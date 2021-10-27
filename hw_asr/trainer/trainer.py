import random
from random import shuffle

import PIL
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
import torchaudio
from tqdm import tqdm

from hw_asr.base import BaseTrainer
from hw_asr.logger.utils import plot_spectrogram_to_buf
from hw_asr.metric.utils import calc_cer, calc_wer
from hw_asr.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            data_loader,
            text_encoder,
            valid_data_loader=None,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.text_encoder = text_encoder
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics if m.use_on_train], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics if m.use_on_val], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "text_encoded"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            if batch_idx >= self.len_epoch:
                break
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(part="train", **batch)
                self._log_spectrogram(batch["spectrogram"])
                self._log_audio(batch['audio'], batch['audio_path'])
                self._log_scalars(self.train_metrics)

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
        batch["log_probs_length"] = self.model.transform_input_lengths(
            batch["spectrogram_length"]
        )
        batch["loss"] = self.criterion(**batch)
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            if met.name in metrics.keys():
                metrics.update(met.name, met(**batch))
        return batch

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(self.valid_data_loader),
                    desc="validation",
                    total=len(self.valid_data_loader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.valid_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, "valid")
            self._log_scalars(self.valid_metrics)
            self._log_predictions(part="val", **batch)
            self._log_spectrogram(batch["spectrogram"])
            self._log_audio(batch["audio"], batch["audio_path"])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            text,
            log_probs,
            log_probs_length,
            examples_to_log=5,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return
        argmax_inds = log_probs.cpu().argmax(-1)
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length)
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]

        beam_search_results = self.text_encoder.ctc_beam_search(log_probs, log_probs_length, beam_size=100)
        top_beam_search_results = [beams[:3] for beams in beam_search_results]

        tuples = list(zip(argmax_texts, text, argmax_texts_raw, top_beam_search_results))

        shuffle(tuples)
        to_log_pred = []
        to_log_pred_raw = []
        for argmax_pred, target, argmax_raw_pred, top_beams in tuples[:examples_to_log]:
            to_log_pred_raw.append(f"true: '{target}' | pred: '{argmax_raw_pred}'<br>")

            argmax_wer = calc_wer(target, argmax_pred) * 100
            argmax_cer = calc_cer(target, argmax_pred) * 100

            log_string = f"true: '{target}'<br>" \
                         f"argmax_pred: '{argmax_pred}' | wer: {argmax_wer:.2f} | cer: {argmax_cer:.2f}<br>"
            for i, (beam_pred, beam_score) in enumerate(top_beams):
                beam_wer = calc_wer(target, beam_pred) * 100
                beam_cer = calc_cer(target, beam_pred) * 100

                log_string += f"beam_{i+1}_pred: '{beam_pred}' | wer: {beam_wer:.2f} | cer: {beam_cer:.2f}<br>"

            to_log_pred.append(log_string)

        self.writer.add_text(f"predictions", "<br>".join(to_log_pred))
        self.writer.add_text(
            f"predictions_raw", "<br>".join(to_log_pred_raw)
        )

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch)
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram.cpu().log()))
        self.writer.add_image("spectrogram", ToTensor()(image))

    def _log_audio(self, audio_batch, audio_path_batch):
        idx = random.randrange(len(audio_batch))
        audio = audio_batch[idx]
        t_info = torchaudio.info(str(audio_path_batch[idx]))
        self.writer.add_audio("audio", audio, sample_rate=t_info.sample_rate)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
