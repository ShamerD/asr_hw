# ASR project barebones

## Installation guide

Install libraries

```shell
pip install -r ./requirements.txt

git clone --recursive https://github.com/parlance/ctcdecode.git
pip install ./ctcdecode
```

```ctcdecode``` can be somewhat painful to build on Windows (issues with compiling C++ code).
However, on Linux there should be no problems.

Download resources (in project directory)

```shell
wget -q -O ./lm.arpa.gz http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz
gunzip ./lm.arpa.gz

# load model from gdrive in your convenient way (manually or wget)
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14tFL0ylwPHDFWZDqlELCsSKl-U_ze6cC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14tFL0ylwPHDFWZDqlELCsSKl-U_ze6cC" -O default_test/checkpoint.pth && rm -rf /tmp/cookies.txt
```

## Evaluating
```shell
python3 ./test.py -c ./default_test/config.json -r ./default_test/checkpoint.pth (-t <test_folder>) 
```

## Results
This model achieved following metrics:
* CER (argmax): 49.09
* WER (argmax): 89.45
* CER (beam-search + Language Model): 48.78
* WER (beam-search + Language Model): 77.70

## Notes

Unit tests can be broken in Datasphere due to PermissionErrors.
Colab worked for me as expected (OK)

All runs can be examined in Wandb project: https://wandb.ai/shamerd/asr_project?workspace=user-shamerd.

## Credits

this repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.