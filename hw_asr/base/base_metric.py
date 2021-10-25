class BaseMetric:
    def __init__(self, name=None, use_on_train=True, use_on_val=True, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__
        self.use_on_train = use_on_train
        self.use_on_val = use_on_val

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
