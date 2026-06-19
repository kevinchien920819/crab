from .baseline import EmotionBaselineModel as _EmotionBaselineModel


class EmotionBaseline(_EmotionBaselineModel):
    def __init__(self, cfg):
        super().__init__(cfg)


class EmotionBaselineModel(_EmotionBaselineModel):
    def __init__(self, cfg):
        super().__init__(cfg)


def build_model(cfg):
    model_class = globals()[cfg.name]
    return model_class(cfg)
