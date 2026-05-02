from ..emotion.models import DeepfakeBaselineModel, DeepfakeBaselineModelWithDuration


class DeepfakeBaselineModel(DeepfakeBaselineModel):
    def __init__(self, cfg):
        super().__init__(cfg)


class DeepfakeBaselineModelWithDuration(DeepfakeBaselineModelWithDuration):
    def __init__(self, cfg):
        super().__init__(cfg)
        
