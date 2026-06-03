from logging import Logger
from typing import Union

from config.deepfake.baseline import DeepfakeBaselineConfig
from linebot import LineBotApi
from linebot.models import TextSendMessage


class LineBot:
    def __init__(self, channel_access_token, user_id, logger: Logger):
        """Initialize the LINE Bot API client and recipient metadata."""
        self.line_bot_api = LineBotApi(channel_access_token)
        self.user_id = user_id
        self.logger = logger
        self.logger.info(f'LineNotifier initialized with user_id: {user_id}')

    def send(self, cfg: Union[DeepfakeBaselineConfig] , total_params, loss, err):

        """Send a training summary notification through LINE Bot."""
        if isinstance(cfg, DeepfakeBaselineConfig):
            message = f'''\
DeepfakeModel Train Finished!
- tag: {cfg.model.tag}
- datasets: {', '.join(dataset.name for dataset in cfg.datasets)}
- model: {cfg.model.name}
- parameters: {total_params:,}
- ssl_model: {cfg.model.ssl_model_str}
- d_model: {cfg.model.d_model}
- n_cls_encoder_layers: {cfg.model.n_cls_encoder_layers}
- n_rhythm_encoder_layers: {cfg.model.n_rhythm_encoder_layers}
- n_inter_encoder_layers: {cfg.model.n_inter_encoder_layers}
- epochs: {cfg.solver.max_epochs}
- eval_err: {err:.6f}'''

        try:
            self.line_bot_api.push_message(
                self.user_id,
                TextSendMessage(text=message)
            )
            self.logger.info(f'Line notification sent to {self.user_id} successfully.')
        except Exception as e:
            self.logger.error('Failed to send Line notification: %s', e)
