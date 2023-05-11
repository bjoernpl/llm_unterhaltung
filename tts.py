from TTS.api import TTS
from loguru import logger

class ThorstenTTS:
    def __init__(self, model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=True, gpu=True, **kwargs):
        self.model = TTS(model_name, progress_bar=progress_bar, gpu=gpu, **kwargs)
        logger.info(f"Loaded TTS model {model_name}")

    def __call__(self, text, **kwargs):
        logger.info(f"Synthesizing text: {text}")
        return 22050, self.model.tts(text, **kwargs)