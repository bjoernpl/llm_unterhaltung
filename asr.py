from faster_whisper import WhisperModel
from librosa import resample
import numpy as np
from loguru import logger
import time

class FasterWhisper:
    def __init__(self, model_name="medium", device="cuda", compute_type="int8", **kwargs):
        """
        model_name: one of tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, or large-v2
        device: "cuda" or "cpu"
        compute_type: "int8" or "fp16" or "fp32"
        """
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type, **kwargs)
        logger.info(f"Loaded ASR model {model_name}")

    def __call__(self, audio_array, sr, language="de"):
        logger.info(f"Transcribing audio of length {len(audio_array)} with sample rate {sr}")
        audio_array = audio_array.astype(np.float32)
        audio_array /= np.iinfo(np.int32).max
        start = time.time()
        if sr != 16000:
            audio_array = resample(audio_array.T, orig_sr=sr, target_sr=16000).T
            logger.info(f"Resampled audio  in {time.time() - start} seconds")

        return self.model.transcribe(audio_array, language=language)
