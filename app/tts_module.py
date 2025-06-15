# app/tts_module.py
# Module for Text-to-Speech using Coqui TTS

from TTS.api import TTS as CoquiTTS
import soundfile as sf
import io
import torch
import logging

logger = logging.getLogger(__name__)

# Determine device (CPU for broader compatibility in local Docker testing)
# If you have a CUDA-enabled Docker setup, this could be "cuda"
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"TTS module will use device: {device}")

# Load TTS model
# This will download the model on first run if not already present in the expected cache path.
# The Dockerfile aims to pre-download this.
tts_model_name = "tts_models/en/ljspeech/vits" # A good quality English VITS model
tts_model = None

try:
    logger.info(f"Loading Coqui TTS model: {tts_model_name}...")
    # progress_bar=False to prevent issues in non-interactive environments like Docker build
    tts_model = CoquiTTS(model_name=tts_model_name, progress_bar=False).to(device)
    logger.info(f"Coqui TTS model '{tts_model_name}' loaded successfully on {device}.")
except Exception as e:
    logger.error(f"Error loading Coqui TTS model '{tts_model_name}': {e}", exc_info=True)
    tts_model = None # Ensure model is None if loading fails

def synthesize_speech_to_bytes(text_input: str) -> bytes:
    """
    Synthesizes speech from text input and returns WAV audio bytes.
    """
    if not tts_model:
        logger.error("TTS synthesis failed: TTS model is not loaded.")
        raise RuntimeError("TTS model is not loaded. Please check logs for errors during startup.")
    
    logger.info(f"Synthesizing speech for text (first 50 chars): '{text_input[:50]}...'")
    
    try:
        # The tts() method of TTS.api.TTS returns a list of floats (waveform)
        waveform = tts_model.tts(text=text_input, speaker=None, language=None) # VITS model might not need speaker/language
        
        # Determine sample rate
        sample_rate = 0
        if hasattr(tts_model, 'synthesizer') and hasattr(tts_model.synthesizer, 'output_sample_rate'):
            sample_rate = tts_model.synthesizer.output_sample_rate
        elif hasattr(tts_model, 'config') and 'audio' in tts_model.config and 'sample_rate' in tts_model.config.audio:
            sample_rate = tts_model.config.audio['sample_rate']
        
        if sample_rate == 0:
            # Fallback for common LJSpeech VITS sample rate if not found
            default_sr = 22050
            logger.warning(f"Could not reliably determine sample rate from model. Defaulting to {default_sr} Hz.")
            sample_rate = default_sr
        else:
            logger.info(f"Using sample rate: {sample_rate} Hz.")
            
        buffer = io.BytesIO()
        sf.write(buffer, waveform, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        logger.info("Speech synthesized and converted to WAV bytes successfully.")
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error during speech synthesis: {e}", exc_info=True)
        raise RuntimeError(f"Failed to synthesize speech: {e}")