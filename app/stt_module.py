# app/stt_module.py
# Module for Speech-to-Text using OpenAI Whisper

import whisper
import tempfile
import os
import shutil
from fastapi import UploadFile
import torch
import logging

logger = logging.getLogger(__name__)

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"STT module will use device: {device}")

# Load Whisper model
# This will download the model on first run to the specified download_root if not already present.
# The Dockerfile aims to pre-download this.
stt_model_name = "base.en"  # Options: "tiny.en", "base.en", "small.en", "medium.en", "large.en"
# Model directory should match the one used in Dockerfile for pre-downloading
model_dir = "/app/models/whisper" 
stt_model = None

try:
    logger.info(f"Loading Whisper STT model: {stt_model_name} from {model_dir}...")
    # Ensure the directory exists for loading, though Dockerfile should create it
    os.makedirs(model_dir, exist_ok=True) 
    stt_model = whisper.load_model(stt_model_name, download_root=model_dir, device=device)
    logger.info(f"Whisper STT model '{stt_model_name}' loaded successfully on {device}.")
except Exception as e:
    logger.error(f"Error loading Whisper STT model '{stt_model_name}': {e}", exc_info=True)
    stt_model = None # Ensure model is None if loading fails


async def transcribe_audio_file(audio_file: UploadFile) -> str:
    """
    Transcribes an audio file using Whisper and returns the transcription text.
    """
    if not stt_model:
        logger.error("STT transcription failed: STT model is not loaded.")
        raise RuntimeError("STT model is not loaded. Please check logs for errors during startup.")

    logger.info(f"Preparing to transcribe audio file: {audio_file.filename}")
    
    # Save UploadFile to a temporary file because whisper.transcribe() expects a file path.
    # Using mkstemp for a secure temporary file.
    fd, tmp_path = tempfile.mkstemp()
    logger.debug(f"Temporary file created for STT: {tmp_path}")

    try:
        with os.fdopen(fd, 'wb') as tmp_file_obj:
            shutil.copyfileobj(audio_file.file, tmp_file_obj)
        logger.debug(f"Audio content copied to temporary file: {tmp_path}")

        # Transcribe
        # For CPU, fp16=False is important. If GPU, fp16=True can be used if supported.
        logger.info(f"Starting transcription with Whisper model ({stt_model_name}) on {device}...")
        result = stt_model.transcribe(tmp_path, fp16=(device == "cuda" and torch.cuda.is_available()))
        transcription = result["text"]
        logger.info("Transcription successful.")
        logger.debug(f"Raw transcription result: {result}")
    except Exception as e:
        logger.error(f"Error during audio transcription: {e}", exc_info=True)
        raise RuntimeError(f"Failed to transcribe audio: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug(f"Temporary file {tmp_path} removed.")
        # Ensure the uploaded file stream is closed
        await audio_file.close()
        logger.debug(f"Uploaded file stream {audio_file.filename} closed.")
        
    return transcription