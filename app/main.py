# app/main.py
# Main FastAPI application for TTS and STT

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import io
import logging

# Import TTS and STT processing functions
from .tts_module import synthesize_speech_to_bytes, tts_model
from .stt_module import transcribe_audio_file, stt_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TTS/STT API",
    description="A REST API for Text-to-Speech and Speech-to-Text using Coqui TTS and OpenAI Whisper.",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    if tts_model is None:
        logger.error("TTS model failed to load. TTS endpoint will not be available.")
    if stt_model is None:
        logger.error("STT model failed to load. STT endpoint will not be available.")
    logger.info("Application startup complete.")

@app.post("/tts/",
          summary="Text-to-Speech",
          description="Converts input text to speech (WAV audio format).",
          tags=["TTS"])
async def text_to_speech(payload: dict):
    """
    Converts text to speech.
    Input: JSON payload with a "text" field.
    Output: WAV audio stream.
    """
    if tts_model is None:
        logger.error("TTS request failed: TTS model not loaded.")
        raise HTTPException(status_code=503, detail="TTS service is unavailable due to model loading issues.")

    text = payload.get("text")
    if not text:
        logger.warning("TTS request failed: 'text' field is missing or empty.")
        raise HTTPException(status_code=400, detail="'text' field is required in the JSON payload.")

    try:
        logger.info(f"Received TTS request for text: \"{text[:50]}...\"")
        audio_bytes = synthesize_speech_to_bytes(text)
        logger.info("Speech synthesized successfully.")
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")
    except RuntimeError as e:
        logger.error(f"TTS Runtime Error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during TTS: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during TTS: {e}")

@app.post("/stt/",
          summary="Speech-to-Text",
          description="Transcribes an uploaded audio file to text.",
          tags=["STT"])
async def speech_to_text(audio_file: UploadFile = File(..., description="Audio file to transcribe.")):
    """
    Transcribes an audio file to text.
    Input: Audio file (e.g., WAV, MP3).
    Output: JSON with "transcription" field.
    """
    if stt_model is None:
        logger.error("STT request failed: STT model not loaded.")
        raise HTTPException(status_code=503, detail="STT service is unavailable due to model loading issues.")

    if not audio_file.filename:
        logger.warning("STT request failed: No audio file provided.")
        raise HTTPException(status_code=400, detail="No audio file provided.")
    
    # Log file type if available
    logger.info(f"Received STT request for file: {audio_file.filename}, content type: {audio_file.content_type}")

    # Supported content types (Whisper is quite flexible, but good to have a basic check)
    # common_audio_types = ["audio/wav", "audio/mpeg", "audio/x-wav", "audio/mp3", "audio/ogg", "audio/flac", "audio/aac"]
    # if audio_file.content_type not in common_audio_types:
    #     logger.warning(f"STT request with potentially unsupported content type: {audio_file.content_type}")
        # Not raising error as Whisper might handle it.

    try:
        transcription = await transcribe_audio_file(audio_file)
        logger.info(f"Audio transcribed successfully. Transcription: \"{transcription[:50]}...\"")
        return {"filename": audio_file.filename, "transcription": transcription}
    except RuntimeError as e:
        logger.error(f"STT Runtime Error: {e}")
        raise HTTPException(status_code=500, detail=f"STT transcription failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during STT: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during STT: {e}")

if __name__ == "__main__":
    import uvicorn
    # This is for local debugging of the main.py file itself, not for Docker.
    # Docker will use the CMD instruction.
    logger.info("Starting Uvicorn server for local debugging...")
    uvicorn.run(app, host="0.0.0.0", port=8000)