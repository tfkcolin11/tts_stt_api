# Dockerfile

# Use a Python base image. slim-bullseye includes more system libraries than alpine or slim.
FROM python:3.10-slim-bullseye

LABEL maintainer="AI Assistant"
LABEL description="Docker image for TTS/STT API using Coqui TTS and OpenAI Whisper."

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    # Set HOME to /app so Coqui TTS downloads models to /app/.local, making them part of the image layer
    HOME=/app \
    # Path for Coqui TTS models (though HOME=/app should handle it)
    TTS_HOME=/app/.local/share/tts \
    # Path for Whisper models
    WHISPER_HOME=/app/models/whisper

# Create working directory
WORKDIR /app

# Install system dependencies
# - ffmpeg: for audio processing by Whisper and potentially Coqui TTS/Pydub
# - libsndfile1: for the soundfile Python library
# - espeak-ng: required by Coqui TTS for phonemization
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
# Ensure PyTorch is installed for CPU. If GPU is intended, this step needs modification.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create directories for models (Whisper specifically, Coqui uses ~/.local/share/tts)
RUN mkdir -p ${WHISPER_HOME} && \
    mkdir -p ${TTS_HOME} && \
    # Ensure these directories are writable by the default user if needed, though models are downloaded as root here.
    chown -R nobody:nogroup /app/.local || true && \
    chown -R nobody:nogroup ${WHISPER_HOME} || true


# Pre-download models to be included in the image
# This can take some time during the build process
# Note: progress_bar=False for CoquiTTS during Docker build to avoid issues.
# Coqui TTS model download (relies on HOME=/app being set)
# The TTS() constructor will download if not found.
RUN echo "Downloading Coqui TTS model (tts_models/en/ljspeech/vits)..." && \
    python -c "from TTS.api import TTS; TTS(model_name='tts_models/en/ljspeech/vits', progress_bar=False)" && \
    echo "Coqui TTS model download attempt finished."

# Whisper model download
RUN echo "Downloading Whisper model (base.en)..." && \
    python -c "import whisper; import os; os.makedirs('${WHISPER_HOME}', exist_ok=True); whisper.load_model('base.en', download_root='${WHISPER_HOME}')" && \
    echo "Whisper model download attempt finished."

# Copy the application code into the image
COPY ./app /app/app

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application using Uvicorn
# Running as non-root user for better security, though model downloads happened as root.
# If files created by root cause permission issues for 'nobody', adjust ownership or run as a user with appropriate permissions.
# For simplicity, running as root here, but 'USER nobody' could be used if permissions are handled.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]