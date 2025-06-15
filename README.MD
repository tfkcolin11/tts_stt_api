# Build:

```batch
docker build -t tts-stt-service .
```

# Run:

```batch
docker run -p 8000:8000 --name my-tts-stt-app tts-stt-service
```

# How to use the API endpoints:

* Text-to-Speech (/tts/):

    * Method: POST
    * URL: http://localhost:8000/tts/
    * Body (JSON):

```JSON
{
    "text": "Hello, this is a test of the text to speech service."
}
```

Response: The API will stream back a WAV audio file. You can use tools like curl to save it or test it:

```Bash

curl -X POST -H "Content-Type: application/json" \
     -d '{"text":"你好，你叫什么名字？"}' \
     http://localhost:8000/tts/ --output output.wav
```

Then you can play output.wav.

* Speech-to-Text (/stt/):

    * Method: POST
    * URL: http://localhost:8000/stt/
    * Body (form-data): Send an audio file (e.g., WAV, MP3) as a form field named audio_file.
Response (JSON):

```JSON
{
    "filename": "your_audio_file.wav",
    "transcription": "The transcribed text will appear here."
}
```

Using curl:

```Bash
# Make sure you have an audio file, e.g., 'test_audio.wav'
curl -X POST -F "audio_file=@test_audio.wav" http://localhost:8000/stt/
```