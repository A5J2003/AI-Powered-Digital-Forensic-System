import whisper

asr_model = whisper.load_model("base")

def transcribe_audio(audio_path):
    result = asr_model.transcribe(
        audio_path,
        language="en",
        fp16=False
    )
    return result["text"]