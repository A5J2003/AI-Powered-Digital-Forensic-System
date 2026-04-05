from input_pipeline.asr import transcribe_audio

audio_path = "My_test.m4a"  # put a short English audio file here
text = transcribe_audio(audio_path)

print("Transcript:")
print(text)


#   python -m input_pipeline.test_asr