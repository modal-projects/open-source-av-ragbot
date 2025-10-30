from pydub import AudioSegment

# Path to your input and output files
input_wav = "server/services/tts/assets/kitt_voice_sample_converted.wav"
output_wav = "server/services/tts/assets/kitt_voice_sample_converted_24000.wav"

# Load the audio file
audio = AudioSegment.from_wav(input_wav)

# Resample to 24000 Hz
audio_24k = audio.set_frame_rate(24000)

# Export the resampled audio
audio_24k.export(output_wav, format="wav")

print(f"Resampled audio saved to {output_wav}")