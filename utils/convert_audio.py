from pydub import AudioSegment

# Path to your input and output files
input_wav = "/Users/shababo/Desktop/canned_intro.wav"
output_wav = "/Users/shababo/Desktop/canned_intro_converted_16000.wav"

# Load the audio file
audio = AudioSegment.from_wav(input_wav)

# Resample to 24000 Hz
audio_converted = audio.set_frame_rate(16000)

# Export the resampled audio
audio_converted.export(output_wav, format="wav")

print(f"Resampled audio saved to {output_wav}")