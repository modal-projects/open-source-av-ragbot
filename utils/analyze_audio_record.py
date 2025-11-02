from moviepy import VideoFileClip
import matplotlib.pyplot as plt
import numpy as np

def extract_audio_from_mov(input_mov_path, output_audio_path):
    """
    Extracts audio from a MOV video file and saves it as a separate audio file.

    Args:
        input_mov_path (str): The path to the input .mov video file.
        output_audio_path (str): The desired path for the output audio file 
                                 (e.g., "output_audio.mp3" or "output_audio.wav").
    """
    try:
        video_clip = VideoFileClip(input_mov_path)
        audio_clip = video_clip.audio
        audio_array = audio_clip.to_soundarray()
        # create interactive plot of audio array
        plt.plot(np.arange(len(audio_array))/audio_clip.fps, audio_array)
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
input_file = "/Users/shababo/Desktop/optimized_rag.mp4"  # Replace with the actual path to your .mov file
output_file = "extracted_audio.mp3" # Or .wav, .ogg, etc.

extract_audio_from_mov(input_file, output_file)