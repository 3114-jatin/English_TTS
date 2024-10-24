import os
import torch
import torchaudio
from speechbrain.inference import Tacotron2, HIFIGAN  # Updated import for SpeechBrain 1.0

# Load models
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tacotron2")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="hifigan")

def sanitize_filename(text):
    """Sanitize text to create a valid filename."""
    return "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in text)

def perform_inference(texts_to_infer, audio_output_dir):
    audio_files = []
    
    # Create output directory if it doesn't exist
    if not os.path.exists(audio_output_dir):
        os.makedirs(audio_output_dir)

    for text in texts_to_infer:
        # Generate mel-spectrogram from text
        mel_output, mel_length, alignment = tacotron2.encode_text(text)

        # Synthesize audio from mel-spectrogram
        waveform = hifi_gan.decode_batch(mel_output)

        # Sanitize the filename
        sanitized_text = sanitize_filename(text)
        output_path = os.path.join(audio_output_dir, f"{sanitized_text}.wav")
        
        # Save the output audio
        torchaudio.save(output_path, waveform.squeeze(1), 22050)  # Save waveform with sample rate
        audio_files.append(output_path)
    
    return audio_files

def collect_mos_ratings(audio_files):
    ratings = []
    
    print("Please rate the following audio samples (1: Poor, 5: Excellent):")
    for audio_file in audio_files:
        print(f"Playing: {audio_file}")
        
        # Use an absolute path to avoid issues
        absolute_path = os.path.abspath(audio_file)
        os.system(f'start {absolute_path}')  # Play audio file (Windows)
        
        while True:
            try:
                rating = int(input(f"Rate '{os.path.basename(audio_file)}': "))
                if 1 <= rating <= 5:
                    ratings.append(rating)
                    break
                else:
                    print("Rating must be between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter an integer between 1 and 5.")
    
    return ratings

def calculate_average_mos(ratings):
    if ratings:
        return sum(ratings) / len(ratings)
    return 0

def main():
    texts_to_infer = [
        "Hello, how are you?",
        "This is a text to speech inference test.",
        "Speech synthesis is interesting!"
    ]
    
    audio_output_dir = "output"  # Specify your output directory
    audio_files = perform_inference(texts_to_infer, audio_output_dir)
    
    print(f"Generated audio files: {audio_files}")
    
    # Collect MOS ratings
    mos_ratings = collect_mos_ratings(audio_files)
    
    # Calculate and display average MOS
    average_mos = calculate_average_mos(mos_ratings)
    print(f"Average Mean Opinion Score (MOS): {average_mos:.2f}")

if __name__ == "__main__":
    main()
