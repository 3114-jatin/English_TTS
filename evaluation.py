import os
import torch
import torchaudio
from datasets import load_from_disk
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
import numpy as np

# Load the dataset
dataset = load_from_disk("tech_tts_dataset")

# Load the TTS model and processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
model.eval()

# Load speaker embeddings from the specified folder
speaker_embeddings_folder = "D:\\SpeechT5\\speaker_embeddings"
speaker_embeddings_list = []

# Iterate over the files in the folder and load the embeddings
for file_name in sorted(os.listdir(speaker_embeddings_folder)):
    if file_name.endswith('.npy'):
        file_path = os.path.join(speaker_embeddings_folder, file_name)
        embeddings = np.load(file_path)
        speaker_embeddings_list.append(embeddings)

# Stack the loaded embeddings into a single array (assuming they have compatible shapes)
speaker_embeddings = np.vstack(speaker_embeddings_list)

# Convert to PyTorch tensor
speaker_embeddings = torch.tensor(speaker_embeddings, dtype=torch.float32)

# Function to generate audio from text and convert spectrogram to audio
def text_to_audio(text):
    # Generate Mel-spectrogram
    inputs = processor(text=text, return_tensors="pt", padding=True)

    # Select the first speaker embedding (or modify this logic if you have specific requirements)
    selected_speaker_embedding = speaker_embeddings[0:1]  # Shape (1, embedding_dim)

    with torch.no_grad():
        mel_spectrogram = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=selected_speaker_embedding  # Provide the selected speaker embedding
        )

    return mel_spectrogram

# Evaluate the TTS on the dataset
for index in range(len(dataset)):
    text = dataset[index]['text']
    print(f"Generating audio for: '{text}'")

    # Generate audio
    audio_waveform = text_to_audio(text)

    # Save the generated audio
    output_file = f"output_audio_{index + 1}.wav"
    torchaudio.save(output_file, audio_waveform.squeeze(0), 24000)  # Specify the desired sample rate
    print(f"Audio saved to {output_file}")

    # Print audio properties
    audio_duration = audio_waveform.shape[-1] / 24000  # Duration in seconds
    print(f"Text: {text}")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Audio sample rate: 24000")
    print(f"Waveform shape: {audio_waveform.shape}")

print("Evaluation completed.")
