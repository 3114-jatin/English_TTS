import os
import torch
from datasets import Dataset, Features, Value, Audio

# Define the data
data = {
    "text": [
        "What is an API?",
        "Explain RESTful services.",
        "How does OAuth work?",
        "What is CUDA?"
    ],
    "audio":  [
        "D:\\SpeechT5\\audio1.wav",
        "D:\\SpeechT5\\audio2.wav",
        "D:\\SpeechT5\\audio3.wav",
        "D:\\SpeechT5\\audio4.wav"
    ]
}

# Define the structure of the dataset
features = Features({
    "text": Value("string"),
    "audio": Audio(sampling_rate=16000)
})

# Create the dataset
dataset = Dataset.from_dict(data, features=features)

# Verify the dataset

full_size = len(dataset)
# Select the full dataset
dataset = dataset.select(range(full_size))
print(dataset)

# Save the dataset to disk
dataset.save_to_disk("tech_tts_dataset")

print("Dataset saved to tech_tts_dataset")

