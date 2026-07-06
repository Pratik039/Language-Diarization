import os
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
import numpy as np
import torch

# Load your language identifier model
language_id = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    hparams_file="hyperparams.yaml",
    savedir="temp"
)

# Define the function to process a single audio file with its corresponding subsegment file
def process_audio_with_subsegments(audio_file, subsegment_file, language_id, output_folder):
    # Load audio to get sampling rate (fs)
    signal, fs = torchaudio.load(audio_file)

    # Load subsegment information from text file
    subsegments = []
    with open(subsegment_file, 'r') as file:
        for line in file:
            subsegment_info = line.strip().split()
            subsegments.append((subsegment_info[0], float(subsegment_info[2]), float(subsegment_info[3])))

    # Extract features from subsegments
    embeddings_list = []
    for i, subsegment_info in enumerate(subsegments):
        subsegment_id, start_time, end_time = subsegment_info
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)
        
        subsegment_signal = signal[:, start_sample:end_sample] 
        
        rel_length = torch.tensor([1.0])
        embeddings = language_id.classify_batch(subsegment_signal, rel_length)
        llkd = embeddings[0].reshape(1, -1)
        b = np.exp(llkd)
        #print(b)
        embeddings_list.append(b)
        print(f"Processed subsegment {i+1}/{len(subsegments)} in {audio_file}")

    # Concatenate embeddings from all subsegments
    embeddings = np.concatenate(embeddings_list, axis=0)
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(audio_file))[0] + "_ecapa_posteriors.npy")
    np.save(output_file, embeddings)
    print(f"Embeddings saved successfully for {audio_file}")

# Define the directories for audio files, subsegment files, and output folder for embeddings
audio_dir = "/home/nitk/Pratik/lang_diarization/eval"
subsegment_dir = "/home/nitk/Pratik/eval_subseg_1_1"
output_folder = "/home/nitk/Pratik/displace_eval_10_0.4"

# List all audio files
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]

# Process each audio file
for audio_file in audio_files:
    # Extract the base filename without extension
    audio_filename = os.path.splitext(os.path.basename(audio_file))[0]
    
    # Construct the corresponding subsegment file path based on naming convention
    subsegment_filename = "subsegments_" + audio_filename + "pyannote.segment.txt"
    subsegment_file = os.path.join(subsegment_dir, subsegment_filename)

    # Check if the corresponding subsegment file exists
    if os.path.exists(subsegment_file):
        process_audio_with_subsegments(audio_file, subsegment_file, language_id, output_folder)
    else:
        print(f"No subsegment file found for {audio_file}")

