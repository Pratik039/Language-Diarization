#
#
import textwrap
import argparse
import logging
import sys
import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier


def get_args():
    #audio_file_path = '/home1/pratik/lang_diarization/data/dev'
    #output_dir = '/home1/pratik/lang_diarization/output/whisper_dev_10_0.4'
    #segment_folder = '/home1/pratik/lang_diarization/output/dev_subseg'
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
        Creates subsegments files from input segments files in a folder.

        ... (rest of the description) ...
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aud_path", type=str,
                        default=None, help=" Directory containing the audio files")
    parser.add_argument("--seg_dir", type=str,
                        default=None, help=" Directory where the windowed segments are available")
    parser.add_argument("--out_dir", type=str,
                        default=None, help=" Directory where features are written to")

    args = parser.parse_args()
    return args


S_RATE = 16000  # Samples/second

# Load SpeechBrain language ID model ONCE
language_id = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    hparams_file="hyperparams.yaml",
    savedir="temp"
)


def whisper_call(wav, segments):
    print('wavfile = ', wav)

    waveform, sr = torchaudio.load(wav)

    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != S_RATE:
        resampler = torchaudio.transforms.Resample(sr, S_RATE)
        waveform = resampler(waveform)

    output = pd.DataFrame()

    for _, speaker_id, start, end in segments:
        start_sample = int(float(start) * S_RATE)
        end_sample = int(float(end) * S_RATE)

        audio_seg = waveform[:, start_sample:end_sample]

        # Skip empty segments
        if audio_seg.shape[1] == 0:
            continue

        with torch.no_grad():
            out_prob, score, index, text_lab = language_id.classify_batch(audio_seg)

        probs = torch.softmax(out_prob.squeeze(), dim=0).cpu().numpy()

        # Create dictionary of language probabilities
        prob_dict = {
            language_id.hparams.label_encoder.decode_ndim(i): float(probs[i])
            for i in range(len(probs))
        }

        df_frame = pd.DataFrame([prob_dict])
        output = pd.concat([output, df_frame], ignore_index=True)

    return output


def save_embeddings(audio_folder_path, output_dir, segment_folder):
    for filename in os.listdir(audio_folder_path):
        try:
            if filename.endswith(".wav"):
                audio_file_path = os.path.join(audio_folder_path, filename)
                segment_file_name = f"subsegments_{filename.replace('.wav', 'pyannote.segment.txt')}"
                segment_file_path = os.path.join(segment_folder, segment_file_name)

                # Read segments from file
                with open(segment_file_path, 'r') as f:
                    segments = [line.strip().split() for line in f]

                pred = whisper_call(audio_file_path, segments)

                # Ensure the output directory exists
                os.makedirs(output_dir, exist_ok=True)

                # Save the numpy array
                output_filename = os.path.splitext(filename)[0] + "_output.npy"
                np.save(os.path.join(output_dir, output_filename), pred.to_numpy())

        except Exception as e:
            print(f"Error processing {filename}: {e}")


############################### MAIN #############################################################

#audio_file_path = '/home1/pratik/lang_diarization/data/dev'
#output_dir = '/home1/pratik/lang_diarization/output/whisper_dev_10_0.4'
#segment_folder = '/home1/pratik/lang_diarization/output/dev_subseg'
#save_embeddings(audio_file_path, output_dir, segment_folder)


def main():
    args = get_args()
    try:
        save_embeddings(args.aud_path, args.out_dir, args.seg_dir)
    except Exception:
        logging.error("Failed creating subsegments", exc_info=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
