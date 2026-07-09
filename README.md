# 🎙️ Language Diarization System

A **Language Diarization** system which performs **Voice Activity Detection (VAD)**, extracts language-related features, and generates language diarization outputs.


## ✨ Features

* 🎤 Voice Activity Detection (Pyannote)
* 🌍 Language Diarization for audio files
* ⚡ Simple command-line interface
* 🔧 Custom output directory support

## 🚀 Getting Started

### 1. Extract the Project

Unzip/clone the provided archive:


## 📦 Requirements

Install Anaconda on your system from **https://repo.anaconda.com/archive/**

Create an environment using the configuration file **baseline.yaml** by running:

```text
conda env create -f baseline.yaml
```

Activate the created environment by running:

```text
conda activate baseline
```
Make sure the pip version is below 24.1

Run this command:
```text
pip install pip==24.0
```
Install the other required Python packages after creating the environment and setting the required version of **pip** by running:
```text
pip install pyannote.audio==0.0.1 numpy==1.26.4 openai-whisper fastcluster==1.2.0 kaldi_io pytorch-lightning==1.6.5
```

### 2. Prepare the Audio Files

Download the DISPLACE 2 Challenge data from:

https://zenodo.org/records/10487448

Download the MERLIon CCS Challenge data from:

https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/ANXS8Z

## Audio Directory Structure

```text
               Data
		         |
		         |
    ---------------------------
    |                         |
    |                         |
    |                         |
    |                         |
   dev                       eval
    |                         |
    |                         |
    |                         |
    |                         |
audio files              audio files

```

### 3. Setting up the scoring tool to compute DER
1. Clone the dscore repository (**https://github.com/nryant/dscore**) in the code directory
2. Install the required packages by running:
   
```text
pip install -r requirements.txt
```

## ▶️ Language Diarization Scripts

## Data Preprocessing 

1. Run VAD on the audio files to detect the speech regions and store the output in .pkl files (pyannote_vad.py)

2. Convert them to segment files and then to Kaldi formatted segment files (seg2kaldi.py)

3. Finally create subsegments from the segments (create_subseg.py)

## Feature Extraction

1. Based on the created subsegments, run the feature extraction to get the Whisper/ECAPA-TDNN posteriors (feat_extr_whisper.py/feat_extr_ecapa.py)

2. Make necessary changes in the run_all.sh script based on the requirement

```text
DISPLACE_DATAPATH="../data/"
DATA="dev"
DISPLACE_DATA="$DISPLACE_DATAPATH/$DATA"

ref_RTTM="$DISPLACE_DATA/${DATA}_original.rttm"
```


## Clustering
Runs the clustering algorithm (AHC) and performs VBHMM on top of it to generate RTTMs (clustering.py)

## Scoring

1. Combine the groundtruth RTTM files into a single RTTM file and rename it as dev_original.rttm/eval_original.rttm

2. Compare groundtruth RTTMs with the system generated RTTMs to generate DER values (score.py)

## How to run

1. Place audio files in the data directory according to the file structure shown above

2. Run the run_all.sh script from within the code directory

3. The run_all.sh takes audio files as input and generates RTTM as output
 
4. The Diarization Error Rate (DER) is computed by comparing the ground-truth RTTM files with the system-generated RTTM files.

## 📁 Output

After successful execution, the system generates an output directory containing the language diarization results for the processed audio files.


## 📂 Project Workflow

```text
 Input Audio (.wav)
        │
        ▼
Voice Activity Detection (Pyannote)
        │
        ▼
Feature Extraction
        │
        ▼
   Clustering
        │
        ▼     
VBx Resegmentation  
        │
        ▼
Generated RTTM File
        │
        ▼
  Output Files
```


## 📊 DOVER-Lap to combine Diarization systems

Clone the dscore repository (**https://github.com/desh2608/dover-lap**) in the output directory

## Installation

Install the required Python packages by running:

```text
pip install -r requirements.txt
```
```text
pip install dover-lap
```

## How to run

After successful installation, run

```text
dover-lap [OPTIONS] OUTPUT_RTTM [INPUT_RTTMS]...
```

**Example**
```text
dover-lap egs/ami/rttm_dl_test egs/ami/rttm_test_*
```

## 🔑 Hugging Face Authentication

The script `VAD_kaldi_segments.py` uses a (**https://huggingface.co/pyannote/segmentation**) model hosted on Hugging Face.

Before running the system:

1. Create a Hugging Face account.
2. Generate an **Access Token** from:

   * **Profile → Settings → Access Tokens → Create New Token**
3. Accept the license agreement for the required Pyannote model (**https://huggingface.co/pyannote/segmentation**) on Hugging Face.
4. Replace the token in `VAD_kaldi_segments.py` with your own token (or authenticate using the Hugging Face CLI).

> **Note:** The VAD step will not run unless a valid Hugging Face token is provided.


## 🛠️ Troubleshooting

* Ensure all required Python packages are installed.
* Verify that the input audio file exists and the path is correct.
* Make sure your Hugging Face token is valid.
* Accept the license agreements for the required Pyannote models before running the VAD module.

## **Filename format:** `<Letter><3 digits>.wav`

Example:
```text
B002.wav
S044.wav
```
