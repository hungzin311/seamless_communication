"""
many speech to text model like seamless4t require finetune on a dataset like google fleurs
from a csv file have 2 columns audio and sentence, where audio is the path to wav file (16khz, mono channel) and sentence is the transcription
create a dataset similar to google fleurs and upload it to huggingface
"""

"""
make sure load wav file into item["audio"]["array"] as numpy array like google fleurs, you can use librosa.load and np.float64 , and get same result
make sure item["audio"]["sampling_rate"] = 16000 already
"""

import pandas as pd
import uuid 
import os
import numpy as np
import librosa
from datasets import load_dataset
import soundfile as sf
from pathlib import Path


wav_dir = Path.cwd() / "data"
wav_dir.mkdir(parents=True, exist_ok=True)

ds_train = []
ds_val = []

data = load_dataset('strongpear/viet_muong_merged_0_200_denoise_silence_speaker101')['train']

train_data, val_data = data.train_test_split(test_size = 0.05)

from tqdm import tqdm
for item in tqdm(train_data, total=len(train_data)):
    id = str(uuid.uuid4())
    audio_array = item["audio"]['array']
    sampling_rate = item["audio"]['sampling_rate']
    audio_path = os.path.join(wav_dir, f"{id}.wav")
    sf.write(audio_path, audio_array, sampling_rate)
    transcription = item["text"]   
    ds_item = {
        "path": audio_path,
        "id": id,
        "transcription": transcription,
        "audio": {
            "path": audio_path,
            "array": audio_array,
            "sampling_rate": sampling_rate
        }
    } 
    ds_train.append(ds_item)

for item in tqdm(val_data, total=len(val_data)):
    id = str(uuid.uuid4())
    audio_array = item["audio"]['array']
    sampling_rate = item["audio"]['sampling_rate']
    audio_path = os.path.join(wav_dir, f"{id}.wav")
    sf.write(audio_path, audio_array, sampling_rate)
    transcription = item["text"]   
    ds_item = {
        "path": audio_path,
        "id": id,
        "transcription": transcription,
        "audio": {
            "path": audio_path,
            "array": audio_array,
            "sampling_rate": sampling_rate
        }
    } 
    ds_val.append(ds_item)

from datasets import Dataset, DatasetDict

big_dict_train = {"path":[], "id":[], "audio":[], "transcription":[]}

for item in ds_train:
    big_dict_train["path"].append(str(item["path"]))
    big_dict_train["id"].append(item["id"])
    big_dict_train["audio"].append(item["audio"])
    big_dict_train["transcription"].append(item["transcription"])

final_ds_train = Dataset.from_dict(big_dict_train)

big_dict_val = {"path":[], "id":[], "audio":[], "transcription":[]}

for item in ds_val:
    big_dict_val["path"].append(str(item["path"]))
    big_dict_val["id"].append(item["id"])
    big_dict_val["audio"].append(item["audio"])
    big_dict_val["transcription"].append(item["transcription"])


final_ds_val = Dataset.from_dict(big_dict_val)


dataset_dict = DatasetDict({
    "train": final_ds_train,
    "validation": final_ds_val
})

# 3. (Optional) Set metadata—description, license, etc.—so your dataset card is populated
dataset_dict.push_to_hub(
    repo_id="hungzin/viet_muong_merged_0_200_denoise_silence_speaker101",
)
