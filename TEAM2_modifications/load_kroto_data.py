import librosa
import pandas as pd
from pathlib import Path
import scipy
from scipy import io  # added during debugging
import scipy.io.wavfile  # added during debugging
import os
import numpy as np
import sounddevice as sd
from tqdm import tqdm
import re
import torch
from torch.utils.data import Dataset, DataLoader
from TEAM2_utils import get_channels_by_name, CHANNEL_MAPPING


class RawKrotoData:
    def __init__(self, data_csv_fpath, parent_dirpath, target_sr=16000):
        self.df = pd.read_csv(data_csv_fpath, index_col="scenario_id")  # TODO: check index col
        self.parent_dirpath = Path(parent_dirpath)
        self.raw_audio_dir = parent_dirpath/"Audio"
        self.target_sr = target_sr

        self.sub_array_dirpaths = [self.parent_dirpath / f"Audio_{channel_name}" for channel_name in CHANNEL_MAPPING.keys()]
        self.gt_transcript_dirpaths = [self.parent_dirpath/f"{side}_gt_transcripts" for side in ("merged", "server", "customer")]

        self.check_if_audio_preprocessed()

    def check_if_audio_preprocessed(self):

        if any([not dirpath.exists() for dirpath in self.sub_array_dirpaths]):
            initiate_separate_channels_and_save_audio = input("Raw audio directory loaded. "
                                                              "Separate out channels, downsample and save audio? (Y/N): ")
            if "y" in initiate_separate_channels_and_save_audio.lower():
                print("Separating audio channels for downsampling and saving...")
                self.preprocess_and_save_audio()
                print("Audio saved.")
        else:
            print("Processed audio data found and ready for use.")

    def check_if_transcripts_preprocessed(self):
        if any([not dirpath.exists() for dirpath in self.gt_transcript_dirpaths]):
            print("Raw transcripts have not been preprocessed. This is fine for now "
                  "but program will fail if you try use transcripts. "
                  "Run process_transcript.py to create new folders and save down normalised transcript.")

    def get_demo_audio_array(self):
        pass

    def preprocess_and_save_audio(self):
        channel_names = list(CHANNEL_MAPPING.keys())
        for sub_array_dirpath in self.sub_array_dirpaths:
            if not sub_array_dirpath.exists():
                os.mkdir(sub_array_dirpath)

        for wav_fpath in tqdm(self.raw_audio_dir.glob("*.wav")):

            # TRIM AUDIO FILE TO GET RID OF STARTING SCENARIO ANNOUNCEMENTS
            audio_array, source_sr = librosa.load(wav_fpath, sr=None, mono=False,
                                                  offset=self.df["time_offset"][wav_fpath.stem])  # TODO: check dataframe naming

            # DOWNSAMPLE
            if source_sr != self.target_sr:
                audio_array = librosa.resample(audio_array, orig_sr=source_sr, target_sr=self.target_sr)

            # BREAK INTO CHANNELS-SPECIFIC AUDIO
            for i, channel_name in enumerate(channel_names):
                sub_array = get_channels_by_name(audio_array, channel_name)
                sub_array_fpath = self.sub_array_dirpaths[i] / f"{wav_fpath.stem}_{channel_name}.wav"

                # NB scipy.io.wavfile.write expects (num_samples, num_channels)
                sub_array = np.swapaxes(sub_array, 0, 1)

                # SAVE .WAV FILE
                scipy.io.wavfile.write(sub_array_fpath, self.target_sr, sub_array)

        print("Audio files saved successfully.")

    def get_torch_dataset(self, side="both", dataset_split="training"):
        subset_df = self.df[self.df["dataset_split"] == dataset_split]
        return ProcessedKrotoDataset(subset_df, self.sub_array_dirpaths,
                                     self.gt_transcript_dirpaths, side=side)


class ProcessedKrotoDataset(Dataset):
    def __init__(self, subset_df, sub_array_dirpaths, gt_transcripts_dirpaths, side="both"):
        self.subset_df = subset_df
        self.sub_array_dirpaths = sub_array_dirpaths
        self.gt_transcripts_dirpaths = gt_transcripts_dirpaths
        self.side = side
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        if self.side == "server":
            pass  # return scenario_id, server_closetalk, customer_closetalk, server_gt_transcript
        elif self.side == "customer":
            pass  # return scenario_id, wall_mic_array, server_closetalk, customer_gt_transcript
        elif self.side == "both":
            pass  # scenario_id, server_closetalk, customer_closetalk, wall_mic_array,
            # server_gt_transcript, customer_gt_transcript, merged_gt_transcript
        else:
            raise ValueError("side must be either 'server', 'customer' or 'both'")


def main():
    pass

if __name__ == "__main__":
    main()