import librosa
import pandas as pd
from pathlib import Path
import scipy
from scipy import io  # added during debugging
import scipy.io.wavfile  # added during debugging
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from TEAM2_utils import get_channels_by_name, CHANNEL_MAPPING


class RawKrotoData:
    def __init__(self, data_csv_fpath, parent_dirpath):
        """
        Wrapper class around the directory containing the raw data to be used in our experiments
        :param data_csv_fpath: (str) the csv filepath containing all information about our training/validation/test datasets
        :param parent_dirpath: (str) the parent dataset dirpath containing audio and transcript subdirectories
        """
        if not Path(data_csv_fpath).exists():
            raise FileNotFoundError(f"CSV of dataset split not found at {data_csv_fpath} - check filepath and confirm.")
        self.df = pd.read_csv(data_csv_fpath)

        self.parent_dirpath = Path(parent_dirpath)
        self.raw_audio_dir = self.parent_dirpath / f"Audio"
        self.target_sr = 16000  # hard-coded in, the sample rate that we're using throughout our project

        self.channel_names = list(CHANNEL_MAPPING.keys())
        self.sub_array_dirpaths = [self.parent_dirpath / f"Audio_{channel_name}" for channel_name in self.channel_names]
        self.gt_transcript_dirpaths = [self.parent_dirpath/f"{side}_gt_transcripts" for side in ("merged", "server", "customer")]

        self.check_if_audio_preprocessed()
        self.parse_csv()

    def parse_csv(self):
        """
        This function adds additional info to the initial csv e.g. filepaths for ease of data loading
        """
        # filename of the raw 13-channel audio file should be in Recording File reference

        self.df["scenario_true_id"] = [(fname.replace(".wav", "")).replace("16k_", "") for fname in self.df["Recording File reference"]]

        for channel_name, dirpath in zip(self.channel_names, self.sub_array_dirpaths):
            new_audio_fpath_col_name = f"{channel_name}_audio_fpath"
            audio_fpaths = []
            for scenario_id in self.df["scenario_true_id"]:
                audio_fpath = dirpath/f"{scenario_id}_{channel_name}.wav"
                audio_fpaths.append(audio_fpath)

            self.df[new_audio_fpath_col_name] = audio_fpaths

    def check_if_audio_preprocessed(self):

        if any([not dirpath.exists() for dirpath in self.sub_array_dirpaths]):
            print("Raw audio directory loaded. Separating out channels, downsampling and saving audio.")
            self.preprocess_and_save_audio()
            print("Audio saved.")
        else:
            print("Processed audio data found and ready for use.")

    def check_if_transcripts_preprocessed(self):
        if any([not dirpath.exists() for dirpath in self.gt_transcript_dirpaths]):
            print("Raw transcripts have not been preprocessed. This is fine for now "
                  "but program will fail if you try use transcripts. "
                  "Run process_transcript.py to create new folders and save down normalised transcript.")
        else:
            print("Processed transcripts found and ready for use.")

    def preprocess_and_save_audio(self):
        channel_names = list(CHANNEL_MAPPING.keys())
        for sub_array_dirpath in self.sub_array_dirpaths:
            if not sub_array_dirpath.exists():
                os.mkdir(sub_array_dirpath)

        for wav_fpath in tqdm(self.raw_audio_dir.glob("*.wav")):
            audio_array, source_sr = librosa.load(wav_fpath, sr=None, mono=False)

            # DOWNSAMPLE
            if source_sr != self.target_sr:
                audio_array = librosa.resample(audio_array, orig_sr=source_sr, target_sr=self.target_sr)

            wav_fpath_stem = str(wav_fpath.stem).replace("16k_", "")
            # BREAK INTO CHANNELS-SPECIFIC AUDIO
            for i, channel_name in enumerate(channel_names):
                sub_array = get_channels_by_name(audio_array, channel_name)
                sub_array_fpath = self.sub_array_dirpaths[i] / f"{wav_fpath_stem}_{channel_name}.wav"

                # NB scipy.io.wavfile.write expects (num_samples, num_channels)
                sub_array = np.swapaxes(sub_array, 0, 1)

                # SAVE .WAV FILE
                scipy.io.wavfile.write(sub_array_fpath, self.target_sr, sub_array)

        print("Audio files saved successfully.")

    def get_torch_dataset(self, dataset_split="Training"):
        """
        Returns a torch dataset that wraps around the preprocessed dataset directory
        :param dataset_split: (str) this should now always have the value "Training"
        :return: a torch dataset
        """
        subset_df = self.df[self.df["Set"] == dataset_split]
        return KrotoAudioDataset(subset_df)


class KrotoAudioDataset(Dataset):

    def __init__(self, subset_df):
        self.subset_df = subset_df

    def __len__(self):
        return len(self.subset_df)

    def __getitem__(self, idx):
        chosen = self.subset_df.iloc[idx]
        scenario_id = chosen["scenario_true_id"]

        server_closetalk_array, _ = librosa.load(chosen["server_closetalk_audio_fpath"], sr=None)
        customer_closetalk_array, _ = librosa.load(chosen["customer_closetalk_audio_fpath"], sr=None)
        top_centre_wall_mic_array, _ = librosa.load(chosen["top_centre_wall_mic_audio_fpath"], sr=None)
        return scenario_id, server_closetalk_array, customer_closetalk_array, top_centre_wall_mic_array

def main():
    print("Loading dataset")
    demo_data = RawKrotoData(data_csv_fpath="kroto_data/demo_dataset_split.csv", parent_dirpath="kroto_data")
    demo_data.check_if_audio_preprocessed()
    demo_data.check_if_transcripts_preprocessed()

    print("Data loaded")
    print(demo_data.df.info())
    print(demo_data.df.head(1))

if __name__ == "__main__":
    main()
