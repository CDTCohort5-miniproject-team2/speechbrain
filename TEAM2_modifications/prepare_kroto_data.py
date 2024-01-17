import librosa
import pandas as pd
from pathlib import Path
import scipy
import os
import numpy as np
import sounddevice as sd
import tqdm
from tqdm import tqdm

CHANNEL_MAPPING = {
    "dashboard_mics": [0, 1, 2, 3],
    # Channels 4, 5 and 6 are not used
    "wall_mics": [7, 8, 9, 10, 11],
    "passengers_closetalk": [12, 13],
    "customer_closetalk": [14],
    "server_closetalk": [15],
}

def get_channels_by_name(multichannel_audio_array, target_channel_name):
    """

    :param multichannel_audio_array:
    :param target_channel_name:
    :return:
    """
    assert multichannel_audio_array.shape[0] == 16, "The function_get_channels_by_name can only operate on the full 16-channel array."
    try:
        target_channels = np.array(CHANNEL_MAPPING[target_channel_name])
    except KeyError:
        print("Target channel name not valid. Returning array with all channels.")
        return multichannel_audio_array
    else:
        return multichannel_audio_array[target_channels, :]


def play_audio_array(audio_array, sr=16000):
    """
    Function for playing a (mono or multichannel) audio array for sense check
    :param audio_array: numpy array (n_samples,) or (n_channels, n_samples,)
    :param sr: int - sample rate to use
    :return:
    """
    if len(audio_array.shape) == 2:
        # down-mixing multichannel audio for playback on laptop speakers
        audio_array = np.mean(audio_array, axis=0)
    sd.play(audio_array, samplerate=sr)
    sd.wait()

def get_audio_array_timeslice(audio_array, start_time, end_time, sr=16000):
    """
    Returns a timeslice of the audio array
    :param audio_array: 1, 2 or 3D - as long as n_samples is the last dimension
    :param start_time: in seconds
    :param end_time: in seconds
    :param sr: int - sample rate to use
    :return:
    """
    if end_time == 0:
        # return the whole array unsliced
        return audio_array

    start_sample, end_sample = int(start_time*sr), int(end_time*sr)

    if end_sample >= audio_array.shape[-1]:
        end_sample = audio_array.shape[-1] - 1
        print(f"End time is greater than audio length, returning a shorter timeslice instead "
              f"from {start_time} seconds to {end_sample/sr} seconds.")

    # TEAM2 fyi: "..." in slicing numpy arrays allows it to accept arrays of an arbitrary number of dimensions
    return audio_array[..., start_sample:end_sample]

def cut_long_audio_files(dirpath, n_seconds=20.0):
    # TODO
    pass

class KrotoData:
    def __init__(self, dirpath_str, target_sr=16000):
        """
        Parses a data folder containing the sub-folders Audio, Text, Logs
        :param dirpath_str: str - path to the folder/directory
        """
        self.parent_dirpath = Path(dirpath_str)
        if not (self.parent_dirpath.exists() and self.parent_dirpath.is_dir()):
            raise FileNotFoundError("Specified folder does not exist - check that you've provided the correct folder path")

        self.raw_audio_dir, self.transcript_dir, self.log_dir = self.parent_dirpath/"Audio", self.parent_dirpath/"Text", self.parent_dirpath/"Log"

        self.audio_catalogue = list(self.raw_audio_dir.glob("*.wav"))

        self.target_sr = target_sr

    def downsample_and_save_audio(self):
        """
        Downsample all audio in the "Audio" directory and save the downsampled audio to "Audio_downsampled".
        TEAM2 note: run this only if you require the downsampled version of the full 16-channel audio
        (e.g. for testing new code).
        Otherwise, downsampling is performed as part of the function separate_channels_and_save_audio.
        :return:
        """
        downsampled_audio_dir = self.parent_dirpath / "Audio_downsampled"

        if not downsampled_audio_dir.exists():
            os.mkdir(downsampled_audio_dir)

        for raw_wav_fpath in self.raw_audio_dir.glob("*.wav"):
            downsampled_audio_fpath = downsampled_audio_dir / raw_wav_fpath.name
            if not downsampled_audio_fpath.exists():
                audio_array, source_sr = librosa.load(raw_wav_fpath, sr=None, mono=False)
                # audio_array has shape (n_channels, n_samples), samples are in dtype float32
                if source_sr != self.target_sr:
                    audio_array = librosa.resample(audio_array, orig_sr=source_sr, target_sr=self.target_sr)

                # NB scipy.io.wavfile.write expects (num_samples, num_channels)
                audio_array = np.swapaxes(audio_array, 0, 1)
                scipy.io.wavfile.write(downsampled_audio_fpath, self.target_sr, audio_array)
        print("Directory with downsampled audio prepared.")

    def get_demo_audio_array(self, audio_fname="", audio_idx=0, downsampled=True, timeslice=(0.0, 0.0)):
        if audio_fname:
            fpath = self.raw_audio_dir/audio_fname
            if not fpath.exists():
                print(f"Filepath invalid. Returning sample audio indexed at 0 in the directory instead.")
                fpath = self.raw_audio_dir / self.audio_catalogue[audio_idx].name
        else:
            fpath = self.raw_audio_dir/self.audio_catalogue[audio_idx].name
        print(f"Getting demo audio from {fpath.stem}")
        demo_array, sr = librosa.load(fpath, sr=None, mono=False)

        if downsampled and (sr != self.target_sr):
            demo_array = librosa.resample(demo_array, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target.sr
        
        return get_audio_array_timeslice(demo_array, start_time=timeslice[0], end_time=timeslice[1], sr=int(sr))

    def separate_channels_and_save_audio(self, downsampling=True):
        """
        Splits the full 16-channel audio files found in the "Audio" folder according to the CHANNEL_MAPPING partitions.
        Saves down the audio files (with channels filtered) into new directories named after these partitions.
        :param downsampling: boolean - if True, performs downsamples the audio to self.target_sr before saving
        :return: None
        """
        channel_names = list(CHANNEL_MAPPING.keys())
        sub_array_dirpaths = [self.parent_dirpath/f"Audio_{channel_name}" for channel_name in channel_names]

        for sub_array_dirpath in sub_array_dirpaths:
            if not sub_array_dirpath.exists():
                os.mkdir(sub_array_dirpath)

        for wav_fpath in tqdm(self.raw_audio_dir.glob("*.wav")):
            audio_array, source_sr = librosa.load(wav_fpath, sr=None, mono=False)

            if downsampling and (source_sr != self.target_sr):
                audio_array = librosa.resample(audio_array, orig_sr=source_sr, target_sr=self.target_sr)

            for i, channel_name in enumerate(channel_names):
                sub_array = get_channels_by_name(audio_array, channel_name)
                sub_array_fpath = sub_array_dirpaths[i]/f"{wav_fpath.stem}_{channel_name}.wav"

                # NB scipy.io.wavfile.write expects (num_samples, num_channels)
                sub_array = np.swapaxes(sub_array, 0, 1)
                scipy.io.wavfile.write(sub_array_fpath, self.target_sr if downsampling else source_sr, sub_array)

        print("Audio files saved successfully.")


if __name__ == "__main__":
    demo_dirpath = "test_kroto_data/18_12_2023"
    # TEAM2 instructions: replace this with the path where you saved the recording session's data folder
    # this data folder should contain the sub-folders "Audio", "Logs" and "Text"
    demo_wav_fname = "20231218_1702902362163_scenariov3_11.wav"  # this is contained in the 18_12_2023 session folder

    demo_kroto_data = KrotoData(demo_dirpath)

    demo_kroto_data.separate_channels_and_save_audio()

    # Example for sense-checking the time-sliced, channel-isolated audio
    demo_array = demo_kroto_data.get_demo_audio_array(demo_wav_fname, timeslice=(4.0, 16.0))
    demo_array_wall_mics_only = get_channels_by_name(demo_array, "wall_mics")
    play_audio_array(demo_array_wall_mics_only)




