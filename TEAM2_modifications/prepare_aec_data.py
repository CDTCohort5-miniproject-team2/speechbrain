"""
Script to process .wav files to satisfy the requirements of the DTLN-aec model.
It modifies the number of channels of the signals, the folder structure, and the naming conventions.
The -o of this script can be directly fed into the -i of DTLN-aec's run_aec.py

Example call:
    $python prepare_aec_data.py -i test_kroto_data/18_12_2023 -o aec_data -m wall_mics -l server_closetalk

    Or, if keeping all default params, simply:
    $python prepare_aec_data.py

    Then, do AEC:
    $python DTLN-aec/run_aec.py -i ../aec_data/ -o ../aec_results/ -m ./pretrained_models/dtln_aec_512

"""

import librosa
from pathlib import Path
import scipy
import scipy.io.wavfile
import os
import shutil
import numpy as np
import argparse


def get_audio_fnames_in_dir(dirpath):
    all_fnames = os.listdir(dirpath)
    wav_fnames = [file for file in all_fnames if file.endswith(".wav")]
    return wav_fnames


def select_single_channel(multichannel_audio_array, channel_idx=0):
    """
    :param multichannel_audio_array:
    :param channel_idx: which channel to select. Default to 0 (the first channel)
    :return: a mono-channel audio array
    """
    try:
        return multichannel_audio_array[channel_idx, :]
    except KeyError:
        print("Select between 0-4 for wall_mics; 0-3 for dashboard_mics; 0-1 for passengers_closetalk")


def prepare_aec_data(recording_dirpath, out_folder, mic_source, lpb_source):
    """
    Creates a folder containing processed data for AEC
    :param recording_dirpath: the recording session's data folder
    :param out_folder: target folder for processed files
    :param mic_source: the near-end microphone signals, default to wall_mics
    :param lpb_source: the far-end microphone or loopback signals, default to server_closetalk
    :return:
    """
    mic_path = os.path.join(recording_dirpath, 'Audio_'+mic_source)
    lpb_path = os.path.join(recording_dirpath, 'Audio_'+lpb_source)

    mic_audios = get_audio_fnames_in_dir(mic_path)
    lpb_audios = get_audio_fnames_in_dir(lpb_path)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for fname in mic_audios:
        fpath = os.path.join(mic_path, fname)
        audio_array, sr = librosa.load(fpath, sr=None, mono=False)
        mono_audio_array = select_single_channel(audio_array)
        fpath_new = os.path.join(out_folder, fname.replace(mic_source, 'mic'))
        scipy.io.wavfile.write(fpath_new, 16000, mono_audio_array)

    # rename and move server audios to aec_data folder
    for fname in lpb_audios:
        shutil.copy2(os.path.join(lpb_path, fname),
                     os.path.join(out_folder, fname.replace(lpb_source, 'lpb')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording_dirpath", "-i", default="test_kroto_data/18_12_2023", help="the recording session's data folder")
    parser.add_argument("--out_folder", "-o", default="aec_data", help="target folder for processed files")
    parser.add_argument("--mic_source", "-m", default="wall_mics", help="the near-end microphone signals, default to wall_mics")
    parser.add_argument("--lpb_source", "-l", default="server_closetalk", help="the far-end microphone or loopback signals, default to server_closetalk")

    args = parser.parse_args()

    prepare_aec_data(args.recording_dirpath, args.out_folder, args.mic_source, args.lpb_source)

