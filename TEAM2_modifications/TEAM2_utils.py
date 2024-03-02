import numpy as np
import sounddevice as sd
import re
from spellchecker import SpellChecker
from num2words import num2words


CHANNEL_MAPPING = {
    "dashboard_mics": [0, 1, 2, 3],
    # "wall_mics": [4, 5, 6, 7, 8],
    "passengers_closetalk": [10, 11],
    "customer_closetalk": [9],
    "server_closetalk": [12],
    "top_centre_wall_mic": [5],
}

def get_channels_by_name(multichannel_audio_array, target_channel_name):
    """

    :param multichannel_audio_array:
    :param target_channel_name:
    :return:
    """
    assert multichannel_audio_array.shape[0] in [13, 16], "The function_get_channels_by_name can only operate on the full 13- or 16-channel array."
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






def main():
    pass

if __name__ == "__main__":
    main()