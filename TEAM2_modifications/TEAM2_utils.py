import numpy as np
import sounddevice as sd
import re
from spellchecker import SpellChecker
from num2words import num2words

CHANNEL_MAPPING = {
    # "dashboard_mics": [0, 1, 2, 3],  # no longer used in our experiments
    # "wall_mics": [4, 5, 6, 7, 8],
    # this is commented out because we're no longer using all 5 mic arrays, but just the top centre one
    # "passengers_closetalk": [10, 11],  # no longer used in our experiments
    "customer_closetalk": [9],
    "server_closetalk": [12],
    "top_centre_wall_mic": [5],
}

GROUND_TRUTH_SUFFIXES = ["gt_merged_transcript.txt",
                         "gt_customer_transcript.txt",
                         "top_centre_wall_mic.wav",
                         "customer_closetalk.wav",
                         "gt_server_transcript.txt",
                         "server_closetalk.wav"]

GROUND_TRUTH_DIRS = ["merged_gt_transcripts",
                     "customer_gt_transcripts",
                     "Audio_top_centre_wall_mic",
                     "Audio_customer_closetalk",
                     "server_gt_transcripts",
                     "Audio_server_closetalk"]

EXPERIMENT_OUTPUT_DIRS = ["merged_pred_transcript_for_nlp",
                          "merged_pred_transcript_for_wer",
                          "customer_pred_transcript",
                          "customer_processed_array",
                          "server_pred_transcript",
                          "server_processed_array"]

EXPERIMENT_FILEPATH_SUFFIXES = ["merged_pred_transcript_for_nlp.txt",
                                "merged_pred_transcript_for_wer.txt",
                                "customer_pred_transcript.txt",
                                "customer_processed_array.wav",
                                "server_pred_transcript.txt",
                                "server_processed_array.wav"]

EXPERIMENT_DESIGNATION = {
    "baseline": "baseline",
    "adding_enhancer": "aec_enhancer_asr",
    "adding_separator": "aec_separator_asr",
    "separator_first": "aec_separator_enhancer_asr",
    "enhancer_first": "aec_enhancer_separator_asr"
}

EXPERIMENT_COMPONENTS = {
    "baseline": ("aec", "asr"),
    "adding_enhancer": ("aec", "enhancer", "asr"),
    "adding_separator": ("aec", "separator", "asr"),
    "separator_first": ("aec", "separator", "enhancer", "asr"),
    "enhancer_first": ("aec", "enhancer", "separator", "asr")
}


def get_channels_by_name(multichannel_audio_array, target_channel_name):
    """

    :param multichannel_audio_array:
    :param target_channel_name:
    :return:
    """
    assert multichannel_audio_array.shape[0] in [13, 16], \
        "The function_get_channels_by_name can only operate on the full 13- or 16-channel array."
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

    start_sample, end_sample = int(start_time * sr), int(end_time * sr)

    if end_sample >= audio_array.shape[-1]:
        end_sample = audio_array.shape[-1] - 1
        print(f"End time is greater than audio length, returning a shorter timeslice instead "
              f"from {start_time} seconds to {end_sample / sr} seconds.")

    # TEAM2 fyi: "..." in slicing numpy arrays allows it to accept arrays of an arbitrary number of dimensions
    return audio_array[..., start_sample:end_sample]


def main():
    pass


if __name__ == "__main__":
    main()
