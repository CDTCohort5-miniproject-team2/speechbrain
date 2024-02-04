import torch
import scipy
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.features import STFT, ISTFT
from speechbrain.processing.multi_mic import Covariance
from speechbrain.processing.multi_mic import GccPhat, SrpPhat
from speechbrain.processing.multi_mic import DelaySum
import numpy as np

import TEAM2_utils

SAMPLING_RATE = 48000

WALL_MIC_LOCATIONS_MAPPING = {4: [-0.10, 0.00, 0.00],
                               5: [0.00, 0.00, 0.00],
                               6: [0.10, 0.00, 0.00],
                               7: [-0.05, -0.10, 0.00],
                               8: [0.05, -0.10, 0.00]}

def get_mic_locations(channels=(4, 5, 6, 7, 8)):
    # function to return locations of selected microphone channels, so that we can experiment with different permutations
    if channels:
        try:
            return torch.FloatTensor([WALL_MIC_LOCATIONS_MAPPING[channel] for channel in channels])
        except KeyError:
            print("Invalid channels specified, returned full mic array (channels 4-8) instead.")
    else:
        raise ValueError("At least one channel must be specified.")


def delaysum_beamforming(audio_tensor, mode="doa", save_file=False, save_fname="demo_audio"):
    """

    :param audio_tensor: 3D torch float-tensor: [batch, time (no. of samples), channels]
    :param mode: str - "doa" or "tdoa" - whether to use direction of arrival or time difference of arrival
    :param save_file: whether to save the beamformed audio for inspection
    :param save_fname: if saving file, the filename to be used - the suffix "_delaysumed_{mode}" will be appended
    :return: 3D torch float-tensor [batch, time(no. of samples), channels]
    """
    # Delay-and-Sum Beamforming - adapting code from SpeechBrain
    # https://colab.research.google.com/drive/1UVoYDUiIrwMpBTghQPbA6rC1mc9IBzi6?usp=sharing#scrollTo=JbLs2iHNBlW9
    # TODO: it seems like the beamforming operation shortened the test audio array by a few hundred samples (sr=16kHz),
    #  find out what's going on

    stft, istft = STFT(sample_rate=SAMPLING_RATE, n_fft=1200), ISTFT(sample_rate=SAMPLING_RATE, n_fft=1200)
    covariance = Covariance()
    delaysum = DelaySum()
    Xs = stft(audio_tensor)
    XXs = covariance(Xs)

    if mode == "doa":
        # if using directions of arrival
        srpphat = SrpPhat(mics=get_mic_locations())
        doas = srpphat(XXs)
        # TODO: at some point we might want to print out the estimated doas and check if it matches up with what we know
        #  i.e. driver is around X meters away from the microphones etc.
        Ys_ds = delaysum(Xs, doas, doa_mode=True, mics=get_mic_locations(), fs=SAMPLING_RATE)

    elif mode == "tdoa":
        # if using time difference of arrival
        gccphat = GccPhat()
        time_diff_of_arrival = gccphat(XXs)
        Ys_ds = delaysum(Xs, time_diff_of_arrival, doa_mode=False, mics=get_mic_locations(), fs=SAMPLING_RATE)
    else:
        raise ValueError("Mode must either be doa or tdoa.")

    ys_ds = istft(Ys_ds)
    # ys_ds has size (batch, n_samples, 1)
    if save_file:
        scipy.io.wavfile.write(f"{save_fname}_delaysumed_{mode}.wav", rate=SAMPLING_RATE, data=ys_ds.squeeze(0).numpy())

    return ys_ds

if __name__ == "__main__":

    demo_audio_fpath = "/home/paulgering/Documents/PGDip_modules/mini_project/ASR_models/test_kroto_data/18_12_2023/Audio/1702901102917_scenariov3_035.wav"
    segment_start, segment_end = int(4.0 * SAMPLING_RATE), int(8.0 * SAMPLING_RATE)
    full_audio_tensor = read_audio(demo_audio_fpath).unsqueeze(0)  # [batch, time (no. of samples), channels]
    audio_tensor = full_audio_tensor[:, segment_start:segment_end, 7:12] # channels 7-11 are wall mics

    scipy.io.wavfile.write("demo_audio_unbeamformed.wav", rate=SAMPLING_RATE, data=audio_tensor.squeeze(0).numpy())
    print(audio_tensor.size())
    delaysum_beamforming(audio_tensor, "demo_audio")
