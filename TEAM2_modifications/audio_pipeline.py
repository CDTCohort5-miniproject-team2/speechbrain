import torch
import sys
import numpy as np

import prepare_kroto_data
import _TEAM2_beamforming as bf

sys.path.append("../DTLN-aec-main")  # so that we can import run_aec below
print("Importing run_aec")
import run_aec
print("run_aec imported.")

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

import transformers


CATALOGUE = [
    # adding new recording sessions as they are available
    "test_kroto_data/18_12_2023",
    "test_kroto_data/01_02_24",
]
kroto_data = prepare_kroto_data.KrotoData(CATALOGUE[-1])

def get_test_sample(audio_fstem="20240201_114729_scenario_28", timeslice=(2.0, 13.6), mics=("wall_mics", "server_closetalk")):
    """
    Get some sample audio arrays for testing
    :param audio_fstem:
    :param timeslice:
    :param mics: tuple of strings - according to channel mapping, e.g. ("wall_mics", "server_closetalk")
    :return: 2D arrays in the specific order
    """
    mic_arrays = []
    for mic_name in mics:
        mic_array = kroto_data.get_demo_audio_array(audio_fname=audio_fstem+".wav", downsampled=True,
                                                         timeslice=timeslice, channel_name=mic_name)
        mic_arrays.append(mic_array)
    return mic_arrays

def initialise_beamforming(channels=(4, 5, 6, 7, 8), mode="doa"):
    """
    This is refactored from _TEAM2_beamforming.py
    :param channels: tuple of int (4, 5, 6, 7, 8)
    :return:
    """
    stft, istft = bf.STFT(sample_rate=bf.SAMPLING_RATE, n_fft=1200), bf.ISTFT(sample_rate=bf.SAMPLING_RATE, n_fft=1200)
    covariance = bf.Covariance()
    delaysum = bf.DelaySum()
    if mode == "doa":
        srpphat_or_gccphat = bf.SrpPhat(mics=bf.get_mic_locations(channels=channels))
    elif mode == "tdoa":
        srpphat_or_gccphat = bf.GccPhat()
    else:
        raise NotImplementedError("Specify an implemented method of beamforming.")

    initialised_beamformer = (stft, istft, covariance, delaysum, srpphat_or_gccphat)
    return initialised_beamformer

def do_beamforming(audio_array_2d, mode, channels, initialised_beamformer):
    audio_tensor = torch.FloatTensor(audio_array_2d.T).unsqueeze(0)
    beamformed_tensor = bf.delaysum_beamforming(audio_tensor, mode=mode, channels=channels,
                                                pre_instantiated=initialised_beamformer)
    beamformed_array = beamformed_tensor.squeeze((0, -1)).numpy()
    return beamformed_array

def initialise_aec(model_size=512):
    """
    Set up the DTLN AEC
    :param model_size: int - must be 128, 256 or 512
    :return: interpreter1, interpreter2 - these will be used by the do_aec function for performing AEC
    """
    if model_size not in [128, 256, 512]:
        raise ValueError("AEC component: model_size must be 128, 256, or 512.")
    aec_pretrained_fpath = f"../DTLN-aec-main/pretrained_models/dtln_aec_{model_size}"
    interpreter1, interpreter2 = run_aec.initialise_interpreters(model=aec_pretrained_fpath)
    return interpreter1, interpreter2

def do_aec(interpreter1, interpreter2, wall_mic_array_1d, server_closetalk_array_1d):
    """
    :param interpreter1: interpreter1 obtained from calling initialise_aec
    :param interpreter2: interpreter2 obtained from calling initialise_aec
    :param wall_mic_array_1d:
    :param server_closetalk_array_1d:
    :return: 1d-array after AEC
    """
    return run_aec.process_audio(interpreter1, interpreter2, wall_mic_array_1d, server_closetalk_array_1d)

def initialise_enhancing():
    print("Initialising enhancer model.")
    # model belongs to <class 'speechbrain.pretrained.interfaces.SepformerSeparation'>
    model = separator.from_hparams(source="speechbrain/sepformer-dns4-16k-enhancement",
                                   savedir="audio_pipeline_pretrained_models/sepformer-dns4-16k-enhancement")
    print("Enhancer model initialised.")
    return model

def initialise_asr():
    pass


def do_enhancing(audio_array_1d, model, normalise=True):
    audio_tensor = torch.FloatTensor(audio_array_1d).unsqueeze(0)
    # audio_tensor has size (1, n_samples)
    est_sources = model.separate_batch(audio_tensor)
    # est_sources has size (1, n_samples, 1)
    enhanced_array = est_sources[:, :, 0].detach().cpu().numpy().squeeze(0)
    # enhanced_array has shape (n_samples,)

    # check for clipping - adapted from DTLN-aec code
    if normalise and (np.max(enhanced_array) > 1):
        enhanced_array = enhanced_array / np.max(enhanced_array) * 0.99
    return enhanced_array

def do_asr(audio_array_1d):
    pass

def first_beamforming_then_aec():
    # First beamforming

    # Then AEC
    pass

def first_aec_then_beamforming():
    # First break apart the wall_mic_array into single channels

    # Then run AEC on each of these

    # Finally do beamforming
    pass

def enhancer_test(play_out=True):
    """
    WARNING: THIS ENHANCER MODEL SEEMS TO OVER-AMPLIFY THE SIGNAL.
    WE MAY NEED TO DO SOME POST-PROCESSING E.G. TO AVOID CLIPPING.
    :param play_out: if true, plays out the pre- and post-enhanced audio for comparison.
    Make sure your speaker/headphone volume is turned down sufficiently before testing
    :return:
    """
    model = initialise_enhancing()
    wall_mics_array, server_closetalk_array = get_test_sample()
    single_wall_mic_array = wall_mics_array[1, :]
    print("Enhancing array.")
    enhanced_array_normalised = do_enhancing(single_wall_mic_array, model, normalise=True)
    print("Array enhanced.")

    if play_out:
        print("Playing pre-enhanced array")
        prepare_kroto_data.play_audio_array(single_wall_mic_array)
        print("Playing enhanced array")
        prepare_kroto_data.play_audio_array(enhanced_array_normalised)


def main():
    enhancer_test()

if __name__ == "__main__":
    main()