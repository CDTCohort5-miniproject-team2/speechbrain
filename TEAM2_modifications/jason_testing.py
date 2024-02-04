import numpy as np
import torch
from pathlib import Path
import scipy
from scipy import io  # added during debugging
import scipy.io.wavfile  #  added during debugging

import _TEAM2_beamforming as bf
import prepare_kroto_data

import sys
sys.path.append("../DTLN-aec-main")  # so that we can import run_aec.py

import run_aec

import full_audio_pipeline

# useful test samples of:
# customer speech to test beamforming only
# - 20240201_110241_scenario_2.wav (4.6, 15.2)
# server and then customer speech to test AEC+beamforming
# - 20240201_110241_scenario_2.wav (1.7, 15.2)
# - 20240201_114729_scenario_28.wav (2.0, 13.6)

kroto_data = prepare_kroto_data.KrotoData(full_audio_pipeline.CATALOGUE[-1])
# kroto_data.separate_channels_and_save_audio() # only needed to run once
def aec_audio_array(audio_fstem="20240201_114729_scenario_28", timeslice=(2.0, 13.6)):

    aec_pretrained_fpath = "../DTLN-aec-main/pretrained_models/dtln_aec_512"
    wall_mic_array = kroto_data.get_demo_audio_array(audio_fname=audio_fstem+".wav", downsampled=True,
                                                     timeslice=timeslice, channel_name="wall_mics")
    top_centre_mic_array = wall_mic_array[2, :]
    server_mic_array = kroto_data.get_demo_audio_array(audio_fname=audio_fstem+".wav", downsampled=True,
                                                       timeslice=timeslice, channel_name="server_closetalk").squeeze(0)

    interpreter1, interpreter2 = run_aec.initialise_interpreters(model=aec_pretrained_fpath)

    post_aec_array = run_aec.process_audio(interpreter1, interpreter2,
                                           wall_mic_1d_array=top_centre_mic_array,
                                           server_closetalk_1d_array=server_mic_array)
    print(f"Pre AEC, top centre mic array has shape {top_centre_mic_array.shape}; "
          f"and server mic array has shape {server_mic_array.shape}")
    print("AEC successfully run.")
    print(f"Post AEC, array has shape {post_aec_array.shape}")

aec_audio_array()

def write_to_aec_input_dir(audio_fstem="20240201_114729_scenario_28", timeslice=(2.0, 13.6)):
    # NOTE: this returns array shape (n_samples,)
    wall_mic_array = kroto_data.get_demo_audio_array(audio_fname=audio_fstem+".wav", downsampled=True,
                                                     timeslice=timeslice, channel_name="wall_mics")
    top_centre_mic_array = wall_mic_array[2, :]

    # whereas this returns array shape (1, n_samples)
    server_mic_array = kroto_data.get_demo_audio_array(audio_fname=audio_fstem+".wav", downsampled=True,
                                                       timeslice=timeslice, channel_name="server_closetalk").squeeze(0)
    # squeezing array into (n_samples,) shape

    aec_input_dir = Path("../DTLN-aec-main/test_audio_files/demo_input_folder")
    lpb_fpath = aec_input_dir / f"{audio_fstem}_lpb.wav"
    mic_fpath = aec_input_dir / f"{audio_fstem}_mic.wav"

    scipy.io.wavfile.write(mic_fpath, 16000, top_centre_mic_array)
    scipy.io.wavfile.write(lpb_fpath, 16000, server_mic_array)

def beamforming_test():
    # demo_customer_array has shape (n_channels, n_samples)
    demo_customer_array = kroto_data.get_demo_audio_array(audio_fname="20240201_110241_scenario_2.wav",
                                                          downsampled=True, timeslice=(4.6, 15.2), channel_name="wall_mics")

    demo_joint_array = kroto_data.get_demo_audio_array(audio_fname="20240201_110241_scenario_2.wav",
                                                       downsampled=True, timeslice=(1.7, 15.2), channel_name="wall_mics")

    # from an informal listening test, there are no audible differences between taking the average of all 5 wall mic arrays
    # vs taking only the signal from the top-centre mic. But FYI below is how we obtain one as opposed to the other
    averaged_array = np.mean(demo_customer_array, axis=0)
    top_centre_array = demo_customer_array[1, :]

    audio_tensor = torch.FloatTensor(demo_customer_array.T).unsqueeze(0)

    print("Beamforming with DOA")
    beamformed_tensor_doa = bf.delaysum_beamforming(audio_tensor, mode="doa")
    print("Beamforming with TDOA")
    beamformed_tensor_tdoa = bf.delaysum_beamforming(audio_tensor, mode="tdoa")
    print("Done")

    # TODO: it seems like the beamforming operation shortened the test audio array by a few hundred samples (sr=16kHz),
    #  find out what's going on
    beamformed_array_doa = beamformed_tensor_doa.squeeze((0, -1)).numpy()
    beamformed_array_tdoa = beamformed_tensor_tdoa.squeeze((0, -1)).numpy()

    # TODO: delay and sum beamforming seems to be only reducing overall amplitude without doing much; to investigate
    # prepare_kroto_data.play_audio_array(demo_customer_array)
    # prepare_kroto_data.play_audio_array(top_centre_array)
    # prepare_kroto_data.play_audio_array(beamformed_array_doa)
    # prepare_kroto_data.play_audio_array(beamformed_array_tdoa)

    # this confirms that we seem to be losing overall amplitude after beamforming, which shouldn't be happening
    for audio_array in [averaged_array, top_centre_array, beamformed_array_doa, beamformed_array_tdoa]:
        print(audio_array.shape)
        avg_displacement = np.mean([np.abs(s) for s in audio_array])
        print(avg_displacement)


