import numpy as np
import torch

import _TEAM2_beamforming as bf
import prepare_kroto_data


CATALOGUE = [
    # adding new recording sessions as they are available
    "test_kroto_data/18_12_2023",
    "test_kroto_data/01_02_24",
]

# useful test samples of:
# customer speech to test beamforming only - 20240201_110241_scenario_2.wav (4.6, 15.2)
# server and then customer speech to test AEC+beamforming - 20240201_110241_scenario_2.wav (1.7, 15.2)

kroto_data = prepare_kroto_data.KrotoData(CATALOGUE[-1])
# kroto_data.separate_channels_and_save_audio() # only need to run once

# demo_customer_array has shape (n_channels, n_samples)
demo_customer_array = kroto_data.get_demo_audio_array(audio_fname="20240201_110241_scenario_2.wav",
                                             downsampled=True, timeslice=(4.6, 15.2), channel_name="wall_mics")

# demo_joint_array = kroto_data.get_demo_audio_array(audio_fname="20240201_110241_scenario_2.wav",
#                                              downsampled=True, timeslice=(1.7, 15.2), channel_name="wall_mics")

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

# NOTES: delay and sum beamforming seems to be reducing amplitude
# prepare_kroto_data.play_audio_array(demo_customer_array)
# prepare_kroto_data.play_audio_array(top_centre_array)
# prepare_kroto_data.play_audio_array(beamformed_array_doa)
# prepare_kroto_data.play_audio_array(beamformed_array_tdoa)

for audio_array in [averaged_array, top_centre_array, beamformed_array_doa, beamformed_array_tdoa]:
    print(audio_array.shape)
    avg_displacement = np.mean([np.abs(s) for s in audio_array])
    print(avg_displacement)

