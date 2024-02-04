import torch
import scipy
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.features import STFT, ISTFT
from speechbrain.processing.multi_mic import Covariance
from speechbrain.processing.multi_mic import GccPhat, SrpPhat
from speechbrain.processing.multi_mic import DelaySum

SAMPLING_RATE = 48000

# TODO: need to return to lab and take measurements (each microphone's distance (in metres) from a fixed reference point)
WALL_MICS_LOCATIONS = torch.FloatTensor([
    [-0.10, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [0.10, 0.00, 0.00],
    [-0.05, -0.10, 0.00],
    [0.05, -0.10, 0.00],
])

def delaysum_beamforming(audio_tensor, save_fname="demo_audio"):
    # Delay-and-Sum Beamforming - adapting code from SpeechBrain
    # https://colab.research.google.com/drive/1UVoYDUiIrwMpBTghQPbA6rC1mc9IBzi6?usp=sharing#scrollTo=JbLs2iHNBlW9
    stft, istft = STFT(sample_rate=SAMPLING_RATE, n_fft=1200), ISTFT(sample_rate=SAMPLING_RATE, n_fft=1200)
    covariance, gccphat = Covariance(), GccPhat()
    delaysum = DelaySum()
    # TODO: for now use time differences of arrival
    #  - once we confirm WALL_MICS_LOCATIONS, we can try using directions of arrival and pass in extra arguments to DelaySum()
    Xs = stft(audio_tensor)
    XXs = covariance(Xs)
    srpphat = SrpPhat(mics=WALL_MICS_LOCATIONS)
    doas = srpphat(XXs)
    time_diff_of_arrival = gccphat(XXs)
    Ys_ds = delaysum(Xs, doas, doa_mode=True, mics=WALL_MICS_LOCATIONS, fs=SAMPLING_RATE)
    ys_ds = istft(Ys_ds)
    print(ys_ds.size())
    scipy.io.wavfile.write(f"{save_fname}_delaysum_beamformed.wav", rate=SAMPLING_RATE, data=ys_ds.squeeze(0).numpy())


if __name__ == "__main__":

    demo_audio_fpath = "/home/paulgering/Documents/PGDip_modules/mini_project/ASR_models/test_kroto_data/18_12_2023/Audio/1702901102917_scenariov3_035.wav"
    segment_start, segment_end = int(4.0 * SAMPLING_RATE), int(8.0 * SAMPLING_RATE)
    full_audio_tensor = read_audio(demo_audio_fpath).unsqueeze(0)  # [batch, time (no. of samples), channels]
    audio_tensor = full_audio_tensor[:, segment_start:segment_end, 7:12] # channels 7-11 are wall mics
    scipy.io.wavfile.write("demo_audio_unbeamformed.wav", rate=SAMPLING_RATE, data=audio_tensor.squeeze(0).numpy())
    print(audio_tensor.size())
    delaysum_beamforming(audio_tensor, "demo_audio")
