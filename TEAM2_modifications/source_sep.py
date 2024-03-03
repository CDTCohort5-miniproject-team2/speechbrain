import numpy as np
import librosa
import scipy.signal as ss
import sounddevice as sd
from ssspy.bss.iva import AuxLaplaceIVA


def load_and_make_stereo(audio_file_path):
    audio_data, sample_rate = librosa.load(audio_file_path, sr=None)

    shifted_audio = np.roll(audio_data, 1)
    stereo_audio = np.array([audio_data, shifted_audio])
    return stereo_audio


def _do_source_sep(audio_array):
    n_fft, hop_length = 4096, 2048
    window = "hann"

    _, _, spectrogram_mix = ss.stft(
       audio_array,
       window=window,
       nperseg=n_fft,
       noverlap=n_fft-hop_length
    )

    iva = AuxLaplaceIVA()
    spectrogram_est = iva(spectrogram_mix)

    _, waveform_est = ss.istft(
       spectrogram_est,
       window=window,
       nperseg=n_fft,
       noverlap=n_fft-hop_length
    )

    waveforms = []
    for idx, waveform in enumerate(waveform_est):
        waveforms.append(waveform)

    return waveforms


def main(audio_file_path):
    stereo_audio = load_and_make_stereo(audio_file_path)
    waveforms = _do_source_sep(stereo_audio)

    for idx, waveform in enumerate(waveforms):
        print("Estimated source: {}".format(idx + 1))
        sd.play(waveform.T, samplerate=16000)
        sd.wait()

    return waveforms


if __name__ == "__main__":
    main('path_to_single_channel_aec_output_audio_file.wav')
