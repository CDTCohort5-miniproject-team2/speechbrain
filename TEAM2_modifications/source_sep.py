import numpy as np
import librosa
import scipy.signal as ss
import sounddevice as sd
from ssspy.bss.iva import AuxLaplaceIVA


def make_stereo(audio_array_1d):
    shifted_audio = np.roll(audio_array_1d, 1)
    stereo_audio = np.array([audio_array_1d, shifted_audio])
    return stereo_audio


def do_source_sep(separator_model, audio_array):
    n_fft, hop_length = 4096, 2048
    window = "hann"

    _, _, spectrogram_mix = ss.stft(
        audio_array,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length
    )

    spectrogram_est = separator_model(spectrogram_mix)

    _, waveform_est = ss.istft(
        spectrogram_est,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length
    )

    waveforms = []
    for idx, waveform in enumerate(waveform_est):
        waveforms.append(waveform)

    return waveforms


def main(audio_file_path):
    audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
    stereo_audio = make_stereo(audio_data)
    iva = AuxLaplaceIVA()

    waveforms = do_source_sep(iva, stereo_audio)

    for idx, waveform in enumerate(waveforms):
        print("Estimated source: {}".format(idx + 1))
        sd.play(waveform.T, samplerate=sample_rate)
        sd.wait()

    return waveforms


if __name__ == "__main__":
    main('/Users/yao/Downloads/DTLN-aec/results/c4_mic.wav')
    # main('path_to_single_channel_aec_output_audio_file.wav')

