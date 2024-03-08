import collections
import numpy as np
from scipy import signal
import pysepm
import audio_pipeline
from scipy.io import wavfile
import pysepm

SAMPLE_RATE = 16000

def _synchronise_target_and_estimate(reference_audio, processed_audio, mode="simple_crop"):
    if mode == "simple_crop":
        length_diff = len(reference_audio) - len(processed_audio)
        if length_diff > 0:
            print("Cropping reference audio for alignment")
            reference_audio = reference_audio[:-length_diff]
        elif length_diff < 0:
            print("Cropping processed audio for alignment")
            processed_audio = processed_audio[:-length_diff]
    else:
        raise NotImplementedError

    return reference_audio, processed_audio

def compute_signal_metrics(reference_audio, processed_audio):

    reference_audio, processed_audio = _synchronise_target_and_estimate(reference_audio, processed_audio)

    pesq_value = pysepm.pesq(reference_audio, processed_audio, SAMPLE_RATE)
    stoi_value = pysepm.stoi(reference_audio, processed_audio, SAMPLE_RATE)
    composite_score = pysepm.composite(reference_audio, processed_audio, SAMPLE_RATE)

    return pesq_value, stoi_value, composite_score


def compute_wer(preds, targets):
    # https: // speechbrain.readthedocs.io / en / latest / API / speechbrain.utils.edit_distance.html
    wer = collections.Counter()
    # compute wer(pred, target)
    # compute batch wer(preds, targets)

def signal_metrics_test():
    pass

def main():
    pass


if __name__ == "__main__":
    main()
