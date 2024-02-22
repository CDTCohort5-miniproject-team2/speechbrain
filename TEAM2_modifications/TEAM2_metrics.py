import collections
import numpy as np
from scipy import signal

import audio_pipeline

from pb_bss_eval.evaluation import pesq, stoi, si_sdr  # FYI you also need to pip install cython

SAMPLE_RATE = 16000

def compute_signal_metrics(reference_audio, processed_audio, pesq_mode="wb"):

    reference_audio, processed_audio = _synchronise_target_and_estimate(reference_audio, processed_audio)

    #pesq mode "wb"/"nb" refers to wide-band and narrowband respectively
    pesq_value = pesq(reference=reference_audio, estimation=processed_audio, sample_rate=SAMPLE_RATE, mode=pesq_mode)
    stoi_value = stoi(reference=reference_audio, estimation=processed_audio, sample_rate=SAMPLE_RATE)
    si_sdr_value = si_sdr(reference=reference_audio, estimation=processed_audio)
    # TODO: si_sdr of customer_closetalk with itself is 101.7038478643427 when it should be positive infinite,
    #  need to double check if this has been implemented correctly

    return {"pesq": pesq_value, "stoi": stoi_value, "si_sdr": si_sdr_value}

def _synchronise_target_and_estimate(reference_audio, processed_audio, mode="simple_crop"):

    if mode == "simple_crop":
        length_diff = len(reference_audio) - len(processed_audio)
        if length_diff > 0:
            print("Cropping reference audio for alignment")
            reference_audio = reference_audio[:-length_diff]
        elif length_diff < 0:
            print("Cropping processed audio for alignment")
            processed_audio = processed_audio[:-length_diff]

    elif mode == "sync_and_crop":
        # TODO: not yet fully implemented, continue if necessary
        wall_mics_array, _ = audio_pipeline.get_test_sample()
        print(wall_mics_array.shape)
        reference_array, processed_array = wall_mics_array[0, :], wall_mics_array[3, :]

        print(sum([np.abs(x - y) for x, y in zip(reference_array, processed_array)]))

        correlation = signal.correlate(reference_array, processed_array, mode="full")
        lags = signal.correlation_lags(reference_array.size, processed_array.size, mode="full")
        print(correlation)
        lag = lags[np.argmax(correlation)]
        print(lag)

    else:
        raise NotImplementedError

    return reference_audio, processed_audio

def compute_wer(preds, targets):
    # https: // speechbrain.readthedocs.io / en / latest / API / speechbrain.utils.edit_distance.html
    wer = collections.Counter()
    # compute wer(pred, target)
    # compute batch wer(preds, targets)




def signal_metrics_test():
    wall_mics_array, server_closetalk, customer_closetalk = \
        audio_pipeline.get_test_sample(mics=("wall_mics", "server_closetalk", "customer_closetalk"))

    customer_closetalk = customer_closetalk.squeeze(axis=0)

    initialised_beamformer, aec_interpreter1, aec_interpreter2, enhancer_model, asr_model = \
        audio_pipeline.initialise_audio_pipeline()

    processed_array = audio_pipeline.first_beamforming_then_aec(aec_interpreter1, aec_interpreter2, wall_mics_array,
                                                                server_closetalk, initialised_beamformer)

    processed_array = audio_pipeline.do_enhancing(processed_array, enhancer_model, normalise=True)

    results_dict1 = compute_signal_metrics(customer_closetalk, customer_closetalk)
    results_dict2 = compute_signal_metrics(customer_closetalk, processed_array)

    print(results_dict1)
    print(results_dict2)

def main():
    pass


if __name__ == "__main__":
    main()
