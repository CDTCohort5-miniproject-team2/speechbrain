import stable_whisper

import audio_pipeline
import prepare_kroto_data
from time import time

def main():
    wall_mics, customer_closetalk, server_closetalk = \
        audio_pipeline.get_test_sample("20240201_104809_scenario_10",
                                       timeslice=(0.0, 30.0),
                                       mics=("wall_mics", "customer_closetalk", "server_closetalk"))

    inference_audio_length = server_closetalk.shape[-1] / 16000
    # the whole of 20240201_104809_scenario_10 is 84.48 seconds long

    aec_interpreter1, aec_interpreter2 = audio_pipeline.initialise_aec()
    # asr_model = audio_pipeline.initialise_asr(model_name="large",
    #                                           long_transcription=True, use_stable_ts=True)

    asr_model = audio_pipeline.initialise_stable_distil_asr()

    wall_mic, customer_closetalk, server_closetalk = wall_mics[0, :], \
                                                     customer_closetalk.squeeze(0), \
                                                     server_closetalk.squeeze(0)

    start_time = time()
    print("Doing AEC")
    wall_mic_aeced = audio_pipeline.do_aec(aec_interpreter1, aec_interpreter2, wall_mic, server_closetalk)
    server_closetalk = audio_pipeline.do_aec(aec_interpreter1, aec_interpreter2, server_closetalk, customer_closetalk)

    aec_end_time = time()
    print(f"AEC took {aec_end_time-start_time}. Doing enhancing")
    # wall_mic_enhanced = audio_pipeline.do_enhancing(wall_mic_aeced, enhancer)
    wall_mic_enhanced = wall_mic_aeced
    enhancing_end_time = time()
    print(f"Enhancing took {enhancing_end_time-aec_end_time}. Doing ASR")

    customer_asr_output = audio_pipeline.do_asr(wall_mic_enhanced, asr_model, stable_ts=True)
    server_asr_output = audio_pipeline.do_asr(server_closetalk, asr_model, stable_ts=True)

    print(f"ASR took {time()-enhancing_end_time}")
    print(f"Inference on {inference_audio_length}-second audio took {time()-start_time} long.")
    print(customer_asr_output)
    print(server_asr_output)


    # stable_ts only
    wall_mic_output_string = stable_whisper.result_to_tsv(customer_asr_output, filepath=None,
                                                          segment_level=True, word_level=False)

    server_output_string = stable_whisper.result_to_tsv(server_asr_output, filepath=None,
                                                        segment_level=True, word_level=False)

    print(wall_mic_output_string)
    print(server_output_string)

if __name__ == "__main__":
    main()
