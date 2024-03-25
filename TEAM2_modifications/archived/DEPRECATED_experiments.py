import os

import audio_pipeline
import load_kroto_data
import process_transcripts
from pathlib import Path
import scipy
from scipy import io  # added during debugging
import scipy.io.wavfile  # added during debugging

OUTPUT_DIRS = ["merged_pred_transcript_for_nlp",
               "merged_pred_transcript_for_wer",
               "customer_pred_transcript",
               "customer_processed_array",
               "server_pred_transcript",
               "server_processed_array"]

FILEPATH_SUFFIXES = ["merged_pred_transcript_for_nlp.txt",
                     "merged_pred_transcript_for_wer.txt",
                     "customer_pred_transcript.txt",
                     "customer_processed_array.wav",
                     "server_pred_transcript.txt",
                     "server_processed_array.wav"]


def baseline():
    # TODO: 1. load the training dataset of Kroto data as a torch dataset
    #  2. set up one audio pipeline (aec, asr) for customer, another one (aec, asr) for server
    #  3. run inference and save down
    #       - all processed audio for signal metrics
    #       - all transcripts (customer, server, merged) for wer
    #       - merged but unnormalised transcript for NLP parsing purpose
    #  4. compute metrics:
    #       - signal metrics w.r.t. ground truth server closetalk and customer_closetalk signals respectively
    #       - WER w.r.t. ground truth transcripts

    data_csv_fpath = ""  # TODO: confirm csv filepath
    raw_data_directory = "kroto_data"

    designation = "baseline_w_whisper_large"

    baseline_output_dirs = [Path(f"kroto_data/{designation}/{subdir}") for subdir in OUTPUT_DIRS]
    for dirpath in baseline_output_dirs:
        if not dirpath.exists():
            os.mkdir(dirpath)

    kroto_data = load_kroto_data.RawKrotoData(data_csv_fpath, raw_data_directory)
    training_dataset = kroto_data.get_torch_dataset(side="both")
    # "both" as in we want both customer and server side audio/transcripts etc.

    customer_audio_pipeline = audio_pipeline.AudioPipeline("aec", "asr")
    server_audio_pipeline = audio_pipeline.AudioPipeline("aec", "asr")

    for i, (scenario_id, num_of_passengers, has_noise, duration_of_recording,
            server_closetalk, customer_closetalk, single_wall_mic_array,
            server_gt_transcript, customer_gt_transcript, merged_gt_transcript) in enumerate(training_dataset):
        saving_fpaths = [baseline_output_dir / f"{scenario_id}_{fpath_suffix}"
                         for baseline_output_dir, fpath_suffix
                         in zip(baseline_output_dirs, FILEPATH_SUFFIXES)]

        c_processed_array, c_transcript = customer_audio_pipeline.run_inference(single_wall_mic_array, server_closetalk)
        s_processed_array, s_transcript = server_audio_pipeline.run_inference(server_closetalk, customer_closetalk)

        # customer's and server's processed arrays need to be saved for comparison against ground truth later
        scipy.io.wavfile.write(saving_fpaths[3], 16000, c_processed_array.unsqueeze(-1))
        scipy.io.wavfile.write(saving_fpaths[5], 16000, s_processed_array.unsqueeze(-1))

        customer_transcript_obj = process_transcripts.WhisperGeneratedTranscript(c_transcript)
        server_transcript_obj = process_transcripts.WhisperGeneratedTranscript(s_transcript)

        customer_transcript_obj.save_normalised_transcript_for_wer(saving_fpaths[2])
        server_transcript_obj.save_normalised_transcript_for_wer(saving_fpaths[4])

        server_transcript_obj.merge_transcripts_chronologically(customer_gt_transcript,
                                                                save_fpath_for_nlp=saving_fpaths[0],
                                                                save_fpath_for_wer=saving_fpaths[1])


def adding_enhancer():
    # TODO: 1. load the training dataset of Kroto data as a torch dataset
    #  2. set up one audio pipeline (aec, enhancer, asr) for customer speech
    #  3. run inference and save down
    #       - the processed customer audio for signal metrics
    #       - customer transcript for wer
    #  4. compute metrics:
    #       - signal metrics w.r.t. ground truth customer_closetalk
    #       - WER w.r.t. ground truth customer transcript
    pass


def swapping_components_around_separator_first():
    # TODO: 1. load the training dataset of Kroto data as a torch dataset
    #  2. set up one audio pipeline (aec, separator, enhancer, asr) for customer speech
    #  3. run inference and save down
    #       - the processed customer audio for signal metrics
    #       - customer transcript for wer
    #  4. compute metrics:
    #       - signal metrics w.r.t. ground truth customer_closetalk
    #       - WER w.r.t. ground truth customer transcript
    pass


def swapping_components_around_enhancer_first():
    # TODO: 1. load the training dataset of Kroto data as a torch dataset
    #  2. set up one audio pipeline (aec, separator, enhancer, asr) for customer speech
    #  3. run inference and save down
    #       - the processed customer audio for signal metrics
    #       - customer transcript for wer
    #  4. compute metrics:
    #       - signal metrics w.r.t. ground truth customer_closetalk
    #       - WER w.r.t. ground truth customer transcript
    pass
