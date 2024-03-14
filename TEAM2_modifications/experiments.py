import os
import audio_pipeline
import load_kroto_data
import process_transcripts
from pathlib import Path
import scipy
from scipy import io  # added during debugging
import scipy.io.wavfile  # added during debugging
import numpy as np
import TEAM2_utils
from time import time
import pandas as pd


class Experiment:
    def __init__(self, rq, data_csv_fpath, set_split="Training", raw_data_directory="kroto_data",
                 output_dir_suffix="w_whisper_large", asr_model_name="whisper-large-v3", test_only=False):
        """
        :param rq: the name of the RQ. Should be one of the following: "baseline", "adding_enhancer", "adding_separator",
        "separator_first", or enhancer_first"
        :param data_csv_fpath: filepath of data csv, this csv should at minimum include columns with the following headings:
            "Recording File reference", "Set"
        :param set_split: (str) specifies which partition to load: "Training", "Validation", or "Test"
        :param asr_model_name: ASR model to use e.g. "whisper-large-v3", "whisper-tiny.en"
        :return:
        """
        self.rq = rq
        self.data_csv_fpath = data_csv_fpath
        self.raw_data_directory = raw_data_directory
        self.side = self._decide_side()
        self.set_split = set_split
        self.asr_model_name = asr_model_name

        self.test_only = test_only

        self.designation = f"{TEAM2_utils.EXPERIMENT_DESIGNATION[self.rq]}_{output_dir_suffix}"

        self.parent_output_dir = Path(f"{raw_data_directory}/{self.designation}")
        self.output_dirs = [Path(f"{raw_data_directory}/{self.designation}/{subdir}") for subdir in
                            TEAM2_utils.EXPERIMENT_OUTPUT_DIRS]

    def _initialise_experiment(self):
        self.training_dataset = self._load_torch_set()
        self._make_output_dirs()
        self.customer_audio_pipeline = self._get_pipeline()
        self.server_audio_pipeline = None
        if self.side == "both":
            self.server_audio_pipeline = self._get_pipeline(server_pipeline=True)

    def _decide_side(self):
        if self.rq == "baseline":
            side = "both"
        else:
            side = "customer"
        return side

    def _make_output_dirs(self):
        if not self.parent_output_dir.exists():
            os.mkdir(self.parent_output_dir)
        for dirpath in self.output_dirs:
            if not dirpath.exists():
                os.mkdir(dirpath)
        # return output_dirs

    def _load_torch_set(self):
        kroto_data = load_kroto_data.RawKrotoData(self.data_csv_fpath, self.raw_data_directory)
        torch_dataset = kroto_data.get_torch_dataset(side=self.side, dataset_split=self.set_split)
        # "both" as in we want both customer and server side audio/transcripts etc.
        return torch_dataset

    def _get_pipeline(self, server_pipeline=False):
        if server_pipeline:
            # server does not need enhancing/separating
            return audio_pipeline.AudioPipeline(("aec", "asr"), asr_model_name=self.asr_model_name)

        return audio_pipeline.AudioPipeline(TEAM2_utils.EXPERIMENT_COMPONENTS[self.rq],
                                            asr_model_name=self.asr_model_name)

    def run_experiment(self):
        self._initialise_experiment()
        # TODO: add in loading partial experiment results to pick up from where stopped
        experiment_record_fpath = self.parent_output_dir/"experiment_duration_records.csv"
        experiment_record = pd.DataFrame(columns=["scenario_id",
                                                  "recording_duration",
                                                  "customer_total_processing_dur",
                                                  "customer_component1_dur",
                                                  "customer_component2_dur",
                                                  "customer_component3_dur",
                                                  "customer_component4_dur",
                                                  "server_total_processing_dur",
                                                  "server_aec_dur",
                                                  "server_asr_dur",
                                                  "end_to_end_processing_dur"])

        customer_time_record_cols = ["customer_total_processing_dur",
                                     "customer_component1_dur",
                                     "customer_component2_dur",
                                     "customer_component3_dur",
                                     "customer_component4_dur"]

        server_time_record_cols = ["server_total_processing_dur",
                                   "server_aec_dur",
                                   "server_asr_dur"]

        master_start_time = time()
        for i, (scenario_id,
                server_closetalk, customer_closetalk, single_wall_mic_array) in enumerate(self.training_dataset):
            saving_fpaths = [output_dir / f"{scenario_id}_{fpath_suffix}"
                             for output_dir, fpath_suffix
                             in zip(self.output_dirs, TEAM2_utils.EXPERIMENT_FILEPATH_SUFFIXES)]
            experiment_record.at[i, "scenario_id"] = scenario_id
            experiment_record.at[i, "recording_duration"] = len(server_closetalk)/16000

            endtoend_start_time = time()
            # customer's and server's processed arrays need to be saved for comparison against ground truth later
            c_processed_array, c_transcript, c_timelist = self.customer_audio_pipeline.run_inference_beta(
                single_wall_mic_array,
                server_closetalk)

            for time_item_i, time_item in enumerate(c_timelist):
                experiment_record.at[i, customer_time_record_cols[time_item_i]] = time_item


            scipy.io.wavfile.write(saving_fpaths[3], 16000, np.expand_dims(c_processed_array, axis=-1))
            customer_transcript_obj = process_transcripts.WhisperGeneratedTranscript(c_transcript, prefix="C")
            customer_transcript_obj.save_normalised_transcript_for_wer(saving_fpaths[2])

            if self.server_audio_pipeline:
                s_processed_array, s_transcript, s_timelist = self.server_audio_pipeline.run_inference_beta(server_closetalk,
                                                                                                          customer_closetalk)

                for time_item_i, time_item in enumerate(s_timelist):
                    experiment_record.at[i, server_time_record_cols[time_item_i]] = time_item

                scipy.io.wavfile.write(saving_fpaths[5], 16000, np.expand_dims(s_processed_array, axis=-1))
                server_transcript_obj = process_transcripts.WhisperGeneratedTranscript(s_transcript, prefix="S")
                server_transcript_obj.save_normalised_transcript_for_wer(saving_fpaths[4])
                server_transcript_obj.merge_transcripts_chronologically(customer_transcript_obj,
                                                                        save_fpath_for_nlp=saving_fpaths[0],
                                                                        save_fpath_for_wer=saving_fpaths[1])

            experiment_record.at[i, "end_to_end_processing_dur"] = time() - endtoend_start_time

            print(f"Processed file no. {i} - Audio duration: {len(server_closetalk) / 16000} - "
                  f"Processing duration: {time() - endtoend_start_time} - Cumulative processing duration: {time() - master_start_time}")

            if (i > 0) and (i % 5 == 0):
                # saving a record for every 5 files parsed
                experiment_record.to_csv(experiment_record_fpath)

            if i == 2 and self.test_only:
                break

        # saving a final record
        experiment_record.to_csv(experiment_record_fpath)

        print("EXPERIMENT COMPLETE.")


def main():
    baseline_exp = Experiment("baseline",
                              data_csv_fpath="kroto_data/demo_dataset_split.csv",
                              set_split="Training",
                              raw_data_directory="kroto_data")
    print("Experiment loaded.")

    # baseline_exp.run_experiment()


if __name__ == "__main__":
    main()
