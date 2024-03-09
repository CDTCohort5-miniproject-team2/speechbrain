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


class Experiment:
    def __init__(self, rq, data_csv_fpath, set_split="Training", raw_data_directory="kroto_data"):
        """
        :param rq: the name of the RQ. Should be one of the following: "baseline", "adding_enhancer", "adding_separator",
        "separator_first", or enhancer_first"
        :param set_split: (str) specifies which partition to load: "Training", "Validation", or "Test"
        :return:
        """
        self.rq = rq
        self.data_csv_fpath = data_csv_fpath
        self.raw_data_directory = raw_data_directory
        self.side = self._decide_side()
        self.set_split = set_split

        self.designation = TEAM2_utils.EXPERIMENT_DESIGNATION[self.rq]

        self.parent_output_dir = Path(f"kroto_data/{self.designation}")
        self.output_dirs = [Path(f"kroto_data/{self.designation}/{subdir}") for subdir in TEAM2_utils.EXPERIMENT_OUTPUT_DIRS]

    def _initialise_experiment(self):
        self.training_dataset = self._load_torch_set()
        self._make_output_dirs()
        self.customer_audio_pipeline = self._get_pipeline()
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
            return audio_pipeline.AudioPipeline(("aec", "asr"))

        return audio_pipeline.AudioPipeline(TEAM2_utils.EXPERIMENT_COMPONENTS[self.rq])

    def run_experiment(self):
        self._initialise_experiment()
        for i, (scenario_id,
                server_closetalk, customer_closetalk, single_wall_mic_array) in enumerate(self.training_dataset):
            saving_fpaths = [output_dir / f"{scenario_id}_{fpath_suffix}"
                             for output_dir, fpath_suffix
                             in zip(self.output_dirs, TEAM2_utils.EXPERIMENT_FILEPATH_SUFFIXES)]
            # customer's and server's processed arrays need to be saved for comparison against ground truth later
            c_processed_array, c_transcript = self.customer_audio_pipeline.run_inference_beta(single_wall_mic_array,
                                                                                              server_closetalk)
            scipy.io.wavfile.write(saving_fpaths[3], 16000, np.expand_dims(c_processed_array, axis=-1))
            customer_transcript_obj = process_transcripts.WhisperGeneratedTranscript(c_transcript, prefix="C")
            customer_transcript_obj.save_normalised_transcript_for_wer(saving_fpaths[2])

            if self.server_audio_pipeline:
                s_processed_array, s_transcript = self.server_audio_pipeline.run_inference_beta(server_closetalk,
                                                                                                customer_closetalk)
                scipy.io.wavfile.write(saving_fpaths[5], 16000, np.expand_dims(s_processed_array, axis=-1))
                server_transcript_obj = process_transcripts.WhisperGeneratedTranscript(s_transcript, prefix="S")
                server_transcript_obj.save_normalised_transcript_for_wer(saving_fpaths[4])
                server_transcript_obj.merge_transcripts_chronologically(customer_transcript_obj,
                                                                        save_fpath_for_nlp=saving_fpaths[0],
                                                                        save_fpath_for_wer=saving_fpaths[1])


def main():
    baseline_exp = Experiment("baseline",
                              data_csv_fpath="kroto_data/demo_dataset_split.csv",
                              set_split="Training",
                              raw_data_directory="kroto_data")
    print("Experiment loaded.")

    # baseline_exp.run_experiment()


if __name__ == "__main__":
    main()
