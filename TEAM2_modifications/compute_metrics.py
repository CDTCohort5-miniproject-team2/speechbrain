import collections
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
from speechbrain.utils.edit_distance import accumulatable_wer_stats
import TEAM2_utils
import pandas as pd
import librosa
import warnings
from numba import NumbaDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
from tqdm import tqdm
import re
from pb_bss_eval.evaluation import pesq, stoi  # FYI you also need to pip install cython

SAMPLE_RATE = 16000


class ExperimentResults:
    def __init__(self, rq, data_csv_fpath, set_split="Training", data_directory="kroto_data", save_as_csv=True):
        """
        :param rq: the name of the RQ. Should be one of the following: "baseline", "adding_enhancer", "adding_separator",
        "separator_first", or enhancer_first"
        :param set_split: (str) specifies which partition to load: "Training", "Validation", or "Test"
        :return:
        """

        self.rq = rq
        self.set_split = set_split
        self.data_directory = data_directory
        self.save_as_csv = save_as_csv
        self.designation = f"{TEAM2_utils.EXPERIMENT_DESIGNATION[self.rq]}_w_medium_en"
        self.parent_output_dir = Path(f"kroto_data/{self.designation}")
        self.save_csv_fpath = self.parent_output_dir / f"{self.designation}_results.csv"
        self.output_dirs = [Path(f"kroto_data/{self.designation}/{subdir}") for subdir in TEAM2_utils.EXPERIMENT_OUTPUT_DIRS]
        self.merged_pred_nlp_dir, self.merged_pred_wer_dir, self.customer_pred_txt_dir, \
        self.customer_pred_wav_dir, self.server_pred_txt_dir, self.server_pred_wav_dir = self.output_dirs
        self._check_output_dirs()

        self.original_df = pd.read_csv(data_csv_fpath)
        self.original_df["scenario_id"] = [fname.replace(".wav", "").replace("16k_", "") for fname in
                                           self.original_df["Recording File reference"]]

    def _check_output_dirs(self):
        if not self.parent_output_dir.exists():
            raise FileNotFoundError("Experiment results not found. Please make sure correct directory is specified."
                                    "The directory you specified was", self.parent_output_dir)
        for output_dir in self.output_dirs:
            if not output_dir.exists():
                print("Warning: result subdirectory not found:", output_dir)
                # this is a warning only, as it might be possible that some are not used

    def get_gt_transcript_prefix_from_scenario_ids(self):

        fname_reobj = re.compile(r"(16k_)?(C_)?(?P<date>\d{8})_?(?P<time>\d{6})_?(scenario_)?(?P<scenario_id>\d{1,3})")

        unified_scenario_ids = []
        for item in self.original_df["scenario_id"]:
            item_match = re.search(fname_reobj, item)
            if item_match:
                unified_scenario_ids.append("-".join([item_match["date"], item_match["time"], item_match["scenario_id"]]))

        merged_gt_dir = Path(f"{self.data_directory}/{TEAM2_utils.GROUND_TRUTH_DIRS[0]}")

        gt_transcript_prefixes = ["" for _ in unified_scenario_ids]

        for fname in merged_gt_dir.glob("*.txt"):
            fname_match = re.search(fname_reobj, fname.stem)
            if fname_match:
                fname_reformed = "-".join([fname_match["date"], fname_match["time"], fname_match["scenario_id"]])
                try:
                    fname_idx = unified_scenario_ids.index(fname_reformed)
                except ValueError:
                    print(f"{fname.stem} is not found.")
                else:
                    gt_transcript_prefixes[fname_idx] = fname.name.replace("_gt_merged_transcript.txt", "")

            else:
                print(f"Regex couldn't process {fname}")

        return gt_transcript_prefixes

    def compute_metrics(self):

        new_df_col_names = ["pesq", "stoi",
                            "wer", "num_tokens_in_gt_transcript"]

        if self.rq == "baseline":
            new_df_col_names += ["server_wer", "server_num_tokens_in_gt_transcript", "merged_wer", "merged_num_tokens_in_gt_transcript"]

        for col_name in new_df_col_names:
            self.original_df[col_name] = np.nan

        self.original_df["already_parsed"] = False

        filter_by_columns = ["already_parsed", "scenario_id", "Set", "has_noise", "num_passengers",
                             "audio_duration_in_s"] + new_df_col_names
        if self.save_csv_fpath.exists():
            print("Continuing from previous results.")
            self.results_df = pd.read_csv(self.save_csv_fpath)
        else:
            print("Making new results CSV.")
            self.results_df = self.original_df[filter_by_columns].copy()

        print(self.results_df.head())
        print(self.results_df.info())

        gt_transcript_prefixes = self.get_gt_transcript_prefix_from_scenario_ids()
        for i, scenario_id in enumerate(self.results_df["scenario_id"]):
            print(f"Parsing file no. {i}: {scenario_id}")
            if self.results_df["Set"][i] != self.set_split:
                print(f"Skipping no. {i} - not used")
                continue
            elif self.results_df.at[i, "already_parsed"]:
                print(self.results_df.at[i, "already_parsed"])
                print(f"Skipping no. {i} - already parsed")
                continue
            # elif i in [210]:
            #     # 20240207_113810_scenario_18,
            #     print("Known problem with signal - skipped")
            #     continue
            else:

                merged_pred_nlp_fpath, merged_pred_wer_fpath, customer_pred_txt_fpath, \
                customer_pred_wav_fpath, server_pred_txt_fpath, server_pred_wav_fpath = \
                    [subdir / f"{scenario_id}_{suf}"
                     for subdir, suf in zip(self.output_dirs, TEAM2_utils.EXPERIMENT_FILEPATH_SUFFIXES)]

                _, _, customer_gt_wall_mic_wav_fpath, \
                customer_gt_closetalk_wav_fpath, _, server_gt_wav_fpath = \
                    [Path(f"{self.data_directory}/{subdir}/{scenario_id}_{suf}")
                     for subdir, suf in zip(TEAM2_utils.GROUND_TRUTH_DIRS, TEAM2_utils.GROUND_TRUTH_SUFFIXES)]

                # call again as filenames are different for ground truth transcripts
                merged_gt_fpath, customer_gt_txt_fpath, _, _, server_gt_txt_fpath, _ = \
                    [Path(f"{self.data_directory}/{subdir}/{gt_transcript_prefixes[i]}_{suf}")
                     for subdir, suf in zip(TEAM2_utils.GROUND_TRUTH_DIRS, TEAM2_utils.GROUND_TRUTH_SUFFIXES)]

                if not customer_pred_wav_fpath.exists():
                    continue
                else:
                    print("Computing customer signal metrics")
                    try:
                        c_pesq, c_stoi = \
                            compute_signal_metrics(customer_gt_closetalk_wav_fpath, customer_pred_wav_fpath)
                    except:
                        print(f"No customer close talk signal for scenario: {scenario_id}")
                        c_pesq, c_stoi = np.nan, np.nan

                    for score, col_name in zip([c_pesq, c_stoi], ["pesq", "stoi"]):
                        self.results_df.at[i, col_name] = score

                    if not customer_gt_txt_fpath.exists():
                        print("No customer ground truth transcript available.")
                        continue
                    else:
                        print("Computing customer WER")
                        c_wer, c_num_tokens = compute_wer(customer_pred_txt_fpath, customer_gt_txt_fpath)
                        self.results_df.at[i, "wer"] = c_wer
                        self.results_df.at[i, "num_tokens_in_gt_transcript"] = c_num_tokens

                    if self.rq == "baseline":

                        if not server_gt_txt_fpath.exists():
                            print("No server ground truth transcript available.")
                            continue
                        else:
                            print("Computing server WER")
                            s_wer, s_num_tokens = compute_wer(server_pred_txt_fpath, server_gt_txt_fpath)
                            self.results_df.at[i, "server_wer"] = s_wer
                            self.results_df.at[i, "server_num_tokens_in_gt_transcript"] = s_num_tokens

                        if not merged_gt_fpath.exists():
                            continue
                        else:
                            print("Computing merged WER")
                            m_wer, m_num_tokens = compute_wer(merged_pred_wer_fpath, merged_gt_fpath)
                            self.results_df.at[i, "merged_wer"] = m_wer
                            self.results_df.at[i, "merged_num_tokens_in_gt_transcript"] = m_num_tokens

            self.results_df.at[i, "already_parsed"] = True
            if self.save_as_csv:
                self.results_df.to_csv(self.save_csv_fpath, index=False, columns=filter_by_columns)

        print(self.results_df.head())
        print(self.results_df.info())
        self.results_df.to_csv(self.save_csv_fpath, index=False, columns=filter_by_columns)


def _synchronise_target_and_estimate(reference_audio, processed_audio, mode="simple_crop"):
    if mode == "simple_crop":
        length_diff = len(reference_audio) - len(processed_audio)
        if length_diff > 0:
            print("Cropping reference audio for alignment")
            reference_audio = reference_audio[:-length_diff]
        elif length_diff < 0:
            print("Cropping processed audio for alignment")
            processed_audio = processed_audio[:length_diff]
    else:
        raise NotImplementedError

    return reference_audio, processed_audio


def compute_signal_metrics(reference_audio_fpath, processed_audio_fpath):
    reference_audio, sr = librosa.load(reference_audio_fpath, sr=None)
    processed_audio, sr = librosa.load(processed_audio_fpath, sr=None)
    reference_audio, processed_audio = _synchronise_target_and_estimate(reference_audio, processed_audio)

    pesq_value = pesq(reference_audio, processed_audio, SAMPLE_RATE)
    stoi_value = stoi(reference_audio, processed_audio, SAMPLE_RATE)

    return pesq_value, stoi_value


def compute_wer(predicted_fpath, ground_truth_fpath):
    # https: // speechbrain.readthedocs.io / en / latest / API / speechbrain.utils.edit_distance.html
    predicted = []
    ground_truth = []
    with open(predicted_fpath) as f_obj:
        for line in f_obj:
            predicted.extend([word.strip().lower() for word in line.strip().split()])

    with open(ground_truth_fpath) as f_obj:
        for line in f_obj:
            ground_truth.extend([word.strip().lower() for word in line.strip().split()])

    stats = collections.Counter()
    stats = accumulatable_wer_stats([ground_truth], [predicted], stats)
    return stats["WER"], stats["num_ref_tokens"]


def main():
    for rq in ["adding_enhancer", "adding_separator", "enhancer_first", "separator_first"]:
        print("Computing results for rq:", rq)
        results = ExperimentResults(rq, "kroto_data/final_data_catalogue.csv",
                                    set_split="Training", data_directory="kroto_data", save_as_csv=True)

        results.compute_metrics()


if __name__ == "__main__":
    main()
