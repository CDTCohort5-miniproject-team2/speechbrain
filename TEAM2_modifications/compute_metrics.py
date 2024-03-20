import collections
import numpy as np
from scipy import signal
import pysepm
from scipy.io import wavfile
import pysepm
from pathlib import Path
from speechbrain.utils.edit_distance import accumulatable_wer_stats
import TEAM2_utils
import pandas as pd
import librosa
import warnings
from numba import NumbaDeprecationWarning
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
from tqdm import tqdm

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
        self.designation = TEAM2_utils.EXPERIMENT_DESIGNATION[self.rq]
        self.parent_output_dir = Path(f"kroto_data/{self.designation}_w_medium_en")
        self.save_csv_fpath = self.parent_output_dir/f"{self.designation}_results.csv"
        self.output_dirs = [Path(f"kroto_data/{self.designation}/{subdir}") for subdir in TEAM2_utils.EXPERIMENT_OUTPUT_DIRS]
        self.merged_pred_nlp_dir, self.merged_pred_wer_dir, self.customer_pred_txt_dir, \
            self.customer_pred_wav_dir, self.server_pred_txt_dir, self.server_pred_wav_dir = self.output_dirs
        self._check_output_dirs()

        self.original_df = pd.read_csv(data_csv_fpath)

    def _check_output_dirs(self):
        if not self.parent_output_dir.exists():
            raise FileNotFoundError("Experiment results not found. Please make sure correct directory is specified."
                                    "The directory you specified was", self.parent_output_dir)
        for output_dir in self.output_dirs:
            if not output_dir.exists():
                print("Warning: result subdirectory not found:", output_dir)
                # this is a warning only, as it might be possible that some are not used
    def compute_metrics(self):
        self.original_df["scenario_id"] = [fname.replace(".wav", "") for fname in self.original_df["Recording File reference"]]
        new_df_col_names = ["pesq", "stoi", "composite_score_sig", "composite_score_bak", "composite_score_ovl",
                            "pesq_v_wall_mic", "stoi_v_wall_mic",
                            "composite_score_v_wall_mic_sig", "composite_score_v_wall_mic_bak",
                            "composite_score_v_wall_mic_ovl",
                            "wer", "num_tokens_in_gt_transcript"]

        if self.rq == "baseline":
            new_df_col_names += ["server_pesq", "server_stoi", "server_composite_score_sig", "server_composite_score_bak",
                                 "server_composite_score_ovl", "server_wer",
                                 "server_num_tokens_in_gt_transcript", "merged_wer", "merged_num_tokens_in_gt_transcript"]

        for col_name in new_df_col_names:
            self.original_df[col_name] = np.nan

        filter_by_columns = ["scenario_id", "Set"] + new_df_col_names
        self.results_df = self.original_df[filter_by_columns].copy()

        print(self.results_df.head())
        print(self.results_df.info())
        for i, scenario_id in enumerate(self.results_df["scenario_id"]):
            if self.results_df["Set"][i] != self.set_split:
                continue
            else:
                merged_pred_nlp_fpath, merged_pred_wer_fpath, customer_pred_txt_fpath, \
                customer_pred_wav_fpath, server_pred_txt_fpath, server_pred_wav_fpath = \
                    [subdir/f"{scenario_id}_{suf}"
                     for subdir, suf in zip(self.output_dirs, TEAM2_utils.EXPERIMENT_FILEPATH_SUFFIXES)]

                merged_gt_fpath, customer_gt_txt_fpath, customer_gt_wall_mic_wav_fpath, \
                customer_gt_closetalk_wav_fpath, server_gt_txt_fpath, server_gt_wav_fpath = \
                    [Path(f"{self.data_directory}/{subdir}/{scenario_id}_{suf}")
                     for subdir, suf in zip(TEAM2_utils.GROUND_TRUTH_DIRS, TEAM2_utils.GROUND_TRUTH_SUFFIXES)]

                if not customer_pred_wav_fpath.exists():
                    continue
                else:
                    c_pesq, c_stoi, c_compsig, c_compbak, c_compovl = compute_signal_metrics(customer_gt_closetalk_wav_fpath, customer_pred_wav_fpath)
                    c_pesq_v_wall_mic, c_stoi_v_wall_mic, c_compsig_v_wall_mic, c_compbak_v_wall_mic, c_compovl_v_wall_mic = \
                        compute_signal_metrics(customer_gt_wall_mic_wav_fpath, customer_pred_wav_fpath)

                    for score, col_name in zip([c_pesq, c_stoi, c_compsig, c_compbak, c_compovl,
                                                c_pesq_v_wall_mic, c_stoi_v_wall_mic, c_compsig_v_wall_mic,
                                                c_compbak_v_wall_mic, c_compovl_v_wall_mic],
                                               ["pesq", "stoi", "composite_score_sig", "composite_score_bak",
                                                "composite_score_ovl", "pesq_v_wall_mic", "stoi_v_wall_mic",
                                                "composite_score_v_wall_mic_sig", "composite_score_v_wall_mic_bak",
                                                "composite_score_v_wall_mic_ovl"]):
                        self.results_df.at[i, col_name] = score

                    if not customer_gt_txt_fpath.exists():
                        continue
                    else:
                        c_wer, c_num_tokens = compute_wer(customer_pred_txt_fpath, customer_gt_txt_fpath)
                        self.results_df.at[i, "wer"] = c_wer
                        self.results_df.at[i, "num_tokens_in_gt_transcript"] = c_num_tokens

                    if self.rq == "baseline":
                        s_pesq, s_stoi, s_compsig, s_compbak, s_compovl = compute_signal_metrics(server_gt_wav_fpath,
                                                                                                 server_pred_wav_fpath)

                        for score, col_name in zip([s_pesq, s_stoi, s_compsig, s_compbak, s_compovl],
                                                   ["server_pesq", "server_stoi", "server_composite_score_sig",
                                                    "server_composite_score_bak", "server_composite_score_ovl"]):
                            self.results_df.at[i, col_name] = score

                        if not server_gt_txt_fpath.exists():
                            continue
                        else:
                            s_wer, s_num_tokens = compute_wer(server_pred_txt_fpath, server_gt_txt_fpath)
                            self.results_df.at[i, "server_wer"] = s_wer
                            self.results_df.at[i, "server_num_tokens_in_gt_transcript"] = s_num_tokens

                        if not merged_gt_fpath.exists():
                            continue
                        else:
                            m_wer, m_num_tokens = compute_wer(merged_pred_wer_fpath, merged_gt_fpath)
                            self.results_df.at[i, "merged_wer"] = m_wer
                            self.results_df.at[i, "merged_num_tokens_in_gt_transcript"] = m_num_tokens

        print(self.results_df.head())
        print(self.results_df.info())
        if self.save_as_csv:
            self.results_df.to_csv(self.save_csv_fpath, index=False, columns=filter_by_columns)

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

def compute_signal_metrics(reference_audio_fpath, processed_audio_fpath):
    reference_audio, sr = librosa.load(reference_audio_fpath, sr=None)
    processed_audio, sr = librosa.load(processed_audio_fpath, sr=None)
    reference_audio, processed_audio = _synchronise_target_and_estimate(reference_audio, processed_audio)
    _, pesq_value = pysepm.pesq(reference_audio, processed_audio, SAMPLE_RATE)
    stoi_value = pysepm.stoi(reference_audio, processed_audio, SAMPLE_RATE)
    c_score_signal, c_score_background, c_score_overall = pysepm.composite(reference_audio, processed_audio, SAMPLE_RATE)
    return pesq_value, stoi_value, c_score_signal, c_score_background, c_score_overall

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
    baseline_demo_results = ExperimentResults("baseline", "kroto_data/temporary_data_catalogue.csv",
                                              set_split="Training", data_directory="kroto_data", save_as_csv=True)

    baseline_demo_results.compute_metrics()


if __name__ == "__main__":
    main()
