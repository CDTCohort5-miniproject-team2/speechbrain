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

DESIGNATION = {
    "baseline": "baseline_w_whisper_large",
    "adding_enhancer": "aec_enhancer_asr",
    "adding_separator": "aec_separator_asr",
    "separator_first": "aec_separator_enhancer_asr",
    "enhancer_first": "aec_enhancer_separator_asr"
}

COMPONENTS = {
    "baseline": ("aec", "asr"),
    "adding_enhancer": ("aec", "enhancer", "asr"),
    "adding_separator": ("aec", "separator", "asr"),
    "separator_first": ("aec", "separator", "enhancer", "asr"),
    "enhancer_first": ("aec", "enhancer", "separator", "asr")
}


class Experiment:
    def __init__(self, rq, data_csv_fpath, raw_data_directory="kroto_data"):
        """
        :param rq: the name of the RQ. Should be one of the following: "baseline", "adding_enhancer", "adding_separator",
        "separator_first", or enhancer_first"
        :return:
        """
        self.rq = rq
        self.data_csv_fpath = data_csv_fpath
        self.raw_data_directory = raw_data_directory
        self.side = self._decide_side()
        self.training_dataset = self._load_train_set()
        self.output_dirs = self._make_output_dirs()
        self.customer_audio_pipeline = self._get_pipeline()
        if self.side == "both":
            self.server_audio_pipeline = self._get_pipeline()

    def _decide_side(self):
        if self.rq == "baseline":
            side = "both"
        else:
            side = "customer"
        return side

    def _make_output_dirs(self):
        designation = DESIGNATION[self.rq]
        output_dirs = [Path(f"kroto_data/{designation}/{subdir}") for subdir in OUTPUT_DIRS]
        for dirpath in output_dirs:
            if not dirpath.exists():
                os.mkdir(dirpath)
        return output_dirs

    def _load_train_set(self):
        kroto_data = load_kroto_data.RawKrotoData(self.data_csv_fpath, self.raw_data_directory)
        training_dataset = kroto_data.get_torch_dataset(side=self.side)
        # "both" as in we want both customer and server side audio/transcripts etc.
        return training_dataset

    def _get_pipeline(self):
        return audio_pipeline.AudioPipeline(COMPONENTS[self.rq])

    def _run_inference(self):
        for i, (scenario_id, num_of_passengers, has_noise, duration_of_recording,
                server_closetalk, customer_closetalk, single_wall_mic_array,
                server_gt_transcript, customer_gt_transcript, merged_gt_transcript) in enumerate(self.training_dataset):
            saving_fpaths = [output_dir / f"{scenario_id}_{fpath_suffix}"
                             for output_dir, fpath_suffix
                             in zip(self.output_dirs, FILEPATH_SUFFIXES)]
            # customer's and server's processed arrays need to be saved for comparison against ground truth later
            c_processed_array, c_transcript = self.customer_audio_pipeline.run_inference_beta(single_wall_mic_array,
                                                                                              server_closetalk)
            scipy.io.wavfile.write(saving_fpaths[3], 16000, c_processed_array.unsqueeze(-1))
            customer_transcript_obj = process_transcripts.WhisperGeneratedTranscript(c_transcript)
            customer_transcript_obj.save_normalised_transcript_for_wer(saving_fpaths[2])

            if self.server_audio_pipeline():
                s_processed_array, s_transcript = self.server_audio_pipeline.run_inference_beta(server_closetalk,
                                                                                                customer_closetalk)
                scipy.io.wavfile.write(saving_fpaths[5], 16000, s_processed_array.unsqueeze(-1))
                server_transcript_obj = process_transcripts.WhisperGeneratedTranscript(s_transcript)
                server_transcript_obj.save_normalised_transcript_for_wer(saving_fpaths[4])
                server_transcript_obj.merge_transcripts_chronologically(customer_gt_transcript,
                                                                        save_fpath_for_nlp=saving_fpaths[0],
                                                                        save_fpath_for_wer=saving_fpaths[1])
