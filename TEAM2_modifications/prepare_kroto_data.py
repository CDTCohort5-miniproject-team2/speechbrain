import librosa
import pandas as pd
from pathlib import Path
import scipy
from scipy import io  # added during debugging
import scipy.io.wavfile  #  added during debugging
import os
import numpy as np
import sounddevice as sd
import tqdm
from tqdm import tqdm
import re
from spellchecker import SpellChecker
from num2words import num2words

# this scheme was used in our earlier recording sessions (e.g. 18_12_2023), retained for ref
OLD_CHANNEL_MAPPING = {
    "dashboard_mics": [0, 1, 2, 3],
    # Channels 4, 5 and 6 are not used
    "wall_mics": [7, 8, 9, 10, 11],
    "passengers_closetalk": [12, 13],
    "customer_closetalk": [14],
    "server_closetalk": [15],
}

# the new scheme that we are using going forward, with unused channels removed
CHANNEL_MAPPING = {
    "dashboard_mics": [0, 1, 2, 3],
    "wall_mics": [4, 5, 6, 7, 8],
    "passengers_closetalk": [10, 11],
    "customer_closetalk": [9],
    "server_closetalk": [12],
}

def get_channels_by_name(multichannel_audio_array, target_channel_name):
    """

    :param multichannel_audio_array:
    :param target_channel_name:
    :return:
    """
    assert multichannel_audio_array.shape[0] in [13, 16], "The function_get_channels_by_name can only operate on the full 13- or 16-channel array."
    try:
        target_channels = np.array(CHANNEL_MAPPING[target_channel_name])
    except KeyError:
        print("Target channel name not valid. Returning array with all channels.")
        return multichannel_audio_array
    else:
        return multichannel_audio_array[target_channels, :]


def play_audio_array(audio_array, sr=16000):
    """
    Function for playing a (mono or multichannel) audio array for sense check
    :param audio_array: numpy array (n_samples,) or (n_channels, n_samples,)
    :param sr: int - sample rate to use
    :return:
    """
    if len(audio_array.shape) == 2:
        # down-mixing multichannel audio for playback on laptop speakers
        audio_array = np.mean(audio_array, axis=0)
    sd.play(audio_array, samplerate=sr)
    sd.wait()

def get_audio_array_timeslice(audio_array, start_time, end_time, sr=16000):
    """
    Returns a timeslice of the audio array
    :param audio_array: 1, 2 or 3D - as long as n_samples is the last dimension
    :param start_time: in seconds
    :param end_time: in seconds
    :param sr: int - sample rate to use
    :return:
    """
    if end_time == 0:
        # return the whole array unsliced
        return audio_array

    start_sample, end_sample = int(start_time*sr), int(end_time*sr)

    if end_sample >= audio_array.shape[-1]:
        end_sample = audio_array.shape[-1] - 1
        print(f"End time is greater than audio length, returning a shorter timeslice instead "
              f"from {start_time} seconds to {end_sample/sr} seconds.")

    # TEAM2 fyi: "..." in slicing numpy arrays allows it to accept arrays of an arbitrary number of dimensions
    return audio_array[..., start_sample:end_sample]

def cut_long_audio_files(dirpath, n_seconds=20.0):
    # TODO
    pass


def spell_check(list_of_lines):
    # TODO: not completed
    checker = SpellChecker()
    corrected_lines = []

    # TODO: TEAM2 add to this word list here as you go, include british spellings
    words_no_need_for_correction = ["whopper", "shwippy", "shwippes", "mambo", "shefburger",
                                    "smarties", "oreo", "fanta", "coca", "cola", "ll", "s", "aren", "isn", "t",
                                    "won", "sec", "didn", "doesn", "sec", "mo", "ve", "BBQ", "flavour", "flavours"]

    # PUT IN COMMON MISTAKES
    common_mistakes = {"shwippie": "shwippy", "erm": "um", "uhh": "uh"}

    for line in list_of_lines:
        corrected_line = line
        corrections_mapping = {}
        words_in_line = re.split(r"\W", line)

        for word_i, word in enumerate(words_in_line):
            if not word:
                continue
            potentially_misspelled = checker.unknown([word])
            if (len(potentially_misspelled) > 0) and \
                    (word.lower() not in list(common_mistakes.keys())) and \
                    (word.lower() not in words_no_need_for_correction):

                print(f"Spell check flagged typo in line: {line}")
                user_correct = input(f"Correct '{word}' -> '{checker.correction(word)}'? \n"
                                     f"(press ENTER to accept, \n"
                                     f"or type in NO (uppercase) to leave unchanged, \n"
                                     f"or type in manual correction (lowercase) then press ENTER): \n")
                if user_correct.lower() == "no":
                    continue
                elif user_correct:
                    corrections_mapping[word] = user_correct
                else:
                    corrections_mapping[word] = checker.correction(word)

        for k, v in common_mistakes.items():
            corrected_line = re.sub(r"(\W)"+k+r"(\W)", r"\1"+v+r"\2", corrected_line)

        for k, v in corrections_mapping.items():
            corrected_line = re.sub(r"(\W)"+k+r"(\W)", r"\1"+v+r"\2", corrected_line)

        corrected_lines.append(corrected_line)

    return corrected_lines

class KrotoData:
    def __init__(self, dirpath_str, target_sr=16000):
        """
        Parses a data folder containing the sub-folders Audio, Text, Logs
        :param dirpath_str: str - path to the folder/directory
        """
        self.parent_dirpath = Path(dirpath_str)
        if not (self.parent_dirpath.exists() and self.parent_dirpath.is_dir()):
            raise FileNotFoundError("Specified folder does not exist - check that you've provided the correct folder path")

        self.raw_audio_dir, self.raw_transcript_dir, self.raw_log_dir = self.parent_dirpath/"Audio", self.parent_dirpath/"Text", self.parent_dirpath/"Log"

        if not self.raw_audio_dir.exists():
            raise ValueError(f"'Audio' folder not found within the specified directory: {dirpath_str}. \n"
                             f"Please ensure audio files are saved in path {dirpath_str}/Audio/some_audio_file.wav")

        self.scenario_catalogue = [file.stem for file in self.raw_audio_dir.glob("*.wav")]

        self.target_sr = target_sr

        self.server_transcript_dir, self.customer_transcript_dir = self.parent_dirpath/"Text_server", \
                                                                   self.parent_dirpath/"Text_customer"

        self.cleaned_server_transcript_dir, self.cleaned_customer_transcript_dir = self.parent_dirpath/"Text_server_cleaned", \
                                                                                   self.parent_dirpath/"Text_customer_cleaned"

        if (not self.server_transcript_dir.exists()) or (not self.customer_transcript_dir.exists()):
            parse_jsonl = input("Raw text directory loaded. Parse jsonl files and "
                                "separate customer/server transcripts? (Y/N): ")
            if "y" in parse_jsonl.lower():
                print("Parsing jsonl and saving...")
                self.preprocess_transcripts()

                print("Transcripts saved.")

        sub_array_dirpaths = [self.parent_dirpath/f"Audio_{channel_name}" for channel_name in CHANNEL_MAPPING.keys()]

        if any([not dirpath.exists() for dirpath in sub_array_dirpaths]):
            initiate_separate_channels_and_save_audio = input("Raw audio directory loaded. "
                                                              "Separate out channels, downsample and save audio? (Y/N): ")
            if "y" in initiate_separate_channels_and_save_audio.lower():
                print("Separating audio channels for downsampling and saving...")
                self.separate_channels_and_save_audio(downsampling=True)
                print("Audio saved.")

    def preprocess_transcripts(self):

        if not self.raw_transcript_dir.exists():
            raise FileNotFoundError("Directory of raw transcripts (.jsonl format) not found. "
                                    "Make sure raw transcripts are filed under {date_of_recording}/Text/*.jsonl")

        if not self.server_transcript_dir.exists():
            os.mkdir(self.server_transcript_dir)
        if not self.customer_transcript_dir.exists():
            os.mkdir(self.customer_transcript_dir)

        for jsonl_fpath in self.raw_transcript_dir.glob("*.jsonl"):
            jsonl_df = pd.read_json(path_or_buf=jsonl_fpath, lines=True)
            for i, (_text, _meta, _path, _input_hash, _task_hash, _is_binary, _field_rows, _field_label, _field_id,
                 _field_autofocus, _transcript, _orig_transcript, _view_id, _audio_spans, _answer, _timestamp,
                 _annotator_id, _session_id) in jsonl_df.iterrows():
                print(f"PARSING TRANSCRIPT FOR {_text}...")
                server_lines, customer_lines = [], []

                transcript_by_line = [line.strip() for line in re.split(r"([CS]: )", _transcript) if line.strip()]
                for i, line in enumerate(transcript_by_line):
                    if line == "S:":
                        server_lines.append(transcript_by_line[i+1])
                    elif line == "C:":
                        customer_lines.append(transcript_by_line[i+1])

                customer_transcript_fpath = self.customer_transcript_dir/f"{_text}_customer_transcript.txt"
                server_transcript_fpath = self.server_transcript_dir/f"{_text}_server_transcript.txt"

                customer_lines = spell_check(customer_lines)
                server_lines = spell_check(server_lines)

                joined_customer_lines = "\n".join(customer_lines)
                joined_server_lines = "\n".join(server_lines)

                with open(customer_transcript_fpath, "w") as f_obj:
                    f_obj.write(joined_customer_lines)

                with open(server_transcript_fpath, "w") as f_obj:
                    f_obj.write(joined_server_lines)

    def clean_transcripts_for_wer(self):
        # TODO: THIS IS NOT YET DONE. THERE ARE QUITE A FEW ERRORS IN THE TRANSCRIPTS
        #  THAT WE NEED TO DISCUSS AND REMEDY
        print("Cleaning transcripts for WER metric.")
        half_words = re.compile(r"\w+-\s]")
        punctuations = re.compile(r"[,.!?;\-]+(\W|$)")
        with open("verbal_fillers.txt") as f_obj:
            filler_words_list = "|".join([line.strip() for line in f_obj])

        filler_words = re.compile(r"(^|\W)("+filler_words_list+r")($|\W)")

        if not self.cleaned_customer_transcript_dir.exists():
            os.mkdir(self.cleaned_customer_transcript_dir)
        if not self.cleaned_server_transcript_dir.exists():
            os.mkdir(self.cleaned_server_transcript_dir)

        mapping = [(self.server_transcript_dir, self.cleaned_server_transcript_dir),
                   (self.customer_transcript_dir, self.cleaned_customer_transcript_dir)]
        for src_folder, tgt_folder in mapping:
            for txt_file in src_folder.glob("*.txt"):
                new_txt_fname = str(txt_file.stem) + "_cleaned.txt"
                new_txt_fpath = tgt_folder/new_txt_fname
                with open(txt_file) as f_obj, open(new_txt_fpath, "w") as new_f_obj:
                    new_lines = []
                    for line in f_obj:
                        line = re.sub(half_words, " ", line.strip().lower())
                        line = re.sub(punctuations, " ", line)
                        line = re.sub(filler_words, " ", line)
                        new_line = []
                        for _word in line.split():
                            if any([char.isdigit() for char in _word]):
                                if "£" in _word or "$" in _word:
                                    _word = _word.replace("£", "")
                                    _word = _word.replace("$", "")
                                    numbers_spelt_out = [num2words(int(num_word)) for num_word in _word.split(".")]
                                    numbers_spelt_out.insert(1, "pounds")
                                    # TODO: WE NEED TO DECIDE IF THIS SHOULD BE INFERRED
                                else:
                                    numbers_spelt_out = [num2words(int(num_word)) for num_word in _word.split(".")]
                                new_line.extend(numbers_spelt_out)
                            else:
                                new_line.append(_word)
                        new_lines.append(" ".join(new_line))
                    new_f_obj.write("\n".join(new_lines))
        print("Transcripts cleaned.")

    def get_demo_audio_array(self, audio_fname="", audio_idx=0, downsampled=True, timeslice=(0.0, 0.0),
                             channel_name=""):
        if audio_fname:
            fpath = self.raw_audio_dir/audio_fname
            if not fpath.exists():
                print(f"Filepath invalid. Returning sample audio indexed at 0 in the directory instead.")
                fpath = self.raw_audio_dir / self.audio_catalogue[audio_idx].name
        else:
            fpath = self.raw_audio_dir/self.audio_catalogue[audio_idx].name
        print(f"Getting demo audio from {fpath.stem}")
        demo_array, sr = librosa.load(fpath, sr=None, mono=False)

        if downsampled and (sr != self.target_sr):
            demo_array = librosa.resample(demo_array, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        if channel_name:
            demo_array = get_channels_by_name(demo_array, channel_name)
        
        return get_audio_array_timeslice(demo_array, start_time=timeslice[0], end_time=timeslice[1], sr=int(sr))

    def separate_channels_and_save_audio(self, downsampling=True):
        """
        Splits the full 13- or 16-channel audio files found in the "Audio" folder according to the CHANNEL_MAPPING partitions.
        Saves down the audio files (with channels filtered) into new directories named after these partitions.
        :param downsampling: boolean - if True, performs downsamples the audio to self.target_sr before saving
        :return: None
        """
        channel_names = list(CHANNEL_MAPPING.keys())
        sub_array_dirpaths = [self.parent_dirpath/f"Audio_{channel_name}" for channel_name in channel_names]

        for sub_array_dirpath in sub_array_dirpaths:
            if not sub_array_dirpath.exists():
                os.mkdir(sub_array_dirpath)

        for wav_fpath in tqdm(self.raw_audio_dir.glob("*.wav")):
            audio_array, source_sr = librosa.load(wav_fpath, sr=None, mono=False)

            if downsampling and (source_sr != self.target_sr):
                audio_array = librosa.resample(audio_array, orig_sr=source_sr, target_sr=self.target_sr)

            for i, channel_name in enumerate(channel_names):
                sub_array = get_channels_by_name(audio_array, channel_name)
                sub_array_fpath = sub_array_dirpaths[i]/f"{wav_fpath.stem}_{channel_name}.wav"

                # NB scipy.io.wavfile.write expects (num_samples, num_channels)
                sub_array = np.swapaxes(sub_array, 0, 1)
                scipy.io.wavfile.write(sub_array_fpath, self.target_sr if downsampling else source_sr, sub_array)

        print("Audio files saved successfully.")

def main():
    demo_dirpath = "test_kroto_data/01_02_24"
    # TEAM2 instructions: replace this with the path where you saved the recording session's data folder
    # this data folder should contain the sub-folders "Audio", "Logs" and "Text"

    demo_kroto_data = KrotoData(demo_dirpath)
    demo_kroto_data.clean_transcripts_for_wer()

if __name__ == "__main__":
    main()





