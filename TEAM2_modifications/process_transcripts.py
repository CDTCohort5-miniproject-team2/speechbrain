"""
This module is responsible for parsing .jsonl ground truth transcripts,
cleaning and saving them down in .txt format ready to be used for computing WER.
"""
import os
from spellchecker import SpellChecker
from num2words import num2words
from pathlib import Path
import pandas as pd
import re
from datetime import datetime

def parse_jsonl_folder(jsonl_dirpath="kroto_data/jsonl_transcripts",
                       save_folders=("kroto_data/raw_transcripts",
                                     "kroto_data/merged_gt_transcripts",
                                     "kroto_data/server_gt_transcripts",
                                     "kroto_data/customer_gt_transcripts"),
                       log_file="kroto_data/jsonl_transcripts/parsed_jsonl.txt"):
    jsonl_dir = Path(jsonl_dirpath)
    log_fpath = Path(log_file)

    for save_dir in save_folders:
        if not Path(save_dir).exists():
            os.mkdir(save_dir)

    if not Path(log_file).exists():
        with open(log_file, "w") as f_obj:
            f_obj.write("")

    with open(log_fpath) as f_obj:
        already_parsed_jsonl = [line.strip() for line in f_obj.readlines()]

    for jsonl_path in jsonl_dir.glob("*.jsonl"):
        if jsonl_path.stem in already_parsed_jsonl:
            print(jsonl_path.stem, "is already parsed.")
            continue

        parse_jsonl(jsonl_path, save_to_folders=save_folders)

        with open(log_fpath, "a") as f_obj:
            f_obj.write(jsonl_path.stem)
            f_obj.write("\n")


def parse_jsonl(jsonl_fpath, save_to_folders=(), check_spelling=False):
    raw_transcript_dir, merged_gt_transcript_dir, server_gt_transcript_dir, customer_gt_transcript_dir = \
        [Path(folder_str) for folder_str in save_to_folders]

    jsonl_df = pd.read_json(path_or_buf=jsonl_fpath, lines=True)
    for i, (_text, _meta, _path, _input_hash, _task_hash, _is_binary, _field_rows, _field_label, _field_id,
            _field_autofocus, _transcript, _orig_transcript, _view_id, _audio_spans, _answer, _timestamp,
            _annotator_id, _session_id) in jsonl_df.iterrows():
        print(f"Parsing {_text}")
        server_lines_idx, customer_lines_idx = [], []

        transcript_by_line = [line.strip() for line in re.split(r"([CS]: )", _transcript) if line.strip()]
        transcript_by_line = [" ".join([line, transcript_by_line[i*2+1]]) for i, line in enumerate(transcript_by_line[0:-1:2])]

        raw_transcript_fpath = raw_transcript_dir/f"{_text}_raw_transcript.txt"
        with open(raw_transcript_fpath, "w") as f_obj:
            f_obj.write("\n".join(transcript_by_line))

        for i, line in enumerate(transcript_by_line):
            if "S:" in line:
                server_lines_idx.append(i)
            elif "C: " in line:
                customer_lines_idx.append(i)

        transcript_by_line = [re.sub(r"[CS]: ", "", line) for line in transcript_by_line]
        transcript_by_line = normalise_for_WER(transcript_by_line)
        if check_spelling:
            transcript_by_line = spell_check(transcript_by_line)

        server_lines, customer_lines = [], []
        for i, line in enumerate(transcript_by_line):
            if i in server_lines_idx:
                server_lines.append(line)
            elif i in customer_lines_idx:
                customer_lines.append(line)

        merged_transcript_fpath = merged_gt_transcript_dir/f"{_text}_gt_merged_transcript.txt"
        server_transcript_fpath = server_gt_transcript_dir/f"{_text}_gt_server_transcript.txt"
        customer_transcript_fpath = customer_gt_transcript_dir/f"{_text}_gt_customer_transcript.txt"

        with open(merged_transcript_fpath, "w") as f_obj:
            f_obj.write("\n".join(transcript_by_line))
        with open(server_transcript_fpath, "w") as f_obj:
            f_obj.write("\n".join(server_lines))
        with open(customer_transcript_fpath, "w") as f_obj:
            f_obj.write("\n".join(customer_lines))


def spell_check(list_of_lines):
    checker = SpellChecker()
    corrected_lines = []

    words_no_need_for_correction = ["whopper", "shwippy", "shwippes", "mambo", "shefburger",
                                    "smarties", "oreo", "fanta", "coca", "cola", "ll", "s", "aren", "isn", "t",
                                    "won", "sec", "didn", "doesn", "sec", "mo", "ve", "BBQ", "flavour", "flavours", "d"]

    # PUT IN COMMON MISTAKES
    common_mistakes = {"shwippie": "shwippy", "erm": "um", "uhh": "uh", "shwippys": "shwippies"}

    for line in list_of_lines:
        corrected_line = line
        corrections_mapping = {}
        words_in_line = re.split(r"\W", line)

        for word_i, word in enumerate(words_in_line):
            if not word:
                continue
            word = word.lower()
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
            corrected_line = re.sub(r"(\W)" + k + r"(\W)", r"\1" + v + r"\2", corrected_line)

        for k, v in corrections_mapping.items():
            corrected_line = re.sub(r"(\W)" + k + r"(\W)", r"\1" + v + r"\2", corrected_line)

        corrected_lines.append(corrected_line)

    return corrected_lines

def normalise_for_WER(list_of_lines):
    half_words = re.compile(r"\w+-\s]")
    punctuations = re.compile(r"[,.!?;\-]+(\s|$)")
    with open("verbal_fillers.txt") as f_obj:
        filler_words_list = "|".join([line.strip() for line in f_obj])
    filler_words = re.compile(r"(^|\W)(" + filler_words_list + r")($|\W)")

    normalised_lines = []
    for line in list_of_lines:
        line = re.sub(half_words, " ", line.strip().lower())
        line = re.sub(punctuations, " ", line)
        line = re.sub(r"[$£,]", "", line)  # rectify currencies and numbers with commas
        line = re.sub(filler_words, " ", line)
        new_line = []
        for _word in line.split():
            if any([char.isdigit() for char in _word]):
                num_list = _word.split(".")
                for num in num_list:
                    try:
                        numword_str = num2words(int(num))
                    except ValueError:
                        pass
                        print(f"Value of {num} unknown.")
                    else:
                        numword_str = numword_str.replace("-", " ").replace(",", "")
                        new_line.append(numword_str)

            else:
                new_line.append(_word)
        normalised_lines.append(" ".join(new_line))

    return normalised_lines


class WhisperGeneratedTranscript:
    def __init__(self, transcript_str,
                 prefix="S"):
        """
        A class for parsing transcripts generated by stable-whisper
        (a) to compute WER against ground truth transcripts
        (b) to
        :param transcript_str:
        :param prefix:

        """

        lines = [line for line in transcript_str.split("\n") if line]
        self.timestamped_lines = []
        self.prefix = prefix

        for line in lines:
            start_time, end_time, segment = line.split("\t")
            self.timestamped_lines.append((prefix, int(start_time), int(end_time), segment))

        self.normalised_lines = None
        self.merged_transcript_lines = None
        self.merged_transcript_lines_normalised = None

    def save_normalised_transcript_for_wer(self, target_fpath):
        self.normalised_lines = normalise_for_WER([line[3].strip() for line in self.timestamped_lines])
        with open(target_fpath, "w") as f_obj:
            f_obj.write("\n".join(self.normalised_lines))

    def merge_transcripts_chronologically(self, target_transcript_obj, save_fpath_for_nlp="", save_fpath_for_wer=""):
        master_transcript = self.timestamped_lines + target_transcript_obj.timestamped_lines
        master_transcript = sorted(master_transcript, key=lambda x: x[2])

        self.merged_transcript_lines = [f"{prefix}: {line.strip()}" for prefix, start, end, line in master_transcript]
        self.merged_transcript_lines_normalised = [re.sub(r"^[CS]: ", "", line) for line in self.merged_transcript_lines]

        if save_fpath_for_nlp:
            with open(save_fpath_for_nlp, "w") as f_obj:
                f_obj.write("\n".join(self.merged_transcript_lines))
        if save_fpath_for_wer:
            with open(save_fpath_for_wer, "w") as f_obj:
                f_obj.write("\n".join(normalise_for_WER(self.merged_transcript_lines_normalised)))

def main():
    # run main to parse jsonl files into transcripts
    parse_jsonl_folder()


if __name__ == "__main__":
    main()
