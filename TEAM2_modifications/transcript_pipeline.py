import re
from num2words import num2words
import pickle

class Transcript:
    def __init__(self, transcript_str="", prefix="S", postprocessing_for_wer=False, scenario_id=""):

        lines = [line for line in transcript_str.split("\n") if line]
        self.timestamped_lines = []
        self.prefix = prefix
        self.scenario_id = scenario_id

        for line in lines:
            start_time, end_time, segment = line.split("\t")
            self.timestamped_lines.append((prefix, int(start_time), int(end_time), segment))

        self.wer_ready_transcript = None
        if postprocessing_for_wer:
            self._postprocess_for_wer()

    def _postprocess_for_wer(self, save_as_txt=False):
        with open("verbal_fillers.txt") as f_obj:
            filler_words = [item.strip().lower() for item in f_obj.read()]
        cleaned_words = []

        for line in self.timestamped_lines:
            cleaned_line = re.sub("[£$,.!?\-]", " ", line[3].strip())

            for word in cleaned_line.split():
                word = word.lower()
                if any([char.isdigit() for char in word]):
                    number_as_words = re.split(r"\s|-", num2words(int(word)))
                    cleaned_words.extend(number_as_words)

                elif word in filler_words:
                    continue

                else:
                    cleaned_words.append(word)

        self.wer_ready_transcript = cleaned_words


    def merge_transcripts_chronologically(self, target_transcript_obj=None, save_fname=""):
        if not target_transcript_obj:
            return self.timestamped_lines
        else:
            master_transcript = self.timestamped_lines + target_transcript_obj.timestamped_lines
            master_transcript = sorted(master_transcript, key=lambda x: x[2])

        if save_fname:
            with open(f"{save_fname}_merged_transcript.txt", "w") as f_obj:
                master_transcript_lines = [f"{prefix}: {line}" for prefix, start, end, line in master_transcript]
                f_obj.write("\n".join(master_transcript_lines))

        return master_transcript



def main():
    demo_txt_file = "customer_side_demo.txt"
    with open(demo_txt_file) as f_obj:
        demo_txt_str = f_obj.read()
    demo_transcript = Transcript(demo_txt_str, prefix="C")

    demo_txt_file2 = "server_side_demo.txt"
    with open(demo_txt_file2) as f_obj:
        demo_txt_str2 = f_obj.read()
    demo_transcript2 = Transcript(demo_txt_str2, prefix="S")

    merged_transcript = demo_transcript.merge_transcripts_chronologically(demo_transcript2, save_fname="demo")

if __name__ == "__main__":
    main()
