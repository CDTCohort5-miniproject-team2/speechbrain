import operator
from stat import ST_CTIME
import os
import time
import json
import sys
from faster_whisper import WhisperModel # Change to SpeechBrain

audio_path = "ENTER AUDIO PATH ON HPC"
transcript_path = "ENTER TRANSCRIPT PATH ON HPC"

def mostRecentFile(path):
    all_files = os.listdir(path)
    file_times = dict()
    for file in all_files:
        file = os.path.join(path, file)
        file_times[file] = time.time() - os.stat(file).st_ctime
    return sorted(file_times.items(), key=operator.itemgetter(1))[0][0]

if __name__ == "__main__":
    if not os.path.exists(audio_path):
        os.mkdir(audio_path)
    if not os.path.exists(transcript_path):
        os.mkdir(transcript_path)

    old_file = ""
    new_file = "PLACEHOLDER.wav"
    old_files = all_files = os.listdir(audio_path)

    print("STARTING TO LISTEN FOR NEW AUDIO FILES!")
    while True:
        old_file = new_file
        new_file = mostRecentFile(audio_path)

        if new_file.split("/")[-1] not in old_files and new_file != old_file:
            print(f"New audio file detected: {new_file}")
            
            # Perform ASR on the new audio file
            model = WhisperModel("base.en")
            segments, info = model.transcribe(new_file)
            transcript = ""
            for segment in segments:
                transcript += "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)

            # Write the transcript to a JSON file
            transcript_filename = f"{new_file.split('/')[-1].split('.')[0]}_transcript.json"
            transcript_filepath = os.path.join(transcript_path, transcript_filename)
            with open(transcript_filepath, "w") as json_file:
                json.dump({"transcript": transcript}, json_file)

            print(f"Transcript saved: {transcript_filepath}")

        else:
            print("No new audio files!")

        time.sleep(2)