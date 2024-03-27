"""
This module was introduced to remedy a technical error in audio_pipeline.py and re-run our RQ1 experiment
with respect to 3 out of 5 conditions:
(a) AEC -> ENHANCER -> ASR
(b) AEC -> ENHANCER -> SEPARATOR -> ASR
(c) AEC -> SEPARATOR -> ENHANCER -> ASR

Since then, we have updated audio_pipeline.py so that our ordinary approach (i.e. using run_experiments.py)
would now produce the equivalent experiment outputs as running this module.

The only technical difference is that this module saves down the partially processed audio output
after each component in the pipeline (for efficiency), whereas the inference function in audio_pipeline.py
only returns the final audio after it has been processed by every component in the specified pipeline.

For completeness, we have retained this module in our codebase and provided a step-by-step walkthrough.
"""

import audio_pipeline
from pathlib import Path
import load_kroto_data
import scipy
from scipy import io
import scipy.io.wavfile
import librosa
import process_transcripts

SAMPLING_RATE = 16000

after_aec_dir = Path("partial_outputs/after_aec")
after_aec_and_separating_dir = Path("partial_outputs/after_aec_and_separating")

aec_component = audio_pipeline.AudioPipeline(components=("aec",),)
separator_component = audio_pipeline.AudioPipeline(components=("separator",),)


kroto_data = load_kroto_data.RawKrotoData("kroto_data/final_data_catalogue.csv", "kroto_data")
kroto_torch_dataset = kroto_data.get_torch_dataset(dataset_split="Training")


# PATCHING STEP 1 - RUN AEC ON SOURCE AUDIO (WALL MICROPHONE) AND SAVE AS WAV FILES
def step1():
    #DONE
    for i, (scenario_id, server_closetalk, customer_closetalk, single_wall_mic_array) in enumerate(kroto_torch_dataset):
        print(f"Processing file no. {i}")
        save_fpath = f"{after_aec_dir}/{scenario_id}_post_aec.wav"
        array_after_aec, _, _ = aec_component.run_inference(single_wall_mic_array, server_closetalk)
        scipy.io.wavfile.write(save_fpath, SAMPLING_RATE, array_after_aec)


# PATCHING STEP 2 - RUN SEPARATOR ON AUDIO FROM STEP 1 AND SAVE AS NEW WAV FILES
def step2():
    for i, wavfile in enumerate(after_aec_dir.glob("*.wav")):
        print(f"Processing file no. {i}")
        save_after_aec_separating_fpath = f"{after_aec_and_separating_dir}/{wavfile.stem}_and_separating.wav"
        wav_array, _ = librosa.load(wavfile, sr=SAMPLING_RATE)

        after_aec_and_sep_array, _, _ = separator_component.run_inference(wav_array, None)
        scipy.io.wavfile.write(save_after_aec_separating_fpath, SAMPLING_RATE, after_aec_and_sep_array)


# PATCHING STEP 3 - RUN ENHANCER ON AUDIO FROM STEPS 1 AND 2
def step3():
    # This was done on command line using
    #
    # "python DTLN-master/run_evaluation.py -i
    #   TEAM2_modifications/partial_outputs/after_aec_and_separating -o
    #   TEAM2_modifications/partial_outputs/after_aec_and_separating_and_enhanced
    #   -m DTLN-master/pretrained_model/model.h5"
    # and
    # "python DTLN-master/run_evaluation.py -i
    #   TEAM2_modifications/partial_outputs/after_aec -o
    #   TEAM2_modifications/partial_outputs/after_aec_and_enhanced
    #   -m DTLN-master/pretrained_model/model.h5"
    # respectively.
    # AFTER STEPS 1-3, WE HAVE (AEC->ENHANCER)-PROCESSED AND (AEC->SEPARATOR->ENHANCER)-PROCESSED AUDIO.
    pass


# PATCHING STEP 4 - TO GET (AEC->ENHANCER->SEPARATOR)-PROCESSED AUDIO,
# WE RUN THE SEPARATOR ON (AEC->ENHANCER)-PROCESSED AUDIO FROM STEP 3 AND SAVE AS NEW FILES
after_aec_and_enhanced_dir = Path("partial_outputs/after_aec_and_enhanced")
after_aec_and_enhanced_and_separating_dir = Path("partial_outputs/after_aec_and_enhanced_and_separating")
def step4():
    for i, wavfile in enumerate(after_aec_and_enhanced_dir.glob("*.wav")):
        print(f"Processing file no. {i}")
        save_after_aec_enh_separating_fpath = f"{after_aec_and_enhanced_and_separating_dir}/{wavfile.stem}_and_separating.wav"
        wav_array, _ = librosa.load(wavfile, sr=SAMPLING_RATE)

        after_aec_and_enh_and_sep_array, _, _ = separator_component.run_inference(wav_array, None)
        scipy.io.wavfile.write(save_after_aec_enh_separating_fpath, SAMPLING_RATE, after_aec_and_enh_and_sep_array)


# PATCHING STEP 5 - STEP 4 CONCLUDES THE AUDIO PROCESSING ELEMENT OF THE EXPERIMENT.
def step5():
    # WE RENAME AND MOVE ALL PROCESSED AUDIO ARRAYS FROM THE "PARTIAL_OUTPUTS" SUBDIRECTORIES
    # TO THE RELEVANT EXPERIMENT OUTPUT SUBDIRECTORIES UNDER "KROTO_DATA"
    # (E.G. "KROTO_DATA/AEC_ENHANCER_ASR_W_MEDIUM_EN/CUSTOMER_PROCESSED_ARRAY/{SCENARIO ID}_CUSTOMER_PROCESSED_ARRAY").
    # THIS IS TO ENSURE CONSISTENCY WITH OUR PREVIOUS EXPERIMENTS AND SO THAT WE CAN USE
    # COMPUTE_METRICS.PY DOWNSTREAM
    pass

# PATCHING STEP 6 - RUN ASR TO GET PREDICTED TRANSCRIPTS W.R.T.
# (A) (AEC->ENHANCER)-PROCESSED AUDIO;
# (B) (AEC->ENHANCER->SEPARATOR)-PROCESSED AUDIO; AND
# (C) (AEC->SEPARATOR->ENHANCER)-PROCESSED AUDIO
# SAVE THESE TRANSCRIPTS UNDER THE RELEVANT EXPERIMENT OUTPUT SUBDIRECTORIES

asr_component = audio_pipeline.AudioPipeline(components=("asr",), asr_model_name="whisper-medium.en")

experiment_output_parent_dirs = [Path("kroto_data/aec_enhancer_asr_w_medium_en"),
                                 Path("kroto_data/aec_enhancer_separator_asr_w_medium_en"),
                                 Path("kroto_data/aec_separator_enhancer_asr_w_medium_en")]

def step6():
    for experiment_output_parent_dir in experiment_output_parent_dirs:
        print(f"Parsing {experiment_output_parent_dir.stem}")
        processed_array_dir = experiment_output_parent_dir/"customer_processed_array"
        transcript_dir = experiment_output_parent_dir/"customer_pred_transcript"

        for i, wav_fpath in enumerate(processed_array_dir.glob("*.wav")):
            print(f"Processing audio no. {i}")
            transcript_fname = wav_fpath.stem.replace("_customer_processed_array", "_customer_pred_transcript.txt")
            pred_transcript_fpath = transcript_dir/transcript_fname
            wav_array, _ = librosa.load(wav_fpath, sr=SAMPLING_RATE)
            _, transcript_str, _ = asr_component.run_inference(wav_array, None)

            customer_transcript_obj = process_transcripts.WhisperGeneratedTranscript(transcript_str, prefix="C")
            customer_transcript_obj.save_normalised_transcript_for_wer(pred_transcript_fpath)

# THIS CONCLUDES THE REMEDIAL WORK FOR OUR EXPERIMENT. WE NOW RE-RUN COMPUTE_METRICS.PY ON THE PREDICTIONS.


if __name__ == "__main__":
    # step1() - DONE
    # step2() - DONE
    # step3() - DONE
    # step4() - DONE
    # step5() - DONE
    # step6() - DONE
    pass
