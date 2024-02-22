import audio_pipeline_paul_tests
import transcript_pipeline
import prepare_kroto_data
import librosa
from speechbrain.utils.edit_distance import accumulatable_wer_stats
import collections

customer_speech_pipeline = audio_pipeline_paul_tests.AudioPipeline()

def parse_folder(folder_name="01_02_24"):
    kroto_data = prepare_kroto_data.KrotoData(f"test_kroto_data/{folder_name}")

    for i, scenario in enumerate(kroto_data.scenario_catalogue):
        print(f"Parsing scenario {scenario}")

        gt_transcript_fpaths = {
            "server": f"test_kroto_data/{folder_name}/Text_server_cleaned/{scenario}_server_transcript_cleaned.txt",
            "customer": f"test_kroto_data/{folder_name}/Text_customer_cleaned/{scenario}_customer_transcript_cleaned.txt"
        }

        try:
            with open(gt_transcript_fpaths["server"]) as f_obj:
                server_ground_truth = []
                for line in f_obj:
                    server_ground_truth.extend(line.strip().split())

            with open(gt_transcript_fpaths["customer"]) as f_obj:
                customer_ground_truth = []
                for line in f_obj:
                    customer_ground_truth.extend(line.strip().split())

        except FileNotFoundError:
            print(f"Ground truth transcripts for scenario {scenario} are not found. Moving to next scenario.")
            continue

        wav_paths = {
            "wall_mics": f"test_kroto_data/{folder_name}/Audio_wall_mics/{scenario}_wall_mics.wav",
            "customer_closetalk": f"test_kroto_data/{folder_name}/Audio_customer_closetalk/{scenario}_customer_closetalk.wav",
            "server_closetalk": f"test_kroto_data/{folder_name}/Audio_server_closetalk/{scenario}_server_closetalk.wav",
        }

        wall_mics, sr = librosa.load(wav_paths["wall_mics"], sr=16000, mono=False)
        customer_closetalk, sr = librosa.load(wav_paths["customer_closetalk"], sr=16000, mono=True)
        server_closetalk, sr = librosa.load(wav_paths["server_closetalk"], sr=16000, mono=True)

        wall_mic = wall_mics[0, :]

        customer_transcript_output = customer_speech_pipeline.run_inference(
            wall_mic, server_closetalk, f"test_kroto_data/{folder_name}/{scenario}_predicted_customer_transcript")

        # server_speech_pipeline = audio_pipeline.AudioPipeline()
        # server_transcript_output = server_speech_pipeline.run_inference(
        #     server_closetalk, customer_closetalk, f"test_kroto_data/{folder_name}/{scenario}_predicted_server_transcript.txt")

        customer_transcript_obj = transcript_pipeline.Transcript(
            customer_transcript_output, prefix="C", postprocessing_for_wer=True, scenario_id=scenario)
        # server_transcript_obj = transcript_pipeline.Transcript(
        #     server_transcript_output, prefix="S", postprocessing_for_wer=True, scenario_id=scenario)

        compute_wer(customer_ground_truth, customer_transcript_obj.wer_ready_transcript)

        if i == 5:
            break


def compute_wer(ground_truth, predicted):
    stats = collections.Counter()
    stats = accumulatable_wer_stats([ground_truth], [predicted], stats)
    print("%WER {WER:.2f}, {num_ref_tokens} ref tokens".format(**stats))

    # results on first six scenarios in 01_02_24
    # %WER 38.54, 96 ref tokens,
    # %WER 30.30, 66 ref tokens,
    # %WER 27.85, 79 ref tokens,
    # %WER 18.33, 60 ref tokens,
    # %WER 21.13, 71 ref tokens,
    # %WER 205.49, 91 ref tokens ?? double check, discount for now


def main():
    parse_folder()


if __name__ == "__main__":
    main()
