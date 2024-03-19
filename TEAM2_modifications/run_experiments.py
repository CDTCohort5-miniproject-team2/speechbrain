import experiments
import process_transcripts

def main():
    baseline_experiment = experiments.Experiment(rq="baseline",
                                                 data_csv_fpath="kroto_data/temporary_data_catalogue.csv",
                                                 set_split="Training",
                                                 raw_data_directory="kroto_data",
                                                 output_dir_suffix="w_medium_en",
                                                 asr_model_name="whisper-medium.en",)

    baseline_experiment.run_experiment()


if __name__ == "__main__":
    main()
