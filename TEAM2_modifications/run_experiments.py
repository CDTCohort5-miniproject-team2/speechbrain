import experiments
import process_transcripts


def main():
    experiment = experiments.Experiment(rq="adding_separator",
                                        data_csv_fpath="kroto_data/final_data_catalogue.csv",
                                        set_split="Training",
                                        raw_data_directory="kroto_data",
                                        output_dir_suffix="w_medium_en",
                                        asr_model_name="whisper-medium.en",)

    experiment.run_experiment()


if __name__ == "__main__":
    main()