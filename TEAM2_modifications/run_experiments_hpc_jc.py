import experiments

def main():
    data_csv_full_fpath = "/mnt/parscratch/users/acp23jlc/private/kroto_data/temporary_data_catalogue.csv"
    raw_data_full_fpath = "/mnt/parscratch/users/acp23jlc/private/kroto_data"

    experiment = experiments.Experiment(rq="separator_first",
                                                 data_csv_fpath=data_csv_full_fpath,
                                                 set_split="Training",
                                                 raw_data_directory=raw_data_full_fpath,
                                                 output_dir_suffix="w_medium_en",
                                                 asr_model_name="whisper-medium.en",
                                                 test_only=True)

    experiment.run_experiment()


if __name__ == "__main__":
    main()
