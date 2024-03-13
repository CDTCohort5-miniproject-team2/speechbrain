import experiments


def main():
    baseline_experiment = experiments.Experiment(rq="baseline",
                                                 data_csv_fpath="temporary_data_csv.csv",
                                                 set_split="Training",
                                                 raw_data_directory="/mnt/parscratch/users/acp23jlc/private/kroto_data",
                                                 output_dir_suffix="w_base_en",
                                                 asr_model_name="whisper-base.en",
                                                 hpc_test_only=True
                                                 )

    baseline_experiment.run_experiment()


if __name__ == "__main__":
    main()
