import experiments

def main():
    experiment_conditions = ["baseline",
                             "adding_enhancer",
                             "adding_separtor",
                             "enhancer_first",
                             "separator_first"]

    for experiment_condition in experiment_conditions:
        experiment = experiments.Experiment(rq=experiment_condition,
                                            data_csv_fpath="kroto_data/final_data_catalogue.csv",
                                            set_split="Training",
                                            raw_data_directory="kroto_data",
                                            output_dir_suffix="w_medium_en",
                                            asr_model_name="whisper-medium.en")

        experiment.run_experiment()


if __name__ == "__main__":
    main()
