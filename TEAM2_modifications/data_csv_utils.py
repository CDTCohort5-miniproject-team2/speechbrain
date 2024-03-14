import pandas as pd
from pathlib import Path

def make_temporary_audio_files_csv(data_dirpath, write_csv_to=""):
    new_df = pd.DataFrame(columns=["Recording File reference", "Set"])
    new_df["Set"] = "Training"

    audio_folder_dirpath = Path(f"{data_dirpath}/Audio")
    if not audio_folder_dirpath.exists():
        raise FileNotFoundError("Directory does not exist")

    for i, audio_path in enumerate(audio_folder_dirpath.glob("*.wav")):
        audio_name = str(audio_path.name)
        new_df.at[i, "Recording File reference"] = audio_name
        new_df.at[i, "Set"] = "Training"

    if write_csv_to == "":
        write_csv_to = f"{data_dirpath}/temporary_data_catalogue.csv"

    new_df.to_csv(write_csv_to)

def main():
    file_storage_dirpath = "kroto_data"
    make_temporary_audio_files_csv(file_storage_dirpath,
                                   write_csv_to="kroto_data/temporary_data_catalogue.csv")


if __name__ == "__main__":
    main()
