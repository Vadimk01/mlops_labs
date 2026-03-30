import os
import argparse
import warnings
import pickle

import pandas as pd

warnings.filterwarnings("ignore")


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не знайдено: {file_path}")
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame, target_column: str = "Class") -> pd.DataFrame:
    if target_column not in df.columns:
        raise ValueError(f"У датасеті немає цільової змінної '{target_column}'")

    df = df.dropna()
    return df


def save_processed_data(df: pd.DataFrame, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump(df, f)

    print("Підготовка даних завершена успішно.")
    print(f"Shape after preprocessing: {df.shape}")
    print(f"Файл збережено: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("input_file", type=str, help="Шлях до сирого датасету")
    parser.add_argument("output_file", type=str, help="Шлях до processed pickle file")
    parser.add_argument("--target_column", type=str, default="Class")
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_data(args.input_file)
    df = preprocess_data(df, target_column=args.target_column)
    save_processed_data(df, args.output_file)


if __name__ == "__main__":
    main()