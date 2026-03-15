import os
import sys
import argparse
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

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


def split_and_save_data(
    df: pd.DataFrame,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    target_column: str = "Class"
):
    os.makedirs(output_dir, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column]
    )

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Підготовка даних завершена успішно.")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print(f"Файли збережено у: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("input_file", type=str, help="Шлях до сирого датасету")
    parser.add_argument("output_dir", type=str, help="Папка для збереження prepared data")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--target_column", type=str, default="Class")
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_data(args.input_file)
    df = preprocess_data(df, target_column=args.target_column)

    split_and_save_data(
        df=df,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        target_column=args.target_column
    )


if __name__ == "__main__":
    main()