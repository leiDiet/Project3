from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_IN = Path("data/used_cars.csv")
OUT_DIR = Path("data/splits")
TRAIN_DIR = OUT_DIR / "train"
TEST_DIR = OUT_DIR / "test"

def main():
    df = pd.read_csv(DATA_IN)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_DIR / "train.csv", index=False)
    test_df.to_csv(TEST_DIR / "test.csv", index=False)

    print(f"Wrote {TRAIN_DIR/'train.csv'}")
    print(f"Wrote {TEST_DIR/'test.csv'}")

if __name__ == "__main__":
    main()
