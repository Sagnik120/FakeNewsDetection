import pandas as pd
import re
import string


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_data():
    fake_df = pd.read_csv("data/Fake.csv")
    true_df = pd.read_csv("data/True.csv")

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    df["content"] = df["text"].fillna("")

    df["content"] = df["content"].apply(clean_text)

    return df[["content", "label"]]


if __name__ == "__main__":
    df = load_data()

    print(df.head())
    print("\nShape:", df.shape)
    print("\nClass Distribution:")
    print(df["label"].value_counts())