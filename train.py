import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping

from preprocess import load_data
from attention import AttentionLayer


MAX_WORDS = 50000
MAX_LEN = 300


df = load_data()

X = df["content"]
y = df["label"]

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)

X_pad = pad_sequences(
    X_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

inputs = Input(shape=(MAX_LEN,))

x = Embedding(
    input_dim=MAX_WORDS,
    output_dim=128
)(inputs)

x = Bidirectional(
    LSTM(
        64,
        return_sequences=True
    )
)(x)

x = AttentionLayer()(x)

x = Dense(64, activation="relu")(x)

x = Dropout(0.3)(x)

outputs = Dense(
    1,
    activation="sigmoid"
)(x)

model = Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=64,
    callbacks=[early_stop]
)

loss, acc = model.evaluate(
    X_test,
    y_test
)

print(f"\nTest Accuracy: {acc:.4f}")

preds = model.predict(X_test)
preds = (preds > 0.5).astype(int)

print(
    classification_report(
        y_test,
        preds
    )
)

model.save(
    "models/lstm_attention.keras"
)

joblib.dump(
    tokenizer,
    "models/tokenizer.pkl"
)

print("\nModel Saved!")