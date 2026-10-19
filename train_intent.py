"""Train the SRM intent classifier and report accuracy on held-out questions.   python train_intent.py"""
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

import intents  # noqa: E402


def main():
    X, y = intents.augmented_train()
    print(f"{len(X)} training questions, {len(intents.TRAIN)} intents")

    pipe = intents.build_pipeline()
    pipe.fit(X, y)

    tx, ty = zip(*intents.TEST)
    pred = pipe.predict(list(tx))
    acc = float(np.mean(np.array(pred) == np.array(ty)))
    print(f"Held-out test accuracy ({len(tx)} unseen questions): {acc:.1%}\n")
    print(classification_report(ty, pred, zero_division=0))
    for q, t, p in zip(tx, ty, pred):
        if t != p:
            print(f"  miss: {q!r}  expected={t} got={p}")

    pipe.fit(X + list(tx), y + list(ty))  # final model also learns from the test questions
    joblib.dump(pipe, intents.MODEL_FILE)
    print(f"Saved model -> {intents.MODEL_FILE}")


if __name__ == "__main__":
    main()
