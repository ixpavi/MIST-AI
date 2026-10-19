"""Train the SRM intent classifier and report accuracy on held-out questions.   python train_intent.py"""
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

import intents  # noqa: E402
import semantic  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


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

    if semantic.load_blocking():
        print("\nTraining embedding intent model (local bge-small) ...")
        ex = semantic.encode_plain(X)
        et = semantic.encode_plain(list(tx))
        emb = LogisticRegression(C=20, max_iter=3000).fit(ex, y)
        pe = emb.predict(et)
        print(f"  embedding model alone: {np.mean(np.array(pe) == np.array(ty)):.1%}")
        ens = 0.5 * pipe.predict_proba(list(tx)) + 0.5 * emb.predict_proba(et)
        pens = pipe.classes_[ens.argmax(axis=1)]
        print(f"  ensemble (TF-IDF + embeddings): {np.mean(np.array(pens) == np.array(ty)):.1%}")
        for q, tt, pp in zip(tx, ty, pens):
            if tt != pp:
                print(f"    ensemble miss: {q!r} expected={tt} got={pp}")
        emb.fit(np.vstack([ex, et]), y + list(ty))
        joblib.dump(emb, intents.EMB_MODEL_FILE)
        print(f"  saved -> {intents.EMB_MODEL_FILE}")
    else:
        print("\n(local embedding model unavailable: only the TF-IDF intent model was trained)")

    pipe.fit(X + list(tx), y + list(ty))  # final model also learns from the test questions
    joblib.dump(pipe, intents.MODEL_FILE)
    print(f"Saved model -> {intents.MODEL_FILE}")


if __name__ == "__main__":
    main()
