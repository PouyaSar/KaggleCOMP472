import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from k_nn import stratified_split
from neural_network import NeuralNetworkClassifier


def l1_rows(X):
    sums = X.sum(axis=1, keepdims=True)
    return (X / np.where(sums == 0, 1.0, sums)).astype(np.float32)


def sqrt_unit_rows(X):
    """Square root, then unit length (Hellinger transform for histograms)."""
    R = np.sqrt(X)
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    return (R / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def class_similarities(X_train, train_labels, X_query, n_classes, top):
    """Top intersection similarities and best cosine similarity to each class."""
    order = np.argsort(train_labels, kind="stable")
    labels = train_labels[order]
    bounds = np.searchsorted(labels, np.arange(n_classes + 1))
    if np.any(np.diff(bounds) < top):
        raise ValueError(f"Every class needs at least {top} training rows")
    inter_train, inter_query = l1_rows(X_train)[order], l1_rows(X_query)
    cos_train, cos_query = sqrt_unit_rows(X_train)[order], sqrt_unit_rows(X_query)
    inter = np.empty((len(X_query), n_classes, top), dtype=np.float32)
    cosine = np.empty((len(X_query), n_classes), dtype=np.float32)
    for start in range(0, len(X_query), 500):
        cos_block = cos_query[start:start + 500] @ cos_train.T
        for offset, query in enumerate(inter_query[start:start + 500]):
            i = start + offset
            # Histogram intersection: sum of elementwise minima, between 0 and 1.
            similarity = np.minimum(inter_train, query).sum(axis=1)
            for c in range(n_classes):
                part = similarity[bounds[c]:bounds[c + 1]]
                best = part[np.argpartition(part, -top)[-top:]]
                inter[i, c] = -np.sort(-best)
                cosine[i, c] = cos_block[offset, bounds[c]:bounds[c + 1]].max()
    return inter, cosine


def combine(inter, cosine, log_probs, weights, beta, lam):
    """score = weighted top matches per class + beta * cosine + lam * log P."""
    return (inter * np.asarray(weights, dtype=np.float32)).sum(axis=2) \
        + beta * cosine + lam * log_probs


def network(epochs, patience, seed):
    return NeuralNetworkClassifier(hidden_layers=(1024,), activation="relu",
                                   learning_rate=0.001, optimizer="adam",
                                   batch_size=128, epochs=epochs,
                                   weight_decay=0.0, dropout=0.3,
                                   normalization="none", patience=patience,
                                   random_state=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train", type=Path)
    parser.add_argument("--weights", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.25],
                        help="Weights of the 1st, 2nd, ... best intersection match per class")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Weight of the best cosine match per class (default: 0.5)")
    parser.add_argument("--lam", type=float, default=0.01,
                        help="Weight of the networks' log-probabilities (default: 0.01)")
    parser.add_argument("--n-networks", type=int, default=4,
                        help="Networks averaged, each with a different seed (default: 4)")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path, default=Path("hybrid_predictions.csv"))
    args = parser.parse_args()
    if args.n_networks < 1:
        parser.error("--n-networks must be at least 1")
    torch.set_num_threads(1)

    frame = pd.read_csv(args.train)
    features = [column for column in frame if column.startswith("feature_")]
    if not features or "label" not in frame:
        parser.error("Training CSV needs feature_* columns and label")
    X = frame[features].to_numpy(dtype=float)
    if np.any(X < 0):
        parser.error("Histogram intersection needs nonnegative features")
    y = frame["label"].to_numpy()
    classes, encoded = np.unique(y, return_inverse=True)
    top = len(args.weights)
    train, validation = stratified_split(encoded, args.validation_fraction, args.seed)
    print(f"Training rows: {len(train)}, validation rows: {len(validation)}", flush=True)

    start = perf_counter()
    inter, cosine = class_similarities(X[train], encoded[train], X[validation],
                                       len(classes), top)
    print(f"Similarities: {perf_counter() - start:.1f} s", flush=True)

    probabilities, epochs = [], []
    for n in range(args.n_networks):
        start = perf_counter()
        model = network(100, 10, args.seed + n).fit(
            sqrt_unit_rows(X[train]), encoded[train],
            sqrt_unit_rows(X[validation]), encoded[validation])
        probabilities.append(model.predict_proba(sqrt_unit_rows(X[validation])))
        epochs.append(model.best_epoch_)
        print(f"Network {n + 1}: {perf_counter() - start:.1f} s, best epoch {model.best_epoch_}, "
              f"validation accuracy {model.best_validation_accuracy_:.4%}", flush=True)
    # Averaging probabilities over seeds reduces the noise of any single network.
    log_probs = np.log(np.mean(probabilities, axis=0) + 1e-9)

    target = encoded[validation]

    def accuracy(weights, beta, lam):
        return np.mean(combine(inter, cosine, log_probs, weights, beta, lam).argmax(1) == target)

    one_nn = [1.0] + [0.0] * (top - 1)
    print("\nValidation accuracy, adding one part at a time:")
    print(f"  1-NN (intersection)                 {accuracy(one_nn, 0, 0):.2%}")
    print(f"  + more matches per class            {accuracy(args.weights, 0, 0):.2%}")
    print(f"  + best cosine match                 {accuracy(args.weights, args.beta, 0):.2%}")
    print(f"  + networks (full model)             "
          f"{accuracy(args.weights, args.beta, args.lam):.2%}")
    print("Full model with other lam values:  " + "  ".join(
        f"{lam:g}: {accuracy(args.weights, args.beta, lam):.2%}"
        for lam in [0.005, 0.01, 0.015, 0.02]), flush=True)

    if args.test:
        test = pd.read_csv(args.test)
        if ("id" not in test or set(features) != {
                c for c in test if c.startswith("feature_")}):
            parser.error("Test CSV needs id and the same feature columns as training")
        if test["id"].isna().any() or test["id"].duplicated().any():
            parser.error("Test IDs must be nonmissing and unique")
        if args.output.resolve() in {args.train.resolve(), args.test.resolve()}:
            parser.error("Output must not overwrite an input CSV")
        T = test[features].to_numpy(dtype=float)
        if np.any(T < 0):
            parser.error("Histogram intersection needs nonnegative features")
        print("\nRefitting on all labeled rows for test predictions...", flush=True)
        inter, cosine = class_similarities(X, encoded, T, len(classes), top)
        probabilities = []
        for n, n_epochs in enumerate(epochs):
            final = network(n_epochs, 0, args.seed + n).fit(sqrt_unit_rows(X), encoded)
            probabilities.append(final.predict_proba(sqrt_unit_rows(T)))
            print(f"Network {n + 1} retrained for {n_epochs} epochs", flush=True)
        log_probs = np.log(np.mean(probabilities, axis=0) + 1e-9)
        labels = classes[combine(inter, cosine, log_probs, args.weights, args.beta,
                                 args.lam).argmax(1)]
        pd.DataFrame({"id": test["id"], "label": labels}).to_csv(args.output, index=False)
        print(f"Saved {len(labels)} predictions to {args.output}")
        print("Combine both tasks by ID with sample_submission.csv before submitting.")


if __name__ == "__main__":
    main()
