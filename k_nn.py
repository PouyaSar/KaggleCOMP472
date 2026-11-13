"""
Examples (run from the directory containing the competition CSVs):
    python k_nn.py image_train.csv --k 5
    python k_nn.py sentiment_train.csv --metric cosine --weights distance
    python k_nn.py image_train.csv --test image_test.csv --output image_predictions.csv

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


class KNNClassifier:
    """Exact k-NN with bounded query batches and deterministic voting.

    Standardization statistics are learned only in fit(). Equal vote totals
    are resolved by choosing the smallest class label. For distance voting,
    exact matches vote exclusively whenever any are present among the k
    selected neighbours. Equal distances are resolved by training row order.
    """

    def __init__(self, k=5, metric="euclidean", weights="uniform",
                 normalization="standard", batch_size=64):
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        if metric not in {"euclidean", "manhattan", "cosine"}:
            raise ValueError("metric must be euclidean, manhattan, or cosine")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be uniform or distance")
        if normalization not in {"none", "standard", "l2"}:
            raise ValueError("normalization must be none, standard, or l2")
        if (not isinstance(batch_size, (int, np.integer))
                or isinstance(batch_size, bool) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        self.k = k
        self.metric = metric
        self.weights = weights
        self.normalization = normalization
        self.batch_size = batch_size

    @staticmethod
    def _matrix(X):
        X = np.array(X, dtype=np.float64, copy=True)
        if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a finite 2D numerical matrix with features")
        return X

    @staticmethod
    def _unit_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms == 0, 1.0, norms)

    def _transform(self, X):
        if self.normalization == "standard":
            X = (X - self.mean_) / self.scale_
        elif self.normalization == "l2":
            X = self._unit_rows(X)
        if self.metric == "cosine":
            X = self._unit_rows(X)
        return X

    def fit(self, X, y):
        X = self._matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or len(y) == 0:
            raise ValueError("y must contain one label per nonempty training row")
        if pd.isna(y).any():
            raise ValueError("y must not contain missing labels")
        if self.k > len(X):
            raise ValueError("k cannot exceed the number of training rows")
        self.n_features_in_ = X.shape[1]
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        self.X_train_ = self._transform(X)
        self.classes_, self.y_encoded_ = np.unique(y, return_inverse=True)
        return self

    def _distances(self, queries):
        # Work one query at a time to avoid a batch x training x features
        # tensor. Direct subtraction also avoids squared-distance cancellation
        # for identical or very close vectors.
        distances = np.empty((len(queries), len(self.X_train_)))
        if self.metric == "cosine":
            distances[:] = np.clip(1.0 - queries @ self.X_train_.T, 0.0, 2.0)
            # Convention: two zero vectors have distance 0; one zero vector
            # and one nonzero vector have distance 1 (cosine is undefined).
            train_zero = ~self.X_train_.any(axis=1)
            query_zero = ~queries.any(axis=1)
            distances[np.ix_(query_zero, train_zero)] = 0.0
        else:
            for i, query in enumerate(queries):
                delta = self.X_train_ - query
                if self.metric == "euclidean":
                    distances[i] = np.sqrt(np.einsum("ij,ij->i", delta, delta))
                else:
                    distances[i] = np.abs(delta).sum(axis=1)
        return distances

    def predict(self, X):
        if not hasattr(self, "X_train_"):
            raise ValueError("Call fit before predict")
        X = self._matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features must match training features")
        X = self._transform(X)
        predictions = np.empty(len(X), dtype=self.classes_.dtype)
        for start in range(0, len(X), self.batch_size):
            distances = self._distances(X[start:start + self.batch_size])
            for offset, row in enumerate(distances):
                # Stable sorting makes ties reproducible, including ties at k.
                neighbours = np.argsort(row, kind="stable")[:self.k]
                selected_distances = row[neighbours]
                votes = np.ones(self.k)
                if self.weights == "distance":
                    exact = selected_distances <= 1e-12
                    if exact.any():
                        votes = exact.astype(float)
                    else:
                        # Equivalent to inverse distance, bounded by 1.
                        votes = selected_distances.min() / selected_distances
                totals = np.bincount(self.y_encoded_[neighbours], weights=votes,
                                     minlength=len(self.classes_))
                predictions[start + offset] = self.classes_[totals.argmax()]
        return predictions

    def score(self, X, y):
        predicted = self.predict(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(predicted) or len(y) == 0:
            raise ValueError("y must contain one label per nonempty prediction row")
        return float(np.mean(predicted == y))


def stratified_split(y, fraction=0.2, seed=42):
    """Return train/validation indices, retaining every class in both sets."""
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        if len(indices) < 2:
            raise ValueError("Each class needs at least two rows for validation")
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    return rng.permutation(train), rng.permutation(validation)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train", type=Path, help="Labeled training CSV")
    parser.add_argument("--test", type=Path, help="Optional unlabeled test CSV")
    parser.add_argument("--output", type=Path, default=Path("knn_predictions.csv"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", choices=["euclidean", "manhattan", "cosine"],
                        default="euclidean")
    parser.add_argument("--weights", choices=["uniform", "distance"], default="uniform")
    parser.add_argument("--normalization", choices=["none", "standard", "l2"],
                        default="standard")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.train)
    features = [c for c in frame.columns if c.startswith("feature_")]
    if not features or "label" not in frame.columns:
        parser.error("Training CSV needs feature_* columns and label")
    X = frame[features].to_numpy(dtype=float)
    y = frame["label"].to_numpy()
    train, validation = stratified_split(y, args.validation_fraction, args.seed)
    model = KNNClassifier(args.k, args.metric, args.weights,
                          args.normalization, args.batch_size)
    model.fit(X[train], y[train])
    print(f"Training rows: {len(train)}, validation rows: {len(validation)}", flush=True)
    print(f"Validation accuracy: {model.score(X[validation], y[validation]):.4%}", flush=True)

    if args.test:
        test = pd.read_csv(args.test)
        test_features = {c for c in test.columns if c.startswith("feature_")}
        if "id" not in test.columns or test_features != set(features):
            parser.error("Test CSV needs id and the same feature columns as training")
        if test["id"].isna().any() or test["id"].duplicated().any():
            parser.error("Test IDs must be nonmissing and unique")
        if args.output.resolve() in {args.train.resolve(), args.test.resolve()}:
            parser.error("Output must not overwrite an input CSV")
        model.fit(X, y)
        labels = model.predict(test[features].to_numpy(dtype=float))
        pd.DataFrame({"id": test["id"], "label": labels}).to_csv(args.output, index=False)
        print(f"Saved {len(labels)} predictions to {args.output}")
        print("Combine both tasks by ID with sample_submission.csv before submitting.")


if __name__ == "__main__":
    main()
