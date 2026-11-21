import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from k_nn import stratified_split


@dataclass
class Node:
    prediction: int 
    n_samples: int
    impurity: float
    feature: int = -1  # -1 indicates node is a leaf.
    threshold: float = 0.0
    left: object = None
    right: object = None


class DecisionTreeClassifier:
    """Numerical classification and regression with exhaustive greedy binary splits.

    max_depth counts edges from the root (depth 0). None grows until another
    stopping condition applies. min_impurity_decrease is the decrease in
    impurity weighted by the node's fraction of the original training rows.
    Feature/threshold ties prefer the first feature and lowest split point.
    No normalization is needed for axis-aligned decision trees.
    """

    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1,
                 criterion="gini", min_impurity_decrease=0.0):
        for name, value, minimum in [
            ("min_samples_split", min_samples_split, 2),
            ("min_samples_leaf", min_samples_leaf, 1),
        ]:
            if (not isinstance(value, (int, np.integer))
                    or isinstance(value, bool) or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if max_depth is not None and (
            not isinstance(max_depth, (int, np.integer))
            or isinstance(max_depth, bool) or max_depth < 0
        ):
            raise ValueError("max_depth must be None or a nonnegative integer")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be gini or entropy")
        if not np.isfinite(min_impurity_decrease) or min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be finite and nonnegative")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.min_impurity_decrease = min_impurity_decrease

    @staticmethod
    def _matrix(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a finite 2D numerical feature matrix")
        return X

    def _impurity(self, counts):
        """Impurity for one count vector or a matrix of count vectors."""
        probabilities = counts / np.sum(counts, axis=-1, keepdims=True)
        if self.criterion == "gini":
            return 1.0 - np.sum(probabilities ** 2, axis=-1)
        # log2(1) substitutes for log2(0); zero-probability terms contribute 0.
        return -np.sum(probabilities * np.log2(np.where(
            probabilities > 0, probabilities, 1.0)), axis=-1)

    def _best_split(self, X, encoded_y, counts, parent_impurity):
        n_rows, n_features = X.shape
        n_left = np.arange(1, n_rows)
        n_right = n_rows - n_left
        size_ok = ((n_left >= self.min_samples_leaf)
                   & (n_right >= self.min_samples_leaf))
        best_gain = 0.0
        best_feature, best_threshold = -1, 0.0
        for feature in range(n_features):
            order = np.argsort(X[:, feature], kind="stable")
            values = X[order, feature]
            candidates = np.flatnonzero(size_ok & (values[:-1] < values[1:]))
            if len(candidates) == 0:
                continue
            cumulative = np.zeros((n_rows, len(counts)), dtype=np.int64)
            cumulative[np.arange(n_rows), encoded_y[order]] = 1
            np.cumsum(cumulative, axis=0, out=cumulative)
            left_counts = cumulative[candidates]
            right_counts = counts - left_counts
            weighted = (n_left[candidates] * self._impurity(left_counts)
                        + n_right[candidates] * self._impurity(right_counts)) / n_rows
            index = int(np.argmin(weighted))
            gain = float(parent_impurity - weighted[index])
            if gain > best_gain + 1e-12:
                position = candidates[index]
                low, high = values[position], values[position + 1]
                # Avoid addition overflow and ensure floating-point rounding cannot move the threshold onto the first right-child value.
                threshold = low / 2.0 + high / 2.0
                if threshold < low or threshold >= high:
                    threshold = low
                best_gain = gain
                best_feature, best_threshold = feature, float(threshold)
        return best_feature, best_threshold, best_gain

    def fit(self, X, y):
        X = self._matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or len(y) == 0 or pd.isna(y).any():
            raise ValueError("y must contain one nonmissing label per training row")
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.n_nodes_ = self.n_leaves_ = self.depth_ = 0
        self.feature_importances_ = np.zeros(self.n_features_in_)
        total_rows = len(X)

        def make_node(indices):
            counts = np.bincount(encoded_y[indices], minlength=len(self.classes_))
            return Node(int(counts.argmax()), len(indices), float(self._impurity(counts)))

        indices = np.arange(total_rows)
        self.root_ = make_node(indices)
        pending = [(self.root_, indices, 0)]
        while pending:
            node, indices, depth = pending.pop()
            self.n_nodes_ += 1
            self.depth_ = max(self.depth_, depth)
            if (node.impurity <= 1e-12
                    or (self.max_depth is not None and depth >= self.max_depth)
                    or len(indices) < max(self.min_samples_split, 2 * self.min_samples_leaf)):
                self.n_leaves_ += 1
                continue
            local_y = encoded_y[indices]
            counts = np.bincount(local_y, minlength=len(self.classes_))
            feature, threshold, gain = self._best_split(
                X[indices], local_y, counts, node.impurity)
            weighted_gain = len(indices) / total_rows * gain
            if feature == -1 or weighted_gain + 1e-12 < self.min_impurity_decrease:
                self.n_leaves_ += 1
                continue
            mask = X[indices, feature] <= threshold
            left_indices, right_indices = indices[mask], indices[~mask]
            node.feature, node.threshold = feature, threshold
            node.left, node.right = make_node(left_indices), make_node(right_indices)
            self.feature_importances_[feature] += weighted_gain
            pending.append((node.right, right_indices, depth + 1))
            pending.append((node.left, left_indices, depth + 1))
        importance_sum = self.feature_importances_.sum()
        if importance_sum > 0:
            self.feature_importances_ /= importance_sum
        return self

    def predict(self, X):
        if not hasattr(self, "root_"):
            raise ValueError("Call fit before predict")
        X = self._matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features must match training features")
        result = np.empty(len(X), dtype=self.classes_.dtype)
        for i, row in enumerate(X):
            node = self.root_
            while node.feature != -1:
                node = node.left if row[node.feature] <= node.threshold else node.right
            result[i] = self.classes_[node.prediction]
        return result

    def score(self, X, y):
        predictions = self.predict(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) == 0 or len(y) != len(predictions):
            raise ValueError("y must contain one label per nonempty prediction row")
        return float(np.mean(predictions == y))


def depth_argument(text):
    return None if text.lower() == "none" else int(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train", type=Path)
    parser.add_argument("--max-depth", type=depth_argument, default=10,
                        help="Maximum depth, or none for no limit (default: 10)")
    parser.add_argument("--criterion", choices=["gini", "entropy"], default="gini")
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--min-impurity-decrease", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tree_predictions.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.train)
    features = [column for column in frame if column.startswith("feature_")]
    if not features or "label" not in frame:
        parser.error("Training CSV needs feature_* columns and label")
    X, y = frame[features].to_numpy(dtype=float), frame["label"].to_numpy()
    train, validation = stratified_split(y, args.validation_fraction, args.seed)
    model = DecisionTreeClassifier(args.max_depth, args.min_samples_split,
                                   args.min_samples_leaf, args.criterion,
                                   args.min_impurity_decrease)
    print(f"Training rows: {len(train)}, validation rows: {len(validation)}", flush=True)
    start = perf_counter()
    model.fit(X[train], y[train])
    print(f"Training time: {perf_counter() - start:.2f} seconds", flush=True)
    print(f"Training accuracy: {model.score(X[train], y[train]):.4%}", flush=True)
    print(f"Validation accuracy: {model.score(X[validation], y[validation]):.4%}", flush=True)
    print(f"Tree depth: {model.depth_}, nodes: {model.n_nodes_}, leaves: {model.n_leaves_}", flush=True)
    if args.test:
        test = pd.read_csv(args.test)
        if ("id" not in test or set(features) != {
                c for c in test if c.startswith("feature_")}):
            parser.error("Test CSV needs id and the same feature columns as training")
        if test["id"].isna().any() or test["id"].duplicated().any():
            parser.error("Test IDs must be nonmissing and unique")
        if args.output.resolve() in {args.train.resolve(), args.test.resolve()}:
            parser.error("Output must not overwrite an input CSV")
        print("Refitting on all labeled rows for test predictions...", flush=True)
        model.fit(X, y)
        predictions = model.predict(test[features].to_numpy(dtype=float))
        pd.DataFrame({"id": test["id"], "label": predictions}).to_csv(args.output, index=False)
        print(f"Saved {len(predictions)} predictions to {args.output}")
        print("Combine both tasks by ID with sample_submission.csv before submitting.")


if __name__ == "__main__":
    main()
