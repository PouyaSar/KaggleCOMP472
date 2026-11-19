"""From-scratch random forest for the COMP472 feature CSVs.

Examples (run alongside decision_tree.py and k_nn.py):
    python3 random_forest.py image_train.csv --n-estimators 50 --max-depth 20
    python3 random_forest.py sentiment_train.csv --max-features sqrt
    python3 random_forest.py image_train.csv --test image_test.csv --output forest_image_predictions.csv

Each tree receives a bootstrap sample (sampling rows with replacement).
At EACH node it searches a fresh random subset of features. Predictions
use majority voting across trees; tied votes choose the smallest label.
NumPy and Pandas are the only third-party dependencies; no sklearn models.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from decision_tree import DecisionTreeClassifier, depth_argument
from k_nn import stratified_split


def feature_count(value, n_features):
    """Resolve sqrt/log2/all, an integer count, or a fractional feature count."""
    if isinstance(value, str):
        if value == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if value == "log2":
            return max(1, int(np.log2(n_features)))
        if value == "all":
            return n_features
    elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if 1 <= value <= n_features:
            return int(value)
    elif isinstance(value, (float, np.floating)):
        if np.isfinite(value) and 0 < value <= 1:
            return max(1, int(value * n_features))
    raise ValueError("max_features must be sqrt, log2, all, an integer from 1 to "
                     "the feature count, or a float in (0, 1]")


class _RandomizedTree(DecisionTreeClassifier):
    """Use the existing CART implementation with random features per node."""

    def __init__(self, n_split_features, seed, **kwargs):
        super().__init__(**kwargs)
        self.n_split_features = n_split_features
        self.seed = seed

    def fit(self, X, y):
        self.rng_ = np.random.default_rng(self.seed)
        return super().fit(X, y)

    def _best_split(self, X, encoded_y, counts, parent_impurity):
        if self.n_split_features == X.shape[1]:
            return super()._best_split(X, encoded_y, counts, parent_impurity)
        features = np.sort(self.rng_.choice(
            X.shape[1], size=self.n_split_features, replace=False))
        feature, threshold, gain = super()._best_split(
            X[:, features], encoded_y, counts, parent_impurity)
        if feature != -1:
            feature = int(features[feature])
        return feature, threshold, gain


class RandomForestClassifier:
    """Bootstrap randomized CART trees with hard majority voting.

    Defaults: 50 trees, maximum depth 20, sqrt(features) candidates per split.
    All tree hyperparameters have the same meaning as in decision_tree.py.
    A fixed random_state reproduces bootstraps, node selections, and votes.
    """

    def __init__(self, n_estimators=50, max_depth=20, max_features="sqrt",
                 min_samples_split=2, min_samples_leaf=1, criterion="gini",
                 min_impurity_decrease=0.0, bootstrap=True, random_state=42):
        if (not isinstance(n_estimators, (int, np.integer))
                or isinstance(n_estimators, bool) or n_estimators < 1):
            raise ValueError("n_estimators must be a positive integer")
        if not isinstance(bootstrap, bool):
            raise ValueError("bootstrap must be True or False")
        if (random_state is not None and (
                not isinstance(random_state, (int, np.integer))
                or isinstance(random_state, bool) or random_state < 0)):
            raise ValueError("random_state must be None or a nonnegative integer")
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.tree_params = dict(max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                criterion=criterion,
                                min_impurity_decrease=min_impurity_decrease)
        DecisionTreeClassifier(**self.tree_params)  # Validate tree parameters now.

    def fit(self, X, y):
        X = DecisionTreeClassifier._matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or len(y) == 0 or pd.isna(y).any():
            raise ValueError("y must contain one nonmissing label per training row")
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.max_features_ = feature_count(self.max_features, self.n_features_in_)
        self.estimators_ = []
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_estimators):
            indices = (rng.integers(0, len(X), size=len(X))
                       if self.bootstrap else np.arange(len(X)))
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            tree = _RandomizedTree(self.max_features_, seed, **self.tree_params)
            tree.fit(X[indices], y[indices])
            self.estimators_.append(tree)
        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.estimators_], axis=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def predict(self, X):
        if not hasattr(self, "estimators_") or not self.estimators_:
            raise ValueError("Call fit before predict")
        X = DecisionTreeClassifier._matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features must match training features")
        # Map original labels to a common class index: bootstrap samples may
        # omit classes, so individual trees can have different classes_ arrays.
        votes = np.zeros((len(X), len(self.classes_)), dtype=np.int64)
        rows = np.arange(len(X))
        for tree in self.estimators_:
            encoded = np.searchsorted(self.classes_, tree.predict(X))
            votes[rows, encoded] += 1
        return self.classes_[votes.argmax(axis=1)]

    def score(self, X, y):
        predictions = self.predict(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) == 0 or len(y) != len(predictions):
            raise ValueError("y must contain one label per nonempty prediction row")
        return float(np.mean(predictions == y))


def feature_argument(text):
    if text in {"sqrt", "log2", "all"}:
        return text
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "Use sqrt, log2, all, an integer count, or a fraction") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train", type=Path)
    parser.add_argument("--n-estimators", type=int, default=50, help="Number of trees (default: 50)")
    parser.add_argument("--max-depth", type=depth_argument, default=20,
                        help="Maximum depth, or none for no limit (default: 20)")
    parser.add_argument("--max-features", type=feature_argument, default="sqrt",
                        help="sqrt, log2, all, integer count, or fraction (default: sqrt)")
    parser.add_argument("--criterion", choices=["gini", "entropy"], default="gini")
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--min-impurity-decrease", type=float, default=0.0)
    parser.add_argument("--no-bootstrap", action="store_true", help="Use all rows for each tree")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path, default=Path("forest_predictions.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.train)
    features = [column for column in frame if column.startswith("feature_")]
    if not features or "label" not in frame:
        parser.error("Training CSV needs feature_* columns and label")
    X, y = frame[features].to_numpy(dtype=float), frame["label"].to_numpy()
    train, validation = stratified_split(y, args.validation_fraction, args.seed)
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        max_features=args.max_features, min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf, criterion=args.criterion,
        min_impurity_decrease=args.min_impurity_decrease,
        bootstrap=not args.no_bootstrap, random_state=args.seed)
    print(f"Training rows: {len(train)}, validation rows: {len(validation)}", flush=True)
    print(f"Training {args.n_estimators} trees...", flush=True)
    start = perf_counter()
    model.fit(X[train], y[train])
    print(f"Training time: {perf_counter() - start:.2f} seconds", flush=True)
    print(f"Training accuracy: {model.score(X[train], y[train]):.4%}", flush=True)
    print(f"Validation accuracy: {model.score(X[validation], y[validation]):.4%}", flush=True)
    print(f"Features considered per split: {model.max_features_}", flush=True)
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
