import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from k_nn import stratified_split


class NeuralNetworkClassifier:
    """
    PyTorch calculates gradients and updates weights using Adam or SGD.
    Cross-entropy trains on raw scores; softmax is used for probabilities.
    Weight decay penalizes weights, not biases.
    Validation selects the highest accuracy, with lower loss breaking ties.
    """

    def __init__(self, hidden_layers=(256, 64), activation='relu',
                 learning_rate=0.01, optimizer='adam', batch_size=32,
                 epochs=100, weight_decay=0.0001, dropout=0.2,
                 normalization='standard', patience=10, random_state=42):
        self.hidden_layers = tuple(hidden_layers)
        for name, value, minimum in [('batch_size', batch_size, 1),
                                      ('epochs', epochs, 1), ('patience', patience, 0),
                                      ('random_state', random_state, 0)]:
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if any(not isinstance(size, (int, np.integer)) or isinstance(size, bool)
               or size < 1 for size in self.hidden_layers):
            raise ValueError('Hidden layer sizes must be positive integers')
        if activation not in {'relu', 'tanh', 'sigmoid'}:
            raise ValueError('activation must be relu, tanh, or sigmoid')
        if optimizer not in {'adam', 'sgd'}:
            raise ValueError('optimizer must be adam or sgd')
        if normalization not in {'standard', 'l2', 'none'}:
            raise ValueError('normalization must be standard, l2, or none')
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')
        if not np.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.normalization = normalization
        self.patience = patience
        self.random_state = random_state

    @staticmethod
    def _matrix(X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError('X must be a finite 2D numerical feature matrix')
        return X

    def _transform(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Prediction features must match training features')
        if self.normalization == 'standard':
            return (X - self.mean_) / self.scale_
        if self.normalization == 'l2':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            return X / np.where(norms == 0, 1.0, norms)
        return X

    def _build_model(self):
        layers = []
        sizes = (self.n_features_in_,) + self.hidden_layers + (len(self.classes_),)
        activation = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}[self.activation]
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(n_in, n_out))
            if i < len(sizes) - 2:
                layers.extend([activation(), nn.Dropout(self.dropout)])
        self.model_ = nn.Sequential(*layers)
        # Keep L2 regularization on weights only, as in the NumPy version.
        groups = [
            {'params': [p for name, p in self.model_.named_parameters() if name.endswith('weight')],
             'weight_decay': self.weight_decay},
            {'params': [p for name, p in self.model_.named_parameters() if name.endswith('bias')],
             'weight_decay': 0.0},
        ]
        optimizer = torch.optim.Adam if self.optimizer == 'adam' else torch.optim.SGD
        self.optimizer_ = optimizer(groups, lr=self.learning_rate)
        self.loss_fn_ = nn.CrossEntropyLoss()

    @torch.inference_mode()
    def _evaluate(self, X, encoded_y):
        # eval() disables dropout, so validation uses the complete network.
        self.model_.eval()
        loss, correct = 0.0, 0
        for start in range(0, len(X), self.batch_size):
            stop = start + self.batch_size
            logits = self.model_(X[start:stop])
            labels = encoded_y[start:stop]
            loss += self.loss_fn_(logits, labels).item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
        return loss / len(X), correct / len(X)

    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        X = self._matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or len(y) == 0 or pd.isna(y).any():
            raise ValueError('y needs one nonmissing label per training row')
        if (X_val is None) != (y_val is None):
            raise ValueError('Provide both X_val and y_val, or neither')
        self.n_features_in_ = X.shape[1]
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1
        X = self._transform(X)
        if X_val is not None:
            X_val = self._transform(self._matrix(X_val))
            y_val = np.asarray(y_val)
            if (y_val.ndim != 1 or len(y_val) != len(X_val) or len(y_val) == 0
                    or pd.isna(y_val).any() or not np.isin(y_val, self.classes_).all()):
                raise ValueError('Validation needs one known class label per row')
            encoded_val = np.searchsorted(self.classes_, y_val)
        torch.manual_seed(self.random_state)
        self._build_model()
        X = torch.from_numpy(np.ascontiguousarray(X))
        encoded_y = torch.as_tensor(encoded_y, dtype=torch.long)
        if X_val is not None:
            X_val = torch.from_numpy(np.ascontiguousarray(X_val))
            encoded_val = torch.as_tensor(encoded_val, dtype=torch.long)
        loader = DataLoader(TensorDataset(X, encoded_y), batch_size=self.batch_size,
                            shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(self.random_state))
        self.history_ = []
        self.best_epoch_ = 0
        self.best_validation_loss_ = float('inf')
        self.best_validation_accuracy_ = -float('inf')
        best_parameters, stale = None, 0
        for epoch in range(1, self.epochs + 1):
            self.model_.train()  # Enable dropout for training batches.
            for batch_X, batch_y in loader:
                self.optimizer_.zero_grad(set_to_none=True)
                logits = self.model_(batch_X)
                loss = self.loss_fn_(logits, batch_y)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Training diverged; try a lower learning rate')
                loss.backward()  # PyTorch calculates gradients automatically.
                self.optimizer_.step()
            train_loss, train_accuracy = self._evaluate(X, encoded_y)
            if not np.isfinite(train_loss):
                raise FloatingPointError('Training diverged; try a lower learning rate')
            entry = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy)
            message = f'Epoch {epoch:3d}: train loss={train_loss:.4f}, accuracy={train_accuracy:.4%}'
            if X_val is not None:
                val_loss, val_accuracy = self._evaluate(X_val, encoded_val)
                if not np.isfinite(val_loss):
                    raise FloatingPointError('Nonfinite validation loss; try a lower learning rate')
                entry.update(validation_loss=val_loss, validation_accuracy=val_accuracy)
                message += f' | validation loss={val_loss:.4f}, accuracy={val_accuracy:.4%}'
                if (val_accuracy > self.best_validation_accuracy_
                        or (val_accuracy == self.best_validation_accuracy_
                            and val_loss < self.best_validation_loss_)):
                    self.best_validation_accuracy_ = val_accuracy
                    self.best_validation_loss_ = val_loss
                    self.best_epoch_ = epoch
                    best_parameters = {key: value.detach().clone() for key, value in self.model_.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
            else:
                self.best_epoch_ = epoch
            self.history_.append(entry)
            if verbose:
                print(message, flush=True)
            if X_val is not None and self.patience and stale >= self.patience:
                if verbose:
                    print(f'Early stopping: restoring epoch {self.best_epoch_}.', flush=True)
                break
        self.n_epochs_ = len(self.history_)
        if best_parameters is not None:
            self.model_.load_state_dict(best_parameters)
        self.model_.eval()
        return self

    @torch.inference_mode()
    def predict_proba(self, X):
        if not hasattr(self, 'model_'):
            raise ValueError('Call fit before predict')
        X = torch.from_numpy(np.ascontiguousarray(self._transform(self._matrix(X))))
        self.model_.eval()
        probabilities = np.empty((len(X), len(self.classes_)), dtype=np.float32)
        for start in range(0, len(X), self.batch_size):
            stop = start + self.batch_size
            probabilities[start:stop] = torch.softmax(self.model_(X[start:stop]), dim=1).numpy()
        return probabilities

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X, y):
        predictions = self.predict(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) == 0 or len(y) != len(predictions):
            raise ValueError('y needs one label per nonempty prediction row')
        return float(np.mean(predictions == y))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('train', type=Path)
    parser.add_argument('--hidden-layers', type=int, nargs='*', default=[256, 64], help='Hidden widths; empty means a linear softmax classifier')
    parser.add_argument('--activation', choices=['relu', 'tanh', 'sigmoid'], default='relu')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--normalization', choices=['standard', 'l2', 'none'], default='standard')
    parser.add_argument('--patience', type=int, default=10, help='Epochs without a better validation checkpoint (accuracy first, loss breaks ties); 0 disables stopping')
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test', type=Path)
    parser.add_argument('--output', type=Path, default=Path('nn_predictions.csv'))
    args = parser.parse_args()
    # Small dense networks are fast on one CPU thread without thread overhead.
    torch.set_num_threads(1)
    frame = pd.read_csv(args.train)
    features = [column for column in frame if column.startswith('feature_')]
    if not features or 'label' not in frame:
        parser.error('Training CSV needs feature_* columns and label')
    X, y = frame[features].to_numpy(dtype=np.float32), frame['label'].to_numpy()
    train, val = stratified_split(y, args.validation_fraction, args.seed)
    settings = dict(hidden_layers=args.hidden_layers, activation=args.activation,
                    learning_rate=args.learning_rate, optimizer=args.optimizer,
                    batch_size=args.batch_size, epochs=args.epochs,
                    weight_decay=args.weight_decay, dropout=args.dropout,
                    normalization=args.normalization, patience=args.patience,
                    random_state=args.seed)
    model = NeuralNetworkClassifier(**settings)
    print(f'Training rows: {len(train)}, validation rows: {len(val)}', flush=True)
    print('Architecture: ' + ' -> '.join(map(str, [len(features)] + args.hidden_layers + [len(np.unique(y))])), flush=True)
    start = perf_counter()
    model.fit(X[train], y[train], X[val], y[val], verbose=True)
    print(f'Training time: {perf_counter() - start:.2f} seconds', flush=True)
    print(f'Selected epoch (highest validation accuracy): {model.best_epoch_}', flush=True)
    print(f'Training accuracy: {model.score(X[train], y[train]):.4%}', flush=True)
    print(f'Validation accuracy: {model.score(X[val], y[val]):.4%}', flush=True)
    if args.test:
        test = pd.read_csv(args.test)
        if ('id' not in test or set(features) != {c for c in test if c.startswith('feature_')}):
            parser.error('Test CSV needs id and the same feature columns as training')
        if test['id'].isna().any() or test['id'].duplicated().any():
            parser.error('Test IDs must be nonmissing and unique')
        if args.output.resolve() in {args.train.resolve(), args.test.resolve()}:
            parser.error('Output must not overwrite an input CSV')
        settings.update(epochs=model.best_epoch_, patience=0)
        print(f'Retraining on all labeled rows for {model.best_epoch_} epochs...', flush=True)
        final_model = NeuralNetworkClassifier(**settings).fit(X, y, verbose=True)
        predictions = final_model.predict(test[features].to_numpy(dtype=np.float32))
        pd.DataFrame({'id': test['id'], 'label': predictions}).to_csv(args.output, index=False)
        print(f'Saved {len(predictions)} predictions to {args.output}')
        print('Combine both tasks by ID with sample_submission.csv before submitting.')


if __name__ == '__main__':
    main()
