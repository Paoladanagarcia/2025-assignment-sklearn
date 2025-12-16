"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.multiclass import type_of_target

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.multiclass import type_of_target


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        -------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 1:
            raise ValueError("`n_neighbors` should be a positive integer.")

        X, y = validate_data(self, X, y, ensure_2d=True)
        y = np.asarray(y)
        y_type = type_of_target(y)
        if y_type not in ('binary', 'multiclass'):
            raise ValueError(f"Unknown label type: {y_type}")

        self._fit_X = X
        self._fit_y = y
        self.classes_ = np.unique(self._fit_y)
        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        -------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self, ['_fit_X', '_fit_y'])
        X = validate_data(self, X, reset=False)
        n_neighbors = min(self.n_neighbors, self._fit_X.shape[0])

        distances = pairwise_distances(X, self._fit_X)
        if n_neighbors == 1:
            return self._fit_y[np.argmin(distances, axis=1)]

        neigh_ind = np.argpartition(
            distances, kth=n_neighbors - 1, axis=1
        )[:, :n_neighbors]

        neighbor_labels = self._fit_y[neigh_ind]
        y_pred = np.empty(X.shape[0], dtype=self._fit_y.dtype)
        for i in range(neighbor_labels.shape[0]):
            labels, counts = np.unique(neighbor_labels[i], return_counts=True)
            y_pred[i] = labels[np.argmax(counts)]
        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        check_is_fitted(self, ['_fit_X', '_fit_y'])
        X, y = validate_data(self, X, y, reset=False)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def __repr__(self):
        return f"MonthlySplit(time_col={self.time_col!r})"

    def _extract_time_index(self, X):
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            raise ValueError("Input must contain datetime information.")
        
        if self.time_col == 'index':
            time_index = X.index
        else:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("datetime column requires a DataFrame.")
            if self.time_col not in X.columns:
                raise ValueError(f"{self.time_col} is not present in X.")
            time_index = X[self.time_col]

        time_index = pd.Index(time_index)
        if not pd.api.types.is_datetime64_any_dtype(time_index):
            raise ValueError("time column must be datetime.")
        return pd.DatetimeIndex(time_index)

    def _group_by_month(self, X):
        time_index = self._extract_time_index(X)
        order = np.argsort(time_index.view('int64'))
        sorted_idx = order.astype(int)
        sorted_periods = time_index[order].to_period('M')

        month_to_indices = {}
        ordered_months = []
        for period, idx in zip(sorted_periods, sorted_idx):
            if period not in month_to_indices:
                month_to_indices[period] = []
                ordered_months.append(period)
            month_to_indices[period].append(int(idx))
        return ordered_months, month_to_indices

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        months, _ = self._group_by_month(X)
        return max(len(months) - 1, 0)

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """

        months, month_to_indices = self._group_by_month(X)
        for i in range(len(months) - 1):
            idx_train = np.array(month_to_indices[months[i]], dtype=int)
            idx_test = np.array(month_to_indices[months[i + 1]], dtype=int)
            if idx_train.size == 0 or idx_test.size == 0:
                continue
            yield idx_train, idx_test
