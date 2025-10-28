import numpy as np
import pandas as pd

class GroupEstimate:
    """
    Group-by estimator for categorical features.

    Parameters
    ----------
    estimate : {"mean", "median"}
        Which statistic of y to learn within each category combination.
    """
    def __init__(self, estimate: str = "mean"):
        est = estimate.lower()
        if est not in {"mean", "median"}:
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = est
        self._columns = None
        self._group_map = None
        self._default_category = None
        self._fallback_map = None

    def fit(self, X: pd.DataFrame, y, default_category: str | None = None):
        """Learn per-group estimates."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        y = pd.Series(y)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if y.isna().any():
            raise ValueError("y must not contain missing values.")

        self._columns = list(X.columns)
        func = np.mean if self.estimate == "mean" else np.median

        df = X.copy()
        df["_y"] = y.values

        grouped = (
            df.groupby(self._columns, observed=True, dropna=False)["_y"]
              .agg(func)
              .reset_index()
        )

        self._group_map = {
            tuple(row[col] for col in self._columns): row["_y"]
            for _, row in grouped.iterrows()
        }

        # Optional fallback
        self._default_category = None
        self._fallback_map = None
        if default_category is not None:
            if default_category not in self._columns:
                raise ValueError(f"default_category '{default_category}' not found in X.")
            self._default_category = default_category
            fallback = (
                df.groupby([default_category], observed=True, dropna=False)["_y"]
                  .agg(func)
                  .reset_index()
            )
            self._fallback_map = {
                row[default_category]: row["_y"] for _, row in fallback.iterrows()
            }

        return self

    def predict(self, X_):
        """Predict estimates for new observations."""
        if self._group_map is None:
            raise RuntimeError("Must call .fit() before .predict().")

        if isinstance(X_, pd.DataFrame):
            missing = set(self._columns) - set(X_.columns)
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            Xn = X_[self._columns].copy()
        else:
            arr = np.asarray(X_)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[1] != len(self._columns):
                raise ValueError(
                    f"Expected {len(self._columns)} columns but got {arr.shape[1]}"
                )
            Xn = pd.DataFrame(arr, columns=self._columns)

        preds, n_missing = [], 0
        for _, row in Xn.iterrows():
            key = tuple(row[col] for col in self._columns)
            val = self._group_map.get(key, None)
            if val is None and self._fallback_map is not None:
                cat_val = row[self._default_category]
                val = self._fallback_map.get(cat_val, np.nan)
            if val is None or pd.isna(val):
                n_missing += 1
                val = np.nan
            preds.append(val)

        if n_missing > 0:
            print(f"[GroupEstimate] {n_missing} unseen group(s) found.")

        return np.array(preds, dtype=float)
