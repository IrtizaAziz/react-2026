"""Small, explicit helpers for past-only feature construction.

These helpers are not automatically applied. They require a caller to sort by
the known event time and exclude the current row before computing aggregates.
"""
import pandas as pd
import numpy as np


def require_time_sorted(frame, time_column):
    values = pd.to_datetime(frame[time_column], errors="raise", utc=True) if not pd.api.types.is_numeric_dtype(frame[time_column]) else frame[time_column]
    if values.isna().any() or not values.is_monotonic_increasing:
        raise ValueError("Temporal features require nonmissing rows sorted in ascending event-time order")
    return values


def past_group_count(frame, group_column, time_column):
    """Return count of prior rows in each group, never including current/future rows."""
    require_time_sorted(frame, time_column)
    return frame.groupby(group_column, dropna=False, sort=False).cumcount().astype("int64")


def past_group_mean(frame, group_column, value_column, time_column):
    """Return prior-observation group mean; the current value is excluded."""
    require_time_sorted(frame, time_column)
    prior_sum = frame.groupby(group_column, dropna=False, sort=False)[value_column].transform(lambda s: s.shift(1).expanding().sum())
    prior_count = frame.groupby(group_column, dropna=False, sort=False)[value_column].transform(lambda s: s.shift(1).expanding().count())
    return prior_sum.div(prior_count.replace(0, np.nan)).astype(float)
