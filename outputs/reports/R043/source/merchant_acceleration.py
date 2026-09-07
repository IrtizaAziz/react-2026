"""Strictly-prior merchant short-term activity and adjacent-hour acceleration."""

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .customer_history import _timestamp_ns


AVAILABLE_FEATURES = [
    "merchant_transactions_1h",
    "merchant_log_rate_change_1h_vs_previous_1h",
]
SOURCE_DEPENDENCIES = (__file__.replace("merchant_acceleration.py", "customer_history.py"),)
_HOUR_NS = 3_600 * 1_000_000_000
_HOUR_SECONDS = 3_600


def _inputs(timestamp, merchant_id):
    times = _timestamp_ns(timestamp)
    merchants = pd.Series(merchant_id).astype("string").fillna("__MISSING_MERCHANT__").to_numpy(dtype=str)
    return times, merchants


def _compute(timestamp, merchant_id, state=None):
    """Score timestamp batches, then append that whole batch to causal state."""
    times, merchants = _inputs(timestamp, merchant_id)
    state = {} if state is None else state
    recent = state.setdefault("recent", defaultdict(deque))
    previous = state.setdefault("previous", defaultdict(deque))
    out_count = np.zeros(len(times), dtype=np.int32)
    out_change = np.zeros(len(times), dtype=np.float32)
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        now = times[start]
        for merchant in set(merchants[start:end]):
            old = previous[merchant]
            while old and old[0] < now - 2 * _HOUR_NS:
                old.popleft()
            current = recent[merchant]
            while current and current[0] < now - _HOUR_NS:
                old.append(current.popleft())
        for i in range(start, end):
            merchant = merchants[i]
            a, b = len(recent[merchant]), len(previous[merchant])
            out_count[i] = a
            out_change[i] = np.log1p(a) - np.log1p(b)
        for merchant in merchants[start:end]:
            recent[merchant].append(now)
    return pd.DataFrame({AVAILABLE_FEATURES[0]: out_count, AVAILABLE_FEATURES[1]: out_change})


def merchant_acceleration_production(timestamp, merchant_id):
    return _compute(timestamp, merchant_id)


def merchant_acceleration_simple_oracle(timestamp, merchant_id):
    """Direct reference implementation for causal certification only."""
    times, merchants = _inputs(timestamp, merchant_id)
    counts = np.zeros(len(times), dtype=np.int32)
    changes = np.zeros(len(times), dtype=np.float32)
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        now = times[start]
        for i in range(start, end):
            same = merchants[:start] == merchants[i]
            a = int(np.sum(same & (times[:start] >= now - _HOUR_NS)))
            b = int(np.sum(same & (times[:start] >= now - 2 * _HOUR_NS) & (times[:start] < now - _HOUR_NS)))
            counts[i] = a
            changes[i] = np.log1p(a) - np.log1p(b)
    return pd.DataFrame({AVAILABLE_FEATURES[0]: counts, AVAILABLE_FEATURES[1]: changes})


def merchant_acceleration_chunked(chunks):
    state, pending, outputs = {}, pd.DataFrame(), []
    for chunk in chunks:
        combined = pd.concat([pending, chunk.copy()])
        if combined.empty:
            continue
        times = _timestamp_ns(combined["timestamp"])
        ready_end = np.searchsorted(times, times[-1], side="left")
        if ready_end:
            ready = combined.iloc[:ready_end]
            outputs.append(_compute(ready.timestamp, ready.merchant_id, state).set_axis(ready.index))
        pending = combined.iloc[ready_end:]
    if not pending.empty:
        outputs.append(_compute(pending.timestamp, pending.merchant_id, state).set_axis(pending.index))
    return pd.concat(outputs).sort_index() if outputs else pd.DataFrame(columns=AVAILABLE_FEATURES)


def build_features(raw_frame, requested_features):
    try:
        requested = tuple(requested_features) if not isinstance(requested_features, str) else ()
    except TypeError:
        requested = ()
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(AVAILABLE_FEATURES):
        raise ValueError("requested_features must be a non-empty ordered subset of AVAILABLE_FEATURES")
    missing = [column for column in ("timestamp", "merchant_id") if column not in raw_frame]
    if missing:
        raise ValueError(f"raw_frame is missing required columns: {missing}")
    values = merchant_acceleration_production(raw_frame["timestamp"], raw_frame["merchant_id"])
    return values.loc[:, list(requested)].set_axis(raw_frame.index)


def _frame():
    h = _HOUR_SECONDS
    return pd.DataFrame({"transaction_id": list("abcdefghi"), "timestamp": [0, h, h, 2*h, 2*h, 3*h, 4*h, 4*h, 5*h], "merchant_id": ["m"] * 8 + ["x"], "fraud": [0, 1] * 4 + [0]})


def certification_cases():
    def actual(frame):
        return merchant_acceleration_production(frame.timestamp, frame.merchant_id)
    def oracle():
        frame = _frame(); pd.testing.assert_frame_equal(actual(frame), merchant_acceleration_simple_oracle(frame.timestamp, frame.merchant_id))
    def strict_past():
        values = actual(_frame())
        if values.merchant_transactions_1h.tolist() != [0, 1, 1, 2, 2, 2, 1, 1, 0]: raise AssertionError("recent-hour count is not strictly prior")
    def equal_timestamp_isolation():
        values = actual(_frame())
        if not values.iloc[1].equals(values.iloc[2]) or not values.iloc[3].equals(values.iloc[4]): raise AssertionError("tied rows saw each other")
    def permutation_invariance():
        frame = _frame(); permuted = frame.iloc[[0, 2, 1, 3, 4, 5, 7, 6, 8]].reset_index(drop=True)
        left = pd.concat([frame.transaction_id, actual(frame)], axis=1).set_index("transaction_id").sort_index()
        right = pd.concat([permuted.transaction_id, actual(permuted)], axis=1).set_index("transaction_id").sort_index()
        pd.testing.assert_frame_equal(left, right)
    def duplicate_same_timestamp_pair():
        frame = _frame(); frame = pd.concat([frame.iloc[:3], frame.iloc[[2]], frame.iloc[3:]], ignore_index=True); values = actual(frame)
        if values.iloc[1].tolist() != values.iloc[2].tolist() or values.iloc[2].tolist() != values.iloc[3].tolist(): raise AssertionError("same-timestamp duplicate was not isolated")
    def future_independence():
        frame = _frame(); expected = actual(frame); future = pd.DataFrame({"timestamp": [10 * _HOUR_SECONDS], "merchant_id": ["m"]})
        pd.testing.assert_frame_equal(expected, actual(pd.concat([frame, future], ignore_index=True)).iloc[:len(frame)].reset_index(drop=True))
    def future_value_mutation_independence():
        frame = _frame(); changed = frame.copy(); changed.loc[6:, "merchant_id"] = "m"; pd.testing.assert_frame_equal(actual(frame).iloc[:6], actual(changed).iloc[:6])
    def label_independence():
        frame = _frame(); pd.testing.assert_frame_equal(actual(frame), actual(frame.assign(fraud=1 - frame.fraud)))
    def chunk_vs_whole():
        frame = _frame(); pd.testing.assert_frame_equal(build_features(frame, AVAILABLE_FEATURES), merchant_acceleration_chunked([frame.iloc[:3], frame.iloc[3:6], frame.iloc[6:]]))
    def raw_frame_compatibility():
        frame = _frame().drop(columns="fraud"); result = build_features(frame, AVAILABLE_FEATURES)
        if not result.index.equals(frame.index): raise AssertionError("raw-frame index was not preserved")
    def requested_subset_order():
        result = build_features(_frame(), list(reversed(AVAILABLE_FEATURES)))
        if list(result) != list(reversed(AVAILABLE_FEATURES)): raise AssertionError("requested order was not preserved")
    def zero_history_behavior():
        values = actual(_frame())
        if values.iloc[0].tolist() != [0, 0.0]: raise AssertionError("zero history is not zero")
    def exact_window_boundaries():
        h = _HOUR_SECONDS
        frame = pd.DataFrame({"timestamp": [-2*h, -h, 0], "merchant_id": ["m"] * 3})
        values = actual(frame)
        if values.merchant_transactions_1h.tolist() != [0, 1, 1]: raise AssertionError("[t-1h,t) boundary is incorrect")
        if not np.isclose(values.iloc[2, 1], np.log(2) - np.log(2)): raise AssertionError("adjacent-hour boundary is incorrect")
    return {"oracle": oracle, "strict_past": strict_past, "equal_timestamp_isolation": equal_timestamp_isolation, "permutation_invariance": permutation_invariance, "duplicate_same_timestamp_pair": duplicate_same_timestamp_pair, "future_independence": future_independence, "future_value_mutation_independence": future_value_mutation_independence, "label_independence": label_independence, "chunk_vs_whole": chunk_vs_whole, "raw_frame_compatibility": raw_frame_compatibility, "requested_subset_order": requested_subset_order, "zero_history": zero_history_behavior, "exact_window_boundaries": exact_window_boundaries}


def diagnostics(candidate, _oof, *, config, added_features):
    relevant = ["merchant_prior_transaction_count", "merchant_transactions_24h", "merchant_unique_customers_24h", "merchant_seconds_since_last"]
    return {"pearson_correlations": {added: {other: float(candidate[added].corr(candidate[other])) for other in relevant if other in candidate} for added in added_features}}
