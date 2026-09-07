"""Strictly-past, timestamp-batched customer temporal-habit shares."""

from collections import defaultdict

import numpy as np
import pandas as pd

from .customer_history import _timestamp_ns


FEATURES = ["customer_daypart_share", "customer_weekday_share"]
AVAILABLE_FEATURES = tuple(FEATURES)
_DAY_SECONDS = 86_400


def _inputs(timestamp, customer_id):
    times = _timestamp_ns(timestamp)
    raw_timestamps = pd.Series(timestamp)
    if pd.api.types.is_numeric_dtype(raw_timestamps):
        seconds = raw_timestamps.to_numpy(dtype=np.int64)
        hours = ((seconds % _DAY_SECONDS) // 3_600).astype(np.int8)
        weekdays = ((seconds // _DAY_SECONDS) + 3) % 7  # Unix epoch was Thursday.
    else:
        parsed = pd.to_datetime(raw_timestamps, errors="raise", format="mixed")
        hours = parsed.dt.hour.to_numpy(dtype=np.int8)
        weekdays = parsed.dt.weekday.to_numpy(dtype=np.int8)
    customers = pd.Series(customer_id).astype("string").fillna("__MISSING_CUSTOMER__").to_numpy(dtype=str)
    return times, customers, hours, weekdays


def _empty(n):
    return {name: np.zeros(n, dtype=float) for name in FEATURES}


def _compute(timestamp, customer_id, state):
    times, customers, hours, weekdays = _inputs(timestamp, customer_id)
    out = _empty(len(times))
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    counts = state.setdefault("counts", defaultdict(int))
    dayparts = state.setdefault("dayparts", defaultdict(lambda: defaultdict(int)))
    weekdays_seen = state.setdefault("weekdays", defaultdict(lambda: defaultdict(int)))
    for start, end in zip(starts, ends):
        for i in range(start, end):
            customer = customers[i]
            prior = counts[customer]
            if prior:
                part = min(hours[i] // 6, 3)
                out["customer_daypart_share"][i] = dayparts[customer][part] / prior
                out["customer_weekday_share"][i] = weekdays_seen[customer][int(weekdays[i])] / prior
        for i in range(start, end):
            customer = customers[i]
            counts[customer] += 1
            dayparts[customer][min(hours[i] // 6, 3)] += 1
            weekdays_seen[customer][int(weekdays[i])] += 1
    return pd.DataFrame(out)


def customer_temporal_habits_oracle(timestamp, customer_id):
    """Direct reference implementation for certification."""
    times, customers, hours, weekdays = _inputs(timestamp, customer_id)
    out = _empty(len(times))
    history = defaultdict(list)
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        for i in range(start, end):
            prior = history[customers[i]]
            if prior:
                part = min(hours[i] // 6, 3)
                out["customer_daypart_share"][i] = sum(p == part for _, p, _ in prior) / len(prior)
                out["customer_weekday_share"][i] = sum(w == weekdays[i] for _, _, w in prior) / len(prior)
        for i in range(start, end):
            history[customers[i]].append((times[i], min(hours[i] // 6, 3), int(weekdays[i])))
    return pd.DataFrame(out)


def customer_temporal_habits_production(timestamp, customer_id):
    """Compute both shares from raw timestamp and customer columns only."""
    return _compute(timestamp, customer_id, {})


def customer_temporal_habits_chunked(chunks):
    """Process chronological chunks without splitting a timestamp batch."""
    state, pending, outputs = {}, pd.DataFrame(), []
    for chunk in chunks:
        combined = pd.concat([pending, chunk.copy()])
        if combined.empty:
            continue
        times = _timestamp_ns(combined["timestamp"])
        ready_end = np.searchsorted(times, times[-1], side="left")
        if ready_end:
            ready = combined.iloc[:ready_end]
            outputs.append(_compute(ready["timestamp"], ready["customer_id"], state).set_axis(ready.index))
        pending = combined.iloc[ready_end:]
    if not pending.empty:
        outputs.append(_compute(pending["timestamp"], pending["customer_id"], state).set_axis(pending.index))
    return pd.concat(outputs).sort_index() if outputs else pd.DataFrame(columns=FEATURES)


def build_features(raw_frame, requested_features):
    try:
        requested = tuple(requested_features) if not isinstance(requested_features, str) else ()
    except TypeError:
        requested = ()
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(AVAILABLE_FEATURES):
        raise ValueError("requested_features must be a non-empty ordered subset of AVAILABLE_FEATURES")
    required = ("timestamp", "customer_id")
    missing = [column for column in required if column not in raw_frame]
    if missing:
        raise ValueError(f"raw_frame is missing required columns: {missing}")
    values = customer_temporal_habits_production(raw_frame["timestamp"], raw_frame["customer_id"])
    return values.loc[:, list(requested)].set_axis(raw_frame.index)


def _certification_frame():
    day = _DAY_SECONDS
    return pd.DataFrame({
        "transaction_id": list("abcdefghijk"),
        "timestamp": [0, 0, 3_600, 21_600, 21_600, day, day + 3_600, day + 43_200, day + 43_200, 2 * day, 2 * day + 21_600],
        "customer_id": ["a", "a", "a", "a", "b", "a", "a", "a", "a", "a", "a"],
        "fraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })


def certification_cases():
    def actual(frame):
        return customer_temporal_habits_production(frame.timestamp, frame.customer_id)

    def oracle():
        frame = _certification_frame()
        pd.testing.assert_frame_equal(actual(frame), customer_temporal_habits_oracle(frame.timestamp, frame.customer_id))

    def strict_past():
        values = actual(_certification_frame())
        expected = [0.0, 0.0, 1.0, 0.0, 0.0, 0.75, 0.8, 0.0, 0.0, 0.625, 0.1111111111111111]
        if values.customer_daypart_share.tolist() != expected:
            raise AssertionError("temporal habit share included current or future transactions")

    def equal_timestamp_isolation():
        values = actual(_certification_frame())
        if values.iloc[7].tolist() != values.iloc[8].tolist():
            raise AssertionError("equal-timestamp rows influenced one another")

    def permutation_invariance():
        frame = _certification_frame(); values = actual(frame)
        permuted = frame.iloc[[1, 0, 2, 4, 3, 5, 6, 7, 8, 9, 10]].reset_index(drop=True)
        observed = actual(permuted)
        pd.testing.assert_frame_equal(pd.concat([frame.transaction_id, values], axis=1).set_index("transaction_id").sort_index(),
                                      pd.concat([permuted.transaction_id, observed], axis=1).set_index("transaction_id").sort_index())

    def duplicate_same_timestamp_pair():
        frame = pd.DataFrame({"timestamp": [0, 3_600, 3_600, 43_200], "customer_id": ["x", "x", "x", "x"]})
        values = actual(frame)
        if values.iloc[1].tolist() != [1.0, 1.0] or values.iloc[2].tolist() != [1.0, 1.0]:
            raise AssertionError("duplicate same-timestamp rows did not share frozen state")

    def future_independence():
        frame = _certification_frame(); values = actual(frame)
        future = pd.concat([frame, pd.DataFrame({"timestamp": [99 * _DAY_SECONDS], "customer_id": ["a"], "fraud": [1]})], ignore_index=True)
        pd.testing.assert_frame_equal(values, actual(future).iloc[:len(frame)].reset_index(drop=True))

    def label_independence():
        frame = _certification_frame(); changed = frame.assign(fraud=1 - frame.fraud)
        pd.testing.assert_frame_equal(actual(frame), actual(changed))

    def chunk_vs_whole():
        frame = _certification_frame()
        pd.testing.assert_frame_equal(actual(frame), customer_temporal_habits_chunked([frame.iloc[:3], frame.iloc[3:7], frame.iloc[7:]]))

    def raw_frame_compatibility():
        frame = _certification_frame().drop(columns="fraud")
        result = build_features(frame, list(AVAILABLE_FEATURES))
        if not result.index.equals(frame.index) or list(result.columns) != list(AVAILABLE_FEATURES):
            raise AssertionError("raw-frame adapter changed requested order or alignment")

    return {"oracle": oracle, "strict_past": strict_past, "equal_timestamp_isolation": equal_timestamp_isolation,
            "permutation_invariance": permutation_invariance, "duplicate_same_timestamp_pair": duplicate_same_timestamp_pair,
            "future_independence": future_independence, "label_independence": label_independence,
            "chunk_vs_whole": chunk_vs_whole, "raw_frame_compatibility": raw_frame_compatibility}
