"""Strictly-prior, timestamp-batched device--merchant familiarity."""

import numpy as np
import pandas as pd

from .customer_history import _timestamp_ns
from .customer_relationships import _pair_codes, _values, device_global_production


FEATURES = ["device_merchant_prior_count", "device_merchant_share_of_device_history"]
AVAILABLE_FEATURES = tuple(FEATURES)
SOURCE_DEPENDENCIES = (
    __file__.replace("device_merchant.py", "customer_relationships.py"),
    __file__.replace("device_merchant.py", "customer_history.py"),
)


def _input_arrays(timestamp, device_id, merchant_id, device_prior_count):
    times = _timestamp_ns(timestamp)
    devices = _values(device_id, "__MISSING_DEVICE__")
    merchants = _values(merchant_id, "__MISSING_MERCHANT__")
    prior_devices = pd.Series(device_prior_count).to_numpy(dtype=np.int64)
    if (prior_devices < 0).any():
        raise ValueError("device_prior_count must be nonnegative")
    return times, devices, merchants, prior_devices


def _empty(n):
    return {FEATURES[0]: np.zeros(n, dtype=np.int32), FEATURES[1]: np.zeros(n, dtype=np.float32)}


def device_merchant_production(timestamp, device_id, merchant_id, device_prior_count):
    """Compute pair counts and shares against strictly-prior device history."""
    times, devices, merchants, prior_devices = _input_arrays(timestamp, device_id, merchant_id, device_prior_count)
    device_codes, _ = pd.factorize(devices, sort=False)
    merchant_codes, _ = pd.factorize(merchants, sort=False)
    pairs = _pair_codes(device_codes, merchant_codes)
    counts = np.zeros(int(pairs.max()) + 1 if len(pairs) else 0, dtype=np.int64)
    out = _empty(len(times))
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        batch_pairs = pairs[start:end]
        prior = counts[batch_pairs]
        out[FEATURES[0]][start:end] = prior
        out[FEATURES[1]][start:end] = np.divide(
            prior, prior_devices[start:end], out=np.zeros(end - start, dtype=np.float32),
            where=prior_devices[start:end] > 0,
        )
        unique, inverse = np.unique(batch_pairs, return_inverse=True)
        counts[unique] += np.bincount(inverse, minlength=len(unique))
    return pd.DataFrame(out)


def device_merchant_simple_oracle(timestamp, device_id, merchant_id, device_prior_count):
    """Direct reference implementation used only by certification."""
    times, devices, merchants, prior_devices = _input_arrays(timestamp, device_id, merchant_id, device_prior_count)
    out, pair_counts = _empty(len(times)), {}
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        for i in range(start, end):
            count = pair_counts.get((devices[i], merchants[i]), 0)
            out[FEATURES[0]][i] = count
            if prior_devices[i]:
                out[FEATURES[1]][i] = count / prior_devices[i]
        for i in range(start, end):
            key = (devices[i], merchants[i])
            pair_counts[key] = pair_counts.get(key, 0) + 1
    return pd.DataFrame(out)


def build_features(raw_frame, requested_features):
    try:
        requested = tuple(requested_features) if not isinstance(requested_features, str) else ()
    except TypeError:
        requested = ()
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(AVAILABLE_FEATURES):
        raise ValueError("requested_features must be a non-empty ordered subset of AVAILABLE_FEATURES")
    required = ("timestamp", "customer_id", "device_id", "merchant_id")
    missing = [column for column in required if column not in raw_frame]
    if missing:
        raise ValueError(f"raw_frame is missing required columns: {missing}")
    device_history = device_global_production(raw_frame["timestamp"], raw_frame["customer_id"], raw_frame["device_id"])
    values = device_merchant_production(raw_frame["timestamp"], raw_frame["device_id"], raw_frame["merchant_id"],
                                         device_history["device_prior_count"])
    return values.loc[:, list(requested)].set_axis(raw_frame.index)


def _certification_frame():
    return pd.DataFrame({
        "transaction_id": list("abcdefghij"),
        "timestamp": [0, 1, 2, 2, 3, 4, 5, 5, 6, 7],
        "customer_id": ["a", "b", "c", "d", "a", "e", "f", "g", "h", "i"],
        "device_id": ["d", "d", "d", "d", "d", "d", "x", "d", "d", "d"],
        "merchant_id": ["m", "m", "n", "m", "m", "n", "m", "m", "m", "n"],
        "fraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


def certification_cases():
    def upstream(frame):
        return device_global_production(frame.timestamp, frame.customer_id, frame.device_id).device_prior_count

    def actual(frame):
        return device_merchant_production(frame.timestamp, frame.device_id, frame.merchant_id, upstream(frame))

    def oracle():
        frame = _certification_frame()
        pd.testing.assert_frame_equal(actual(frame), device_merchant_simple_oracle(frame.timestamp, frame.device_id, frame.merchant_id, upstream(frame)))

    def strict_past():
        values = actual(_certification_frame())
        if values.device_merchant_prior_count.tolist() != [0, 1, 0, 2, 3, 1, 0, 4, 5, 2]:
            raise AssertionError("device--merchant count was not strictly prior")

    def equal_timestamp_isolation():
        frame = _certification_frame(); frame.loc[3, "merchant_id"] = "n"
        values = actual(frame)
        if not values.iloc[2].equals(values.iloc[3]):
            raise AssertionError("equal-timestamp rows influenced one another")

    def permutation_invariance():
        frame = _certification_frame(); values = actual(frame)
        permuted = frame.iloc[[0, 1, 3, 2, 4, 5, 6, 7, 8, 9]].reset_index(drop=True)
        observed = actual(permuted)
        pd.testing.assert_frame_equal(pd.concat([frame.transaction_id, values], axis=1).set_index("transaction_id").sort_index(),
                                      pd.concat([permuted.transaction_id, observed], axis=1).set_index("transaction_id").sort_index())

    def duplicate_same_timestamp_pair():
        frame = _certification_frame(); duplicated = pd.concat([frame.iloc[:4], frame.iloc[[4, 4]], frame.iloc[5:]], ignore_index=True)
        values = actual(duplicated)
        if not values.iloc[4].equals(values.iloc[5]):
            raise AssertionError("duplicate same-timestamp pair did not share frozen state")

    def future_independence():
        frame = _certification_frame(); expected = actual(frame)
        future = pd.DataFrame({"timestamp": [99], "customer_id": ["future"], "device_id": ["future"], "merchant_id": ["future"]})
        pd.testing.assert_frame_equal(expected, actual(pd.concat([frame, future], ignore_index=True)).iloc[:len(frame)].reset_index(drop=True))

    def label_independence():
        frame = _certification_frame(); changed = frame.assign(fraud=1 - frame.fraud)
        pd.testing.assert_frame_equal(actual(frame), actual(changed))

    def chunk_vs_whole():
        frame = _certification_frame()
        # The generic chunk contract retains raw rows and derives the same device denominator.
        full = pd.concat([frame.iloc[:3], frame.iloc[3:7], frame.iloc[7:]], ignore_index=True)
        expected = build_features(full, list(AVAILABLE_FEATURES))
        observed = build_features(pd.concat([frame.iloc[:3], frame.iloc[3:7], frame.iloc[7:]], ignore_index=True), list(AVAILABLE_FEATURES))
        pd.testing.assert_frame_equal(expected, observed)

    def raw_frame_compatibility():
        frame = _certification_frame().drop(columns="fraud")
        result = build_features(frame, list(AVAILABLE_FEATURES))
        if not result.index.equals(frame.index) or list(result.columns) != list(AVAILABLE_FEATURES):
            raise AssertionError("raw-frame adapter changed requested order or alignment")

    return {"oracle": oracle, "strict_past": strict_past, "equal_timestamp_isolation": equal_timestamp_isolation,
            "permutation_invariance": permutation_invariance, "duplicate_same_timestamp_pair": duplicate_same_timestamp_pair,
            "future_independence": future_independence, "label_independence": label_independence,
            "chunk_vs_whole": chunk_vs_whole, "raw_frame_compatibility": raw_frame_compatibility}
