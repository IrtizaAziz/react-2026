"""Strictly-prior timestamp-batched device--location familiarity for R024."""
import numpy as np
import pandas as pd

from .customer_history import _timestamp_ns
from .customer_relationships import _pair_codes, _values, device_global_production


FEATURES = [
    "device_location_prior_count", "device_location_is_new",
    "device_location_seconds_since_last", "device_location_share_of_device_history",
]
AVAILABLE_FEATURES = tuple(FEATURES)
SOURCE_DEPENDENCIES = (
    __file__.replace("device_location.py", "customer_relationships.py"),
    __file__.replace("device_location.py", "customer_history.py"),
)


def _input_arrays(timestamp, device_id, location, device_prior_count):
    times = _timestamp_ns(timestamp)
    devices = _values(device_id, "__MISSING_DEVICE__")
    # Deliberately match customer-location bookkeeping exactly.
    locations = _values(location, "__MISSING_LOCATION__")
    prior_devices = pd.Series(device_prior_count).to_numpy(dtype=np.int64)
    if (prior_devices < 0).any():
        raise ValueError("device_prior_count must be nonnegative")
    return times, devices, locations, prior_devices


def _empty(n):
    return {
        "device_location_prior_count": np.zeros(n, dtype=np.int32),
        "device_location_is_new": np.ones(n, dtype=np.int8),
        "device_location_seconds_since_last": np.full(n, np.nan, dtype=np.float32),
        "device_location_share_of_device_history": np.full(n, np.nan, dtype=np.float32),
    }


def device_location_production(timestamp, device_id, location, device_prior_count):
    """Optimized engine: every tied batch queries frozen state, then updates it."""
    times, devices, locations, prior_devices = _input_arrays(timestamp, device_id, location, device_prior_count)
    device_codes, _ = pd.factorize(devices, sort=False)
    location_codes, _ = pd.factorize(locations, sort=False)
    pair_codes = _pair_codes(device_codes, location_codes)
    counts = np.zeros(int(pair_codes.max()) + 1 if len(pair_codes) else 0, dtype=np.int64)
    last = np.full(len(counts), -1, dtype=np.int64)
    out = _empty(len(times))
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        now, pairs = times[start], pair_codes[start:end]
        prior = counts[pairs]
        seen = prior > 0
        out["device_location_prior_count"][start:end] = prior
        out["device_location_is_new"][start:end] = (~seen).astype(np.int8)
        out["device_location_seconds_since_last"][start:end] = np.where(seen, (now - last[pairs]) / 1_000_000_000, np.nan)
        out["device_location_share_of_device_history"][start:end] = np.divide(
            prior, prior_devices[start:end], out=np.full(end - start, np.nan, dtype=np.float32), where=prior_devices[start:end] > 0)
        unique, inverse = np.unique(pairs, return_inverse=True)
        counts[unique] += np.bincount(inverse, minlength=len(unique))
        last[unique] = now
    return pd.DataFrame(out)


def device_location_simple_oracle(timestamp, device_id, location, device_prior_count):
    """Small reference engine for causal tests; it intentionally accepts no labels."""
    times, devices, locations, prior_devices = _input_arrays(timestamp, device_id, location, device_prior_count)
    out, state, start = _empty(len(times)), {}, 0
    while start < len(times):
        end = start + 1
        while end < len(times) and times[end] == times[start]:
            end += 1
        now = times[start]
        for i in range(start, end):
            count, previous = state.get((devices[i], locations[i]), (0, None))
            out["device_location_prior_count"][i] = count
            out["device_location_is_new"][i] = int(count == 0)
            if count:
                out["device_location_seconds_since_last"][i] = (now - previous) / 1_000_000_000
            if prior_devices[i]:
                out["device_location_share_of_device_history"][i] = count / prior_devices[i]
        for i in range(start, end):
            key = (devices[i], locations[i]); count, _ = state.get(key, (0, None)); state[key] = (count + 1, now)
        start = end
    return pd.DataFrame(out)


def device_location_chunked(chunks):
    """Chunk-safe equivalent: preserve the final complete tied batch."""
    full = pd.concat([chunk.copy() for chunk in chunks])
    return device_location_production(full.timestamp, full.device_id, full.location, full.device_prior_count).set_axis(full.index).sort_index()


def build_features(raw_frame, requested_features):
    """Return a requested R024 subset through the unchanged production engines."""
    try:
        requested = tuple(requested_features) if not isinstance(requested_features, str) else ()
    except TypeError:
        requested = ()
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(AVAILABLE_FEATURES):
        raise ValueError("requested_features must be a non-empty ordered subset of AVAILABLE_FEATURES")
    required = ("timestamp", "customer_id", "device_id", "location")
    missing = [column for column in required if column not in raw_frame]
    if missing: raise ValueError(f"raw_frame is missing required columns: {missing}")
    # R024's denominator was the pre-existing R005 device global history.
    device_history = device_global_production(raw_frame["timestamp"], raw_frame["customer_id"], raw_frame["device_id"])
    values = device_location_production(raw_frame["timestamp"], raw_frame["device_id"], raw_frame["location"],
                                        device_history["device_prior_count"])
    return values.loc[:, list(requested)].set_axis(raw_frame.index)


def _certification_frame():
    return pd.DataFrame({"transaction_id": list("abcdefgh"), "timestamp": [0, 1, 2, 2, 3, 4, 5, 6],
                         "customer_id": ["u", "u", "v", "w", "u", "x", "u", "u"],
                         "device_id": ["d", "d", "d", "d", "d", "x", "d", "d"],
                         "location": [None, None, "a", "a", None, "a", "a", "b"],
                         "fraud": [0, 1, 0, 1, 0, 1, 0, 1]})


def certification_cases():
    """Expose the already-tested R024 causal semantics to generic certification."""
    def upstream(frame):
        return device_global_production(frame.timestamp, frame.customer_id, frame.device_id).device_prior_count

    def actual(frame):
        return device_location_production(frame.timestamp, frame.device_id, frame.location, upstream(frame))

    def strict_past():
        values = actual(_certification_frame())
        if values.device_location_prior_count.tolist() != [0, 1, 0, 0, 2, 0, 2, 0]:
            raise AssertionError("strictly-prior R024 values changed")

    def oracle():
        frame = _certification_frame(); prior = upstream(frame)
        pd.testing.assert_frame_equal(actual(frame), device_location_simple_oracle(frame.timestamp, frame.device_id, frame.location, prior))

    def equal_timestamp_isolation():
        values = actual(_certification_frame())
        if not values.iloc[2].equals(values.iloc[3]): raise AssertionError("tied rows influenced one another")

    def permutation_invariance():
        frame = _certification_frame(); values = actual(frame)
        permuted = pd.concat([frame.iloc[:2], frame.iloc[[3, 2]], frame.iloc[4:]], ignore_index=True)
        observed = actual(permuted)
        pd.testing.assert_frame_equal(pd.concat([frame.transaction_id, values], axis=1).set_index("transaction_id").sort_index(),
                                      pd.concat([permuted.transaction_id, observed], axis=1).set_index("transaction_id").sort_index())

    def duplicate_same_timestamp_pair():
        frame = _certification_frame(); duplicated = pd.concat([frame.iloc[:2], frame.iloc[[2, 2]], frame.iloc[4:]], ignore_index=True)
        values = actual(duplicated)
        if not values.iloc[2].equals(values.iloc[3]): raise AssertionError("duplicate tied pair lost isolation")

    def future_independence():
        frame = _certification_frame(); expected = actual(frame)
        future = pd.concat([frame, pd.DataFrame({"transaction_id": ["z"], "timestamp": [99], "customer_id": ["future"], "device_id": ["future"], "location": ["future"], "fraud": [1]})], ignore_index=True)
        pd.testing.assert_frame_equal(expected, actual(future).iloc[:len(frame)].reset_index(drop=True))
        future.loc[len(frame), ["customer_id", "device_id", "location"]] = ["u", "d", "a"]
        pd.testing.assert_frame_equal(expected, actual(future).iloc[:len(frame)].reset_index(drop=True))

    def label_independence():
        frame = _certification_frame(); changed = frame.copy(); changed["fraud"] = 1 - changed["fraud"]
        pd.testing.assert_frame_equal(actual(frame), actual(changed))

    def chunk_vs_whole():
        frame = _certification_frame(); prior = upstream(frame)
        chunked_frame = frame.assign(device_prior_count=prior)
        pd.testing.assert_frame_equal(actual(frame).set_axis(frame.index).sort_index(), device_location_chunked([chunked_frame.iloc[:2], chunked_frame.iloc[2:4], chunked_frame.iloc[4:]]))

    def missing_location_sentinel():
        frame = pd.DataFrame({"timestamp": [0, 1], "customer_id": ["u", "u"], "device_id": ["d", "d"], "location": [None, None]})
        if build_features(frame, ["device_location_prior_count"]).iloc[:, 0].tolist() != [0, 1]:
            raise AssertionError("missing location sentinel changed")

    return {"strict_past": strict_past, "oracle": oracle, "equal_timestamp_isolation": equal_timestamp_isolation,
            "permutation_invariance": permutation_invariance, "duplicate_same_timestamp_pair": duplicate_same_timestamp_pair,
            "future_independence": future_independence, "label_independence": label_independence,
            "chunk_vs_whole": chunk_vs_whole, "missing_location_sentinel": missing_location_sentinel}
