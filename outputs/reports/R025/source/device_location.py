"""Strictly-prior timestamp-batched device--location familiarity for R024."""
import numpy as np
import pandas as pd

from .customer_history import _timestamp_ns
from .customer_relationships import _pair_codes, _values


FEATURES = [
    "device_location_prior_count", "device_location_is_new",
    "device_location_seconds_since_last", "device_location_share_of_device_history",
]


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
