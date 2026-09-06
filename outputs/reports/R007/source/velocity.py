"""Strictly-prior one-hour customer and device transaction velocity."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .customer_history import _timestamp_ns


FEATURES = ["customer_prior_1h_count", "device_prior_1h_count"]
WINDOW_NS = np.int64(3_600_000_000_000)


def _input_arrays(timestamp, customer_id, device_id):
    times = _timestamp_ns(timestamp)
    customers = pd.Series(customer_id).astype("string").fillna("__MISSING_CUSTOMER__").to_numpy(dtype=str)
    devices = pd.Series(device_id).astype("string").fillna("__MISSING_DEVICE__").to_numpy(dtype=str)
    return times, customers, devices


def _empty(n):
    return {name: np.zeros(n, dtype=np.int64) for name in FEATURES}


def _evict(history, current):
    # [t - 1 hour, t): equality at the left boundary is retained.
    cutoff = current - WINDOW_NS
    while history and history[0] < cutoff:
        history.popleft()


def _velocity_with_state(timestamp, customer_id, device_id, state):
    times, customers, devices = _input_arrays(timestamp, customer_id, device_id)
    customer_state, device_state = state
    out = _empty(len(times))
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        current = times[start]
        # Query every row in the timestamp batch against frozen pre-t state.
        for key, histories, name in ((customers, customer_state, "customer_prior_1h_count"),
                                     (devices, device_state, "device_prior_1h_count")):
            for entity in np.unique(key[start:end]):
                _evict(histories[entity], current)
            out[name][start:end] = [len(histories[entity]) for entity in key[start:end]]
        # Only then allow the complete batch to enter subsequent histories.
        for entity in customers[start:end]:
            customer_state[entity].append(current)
        for entity in devices[start:end]:
            device_state[entity].append(current)
    return pd.DataFrame(out)


def velocity_oracle(timestamp, customer_id, device_id):
    """Deliberately simple reference for test-only semantic comparison."""
    times, customers, devices = _input_arrays(timestamp, customer_id, device_id)
    out = _empty(len(times))
    for i, current in enumerate(times):
        prior = times[:i] < current
        in_window = prior & (times[:i] >= current - WINDOW_NS)
        out["customer_prior_1h_count"][i] = int(np.sum(in_window & (customers[:i] == customers[i])))
        out["device_prior_1h_count"][i] = int(np.sum(in_window & (devices[:i] == devices[i])))
    return pd.DataFrame(out)


def velocity_production(timestamp, customer_id, device_id):
    """Timestamp-batched rolling implementation used by R007."""
    return _velocity_with_state(timestamp, customer_id, device_id, (defaultdict(deque), defaultdict(deque)))


def velocity_from_prior_stream(prior_timestamp, prior_customer_id, prior_device_id,
                               current_timestamp, current_customer_id, current_device_id):
    """Continue an unlabeled chronological stream after strictly earlier raw rows."""
    prior_times = _timestamp_ns(prior_timestamp)
    current_times = _timestamp_ns(current_timestamp)
    if len(prior_times) and len(current_times) and prior_times[-1] >= current_times[0]:
        raise ValueError("Prior velocity stream must end strictly before the current stream")
    state = (defaultdict(deque), defaultdict(deque))
    _velocity_with_state(prior_timestamp, prior_customer_id, prior_device_id, state)
    return _velocity_with_state(current_timestamp, current_customer_id, current_device_id, state)


def velocity_chunked(chunks):
    """Chunked stream processor retaining a final tied batch until complete."""
    state, pending, outputs = (defaultdict(deque), defaultdict(deque)), pd.DataFrame(), []
    for chunk in chunks:
        combined = pd.concat([pending, chunk.copy()])
        if not len(combined):
            continue
        times = _timestamp_ns(combined["timestamp"])
        ready_end = np.searchsorted(times, times[-1], side="left")
        if ready_end:
            ready = combined.iloc[:ready_end]
            outputs.append(_velocity_with_state(ready["timestamp"], ready["customer_id"], ready["device_id"], state).set_axis(ready.index))
        pending = combined.iloc[ready_end:]
    if len(pending):
        outputs.append(_velocity_with_state(pending["timestamp"], pending["customer_id"], pending["device_id"], state).set_axis(pending.index))
    return pd.concat(outputs).sort_index() if outputs else pd.DataFrame(columns=FEATURES)
