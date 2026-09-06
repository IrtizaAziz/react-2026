"""Strictly-past timestamp-batched customer-device and customer-location state."""
import numpy as np
import pandas as pd

from .customer_history import _timestamp_ns


FEATURES = [
    "customer_device_prior_count", "customer_device_new", "customer_device_seconds_since_last", "customer_device_count_share",
    "customer_location_prior_count", "customer_location_new", "customer_location_seconds_since_last", "customer_location_count_share",
]

DEVICE_GLOBAL_FEATURES = [
    "device_prior_count", "device_prior_distinct_customer_count",
    "device_prior_distinct_customer_count_excluding_current", "device_seconds_since_last",
    "device_observed_age_seconds", "device_prior_other_customer_transaction_count",
    "customer_share_of_device_prior_transactions",
]


def _values(values, missing):
    return pd.Series(values).astype("string").fillna(missing).to_numpy(dtype=str)


def _input_arrays(timestamp, customer_id, device_id, location):
    return (_timestamp_ns(timestamp), _values(customer_id, "__MISSING_CUSTOMER__"),
            _values(device_id, "__MISSING_DEVICE__"), _values(location, "__MISSING_LOCATION__"))


def _empty(n):
    result = {name: np.full(n, np.nan, dtype=np.float32) for name in FEATURES}
    for name in ("customer_device_prior_count", "customer_device_new", "customer_location_prior_count", "customer_location_new"):
        result[name] = np.zeros(n, dtype=np.int32 if name.endswith("count") else np.int8)
    return result


def _oracle_with_state(timestamp, customer_id, device_id, location, state):
    times, customers, devices, locations = _input_arrays(timestamp, customer_id, device_id, location)
    out, start = _empty(len(times)), 0
    customer_counts = state.setdefault("customer_counts", {})
    device_pairs = state.setdefault("device_pairs", {})
    location_pairs = state.setdefault("location_pairs", {})
    while start < len(times):
        end = start + 1
        while end < len(times) and times[end] == times[start]:
            end += 1
        for i in range(start, end):
            customer_count = customer_counts.get(customers[i], 0)
            for prefix, pairs, value in (("customer_device", device_pairs, devices[i]),
                                         ("customer_location", location_pairs, locations[i])):
                count, last = pairs.get((customers[i], value), (0, None))
                out[f"{prefix}_prior_count"][i] = count
                out[f"{prefix}_new"][i] = int(count == 0)
                if count:
                    out[f"{prefix}_seconds_since_last"][i] = (times[i] - last) / 1_000_000_000
                if customer_count:
                    out[f"{prefix}_count_share"][i] = count / customer_count
        for i in range(start, end):
            customer_counts[customers[i]] = customer_counts.get(customers[i], 0) + 1
            for pairs, value in ((device_pairs, devices[i]), (location_pairs, locations[i])):
                key = (customers[i], value)
                count, _ = pairs.get(key, (0, None))
                pairs[key] = (count + 1, times[i])
        start = end
    return pd.DataFrame(out)


def customer_relationship_oracle(timestamp, customer_id, device_id, location):
    """Reference implementation. It deliberately accepts no labels or arbitrary frames."""
    return _oracle_with_state(timestamp, customer_id, device_id, location, {})


def _pair_codes(left, right):
    """Exact dense codes for integer pairs without materializing an object MultiIndex."""
    order = np.lexsort((right, left))
    ordered_left, ordered_right = left[order], right[order]
    starts = np.r_[True, (ordered_left[1:] != ordered_left[:-1]) | (ordered_right[1:] != ordered_right[:-1])]
    result = np.empty(len(left), dtype=np.int64)
    result[order] = np.cumsum(starts, dtype=np.int64) - 1
    return result


def customer_relationship_production(timestamp, customer_id, device_id, location):
    """Vectorized timestamp-batch implementation for the eight approved pair features."""
    times, customers, devices, locations = _input_arrays(timestamp, customer_id, device_id, location)
    customer_codes, customer_values = pd.factorize(customers, sort=False)
    device_values, _ = pd.factorize(devices, sort=False)
    location_values, _ = pd.factorize(locations, sort=False)
    device_codes = _pair_codes(customer_codes, device_values)
    location_codes = _pair_codes(customer_codes, location_values)
    n = len(times)
    customer_count = np.zeros(len(customer_values), dtype=np.int64)
    device_count, location_count = np.zeros(device_codes.max() + 1, dtype=np.int64), np.zeros(location_codes.max() + 1, dtype=np.int64)
    device_last, location_last = np.full(len(device_count), -1, dtype=np.int64), np.full(len(location_count), -1, dtype=np.int64)
    out = _empty(n)
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    for start, end in zip(starts, ends):
        current = times[start]
        c, d, l = customer_codes[start:end], device_codes[start:end], location_codes[start:end]
        prior_customer = customer_count[c]
        for prefix, pair_codes, pair_count, pair_last in (("customer_device", d, device_count, device_last),
                                                          ("customer_location", l, location_count, location_last)):
            prior_pair = pair_count[pair_codes]
            seen = prior_pair > 0
            out[f"{prefix}_prior_count"][start:end] = prior_pair
            out[f"{prefix}_new"][start:end] = (~seen).astype(np.int8)
            out[f"{prefix}_seconds_since_last"][start:end] = np.where(seen, (current - pair_last[pair_codes]) / 1_000_000_000, np.nan)
            out[f"{prefix}_count_share"][start:end] = np.divide(prior_pair, prior_customer,
                                                                   out=np.full(end - start, np.nan, dtype=np.float32), where=prior_customer > 0)
            unique, inverse = np.unique(pair_codes, return_inverse=True)
            pair_count[unique] += np.bincount(inverse, minlength=len(unique))
            pair_last[unique] = current
        unique_customer, inverse_customer = np.unique(c, return_inverse=True)
        customer_count[unique_customer] += np.bincount(inverse_customer, minlength=len(unique_customer))
    return pd.DataFrame(out)


def customer_relationship_from_prior_stream(prior_timestamp, prior_customer_id, prior_device_id, prior_location,
                                            current_timestamp, current_customer_id, current_device_id, current_location):
    state = {}
    _oracle_with_state(prior_timestamp, prior_customer_id, prior_device_id, prior_location, state)
    return _oracle_with_state(current_timestamp, current_customer_id, current_device_id, current_location, state)


def customer_relationship_chunked(chunks):
    """Streaming reference that defers each chunk's final tied timestamp batch."""
    state, pending, outputs = {}, pd.DataFrame(), []
    for chunk in chunks:
        combined = pd.concat([pending, chunk.copy()])
        if not len(combined):
            continue
        times = _timestamp_ns(combined["timestamp"])
        ready_end = np.searchsorted(times, times[-1], side="left")
        if ready_end:
            ready = combined.iloc[:ready_end]
            outputs.append(_oracle_with_state(ready["timestamp"], ready["customer_id"], ready["device_id"], ready["location"], state).set_axis(ready.index))
        pending = combined.iloc[ready_end:]
    if len(pending):
        outputs.append(_oracle_with_state(pending["timestamp"], pending["customer_id"], pending["device_id"], pending["location"], state).set_axis(pending.index))
    return pd.concat(outputs).sort_index() if outputs else pd.DataFrame(columns=FEATURES)


def _device_global_empty(n):
    out = {name: np.full(n, np.nan, dtype=np.float32) for name in DEVICE_GLOBAL_FEATURES}
    for name in ("device_prior_count", "device_prior_distinct_customer_count",
                 "device_prior_distinct_customer_count_excluding_current",
                 "device_prior_other_customer_transaction_count"):
        out[name] = np.zeros(n, dtype=np.int32)
    return out


def _device_global_oracle_with_state(timestamp, customer_id, device_id, state):
    """Small strictly-past oracle; state contains no labels or arbitrary frames."""
    times, customers, devices, _ = _input_arrays(timestamp, customer_id, device_id, [""] * len(timestamp))
    out, start = _device_global_empty(len(times)), 0
    device_state = state.setdefault("devices", {})
    pair_counts = state.setdefault("device_customer_counts", {})
    while start < len(times):
        end = start + 1
        while end < len(times) and times[end] == times[start]:
            end += 1
        for i in range(start, end):
            count, customers_seen, first, last = device_state.get(devices[i], (0, set(), None, None))
            pair_count = pair_counts.get((customers[i], devices[i]), 0)
            out["device_prior_count"][i] = count
            out["device_prior_distinct_customer_count"][i] = len(customers_seen)
            out["device_prior_distinct_customer_count_excluding_current"][i] = len(customers_seen) - int(pair_count > 0)
            out["device_prior_other_customer_transaction_count"][i] = count - pair_count
            if count:
                out["device_seconds_since_last"][i] = (times[i] - last) / 1_000_000_000
                out["device_observed_age_seconds"][i] = (times[i] - first) / 1_000_000_000
                out["customer_share_of_device_prior_transactions"][i] = pair_count / count
        for i in range(start, end):
            count, customers_seen, first, _ = device_state.get(devices[i], (0, set(), times[i], None))
            customers_seen = set(customers_seen)
            customers_seen.add(customers[i])
            device_state[devices[i]] = (count + 1, customers_seen, first, times[i])
            key = (customers[i], devices[i])
            pair_counts[key] = pair_counts.get(key, 0) + 1
        start = end
    return pd.DataFrame(out)


def device_global_oracle(timestamp, customer_id, device_id):
    return _device_global_oracle_with_state(timestamp, customer_id, device_id, {})


def device_global_production(timestamp, customer_id, device_id):
    """Vectorized timestamp-batched device support/sharing state for R005."""
    times, customers, devices, _ = _input_arrays(timestamp, customer_id, device_id, [""] * len(timestamp))
    customer_codes, _ = pd.factorize(customers, sort=False)
    device_codes, device_values = pd.factorize(devices, sort=False)
    pair_codes = _pair_codes(customer_codes, device_codes)
    n_devices, n_pairs, n = len(device_values), int(pair_codes.max()) + 1 if len(pair_codes) else 0, len(times)
    device_count = np.zeros(n_devices, dtype=np.int64)
    device_distinct = np.zeros(n_devices, dtype=np.int64)
    device_first = np.full(n_devices, -1, dtype=np.int64)
    device_last = np.full(n_devices, -1, dtype=np.int64)
    pair_count = np.zeros(n_pairs, dtype=np.int64)
    pair_device = np.empty(n_pairs, dtype=np.int64)
    pair_device[pair_codes] = device_codes
    out = _device_global_empty(n)
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    for start, end in zip(starts, ends):
        current, d, p = times[start], device_codes[start:end], pair_codes[start:end]
        prior_device, prior_pair = device_count[d], pair_count[p]
        seen = prior_device > 0
        out["device_prior_count"][start:end] = prior_device
        out["device_prior_distinct_customer_count"][start:end] = device_distinct[d]
        out["device_prior_distinct_customer_count_excluding_current"][start:end] = device_distinct[d] - (prior_pair > 0)
        out["device_prior_other_customer_transaction_count"][start:end] = prior_device - prior_pair
        out["device_seconds_since_last"][start:end] = np.where(seen, (current - device_last[d]) / 1_000_000_000, np.nan)
        out["device_observed_age_seconds"][start:end] = np.where(seen, (current - device_first[d]) / 1_000_000_000, np.nan)
        out["customer_share_of_device_prior_transactions"][start:end] = np.divide(
            prior_pair, prior_device, out=np.full(end - start, np.nan, dtype=np.float32), where=seen)
        unique_pairs, inverse = np.unique(p, return_inverse=True)
        new_pairs = pair_count[unique_pairs] == 0
        pair_count[unique_pairs] += np.bincount(inverse, minlength=len(unique_pairs))
        if new_pairs.any():
            device_distinct += np.bincount(pair_device[unique_pairs[new_pairs]], minlength=n_devices)
        unique_devices, inverse_device = np.unique(d, return_inverse=True)
        device_count[unique_devices] += np.bincount(inverse_device, minlength=len(unique_devices))
        device_first[unique_devices[device_first[unique_devices] < 0]] = current
        device_last[unique_devices] = current
    return pd.DataFrame(out)


def device_global_from_prior_stream(prior_timestamp, prior_customer_id, prior_device_id,
                                    current_timestamp, current_customer_id, current_device_id):
    state = {}
    _device_global_oracle_with_state(prior_timestamp, prior_customer_id, prior_device_id, state)
    return _device_global_oracle_with_state(current_timestamp, current_customer_id, current_device_id, state)


def device_global_chunked(chunks):
    state, pending, outputs = {}, pd.DataFrame(), []
    for chunk in chunks:
        combined = pd.concat([pending, chunk.copy()])
        if not len(combined):
            continue
        times = _timestamp_ns(combined["timestamp"])
        ready_end = np.searchsorted(times, times[-1], side="left")
        if ready_end:
            ready = combined.iloc[:ready_end]
            outputs.append(_device_global_oracle_with_state(ready["timestamp"], ready["customer_id"], ready["device_id"], state).set_axis(ready.index))
        pending = combined.iloc[ready_end:]
    if len(pending):
        outputs.append(_device_global_oracle_with_state(pending["timestamp"], pending["customer_id"], pending["device_id"], state).set_axis(pending.index))
    return pd.concat(outputs).sort_index() if outputs else pd.DataFrame(columns=DEVICE_GLOBAL_FEATURES)
