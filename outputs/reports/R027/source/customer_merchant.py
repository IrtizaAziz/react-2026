"""Strictly-prior timestamp-batched customer--merchant familiarity features."""
import numpy as np
import pandas as pd

from .customer_history import _timestamp_ns
from .customer_relationships import _pair_codes, _values


FEATURES = ["customer_merchant_prior_count", "customer_merchant_is_new",
            "customer_merchant_seconds_since_last", "customer_merchant_share_of_customer_history"]


def _input_arrays(timestamp, customer_id, merchant_id, customer_prior_count):
    times = _timestamp_ns(timestamp)
    customers = _values(customer_id, "__MISSING_CUSTOMER__")
    merchants = _values(merchant_id, "__MISSING_MERCHANT__")
    prior_customers = pd.Series(customer_prior_count).to_numpy(dtype=np.int64)
    if (prior_customers < 0).any():
        raise ValueError("customer_prior_count must be nonnegative")
    return times, customers, merchants, prior_customers


def _empty(n):
    return {"customer_merchant_prior_count": np.zeros(n, dtype=np.int32),
            "customer_merchant_is_new": np.ones(n, dtype=np.int8),
            "customer_merchant_seconds_since_last": np.full(n, np.nan, dtype=np.float32),
            "customer_merchant_share_of_customer_history": np.zeros(n, dtype=np.float32)}


def customer_merchant_production(timestamp, customer_id, merchant_id, customer_prior_count):
    """Production state engine; accepts only raw identities and R007's prior count."""
    times, customers, merchants, prior_customers = _input_arrays(timestamp, customer_id, merchant_id, customer_prior_count)
    customer_codes, _ = pd.factorize(customers, sort=False)
    merchant_codes, _ = pd.factorize(merchants, sort=False)
    pair_codes = _pair_codes(customer_codes, merchant_codes)
    pair_count = np.zeros(int(pair_codes.max()) + 1 if len(pair_codes) else 0, dtype=np.int64)
    pair_last = np.full(len(pair_count), -1, dtype=np.int64)
    out = _empty(len(times))
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        current, pairs = times[start], pair_codes[start:end]
        prior_pair = pair_count[pairs]
        seen = prior_pair > 0
        out["customer_merchant_prior_count"][start:end] = prior_pair
        out["customer_merchant_is_new"][start:end] = (~seen).astype(np.int8)
        out["customer_merchant_seconds_since_last"][start:end] = np.where(
            seen, (current - pair_last[pairs]) / 1_000_000_000, np.nan
        )
        # This denominator is intentionally supplied by immutable R007 customer history.
        out["customer_merchant_share_of_customer_history"][start:end] = prior_pair / np.maximum(prior_customers[start:end], 1)
        unique_pairs, inverse = np.unique(pairs, return_inverse=True)
        pair_count[unique_pairs] += np.bincount(inverse, minlength=len(unique_pairs))
        pair_last[unique_pairs] = current
    return pd.DataFrame(out)


def customer_merchant_chunked(chunks):
    """Reference streaming implementation retaining a final complete timestamp batch."""
    pending, outputs = pd.DataFrame(), []
    state_counts, state_last = {}, {}
    for chunk in chunks:
        combined = pd.concat([pending, chunk.copy()])
        if not len(combined):
            continue
        times = _timestamp_ns(combined["timestamp"])
        ready_end = np.searchsorted(times, times[-1], side="left")
        if ready_end:
            ready = combined.iloc[:ready_end]
            outputs.append(_chunk_with_state(ready, state_counts, state_last).set_axis(ready.index))
        pending = combined.iloc[ready_end:]
    if len(pending):
        outputs.append(_chunk_with_state(pending, state_counts, state_last).set_axis(pending.index))
    return pd.concat(outputs).sort_index() if outputs else pd.DataFrame(columns=FEATURES)


def _chunk_with_state(frame, counts, last):
    times, customers, merchants, prior_customers = _input_arrays(frame["timestamp"], frame["customer_id"], frame["merchant_id"], frame["customer_prior_count"])
    out, start = _empty(len(frame)), 0
    while start < len(frame):
        end = start + 1
        while end < len(frame) and times[end] == times[start]:
            end += 1
        for i in range(start, end):
            key, count = (customers[i], merchants[i]), counts.get((customers[i], merchants[i]), 0)
            out["customer_merchant_prior_count"][i] = count
            out["customer_merchant_is_new"][i] = int(count == 0)
            out["customer_merchant_share_of_customer_history"][i] = count / max(prior_customers[i], 1)
            if count:
                out["customer_merchant_seconds_since_last"][i] = (times[i] - last[key]) / 1_000_000_000
        for i in range(start, end):
            key = (customers[i], merchants[i]); counts[key] = counts.get(key, 0) + 1; last[key] = times[i]
        start = end
    return pd.DataFrame(out)
