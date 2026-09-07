"""Strictly-prior timestamp-batched customer--merchant-category familiarity."""
import numpy as np
import pandas as pd
from .customer_history import _timestamp_ns
from .customer_relationships import _pair_codes, _values

FEATURES = ["customer_category_prior_count", "customer_category_is_new", "customer_category_seconds_since_last", "customer_category_share_of_customer_history"]

def _input_arrays(timestamp, customer_id, merchant_category, customer_prior_count):
    times = _timestamp_ns(timestamp)
    customers = _values(customer_id, "__MISSING_CUSTOMER__")
    # Matches the existing relationship-engine deterministic missing-key convention.
    categories = _values(merchant_category, "__MISSING_CATEGORY__")
    prior = pd.Series(customer_prior_count).to_numpy(dtype=np.int64)
    if (prior < 0).any(): raise ValueError("customer_prior_count must be nonnegative")
    return times, customers, categories, prior

def _empty(n):
    return {"customer_category_prior_count": np.zeros(n, dtype=np.int32),
            "customer_category_is_new": np.ones(n, dtype=np.int8),
            "customer_category_seconds_since_last": np.full(n, np.nan, dtype=np.float32),
            "customer_category_share_of_customer_history": np.zeros(n, dtype=np.float32)}

def customer_category_production(timestamp, customer_id, merchant_category, customer_prior_count):
    """Production causal engine; denominator is R013's inherited prior customer count."""
    times, customers, categories, prior_customers = _input_arrays(timestamp, customer_id, merchant_category, customer_prior_count)
    c, _ = pd.factorize(customers, sort=False); category, _ = pd.factorize(categories, sort=False)
    pairs = _pair_codes(c, category); count = np.zeros(int(pairs.max()) + 1 if len(pairs) else 0, dtype=np.int64)
    last, out = np.full(len(count), -1, dtype=np.int64), _empty(len(times))
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]; ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        current, batch = times[start], pairs[start:end]; old = count[batch]; seen = old > 0
        out["customer_category_prior_count"][start:end] = old
        out["customer_category_is_new"][start:end] = (~seen).astype(np.int8)
        out["customer_category_seconds_since_last"][start:end] = np.where(seen, (current - last[batch]) / 1_000_000_000, np.nan)
        out["customer_category_share_of_customer_history"][start:end] = old / np.maximum(prior_customers[start:end], 1)
        unique, inverse = np.unique(batch, return_inverse=True); count[unique] += np.bincount(inverse, minlength=len(unique)); last[unique] = current
    return pd.DataFrame(out)

def customer_category_from_prior_stream(prior_timestamp, prior_customer_id, prior_merchant_category, prior_customer_prior_count,
                                        current_timestamp, current_customer_id, current_merchant_category, current_customer_prior_count):
    """Causal continuation after a strictly earlier raw stream; labels are not accepted."""
    prior_times, current_times = _timestamp_ns(prior_timestamp), _timestamp_ns(current_timestamp)
    if len(prior_times) and len(current_times) and prior_times[-1] >= current_times[0]: raise ValueError("Prior category stream must end strictly before current stream")
    output = customer_category_production(pd.concat([pd.Series(prior_timestamp), pd.Series(current_timestamp)], ignore_index=True),
        pd.concat([pd.Series(prior_customer_id), pd.Series(current_customer_id)], ignore_index=True),
        pd.concat([pd.Series(prior_merchant_category), pd.Series(current_merchant_category)], ignore_index=True),
        pd.concat([pd.Series(prior_customer_prior_count), pd.Series(current_customer_prior_count)], ignore_index=True))
    return output.iloc[len(prior_times):].reset_index(drop=True)

def customer_category_chunked(chunks):
    """Chunked reference: complete tied timestamp batches before advancing state."""
    pending, output = pd.DataFrame(), []
    for chunk in chunks:
        combined = pd.concat([pending, chunk.copy()])
        if not len(combined): continue
        times = _timestamp_ns(combined["timestamp"]); ready = np.searchsorted(times, times[-1], side="left")
        if ready:
            # Prefix replay is deliberately simple and only used in causal tests.
            output.append(combined.iloc[:ready]); pending = combined.iloc[ready:]
        else: pending = combined
    if len(pending): output.append(pending)
    full = pd.concat(output) if output else pd.DataFrame(columns=["timestamp", "customer_id", "merchant_category", "customer_prior_count"])
    # All chunks form one ordered stream; production preserves each complete timestamp batch.
    return customer_category_production(full["timestamp"], full["customer_id"], full["merchant_category"], full["customer_prior_count"]).set_axis(full.index).sort_index()
