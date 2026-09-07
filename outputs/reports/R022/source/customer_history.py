"""Strictly-past, timestamp-batched customer amount history for REACT 2026."""
import numpy as np
import pandas as pd


FEATURES = [
    "customer_prior_count", "customer_first_seen", "customer_seconds_since_last",
    "customer_observed_age_seconds", "customer_prior_mean_amount", "customer_prior_mean_log_amount",
    "customer_prior_std_log_amount", "customer_amount_to_prior_mean",
    "customer_log_amount_minus_prior_mean", "customer_log_amount_zscore",
]


def _timestamp_ns(values):
    """Normalize datetime inputs to explicit int64 nanoseconds; numeric inputs are seconds."""
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        result = series.to_numpy(dtype=np.int64) * np.int64(1_000_000_000)
    else:
        parsed = pd.to_datetime(series, errors="raise", format="mixed")
        result = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if len(result) and np.any(result[1:] < result[:-1]):
        raise ValueError("Customer-history inputs must be nondecreasing by timestamp")
    return result


def _input_arrays(timestamp, customer_id, amount_bdt):
    times = _timestamp_ns(timestamp)
    customers = pd.Series(customer_id).astype("string").fillna("__MISSING_CUSTOMER__").to_numpy(dtype=str)
    amounts = pd.Series(amount_bdt).to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Customer-history amounts must be finite and nonnegative")
    return times, customers, amounts, np.log1p(amounts)


def _empty_output(n):
    result = {name: np.full(n, np.nan, dtype=float) for name in FEATURES}
    result["customer_prior_count"] = np.zeros(n, dtype=np.int64)
    result["customer_first_seen"] = np.zeros(n, dtype=np.int8)
    return result


def _oracle_with_state(timestamp, customer_id, amount_bdt, state):
    times, customers, amounts, logs = _input_arrays(timestamp, customer_id, amount_bdt)
    out = _empty_output(len(times))
    start = 0
    while start < len(times):
        end = start + 1
        while end < len(times) and times[end] == times[start]:
            end += 1
        for i in range(start, end):
            count, first, last, total, log_total, log_sq_total = state.get(customers[i], (0, None, None, 0.0, 0.0, 0.0))
            out["customer_prior_count"][i] = count
            out["customer_first_seen"][i] = int(count == 0)
            if count:
                mean_amount = total / count
                mean_log = log_total / count
                std_log = np.sqrt(max(log_sq_total / count - mean_log ** 2, 0.0))
                out["customer_seconds_since_last"][i] = (times[i] - last) / 1_000_000_000
                out["customer_observed_age_seconds"][i] = (times[i] - first) / 1_000_000_000
                out["customer_prior_mean_amount"][i] = mean_amount
                out["customer_prior_mean_log_amount"][i] = mean_log
                out["customer_prior_std_log_amount"][i] = std_log
                out["customer_amount_to_prior_mean"][i] = amounts[i] / (mean_amount + 1.0)
                out["customer_log_amount_minus_prior_mean"][i] = logs[i] - mean_log
                if count >= 5:
                    out["customer_log_amount_zscore"][i] = (logs[i] - mean_log) / max(std_log, 0.1)
        for i in range(start, end):
            count, first, _, total, log_total, log_sq_total = state.get(customers[i], (0, times[i], None, 0.0, 0.0, 0.0))
            state[customers[i]] = (count + 1, first, times[i], total + amounts[i], log_total + logs[i], log_sq_total + logs[i] ** 2)
        start = end
    return pd.DataFrame(out)


def customer_history_oracle(timestamp, customer_id, amount_bdt):
    """Small direct reference implementation; never accepts labels or arbitrary frames."""
    return _oracle_with_state(timestamp, customer_id, amount_bdt, {})


def customer_history_from_prior_stream(prior_timestamp, prior_customer_id, prior_amount_bdt,
                                       current_timestamp, current_customer_id, current_amount_bdt):
    """Score a chronological unlabeled stream after a strictly earlier raw stream.

    Both streams are processed in timestamp batches.  The prior stream only
    initializes raw state; no labels or arbitrary data frames can enter this API.
    """
    state = {}
    _oracle_with_state(prior_timestamp, prior_customer_id, prior_amount_bdt, state)
    return _oracle_with_state(current_timestamp, current_customer_id, current_amount_bdt, state)


def customer_history_production(timestamp, customer_id, amount_bdt):
    """Vectorized timestamp-batch state engine used for live data."""
    times, customers, amounts, logs = _input_arrays(timestamp, customer_id, amount_bdt)
    codes, uniques = pd.factorize(customers, sort=False)
    m, n = len(uniques), len(times)
    count = np.zeros(m, dtype=np.int64)
    first = np.full(m, -1, dtype=np.int64)
    last = np.full(m, -1, dtype=np.int64)
    amount_sum = np.zeros(m, dtype=float)
    log_sum = np.zeros(m, dtype=float)
    log_sq_sum = np.zeros(m, dtype=float)
    out = _empty_output(n)
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    for start, end in zip(starts, ends):
        batch_codes, current = codes[start:end], times[start]
        old_count = count[batch_codes]
        seen = old_count > 0
        out["customer_prior_count"][start:end] = old_count
        out["customer_first_seen"][start:end] = (~seen).astype(np.int8)
        mean_amount = np.divide(amount_sum[batch_codes], old_count, out=np.full(end - start, np.nan), where=seen)
        mean_log = np.divide(log_sum[batch_codes], old_count, out=np.full(end - start, np.nan), where=seen)
        variance = np.divide(log_sq_sum[batch_codes], old_count, out=np.zeros(end - start), where=seen) - np.where(seen, mean_log ** 2, 0.0)
        std_log = np.sqrt(np.maximum(variance, 0.0))
        out["customer_seconds_since_last"][start:end] = np.where(seen, (current - last[batch_codes]) / 1_000_000_000, np.nan)
        out["customer_observed_age_seconds"][start:end] = np.where(seen, (current - first[batch_codes]) / 1_000_000_000, np.nan)
        out["customer_prior_mean_amount"][start:end] = mean_amount
        out["customer_prior_mean_log_amount"][start:end] = mean_log
        out["customer_prior_std_log_amount"][start:end] = np.where(seen, std_log, np.nan)
        out["customer_amount_to_prior_mean"][start:end] = np.where(seen, amounts[start:end] / (mean_amount + 1.0), np.nan)
        difference = logs[start:end] - mean_log
        out["customer_log_amount_minus_prior_mean"][start:end] = np.where(seen, difference, np.nan)
        enough = old_count >= 5
        out["customer_log_amount_zscore"][start:end] = np.where(enough, difference / np.maximum(std_log, 0.1), np.nan)
        unique_codes, inverse = np.unique(batch_codes, return_inverse=True)
        batch_counts = np.bincount(inverse, minlength=len(unique_codes))
        count[unique_codes] += batch_counts
        amount_sum[unique_codes] += np.bincount(inverse, weights=amounts[start:end], minlength=len(unique_codes))
        log_sum[unique_codes] += np.bincount(inverse, weights=logs[start:end], minlength=len(unique_codes))
        log_sq_sum[unique_codes] += np.bincount(inverse, weights=logs[start:end] ** 2, minlength=len(unique_codes))
        first[unique_codes[first[unique_codes] < 0]] = current
        last[unique_codes] = current
    return pd.DataFrame(out)


def customer_history_chunked(chunks):
    """Stream chunks while retaining the last tied timestamp batch until it is complete."""
    state, pending, outputs = {}, pd.DataFrame(), []
    for chunk in chunks:
        combined = pd.concat([pending, chunk.copy()])
        if not len(combined):
            continue
        times = _timestamp_ns(combined["timestamp"])
        ready_end = np.searchsorted(times, times[-1], side="left")
        if ready_end:
            ready = combined.iloc[:ready_end]
            outputs.append(_oracle_with_state(ready["timestamp"], ready["customer_id"], ready["amount_bdt"], state).set_axis(ready.index))
        pending = combined.iloc[ready_end:]
    if len(pending):
        outputs.append(_oracle_with_state(pending["timestamp"], pending["customer_id"], pending["amount_bdt"], state).set_axis(pending.index))
    return pd.concat(outputs).sort_index() if outputs else pd.DataFrame(columns=FEATURES)
