"""Strictly-past timestamp-batched merchant amount features for R022."""
import numpy as np
import pandas as pd

from .customer_history import _timestamp_ns
from .customer_relationships import _values


FEATURES = [
    "merchant_prior_mean_amount", "merchant_prior_mean_log_amount",
    "merchant_prior_std_log_amount", "merchant_amount_to_prior_mean",
    "merchant_log_amount_zscore",
]


def _inputs(timestamp, merchant_id, amount_bdt):
    times = _timestamp_ns(timestamp)
    merchants = _values(merchant_id, "__MISSING_MERCHANT__")
    amounts = pd.Series(amount_bdt).to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Merchant amount-history amounts must be finite and nonnegative")
    return times, merchants, amounts, np.log1p(amounts)


def _empty(n):
    return {name: np.full(n, np.nan, dtype=float) for name in FEATURES}


def merchant_amount_history_production(timestamp, merchant_id, amount_bdt):
    """Optimized raw-event engine; a timestamp batch reads frozen prior state."""
    times, merchants, amounts, logs = _inputs(timestamp, merchant_id, amount_bdt)
    codes, uniques = pd.factorize(merchants, sort=False)
    count = np.zeros(len(uniques), dtype=np.int64)
    amount_sum = np.zeros(len(uniques), dtype=float)
    log_sum = np.zeros(len(uniques), dtype=float)
    log_sq_sum = np.zeros(len(uniques), dtype=float)
    out = _empty(len(times))
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        batch_codes = codes[start:end]
        old_count = count[batch_codes]
        seen = old_count > 0
        mean_amount = np.divide(amount_sum[batch_codes], old_count, out=np.full(end-start, np.nan), where=seen)
        mean_log = np.divide(log_sum[batch_codes], old_count, out=np.full(end-start, np.nan), where=seen)
        variance = np.divide(log_sq_sum[batch_codes], old_count, out=np.zeros(end-start), where=seen) - np.where(seen, mean_log ** 2, 0.0)
        std_log = np.sqrt(np.maximum(variance, 0.0))
        out["merchant_prior_mean_amount"][start:end] = mean_amount
        out["merchant_prior_mean_log_amount"][start:end] = mean_log
        out["merchant_prior_std_log_amount"][start:end] = np.where(seen, std_log, np.nan)
        out["merchant_amount_to_prior_mean"][start:end] = np.where(seen, amounts[start:end] / (mean_amount + 1.0), np.nan)
        difference = logs[start:end] - mean_log
        out["merchant_log_amount_zscore"][start:end] = np.where(old_count >= 5, difference / np.maximum(std_log, 0.1), np.nan)
        unique_codes, inverse = np.unique(batch_codes, return_inverse=True)
        count[unique_codes] += np.bincount(inverse, minlength=len(unique_codes))
        amount_sum[unique_codes] += np.bincount(inverse, weights=amounts[start:end], minlength=len(unique_codes))
        log_sum[unique_codes] += np.bincount(inverse, weights=logs[start:end], minlength=len(unique_codes))
        log_sq_sum[unique_codes] += np.bincount(inverse, weights=logs[start:end] ** 2, minlength=len(unique_codes))
    return pd.DataFrame(out)


def merchant_amount_history_simple_oracle(timestamp, merchant_id, amount_bdt):
    """Deliberately simple O(n²) reference implementation for causal tests."""
    times, merchants, amounts, logs = _inputs(timestamp, merchant_id, amount_bdt)
    out = _empty(len(times)); history = []
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]; ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        for i in range(start, end):
            prior = [amount for _, merchant, amount in history if merchant == merchants[i]]
            if prior:
                prior = np.asarray(prior, dtype=float); prior_logs = np.log1p(prior)
                mean_amount, mean_log = float(prior.mean()), float(prior_logs.mean())
                std_log = float(np.sqrt(max((prior_logs ** 2).mean() - mean_log ** 2, 0.0)))
                out["merchant_prior_mean_amount"][i] = mean_amount
                out["merchant_prior_mean_log_amount"][i] = mean_log
                out["merchant_prior_std_log_amount"][i] = std_log
                out["merchant_amount_to_prior_mean"][i] = amounts[i] / (mean_amount + 1.0)
                if len(prior) >= 5:
                    out["merchant_log_amount_zscore"][i] = (logs[i] - mean_log) / max(std_log, 0.1)
        history.extend((times[i], merchants[i], amounts[i]) for i in range(start, end))
    return pd.DataFrame(out)


def merchant_amount_history_chunked(chunks):
    full = pd.concat([chunk.copy() for chunk in chunks])
    return merchant_amount_history_production(full.timestamp, full.merchant_id, full.amount_bdt).set_axis(full.index).sort_index()
