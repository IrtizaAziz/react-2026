"""Strictly-prior, timestamp-batched 30-day customer amount history."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .customer_history import _timestamp_ns


FEATURES = [
    "customer_prior_30d_count", "customer_prior_30d_mean_amount",
    "customer_prior_30d_mean_log_amount", "customer_amount_to_prior_30d_mean",
    "customer_log_amount_minus_prior_30d_mean", "customer_30d_to_lifetime_mean_ratio",
]
WINDOW_NS = np.int64(30 * 86_400_000_000_000)


def _input_arrays(timestamp, customer_id, amount_bdt, lifetime_mean_amount):
    times = _timestamp_ns(timestamp)
    customers = pd.Series(customer_id).astype("string").fillna("__MISSING_CUSTOMER__").to_numpy(dtype=str)
    amounts = pd.Series(amount_bdt).to_numpy(dtype=float)
    lifetime_means = pd.Series(lifetime_mean_amount).to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("30-day customer amounts must be finite and nonnegative")
    return times, customers, amounts, np.log1p(amounts), lifetime_means


def _empty_output(n):
    output = {name: np.full(n, np.nan, dtype=float) for name in FEATURES}
    output["customer_prior_30d_count"] = np.zeros(n, dtype=np.int64)
    return output


def _evict(history, current, totals):
    """Keep exactly [t - 30 days, t): the left boundary is inclusive."""
    cutoff = current - WINDOW_NS
    while history and history[0][0] < cutoff:
        _, amount, log_amount = history.popleft()
        totals[0] -= amount
        totals[1] -= log_amount


def _with_state(timestamp, customer_id, amount_bdt, lifetime_mean_amount, state):
    times, customers, amounts, logs, lifetime_means = _input_arrays(
        timestamp, customer_id, amount_bdt, lifetime_mean_amount
    )
    histories, totals = state
    output = _empty_output(len(times))
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        current = times[start]
        batch_customers = customers[start:end]
        for customer in np.unique(batch_customers):
            _evict(histories[customer], current, totals[customer])
        counts = np.asarray([len(histories[customer]) for customer in batch_customers], dtype=np.int64)
        output["customer_prior_30d_count"][start:end] = counts
        seen = counts > 0
        sums = np.asarray([totals[customer][0] for customer in batch_customers], dtype=float)
        log_sums = np.asarray([totals[customer][1] for customer in batch_customers], dtype=float)
        means = np.divide(sums, counts, out=np.full(end - start, np.nan), where=seen)
        log_means = np.divide(log_sums, counts, out=np.full(end - start, np.nan), where=seen)
        output["customer_prior_30d_mean_amount"][start:end] = means
        output["customer_prior_30d_mean_log_amount"][start:end] = log_means
        output["customer_amount_to_prior_30d_mean"][start:end] = np.where(seen, amounts[start:end] / (means + 1.0), np.nan)
        output["customer_log_amount_minus_prior_30d_mean"][start:end] = np.where(seen, logs[start:end] - log_means, np.nan)
        # The lifetime mean is produced by the immutable R007 state engine.
        output["customer_30d_to_lifetime_mean_ratio"][start:end] = np.where(
            seen, (means + 1.0) / (lifetime_means[start:end] + 1.0), np.nan
        )
        # Update only after every row at this timestamp has queried frozen state.
        for customer, amount, log_amount in zip(batch_customers, amounts[start:end], logs[start:end]):
            histories[customer].append((current, amount, log_amount))
            totals[customer][0] += amount
            totals[customer][1] += log_amount
    return pd.DataFrame(output)


def customer_amount_30d_production(timestamp, customer_id, amount_bdt, lifetime_mean_amount):
    """Production causal engine. Its API intentionally accepts no labels or frames."""
    return _with_state(timestamp, customer_id, amount_bdt, lifetime_mean_amount,
                       (defaultdict(deque), defaultdict(lambda: [0.0, 0.0])))


def customer_amount_30d_from_prior_stream(prior_timestamp, prior_customer_id, prior_amount_bdt,
                                           prior_lifetime_mean_amount, current_timestamp,
                                           current_customer_id, current_amount_bdt,
                                           current_lifetime_mean_amount):
    """Continue after a strictly earlier unlabeled raw stream."""
    prior_times = _timestamp_ns(prior_timestamp)
    current_times = _timestamp_ns(current_timestamp)
    if len(prior_times) and len(current_times) and prior_times[-1] >= current_times[0]:
        raise ValueError("Prior 30-day stream must end strictly before the current stream")
    state = (defaultdict(deque), defaultdict(lambda: [0.0, 0.0]))
    _with_state(prior_timestamp, prior_customer_id, prior_amount_bdt, prior_lifetime_mean_amount, state)
    return _with_state(current_timestamp, current_customer_id, current_amount_bdt, current_lifetime_mean_amount, state)


def customer_amount_30d_chunked(chunks):
    """Chunked causal stream processor; never splits a timestamp batch."""
    state = (defaultdict(deque), defaultdict(lambda: [0.0, 0.0]))
    pending, outputs = pd.DataFrame(), []
    required = ["timestamp", "customer_id", "amount_bdt", "customer_prior_mean_amount"]
    for chunk in chunks:
        combined = pd.concat([pending, chunk.copy()])
        if not len(combined):
            continue
        if not set(required) <= set(combined):
            raise ValueError("30-day chunk requires timestamp, customer_id, amount_bdt, and customer_prior_mean_amount")
        times = _timestamp_ns(combined["timestamp"])
        ready_end = np.searchsorted(times, times[-1], side="left")
        if ready_end:
            ready = combined.iloc[:ready_end]
            outputs.append(_with_state(ready["timestamp"], ready["customer_id"], ready["amount_bdt"],
                                       ready["customer_prior_mean_amount"], state).set_axis(ready.index))
        pending = combined.iloc[ready_end:]
    if len(pending):
        outputs.append(_with_state(pending["timestamp"], pending["customer_id"], pending["amount_bdt"],
                                   pending["customer_prior_mean_amount"], state).set_axis(pending.index))
    return pd.concat(outputs).sort_index() if outputs else pd.DataFrame(columns=FEATURES)
