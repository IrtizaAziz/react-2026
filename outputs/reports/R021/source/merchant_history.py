"""Strictly-past timestamp-batched merchant behavior features for R017."""
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from .customer_history import _timestamp_ns
from .customer_relationships import _values

FEATURES = ["merchant_prior_transaction_count", "merchant_seconds_since_last", "merchant_prior_unique_customers", "merchant_transactions_24h", "merchant_unique_customers_24h"]
DAY = np.int64(86_400_000_000_000)
def _input(timestamp, customer_id, merchant_id): return _timestamp_ns(timestamp), _values(customer_id, "__MISSING_CUSTOMER__"), _values(merchant_id, "__MISSING_MERCHANT__")
def _empty(n): return {"merchant_prior_transaction_count": np.zeros(n, dtype=np.int32), "merchant_seconds_since_last": np.full(n, np.nan, dtype=np.float32), "merchant_prior_unique_customers": np.zeros(n, dtype=np.int32), "merchant_transactions_24h": np.zeros(n, dtype=np.int32), "merchant_unique_customers_24h": np.zeros(n, dtype=np.int32)}

def merchant_history_production(timestamp, customer_id, merchant_id):
    """Optimized raw-event engine: each tied timestamp batch reads before updating."""
    times, customers, merchants = _input(timestamp, customer_id, merchant_id); out = _empty(len(times)); total = defaultdict(int); last = {}; lifetime = defaultdict(set); recent = defaultdict(deque); recent_counts = defaultdict(lambda: defaultdict(int))
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]; ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        now = times[start]
        for merchant in np.unique(merchants[start:end]):
            queue, counts = recent[merchant], recent_counts[merchant]
            while queue and queue[0][0] < now - DAY:
                _, customer = queue.popleft(); counts[customer] -= 1
                if counts[customer] == 0: del counts[customer]
        for i in range(start, end):
            merchant = merchants[i]; out["merchant_prior_transaction_count"][i] = total[merchant]; out["merchant_prior_unique_customers"][i] = len(lifetime[merchant]); out["merchant_transactions_24h"][i] = len(recent[merchant]); out["merchant_unique_customers_24h"][i] = len(recent_counts[merchant])
            if merchant in last: out["merchant_seconds_since_last"][i] = (now - last[merchant]) / 1_000_000_000
        for i in range(start, end):
            merchant, customer = merchants[i], customers[i]; total[merchant] += 1; lifetime[merchant].add(customer); last[merchant] = now; recent[merchant].append((now, customer)); recent_counts[merchant][customer] += 1
    return pd.DataFrame(out)

def merchant_history_simple_oracle(timestamp, customer_id, merchant_id):
    """Deliberately simple O(n²) oracle used only in preflight parity tests."""
    times, customers, merchants = _input(timestamp, customer_id, merchant_id); out = _empty(len(times)); history = []; starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]; ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        now = times[start]
        for i in range(start, end):
            merchant = merchants[i]; prior = [(t, c, m) for t, c, m in history if m == merchant]
            out["merchant_prior_transaction_count"][i] = len(prior); out["merchant_prior_unique_customers"][i] = len({c for _, c, _ in prior}); out["merchant_transactions_24h"][i] = sum(t >= now - DAY for t, _, _ in prior); out["merchant_unique_customers_24h"][i] = len({c for t, c, _ in prior if t >= now - DAY})
            if prior: out["merchant_seconds_since_last"][i] = (now - prior[-1][0]) / 1_000_000_000
        history.extend((now, customers[i], merchants[i]) for i in range(start, end))
    return pd.DataFrame(out)

def merchant_history_chunked(chunks):
    full = pd.concat([chunk.copy() for chunk in chunks]); return merchant_history_production(full.timestamp, full.customer_id, full.merchant_id).set_axis(full.index).sort_index()
