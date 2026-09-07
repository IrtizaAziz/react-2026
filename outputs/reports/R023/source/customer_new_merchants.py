"""Strictly-past customer new-merchant relationship creation features for R023."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .customer_history import _timestamp_ns
from .customer_relationships import _pair_codes, _values


FEATURES = [
    "customer_new_merchants_7d", "customer_new_merchants_30d",
    "customer_new_merchant_share_7d", "customer_new_merchant_share_30d",
    "customer_seconds_since_last_new_merchant",
]
_DAY_NS = 86_400 * 1_000_000_000


def _inputs(timestamp, customer_id, merchant_id):
    return (_timestamp_ns(timestamp), _values(customer_id, "__MISSING_CUSTOMER__"),
            _values(merchant_id, "__MISSING_MERCHANT__"))


def _empty(n):
    return {
        "customer_new_merchants_7d": np.zeros(n, dtype=np.int32),
        "customer_new_merchants_30d": np.zeros(n, dtype=np.int32),
        "customer_new_merchant_share_7d": np.zeros(n, dtype=np.float32),
        "customer_new_merchant_share_30d": np.zeros(n, dtype=np.float32),
        "customer_seconds_since_last_new_merchant": np.full(n, np.nan, dtype=np.float32),
    }


def _compute(timestamp, customer_id, merchant_id, state):
    times, customers, merchants = _inputs(timestamp, customer_id, merchant_id)
    out = _empty(len(times)); seen_pairs = state.setdefault("seen_pairs", set())
    new_events_7 = state.setdefault("new_events_7", defaultdict(deque))
    new_events_30 = state.setdefault("new_events_30", defaultdict(deque))
    transactions_7 = state.setdefault("transactions_7", defaultdict(deque))
    transactions_30 = state.setdefault("transactions_30", defaultdict(deque))
    last_new = state.setdefault("last_new", {})
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        current = times[start]
        batch_customers = customers[start:end]
        for customer in np.unique(batch_customers):
            for queue, width in ((new_events_7[customer], 7), (new_events_30[customer], 30),
                                 (transactions_7[customer], 7), (transactions_30[customer], 30)):
                while queue and queue[0] <= current - width * _DAY_NS: queue.popleft()
        for i in range(start, end):
            customer = customers[i]; event_7d, event_30d = len(new_events_7[customer]), len(new_events_30[customer])
            txn_7d, txn_30d = len(transactions_7[customer]), len(transactions_30[customer])
            out["customer_new_merchants_7d"][i] = event_7d
            out["customer_new_merchants_30d"][i] = event_30d
            out["customer_new_merchant_share_7d"][i] = event_7d / txn_7d if txn_7d else 0.0
            out["customer_new_merchant_share_30d"][i] = event_30d / txn_30d if txn_30d else 0.0
            if customer in last_new:
                out["customer_seconds_since_last_new_merchant"][i] = (current - last_new[customer]) / 1_000_000_000
        # The state changes only after every row in this timestamp batch is queried.
        unique_pairs = set(zip(customers[start:end], merchants[start:end]))
        created = {(customer, merchant) for customer, merchant in unique_pairs if (customer, merchant) not in seen_pairs}
        seen_pairs.update(unique_pairs)
        for customer, _ in created:
            new_events_7[customer].append(current); new_events_30[customer].append(current)
            last_new[customer] = current
        for customer in batch_customers:
            transactions_7[customer].append(current); transactions_30[customer].append(current)
    return pd.DataFrame(out)


def customer_new_merchants_production(timestamp, customer_id, merchant_id):
    """Timestamp-batched production engine; each new pair creates one event once."""
    times, customers, merchants = _inputs(timestamp, customer_id, merchant_id)
    customer_codes, customer_values = pd.factorize(customers, sort=False)
    merchant_codes, _ = pd.factorize(merchants, sort=False)
    pair_codes = _pair_codes(customer_codes, merchant_codes)
    pair_seen = np.zeros(int(pair_codes.max()) + 1 if len(pair_codes) else 0, dtype=bool)
    pair_customer = np.empty(len(pair_seen), dtype=np.int64); pair_customer[pair_codes] = customer_codes
    event7, event30, txn7, txn30 = ([deque() for _ in customer_values] for _ in range(4))
    last_new = np.full(len(customer_values), -1, dtype=np.int64); out = _empty(len(times))
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]; ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        current, batch_customers = times[start], customer_codes[start:end]
        for customer in set(batch_customers):
            for queue, width in ((event7[customer], 7), (event30[customer], 30),
                                 (txn7[customer], 7), (txn30[customer], 30)):
                while queue and queue[0] <= current - width * _DAY_NS: queue.popleft()
        for i in range(start, end):
            customer = customer_codes[i]; e7, e30, t7, t30 = len(event7[customer]), len(event30[customer]), len(txn7[customer]), len(txn30[customer])
            out["customer_new_merchants_7d"][i], out["customer_new_merchants_30d"][i] = e7, e30
            out["customer_new_merchant_share_7d"][i] = e7 / t7 if t7 else 0.0
            out["customer_new_merchant_share_30d"][i] = e30 / t30 if t30 else 0.0
            if last_new[customer] >= 0: out["customer_seconds_since_last_new_merchant"][i] = (current - last_new[customer]) / 1_000_000_000
        unique_pairs = np.unique(pair_codes[start:end]); created = unique_pairs[~pair_seen[unique_pairs]]; pair_seen[unique_pairs] = True
        for pair in created:
            customer = pair_customer[pair]
            event7[customer].append(current); event30[customer].append(current); last_new[customer] = current
        for customer in batch_customers:
            txn7[customer].append(current); txn30[customer].append(current)
    return pd.DataFrame(out)


def customer_new_merchants_simple_oracle(timestamp, customer_id, merchant_id):
    """Deliberately direct strictly-past oracle used only for causal verification."""
    times, customers, merchants = _inputs(timestamp, customer_id, merchant_id)
    out = _empty(len(times)); prior_rows, prior_pairs = [], set()
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]; ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        current = times[start]
        for i in range(start, end):
            customer = customers[i]
            events = [t for t, c, kind in prior_rows if c == customer and kind == "new"]
            txns = [t for t, c, kind in prior_rows if c == customer and kind == "txn"]
            e7, e30 = sum(t > current - 7 * _DAY_NS for t in events), sum(t > current - 30 * _DAY_NS for t in events)
            t7, t30 = sum(t > current - 7 * _DAY_NS for t in txns), sum(t > current - 30 * _DAY_NS for t in txns)
            out["customer_new_merchants_7d"][i], out["customer_new_merchants_30d"][i] = e7, e30
            out["customer_new_merchant_share_7d"][i] = e7 / t7 if t7 else 0.0
            out["customer_new_merchant_share_30d"][i] = e30 / t30 if t30 else 0.0
            if events: out["customer_seconds_since_last_new_merchant"][i] = (current - max(events)) / 1_000_000_000
        unique_pairs = set(zip(customers[start:end], merchants[start:end])); created = unique_pairs - prior_pairs
        prior_pairs.update(unique_pairs)
        prior_rows.extend((current, customer, "new") for customer, _ in created)
        prior_rows.extend((current, customer, "txn") for customer in customers[start:end])
    return pd.DataFrame(out)


def customer_new_merchants_chunked(chunks):
    """Streaming implementation that retains a final complete timestamp batch."""
    state, pending, outputs = {}, pd.DataFrame(), []
    for chunk in chunks:
        combined = pd.concat([pending, chunk.copy()])
        if not len(combined): continue
        times = _timestamp_ns(combined["timestamp"]); ready_end = np.searchsorted(times, times[-1], side="left")
        if ready_end:
            ready = combined.iloc[:ready_end]
            outputs.append(_compute(ready.timestamp, ready.customer_id, ready.merchant_id, state).set_axis(ready.index))
        pending = combined.iloc[ready_end:]
    if len(pending): outputs.append(_compute(pending.timestamp, pending.customer_id, pending.merchant_id, state).set_axis(pending.index))
    return pd.concat(outputs).sort_index() if outputs else pd.DataFrame(columns=FEATURES)
