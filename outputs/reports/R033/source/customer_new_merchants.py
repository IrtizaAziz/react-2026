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
AVAILABLE_FEATURES = tuple(FEATURES)
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


def build_features(raw_frame, requested_features):
    try:
        requested = tuple(requested_features) if not isinstance(requested_features, str) else ()
    except TypeError:
        requested = ()
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(AVAILABLE_FEATURES):
        raise ValueError("requested_features must be a non-empty ordered subset of AVAILABLE_FEATURES")
    missing = [column for column in ("timestamp", "customer_id", "merchant_id") if column not in raw_frame]
    if missing:
        raise ValueError(f"raw_frame is missing required columns: {missing}")
    values = customer_new_merchants_production(raw_frame.timestamp, raw_frame.customer_id, raw_frame.merchant_id)
    return values.loc[:, list(requested)].set_axis(raw_frame.index)


def _certification_frame():
    day = 86_400
    return pd.DataFrame({"transaction_id": list("abcdefghij"), "timestamp": [0, 0, day, day, 2*day, 8*day, 8*day, 31*day, 32*day, 32*day],
        "customer_id": ["a", "a", "a", "a", "a", "a", "b", "a", "a", "a"],
        "merchant_id": ["m", "m", "m", "n", "o", "m", "x", "p", "q", "q"], "fraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})


def certification_cases():
    def actual(frame): return customer_new_merchants_production(frame.timestamp, frame.customer_id, frame.merchant_id)
    def strict_past():
        values = actual(_certification_frame())
        assert values.customer_new_merchants_7d.tolist()[:5] == [0, 0, 1, 1, 2]
    def oracle():
        frame = _certification_frame(); pd.testing.assert_frame_equal(actual(frame), customer_new_merchants_simple_oracle(frame.timestamp, frame.customer_id, frame.merchant_id))
    def equal_timestamp_isolation():
        values = actual(_certification_frame()); assert values.iloc[0].equals(values.iloc[1])
    def permutation_invariance():
        frame = _certification_frame(); values = actual(frame); permuted = pd.concat([frame.iloc[:2], frame.iloc[[3, 2]], frame.iloc[4:]], ignore_index=True)
        pd.testing.assert_frame_equal(pd.concat([frame.transaction_id, values], axis=1).set_index("transaction_id").sort_index(), pd.concat([permuted.transaction_id, actual(permuted)], axis=1).set_index("transaction_id").sort_index())
    def duplicate_same_timestamp_customer_merchant():
        frame = _certification_frame(); values = actual(pd.concat([frame.iloc[:2], frame.iloc[[2, 2]], frame.iloc[3:]], ignore_index=True)); assert values.iloc[2].equals(values.iloc[3])
    def future_independence():
        frame = _certification_frame(); expected = actual(frame); future = pd.concat([frame, pd.DataFrame({"timestamp": [99*86_400], "customer_id": ["a"], "merchant_id": ["m"]})], ignore_index=True); pd.testing.assert_frame_equal(expected, actual(future).iloc[:len(frame)].reset_index(drop=True))
    def label_independence():
        frame = _certification_frame(); changed = frame.assign(fraud=1-frame.fraud); pd.testing.assert_frame_equal(actual(frame), actual(changed))
    def chunk_vs_whole():
        frame = _certification_frame(); pd.testing.assert_frame_equal(actual(frame), customer_new_merchants_chunked([frame.iloc[:2], frame.iloc[2:6], frame.iloc[6:]]))
    return {"strict_past": strict_past, "oracle": oracle, "equal_timestamp_isolation": equal_timestamp_isolation,
            "permutation_invariance": permutation_invariance, "duplicate_same_timestamp_customer_merchant": duplicate_same_timestamp_customer_merchant,
            "future_independence": future_independence, "label_independence": label_independence, "chunk_vs_whole": chunk_vs_whole}
