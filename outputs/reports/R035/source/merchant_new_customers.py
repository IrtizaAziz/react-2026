"""Strictly-past merchant-side new-customer relationship creation features for R025."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .customer_history import _timestamp_ns
from .customer_relationships import _pair_codes, _values


FEATURES = [
    "merchant_new_customers_24h", "merchant_new_customers_7d",
    "merchant_new_customer_share_24h", "merchant_new_customer_share_7d",
    "merchant_seconds_since_last_new_customer",
]
# Generic feature-spec interface.  This is intentionally an alias rather than
# a second manifest: R025 remains the sole definition of this feature family.
AVAILABLE_FEATURES = tuple(FEATURES)
_DAY_NS = 86_400 * 1_000_000_000


def _inputs(timestamp, customer_id, merchant_id):
    return (_timestamp_ns(timestamp), _values(customer_id, "__MISSING_CUSTOMER__"),
            _values(merchant_id, "__MISSING_MERCHANT__"))


def _empty(n):
    return {
        "merchant_new_customers_24h": np.zeros(n, dtype=np.int32),
        "merchant_new_customers_7d": np.zeros(n, dtype=np.int32),
        "merchant_new_customer_share_24h": np.zeros(n, dtype=np.float32),
        "merchant_new_customer_share_7d": np.zeros(n, dtype=np.float32),
        "merchant_seconds_since_last_new_customer": np.full(n, np.nan, dtype=np.float32),
    }


def _compute(timestamp, customer_id, merchant_id, state):
    """Streaming string-key engine; caller retains any trailing tied batch."""
    times, customers, merchants = _inputs(timestamp, customer_id, merchant_id)
    out = _empty(len(times)); seen_pairs = state.setdefault("seen_pairs", set())
    event24 = state.setdefault("event24", defaultdict(deque)); event7 = state.setdefault("event7", defaultdict(deque))
    txn24 = state.setdefault("txn24", defaultdict(deque)); txn7 = state.setdefault("txn7", defaultdict(deque))
    txn24_counts = state.setdefault("txn24_counts", defaultdict(lambda: defaultdict(int)))
    txn7_counts = state.setdefault("txn7_counts", defaultdict(lambda: defaultdict(int)))
    last_new = state.setdefault("last_new", {})
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]; ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        now, batch_merchants = times[start], merchants[start:end]
        for merchant in np.unique(batch_merchants):
            for queue, width in ((event24[merchant], 1), (event7[merchant], 7)):
                while queue and queue[0] <= now - width * _DAY_NS: queue.popleft()
            for queue, counts, width in ((txn24[merchant], txn24_counts[merchant], 1), (txn7[merchant], txn7_counts[merchant], 7)):
                while queue and queue[0][0] <= now - width * _DAY_NS:
                    _, customer = queue.popleft(); counts[customer] -= 1
                    if counts[customer] == 0: del counts[customer]
        for i in range(start, end):
            merchant = merchants[i]; e24, e7 = len(event24[merchant]), len(event7[merchant])
            d24, d7 = len(txn24_counts[merchant]), len(txn7_counts[merchant])
            out["merchant_new_customers_24h"][i], out["merchant_new_customers_7d"][i] = e24, e7
            out["merchant_new_customer_share_24h"][i] = e24 / d24 if d24 else 0.0
            out["merchant_new_customer_share_7d"][i] = e7 / d7 if d7 else 0.0
            if merchant in last_new: out["merchant_seconds_since_last_new_customer"][i] = (now - last_new[merchant]) / 1_000_000_000
        unique_pairs = set(zip(customers[start:end], merchants[start:end]))
        created = {(customer, merchant) for customer, merchant in unique_pairs if (customer, merchant) not in seen_pairs}
        seen_pairs.update(unique_pairs)
        for _, merchant in created:
            event24[merchant].append(now); event7[merchant].append(now); last_new[merchant] = now
        for customer, merchant in zip(customers[start:end], merchants[start:end]):
            txn24[merchant].append((now, customer)); txn24_counts[merchant][customer] += 1
            txn7[merchant].append((now, customer)); txn7_counts[merchant][customer] += 1
    return pd.DataFrame(out)


def merchant_new_customers_production(timestamp, customer_id, merchant_id):
    """Optimized timestamp-batched engine; unseen customer--merchant pairs create one event."""
    times, customers, merchants = _inputs(timestamp, customer_id, merchant_id)
    customer_codes, _ = pd.factorize(customers, sort=False); merchant_codes, merchant_values = pd.factorize(merchants, sort=False)
    pair_codes = _pair_codes(customer_codes, merchant_codes); pair_seen = np.zeros(int(pair_codes.max()) + 1 if len(pair_codes) else 0, dtype=bool)
    pair_merchant = np.empty(len(pair_seen), dtype=np.int64); pair_merchant[pair_codes] = merchant_codes
    event24, event7, txn24, txn7 = ([deque() for _ in merchant_values] for _ in range(4))
    txn24_counts = [defaultdict(int) for _ in merchant_values]; txn7_counts = [defaultdict(int) for _ in merchant_values]
    last_new = np.full(len(merchant_values), -1, dtype=np.int64); out = _empty(len(times))
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]; ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        now, batch_merchants = times[start], merchant_codes[start:end]
        for merchant in set(batch_merchants):
            for queue, width in ((event24[merchant], 1), (event7[merchant], 7)):
                while queue and queue[0] <= now - width * _DAY_NS: queue.popleft()
            for queue, counts, width in ((txn24[merchant], txn24_counts[merchant], 1), (txn7[merchant], txn7_counts[merchant], 7)):
                while queue and queue[0][0] <= now - width * _DAY_NS:
                    _, customer = queue.popleft(); counts[customer] -= 1
                    if counts[customer] == 0: del counts[customer]
        for i in range(start, end):
            merchant = merchant_codes[i]; e24, e7 = len(event24[merchant]), len(event7[merchant])
            d24, d7 = len(txn24_counts[merchant]), len(txn7_counts[merchant])
            out["merchant_new_customers_24h"][i], out["merchant_new_customers_7d"][i] = e24, e7
            out["merchant_new_customer_share_24h"][i] = e24 / d24 if d24 else 0.0
            out["merchant_new_customer_share_7d"][i] = e7 / d7 if d7 else 0.0
            if last_new[merchant] >= 0: out["merchant_seconds_since_last_new_customer"][i] = (now - last_new[merchant]) / 1_000_000_000
        unique_pairs = np.unique(pair_codes[start:end]); created = unique_pairs[~pair_seen[unique_pairs]]; pair_seen[unique_pairs] = True
        for pair in created:
            merchant = pair_merchant[pair]; event24[merchant].append(now); event7[merchant].append(now); last_new[merchant] = now
        for i in range(start, end):
            merchant, customer = merchant_codes[i], customer_codes[i]
            txn24[merchant].append((now, customer)); txn24_counts[merchant][customer] += 1
            txn7[merchant].append((now, customer)); txn7_counts[merchant][customer] += 1
    return pd.DataFrame(out)


def merchant_new_customers_simple_oracle(timestamp, customer_id, merchant_id):
    """Direct O(n²) oracle for causal verification only."""
    times, customers, merchants = _inputs(timestamp, customer_id, merchant_id); out = _empty(len(times)); history, prior_pairs = [], set()
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]; ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        now = times[start]
        for i in range(start, end):
            merchant = merchants[i]; events = [t for t, m, kind, _ in history if m == merchant and kind == "new"]
            txns = [(t, c) for t, m, kind, c in history if m == merchant and kind == "txn"]
            e24, e7 = sum(t > now - _DAY_NS for t in events), sum(t > now - 7 * _DAY_NS for t in events)
            d24 = len({c for t, c in txns if t > now - _DAY_NS}); d7 = len({c for t, c in txns if t > now - 7 * _DAY_NS})
            out["merchant_new_customers_24h"][i], out["merchant_new_customers_7d"][i] = e24, e7
            out["merchant_new_customer_share_24h"][i] = e24 / d24 if d24 else 0.0; out["merchant_new_customer_share_7d"][i] = e7 / d7 if d7 else 0.0
            if events: out["merchant_seconds_since_last_new_customer"][i] = (now - max(events)) / 1_000_000_000
        unique_pairs = set(zip(customers[start:end], merchants[start:end])); created = unique_pairs - prior_pairs; prior_pairs.update(unique_pairs)
        history.extend((now, merchant, "new", customer) for customer, merchant in created)
        history.extend((now, merchants[i], "txn", customers[i]) for i in range(start, end))
    return pd.DataFrame(out)


def merchant_new_customers_chunked(chunks):
    state, pending, outputs = {}, pd.DataFrame(), []
    for chunk in chunks:
        combined = pd.concat([pending, chunk.copy()])
        if combined.empty: continue
        times = _timestamp_ns(combined["timestamp"]); ready_end = np.searchsorted(times, times[-1], side="left")
        if ready_end:
            ready = combined.iloc[:ready_end]; outputs.append(_compute(ready.timestamp, ready.customer_id, ready.merchant_id, state).set_axis(ready.index))
        pending = combined.iloc[ready_end:]
    if not pending.empty: outputs.append(_compute(pending.timestamp, pending.customer_id, pending.merchant_id, state).set_axis(pending.index))
    return pd.concat(outputs).sort_index() if outputs else pd.DataFrame(columns=FEATURES)


def build_features(raw_frame, requested_features):
    """Return an ordered R025 feature subset without changing its causal engine."""
    try:
        requested = tuple(requested_features) if not isinstance(requested_features, str) else ()
    except TypeError:
        requested = ()
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(AVAILABLE_FEATURES):
        raise ValueError("requested_features must be a non-empty ordered subset of AVAILABLE_FEATURES")
    required = ("timestamp", "customer_id", "merchant_id")
    missing = [column for column in required if column not in raw_frame]
    if missing:
        raise ValueError(f"raw_frame is missing required columns: {missing}")
    values = merchant_new_customers_production(raw_frame["timestamp"], raw_frame["customer_id"], raw_frame["merchant_id"])
    return values.loc[:, list(requested)].set_axis(raw_frame.index)


def _certification_frame():
    day = 86_400
    return pd.DataFrame({"transaction_id": list("abcdefghijk"),
        "timestamp": [0, 0, day - 1, day - 1, day - 1, 2 * day - 1, 2 * day - 1, 8 * day, 8 * day, 9 * day, 9 * day],
        "customer_id": ["a", "a", "a", "b", "b", "a", "c", "a", "d", "e", "e"],
        "merchant_id": ["m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "n"],
        "fraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]})


def certification_cases():
    """Expose R025's existing causal checks to the spec-driven runner."""
    def actual(frame):
        return merchant_new_customers_production(frame.timestamp, frame.customer_id, frame.merchant_id)

    def strict_past():
        values = actual(_certification_frame())
        if values.merchant_new_customers_24h.tolist()[:7] != [0, 0, 1, 1, 1, 0, 0]:
            raise AssertionError("strictly-past 24-hour relationship events changed")
        if values.merchant_new_customers_7d.tolist()[:7] != [0, 0, 1, 1, 1, 2, 2]:
            raise AssertionError("strictly-past 7-day relationship events changed")

    def oracle():
        frame = _certification_frame()
        pd.testing.assert_frame_equal(actual(frame), merchant_new_customers_simple_oracle(frame.timestamp, frame.customer_id, frame.merchant_id))

    def equal_timestamp_isolation():
        values = actual(_certification_frame())
        if not values.iloc[2].equals(values.iloc[3]):
            raise AssertionError("equal-timestamp rows influenced one another")

    def permutation_invariance():
        frame = _certification_frame(); values = actual(frame)
        permuted = pd.concat([frame.iloc[:2], frame.iloc[[4, 2, 3]], frame.iloc[5:]], ignore_index=True)
        observed = actual(permuted)
        pd.testing.assert_frame_equal(pd.concat([frame.transaction_id, values], axis=1).set_index("transaction_id").sort_index(),
                                      pd.concat([permuted.transaction_id, observed], axis=1).set_index("transaction_id").sort_index())

    def future_independence():
        frame = _certification_frame(); values = actual(frame); future = pd.concat([frame, pd.DataFrame({"transaction_id": ["z"], "timestamp": [99 * 86_400], "customer_id": ["future_customer"], "merchant_id": ["future_merchant"], "fraud": [1]})], ignore_index=True)
        pd.testing.assert_frame_equal(values, actual(future).iloc[:len(frame)].reset_index(drop=True))
        mutated = future.copy(); mutated.loc[len(frame), ["customer_id", "merchant_id"]] = ["a", "m"]
        pd.testing.assert_frame_equal(values, actual(mutated).iloc[:len(frame)].reset_index(drop=True))

    def label_independence():
        frame = _certification_frame(); labels = frame.copy(); labels["fraud"] = 1 - labels["fraud"]
        pd.testing.assert_frame_equal(actual(frame), actual(labels))

    def chunk_vs_whole():
        frame = _certification_frame()
        pd.testing.assert_frame_equal(actual(frame), merchant_new_customers_chunked([frame.iloc[:3], frame.iloc[3:7], frame.iloc[7:]]))

    return {"strict_past": strict_past, "oracle": oracle, "equal_timestamp_isolation": equal_timestamp_isolation,
            "permutation_invariance": permutation_invariance, "future_independence": future_independence,
            "label_independence": label_independence, "chunk_vs_whole": chunk_vs_whole}
