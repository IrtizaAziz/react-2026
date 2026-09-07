"""Strictly-prior, timestamp-batched merchant--location familiarity."""

import numpy as np
import pandas as pd

from .customer_history import _timestamp_ns
from .customer_relationships import _pair_codes, _values
from .merchant_history import merchant_history_production


AVAILABLE_FEATURES = [
    "merchant_location_prior_count",
    "merchant_location_share_of_merchant_history",
]
SOURCE_DEPENDENCIES = (
    __file__.replace("merchant_location.py", "merchant_history.py"),
    __file__.replace("merchant_location.py", "customer_relationships.py"),
    __file__.replace("merchant_location.py", "customer_history.py"),
)


def _inputs(timestamp, merchant_id, location, merchant_prior_count):
    times = _timestamp_ns(timestamp)
    merchants = _values(merchant_id, "__MISSING_MERCHANT__")
    locations = _values(location, "__MISSING_LOCATION__")
    prior = pd.Series(merchant_prior_count).to_numpy(dtype=np.int64)
    if (prior < 0).any():
        raise ValueError("merchant_prior_transaction_count must be nonnegative")
    return times, merchants, locations, prior


def merchant_location_production(timestamp, merchant_id, location, merchant_prior_count):
    """Score complete timestamp batches before adding their raw pair events."""
    times, merchants, locations, merchant_prior = _inputs(timestamp, merchant_id, location, merchant_prior_count)
    merchant_codes, _ = pd.factorize(merchants, sort=False)
    location_codes, _ = pd.factorize(locations, sort=False)
    pairs = _pair_codes(merchant_codes, location_codes)
    counts = np.zeros(int(pairs.max()) + 1 if len(pairs) else 0, dtype=np.int64)
    out_count = np.zeros(len(times), dtype=np.int32)
    out_share = np.zeros(len(times), dtype=np.float32)
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        batch_pairs = pairs[start:end]
        prior = counts[batch_pairs]
        out_count[start:end] = prior
        out_share[start:end] = np.divide(prior, merchant_prior[start:end], out=np.zeros(end - start, dtype=np.float32), where=merchant_prior[start:end] > 0)
        unique, inverse = np.unique(batch_pairs, return_inverse=True)
        counts[unique] += np.bincount(inverse, minlength=len(unique))
    return pd.DataFrame({AVAILABLE_FEATURES[0]: out_count, AVAILABLE_FEATURES[1]: out_share})


def merchant_location_simple_oracle(timestamp, merchant_id, location, merchant_prior_count):
    """Direct reference implementation used only for causal certification."""
    times, merchants, locations, merchant_prior = _inputs(timestamp, merchant_id, location, merchant_prior_count)
    result = {AVAILABLE_FEATURES[0]: np.zeros(len(times), dtype=np.int32), AVAILABLE_FEATURES[1]: np.zeros(len(times), dtype=np.float32)}
    history = {}
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]; ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        for i in range(start, end):
            count = history.get((merchants[i], locations[i]), 0)
            result[AVAILABLE_FEATURES[0]][i] = count
            if merchant_prior[i]: result[AVAILABLE_FEATURES[1]][i] = count / merchant_prior[i]
        for i in range(start, end):
            key = (merchants[i], locations[i]); history[key] = history.get(key, 0) + 1
    return pd.DataFrame(result)


def build_features(raw_frame, requested_features):
    try: requested = tuple(requested_features) if not isinstance(requested_features, str) else ()
    except TypeError: requested = ()
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(AVAILABLE_FEATURES):
        raise ValueError("requested_features must be a non-empty ordered subset of AVAILABLE_FEATURES")
    required = ("timestamp", "customer_id", "merchant_id", "location")
    missing = [name for name in required if name not in raw_frame]
    if missing: raise ValueError(f"raw_frame is missing required columns: {missing}")
    merchant_history = merchant_history_production(raw_frame.timestamp, raw_frame.customer_id, raw_frame.merchant_id)
    result = merchant_location_production(raw_frame.timestamp, raw_frame.merchant_id, raw_frame.location, merchant_history.merchant_prior_transaction_count)
    return result.loc[:, list(requested)].set_axis(raw_frame.index)


def _frame():
    return pd.DataFrame({"transaction_id": list("abcdefghijkl"), "timestamp": [0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9], "customer_id": list("abcdefghijkl"), "merchant_id": ["m"] * 10 + ["x", "m"], "location": ["l", "x", "l", "l", "x", "l", "l", "l", "x", "l", "l", "l"], "fraud": [0, 1] * 6})


def certification_cases():
    def denominator(frame): return merchant_history_production(frame.timestamp, frame.customer_id, frame.merchant_id).merchant_prior_transaction_count
    def actual(frame): return merchant_location_production(frame.timestamp, frame.merchant_id, frame.location, denominator(frame))
    def oracle():
        frame = _frame(); pd.testing.assert_frame_equal(actual(frame), merchant_location_simple_oracle(frame.timestamp, frame.merchant_id, frame.location, denominator(frame)))
    def strict_past():
        values = actual(_frame())
        if values.merchant_location_prior_count.tolist() != [0, 0, 1, 1, 1, 3, 4, 4, 2, 6, 0, 7]: raise AssertionError("pair counts were not strictly past timestamp-batched counts")
        if not np.isclose(values.merchant_location_share_of_merchant_history.iloc[5], 3 / 5): raise AssertionError("merchant-normalized share is incorrect")
    def equal_timestamp_isolation():
        values = actual(_frame())
        if not values.iloc[2].equals(values.iloc[3]): raise AssertionError("tied rows saw each other")
    def permutation_invariance():
        frame = _frame(); values = actual(frame); permuted = frame.iloc[[0, 1, 3, 2, *range(4, len(frame))]].reset_index(drop=True); observed = actual(permuted)
        pd.testing.assert_frame_equal(pd.concat([frame.transaction_id, values], axis=1).set_index("transaction_id").sort_index(), pd.concat([permuted.transaction_id, observed], axis=1).set_index("transaction_id").sort_index())
    def duplicate_same_timestamp_pair():
        frame = _frame(); duplicated = pd.concat([frame.iloc[:5], frame.iloc[[5, 5]], frame.iloc[6:]], ignore_index=True); values = actual(duplicated)
        if not values.iloc[5].equals(values.iloc[6]) or values.merchant_location_prior_count.iloc[5] != 3: raise AssertionError("same-timestamp duplicate pair was not isolated")
    def future_independence():
        frame = _frame(); expected = actual(frame); future = pd.DataFrame({"timestamp": [99], "customer_id": ["z"], "merchant_id": ["m"], "location": ["l"]})
        pd.testing.assert_frame_equal(expected, actual(pd.concat([frame, future], ignore_index=True)).iloc[:len(frame)].reset_index(drop=True))
    def future_value_mutation_independence():
        frame = _frame(); changed = frame.copy(); changed.loc[10:, "location"] = "changed"; pd.testing.assert_frame_equal(actual(frame).iloc[:10], actual(changed).iloc[:10])
    def label_independence():
        frame = _frame(); pd.testing.assert_frame_equal(actual(frame), actual(frame.assign(fraud=1 - frame.fraud)))
    def chunk_vs_whole():
        frame = _frame(); whole = build_features(frame, AVAILABLE_FEATURES); chunks = pd.concat([frame.iloc[:3], frame.iloc[3:8], frame.iloc[8:]], ignore_index=True); pd.testing.assert_frame_equal(whole, build_features(chunks, AVAILABLE_FEATURES))
    def raw_frame_compatibility():
        frame = _frame().drop(columns="fraud"); result = build_features(frame, AVAILABLE_FEATURES)
        if not result.index.equals(frame.index) or list(result) != AVAILABLE_FEATURES: raise AssertionError("raw adapter did not preserve order/index")
    def requested_subset_order():
        frame = _frame(); result = build_features(frame, list(reversed(AVAILABLE_FEATURES)))
        if list(result) != list(reversed(AVAILABLE_FEATURES)): raise AssertionError("requested feature order was not preserved")
    def zero_history_behavior():
        values = actual(_frame())
        if values.iloc[0].tolist() != [0, 0.0]: raise AssertionError("zero history must produce finite zeros")
    def denominator_parity():
        frame = _frame(); existing = merchant_history_production(frame.timestamp, frame.customer_id, frame.merchant_id).merchant_prior_transaction_count
        independently = np.array([sum((frame.merchant_id.iloc[j] == frame.merchant_id.iloc[i]) and (frame.timestamp.iloc[j] < frame.timestamp.iloc[i]) for j in range(len(frame))) for i in range(len(frame))])
        np.testing.assert_array_equal(existing.to_numpy(), independently)
    return {"oracle": oracle, "strict_past": strict_past, "equal_timestamp_isolation": equal_timestamp_isolation, "permutation_invariance": permutation_invariance, "duplicate_same_timestamp_pair": duplicate_same_timestamp_pair, "future_independence": future_independence, "future_value_mutation_independence": future_value_mutation_independence, "label_independence": label_independence, "chunk_vs_whole": chunk_vs_whole, "raw_frame_compatibility": raw_frame_compatibility, "requested_subset_order": requested_subset_order, "zero_history_behavior": zero_history_behavior, "denominator_parity": denominator_parity}


def diagnostics(candidate, _oof, *, config, added_features):
    relevant = ["merchant_prior_transaction_count", "merchant_prior_unique_customers", "customer_location_prior_count", "customer_location_count_share", "customer_merchant_prior_count", "customer_merchant_share_of_customer_history"]
    return {"pearson_correlations": {added: {other: float(candidate[added].corr(candidate[other])) for other in relevant if other in candidate} for added in added_features}}
