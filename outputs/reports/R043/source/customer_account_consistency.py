"""Strictly-prior customer account-creation-day consistency features.

Missing ``account_age_days`` follows the project's numeric missing-value
contract: it remains missing rather than being imputed.  Such an event counts
as customer history, but contributes no implied creation day to the modal
histogram; its own three derived values are finite zeros.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from .customer_history import _timestamp_ns


AVAILABLE_FEATURES = [
    "customer_creation_day_signed_deviation",
    "customer_creation_day_abs_deviation",
    "customer_creation_day_mode_share",
]
SOURCE_DEPENDENCIES = (__file__.replace("customer_account_consistency.py", "customer_history.py"),)
_DAY_NS = np.int64(86_400 * 1_000_000_000)


def _inputs(timestamp, customer_id, account_age_days):
    times = _timestamp_ns(timestamp)
    customers = pd.Series(customer_id).astype("string").fillna("__MISSING_CUSTOMER__").to_numpy(dtype=str)
    ages = pd.to_numeric(pd.Series(account_age_days), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(ages)
    if np.any(finite & (ages != np.floor(ages))):
        raise ValueError("account_age_days must be an integer calendar-day quantity")
    implied_days = np.zeros(len(times), dtype=np.int64)
    implied_days[finite] = times[finite] // _DAY_NS - ages[finite].astype(np.int64)
    return times, customers, implied_days, finite


def _compute(timestamp, customer_id, account_age_days, state=None):
    """Score each timestamp batch from strictly earlier customer state."""
    times, customers, implied_days, valid = _inputs(timestamp, customer_id, account_age_days)
    state = {} if state is None else state
    counts = state.setdefault("customer_counts", defaultdict(int))
    histograms = state.setdefault("creation_day_histograms", defaultdict(lambda: defaultdict(int)))
    signed = np.zeros(len(times), dtype=np.int32)
    absolute = np.zeros(len(times), dtype=np.int32)
    shares = np.zeros(len(times), dtype=np.float32)
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        # A customer can appear several times in a batch, but every one sees
        # the identical, frozen strictly-prior histogram.
        modes = {}
        for customer in set(customers[start:end]):
            histogram = histograms[customer]
            if histogram:
                top = max(histogram.values())
                modes[customer] = (min(day for day, count in histogram.items() if count == top), top)
        for i in range(start, end):
            customer = customers[i]
            if valid[i] and customer in modes:
                mode_day, mode_count = modes[customer]
                deviation = int(implied_days[i] - mode_day)
                signed[i] = deviation
                absolute[i] = abs(deviation)
                shares[i] = mode_count / counts[customer]
        for i in range(start, end):
            customer = customers[i]
            counts[customer] += 1
            if valid[i]:
                histograms[customer][int(implied_days[i])] += 1
    return pd.DataFrame({
        AVAILABLE_FEATURES[0]: signed,
        AVAILABLE_FEATURES[1]: absolute,
        AVAILABLE_FEATURES[2]: shares,
    })


def customer_account_consistency_production(timestamp, customer_id, account_age_days):
    return _compute(timestamp, customer_id, account_age_days)


def customer_account_consistency_simple_oracle(timestamp, customer_id, account_age_days):
    """Deliberately direct, independent reference for causal certification."""
    times, customers, implied_days, valid = _inputs(timestamp, customer_id, account_age_days)
    result = {AVAILABLE_FEATURES[0]: np.zeros(len(times), dtype=np.int32),
              AVAILABLE_FEATURES[1]: np.zeros(len(times), dtype=np.int32),
              AVAILABLE_FEATURES[2]: np.zeros(len(times), dtype=np.float32)}
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]; ends = np.r_[starts[1:], len(times)]
    for start, end in zip(starts, ends):
        for i in range(start, end):
            prior = [j for j in range(start) if customers[j] == customers[i]]
            prior_valid = [j for j in prior if valid[j]]
            if valid[i] and prior_valid:
                frequencies = defaultdict(int)
                for j in prior_valid: frequencies[int(implied_days[j])] += 1
                top = max(frequencies.values())
                mode_day = min(day for day, count in frequencies.items() if count == top)
                deviation = int(implied_days[i] - mode_day)
                result[AVAILABLE_FEATURES[0]][i] = deviation
                result[AVAILABLE_FEATURES[1]][i] = abs(deviation)
                result[AVAILABLE_FEATURES[2]][i] = top / len(prior)
    return pd.DataFrame(result)


def customer_account_consistency_chunked(chunks):
    state, pending, outputs = {}, pd.DataFrame(), []
    for chunk in chunks:
        combined = pd.concat([pending, chunk.copy()])
        if combined.empty:
            continue
        times = _timestamp_ns(combined["timestamp"])
        ready_end = np.searchsorted(times, times[-1], side="left")
        if ready_end:
            ready = combined.iloc[:ready_end]
            outputs.append(_compute(ready.timestamp, ready.customer_id, ready.account_age_days, state).set_axis(ready.index))
        pending = combined.iloc[ready_end:]
    if not pending.empty:
        outputs.append(_compute(pending.timestamp, pending.customer_id, pending.account_age_days, state).set_axis(pending.index))
    return pd.concat(outputs).sort_index() if outputs else pd.DataFrame(columns=AVAILABLE_FEATURES)


def build_features(raw_frame, requested_features):
    try: requested = tuple(requested_features) if not isinstance(requested_features, str) else ()
    except TypeError: requested = ()
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(AVAILABLE_FEATURES):
        raise ValueError("requested_features must be a non-empty ordered subset of AVAILABLE_FEATURES")
    required = ("timestamp", "customer_id")
    missing = [column for column in required if column not in raw_frame]
    if missing: raise ValueError(f"raw_frame is missing required columns: {missing}")
    # The generic raw-frame contract may omit an optional numeric source
    # column.  Treat that exactly as an all-missing numeric column: retain
    # every row/customer count, emit finite zeros, and update no modal values.
    ages = raw_frame.account_age_days if "account_age_days" in raw_frame else pd.Series(np.nan, index=raw_frame.index)
    values = customer_account_consistency_production(raw_frame.timestamp, raw_frame.customer_id, ages)
    return values.loc[:, list(requested)].set_axis(raw_frame.index)


def _frame():
    return pd.DataFrame({"transaction_id": list("abcdefgh"),
        "timestamp": ["2026-01-10 23:30", "2026-01-11 00:15", "2026-01-12 12:00", "2026-01-13 12:00", "2026-01-13 12:00", "2026-01-14 12:00", "2026-01-15 12:00", "2026-01-16 12:00"],
        "customer_id": ["a", "a", "a", "a", "a", "a", "b", "a"],
        "account_age_days": [9, 10, 11, 11, 11, 13, 1, 15], "fraud": [0, 1, 0, 1, 0, 1, 0, 1]})


def certification_cases():
    def actual(frame): return customer_account_consistency_production(frame.timestamp, frame.customer_id, frame.account_age_days)
    def oracle():
        frame = _frame(); pd.testing.assert_frame_equal(actual(frame), customer_account_consistency_simple_oracle(frame.timestamp, frame.customer_id, frame.account_age_days))
    def strict_past():
        values = actual(_frame())
        if values.customer_creation_day_signed_deviation.tolist() != [0, 0, 0, 1, 1, 0, 0, 0]: raise AssertionError("state was not strictly past-only")
    def equal_timestamp_isolation():
        values = actual(_frame())
        if not values.iloc[3].equals(values.iloc[4]): raise AssertionError("equal-timestamp rows saw each other")
    def permutation_invariance():
        frame = _frame(); permuted = frame.iloc[[0, 1, 2, 4, 3, 5, 6, 7]].reset_index(drop=True)
        left = pd.concat([frame.transaction_id, actual(frame)], axis=1).set_index("transaction_id").sort_index()
        right = pd.concat([permuted.transaction_id, actual(permuted)], axis=1).set_index("transaction_id").sort_index()
        pd.testing.assert_frame_equal(left, right)
    def duplicate_same_timestamp_pair():
        frame = _frame(); duplicated = pd.concat([frame.iloc[:3], frame.iloc[[3, 3]], frame.iloc[4:]], ignore_index=True); values = actual(duplicated)
        if not values.iloc[3].equals(values.iloc[4]): raise AssertionError("duplicate same-timestamp rows did not see frozen state")
    def future_independence():
        frame = _frame(); future = pd.DataFrame({"timestamp": ["2026-02-01"], "customer_id": ["a"], "account_age_days": [31]})
        pd.testing.assert_frame_equal(actual(frame), actual(pd.concat([frame, future], ignore_index=True)).iloc[:len(frame)].reset_index(drop=True))
    def future_value_mutation_independence():
        frame = _frame(); changed = frame.copy(); changed.loc[5:, "account_age_days"] = 999
        pd.testing.assert_frame_equal(actual(frame).iloc[:5], actual(changed).iloc[:5])
    def label_independence():
        frame = _frame(); pd.testing.assert_frame_equal(actual(frame), actual(frame.assign(fraud=1 - frame.fraud)))
    def raw_frame_compatibility():
        frame = _frame().drop(columns="fraud"); result = build_features(frame, AVAILABLE_FEATURES)
        if not result.index.equals(frame.index): raise AssertionError("raw row identity was not preserved")
    def requested_subset_order():
        result = build_features(_frame(), [AVAILABLE_FEATURES[2], AVAILABLE_FEATURES[0]])
        if list(result) != [AVAILABLE_FEATURES[2], AVAILABLE_FEATURES[0]]: raise AssertionError("requested subset order was not preserved")
    def zero_history_behavior():
        if actual(_frame()).iloc[0].tolist() != [0, 0, 0.0]: raise AssertionError("zero history must be finite zeros")
    def modal_tie_earliest():
        frame = pd.DataFrame({"timestamp": ["2026-01-02", "2026-01-03", "2026-01-04"], "customer_id": ["x"] * 3, "account_age_days": [1, 1, 1]})
        # Prior implied days are Jan 1 and Jan 2; Jan 1 is the required tie mode.
        if actual(frame).customer_creation_day_signed_deviation.iloc[2] != 2: raise AssertionError("modal ties must select earliest calendar day")
    def mode_robustness():
        frame = pd.DataFrame({"timestamp": ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"], "customer_id": ["x"] * 4, "account_age_days": [1, 2, 0, 3]})
        if actual(frame).customer_creation_day_signed_deviation.iloc[3] != 1: raise AssertionError("one anomalous history row moved the modal invariant")
    def calendar_day_semantics():
        values = actual(_frame())
        if values.customer_creation_day_abs_deviation.iloc[1] != 0: raise AssertionError("calendar-day flooring across midnight is incorrect")
    def denominator_parity():
        frame = _frame(); values = actual(frame)
        # At row 5, the modal count is 3 and every one of five prior rows is
        # in the customer denominator, matching existing prior-count semantics.
        if not np.isclose(values.customer_creation_day_mode_share.iloc[5], 3 / 5): raise AssertionError("modal share denominator is not all strictly-prior customer rows")
    def chunk_vs_whole():
        frame = _frame(); pd.testing.assert_frame_equal(build_features(frame, AVAILABLE_FEATURES), customer_account_consistency_chunked([frame.iloc[:4], frame.iloc[4:6], frame.iloc[6:]]))
    return {"oracle": oracle, "strict_past": strict_past, "equal_timestamp_isolation": equal_timestamp_isolation,
            "permutation_invariance": permutation_invariance, "duplicate_same_timestamp_pair": duplicate_same_timestamp_pair,
            "future_independence": future_independence, "future_value_mutation_independence": future_value_mutation_independence,
            "label_independence": label_independence, "raw_frame_compatibility": raw_frame_compatibility,
            "requested_subset_order": requested_subset_order, "zero_history_behavior": zero_history_behavior,
            "modal_tie_earliest": modal_tie_earliest, "mode_robustness": mode_robustness,
            "calendar_day_semantics": calendar_day_semantics, "denominator_parity": denominator_parity, "chunk_vs_whole": chunk_vs_whole}


def diagnostics(candidate, oof, *, config, added_features):
    folds = oof["__fold__"].to_numpy()
    times = candidate[config.time_column].astype("string").to_numpy()
    labels = candidate[config.target].to_numpy()
    absolute = candidate["customer_creation_day_abs_deviation"].to_numpy()
    shares = candidate["customer_creation_day_mode_share"].to_numpy()
    windows = {"F2": folds == 1, "F2_early": (folds == 1) & (times >= "2026-05-15") & (times < "2026-06-15"),
               "F2_late_corrected": (folds == 1) & (times >= "2026-06-15") & (times < "2026-07-16"),
               "June_15_30": (folds == 1) & (times >= "2026-06-15") & (times < "2026-07-01"),
               "July_1_15": (folds == 1) & (times >= "2026-07-01") & (times < "2026-07-16"),
               "July_1_7": (folds == 1) & (times >= "2026-07-01") & (times < "2026-07-08"),
               "July_8_15": (folds == 1) & (times >= "2026-07-08") & (times < "2026-07-16")}
    result = {}
    for name, mask in windows.items():
        nonzero = mask & (absolute > 0)
        vals = absolute[nonzero]; mode_shares = shares[nonzero]
        result[name] = {"rows": int(mask.sum()), "abs_deviation_zero_fraction": float((absolute[mask] == 0).mean()),
                        "abs_deviation_nonzero_fraction": float(nonzero.sum() / mask.sum()), "abs_deviation_nonzero_positive_count": int(labels[nonzero].sum()),
                        "abs_deviation_nonzero_fraud_prevalence": float(labels[nonzero].mean()) if nonzero.any() else None,
                        "abs_deviation_nonzero_median": float(np.median(vals)) if len(vals) else None,
                        "abs_deviation_nonzero_p90": float(np.percentile(vals, 90)) if len(vals) else None,
                        "mode_share_nonzero_distribution": {"min": float(mode_shares.min()) if len(mode_shares) else None, "median": float(np.median(mode_shares)) if len(mode_shares) else None, "p90": float(np.percentile(mode_shares, 90)) if len(mode_shares) else None, "max": float(mode_shares.max()) if len(mode_shares) else None}}
    return {"standard_window_descriptive_diagnostics": result}
