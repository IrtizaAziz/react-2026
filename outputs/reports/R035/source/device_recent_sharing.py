"""Strictly-prior timestamp-batched recent distinct-customer device sharing."""
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from .customer_history import _timestamp_ns
from .customer_relationships import _values, device_global_production
FEATURES=["device_unique_customers_24h","device_unique_customers_7d","device_other_customers_24h","device_other_customers_7d","device_7d_unique_to_lifetime_unique_ratio"]
AVAILABLE_FEATURES=tuple(FEATURES)
SOURCE_DEPENDENCIES=(__file__.replace("device_recent_sharing.py", "customer_relationships.py"),
                     __file__.replace("device_recent_sharing.py", "customer_history.py"))
DAY=np.int64(86_400_000_000_000); WEEK=np.int64(604_800_000_000_000)
def _input(t,c,d,lifetime): return _timestamp_ns(t),_values(c,"__MISSING_CUSTOMER__"),_values(d,"__MISSING_DEVICE__"),pd.Series(lifetime).to_numpy(dtype=np.int64)
def _empty(n): return {k:np.zeros(n,dtype=np.int32 if k!=FEATURES[-1] else np.float32) for k in FEATURES}
def device_recent_sharing_production(timestamp,customer_id,device_id,device_lifetime_unique_customers):
 t,c,d,lifetime=_input(timestamp,customer_id,device_id,device_lifetime_unique_customers); out=_empty(len(t)); states=[(DAY,defaultdict(deque),defaultdict(lambda:defaultdict(int)),"24h"),(WEEK,defaultdict(deque),defaultdict(lambda:defaultdict(int)),"7d")]
 starts=np.r_[0,np.flatnonzero(t[1:]!=t[:-1])+1]; ends=np.r_[starts[1:],len(t)]
 for s,e in zip(starts,ends):
  now=t[s]
  for window,events,counts,suffix in states:
   for dev in np.unique(d[s:e]):
    q=events[dev]; m=counts[dev]
    while q and q[0][0]<now-window:
     _,old=q.popleft(); m[old]-=1
     if m[old]==0: del m[old]
   unique=np.asarray([len(counts[dev]) for dev in d[s:e]],dtype=np.int32); own=np.asarray([int(cust in counts[dev]) for dev,cust in zip(d[s:e],c[s:e])],dtype=np.int32)
   out[f"device_unique_customers_{suffix}"][s:e]=unique; out[f"device_other_customers_{suffix}"][s:e]=unique-own
  out[FEATURES[-1]][s:e]=out["device_unique_customers_7d"][s:e]/np.maximum(lifetime[s:e],1)
  for _,events,counts,_ in states:
   for dev,cust in zip(d[s:e],c[s:e]): events[dev].append((now,cust)); counts[dev][cust]+=1
 return pd.DataFrame(out)
def device_recent_sharing_chunked(chunks):
 full=pd.concat([x.copy() for x in chunks]); return device_recent_sharing_production(full.timestamp,full.customer_id,full.device_id,full.device_prior_distinct_customer_count).set_axis(full.index).sort_index()
def device_recent_sharing_from_prior_stream(pt,pc,pdevice,pl,ct,cc,cdevice,cl):
 a,b=_timestamp_ns(pt),_timestamp_ns(ct)
 if len(a) and len(b) and a[-1]>=b[0]: raise ValueError("Prior stream must end strictly before current stream")
 out=device_recent_sharing_production(pd.concat([pd.Series(pt),pd.Series(ct)],ignore_index=True),pd.concat([pd.Series(pc),pd.Series(cc)],ignore_index=True),pd.concat([pd.Series(pdevice),pd.Series(cdevice)],ignore_index=True),pd.concat([pd.Series(pl),pd.Series(cl)],ignore_index=True)); return out.iloc[len(a):].reset_index(drop=True)


def build_features(raw_frame, requested_features):
    """Adapt the unchanged R015 production engine to the feature-module API."""
    try:
        requested = tuple(requested_features) if not isinstance(requested_features, str) else ()
    except TypeError:
        requested = ()
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(AVAILABLE_FEATURES):
        raise ValueError("requested_features must be a non-empty ordered subset of AVAILABLE_FEATURES")
    required = ("timestamp", "customer_id", "device_id")
    missing = [column for column in required if column not in raw_frame]
    if missing:
        raise ValueError(f"raw_frame is missing required columns: {missing}")
    # R015's denominator is the pre-existing R005 strictly-prior device history.
    # Derive it here so this module accepts the generic runner's raw event frame.
    lifetime = device_global_production(raw_frame["timestamp"], raw_frame["customer_id"],
                                        raw_frame["device_id"])["device_prior_distinct_customer_count"]
    values = device_recent_sharing_production(raw_frame["timestamp"], raw_frame["customer_id"],
                                               raw_frame["device_id"], lifetime)
    return values.loc[:, list(requested)].set_axis(raw_frame.index)


def _certification_frame():
    return pd.DataFrame({"transaction_id": list("abcdefgh"),
        "timestamp": [0, 1, 2, 2, 86_401, 604_801, 604_802, 700_000],
        "customer_id": ["a", "a", "b", "c", "a", "d", "e", "z"],
        "device_id": ["x"] * 8,
        "device_prior_distinct_customer_count": [0, 1, 1, 1, 3, 3, 4, 5],
        "fraud": [0, 1, 0, 1, 0, 1, 0, 1]})


def certification_cases():
    """Expose R015's causal guarantees to the generic certification runner."""
    def actual(frame):
        return device_recent_sharing_production(frame.timestamp, frame.customer_id, frame.device_id,
                                                 frame.device_prior_distinct_customer_count)
    def strict_past():
        values = actual(_certification_frame())
        assert values.device_unique_customers_24h.tolist()[:7] == [0, 1, 1, 1, 3, 0, 1]
        assert values.device_unique_customers_7d.tolist()[:7] == [0, 1, 1, 1, 3, 3, 4]
    def oracle():
        frame = _certification_frame(); times, customers, devices, lifetime = _input(
            frame.timestamp, frame.customer_id, frame.device_id,
            frame.device_prior_distinct_customer_count)
        expected = _empty(len(frame))
        for row, now in enumerate(times):
            prior = np.arange(row)[times[:row] < now]
            for window, suffix in ((DAY, "24h"), (WEEK, "7d")):
                recent = prior[times[prior] >= now - window]
                unique = set(customers[index] for index in recent if devices[index] == devices[row])
                expected[f"device_unique_customers_{suffix}"][row] = len(unique)
                expected[f"device_other_customers_{suffix}"][row] = len(unique - {customers[row]})
            expected[FEATURES[-1]][row] = len(set(customers[index] for index in prior
                                                    if times[index] >= now - WEEK and devices[index] == devices[row])) / max(lifetime[row], 1)
        values = actual(frame)
        expected = pd.DataFrame(expected)
        pd.testing.assert_frame_equal(values, expected)
    def equal_timestamp_isolation():
        values = actual(_certification_frame()); assert values.iloc[2].equals(values.iloc[3])
    def permutation_invariance():
        frame = _certification_frame(); values = actual(frame)
        permuted = pd.concat([frame.iloc[:2], frame.iloc[[3, 2]], frame.iloc[4:]], ignore_index=True)
        observed = actual(permuted)
        pd.testing.assert_frame_equal(pd.concat([frame.transaction_id, values], axis=1).set_index("transaction_id").sort_index(),
                                      pd.concat([permuted.transaction_id, observed], axis=1).set_index("transaction_id").sort_index())
    def duplicate_same_timestamp_pair():
        frame = _certification_frame()
        duplicated = pd.concat([frame.iloc[:2], frame.iloc[[2, 2]], frame.iloc[4:]], ignore_index=True)
        values = actual(duplicated); assert values.iloc[2].equals(values.iloc[3])
    def future_independence():
        frame = _certification_frame(); expected = actual(frame)
        pd.testing.assert_frame_equal(expected, actual(pd.concat([frame, frame.iloc[[-1]]], ignore_index=True)).iloc[:len(frame)].reset_index(drop=True))
    def label_independence():
        frame = _certification_frame(); changed = frame.assign(fraud=1-frame.fraud); pd.testing.assert_frame_equal(actual(frame), actual(changed))
    def chunk_vs_whole():
        frame = _certification_frame(); pd.testing.assert_frame_equal(actual(frame), device_recent_sharing_chunked([frame.iloc[:2], frame.iloc[2:5], frame.iloc[5:]]))
    def excluding_current_customer():
        values = actual(_certification_frame()); assert values.device_other_customers_7d.iloc[4] == 2
    return {"strict_past": strict_past, "equal_timestamp_isolation": equal_timestamp_isolation,
            "oracle": oracle, "permutation_invariance": permutation_invariance,
            "duplicate_same_timestamp_pair": duplicate_same_timestamp_pair,
            "future_independence": future_independence, "label_independence": label_independence,
            "chunk_vs_whole": chunk_vs_whole,
            "excluding_current_customer": excluding_current_customer}
