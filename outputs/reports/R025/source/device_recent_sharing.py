"""Strictly-prior timestamp-batched recent distinct-customer device sharing."""
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from .customer_history import _timestamp_ns
from .customer_relationships import _values
FEATURES=["device_unique_customers_24h","device_unique_customers_7d","device_other_customers_24h","device_other_customers_7d","device_7d_unique_to_lifetime_unique_ratio"]
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
