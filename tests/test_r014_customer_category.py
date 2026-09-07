"""Mandatory causal/parity preflight for R014 customer--category familiarity."""
import inspect
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.config import ROOT, load_config
from src.customer_category import FEATURES, customer_category_chunked, customer_category_from_prior_stream, customer_category_production
from src.data import REACT2026_CUSTOMER_CATEGORY_FEATURES, load_training
from src.validation import make_splits

class R014CustomerCategoryTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame({"transaction_id":list("abcdefg"),"timestamp":[0,1,2,2,3,4,5],"customer_id":["x","x","x","x","x","y","x"],"merchant_category":[None,None,"food","food",None,None,None],"customer_prior_count":[0,1,2,2,4,0,5],"fraud":[0,1,0,1,0,1,0]})
    def test_counts_recency_share_missing_and_label_isolation(self):
        base=self._frame(); out=customer_category_production(base.timestamp,base.customer_id,base.merchant_category,base.customer_prior_count)
        self.assertEqual(list(out),FEATURES); self.assertEqual(out.customer_category_prior_count.tolist(),[0,1,0,0,2,0,3])
        self.assertEqual(out.customer_category_is_new.tolist(),[1,0,1,1,0,1,0]); self.assertEqual(out.customer_category_seconds_since_last.iloc[1],1.)
        self.assertEqual(out.customer_category_seconds_since_last.iloc[4],2.); self.assertEqual(out.customer_category_share_of_customer_history.tolist()[:5],[0.,1.,0.,0.,.5])
        self.assertNotIn("fraud",inspect.signature(customer_category_production).parameters); self.assertNotIn("label",inspect.signature(customer_category_production).parameters)
    def test_causal_permutation_future_chunk_and_continuation(self):
        base=self._frame(); out=customer_category_production(base.timestamp,base.customer_id,base.merchant_category,base.customer_prior_count)
        # Tied food rows both see pre-t state, which is zero.
        self.assertEqual(out.customer_category_prior_count.iloc[2],0); self.assertEqual(out.customer_category_prior_count.iloc[3],0)
        permuted=pd.concat([base.iloc[:2],base.iloc[[3,2]],base.iloc[4:]],ignore_index=True); shuffled=customer_category_production(permuted.timestamp,permuted.customer_id,permuted.merchant_category,permuted.customer_prior_count)
        pd.testing.assert_frame_equal(pd.concat([base.transaction_id,out],axis=1).set_index("transaction_id").sort_index(),pd.concat([permuted.transaction_id,shuffled],axis=1).set_index("transaction_id").sort_index())
        future=pd.concat([base,pd.DataFrame({"transaction_id":["z"],"timestamp":[99],"customer_id":["x"],"merchant_category":["mutated"],"customer_prior_count":[6],"fraud":[0]})],ignore_index=True)
        pd.testing.assert_frame_equal(out,customer_category_production(future.timestamp,future.customer_id,future.merchant_category,future.customer_prior_count).iloc[:len(base)].reset_index(drop=True))
        pd.testing.assert_frame_equal(out,customer_category_chunked([base.iloc[:2],base.iloc[2:4],base.iloc[4:]]))
        continued=customer_category_from_prior_stream(base.iloc[:2].timestamp,base.iloc[:2].customer_id,base.iloc[:2].merchant_category,base.iloc[:2].customer_prior_count,base.iloc[2:].timestamp,base.iloc[2:].customer_id,base.iloc[2:].merchant_category,base.iloc[2:].customer_prior_count)
        pd.testing.assert_frame_equal(out.iloc[2:].reset_index(drop=True),continued)
    def test_r013_parity_manifest_and_split(self):
        root=Path(ROOT); parent,candidate=load_config(root/"config_r013.json"),load_config(root/"config_r014.json"); candidate.validate()
        self.assertEqual(candidate.features[:len(parent.features)],parent.features); self.assertEqual(candidate.features[len(parent.features):],REACT2026_CUSTOMER_CATEGORY_FEATURES); self.assertEqual(len(candidate.features),49)
        frame,fingerprint=load_training(candidate,root); parent_frame,_=load_training(parent,root); pd.testing.assert_frame_equal(frame[parent.features],parent_frame[parent.features],check_dtype=True,check_exact=True)
        _,_,splits=make_splits(frame,candidate,fingerprint,reuse=root/candidate.splits_file); self.assertEqual(splits["signature"],"6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47")
if __name__=="__main__": unittest.main()
