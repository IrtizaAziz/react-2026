"""Mandatory causal/parity preflight for R017 merchant behavior."""
import inspect
from pathlib import Path
import unittest
import pandas as pd
from src.config import ROOT, load_config
from src.data import REACT2026_MERCHANT_HISTORY_FEATURES, load_training
from src.merchant_history import FEATURES, merchant_history_chunked, merchant_history_production, merchant_history_simple_oracle
from src.validation import make_splits
class R017MerchantHistoryTests(unittest.TestCase):
 def _frame(self): return pd.DataFrame({"transaction_id":list("abcdefg"),"timestamp":[0,1,2,2,3,86402,86403],"customer_id":["a","a","b","c","a","d","a"],"merchant_id":["m"]*7,"fraud":[0,1,0,1,0,1,0]})
 def test_strict_past_ties_permutation_future_labels_and_oracle(self):
  base=self._frame(); actual=merchant_history_production(base.timestamp,base.customer_id,base.merchant_id); self.assertEqual(list(actual),FEATURES); self.assertEqual(actual.merchant_prior_transaction_count.tolist(),[0,1,2,2,4,5,6]); self.assertEqual(actual.merchant_transactions_24h.tolist(),[0,1,2,2,4,3,2]); self.assertEqual(actual.merchant_prior_unique_customers.tolist(),[0,1,1,1,3,3,4]); self.assertEqual(actual.merchant_unique_customers_24h.tolist(),[0,1,1,1,3,3,2]); self.assertEqual(actual.merchant_prior_transaction_count.iloc[2],actual.merchant_prior_transaction_count.iloc[3]); self.assertNotIn("fraud",inspect.signature(merchant_history_production).parameters); pd.testing.assert_frame_equal(actual,merchant_history_simple_oracle(base.timestamp,base.customer_id,base.merchant_id))
  permuted=pd.concat([base.iloc[:2],base.iloc[[3,2]],base.iloc[4:]],ignore_index=True); got=merchant_history_production(permuted.timestamp,permuted.customer_id,permuted.merchant_id); pd.testing.assert_frame_equal(pd.concat([base.transaction_id,actual],axis=1).set_index("transaction_id").sort_index(),pd.concat([permuted.transaction_id,got],axis=1).set_index("transaction_id").sort_index())
  future=pd.concat([base,pd.DataFrame({"transaction_id":["z"],"timestamp":[99_999_999_999_999],"customer_id":["x"],"merchant_id":["m"],"fraud":[1]})],ignore_index=True); pd.testing.assert_frame_equal(actual,merchant_history_production(future.timestamp,future.customer_id,future.merchant_id).iloc[:len(base)].reset_index(drop=True)); pd.testing.assert_frame_equal(actual,merchant_history_chunked([base.iloc[:2],base.iloc[2:4],base.iloc[4:]]))
 def test_r013_parity_row_ids_and_saved_replay_split(self):
  root=Path(ROOT); parent,candidate=load_config(root/"config_r013.json"),load_config(root/"config_r017.json"); candidate.validate(); self.assertEqual(candidate.features[:len(parent.features)],parent.features); self.assertEqual(candidate.features[len(parent.features):],REACT2026_MERCHANT_HISTORY_FEATURES); self.assertEqual(len(candidate.features),50); self.assertEqual(candidate.model_params,parent.model_params)
  candidate_frame,fingerprint=load_training(candidate,root); parent_frame,_=load_training(parent,root); pd.testing.assert_frame_equal(candidate_frame[parent.features],parent_frame[parent.features],check_dtype=True,check_exact=True); self.assertTrue(candidate_frame.transaction_id.astype(str).equals(parent_frame.transaction_id.astype(str))); _,_,splits=make_splits(candidate_frame,candidate,fingerprint,reuse=root/candidate.splits_file); self.assertEqual(splits["signature"],"6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47")
if __name__=="__main__": unittest.main()
