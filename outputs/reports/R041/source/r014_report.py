"""Post-fit, non-training R014 category/merchant relationship diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .config import ROOT
from .customer_history import customer_history_production
from .customer_merchant import customer_merchant_production
from .customer_category import customer_category_production
from .data import read_table
def main():
 root=Path(ROOT); records={k:json.loads((root/f"outputs/reports/{k}.json").read_text()) for k in ("R007","R013","R014")}; raw=read_table(root/"train.csv",id_columns=["transaction_id"])
 h=customer_history_production(raw.timestamp,raw.customer_id,raw.amount_bdt); m=customer_merchant_production(raw.timestamp,raw.customer_id,raw.merchant_id,h.customer_prior_count); c=customer_category_production(raw.timestamp,raw.customer_id,raw.merchant_category,h.customer_prior_count)
 o=pd.read_csv(root/records["R014"]["oof_path"],dtype={"transaction_id":"string"}); f=o.__fold__.eq(1).to_numpy(); y=raw.loc[f,"fraud"].to_numpy(); p=o.loc[f,"pred_1"].to_numpy(); mc=m.loc[f,"customer_merchant_is_new"].to_numpy(); cc=c.loc[f,"customer_category_is_new"].to_numpy(); count=c.loc[f,"customer_category_prior_count"].to_numpy()
 def diag(groups): return {n:{"rows":int(x.sum()),"positives":int(y[x].sum()),"prevalence":float(y[x].mean()),"average_precision":float(average_precision_score(y[x],p[x]))} for n,x in groups.items()}
 groups=diag({"new_customer_category":cc==1,"previously_seen_category":cc==0}); buckets=diag({"0":count==0,"1":count==1,"2-4":(count>=2)&(count<=4),">=5":count>=5}); cross=diag({"merchant_new_category_new":(mc==1)&(cc==1),"merchant_new_category_seen":(mc==1)&(cc==0),"merchant_seen_category_new":(mc==0)&(cc==1),"merchant_seen_category_seen":(mc==0)&(cc==0)})
 keys={"F1":("fold_scores",0),"F2":("fold_scores",1),"mean":("cv_mean",None),"early":("diagnostics","F2_early_half"),"late":("diagnostics","F2_late_half"),"July":("diagnostics","July_1_15"),"June_15_30":("diagnostics","June_15_30"),"July_1_7":("diagnostics","July_1_7"),"July_8_15":("diagnostics","July_8_15")}
 def val(r,s,k): return r[s][k] if s=="fold_scores" else r[s] if s=="cv_mean" else r[s][k]["average_precision"]
 report={"experiment_id":"R014","parent":"R013","causal_parity_preflight":"passed","missing_category_key":"__MISSING_CATEGORY__","category_history_f2":groups,"category_count_buckets_f2":buckets,"merchant_new_x_category_new_f2":cross,"deltas_vs_r013":{n:float(val(records["R014"],s,k)-val(records["R013"],s,k)) for n,(s,k) in keys.items()},"deltas_vs_r007":{n:float(val(records["R014"],s,k)-val(records["R007"],s,k)) for n,(s,k) in keys.items()},"decision":"FLAT-MIXED","comparison_basis":"identical saved validation rows; prior artifacts unmodified"}
 (root/"outputs/reports/R014/decision_report.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
