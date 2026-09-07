"""Locked final inference for R040 = .75 R029 + .25 R037; never an experiment."""
from __future__ import annotations

import hashlib, importlib, importlib.util, json, pickle, shutil, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R029_MATRIX_SHA = "ab767d22d77a44ef3514e8580d9287be720b210475037158514883c1c249771c"
R029_PRED_SHA = "0d4e94287d30817a717b071860734a5ed80bb8b1d5e88b26f2fa2f9a8aca571c"
R029_PACKAGE = "outputs/final/R029_final_refit_20260907T130951521136Z"
R037_PARAMS = {"objective":"binary", "n_estimators":800, "learning_rate":0.05,
    "num_leaves":31, "min_child_samples":100, "reg_lambda":5, "colsample_bytree":0.9,
    "subsample":0.8, "subsample_freq":1, "random_state":42, "n_jobs":8,
    "deterministic":True, "force_col_wise":True, "verbosity":-1}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def stamp(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def write_json(path, value): Path(path).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")

def load_frozen(root, report):
    source=root/f"outputs/reports/{report}/source"; name=f"_r040_{report.lower()}_{stamp().lower()}"
    temp=Path(tempfile.mkdtemp(prefix=name)); package=temp/name; shutil.copytree(source, package)
    spec=importlib.util.spec_from_file_location(name, package/"__init__.py", submodule_search_locations=[str(package)])
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module)
    return temp,name,source

def build_r029_test_features(train, test, config, pkg):
    data=importlib.import_module(f"{pkg}.data"); hist=importlib.import_module(f"{pkg}.customer_history")
    rel=importlib.import_module(f"{pkg}.customer_relationships"); velocity=importlib.import_module(f"{pkg}.velocity")
    cm=importlib.import_module(f"{pkg}.customer_merchant"); mh=importlib.import_module(f"{pkg}.merchant_history")
    new=importlib.import_module(f"{pkg}.merchant_new_customers")
    train_time=pd.to_datetime(train.timestamp, errors="raise"); test_time=pd.to_datetime(test.timestamp, errors="raise")
    if train_time.max() >= test_time.min(): raise ValueError("train must end strictly before test")
    order=np.argsort(test_time.to_numpy(dtype="datetime64[ns]"), kind="stable")
    chrono=test.iloc[order].reset_index(drop=False).rename(columns={"index":"__original_index__"})
    combined=pd.concat([train.drop(columns=[config["target"]]), chrono.drop(columns=["__original_index__"])], ignore_index=True)
    history=hist.customer_history_production(combined.timestamp,combined.customer_id,combined.amount_bdt)
    relationships=rel.customer_relationship_production(combined.timestamp,combined.customer_id,combined.device_id,combined.location)
    devices=rel.device_global_production(combined.timestamp,combined.customer_id,combined.device_id)
    speeds=velocity.velocity_production(combined.timestamp,combined.customer_id,combined.device_id)
    familiar=cm.customer_merchant_production(combined.timestamp,combined.customer_id,combined.merchant_id,history.customer_prior_count)
    merchants=mh.merchant_history_production(combined.timestamp,combined.customer_id,combined.merchant_id)
    newcomers=new.build_features(combined,["merchant_new_customer_share_24h","merchant_seconds_since_last_new_customer"])
    cut=len(train); pieces=[x.iloc[cut:].reset_index(drop=True) for x in (history,relationships,devices,speeds,familiar,merchants)]
    static=data.add_feature_profile(chrono.drop(columns=["__original_index__"]),"react2026_static").reset_index(drop=True)
    result=pd.concat([static,*pieces,newcomers.iloc[cut:].reset_index(drop=True)],axis=1)
    result.index=chrono["__original_index__"].to_numpy(); result=result.reindex(test.index)
    if list(result[config["features"]].columns)!=list(config["features"]): raise ValueError("52-feature test order mismatch")
    return result,{"implementation":"frozen_R029_final_feature_generation","train_seeded_raw_state":True,"test_processed_chronologically":True,"strict_less_than_timestamp":True,"equal_timestamp_isolation":True,"test_labels_used":False,"test_fitting":False,"timestamp_batches":int(test_time.nunique()),"tie_rows":int(test_time.duplicated(keep=False).sum()),"original_sample_order_restored":True}

def validate(frame, sample, label):
    if list(frame.columns)!=["transaction_id","fraud"] or len(frame)!=262648: raise ValueError(f"{label}: schema/row count")
    if not frame.transaction_id.astype("string").equals(sample.transaction_id.astype("string")): raise ValueError(f"{label}: sample order")
    if frame.transaction_id.duplicated().any() or frame.fraud.isna().any() or not np.isfinite(frame.fraud).all() or ((frame.fraud<0)|(frame.fraud>1)).any(): raise ValueError(f"{label}: invalid probabilities")

def main():
    started=time.perf_counter(); root=ROOT
    matrix=root/"outputs/feature_matrices/R029.pkl"; r029_pred=root/R029_PACKAGE/"test_predictions.csv"
    if sha256(matrix)!=R029_MATRIX_SHA: raise ValueError("R029 matrix SHA mismatch")
    if sha256(r029_pred)!=R029_PRED_SHA: raise ValueError("R029 frozen prediction SHA mismatch")
    r029cfg=json.loads((root/"outputs/reports/R029/config.json").read_text()); r037cfg=json.loads((root/"outputs/reports/R037/config.json").read_text())
    if r037cfg["model"]!="lightgbm" or r037cfg["model_params"]!=R037_PARAMS or r037cfg["predict_test"]: raise ValueError("immutable R037 recipe mismatch")
    if r029cfg["features"]!=r037cfg["features"] or len(r037cfg["features"])!=52: raise ValueError("R037/R029 feature manifest mismatch")
    train=pd.read_csv(root/r037cfg["train_file"],dtype={"transaction_id":"string"}); test=pd.read_csv(root/r037cfg["test_file"],dtype={"transaction_id":"string"}); sample=pd.read_csv(root/r037cfg["sample_file"],dtype={"transaction_id":"string"})
    if list(sample.columns)!=["transaction_id","fraud"] or len(test)!=262648 or not sample.transaction_id.equals(test.transaction_id): raise ValueError("raw test/sample identity mismatch")
    saved=pd.read_pickle(matrix)
    if not saved.transaction_id.astype("string").equals(train.transaction_id.astype("string")): raise ValueError("R029 matrix train IDs mismatch")
    if list(saved[r037cfg["features"]].columns)!=r037cfg["features"]: raise ValueError("R029 matrix exact feature order mismatch")
    if not saved[r037cfg["target"]].equals(train[r037cfg["target"]]): raise ValueError("R029 matrix labels mismatch")
    parity={"passed":True,"matrix_path":str(matrix.relative_to(root)),"matrix_sha256":R029_MATRIX_SHA,"rows":len(saved),"feature_count":52,"exact_feature_order":True,"transaction_ids_order":True,"labels_exact":True}
    temp29,pkg29,source29=load_frozen(root,"R029"); temp37,pkg37,source37=load_frozen(root,"R037")
    try:
        test_features, causal=build_r029_test_features(train,test,r037cfg,pkg29)
        from importlib import import_module
        Config=import_module(f"{pkg37}.config").Config; build_pipeline=import_module(f"{pkg37}.features").build_pipeline
        final_config=Config(**r037cfg); model=build_pipeline(final_config)
        if model.named_steps["model"].get_params() != {**model.named_steps["model"].get_params(), **R037_PARAMS}: raise ValueError("R037 LGBM parameter parity failed")
        model.fit(saved[r037cfg["features"]], train[r037cfg["target"]].astype(int).to_numpy())
        r037_values=np.asarray(model.predict_proba(test_features[r037cfg["features"]])[:,1],dtype=float)
        r037_frame=pd.DataFrame({"transaction_id":sample.transaction_id,"fraud":r037_values}); validate(r037_frame,sample,"R037")
        r029_raw=pd.read_csv(r029_pred,dtype={"transaction_id":"string"})
        if list(r029_raw.columns)!=["transaction_id","pred_0","pred_1"]: raise ValueError("unexpected frozen R029 prediction schema")
        r029_frame=r029_raw[["transaction_id","pred_1"]].rename(columns={"pred_1":"fraud"}); validate(r029_frame,sample,"R029")
        if set(r029_frame.transaction_id)!=set(r037_frame.transaction_id): raise ValueError("R029/R037 ID set mismatch")
        aligned=r029_frame.merge(r037_frame,on="transaction_id",how="inner",validate="one_to_one",suffixes=("_R029","_R037"))
        aligned=aligned.set_index("transaction_id").loc[sample.transaction_id].reset_index()
        if len(aligned)!=len(sample) or not aligned.transaction_id.equals(sample.transaction_id): raise ValueError("alignment restoration failed")
        blend=.75*aligned.fraud_R029.to_numpy(float)+.25*aligned.fraud_R037.to_numpy(float)
        final=pd.DataFrame({"transaction_id":sample.transaction_id,"fraud":blend}); validate(final,sample,"R040")
        package=root/"outputs/final"/f"R040_final_blend_{stamp()}"; package.mkdir(parents=True,exist_ok=False)
        shutil.copytree(source29,package/"frozen_R029_source"); shutil.copytree(source37,package/"frozen_R037_source")
        shutil.copy2(Path(__file__),package/"final_r040_submission.py")
        shutil.copy2(root/"outputs/reports/R037/config.json",package/"R037_frozen_config.json"); shutil.copy2(root/"outputs/reports/R037/model_parameters.json",package/"R037_frozen_model_parameters.json")
        write_json(package/"feature_manifest.json",{"features":r037cfg["features"],"categorical_features":r037cfg["categorical_features"],"count":52,"matches_R029":True})
        write_json(package/"R037_train_matrix_parity.json",parity); write_json(package/"test_feature_generation_provenance.json",causal)
        with (package/"R037_final_fit_model.pkl").open("xb") as f: pickle.dump(model,f,pickle.HIGHEST_PROTOCOL)
        with (package/"R037_fitted_preprocessing.pkl").open("xb") as f: pickle.dump(model.named_steps["preprocess"],f,pickle.HIGHEST_PROTOCOL)
        r037_path=package/"R037_test_predictions.csv"; r037_frame.to_csv(r037_path,index=False)
        final_path=package/"R040_blended_test_predictions.csv"; final.to_csv(final_path,index=False)
        submission=root/"submissions"/f"submission_R040_{package.name.rsplit('_',1)[-1]}.csv"; final.to_csv(submission,index=False)
        alignment={"passed":True,"rows":len(aligned),"method":"explicit transaction_id one_to_one merge then sample-order restoration","r029_id_order_matches_sample":True,"r037_id_order_matches_sample":True,"id_sets_exact":True,"duplicate_ids":False}
        write_json(package/"member_alignment_report.json",alignment); write_json(package/"fixed_blend_specification.json",{"formula":"fraud_R040 = 0.75 * fraud_R029 + 0.25 * fraud_R037","weights":{"R029":0.75,"R037":0.25},"rank_blending":False,"calibration":False,"clipping":False})
        write_json(package/"R037_prediction_metadata.json",{"rows":len(r037_frame),"columns":["transaction_id","fraud"],"test_fitting":False,"probabilities_finite":True,"probabilities_in_unit_interval":True,"sha256":sha256(r037_path)})
        write_json(package/"environment_provenance.json",{"python":sys.version,"platform":sys.platform,"lightgbm":import_module("lightgbm").__version__,"sklearn":import_module("sklearn").__version__,"matrix_sha256":R029_MATRIX_SHA,"r029_prediction_sha256":R029_PRED_SHA})
        report={"kind":"R040_final_fixed_blend_submission","r029_member":{"path":str(r029_pred.relative_to(root)),"sha256":sha256(r029_pred),"verified":sha256(r029_pred)==R029_PRED_SHA},"r037_matrix_parity":parity,"r037_final_model":{"path":str((package/"R037_final_fit_model.pkl").relative_to(root)),"sha256":sha256(package/"R037_final_fit_model.pkl")},"r037_preprocessor":{"path":str((package/"R037_fitted_preprocessing.pkl").relative_to(root)),"sha256":sha256(package/"R037_fitted_preprocessing.pkl")},"r037_test_predictions":{"path":str(r037_path.relative_to(root)),"sha256":sha256(r037_path)},"alignment":alignment,"blend":{"R029":0.75,"R037":0.25},"r040_predictions":{"path":str(final_path.relative_to(root)),"sha256":sha256(final_path)},"submission":{"path":str(submission.relative_to(root)),"sha256":sha256(submission),"rows":len(final),"columns":list(final.columns),"sample_order_exact":True,"uploaded":False},"test_fitting":False,"equal_timestamp_isolation":True,"no_historical_artifacts_overwritten":True,"launcher_sha256":sha256(package/"final_r040_submission.py"),"elapsed_seconds":time.perf_counter()-started}
        write_json(package/"final_submission_report.json",report); print(json.dumps(report,indent=2))
    finally:
        for n in list(sys.modules):
            if n in (pkg29,pkg37) or n.startswith(pkg29+".") or n.startswith(pkg37+"."): sys.modules.pop(n,None)
        shutil.rmtree(temp29,ignore_errors=True); shutil.rmtree(temp37,ignore_errors=True)

if __name__=="__main__": main()
