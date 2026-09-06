# Launch checklist

1. Manually read Kaggle Overview, Rules, Data, and Evaluation. Confirm the official metric and worked example.
2. Download exactly `train.csv`, `test.csv`, `sample_submission.csv`, and `data_dictionary.csv` into `data/raw/`.
3. Run `python src/audit.py --full`; inspect target, fraud prevalence, sample schema, value-validated time structure, entities, drift, and leakage risks.
4. Configure the exact metric, `target=fraud`, `id_columns=[transaction_id]`, probability output, and the timestamp/order semantics. Confirm the sample remains exactly `transaction_id,fraud`.
5. Use `validation_type=time_holdout` for an initial chronological holdout or `validation_type=time` for expanding-window CV; set `shuffle=false` and choose any gap only after audit.
6. Start live experiments at `R001`, not E007. Run the simple baseline, then LightGBM/CatBoost/XGBoost with OOF artifacts.
7. Run `python src/compare.py`; inspect CV, variance, temporal coverage, and OOF diversity.
8. Use `python src/screen.py --experiment R001 --fold 0` only as a directional kill switch; full CV remains authoritative.
9. Generate submissions only after human review; confirm exact columns/order and `[0,1]` probabilities. Record the attempt in `reports/submission_log.md`.
10. Export the selected run with `python src/export_notebook.py --experiment R###`, then fresh Kaggle runtime → Run All → checks pass → Save Version → record Version ID.

## Responsibility boundary

Manual Kaggle actions: join/team setup, reading Overview/Rules/Data/Evaluation, downloading files, uploading submissions, selecting final submissions, Save Version, and organizer sharing. Codex actions: audit, validation/leakage analysis, modeling, experiments, OOF comparison, blending, local submission generation, replay checks, and report updates.

The synthetic rehearsal remains at `reports/synthetic_smoke_rehearsal.ipynb`: fresh Kaggle runtime → Run All → all checks pass → Save Version → record Version ID. Replay tolerances remain `rtol=1e-7`, `atol=1e-9` where applicable.
