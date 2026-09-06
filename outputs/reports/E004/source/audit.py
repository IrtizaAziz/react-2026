"""Descriptive audit only: never fit a model, infer test labels, or change configuration."""
if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .config import ROOT, load_config
from .data import TABLE_SUFFIXES, read_table


def sampled_table(path, limit, seed):
    """CSV/TSV/JSONL: scan rows, retain a bounded uniform priority sample."""
    suffix = path.suffix.lower()
    if limit is None or suffix not in {".csv", ".tsv", ".jsonl"}:
        frame = read_table(path)
        total = len(frame)
        return (frame.sample(n=limit, random_state=seed).sort_index() if limit and total > limit else frame), total
    chunks = (pd.read_json(path, lines=True, chunksize=50000) if suffix == ".jsonl" else
              pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", chunksize=50000))
    rng, sample, keys, total = np.random.default_rng(seed), None, np.array([]), 0
    for chunk in chunks:
        total += len(chunk)
        sample = chunk if sample is None else pd.concat([sample, chunk])
        keys = np.concatenate([keys, rng.random(len(chunk))])
        if len(sample) > limit:
            selected = np.argpartition(keys, limit - 1)[:limit]
            sample, keys = sample.iloc[selected], keys[selected]
    return (sample.sort_index() if sample is not None else read_table(path)), total


def clean_nested(frame):
    frame = frame.copy()
    for column in frame.select_dtypes(include=["object"]).columns:
        frame[column] = frame[column].map(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (dict, list)) else x)
    return frame


def id_candidates(frame):
    return [str(c) for c in frame if re.search(r"(^id$|_id$|^id_|uuid|^index$|^unnamed:|_key$)", str(c), re.I)]


def brief(value):
    return str(value).replace("\n", " ").replace("|", "\\|")[:180]


def audit(root=ROOT, *, config=None, full=False, max_rows=100000, seed=42):
    root = Path(root)
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    raw = root / "data/raw"
    paths = sorted(p for p in raw.rglob("*") if p.is_file() and p.name != ".gitkeep") if raw.exists() else []
    lines = ["# Data audit", "", f"Generated: {datetime.now(timezone.utc).isoformat()}",
             "", "Diagnostics only. No features are removed, no model is fitted, and no test labels are inferred.",
             f"Default sample cap: {max_rows}; mode: {'FULL' if full else 'bounded deterministic sample'}; seed: {seed}.",
             "Counts and distributions below refer to analyzed rows unless explicitly called exact file totals.",
             "JSON, Parquet, and Excel readers may load a complete table before sampling; CSV/TSV/JSONL use bounded samples.", "", "## File inventory"]
    tables, risks = {}, []
    for path in paths:
        lines.append(f"- {path.relative_to(root)}: {path.stat().st_size:,} bytes")
    if not paths:
        lines.append("No competition files found. Place organizer-provided files in data/raw/ after launch.")
    if config:
        for name in ("train_file", "test_file", "sample_file"):
            value = getattr(config, name)
            if value:
                path = config.path(root, value).resolve()
                if path not in [p.resolve() for p in paths]:
                    paths.append(path)
                    lines.append(f"- Explicit {name}: {path}")
    for path in paths:
        if path.suffix.lower() not in TABLE_SUFFIXES:
            lines.append(f"- {path.name}: inventory only (media/archive/unsupported format); not extracted.")
            continue
        try:
            frame, total = sampled_table(path, None if full else max_rows, seed)
            frame = clean_nested(frame)
            configured_target = getattr(config, "target", None) if config else None
            explicit_test = getattr(config, "test_file", None) if config else None
            is_test = (path.resolve() == config.path(root, explicit_test).resolve()) if explicit_test else bool(re.fullmatch(r"test|testing", path.stem, re.I))
            if is_test and configured_target and configured_target in frame:
                lines.append(f"- {path.name}: configured target column is present in the test file. Its values were excluded from all diagnostics; investigate the file role/schema before proceeding.")
                risks.append(("CRITICAL", f"{path.name}: configured target appears in test schema; do not inspect or use its values"))
                frame = frame.drop(columns=[configured_target])
            tables[path.resolve()] = frame
            sampled = len(frame) < total
            scope = "SAMPLED: duplicate counts are lower bounds; absence is not established" if sampled else "COMPLETE TABLE"
            lines += ["", f"## {path.name}", f"Exact file rows: {total:,}; columns: {len(frame.columns)}; analyzed rows: {len(frame):,}. {scope}.",
                      f"Analyzed memory: {frame.memory_usage(deep=True).sum():,} bytes.",
                      f"Columns: {brief(list(frame.columns))}", f"Candidate IDs/keys (name heuristic): {id_candidates(frame)}",
                      f"Exact duplicate rows in analyzed data: {int(frame.duplicated().sum())}",
                      "", "| Column | Type | Missing count (%) | Unique (ratio) | Signals |", "|---|---|---|---|---|"]
            for column in frame:
                series = frame[column]
                missing, unique = int(series.isna().sum()), int(series.nunique())
                ratio = unique / max(len(series), 1)
                top = series.value_counts(normalize=True, dropna=False)
                signals = []
                numeric = pd.api.types.is_numeric_dtype(series)
                if unique <= 1:
                    signals.append("constant")
                    risks.append(("LOW", f"{path.name}.{column}: constant/all-missing in analyzed rows"))
                elif len(top) and top.iloc[0] >= .99:
                    signals.append("near-constant (>=99%)")
                signals.append("likely categorical" if not numeric or unique <= 20 else "likely numeric/continuous; confirm semantics")
                if not numeric and unique > 100:
                    signals.append("high cardinality")
                if ratio > .98:
                    signals.append("near-unique; possible identifier")
                name = str(column)
                if re.search(r"date|time|timestamp|year|month", name, re.I):
                    signals.append("possible time structure")
                    risks.append(("HIGH", f"{path.name}.{column}: possible time field; random CV may train on future observations"))
                if re.search(r"user|customer|patient|device|machine|entity|session|account", name, re.I):
                    signals.append("possible entity/group")
                    risks.append(("HIGH", f"{path.name}.{column}: possible repeated entity ({len(series)-unique} repeated/nonunique rows including missing); investigate group isolation"))
                if re.search(r"lat|lon|country|city|region|postal|location|geo", name, re.I):
                    signals.append("possible geography")
                if re.search(r"sequence|order|index|unnamed|uuid|encoded", name, re.I):
                    risks.append(("MEDIUM", f"{path.name}.{column}: index/order/encoded ID may reveal collection structure"))
                lines.append(f"| {brief(column)} | {series.dtype} | {missing} ({100*missing/max(len(series),1):.2f}%) | {unique} ({ratio:.3f}) | {', '.join(signals)} |")
            patterns = frame.isna().astype(int).astype(str).agg("".join, axis=1).value_counts().head(5) if len(frame) else pd.Series(dtype=int)
            lines.append(f"\nTop missingness masks (1=missing, column order above): {brief(patterns.to_dict())}")
            candidate_features = [c for c in frame if c not in id_candidates(frame) and c != configured_target]
            if candidate_features:
                lines.append(f"Duplicate candidate-feature rows excluding named ID candidates and any configured target: {int(frame.duplicated(candidate_features).sum())}; confirm feature roles before interpretation.")
            for column in frame:
                series = frame[column]
                if column in id_candidates(frame):
                    count = int(series.duplicated(keep=False).sum())
                    lines.append(f"- {column}: rows with duplicated candidate ID, including missing: {count}")
                if pd.api.types.is_numeric_dtype(series):
                    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
                    stats = clean.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).to_dict()
                    q1, q3 = clean.quantile(.25), clean.quantile(.75)
                    extremes = int(((clean < q1 - 3*(q3-q1)) | (clean > q3 + 3*(q3-q1))).sum())
                    lines.append(f"- Numeric {column}: {stats}; skew={clean.skew()}; 3-IQR extremes={extremes}; nonfinite={int(np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).sum())}.")
                else:
                    counts = series.value_counts()
                    lines.append(f"- Categorical {column}: common={brief(counts.head(10).to_dict())}; singleton values={int((counts == 1).sum())}; rare examples={brief(counts[counts <= 2].head(10).to_dict())}.")
        except (ValueError, OSError, ImportError, TypeError) as exc:
            lines.append(f"- Could not analyze {path.name}: {type(exc).__name__}: {exc}")
    def role(name, pattern):
        explicit = getattr(config, name, None) if config else None
        if explicit:
            return tables.get(config.path(root, explicit).resolve())
        candidates = [frame for path, frame in tables.items() if re.fullmatch(pattern, path.stem, re.I)]
        if len(candidates) != 1:
            lines.append(f"{name}: {'ambiguous' if candidates else 'unavailable'}; configure its path explicitly.")
            return None
        return candidates[0]
    lines += ["", "## Roles, target, and train/test diagnostics"]
    train, test, sample = role("train_file", r"train|training"), role("test_file", r"test|testing"), role("sample_file", r"sample_submission|samplesubmission")
    if sample is not None:
        lines.append(f"Sample submission schema: {dict(sample.dtypes.astype(str))}; analyzed rows={len(sample)}; consult file section for exact total.")
    target = getattr(config, "target", None) if config else None
    if train is not None and test is not None:
        train_only, test_only = sorted(set(train)-set(test)), sorted(set(test)-set(train))
        lines += [f"Train-only columns / target candidates: {train_only}", f"Test-only columns: {test_only}"]
        if target is None and len(train_only) == 1 and sample is not None and train_only[0] in sample:
            target = train_only[0]
            lines.append(f"Strong target candidate: {target}; sole train-only column also appears in sample submission. This does not configure training.")
        common = [c for c in train if c in test and c != target]
        feature_columns = [c for c in common if c not in set(id_candidates(train) + id_candidates(test))]
        if feature_columns:
            overlap = train[feature_columns].drop_duplicates().merge(test[feature_columns].drop_duplicates(), on=feature_columns, how="inner")
            lines.append(f"Distinct matching train/test feature vectors in analyzed rows, excluding candidate IDs and target: {len(overlap)}. No labels transferred.")
            if len(overlap):
                risks.append(("HIGH", f"{len(overlap)} train/test feature vectors overlap in analyzed data; investigate legitimate duplicate structure only"))
        for c in common:
            a, b = train[c], test[c]
            lines.append(f"- {c}: missingness train={a.isna().mean():.3%}, test={b.isna().mean():.3%}.")
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                lines.append(f"  Numeric drift diagnostic: train quantiles={a.quantile([.05,.5,.95]).to_dict()}; test quantiles={b.quantile([.05,.5,.95]).to_dict()}.")
            else:
                fa, fb = a.astype("string").value_counts(normalize=True), b.astype("string").value_counts(normalize=True)
                keys = fa.index.union(fb.index)
                distance = float((fa.reindex(keys, fill_value=0)-fb.reindex(keys, fill_value=0)).abs().sum()/2)
                only_a, only_b = fa.index.difference(fb.index), fb.index.difference(fa.index)
                lines.append(f"  Category drift diagnostic: total variation={distance:.4f}; train-only={len(only_a)} {brief(list(only_a[:10]))}; test-only={len(only_b)} {brief(list(only_b[:10]))}.")
    if target and train is not None and target in train:
        y = train[target]
        lines += ["", f"### Target analysis: {target}", f"Classes/unique values: {y.nunique()}; missing labels: {y.isna().sum()}."]
        likely_class = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20
        lines.append("Likely classification; numeric codes may still represent regression. Confirm from the problem statement." if likely_class else "Likely regression or high-cardinality labels; confirm from the problem statement.")
        lines.append(f"Summary: {y.describe().to_dict()}; distribution/common labels: {brief(y.value_counts().head(20).to_dict())}; rare labels (<=2): {int((y.value_counts()<=2).sum())}.")
        feature_columns = [c for c in train if c != target and c not in id_candidates(train)]
        if feature_columns:
            duplicates = int(train.duplicated(subset=feature_columns).sum())
            conflicts = int((train.groupby(feature_columns, dropna=False, observed=True)[target].nunique() > 1).sum())
            lines.append(f"Duplicate feature rows excluding target/candidate IDs: {duplicates}; feature groups with conflicting labels: {conflicts}.")
            if duplicates:
                risks.append(("HIGH", f"{duplicates} duplicate training feature rows; random folds may share records"))
            if conflicts:
                risks.append(("HIGH", f"{conflicts} duplicate feature groups have conflicting labels; investigate noise or missing structure"))
        for c in feature_columns:
            if train[c].equals(y):
                risks.append(("CRITICAL", f"{c} exactly equals target in analyzed training rows; possible direct target copy"))
            elif pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(y):
                corr = train[c].corr(y)
                if pd.notna(corr) and abs(corr) > .995:
                    risks.append(("HIGH", f"{c}: absolute target correlation {abs(corr):.6f}; investigate post-outcome or encoded-target information"))
    else:
        lines.append("Target uncertain or unavailable. Target-dependent diagnostics were not performed.")
    lines += ["", "## POTENTIAL LEAKAGE RISKS", "Heuristics and drift signals are diagnostics, not proof. Review before using or removing features."]
    for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        lines.append(f"### {level}")
        findings = list(dict.fromkeys(reason for severity, reason in risks if severity == level))
        lines.extend(f"- {finding}" for finding in findings)
        if not findings:
            lines.append("- No signal found by these checks; this does not establish absence of leakage.")
    lines += ["", "Still requires human review: collection process, post-outcome features, time/groups/entities, duplicate policies, and fold-local preprocessing."]
    report = "\n".join(lines) + "\n"
    destination = root / "reports/data_audit.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination.rename(destination.with_name(f"data_audit_{stamp}.md"))
    destination.write_text(report, encoding="utf-8")
    print(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train", dest="train_file")
    parser.add_argument("--test", dest="test_file")
    parser.add_argument("--sample", dest="sample_file")
    parser.add_argument("--target")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("train_file", "test_file", "sample_file", "target"):
        if getattr(args, key):
            setattr(config, key, getattr(args, key))
    audit(args.root, config=config, full=args.full, max_rows=args.max_rows, seed=args.seed)


if __name__ == "__main__":
    main()
