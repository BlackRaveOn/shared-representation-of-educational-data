"""Prepare Moodle event windows, participant outcomes, and source metadata."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import pandas as pd
import pyarrow


P = Path(__file__).resolve().parents[1]
PERIODS = ("2023 P2", "2024 P1", "2024 P2", "2025 P1")
SPLITS = {"train": PERIODS[:2], "calibration": PERIODS[2:3], "test": PERIODS[3:]}
ID = "Nombre completo del usuario"
COMPONENT = "Componente"
TARGET = "Status_encoded"
ENCODING = {"Not accredited": 0, "Accredited": 1}
USECOLS = [ID, "Fecha", "Hora", COMPONENT, "Nombre evento", TARGET, "Status", "Year"]


def save(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(source_zip: Path, raw_dir: Path) -> None:
    """Extract only expected input basenames; never trust archive paths."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f"Records_{p}.csv" for p in PERIODS} | {"Readme.txt"}
    with zipfile.ZipFile(source_zip) as archive:
        for name in wanted:
            matches = [i for i in archive.infolist() if Path(i.filename).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected one archive member for {name}, got {len(matches)}")
            (raw_dir / name).write_bytes(archive.read(matches[0]))


def prepare(raw_dir: Path, out: Path, source_zip: Path, provenance: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty output directory.')
    out.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)
    protocol_path = P / "configs/moodle.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf8"))
    assert protocol["period_order"] == list(PERIODS)
    assert protocol["train_periods"] == list(SPLITS["train"])
    assert protocol["observation_days"] == 56
    assert protocol["catalog_bin_days"] == [1, 7, 14, 28, 56]
    protocol_sha = sha(protocol_path)
    frames, audits, source_entries = {}, {}, []
    prior_ids: set[str] = set()
    all_ids: set[str] = set()
    source_offset = 0
    excluded_grade_columns = None
    for period in PERIODS:
        path = raw_dir / f"Records_{period}.csv"
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        grade_columns = [c for c in columns if c not in USECOLS]
        if len(grade_columns) != 13:
            raise ValueError(f"Expected 13 excluded grade columns in {path.name}")
        if excluded_grade_columns is None:
            excluded_grade_columns = grade_columns
        assert excluded_grade_columns == grade_columns
        # Numeric grade fields are not loaded at all.
        d = pd.read_csv(path, usecols=USECOLS, dtype={ID: "string", "Fecha": "string", "Hora": "string", COMPONENT: "string", "Nombre evento": "string", "Status": "string", "Year": "string"})
        if d[USECOLS].isna().any().any():
            raise ValueError(f"Missing required values in {path.name}")
        assert d["Year"].eq(period).all()
        assert d[TARGET].isin([0, 1]).all()
        assert d["Status"].map(ENCODING).notna().all()
        assert d["Status"].map(ENCODING).eq(d[TARGET]).all()
        assert d.groupby(ID)[TARGET].nunique().eq(1).all(), "Conflicting participant targets"
        format_string = "%d/%m/%Y %H:%M:%S" if period == "2023 P2" else "%Y-%m-%d %H:%M:%S"
        dt = pd.to_datetime(d["Fecha"].str.strip() + " " + d["Hora"].str.strip(), format=format_string, errors="raise")
        assert dt.notna().all()
        anchor = dt.min().normalize()
        day = (dt.dt.normalize() - anchor).dt.days
        mask = day.ge(0) & day.lt(protocol["observation_days"])
        raw_ids = set(d[ID].tolist())
        active_ids = set(d.loc[mask, ID].tolist())
        participants = d.drop_duplicates(ID).set_index(ID).loc[sorted(active_ids), [TARGET]].reset_index()
        participants = participants.rename(columns={ID: "public_id", TARGET: "y"})
        participants.insert(0, "period", period)
        participants["y"] = participants["y"].astype("uint8")
        participants["seen_in_earlier_period"] = participants["public_id"].isin(prior_ids)
        events = pd.DataFrame({
            "eid": (np.arange(len(d), dtype="uint64") + np.uint64(source_offset))[mask.to_numpy()],
            "public_id": d.loc[mask, ID].reset_index(drop=True),
            "day": day.loc[mask].astype("uint8").reset_index(drop=True),
            "component": d.loc[mask, COMPONENT].reset_index(drop=True),
        })
        dup_cols = [ID, "Fecha", "Hora", COMPONENT, "Nombre evento"]
        audits[period] = {
            "source_file": path.name,
            "source_sha256": sha(path),
            "raw_records": len(d),
            "raw_participants": len(raw_ids),
            "window_records": len(events),
            "window_participants": len(participants),
            "participants_excluded_no_window_events": len(raw_ids - active_ids),
            "raw_public_ids_seen_in_earlier_period": len(raw_ids & prior_ids),
            "window_participants_seen_in_earlier_period": int(participants["seen_in_earlier_period"].sum()),
            "source_min_timestamp": str(dt.min()),
            "source_max_timestamp": str(dt.max()),
            "window_start_inclusive": str(anchor),
            "window_end_exclusive": str(anchor + pd.Timedelta(days=protocol["observation_days"])),
            "timestamp_format": format_string,
            "timestamp_timezone": "unspecified; source timestamps treated as naive",
            "invalid_timestamps": 0,
            "missing_required_values": 0,
            "conflicting_participant_targets": 0,
            "original_row_eid_offset": source_offset,
            "identical_selected_field_rows": int(d.duplicated(dup_cols).sum()),
            "duplicate_audit_columns": dup_cols,
            "identical_selected_field_rows_in_window": int(d.loc[mask].duplicated(dup_cols).sum()),
            "first_date_records": int(day.eq(0).sum()),
            "first_date_participants": int(d.loc[day.eq(0), ID].nunique()),
        }
        source_entries.append({"file": path.name, "bytes": path.stat().st_size, "sha256": audits[period]["source_sha256"]})
        frames[period] = (events, participants)
        prior_ids.update(raw_ids)
        all_ids.update(raw_ids)
        source_offset += len(d)
    channels = sorted(set().union(*(set(frames[p][0]["component"]) for p in SPLITS["train"])))
    assert "__OTHER__" not in channels
    channels.append("__OTHER__")
    assert len(channels) <= 256
    channel_index = {name: i for i, name in enumerate(channels)}
    other_id = channel_index["__OTHER__"]
    vocabulary = {"channels": channels, "fitted_on": "training_window_only", "train_periods": list(SPLITS["train"]), "other_channel_id": other_id}
    save(out / "channel_vocabulary.json", vocabulary)
    split_audits = {}
    for split, periods in SPLITS.items():
        split_events, split_participants = [], []
        qid_offset = 0
        for period in periods:
            event_frame, participant_frame = frames[period]
            event_frame, participant_frame = event_frame.copy(), participant_frame.copy()
            participant_frame.insert(0, "qid", np.arange(qid_offset, qid_offset + len(participant_frame), dtype="uint32"))
            lookup = participant_frame.set_index("public_id")["qid"]
            event_frame["qid"] = event_frame["public_id"].map(lookup).astype("uint32")
            event_frame["channel"] = event_frame["component"].map(channel_index).fillna(other_id).astype("uint8")
            split_events.append(event_frame[["eid", "qid", "day", "channel"]])
            split_participants.append(participant_frame[["qid", "period", "public_id", "y", "seen_in_earlier_period"]])
            qid_offset += len(participant_frame)
        events = pd.concat(split_events, ignore_index=True)
        participants = pd.concat(split_participants, ignore_index=True)
        assert events["eid"].is_unique
        assert participants["qid"].is_unique
        assert participants["qid"].tolist() == list(range(len(participants)))
        assert set(events["qid"]) == set(participants["qid"])
        assert events["day"].between(0, 55).all()
        assert events["channel"].between(0, len(channels) - 1).all()
        assert [str(t) for t in events.dtypes] == ["uint64", "uint32", "uint8", "uint8"]
        event_path, participant_path = out / f"events_{split}.parquet", out / f"participants_{split}.parquet"
        events.to_parquet(event_path, index=False, engine="pyarrow", compression="snappy")
        participants.to_parquet(participant_path, index=False, engine="pyarrow", compression="snappy")
        split_audits[split] = {
            "periods": list(periods), "events": len(events), "participants": len(participants),
            "other_channel_events": int(events["channel"].eq(other_id).sum()),
            "participants_seen_in_earlier_period": int(participants["seen_in_earlier_period"].sum()),
            "new_public_id_participants": int((~participants["seen_in_earlier_period"]).sum()),
            "synthetic_deleted_events": int(events["eid"].mod(100).eq(0).sum()),
            "events_sha256": sha(event_path), "participants_sha256": sha(participant_path),
            "event_dtypes": {k: str(v) for k, v in events.dtypes.items()},
            "participant_dtypes": {k: str(v) for k, v in participants.dtypes.items()},
        }
        print(split, json.dumps(split_audits[split], ensure_ascii=False), flush=True)
    readme = raw_dir / "Readme.txt"
    source_entries.append({"file": readme.name, "bytes": readme.stat().st_size, "sha256": sha(readme)})
    source_manifest = {
        "source_url": protocol["dataset_url"], "version": 1,
        "acquisition": "Published source archive, preserved without record correction.",
        "source_zip": {"file": source_zip.name, "bytes": source_zip.stat().st_size, "sha256": sha(source_zip)},
        "extracted_files": source_entries,
    }
    save(provenance / "source_manifest.json", source_manifest)
    audit = {
        "source_url": protocol["dataset_url"], "source_zip_sha256": source_manifest["source_zip"]["sha256"],
        "protocol_sha256": protocol_sha,
        "channel_vocabulary_sha256": sha(out / "channel_vocabulary.json"),
        "target_encoding": ENCODING,
        "source_rows": source_offset,
        "raw_period_participations": sum(a["raw_participants"] for a in audits.values()),
        "distinct_public_ids": len(all_ids),
        "readme_student_count": 5717,
        "readme_count_note": "The published README count is recorded separately from counts computed from the supplied records. Public identifiers are not verified real-person identities.",
        "excluded_grade_columns": excluded_grade_columns,
        "feature_source_columns": [ID, "Fecha", "Hora", COMPONENT],
        "audit_only_source_columns": ["Nombre evento", "Status", "Year"],
        "target_column": TARGET,
        "multiplicity": "All published records are retained. Matching selected fields do not establish that events are erroneous duplicates because content identifiers and original event IDs are absent.",
        "window_anchor_limitation": protocol["window_anchor"],
        "period_label_note": "Chronology follows the observed timestamp ranges recorded for each period; period labels need not match the year of those timestamps.",
        "outcome_access": "Targets were accessed only for binary encoding and within-participation consistency checks and canonical labels; no class prevalence, fitting, or predictive quality was calculated in preparation.",
        "periods": audits, "splits": split_audits,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__},
    }
    save(out / "input_audit.json", audit)
    canonical_files = sorted(out.glob("*.parquet")) + [out / "channel_vocabulary.json", out / "input_audit.json"]
    canonical_manifest = {"files": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in canonical_files]}
    save(provenance / "canonical_manifest.json", canonical_manifest)
    save(provenance / "preparation_freeze.json", {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Preparation complete before any model fitting or predictive-quality calculation",
        "protocol_sha256": protocol_sha,
        "prepare_sha256": sha(Path(__file__)),
        "source_manifest_sha256": sha(provenance / "source_manifest.json"),
        "canonical_manifest_sha256": sha(provenance / "canonical_manifest.json"),
        "channel_vocabulary_sha256": sha(out / "channel_vocabulary.json"),
    })
    print("Frozen protocol SHA256:", protocol_sha, flush=True)
    return audit


def prepare_moodle(
    source_zip: Path | str,
    raw_dir: Path | str,
    out_dir: Path | str,
    *,
    provenance_dir: Path | str | None = None,
    extract_source: bool = True,
) -> dict:
    """Prepare Moodle windows and source metadata from the published archive."""
    source_zip, raw_dir, out_dir = Path(source_zip), Path(raw_dir), Path(out_dir)
    provenance = Path(provenance_dir) if provenance_dir is not None else out_dir / 'provenance'
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError('Use an empty output directory.')
    if provenance.exists() and any(provenance.iterdir()):
        raise FileExistsError('Use an empty provenance directory.')
    if extract_source:
        extract(source_zip, raw_dir)
    return prepare(raw_dir, out_dir, source_zip, provenance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--raw-dir", type=Path, default=P / "local_data" / "moodle")
    parser.add_argument("--out-dir", type=Path, default=P / "outputs" / "prepared_moodle")
    parser.add_argument("--provenance-dir", type=Path, default=P / "outputs" / "moodle_provenance")
    parser.add_argument("--extract", action="store_true", help="Extract expected files from the supplied ZIP before preparation")
    args = parser.parse_args()
    prepare_moodle(args.source_zip, args.raw_dir, args.out_dir,
                   provenance_dir=args.provenance_dir, extract_source=args.extract)
