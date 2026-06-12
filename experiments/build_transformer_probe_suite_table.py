from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"

PROBES = [
    {
        "label": "DeBERTa-v3-small",
        "manifest": "vacs_transformer_embedding_probe_deberta_local_manifest.json",
        "summary": "vacs_transformer_embedding_probe_deberta_local_summary.csv",
    },
    {
        "label": "SmolLM2-360M, maxlen 64",
        "manifest": "transformer_embedding_probe_manifest.json",
        "summary": "vacs_transformer_embedding_probe_summary.csv",
    },
    {
        "label": "SmolLM2-360M, maxlen 128",
        "manifest": "vacs_transformer_embedding_probe_smolm2_manifest.json",
        "summary": "vacs_transformer_embedding_probe_smolm2_summary.csv",
    },
    {
        "label": "Qwen2.5-0.5B",
        "manifest": "vacs_transformer_embedding_probe_qwen2_5_0_5b_local_manifest.json",
        "summary": "vacs_transformer_embedding_probe_qwen2_5_0_5b_local_summary.csv",
    },
]

METHOD_NAMES = {
    "random": "Random",
    "herding": "Herding",
    "kcenter": "K-center",
    "boundary": "Boundary",
    "kmeans": "K-means",
    "marc": "MARC",
    "vacs": "VACS",
}


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def points(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for probe in PROBES:
        manifest_path = RESULTS / probe["manifest"]
        summary_path = TABLES / probe["summary"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = read_summary(summary_path)
        vacs = next(row for row in summary if row["method"] == "vacs")
        statics = [row for row in summary if row["method"] != "vacs"]
        best_static = max(statics, key=lambda row: float(row["accuracy_mean"]))
        rows.append(
            {
                "probe": probe["label"],
                "dataset": manifest["dataset"],
                "documents": int(manifest["documents"]),
                "embedding_dim": int(manifest["embedding_dim"]),
                "vacs_accuracy": float(vacs["accuracy_mean"]),
                "best_static_method": best_static["method"],
                "best_static_accuracy": float(best_static["accuracy_mean"]),
                "gain": float(vacs["accuracy_mean"]) - float(best_static["accuracy_mean"]),
                "runtime_sec": float(manifest["runtime_sec"]),
                "embedding_runtime_sec": float(manifest["embedding_runtime_sec"]),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    fields = [
        "probe",
        "dataset",
        "documents",
        "embedding_dim",
        "vacs_accuracy",
        "best_static_method",
        "best_static_accuracy",
        "gain",
        "runtime_sec",
        "embedding_runtime_sec",
    ]
    with (TABLES / "vacs_transformer_probe_suite_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_tex(rows: list[dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Probe & Dim. & VACS & Best static & Gain \\\\",
        "\\midrule",
    ]
    for row in rows:
        best = METHOD_NAMES[str(row["best_static_method"])]
        lines.append(
            f"{row['probe']} & {row['embedding_dim']} & {pct(float(row['vacs_accuracy']))} & "
            f"{best} {pct(float(row['best_static_accuracy']))} & {points(float(row['gain']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_transformer_probe_suite_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_manifest(rows: list[dict[str, object]]) -> None:
    gains = [float(row["gain"]) for row in rows]
    manifest = {
        "probes": rows,
        "interpretation": (
            "Frozen Transformer text embeddings on full 20 Newsgroups are "
            "reported as boundary evidence. The suite contains no broad "
            "positive modern-embedding claim because gains are near zero or "
            "negative across the checked encoders."
        ),
        "probe_count": len(rows),
        "positive_probe_count": sum(1 for gain in gains if gain > 0),
        "max_gain": max(gains),
        "min_gain": min(gains),
    }
    (RESULTS / "vacs_transformer_probe_suite_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_tex(rows)
    write_manifest(rows)
    print(json.dumps({"rows": rows, "status": "TRANSFORMER_PROBE_SUITE_OK"}, indent=2))


if __name__ == "__main__":
    main()
