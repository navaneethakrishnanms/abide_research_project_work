"""Subject-ID overlap audit. Reads ONLY subject-ID columns / ID lists (+ file names). No labels, features,
predictions or performance of any holdout subject are read or printed. Developer-only data (the 600 development
subjects' time series) is read in part 2 for the dynamic-data inventory. Nothing is modified outside this folder."""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
R = Path(r"C:\abide\results")
OUT = R / "evaluation_independence_audit"
OUT.mkdir(exist_ok=True)


def norm(s):
    s = str(s).strip()
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def ids_txt(p):
    return [norm(l) for l in open(p) if l.strip()]


holdout = ids_txt(r"C:\abide\splits\phase2_holdout_subjects.txt")
train = ids_txt(r"C:\abide\splits\phase2_train_subjects.txt")
P871 = [norm(x) for x in pd.read_csv(r"C:\abide\abide_pcp\phenotypic_qc.csv", usecols=["SUB_ID"])["SUB_ID"]]
C780 = [norm(x) for x in pd.read_csv(R / "phase2" / "fd_filtered_subjects.csv", usecols=["subject_id"], dtype=str)["subject_id"]]
FDX = [norm(x) for x in pd.read_csv(R / "phase2" / "fd_removed_subjects.csv", usecols=["subject_id"], dtype=str)["subject_id"]]
DEV600 = [norm(x) for x in pd.read_csv(R / "fast_accuracy_experiment" / "data" / "dev_split_manifest.csv", usecols=["sid"], dtype=str)["sid"]]
P6 = [norm(x) for x in pd.read_csv(R / "phase6_strategyC_all_nonzero" / "data_checks" / "cohort_verification.csv", usecols=["subject_id"], dtype=str)["subject_id"]]
S = {k: set(v) for k, v in dict(holdout=holdout, train=train, P871=P871, C780=C780, FDX=FDX, DEV600=DEV600, P6=P6).items()}
H, T = S["holdout"], S["train"]
pool846 = H | T

facts = {
    "holdout list size / unique": (len(holdout), len(H)),
    "train list size / unique": (len(train), len(T)),
    "holdout ∩ train": len(H & T),
    "pool846 = train ∪ holdout": len(pool846),
    "pool846 ⊂ 871-subject phenotypic table": pool846 <= S["P871"],
    "871-table subjects not in pool846 (<100 timepoints filter in split script)": len(S["P871"] - pool846),
    "FD-removed list size": len(S["FDX"]),
    "871 = 780 cohort ∪ FD-removed, disjoint": (S["C780"] | S["FDX"]) == S["P871"] and not (S["C780"] & S["FDX"]),
    "Phase 6 cohort_verification IDs == 780 cohort": S["P6"] == S["C780"],
    "dev600 ⊂ train": S["DEV600"] <= T,
    "dev600 ⊂ 780 cohort": S["DEV600"] <= S["C780"],
    "dev600 == train ∩ 780 cohort": S["DEV600"] == (T & S["C780"]),
}
rows = []


def add(group, a, b, note=""):
    rows.append({"set_A": a, "set_B": b, "|A|": len(S_[a]), "|B|": len(S_[b]), "|A∩B|": len(S_[a] & S_[b]), "group": group, "note": note})


S_ = {"holdout (170)": H, "train list (676)": T, "pool846 (train+holdout)": pool846, "871 phenotypic table (Phase 1)": S["P871"],
      "780 FD-filtered cohort (Phases 3-10)": S["C780"], "FD-removed (91)": S["FDX"], "dev600 (fast/site/motion experiments)": S["DEV600"]}
for b in ["train list (676)", "pool846 (train+holdout)", "871 phenotypic table (Phase 1)", "780 FD-filtered cohort (Phases 3-10)", "FD-removed (91)", "dev600 (fast/site/motion experiments)"]:
    add("holdout vs cohort", "holdout (170)", b)
S_["780 minus train list"] = S["C780"] - T
S_["train list minus 780 (FD-removed or other)"] = T - S["C780"]
add("composition", "780 minus train list", "holdout (170)", "cohort-780 subjects not in the train list: how many are holdout")
add("composition", "train list minus 780 (FD-removed or other)", "FD-removed (91)")
add("composition", "FD-removed (91)", "pool846 (train+holdout)", "FD-removed subjects inside the 846-pool")
cnt = pd.DataFrame(rows)

# Which phases/experiments touched which cohort (from scripts, manifests and summaries)
phase_rows = [
    ("Phase 1 inventory (871)", "871 phenotypic table", len(S["P871"]), len(S["P871"] & H)),
    ("Phase 2 FD filter (871 -> 780)", "871 phenotypic table", len(S["P871"]), len(S["P871"] & H)),
    ("Phases 3-5 (AAL feasibility, FC diagnostics, missing-FC analysis)", "780 cohort", len(S["C780"]), len(S["C780"] & H)),
    ("Phase 6 (Strategy C + EN + SSAE, XAI, ROAR; 61.5%)", "780 cohort", len(S["C780"]), len(S["C780"] & H)),
    ("Phases 7-8 (synthesis, audit)", "780 cohort", len(S["C780"]), len(S["C780"] & H)),
    ("Phase 9 (accuracy optimization; 64.9%)", "780 cohort", len(S["C780"]), len(S["C780"] & H)),
    ("Phase 10 old run and corrected nested CV (63.0%)", "780 cohort", len(S["C780"]), len(S["C780"] & H)),
    ("Phase 2c-2h dev-only experiments (CC200, 67.2%)", "train list (676) via holdout_guard", len(T), len(T & H)),
    ("fast / site-centering / motion / diagnostic experiments", "dev600", len(S["DEV600"]), len(S["DEV600"] & H)),
]
ph = pd.DataFrame(phase_rows, columns=["phase_or_experiment", "subjects_used", "n_subjects", "n_overlapping_with_phase2_holdout"])
ph["share_overlapping"] = (ph.n_overlapping_with_phase2_holdout / ph.n_subjects).round(3)
out_counts = pd.concat([cnt.rename(columns={"set_A": "phase_or_set_A"}).assign(kind="set comparison"), ph.rename(columns={"phase_or_experiment": "phase_or_set_A", "subjects_used": "set_B", "n_subjects": "|B|", "n_overlapping_with_phase2_holdout": "|A∩B|"}).assign(kind="phase vs holdout")], ignore_index=True)
out_counts.to_csv(OUT / "cohort_overlap_counts.csv", index=False)

# restricted ID-level artifact: membership flags only (no labels, no features)
allids = sorted(S["P871"] | pool846 | S["C780"], key=lambda x: int(x) if x.isdigit() else x)
pd.DataFrame({"subject_id": allids, "in_871": [i in S["P871"] for i in allids], "in_pool846": [i in pool846 for i in allids], "in_holdout": [i in H for i in allids],
              "in_train_list": [i in T for i in allids], "in_780_cohort": [i in S["C780"] for i in allids], "in_fd_removed": [i in S["FDX"] for i in allids],
              "in_dev600": [i in S["DEV600"] for i in allids]}).to_csv(OUT / "RESTRICTED_id_membership.csv", index=False)

json.dump({"facts": {k: (list(v) if isinstance(v, tuple) else v) for k, v in facts.items()}, "phases": ph.to_dict(orient="records"),
           "composition": cnt[cnt.group == "composition"].to_dict(orient="records")}, open(OUT / "overlap_facts.json", "w"), indent=1, default=str)
print(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in facts.items()}, indent=1, default=str))
print(cnt.to_string(index=False))
print(ph.to_string(index=False))
