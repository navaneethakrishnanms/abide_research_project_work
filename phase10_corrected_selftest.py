"""Non-training fold-isolation / checkpoint self-test for phase10_corrected_nested_cv.py (no model is trained)."""
import inspect
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase10_corrected_nested_cv as m

res, notes = {}, {}


def rec(name, ok, note=""):
    res[name] = bool(ok)
    notes[name] = note
    print(("PASS " if ok else "FAIL ") + name + (f" -- {note}" if note else ""), flush=True)


fc_raw, y, site, _ = m.load_data()
rng = np.random.default_rng(0)

# 1. partitions -----------------------------------------------------------------------------
ok, msg = True, []
for track in ("ordinary", "site"):
    outer = m.make_outer_splits(track, fc_raw, y, site)
    all_te = np.concatenate([te for _, te in outer])
    ok &= len(all_te) == 780 and len(set(all_te)) == 780
    for k, (tr, te) in enumerate(outer):
        ok &= not (set(tr) & set(te)) and len(set(y[te])) == 2
        if track == "site":
            ok &= not (set(site[tr]) & set(site[te]))
        for a, b in m.make_inner_splits(track, y[tr], site[tr]):
            itr, iva = tr[a], tr[b]
            ok &= not (set(itr) & set(iva)) and set(itr) | set(iva) == set(tr) and not ((set(itr) | set(iva)) & set(te))
            if track == "site":
                ok &= not (set(site[itr]) & set(site[iva]))
    msg.append(f"{track}: test folds partition 780 subjects")
rec("partitions", ok, "; ".join(msg) + "; inner subsets disjoint from outer test; site folds share no sites")

# 2. inner preprocessing isolation ---------------------------------------------------------------
outer = m.make_outer_splits("ordinary", fc_raw, y, site)
tr, te = outer[0]
isp = m.make_inner_splits("ordinary", y[tr], site[tr])
base = m.build_inner(fc_raw, y, tr, isp)
junk = fc_raw.copy()
junk[te] = rng.uniform(-0.9, 0.9, junk[te].shape)
junk[te][rng.random(junk[te].shape) < 0.1] = np.nan
junk[tr[isp[0][1]]] = rng.uniform(-0.9, 0.9, junk[tr[isp[0][1]]].shape)   # inner-val rows of inner fold 0
pert = m.build_inner(junk, y, tr, isp)
same_train = np.array_equal(base[0]["Xtr"], pert[0]["Xtr"])
val_changed = not np.array_equal(base[0]["Xva"], pert[0]["Xva"])
# outer-test-only perturbation must change nothing in ANY inner fold
junk2 = fc_raw.copy()
junk2[te] = rng.uniform(-0.9, 0.9, junk2[te].shape)
pert2 = m.build_inner(junk2, y, tr, isp)
same_all = all(np.array_equal(a["Xtr"], b["Xtr"]) and np.array_equal(a["Xva"], b["Xva"]) for a, b in zip(base, pert2))
en = {"C": 0.03, "l1_ratio": 0.5}
sel_a = m.fit_en(base[0]["Xtr"], base[0]["ytr"], en["C"], en["l1_ratio"])
sel_b = m.fit_en(pert[0]["Xtr"], pert[0]["ytr"], en["C"], en["l1_ratio"])
rec("inner_prep_isolation", same_train and val_changed and same_all and np.array_equal(sel_a, sel_b),
    f"inner-train matrix identical when inner-val + outer-test rows are replaced by junk ({same_train}); inner-val matrix does change ({val_changed}); "
    f"outer-test junk changes nothing ({same_all}); EN selection identical ({np.array_equal(sel_a, sel_b)})")

# 3. outer preprocessing isolation ------------------------------------------------------------------
m._OUTER_CACHE.clear()
Xtr1, Xte1, s1 = m.outer_prepared("ordinary", 0, fc_raw, y, tr, te, en)
m._OUTER_CACHE.clear()
Xtr2, Xte2, s2 = m.outer_prepared("ordinary", 0, junk2, y, tr, te, en)
m._OUTER_CACHE.clear()
rec("outer_prep_isolation", np.array_equal(Xtr1, Xtr2) and np.array_equal(s1, s2) and not np.array_equal(Xte1, Xte2),
    "outer-train matrix and EN-selected features identical when outer-test rows are replaced by junk; outer-test matrix changes (sanity)")

# 4. selection code has no access to outer test --------------------------------------------------------
sel_funcs = [m.stage_EN, m.stage_ARCH, m.stage_WD, m.stage_DROPOUT, m.stage_LR, m.stage_ES, m.stage_B, m.stage_H, m.stage_CLF,
             m.prefetch_en, m.eval_candidate, m._sweep, m.pick, m.summarize, m.build_inner, m.fit_prep, m.get_en, m.fd_en]
bad_tokens = re.compile(r"\b(te_idx|X_te|Xte|y_te|yte|fc_raw|outer_prepared|final_unit)\b")
offenders = [f.__name__ for f in sel_funcs if bad_tokens.search(inspect.getsource(f)) and f.__name__ != "build_inner"]
bi_src = inspect.getsource(m.build_inner)
build_inner_ok = "te_idx" not in bi_src and "tr_idx" in bi_src
modsrc = inspect.getsource(m)
no_vote = not re.search(r"most_common|Counter|majority", inspect.getsource(m.run_fold) + inspect.getsource(m.main))
rec("selection_has_no_test_access", not offenders and build_inner_ok and no_vote,
    f"offenders={offenders}; build_inner uses only tr_idx={build_inner_ok}; no cross-fold majority vote in run_fold/main={no_vote}")

# 5. checkpoint mechanism -----------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    ok = True
    m.save_checkpoint("u1", {"x": np.float64(1.5), "n": np.int64(3)}, d)
    ok &= not list(d.glob("*.tmp")) and m.load_checkpoint("u1", d)["payload"]["n"] == 3
    (d / "u2.json").write_text('{"payload": {"a": 1}, "_complete": tru')                         # truncated
    (d / "u3.json").write_text(json.dumps({"payload": {"a": 1}, "_unit_id": "u3"}))                # no _complete
    (d / "u4.json").write_text(json.dumps({"payload": {"a": 1}, "_complete": True, "_unit_id": "other"}))  # wrong id
    (d / "u5.json").write_text(json.dumps({"payload": {}, "_complete": True, "_unit_id": "u5"}))   # empty payload
    (d / "u6.json.tmp").write_text("{}")                                                           # leftover tmp
    ok &= all(m.load_checkpoint(u, d) is None for u in ("u2", "u3", "u4", "u5", "u6"))
    calls = []
    fn = lambda: (calls.append(1) or {"v": 1})
    m.checkpointed("u7", fn, d); m.checkpointed("u7", fn, d)
    ok &= len(calls) == 1
    m.checkpointed("u2", fn, d)                                                                    # malformed -> recomputed
    ok &= len(calls) == 2 and m.load_checkpoint("u2", d) is not None
rec("checkpoint_resume", ok, "atomic write leaves no .tmp; malformed/incomplete/wrong-id/empty/tmp files are NOT treated as complete; valid units skipped on resume; malformed recomputed")

# 6. planned unit count + id uniqueness ------------------------------------------------------------
u = m.planned_units()
rec("unit_plan", len(u) == 95 and len(set(u)) == 95, f"{len(u)} planned units (ordinary 60 + site 35)")

# 7. speed optimization equivalence: parallel EN prefetch == stored serial-run checkpoint (ordinary fold 0) -----------
import time
ck = m.load_checkpoint("ord_outer0_EN")["payload"]
stored = {(r["C"], r["l1_ratio"], r["inner_fold"]): r["n_features"] for r in ck["inner"]}
key = ("ordinary", 0)
m._EN_CACHE.clear()
t0 = time.time()
m.prefetch_en(key, base, m.EN_CONFIGS)
t_par = time.time() - t0
mism = [(c, l, i) for (c, l, i), n in stored.items() if len(m._EN_CACHE[((key, i), c, l)]) != n]
serial_sel = m.fit_en(base[1]["Xtr"], base[1]["ytr"], 0.05, 0.5)
same_idx = np.array_equal(serial_sel, m._EN_CACHE[((key, 1), 0.05, 0.5)])
rec("parallel_en_equivalence", not mism and same_idx and len(stored) == 22,
    f"22 parallel EN fits reproduce the stored feature counts of the original serial run (mismatches={mism}); index arrays identical to serial fit ({same_idx}); parallel time {t_par:.0f}s vs stored stage runtime {ck['_runtime_sec']}s (includes 22 SSAE trainings)")

# 8. stage_B (parallelized) == stored serial result ------------------------------------------------------
ckb = m.load_checkpoint("ord_outer0_B")["payload"]
t0 = time.time()
newb = m.stage_B(base, key, ck["best"])
t_b = time.time() - t0
oldr = {r["stability_threshold"]: r for r in ckb["rows"]}
okb = all(abs(r["mean_inner_roc_auc_LR_probe"] - oldr[r["stability_threshold"]]["mean_inner_roc_auc_LR_probe"]) < 1e-12 and
          r["n_stable_features"] == oldr[r["stability_threshold"]]["n_stable_features"] for r in newb["rows"]) and len(newb["rows"]) == len(ckb["rows"])
rec("parallel_stageB_equivalence", okb and newb["best"] == ckb["best"], f"stage_B rerun in parallel matches stored result exactly; {t_b:.0f}s vs stored {ckb['_runtime_sec']}s")

out = {"generated_utc": datetime.now(timezone.utc).isoformat(), "tests": res, "notes": notes, "all_passed": all(res.values())}
(m.OUT / "logs" / "unit_tests.json").write_text(json.dumps(out, indent=2))
print("ALL PASSED" if out["all_passed"] else "SOME FAILED")
