"""Synthetic-only tests of the demo plumbing. They check that validation rejects bad input and that the saved artifact loads and
returns a probability for well-formed random input. They say NOTHING about ASD classification performance."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model_defs  # noqa: F401
import validate

meta = json.loads((HERE / "baseline_lr_metadata.json").read_text(encoding="utf-8"))
hdr = meta["input"]["roi_labels_in_order"]
model = joblib.load(HERE / "baseline_lr.joblib")
rng = np.random.default_rng(0)


def make(ts, header=hdr, sep="\t"):
    lines = ["#" + (sep + "#").join(header)] if header is not None else []
    lines += [sep.join(f"{v:.6f}" for v in r) for r in ts]
    return ("\n".join(lines)).encode()


def rejects(raw, why):
    try:
        validate.parse_timeseries(raw, hdr)
    except validate.InputError as e:
        print("PASS reject:", why, "->", str(e)[:70]); return
    raise AssertionError("accepted: " + why)


good = rng.standard_normal((150, 116))
ts = validate.parse_timeseries(make(good), hdr)
X = validate.build_features(ts)
assert X.shape == (1, 6670) and np.isfinite(X).all()
iu = np.triu_indices(116, 1)
assert np.allclose(X[0], np.corrcoef(good, rowvar=False)[iu], atol=1e-5)          # feature order = training construction
p = float(model.predict_proba(X)[0, 1]); assert 0.0 <= p <= 1.0
print("PASS well-formed synthetic input -> plumbing works, score", round(p, 3), "(meaningless for ASD)")
assert model.named_steps["sel"].get_support().sum() == 2000 and model.named_steps["clf"].coef_.shape == (1, 2000)

rejects(make(good, header=None), "no header")
rejects(make(good, header=hdr[::-1]), "ROI order reversed")
rejects(make(good, header=hdr[:115]), "115 ROIs")
rejects(make(good[:50]), "only 50 timepoints")
rejects(make(good).replace(b"0.", b"x", 1), "non-numeric")
bad = good.copy(); bad[3, 5] = np.nan
rejects(make(bad), "NaN value")
const = good.copy(); const[:, :30] = 1.0
rejects(make(const), "30 constant ROIs")
rejects(b"\xff\xfe\x00", "binary")
rejects(b"", "empty")
print("ALL SYNTHETIC CHECKS PASSED")
