import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; EXP = HERE.parent
def inv():
    d = {}
    for root in (EXP, EXP.parent / "stage1_2_b0_baseline"):
        for p in sorted(root.rglob("*")):
            if p.is_file() and HERE not in p.parents and "__pycache__" not in p.parts and p.suffix != ".pyc":
                d[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return d
if __name__ == "__main__":
    tag = sys.argv[1]; d = inv()
    if tag == "before": json.dump(d, open(HERE / "hashes_before.json", "w"), indent=0); print(len(d), "files hashed")
    else:
        b = json.load(open(HERE / "hashes_before.json")); diff = [k for k in set(b) | set(d) if b.get(k) != d.get(k)]
        json.dump(d, open(HERE / f"hashes_{tag}.json", "w"), indent=0); print(len(d), "files;", "IDENTICAL" if not diff else f"DIFF {diff[:5]}")
