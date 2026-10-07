"""Phase 1d: exact recovery of start states (incl. proprioception) for the step3b validation set (needed for the
model-error floor of proprio models). Reuses eval/phase1b_data.py's recovery replay with a val folder added."""
import json
from pathlib import Path
import multiprocessing as mp
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]


def job(j):
    import phase1b_data as pb
    pb.RUNS["step3b"][1]["val"] = ROOT / "results/program_b/step3b/val"
    return pb.recover_job(j)


def main():
    import phase1b_data as pb
    sel = json.loads((ROOT / "results/program_b/step3/selection.json").read_text())["roles"]
    jobs = []
    for chain, roles in sel.items():
        keep = {int(i): r for i, r in roles.items() if "val" in (r if isinstance(r, list) else [r])}
        if keep:
            e, s = chain.split(":")
            jobs.append(("step3b", e, int(s), keep))
    print(len(jobs), "chains", flush=True)
    diffs, fails, per = [], [], {}
    with mp.get_context("spawn").Pool(12) as pool:
        for e, s, report, err in pool.imap_unordered(job, jobs):
            if err:
                fails.append((e, s, err[-300:]))
            else:
                for k, v in report.items():
                    if k.startswith("val:"):
                        diffs.append(v); per[f"ep{e}_seed{s}_s{k.split(':')[1]}"] = v
    out = {"states": len(diffs), "exact_lt_1e-6": int(sum(d < 1e-6 for d in diffs)),
           "max_diff": max(diffs) if diffs else None, "failures": fails,
           "inexact": sorted(k for k, v in per.items() if v >= 1e-6)}
    (ROOT / "results/program_b/step3b/val_recovery.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out)[:500])


if __name__ == "__main__":
    main()
