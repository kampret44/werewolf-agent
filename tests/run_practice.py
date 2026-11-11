#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run main.py against the mock env on the kit's practice trials and score it.

Usage: python3 tests/run_practice.py [trial ...] [--twice]
Default: trials 0, 1, 2. --twice re-runs trial 0 and checks the two runs
are identical (determinism check).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KIT = os.path.join(ROOT, "werewolf-agent-01-kit")
sys.path.insert(0, HERE)
import mock_env  # noqa: E402


def run_trial(trial_no, seed):
    with open(os.path.join(KIT, "practice", "trial-%d" % trial_no, "input", "env", "table.json")) as f:
        table = json.load(f)
    prompt = os.path.join(KIT, "practice", "trial-%d" % trial_no, "input", "prompt.txt")
    trial = mock_env.Trial(table, seed)
    server = mock_env.serve(trial)
    port = server.server_address[1]
    env = dict(os.environ)
    env["MAKEFUN_ENV_URL"] = "http://127.0.0.1:%d" % port
    env["MAKEFUN_PROMPT_FILE"] = prompt
    env["MAKEFUN_TRIAL_SEED"] = seed
    start = time.time()
    proc = subprocess.run(
        [sys.executable, "-P", "-s", os.path.join(ROOT, "main.py")],
        env=env, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start
    server.shutdown()
    results = trial.results()
    wolf = [r for r in results if r["side"] == "werewolf"]
    vil = [r for r in results if r["side"] == "village"]
    wolf_wins = sum(r["won"] for r in wolf)
    vil_wins = sum(r["won"] for r in vil)
    score = round(1000 * (wolf_wins / len(wolf) + vil_wins / len(vil)) / 2) if wolf and vil else 0
    by_role = {}
    for r in results:
        by_role.setdefault(r["role"], [0, 0])
        by_role[r["role"]][1] += 1
        by_role[r["role"]][0] += r["won"]
    return {
        "trial": trial_no, "rc": proc.returncode, "elapsed": round(elapsed, 1),
        "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()[-2000:],
        "acts": trial.acts, "illegal": trial.illegal, "max_statement": trial.max_statement,
        "finished": sum(1 for r in results if r["winner"] is not None),
        "wolf": (wolf_wins, len(wolf)), "village": (vil_wins, len(vil)),
        "by_role": by_role, "score": score,
        "signature": [(r["game"], r["role"], r["winner"]) for r in results],
    }


def main():
    args = sys.argv[1:]
    twice = "--twice" in args
    args = [a for a in args if a != "--twice"]
    trials = [int(a) for a in args] if args else [0, 1, 2]
    seed = "371c9c0e0aace432ba4ad68194bada8740441c8bc3b848df49ffee68e7e9a38a"
    all_scores = []
    sig0 = None
    for t in trials:
        r = run_trial(t, seed)
        all_scores.append(r["score"])
        print("trial %d: rc=%d elapsed=%.1fs acts=%d illegal=%s max_statement=%d chars finished=%d/30 "
              "wolf %d/%d village %d/%d by_role=%s score=%d" % (
                  r["trial"], r["rc"], r["elapsed"], r["acts"], r["illegal"],
                  r["max_statement"], r["finished"], r["wolf"][0], r["wolf"][1],
                  r["village"][0], r["village"][1], r["by_role"], r["score"]))
        print("  agent: %s" % r["stdout"])
        if r["stderr"]:
            print("  stderr: %s" % r["stderr"])
        if t == 0:
            sig0 = r["signature"]
    if twice:
        r = run_trial(0, seed)
        print("trial 0 rerun: score=%d signature-identical=%s" % (
            r["score"], r["signature"] == sig0))
    if all_scores:
        print("mean practice score (mock house bots): %d" % round(sum(all_scores) / len(all_scores)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
