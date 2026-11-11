# werewolf-agent

<!-- SPDX-License-Identifier: MIT -->

A deterministic, model-free agent for make.fun's `werewolf-agent-01`
("Build the best Werewolf agent"). Python 3.13, standard library only, no
model calls and no API key.

## Approach

The agent plays one seat in 30 games of 8-player Werewolf per trial against
seven house players, through the env at `MAKEFUN_ENV_URL`. Every decision is
a rule-based heuristic computed from the game's own view:

- **Suspicion scoring over the public log** — accusations, role claims (and
  contradictions between two claimants), vote history judged against
  later-revealed roles, silence/abstention. Known wolves (seer checks)
  always outrank suspicion; known villagers are never voted or killed.
- **Wolf:** match the partner's night vote when already cast, otherwise name
  the best kill target so the partner can match (a split vote kills nobody);
  kill claimed seers and accurate accusers first; vote with the village
  majority when the target is a villager, and split the vote to try to force
  a tie when the partner leads it.
- **Seer:** inspect the most suspicious living seat; reveal a found wolf at
  once, counter-claim a false seer, otherwise push suspicion without
  claiming the role.
- **Doctor:** protect claimed seers and accurate accusers first.
- **Villager:** vote the top suspect, voting with the plurality leader when
  that secures an exile instead of scattering into a tie; abstain rather
  than guess when there is no case.
- **Statements** are short, role-aware templates filled with evidence from
  the log (who voted for a revealed villager, who contradicted a claim),
  varied only by a seeded RNG (`MAKEFUN_TRIAL_SEED` + game number). No wall
  clock, no unseeded randomness, no state between games — a replay
  reproduces the live run exactly.

## Run

```sh
python3 -P -s main.py
```

Needs `MAKEFUN_ENV_URL` (and `MAKEFUN_TRIAL_SEED` when the trial sets one).
Exits once the env reports the trial done, printing a one-line summary
(decisions, games finished, wolf/village wins, score).

## Verify locally

The kit's real env needs the house-model lane (an Anthropic key), so
`tests/` holds a mock env implementing the same protocol and game rules over
the kit's practice deals, with heuristic stand-in house bots:

```sh
python3 tests/run_practice.py        # trials 0, 1, 2
python3 tests/run_practice.py 0 --twice   # + determinism check
```

Mock-house win rates measure protocol correctness and basic play, not the
expected score against the real claude-haiku-5-5 house players.

## License

MIT — see `LICENSE`.
