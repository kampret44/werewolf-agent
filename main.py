#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""werewolf-agent-01: a deterministic, model-free Werewolf agent (Python 3.13 stdlib only).

Plays one seat in 30 games of 8-player Werewolf against seven house players
per trial, through the env at MAKEFUN_ENV_URL. No model calls, no API key:
every decision is a rule-based heuristic computed from the game's own view
(the public log, revealed roles, and the seat's private knowledge), plus a
seeded RNG (MAKEFUN_TRIAL_SEED + game number) used only for tie-breaking and
for varying statement templates. No wall clock, no unseeded randomness, no
state carried between games: the same seed and the same views give the same
actions, so a replay reproduces the live run exactly.

Approach, in short:
  * Suspicion scoring over the public log: accusations, role claims (and
    contradictions between them), vote history against later-revealed roles,
    and abstention/silence. Known wolves (seer checks) always outrank
    suspicion; known villagers are never voted or killed.
  * Wolf: coordinate the kill with the partner (match the partner's night
    vote when it is already cast; otherwise name the best target so the
    partner can match), kill claimed seers and accurate accusers first,
    and vote with the village majority whenever the target is a villager.
  * Seer: inspect the most suspicious living seat; reveal a found wolf at
    once, counter-claim a false seer, otherwise push suspicion without
    claiming.
  * Doctor: protect claimed seers and accurate accusers; never self-protect
    (the rules forbid it) and never repeat the previous night's target.
  * Villager: vote the top suspect, and vote with the plurality leader when
    that secures an exile of a suspect instead of scattering the vote.
  * Statements are short, role-aware templates filled with evidence drawn
    from the log (who voted for a revealed villager, who contradicted a
    claim, ...), so the agent argues from the game instead of chatting.

Run: python3 -P -s main.py  (needs MAKEFUN_ENV_URL; MAKEFUN_PROMPT_FILE and
MAKEFUN_TRIAL_SEED are read when present, the seed defaulting to "default").
Exit code 0 once the env reports done; a summary is printed to stdout.
"""

import hashlib
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request

# ---------------------------------------------------------------- env client

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post(base, path, body, timeout=90):
    """One POST to the env; returns (status, parsed JSON dict)."""
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _OPENER.open(req, timeout=timeout) as res:
            return res.status, json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        try:
            return err.code, json.loads(err.read().decode("utf-8"))
        except Exception:
            return err.code, {}


def _fold(text):
    """Fold a statement the env's way: line-break/tab runs become one space,
    other control characters are dropped, ends trimmed."""
    text = re.sub(r"[\t\n\v\f\r\x85\u2028\u2029]+", " ", text)
    text = "".join(ch for ch in text if ord(ch) >= 32 and ord(ch) != 127)
    return text.strip()


def _cut(text, max_chars):
    folded = _fold(text)
    if len(folded) <= max_chars:
        return folded
    return _fold("".join(list(folded)[:max_chars]))


def _rng(seed, *parts):
    """Deterministic RNG stream for one purpose: sha256(seed, parts)."""
    h = hashlib.sha256("\0".join([str(seed)] + [str(p) for p in parts]).encode("utf-8")).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


# ---------------------------------------------------------------- view analysis

_NAME_WORD = r"[A-Za-z][A-Za-z\-']*"


class Analysis:
    """Everything the agent believes about one game, derived from one view.

    Rebuilt from scratch on every decision, so no state is carried between
    decisions or games and a replay sees exactly the same beliefs.
    """

    def __init__(self, view, game_no, seed):
        self.view = view
        self.game_no = game_no
        self.seed = seed
        self.day = view.get("day", 1)
        self.phase = view.get("phase", "")
        you = view.get("you") or {}
        self.me = you.get("seat")
        self.my_name = you.get("name", "")
        self.role = you.get("role", "villager")
        partner = you.get("partner") or {}
        self.partner = partner.get("seat") if isinstance(partner, dict) else None
        self.alive = you.get("alive", True)
        self.players = {}
        self.names = {}
        for p in view.get("players") or []:
            self.players[p["seat"]] = p
            self.names[p["seat"]] = p.get("name", "seat %d" % p["seat"])
        self.name_to_seat = {n.lower(): s for s, n in self.names.items()}
        self.living = [s for s, p in self.players.items() if p.get("alive")]
        self.revealed = {s: p.get("role") for s, p in self.players.items() if p.get("role")}
        self.log = view.get("log") or []
        # Private knowledge.
        self.checked_wolf = set()
        self.checked_clear = set()
        for r in view.get("seerResults") or []:
            (self.checked_wolf if r.get("werewolf") else self.checked_clear).add(r.get("seat"))
        self.last_protect = (view.get("lastProtect") or {}).get("seat")
        self.wolf_votes = view.get("wolfVotes") or []
        # Public claims and accusations, parsed from speeches.
        self.seer_claims = {}   # seat -> [day, ...]
        self.doctor_claims = {}  # seat -> [day, ...]
        self.accusations = []   # (accuser, target, day, strong)
        self.clears_claimed = []  # (claimer, target, day)
        self.speeches = {}      # seat -> count of speeches
        self.passes = {}        # seat -> count of passes
        self.votes = []         # (day, voter, target or None)
        self._parse_log()
        # Suspicion of each living seat other than me (village perspective).
        self.suspicion = self._suspicion()
        self.known_wolves = set(self.checked_wolf)
        self.known_wolves |= {s for s, r in self.revealed.items() if r == "werewolf"}
        if self.role == "werewolf":
            self.known_wolves = {self.me, self.partner} if self.partner else {self.me}
        self.known_village = set(self.checked_clear)
        self.known_village |= {s for s, r in self.revealed.items() if r in ("seer", "doctor", "villager")}
        # Seats the village has reason to trust (accused a revealed wolf).
        self.trusted = self._trusted()

    # -- log parsing ------------------------------------------------------
    def _seat_of_mention(self, text, m):
        """Resolve a name or seat-number mention to a seat, or None."""
        if m is None:
            return None
        tok = m.group(1)
        if tok.isdigit():
            s = int(tok)
            return s if s in self.players else None
        return self.name_to_seat.get(tok.lower())

    def _parse_log(self):
        day = 1
        for e in self.log:
            kind = e.get("kind")
            day = e.get("day", day)
            if kind == "speech":
                seat = e.get("seat")
                text = e.get("text") or ""
                self.speeches[seat] = self.speeches.get(seat, 0) + 1
                self._parse_speech(seat, text, day)
            elif kind == "pass":
                seat = e.get("seat")
                self.passes[seat] = self.passes.get(seat, 0) + 1
            elif kind == "vote":
                self.votes.append((day, e.get("seat"), e.get("target")))

    def _parse_speech(self, seat, text, day):
        t = text.lower()
        if re.search(r"\bi(?:'m| am) (?:the )?seer\b|\bseer here\b|\bi checked\b|\bmy check\b", t):
            self.seer_claims.setdefault(seat, []).append(day)
        if re.search(r"\bi(?:'m| am) (?:the )?doctor\b|\bi protected\b|\bmy protection\b", t):
            self.doctor_claims.setdefault(seat, []).append(day)
        # "seat N is a werewolf / is a wolf" and "<Name> is a werewolf".
        for pat in (r"seat\s+(\d+)\s+is\s+(?:a\s+|the\s+)?werewolf",
                    r"seat\s+(\d+)\s+is\s+(?:a\s+)?wolf\b",
                    r"\b(" + _NAME_WORD + r")\s+is\s+(?:a\s+|the\s+)?werewolf",
                    r"\b(" + _NAME_WORD + r")\s+is\s+(?:a\s+)?wolf\b"):
            for m in re.finditer(pat, t):
                target = self._seat_of_mention(t, m)
                if target is not None and target != seat:
                    self.accusations.append((seat, target, day, True))
        # "I checked seat N: (not) a wolf" / "seat N is a villager / innocent".
        # A checked-wolf reveal ("I checked seat N ... they are a werewolf")
        # is a strong accusation of N, not just a vote call: the seer's
        # reveal must land as evidence, or the village never follows it.
        for m in re.finditer(r"checked\s+seat\s+(\d+)[^.]{0,80}?(?:they|he|she)\s+(?:are|is)\s+(?:a\s+|the\s+)?(?:werewolf|wolf)\b", t):
            target = self._seat_of_mention(t, m)
            if target is not None and target != seat:
                self.accusations.append((seat, target, day, True))
        for m in re.finditer(r"checked\s+(" + _NAME_WORD + r")[^.]{0,80}?(?:they|he|she)\s+(?:are|is)\s+(?:a\s+|the\s+)?(?:werewolf|wolf)\b", t):
            target = self._seat_of_mention(t, m)
            if target is not None and target != seat:
                self.accusations.append((seat, target, day, True))
        for m in re.finditer(r"checked\s+seat\s+(\d+)[^.]{0,50}?(not a wolf|not a werewolf|villager|innocent|clear)", t):
            target = self._seat_of_mention(t, m)
            if target is not None and target != seat:
                self.clears_claimed.append((seat, target, day))
        for pat in (r"seat\s+(\d+)\s+is\s+(?:a\s+)?villager",
                    r"seat\s+(\d+)\s+is\s+innocent",
                    r"seat\s+(\d+)\s+is\s+not\s+(?:a\s+)?wolf",
                    r"\b(" + _NAME_WORD + r")\s+is\s+(?:a\s+)?villager"):
            for m in re.finditer(pat, t):
                target = self._seat_of_mention(t, m)
                if target is not None and target != seat:
                    self.clears_claimed.append((seat, target, day))
        # A call to vote for a seat is a weak accusation.
        for m in re.finditer(r"vote\s+(?:for\s+)?seat\s+(\d+)", t):
            target = self._seat_of_mention(t, m)
            if target is not None and target != seat:
                self.accusations.append((seat, target, day, False))

    # -- derived beliefs --------------------------------------------------
    def _trusted(self):
        trusted = set()
        for (accuser, target, _d, _s) in self.accusations:
            if self.revealed.get(target) == "werewolf":
                trusted.add(accuser)
        for (_d, voter, target) in self.votes:
            if self.revealed.get(target) == "werewolf":
                trusted.add(voter)
        return trusted

    def _suspicion(self):
        s = {seat: 0.0 for seat in self.living if seat != self.me}
        # Accusations: each distinct accuser adds weight; a claimed seer counts double.
        seen = set()
        for (accuser, target, _d, strong) in self.accusations:
            if target not in s or (accuser, target) in seen:
                continue
            seen.add((accuser, target))
            w = 1.2 if strong else 0.5
            if accuser in self.seer_claims:
                w += 1.3
            s[target] = s.get(target, 0.0) + w
        # Two seer claims (or two doctor claims): one of them is lying.
        for claims in (self.seer_claims, self.doctor_claims):
            living_claimants = [c for c in claims if c in s]
            if len(living_claimants) >= 2:
                for c in living_claimants:
                    s[c] = s.get(c, 0.0) + 2.0
        # A claim that later proved false: the claimer is likely a wolf.
        for (claimer, target, _d) in self.clears_claimed:
            if self.revealed.get(target) == "werewolf" and claimer in s:
                s[claimer] = s.get(claimer, 0.0) + 3.0
        for (accuser, target, _d, _strong) in self.accusations:
            if self.revealed.get(target) in ("seer", "doctor", "villager") and accuser in s:
                s[accuser] = s.get(accuser, 0.0) + 1.0
        # Vote history against revealed roles.
        for (_d, voter, target) in self.votes:
            if voter not in s or target is None:
                continue
            if self.revealed.get(target) == "werewolf":
                s[voter] = s.get(voter, 0.0) - 2.5
            elif self.revealed.get(target) in ("seer", "doctor", "villager"):
                s[voter] = s.get(voter, 0.0) + 1.2
            if target is None:
                s[voter] = s.get(voter, 0.0) + 0.2
        # Silence: never spoke while others debated.
        if self.day >= 2:
            for seat in s:
                if self.speeches.get(seat, 0) == 0 and self.passes.get(seat, 0) >= 2:
                    s[seat] = s.get(seat, 0.0) + 0.5
        # A lone uncontradicted seer claim is more likely real: less suspicious.
        if len([c for c in self.seer_claims if c in self.living]) == 1:
            only = [c for c in self.seer_claims if c in self.living][0]
            if only in s:
                s[only] = s.get(only, 0.0) - 1.0
        return s

    # -- helpers ----------------------------------------------------------
    def top_suspects(self, exclude=()):
        """Living seats other than me, most suspicious first (seeded tie-break)."""
        cands = [s for s in self.living if s != self.me and s not in exclude]
        rng = _rng(self.seed, "tie", self.game_no, self.day, len(self.log))
        order = sorted(cands, key=lambda s: (-self.suspicion.get(s, 0.0), rng.random()))
        return order

    def current_votes(self):
        """Votes cast in the voting phase now under way: {seat: count}."""
        tally = {}
        # Only votes after the latest "voting" log entry belong to this phase.
        start = 0
        for i, e in enumerate(self.log):
            if e.get("kind") == "voting":
                start = i
        for e in self.log[start:]:
            if e.get("kind") == "vote" and e.get("target") is not None:
                tally[e["target"]] = tally.get(e["target"], 0) + 1
        return tally

    def evidence_reason(self, target):
        """One short evidence string for suspecting a seat, from the log."""
        name = self.names.get(target, "seat %d" % target)
        for (_d, voter, tgt) in self.votes:
            if voter == target and self.revealed.get(tgt) in ("seer", "doctor", "villager"):
                return "They voted to exile %s, who turned out to be a %s." % (
                    self.names.get(tgt, "seat %d" % tgt), self.revealed[tgt])
        if target in self.seer_claims and len([c for c in self.seer_claims if c in self.living]) >= 2:
            other = next(c for c in self.seer_claims if c in self.living and c != target)
            return "They claimed to be the seer, but %s claimed it too — one of them is lying." % (
                self.names.get(other, "seat %d" % other))
        for (claimer, tgt, _d) in self.clears_claimed:
            if claimer == target and self.revealed.get(tgt) == "werewolf":
                return "They vouched for %s, who turned out to be a werewolf." % self.names.get(tgt, "")
        accusers = sorted({a for (a, t, _d, _s) in self.accusations if t == target})
        if accusers:
            return "%s named them as a wolf." % ", ".join(self.names.get(a, "?") for a in accusers[:2])
        if self.speeches.get(target, 0) == 0 and self.day >= 2:
            return "They have said almost nothing while the village did the talking."
        return "Their votes have not lined up with what we later learned."

    def am_accused(self):
        return any(t == self.me for (_a, t, _d, _s) in self.accusations)

    def claimed_seer_alive(self):
        return [c for c in self.seer_claims if c in self.living and c != self.me]

    def have_i_revealed(self):
        """Did I already claim seer / name a checked wolf in my own speeches?"""
        for e in self.log:
            if e.get("kind") == "speech" and e.get("seat") == self.me:
                t = (e.get("text") or "").lower()
                if "i am the seer" in t or "i checked" in t:
                    return True
        return False

# ---------------------------------------------------------------- decisions

def _best_by_score(options, score, seed, *salt):
    """Highest-scoring option; seeded tie-break among (near-)tied best."""
    if not options:
        return None
    best = max(score(o) for o in options)
    tied = [o for o in options if score(o) >= best - 0.001]
    return _rng(seed, *salt).choice(tied)


def _entry_kill(a, options):
    """Entry-only wolf kill (the house never runs this; decide() calls it
    instead of decide_kill). Coordination first (match the partner's pick
    already cast tonight), then one rule the house never uses: a blocked
    kill last night means the doctor protected that target -- and the rules
    forbid the doctor from protecting the same seat two nights running. So
    if both wolves agreed on a target last night and nobody died, name it
    again tonight: the doctor cannot save it a second time, so the kill is
    guaranteed. Otherwise fall back to the shared kill scoring."""
    if not options:
        return None
    for v in a.wolf_votes:
        if v.get("night") == a.day and v.get("seat") == a.partner and v.get("target") in options:
            return v["target"]
    if a.day >= 2:
        last = [v for v in a.wolf_votes
                if v.get("night") == a.day - 1 and v.get("target") is not None]
        picks = {v["target"] for v in last}
        if len(picks) == 1:
            target = next(iter(picks))
            dawn = None
            for e in a.log:
                if e.get("kind") == "dawn" and e.get("day") == a.day - 1:
                    dawn = e
            if dawn is not None and dawn.get("died") is None and target in options:
                return target
    return decide_kill(a, options)


def decide_kill(a, options):
    """Wolf night kill. Match the partner's vote when cast; else name the
    best target so the partner can match (a split vote kills nobody)."""
    if not options:
        return None
    for v in a.wolf_votes:
        if v.get("night") == a.day and v.get("seat") == a.partner and v.get("target") in options:
            return v["target"]

    def score(t):
        s = 0.0
        if t in a.seer_claims:
            s += 10.0  # a claimed seer is the most dangerous villager alive
        if t in a.doctor_claims:
            s += 4.0
        # Target accused me or my partner: they are onto us.
        if any(tgt in (a.me, a.partner) for (_acc, tgt, _d, _s) in a.accusations if _acc == t):
            s += 4.0
        if t in a.trusted:
            s += 3.0  # they have already found a wolf once
        if a.suspicion.get(t, 0.0) >= 2.0:
            s -= 3.0  # the village may exile them for us; don't waste a kill
        elif a.suspicion.get(t, 0.0) <= 0.0:
            s += 1.0  # a trusted, vocal villager shapes the debate
        return s

    return _best_by_score(options, score, a.seed, "kill", a.game_no, a.day)


def decide_protect(a, options):
    """Doctor: protect the claimed seer first, then whoever the wolves most
    want dead (accurate accusers, trusted villagers). Never the previous
    night's target (the env forbids it; it is not in the options anyway)."""
    if not options:
        return None
    claimants = [c for c in a.claimed_seer_alive() if c in options]
    if claimants:
        return claimants[0]

    def score(t):
        s = 0.0
        if t in a.trusted:
            s += 3.0
        if any(tgt in a.known_wolves for (_acc, tgt, _d, _s) in a.accusations if _acc == t):
            s += 2.0
        if a.suspicion.get(t, 0.0) <= 0.0:
            s += 1.0
        return s

    return _best_by_score(options, score, a.seed, "protect", a.game_no, a.day)


def decide_inspect(a, options):
    """Seer: inspect the most suspicious living seat not yet checked. A
    power-role claimant is worth verifying early."""
    if not options:
        return None
    unchecked = [o for o in options if o not in a.checked_wolf and o not in a.checked_clear]
    if not unchecked:
        return None
    pool = [o for o in unchecked if o in a.seer_claims or o in a.doctor_claims]
    cands = pool if pool else unchecked

    def score(t):
        return a.suspicion.get(t, 0.0) + (2.0 if t in a.seer_claims else 0.0)

    return _best_by_score(cands, score, a.seed, "inspect", a.game_no, a.day)


def decide_vote_village(a, options):
    """Village side: a known wolf first; else the top suspect, but vote with
    the plurality leader when that secures an exile of a suspect instead of
    scattering the vote into a tie (a tie exiles nobody)."""
    if not options:
        return None
    for w in a.known_wolves:
        if w in options:
            return w
    cands = [o for o in options if o not in a.known_village]
    if not cands:
        return None
    ranked = sorted(cands, key=lambda s: -a.suspicion.get(s, 0.0))
    top = ranked[0]
    tally = a.current_votes()
    leader = max(tally, key=lambda s: (tally[s], -s)) if tally else None
    if leader is not None and leader in cands and leader != top:
        if a.suspicion.get(leader, 0.0) >= 1.0 and tally.get(leader, 0) >= tally.get(top, 0) + 1:
            return leader
    if a.suspicion.get(top, 0.0) >= 1.0:
        return top
    if leader is not None and leader in cands and a.suspicion.get(leader, 0.0) >= 0.5:
        return leader
    return None  # no case worth an exile: abstain rather than guess


def _entry_vote_village(a, options):
    """Entry-only village vote (the house never runs this; decide() calls it
    instead of decide_vote_village). Same spine -- known wolf first, then a
    real case -- plus one thing the house never does: on a weak-evidence day
    a random exile hurts the village far more than no exile (in practice
    day-1 exiles hit a villager about four times in five, power roles
    included), so when the forming leader's case is weak, or the leader is
    someone the entry knows is village, or the entry itself is leading,
    vote for the runner-up to tie the leader instead of joining the
    bandwagon. With no leader yet and no case, abstain rather than create
    a leader out of guesswork."""
    if not options:
        return None
    for w in a.known_wolves:
        if w in options:
            return w
    cands = [o for o in options if o not in a.known_village]
    if not cands:
        return None
    tally = a.current_votes()
    ranked = sorted(cands, key=lambda s: -a.suspicion.get(s, 0.0))
    top = ranked[0]
    top_score = a.suspicion.get(top, 0.0)
    # Endgame: when one more night kill brings parity (non-wolves left at
    # most one more than wolves alive), a no-exile day loses outright, so
    # the tie-forcing and abstaining below are switched off -- vote the
    # best suspect available, however weak the case.
    dead_wolves = sum(1 for r in a.revealed.values() if r == "werewolf")
    wolves_alive = 2 - dead_wolves
    non_wolves = len(a.living) - wolves_alive
    if non_wolves <= wolves_alive + 1:
        return top
    leader = max(tally, key=lambda s: (tally[s], -s)) if tally else None

    def runner_up():
        alts = [o for o in cands if o != leader]
        if not alts:
            return None
        return max(alts, key=lambda s: (tally.get(s, 0), a.suspicion.get(s, 0.0), -s))

    if leader is not None:
        if leader == a.me:
            return runner_up()  # tie myself out of the lead, or abstain
        if leader in a.known_village:
            return runner_up()  # never help exile someone known innocent
        if leader in cands and a.suspicion.get(leader, 0.0) < 1.0:
            return runner_up()  # weak case: tie the leader, don't crown them
    if top_score >= 1.0:
        if (leader is not None and leader in cands and leader != top
                and a.suspicion.get(leader, 0.0) >= 1.0
                and tally.get(leader, 0) >= tally.get(top, 0) + 1):
            return leader  # consolidate a plausible exile already forming
        return top
    return None  # no case worth an exile: abstain rather than guess


def _entry_vote_wolf(a, options):
    """Entry-only wolf vote (the house never runs this; decide() calls it
    instead of decide_vote_wolf). Same spine -- join a villager leader,
    split when the partner leads -- plus one thing the house never does:
    pivotal voting. Every vote for a villager who is then exiled costs the
    voter suspicion, while an abstention costs nothing, so when a villager
    leader's exile is already assured without the entry's vote (their lead
    over the runner-up beats the number of voters still to come), abstain
    and let the village exile them on its own votes. Vote only when the
    entry's ballot can still change the outcome."""
    if not options:
        return None
    villagers = [o for o in options if o != a.partner]
    if not villagers:
        return None
    tally = a.current_votes()
    # Votes already cast in this voting phase (abstentions included).
    start = 0
    for i, e in enumerate(a.log):
        if e.get("kind") == "voting":
            start = i
    cast = sum(1 for e in a.log[start:] if e.get("kind") == "vote")
    remaining = len(a.living) - cast - 1  # voters still to come after me
    leader = max(tally, key=lambda s: (tally[s], -s)) if tally else None
    if leader == a.partner:
        # Partner is being exiled: split onto the best villager alternative.
        others = [v for v in villagers if tally.get(v, 0) > 0]
        pool = others if others else villagers
        return _best_by_score(pool, lambda t: a.suspicion.get(t, 0.0), a.seed,
                              "wolfvote", a.game_no, a.day, len(a.log))
    if leader is not None and leader in villagers:
        runner_votes = max([v for s, v in tally.items() if s != leader], default=0)
        if tally[leader] - runner_votes > remaining:
            return None  # exile assured without me: abstain, dodge the penalty
        return leader
    return _best_by_score(villagers, lambda t: a.suspicion.get(t, 0.0), a.seed,
                          "wolfvote", a.game_no, a.day, len(a.log))


def decide_vote_wolf(a, options):
    """Wolf: vote with the village when the target is a villager. If the
    partner leads the vote, vote elsewhere to try to force a tie instead."""
    if not options:
        return None
    villagers = [o for o in options if o != a.partner]
    if not villagers:
        return None
    tally = a.current_votes()
    leader = max(tally, key=lambda s: (tally[s], -s)) if tally else None
    if leader == a.partner:
        # Partner is being exiled: split onto the best villager alternative.
        others = [v for v in villagers if tally.get(v, 0) > 0]
        pool = others if others else villagers
        return _best_by_score(pool, lambda t: a.suspicion.get(t, 0.0), a.seed,
                              "wolfvote", a.game_no, a.day, len(a.log))
    if leader is not None and leader in villagers:
        return leader
    return _best_by_score(villagers, lambda t: a.suspicion.get(t, 0.0), a.seed,
                          "wolfvote", a.game_no, a.day, len(a.log))


# ---------------------------------------------------------------- statements

def _pick(rng, templates):
    return templates[rng.randint(0, len(templates) - 1)]


def _entry_speak(a, max_chars):
    """Entry-only speak gate (the house never runs this; decide() calls it
    instead of speak()). A baseless accusation is worse than silence: the
    village's own suspicion rules punish whoever accused a later-revealed
    villager, so a day-one guess usually paints the accuser and misleads
    the table. When the entry has no real case -- no checked wolf, no false
    seer to counter, no accusation to answer, and the top suspect scores
    under 1.0 -- say plainly that there is no case yet instead of naming
    one. Reveals, counter-claims and defences still go through speak()."""
    suspects = a.top_suspects()
    top = suspects[0] if suspects else None
    top_score = a.suspicion.get(top, 0.0) if top is not None else 0.0
    has_wolf = any(w in a.living for w in a.checked_wolf)
    # Only the seer needs the counter-claim exception: for other village
    # roles somebody else's seer claim is no reason to accuse on a guess.
    false_claim = a.role == "seer" and bool(a.claimed_seer_alive())
    if (a.role != "werewolf" and not has_wolf and not false_claim
            and not a.am_accused() and top_score < 1.0):
        rng = _rng(a.seed, "speak-neutral", a.game_no, a.day, len(a.log))
        templates = [
            "Day %d, and I have no solid read yet. I would rather hear everyone than push a name on guesswork." % a.day,
            "Little to go on so far. I am listening, and I will vote on what the day actually shows.",
            "No strong case yet today. Let us hear the whole table before anyone's name sticks.",
            "Nothing has convinced me yet. I will keep watching and vote on evidence, not noise.",
        ]
        return _cut(_pick(rng, templates), max_chars)
    if a.role == "werewolf":
        return _entry_speak_wolf(a, max_chars)
    return speak(a, max_chars)


def speak(a, max_chars):
    """One debate statement (or None to pass), role-aware and evidence-based."""
    rng = _rng(a.seed, "speak", a.game_no, a.day, len(a.log))
    suspects = a.top_suspects()
    top = suspects[0] if suspects else None
    # Pass sometimes on a no-information first day, and rarely otherwise.
    if a.day == 1 and rng.random() < 0.15:
        return None
    if rng.random() < 0.05:
        return None

    if a.role == "werewolf":
        return _speak_wolf(a, rng, top, max_chars)
    if a.role == "seer":
        return _speak_seer(a, rng, top, max_chars)
    return _speak_villager(a, rng, top, max_chars)


def _entry_speak_wolf(a, max_chars):
    """Entry-only wolf speak: the same voice as _speak_wolf, but it never
    names the partner as a suspect -- not as the accusation target, and not
    even inside a 'before we settle on X' deflection, which only puts the
    partner's name in the frame as a candidate for exile. The partner is
    defended by redirecting at the best non-partner suspect, never by
    discussing them. (The shared _speak_wolf can bus the partner whenever
    the village's own suspicion puts them on top; the entry must not.)"""
    rng = _rng(a.seed, "speak", a.game_no, a.day, len(a.log))
    if a.day == 1 and rng.random() < 0.15:
        return None
    if rng.random() < 0.05:
        return None
    suspects = [s for s in a.top_suspects() if s != a.partner]
    top = suspects[0] if suspects else None
    if a.am_accused() and top is not None:
        text = ("I am a villager, and my votes have followed the evidence. "
                "Look at %s instead: %s" % (a.names[top], a.evidence_reason(top)))
        return _cut(text, max_chars)
    if top is None:
        return _cut("Not much to go on yet. I will listen and vote on what the day shows.", max_chars)
    if a.partner is not None and a.suspicion.get(a.partner, 0.0) >= 2.0:
        templates = [
            "%s worries me more than the names so far: %s" % (
                a.names[top], a.evidence_reason(top)),
            "Before we settle on anyone, look at %s. %s" % (
                a.names[top], a.evidence_reason(top)),
        ]
        return _cut(_pick(rng, templates), max_chars)
    reason = a.evidence_reason(top)
    name = a.names[top]
    templates = [
        "Seat %d (%s) is where my vote is going. %s" % (top, name, reason),
        "I keep coming back to seat %d (%s). %s I am open to another name, but that is my read." % (top, name, reason),
        "Seat %d (%s) has not added up for me. %s" % (top, name, reason),
        "I agree with the case on seat %d (%s). %s" % (top, name, reason),
        "Seat %d, why did you vote the way you did yesterday? %s" % (top, reason),
        "The night kill points at someone who wanted the village weaker. Seat %d (%s) fits that best: %s" % (top, name, reason),
    ]
    if a.day == 1:
        templates.append(
            "First day, little to go on. I will watch seat %d (%s) for now: %s" % (top, name, reason))
    return _cut(_pick(rng, templates), max_chars)


def _speak_wolf(a, rng, top, max_chars):
    # Accused? Defend once, briefly, and redirect at the top suspect.
    if a.am_accused() and top is not None:
        text = ("I am a villager, and my votes have followed the evidence. "
                "Look at %s instead: %s" % (a.names[top], a.evidence_reason(top)))
        return _cut(text, max_chars)
    if top is None:
        return _cut("Not much to go on yet. I will listen and vote on what the day shows.", max_chars)
    # Partner under suspicion: gently deflect, never defend them loudly.
    if a.partner is not None and a.suspicion.get(a.partner, 0.0) >= 2.0:
        others = [s for s in a.top_suspects() if s != a.partner]
        alt = others[0] if others else top
        templates = [
            "I am not convinced on %s yet. %s worries me more: %s" % (
                a.names[a.partner], a.names[alt], a.evidence_reason(alt)),
            "Before we settle on %s, look at %s. %s" % (
                a.names[a.partner], a.names[alt], a.evidence_reason(alt)),
        ]
        return _cut(_pick(rng, templates), max_chars)
    reason = a.evidence_reason(top)
    name = a.names[top]
    templates = [
        "Seat %d (%s) is where my vote is going. %s" % (top, name, reason),
        "I keep coming back to seat %d (%s). %s I am open to another name, but that is my read." % (top, name, reason),
        "Seat %d (%s) has not added up for me. %s" % (top, name, reason),
        "I agree with the case on seat %d (%s). %s" % (top, name, reason),
        "Seat %d, why did you vote the way you did yesterday? %s" % (top, reason),
        "The night kill points at someone who wanted the village weaker. Seat %d (%s) fits that best: %s" % (top, name, reason),
    ]
    if a.day == 1:
        templates.append(
            "First day, little to go on. I will watch seat %d (%s) for now: %s" % (top, name, reason))
    return _cut(_pick(rng, templates), max_chars)


def _speak_seer(a, rng, top, max_chars):
    # A found wolf, still alive: reveal it. Say it once per game, then push the vote.
    wolves = [w for w in a.checked_wolf if w in a.living]
    if wolves:
        w = wolves[0]
        if not a.have_i_revealed():
            return _cut("I am the seer. I checked seat %d (%s) last night and they are a werewolf. "
                        "Vote seat %d." % (w, a.names[w], w), max_chars)
        return _cut("Seat %d (%s) is a werewolf — I checked them. Do not let this vote scatter." % (
            w, a.names[w]), max_chars)
    # A false seer claim exists: counter-claim, with what I actually know.
    false_claimants = a.claimed_seer_alive()
    if false_claimants:
        c = false_claimants[0]
        clear = next(iter(a.checked_clear), None)
        if clear is not None:
            text = ("Seat %d (%s) is lying about being the seer — I am the seer. "
                    "I checked seat %d (%s): not a wolf." % (
                        c, a.names[c], clear, a.names[clear]))
        else:
            text = ("Seat %d (%s) is lying about being the seer — I am the seer, "
                    "and I have found no wolf yet." % (c, a.names[c]))
        return _cut(text, max_chars)
    # No wolf found yet: push suspicion without claiming the role.
    if top is None:
        return _cut("No strong read yet. I will keep watching and vote on the evidence.", max_chars)
    reason = a.evidence_reason(top)
    templates = [
        "Seat %d (%s) is my strongest suspicion. %s" % (top, a.names[top], reason),
        "I would vote seat %d (%s) today. %s" % (top, a.names[top], reason),
        "Look closely at seat %d (%s). %s" % (top, a.names[top], reason),
    ]
    return _cut(_pick(rng, templates), max_chars)


def _speak_villager(a, rng, top, max_chars):
    if a.am_accused() and top is not None:
        return _cut("I am a villager. Check my votes: they followed the evidence each day. "
                    "Seat %d (%s) is the better case: %s" % (
                        top, a.names[top], a.evidence_reason(top)), max_chars)
    if top is None:
        return _cut("No clear suspect yet. Let us hear everyone before we vote.", max_chars)
    reason = a.evidence_reason(top)
    name = a.names[top]
    templates = [
        "Seat %d (%s) is my vote today. %s" % (top, name, reason),
        "I think seat %d (%s) is a wolf. %s" % (top, name, reason),
        "The case on seat %d (%s) is the strongest we have. %s" % (top, name, reason),
        "Seat %d (%s) needs to explain themselves. %s" % (top, name, reason),
    ]
    if a.day == 1:
        templates.append("Day one, but seat %d (%s) already stands out: %s" % (top, name, reason))
    return _cut(_pick(rng, templates), max_chars)


# ---------------------------------------------------------------- driver

def decide(a, pending):
    """Answer one pending decision; returns the /act body."""
    kind = pending.get("decision")
    game = pending.get("game")
    if kind == "speak":
        text = _entry_speak(a, pending.get("max_chars") or 500)
        if text is None:
            return {"game": game, "decision": "speak", "pass": True}
        return {"game": game, "decision": "speak", "text": text}
    options = pending.get("options") or []
    if kind == "kill":
        return {"game": game, "decision": "kill", "target": _entry_kill(a, options)}
    if kind == "protect":
        return {"game": game, "decision": "protect", "target": decide_protect(a, options)}
    if kind == "inspect":
        return {"game": game, "decision": "inspect", "target": decide_inspect(a, options)}
    if kind == "vote":
        if a.role == "werewolf":
            return {"game": game, "decision": "vote", "target": _entry_vote_wolf(a, options)}
        return {"game": game, "decision": "vote", "target": _entry_vote_village(a, options)}
    return {"game": game, "decision": kind, "target": None}


def fallback(pending):
    if pending.get("decision") == "speak":
        return {"game": pending.get("game"), "decision": "speak", "pass": True}
    return {"game": pending.get("game"), "decision": pending.get("decision"), "target": None}


def main():
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    seed = os.environ.get("MAKEFUN_TRIAL_SEED") or "default"
    if not env_url:
        sys.stderr.write("werewolf-agent: MAKEFUN_ENV_URL is not set\n")
        return 1
    decisions = 0
    acts = 0
    while True:
        status, obs = _post(env_url, "/observe", {})
        if status != 200 or not isinstance(obs, dict):
            sys.stderr.write("werewolf-agent: /observe answered %s\n" % status)
            return 1
        if obs.get("done"):
            break
        pending = obs.get("pending") or []
        if not pending:
            continue  # /observe long-polls; a new round will list decisions
        for p in pending:
            decisions += 1
            try:
                a = Analysis(p.get("view") or {}, p.get("game"), seed)
                act = decide(a, p)
            except Exception as err:  # never stall a game on an analysis bug
                sys.stderr.write("werewolf-agent: game %s decide failed: %s\n" % (p.get("game"), err))
                act = fallback(p)
            status, _resp = _post(env_url, "/act", act)
            if status == 400:
                status, _resp = _post(env_url, "/act", fallback(p))
            if status == 200:
                acts += 1
            # 409 (not pending: the game moved on) needs no retry.
    # Final tally, for the run log only.
    wins = {"werewolf": [0, 0], "village": [0, 0]}
    finished = 0
    games = obs.get("games") or 0
    for n in range(1, games + 1):
        status, view = _post(env_url, "/game", {"game": n})
        if status != 200 or not isinstance(view, dict):
            continue
        # /game answers the view itself (possibly wrapped); accept either shape.
        v = view.get("view") if isinstance(view.get("view"), dict) else view
        outcome = v.get("outcome")
        if not outcome:
            continue
        finished += 1
        you = v.get("you") or {}
        side = "werewolf" if you.get("role") == "werewolf" else "village"
        wins[side][1] += 1
        if outcome.get("winner") == ("werewolves" if side == "werewolf" else "village"):
            wins[side][0] += 1
    wolf_rate = wins["werewolf"][0] / wins["werewolf"][1] if wins["werewolf"][1] else 0.0
    vil_rate = wins["village"][0] / wins["village"][1] if wins["village"][1] else 0.0
    score = round(1000 * (wolf_rate + vil_rate) / 2)
    print("werewolf-agent: %d decisions, %d acted; finished %d/%d games; "
          "wolf %d/%d, village %d/%d; score %d" % (
              decisions, acts, finished, games,
              wins["werewolf"][0], wins["werewolf"][1],
              wins["village"][0], wins["village"][1], score))
    return 0


if __name__ == "__main__":
    sys.exit(main())
