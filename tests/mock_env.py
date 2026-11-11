#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Mock env for werewolf-agent-01: a local stand-in for program/env/server.mjs.

Implements the same HTTP protocol (POST /observe, /game, /act) and the same
game rules as the prompt describes, over the kit's practice deals. The seven
house seats are heuristic bots built from main.py's own Analysis/decide
functions plus seeded noise -- stand-ins for claude-haiku-5-5, NOT the real
house model. Purpose: verify the entry completes full practice trials
without crashing, without illegal moves, and within time limits. Win rates
against these mock house bots measure protocol correctness and basic play,
not the expected score against Haiku.

Run: python3 tests/run_practice.py [trial ...]   (trials 0, 1, 2; default all)
"""
import hashlib
import json
import os
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as entry  # noqa: E402


def _rng(seed, *parts):
    h = hashlib.sha256("\0".join([str(seed)] + [str(p) for p in parts]).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


class Game:
    def __init__(self, deal, seed):
        self.n = deal["game"]
        self.entry_seat = deal["entry"]
        self.roles = list(deal["roles"])
        self.names = list(deal["names"])
        self.start = deal["start"]
        self.seed = seed
        self.day = 1
        self.phase = "night"
        self.alive = [True] * 8
        self.log = []
        self.seq = 0
        self.outcome = None
        self.seer_results = {s: [] for s in range(1, 9)}  # seat -> [{night, seat, werewolf}]
        self.last_protect = {s: None for s in range(1, 9)}
        self.wolf_votes = []  # [{night, seat, target}]
        self.pending = None  # (decision, options, max_chars) waiting on the entry
        self.night = {"wolf_picks": {}, "protect": None, "protect_by": None}
        self.vote_tally = {}
        self._house_calls = 0
        self._emit("night")

    # -- log / views ------------------------------------------------------
    def _emit(self, kind, **kw):
        self.seq += 1
        e = {"seq": self.seq, "day": self.day, "kind": kind}
        e.update(kw)
        self.log.append(e)
        return e

    def living_seats(self):
        return [s for s in range(1, 9) if self.alive[s - 1]]

    def wolves(self):
        return [s for s in range(1, 9) if self.roles[s - 1] == "werewolf" and self.alive[s - 1]]

    def view_for(self, seat):
        role = self.roles[seat - 1]
        partner = None
        if role == "werewolf":
            others = [s for s in range(1, 9) if s != seat and self.roles[s - 1] == "werewolf"]
            if others:
                p = others[0]
                partner = {"seat": p, "name": self.names[p - 1], "alive": self.alive[p - 1]}
        return {
            "day": self.day, "phase": "over" if self.outcome else self.phase,
            "you": {"seat": seat, "name": self.names[seat - 1], "role": role,
                    "alive": self.alive[seat - 1], "partner": partner},
            "players": [{"seat": s, "name": self.names[s - 1], "alive": self.alive[s - 1],
                         "role": self.roles[s - 1] if not self.alive[s - 1] else None}
                        for s in range(1, 9)],
            "seerResults": self.seer_results[seat] if role == "seer" else None,
            "lastProtect": self.last_protect[seat] if role == "doctor" else None,
            "wolfVotes": list(self.wolf_votes) if role == "werewolf" else None,
            "log": [dict(e) for e in self.log],
            "outcome": self.outcome,
        }

    # -- house bots -------------------------------------------------------
    def _house_rng(self, seat):
        self._house_calls += 1
        return _rng(self.seed, "house", self.n, seat, self._house_calls)

    def _house_analysis(self, seat):
        a = entry.Analysis(self.view_for(seat), self.n, self.seed)
        # House players are independent minds, not copies of one another:
        # each reads the same log through its own noisy suspicion, so their
        # tie-breaks and early accusations do not all collapse onto one seat.
        rng = self._house_rng(seat)
        for s in a.suspicion:
            a.suspicion[s] = a.suspicion.get(s, 0.0) + rng.uniform(-0.8, 0.8)
        return a

    def _house_act(self, seat, decision, options, max_chars=500):
        rng = self._house_rng(seat)
        a = self._house_analysis(seat)
        if decision == "speak":
            if rng.random() < 0.25:
                return None
            return entry.speak(a, max_chars)
        if decision == "kill":
            # Second wolf matches the first's pick (usually).
            picks = self.night["wolf_picks"]
            if picks and rng.random() < 0.9:
                t = next(iter(picks.values()))
                return t if t in options else None
            return entry.decide_kill(a, options)
        if decision == "protect":
            return entry.decide_protect(a, options)
        if decision == "inspect":
            return entry.decide_inspect(a, options)
        if decision == "vote":
            if a.role == "werewolf":
                t = entry.decide_vote_wolf(a, options)
            else:
                t = entry.decide_vote_village(a, options)
            if t is None:
                return None
            # Noise: sometimes follow the plurality leader instead.
            tally = a.current_votes()
            if tally and rng.random() < 0.25:
                leader = max(tally, key=lambda s: (tally[s], -s))
                if leader in options:
                    return leader
            return t
        return None

    # -- engine -----------------------------------------------------------
    def _win_check(self):
        wolves = self.wolves()
        non_wolves = [s for s in self.living_seats() if self.roles[s - 1] != "werewolf"]
        if not wolves:
            winners = [s for s in range(1, 9) if self.roles[s - 1] != "werewolf"]
            return {"winner": "village", "reason": "no_wolves", "day": self.day, "winners": winners}
        if len(wolves) >= len(non_wolves):
            winners = [s for s in range(1, 9) if self.roles[s - 1] == "werewolf"]
            return {"winner": "werewolves", "reason": "parity", "day": self.day, "winners": winners}
        return None

    def _end(self, outcome):
        self.outcome = outcome
        self.phase = "over"
        self.pending = None
        self._emit("end", winner=outcome["winner"], reason=outcome["reason"],
                   winners=outcome["winners"])

    def _do_turn(self, seat, decision, options, max_chars=500):
        """Resolve one seat's turn; returns True if the engine must wait on the entry."""
        if seat == self.entry_seat and self.alive[seat - 1]:
            self.pending = (decision, list(options), max_chars)
            return True
        if not self.alive[seat - 1]:
            return False
        return self._apply(seat, decision,
                           self._house_act(seat, decision, list(options), max_chars))

    def _apply(self, seat, decision, target_or_text):
        if decision == "speak":
            if target_or_text:
                self._emit("speech", seat=seat, round=0, text=entry._cut(target_or_text, 500))
            else:
                self._emit("pass", seat=seat, round=0)
        elif decision == "vote":
            self._emit("vote", seat=seat, target=target_or_text)
            if target_or_text is not None:
                self.vote_tally[target_or_text] = self.vote_tally.get(target_or_text, 0) + 1
        elif decision == "kill":
            self.night["wolf_picks"][seat] = target_or_text
            self.wolf_votes.append({"night": self.day, "seat": seat, "target": target_or_text})
        elif decision == "protect":
            self.night["protect"] = target_or_text
            self.night["protect_by"] = seat
            if target_or_text is not None:
                self.last_protect[seat] = {"night": self.day, "seat": target_or_text}
        elif decision == "inspect":
            if target_or_text is not None:
                self.seer_results[seat].append(
                    {"night": self.day, "seat": target_or_text,
                     "werewolf": self.roles[target_or_text - 1] == "werewolf"})
        return False

    def debate_order(self):
        order = []
        s = self.start + (self.day - 1)
        for i in range(8):
            seat = (s - 1 + i) % 8 + 1
            if self.alive[seat - 1]:
                order.append(seat)
        return order

    def step(self):
        """Advance until the entry owes a decision or the game ends."""
        while self.outcome is None and self.pending is None:
            if self.phase == "night":
                if self._night_step():
                    return
            elif self.phase == "debate":
                if self._debate_step():
                    return
            elif self.phase == "voting":
                if self._voting_step():
                    return
            else:
                return

    def _night_step(self):
        # Werewolves vote in turn: odd nights lower seat first, even nights the other.
        wolves = sorted(self.wolves())
        if self.day % 2 == 0:
            wolves = wolves[::-1]
        stage = getattr(self, "_night_stage", 0)
        if stage < len(wolves):
            seat = wolves[stage]
            self._night_stage = stage + 1
            options = [s for s in self.living_seats() if self.roles[s - 1] != "werewolf"]
            return self._do_turn(seat, "kill", options)
        if stage == len(wolves):
            self._night_stage = stage + 1
            doc = next((s for s in self.living_seats() if self.roles[s - 1] == "doctor"), None)
            if doc is not None:
                last = (self.last_protect[doc] or {}).get("seat")
                options = [s for s in self.living_seats()
                           if s != doc and s != last]
                return self._do_turn(doc, "protect", options)
        if stage == len(wolves) + 1:
            self._night_stage = stage + 1
            seer = next((s for s in self.living_seats() if self.roles[s - 1] == "seer"), None)
            if seer is not None:
                options = [s for s in self.living_seats() if s != seer]
                return self._do_turn(seer, "inspect", options)
        # Resolve the night.
        self._night_stage = 0
        picks = {t for t in self.night["wolf_picks"].values() if t is not None}
        target = picks.pop() if len(picks) == 1 else None
        died, role = None, None
        if target is not None and target != self.night["protect"]:
            self.alive[target - 1] = False
            died, role = target, self.roles[target - 1]
        self.night = {"wolf_picks": {}, "protect": None, "protect_by": None}
        self._emit("dawn", died=died, role=role)
        outcome = self._win_check()
        if outcome:
            self._end(outcome)
            return False
        self.phase = "debate"
        self._debate_rounds = 0
        self._debate_idx = 0
        self._emit("debate", order=self.debate_order())
        return False

    def _debate_step(self):
        order = self.debate_order()
        if self._debate_rounds >= 2:
            self.phase = "voting"
            self.vote_tally = {}
            self._vote_idx = 0
            self._emit("voting", order=self.debate_order())
            return False
        if self._debate_idx >= len(order):
            self._debate_idx = 0
            self._debate_rounds += 1
            return False
        seat = order[self._debate_idx]
        self._debate_idx += 1
        return self._do_turn(seat, "speak", [], 500)

    def _voting_step(self):
        order = self.debate_order()
        if self._vote_idx >= len(order):
            tally = self.vote_tally
            exiled = None
            if tally:
                top = max(tally.values())
                leaders = [s for s, v in tally.items() if v == top]
                if len(leaders) == 1:
                    exiled = leaders[0]
            if exiled is not None:
                self.alive[exiled - 1] = False
                self._emit("exile", seat=exiled, role=self.roles[exiled - 1],
                           tally=[{"seat": s, "votes": v} for s, v in sorted(tally.items())])
            outcome = self._win_check()
            if outcome:
                self._end(outcome)
                return False
            if self.day >= 10:
                winners = []
                self._end({"winner": None, "reason": "max_days", "day": self.day,
                           "winners": winners})
                return False
            self.day += 1
            self.phase = "night"
            self._night_stage = 0
            self._emit("night")
            return False
        seat = order[self._vote_idx]
        self._vote_idx += 1
        options = [s for s in self.living_seats() if s != seat]
        return self._do_turn(seat, "vote", options)


class Trial:
    def __init__(self, table, seed):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.seed = seed
        self.games = [Game(d, seed) for d in table["games"]]
        self.opened = 0
        self.illegal = []
        self.acts = 0
        self.max_statement = 0
        self.decision_counts = {}
        self._open_up_to(10)
        for g in self.games[: self.opened]:
            g.step()

    def _open_up_to(self, n):
        while self.opened < len(self.games) and self.opened < n:
            self.opened += 1

    def _maybe_open_next(self):
        live = sum(1 for g in self.games[: self.opened] if g.outcome is None)
        if live < 10 and self.opened < len(self.games):
            nxt = self.games[self.opened]
            self.opened += 1
            nxt.step()

    def observe(self):
        with self.cond:
            for _ in range(20):  # up to ~2s long-poll (mock; the real env holds 20s)
                pending = []
                for g in self.games[: self.opened]:
                    if g.pending is not None:
                        decision, options, max_chars = g.pending
                        p = {"game": g.n, "decision": decision, "view": g.view_for(g.entry_seat)}
                        if decision == "speak":
                            p["max_chars"] = max_chars
                        else:
                            p["options"] = options
                        pending.append(p)
                done = all(g.outcome is not None for g in self.games)
                if pending or done:
                    return {"games": self.opened,
                            "finished": sum(1 for g in self.games if g.outcome is not None),
                            "done": done, "pending": pending}
                self.cond.wait(timeout=0.1)
            return {"games": self.opened,
                    "finished": sum(1 for g in self.games if g.outcome is not None),
                    "done": False, "pending": []}

    def act(self, body):
        with self.cond:
            n = body.get("game")
            if not isinstance(n, int) or not 1 <= n <= len(self.games):
                return 400, {"error": "bad_request"}
            g = self.games[n - 1]
            if g.pending is None or g.outcome is not None:
                return 409, {"error": "not_pending"}
            decision, options, max_chars = g.pending
            if body.get("decision") != decision:
                return 409, {"error": "not_pending"}
            if decision == "speak":
                if body.get("pass") is True:
                    value = None
                elif isinstance(body.get("text"), str):
                    value = entry._fold(body["text"])
                    if value == "":
                        return 400, {"error": "empty"}
                    if len(value) > (max_chars or 500):
                        return 400, {"error": "too_long"}
                    self.max_statement = max(self.max_statement, len(value))
                else:
                    return 400, {"error": "bad_request"}
            else:
                value = body.get("target")
                if value is not None and value not in options:
                    self.illegal.append((n, decision, value, list(options)))
                    return 400, {"error": "bad_target"}
            g.pending = None
            self.acts += 1
            g._apply(g.entry_seat, decision, value)
            g.step()
            self._maybe_open_next()
            self.cond.notify_all()
            return 200, {"ok": True}

    def game_view(self, n):
        with self.lock:
            if not 1 <= n <= len(self.games):
                return 400, {"error": "bad_request"}
            return 200, {"view": self.games[n - 1].view_for(self.games[n - 1].entry_seat)}

    def results(self):
        out = []
        for g in self.games:
            role = g.roles[g.entry_seat - 1]
            side = "werewolf" if role == "werewolf" else "village"
            won = g.outcome is not None and (
                (side == "werewolf" and g.outcome["winner"] == "werewolves") or
                (side == "village" and g.outcome["winner"] == "village"))
            out.append({"game": g.n, "role": role, "side": side,
                        "winner": g.outcome["winner"] if g.outcome else None,
                        "won": won, "days": g.outcome["day"] if g.outcome else None})
        return out


def make_handler(trial):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, status, obj):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                return self._json(400, {"error": "bad_request"})
            if self.path == "/observe":
                return self._json(200, trial.observe())
            if self.path == "/game":
                status, obj = trial.game_view(body.get("game"))
                return self._json(status, obj)
            if self.path == "/act":
                status, obj = trial.act(body)
                return self._json(status, obj)
            return self._json(404, {"error": "not_found"})
    return Handler


def serve(trial, port=0):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(trial))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server
