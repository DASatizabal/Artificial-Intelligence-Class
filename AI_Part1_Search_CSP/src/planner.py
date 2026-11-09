"""AI Part 1 (healthcare version): a rational planner that assigns members to in-network providers.

PEAS
  Performance: share of members placed inside the CMS distance standard (target >= 90%), average miles,
               open-slot use, high-cost members spread under each doctor's cap, search effort
  Environment: county grid, provider directory, open new-patient slots
  Actuators:   member -> provider assignments (or an access-gap flag for the network team)
  Sensors:     provider directory, slots, member home cells, A* travel miles

CSP
  Variables:   members who need a doctor
  Domains:     in-network providers of the needed specialty, accepting patients, with open slots,
               within the CMS distance standard (A* miles). Sorted closest first.
  Constraints: (1) a provider gets no more members than its open slots (capacity)
               (2) a provider gets no more high-cost members than its high-cost cap
  Each specialty is an independent sub-problem (no constraint links a cardiologist to a dermatologist),
  so each one is solved on its own. Splitting a CSP into independent parts is a standard speed-up.
Why no AC-3: AC-3 works on binary constraints (pairs of variables). Capacity is an n-ary constraint
  (it involves every member who could pick that doctor), so the matching tool here is forward checking
  on the capacity counts, plus node consistency when domains are built.
"""
import random, sys, time
from collections import defaultdict
import networkx as nx
from networkx.algorithms.flow import edmonds_karp

sys.setrecursionlimit(10000)


class Planner:
    def __init__(self, kb, members, slots):
        self.kb = kb
        self.members = members.set_index("member_id")
        prov = kb.providers.merge(slots, on="provider_id")
        self.prov = prov.set_index("provider_id")
        self.slots = self.prov["new_slots"].to_dict()
        self.cap = self.prov["high_cost_cap"].to_dict()
        self.domains, self.miles, self.gaps = {}, {}, []

    # ---------- node consistency: build each member's allowed doctors ----------
    def build_domains(self, removed=()):
        self.domains, self.gaps = {}, []
        p = self.prov[(self.prov["new_slots"] > 0) & (~self.prov.index.isin(list(removed)))]
        for mid, m in self.members.iterrows():
            std = self.kb.standards[m["need_specialty"]]
            cand = []
            for pid, pr in p[p["specialty"] == m["need_specialty"]].iterrows():
                if abs(pr["row"] - m["row"]) + abs(pr["col"] - m["col"]) > std:
                    continue                     # Manhattan is a lower bound on A*: skip without searching
                d = self.kb.astar((int(m["row"]), int(m["col"])), (int(pr["row"]), int(pr["col"])))
                if d is not None and d <= std:
                    cand.append((d, pid)); self.miles[(mid, pid)] = d
            if cand:
                self.domains[mid] = [pid for d, pid in sorted(cand)]   # closest doctor first
            else:
                self.gaps.append(mid)            # nobody qualifies inside the standard: an access gap
        return self

    # ---------- feasibility check: max flow says how many members CAN be placed ----------
    def max_placeable(self, mids):
        g = nx.DiGraph()
        for mid in mids:
            g.add_edge("S", mid, capacity=1)
            hc = self.members.at[mid, "high_cost"] == 1
            for pid in self.domains[mid]:
                g.add_edge(mid, f"{pid}_hc" if hc else pid, capacity=1)
        for pid in sorted({p for mid in mids for p in self.domains[mid]}):
            g.add_edge(f"{pid}_hc", pid, capacity=int(self.cap[pid]))
            g.add_edge(pid, "T", capacity=int(self.slots[pid]))
        if "S" not in g:
            return 0, sorted(mids)
        val, flow = nx.maximum_flow(g, "S", "T", flow_func=edmonds_karp)   # BFS-based: same answer every run
        placed = {mid for mid in mids if flow["S"].get(mid, 0) > 0}
        return val, sorted(set(mids) - placed)        # sorted so every run is identical

    # ---------- backtracking search ----------
    def solve(self, mids, mrv=True, fc=True, lcv=False, lookahead=False, node_limit=200_000, seed=42):
        mids = list(mids)
        dom = {m: list(self.domains[m]) for m in mids}
        used, hc_used = defaultdict(int), defaultdict(int)
        hc = {m: self.members.at[m, "high_cost"] == 1 for m in mids}
        assign, stats = {}, {"nodes": 0, "backtracks": 0}
        watchers = defaultdict(list)                       # provider -> members whose domain contains it
        for m in mids:
            for p in dom[m]:
                watchers[p].append(m)

        def consistent(m, p):
            return used[p] < self.slots[p] and (not hc[m] or hc_used[p] < self.cap[p])

        def pick_var():
            free = [m for m in mids if m not in assign]
            if not mrv:
                return free[0]                             # fixed order (member id)
            # MRV: fewest remaining values; tie-break: high-cost first (more constrained), then id
            return min(free, key=lambda m: (sum(consistent(m, p) for p in dom[m]) if not fc else len(dom[m]), not hc[m], m))

        def prune(p):
            """Forward checking: once p is full (or its high-cost cap is full), remove it from the
            domains of unassigned members. Return the removals so they can be undone, or None on wipeout."""
            removed = []
            full = used[p] >= self.slots[p]
            hc_full = hc_used[p] >= self.cap[p]
            for m in watchers[p]:
                if m in assign or p not in dom[m]:
                    continue
                if full or (hc_full and hc[m]):
                    dom[m].remove(p); removed.append((m, p))
                    if not dom[m]:
                        return removed, False
            return removed, True

        def everyone_fits():
            """Look-ahead: can every member still waiting fit into the seats that are left?
            Max flow answers it exactly. Forward checking only notices when ONE member has no seat left;
            this notices when a whole GROUP of members is competing for too few seats."""
            free = [o for o in mids if o not in assign]
            if not free:
                return True
            g = nx.DiGraph()
            for o in free:
                g.add_edge("S", o, capacity=1)
                for q in dom[o]:
                    g.add_edge(o, f"{q}_hc" if hc[o] else q, capacity=1)
            for q in sorted({q for o in free for q in dom[o]}):
                g.add_edge(f"{q}_hc", q, capacity=max(0, int(self.cap[q] - hc_used[q])))
                g.add_edge(q, "T", capacity=max(0, int(self.slots[q] - used[q])))
            stats["lookaheads"] = stats.get("lookaheads", 0) + 1
            return "T" in g and nx.maximum_flow_value(g, "S", "T") == len(free)

        def bt():
            if len(assign) == len(mids):
                return True
            if stats["nodes"] >= node_limit:
                return False
            m = pick_var()
            values = list(dom[m])
            if lcv:
                # LCV: try the doctor that knocks out the fewest options for everyone else first.
                # A doctor only knocks out options when this member fills its last seat (or last high-cost seat).
                def knocked(p):
                    others = [o for o in watchers[p] if o not in assign and o != m and p in dom[o]]
                    if used[p] + 1 >= self.slots[p]:
                        return len(others)
                    if hc[m] and hc_used[p] + 1 >= self.cap[p]:
                        return sum(hc[o] for o in others)
                    return 0
                values.sort(key=lambda p: (knocked(p), self.miles[(m, p)]))
            for p in values:
                if not consistent(m, p):
                    continue
                stats["nodes"] += 1
                assign[m] = p; used[p] += 1; hc_used[p] += hc[m]
                removed, ok = prune(p) if fc else ([], True)
                if ok and lookahead:
                    ok = everyone_fits()           # can everyone still waiting fit after this choice?
                if ok and bt():
                    return True
                for mm, pp in removed:                     # undo forward-checking removals
                    dom[mm].append(pp); dom[mm].sort(key=lambda q: self.miles[(mm, q)])
                del assign[m]; used[p] -= 1; hc_used[p] -= hc[m]
                if stats["nodes"] >= node_limit:
                    return False
            stats["backtracks"] += 1
            return False

        t0 = time.perf_counter()
        ok = bt()
        stats["seconds"] = round(time.perf_counter() - t0, 3)
        stats["solved"] = ok
        return (assign if ok else {}), stats

    # ---------- min-conflicts repair after providers leave ----------
    def min_conflicts(self, assign, removed, max_steps=3000, seed=42):
        """Repair the old seating chart instead of starting over.
        Members whose doctor left are in conflict. Each step picks one conflicted member and gives them the
        value with the fewest conflicts:
          a doctor with a free seat         -> 0 conflicts
          squeeze into a full doctor        -> 1 conflict (that doctor is now over by one, so someone there
                                               becomes conflicted and may move to their own free option)
          waitlist (escalate to network team) -> 1.5 conflicts (only when nothing better exists)
        Ties go to fewer miles. Any seat still over capacity at the end waitlists its farthest member."""
        rng = random.Random(seed)
        removed = set(removed)
        assign = dict(assign)
        hc = lambda m: self.members.at[m, "high_cost"] == 1
        used, hc_used = defaultdict(int), defaultdict(int)
        for m, p in assign.items():
            if p not in removed:
                used[p] += 1; hc_used[p] += hc(m)
        displaced = [m for m, p in assign.items() if p in removed]
        for m in displaced:
            assign[m] = None                                   # None = not handled yet
        over = lambda p, m: used[p] > self.slots[p] or (hc(m) and hc_used[p] > self.cap[p])

        def cost(m, p):
            if p == "WAIT":
                return 1.5
            extra = used[p] + 1 - self.slots[p] + (hc_used[p] + 1 - self.cap[p] if hc(m) else 0)
            return 0 if (used[p] < self.slots[p] and (not hc(m) or hc_used[p] < self.cap[p])) else max(1, extra)

        def conflicted():
            out = []
            for m, p in assign.items():
                if p is None:
                    out.append(m)
                elif p == "WAIT":
                    if any(cost(m, q) == 0 for q in self.domains.get(m, [])):
                        out.append(m)                          # a seat opened up
                elif over(p, m):
                    out.append(m)
            return out

        start = dict(assign)
        steps = 0
        while steps < max_steps:
            c = conflicted()
            if not c:
                break
            m = rng.choice(c)
            cur = assign[m]
            if cur not in (None, "WAIT"):
                used[cur] -= 1; hc_used[cur] -= hc(m)
            options = list(self.domains.get(m, [])) + ["WAIT"]
            best = min(options, key=lambda q: (cost(m, q), 0 if q == "WAIT" else self.miles[(m, q)]))
            assign[m] = best
            if best != "WAIT":
                used[best] += 1; hc_used[best] += hc(m)
            steps += 1
        # clean-up: never leave a doctor over capacity
        for p in list(used):
            while used[p] > self.slots[p] or hc_used[p] > self.cap[p]:
                there = [m for m, q in assign.items() if q == p and (used[p] > self.slots[p] or hc(m))]
                far = max(there, key=lambda m: self.miles[(m, p)])
                assign[far] = "WAIT"; used[p] -= 1; hc_used[p] -= hc(far)
        final = {m: (None if p in (None, "WAIT") else p) for m, p in assign.items()}
        moved_others = sum(1 for m in start if m not in displaced and final[m] != start[m])
        return final, {"steps": steps, "displaced": len(displaced),
                       "displaced_replaced": sum(1 for m in displaced if final[m]),
                       "waitlisted": sum(1 for p in final.values() if p is None),
                       "other_members_moved": moved_others}
