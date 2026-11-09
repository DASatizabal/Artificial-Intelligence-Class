"""Knowledge base for the planner: county grid, ZIP cells, provider directory, access standards.

Travel distance is the A* path length on the grid (1 cell = 1 mile), the same idea as Part 1.
"""
import heapq
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class KnowledgeBase:
    def __init__(self, kb_dir=None):
        d = Path(kb_dir) if kb_dir else ROOT / "data" / "kb"
        self.grid = pd.read_csv(d / "county_grid.csv", header=None).values
        self.zips = pd.read_csv(d / "zip_cells.csv", dtype={"zip": str}).set_index("zip")
        self.providers = pd.read_csv(d / "providers.csv")
        self.standards = pd.read_csv(d / "access_standards.csv").set_index("specialty")["max_miles"].to_dict()
        self._by_name = {n.lower(): i for i, n in enumerate(self.providers["name"])}
        self._cache = {}
        self.astar_calls = 0
        self.astar_expansions = 0

    # ---------- search ----------
    def astar(self, start, goal):
        """A* on a 4-connected grid with the Manhattan heuristic (admissible, so the path is the shortest)."""
        key = (start, goal)
        if key in self._cache:
            return self._cache[key]
        self.astar_calls += 1
        rows, cols = self.grid.shape
        h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        frontier = [(h(start), 0, start)]
        best = {start: 0}
        while frontier:
            f, g, cur = heapq.heappop(frontier)
            if cur == goal:
                self._cache[key] = g
                return g
            if g > best.get(cur, 1e9):
                continue
            self.astar_expansions += 1
            r, c = cur
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < rows and 0 <= nc < cols and self.grid[nr, nc] == 0:
                    ng = g + 1
                    if ng < best.get((nr, nc), 1e9):
                        best[(nr, nc)] = ng
                        heapq.heappush(frontier, (ng + h((nr, nc)), ng, (nr, nc)))
        self._cache[key] = None
        return None

    def zip_cell(self, z):
        if z is None or str(z) not in self.zips.index:
            return None
        r = self.zips.loc[str(z)]
        return int(r["row"]), int(r["col"])

    def nearest_providers(self, z, specialty, language=None, k=3, in_network=True, accepting=True, open_hour=None):
        """Return up to k matching providers sorted by A* travel miles.
        Manhattan distance is a lower bound on the A* distance, so we can stop early once the next
        candidate's lower bound is already worse than the k-th best real distance (fewer A* calls)."""
        start = self.zip_cell(z)
        if start is None:
            return []
        p = self.providers[self.providers["specialty"] == specialty]
        if in_network:
            p = p[p["in_network"] == 1]
        if accepting:
            p = p[(p["accepting_new"] == 1) & (p["panel_size"] < p["panel_capacity"])]
        if language:
            p = p[p["languages"].str.contains(language, case=False)]
        if open_hour is not None:
            p = p[(p["open_hour"] <= open_hour) & (open_hour < p["close_hour"])]
        cand = sorted(((abs(r - start[0]) + abs(c - start[1]), i) for i, r, c in zip(p.index, p["row"], p["col"])))
        found = []
        for lb, i in cand:
            if len(found) >= k and lb >= found[-1][0]:
                break                                   # no remaining candidate can beat the current top k
            d = self.astar(start, (int(self.providers.at[i, "row"]), int(self.providers.at[i, "col"])))
            if d is not None:
                found.append((d, i)); found.sort()
                found = found[:k]
        out = []
        for d, i in found:
            row = self.providers.loc[i]
            out.append({"provider_id": row["provider_id"], "name": row["name"], "specialty": row["specialty"],
                        "miles": int(d), "languages": row["languages"].split(";"),
                        "hours": f"{int(row['open_hour'])}:00-{int(row['close_hour'])}:00"})
        return out

    def provider(self, name):
        if not name:
            return None
        i = self._by_name.get(name.lower())
        return None if i is None else self.providers.loc[i].to_dict()
