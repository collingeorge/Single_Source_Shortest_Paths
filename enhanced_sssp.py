# enhanced_sssp.py
# Scientifically rigorous SPSP benchmarking with counters and correctness checks.
# Supports: Baseline Dijkstra (early-stop), A*, Bidirectional ALT (A* + Landmarks).
# Author: you
# License: MIT

from __future__ import annotations
import heapq
import math
import random
import time
import statistics
import tracemalloc
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# -----------------------------
# Types
# -----------------------------
Node = Any
Weight = float
Graph = Dict[Node, List[Tuple[Node, Weight]]]
Heuristic = Callable[[Node, Node], float]

# -----------------------------
# Stats container
# -----------------------------
@dataclass
class Stats:
    algo: str
    distance: float
    time_s: float
    pops: int
    pushes: int
    relaxes: int
    pq_peak: int
    mem_peak_kb: int
    path_len: int
    ok: bool

# -----------------------------
# Utilities
# -----------------------------
def reconstruct_path(prev: Dict[Node, Node], s: Node, t: Node) -> List[Node]:
    if t not in prev and s != t:
        return []
    path = [t]
    while path[-1] != s:
        path.append(prev[path[-1]])
    path.reverse()
    return path

def manhattan(pos: Dict[Node, Tuple[int,int]]) -> Heuristic:
    def h(u: Node, v: Node) -> float:
        (x1, y1), (x2, y2) = pos[u], pos[v]
        return abs(x1 - x2) + abs(y1 - y2)
    return h

# -----------------------------
# Baseline: Dijkstra (SPSP, early-stop)
# -----------------------------
def dijkstra_spsp(
    G: Graph,
    s: Node,
    t: Node,
) -> Tuple[float, List[Node], Stats]:
    dist: Dict[Node, float] = {s: 0.0}
    prev: Dict[Node, Node] = {}
    pq: List[Tuple[float, Node]] = [(0.0, s)]
    visited: set = set()

    pops = pushes = relaxes = 0
    pq_peak = 1

    tracemalloc.start()
    t0 = time.perf_counter()

    while pq:
        d, u = heapq.heappop(pq); pops += 1
        if u in visited:
            continue
        visited.add(u)

        if u == t:
            break

        for v, w in G.get(u, []):
            relaxes += 1
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v)); pushes += 1
        pq_peak = max(pq_peak, len(pq))

    elapsed = time.perf_counter() - t0
    mem_peak_kb = tracemalloc.get_traced_memory()[1] // 1024
    tracemalloc.stop()

    D = dist.get(t, math.inf)
    path = reconstruct_path(prev, s, t)
    return D, path, Stats(
        algo="Dijkstra",
        distance=D, time_s=elapsed,
        pops=pops, pushes=pushes, relaxes=relaxes,
        pq_peak=pq_peak, mem_peak_kb=mem_peak_kb,
        path_len=len(path), ok=math.isfinite(D),
    )

# -----------------------------
# A* (SPSP)
# -----------------------------
def astar_spsp(
    G: Graph,
    s: Node,
    t: Node,
    heuristic: Heuristic,
) -> Tuple[float, List[Node], Stats]:
    g: Dict[Node, float] = {s: 0.0}
    prev: Dict[Node, Node] = {}
    pq: List[Tuple[float, Node]] = [(heuristic(s, t), s)]
    closed: set = set()

    pops = pushes = relaxes = 0
    pq_peak = 1

    tracemalloc.start()
    t0 = time.perf_counter()

    while pq:
        f, u = heapq.heappop(pq); pops += 1
        if u in closed:
            continue
        closed.add(u)

        if u == t:
            break

        gu = g[u]
        for v, w in G.get(u, []):
            relaxes += 1
            ng = gu + w
            if ng < g.get(v, math.inf):
                g[v] = ng
                prev[v] = u
                heapq.heappush(pq, (ng + heuristic(v, t), v)); pushes += 1
        pq_peak = max(pq_peak, len(pq))

    elapsed = time.perf_counter() - t0
    mem_peak_kb = tracemalloc.get_traced_memory()[1] // 1024
    tracemalloc.stop()

    D = g.get(t, math.inf)
    path = reconstruct_path(prev, s, t)
    return D, path, Stats(
        algo="A*",
        distance=D, time_s=elapsed,
        pops=pops, pushes=pushes, relaxes=relaxes,
        pq_peak=pq_peak, mem_peak_kb=mem_peak_kb,
        path_len=len(path), ok=math.isfinite(D),
    )

# -----------------------------
# ALT: Landmarks preprocessing
# -----------------------------
def dijkstra_all_nodes(G: Graph, start: Node) -> Dict[Node, float]:
    dist: Dict[Node, float] = {start: 0.0}
    pq: List[Tuple[float, Node]] = [(0.0, start)]
    closed: set = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        for v, w in G.get(u, []):
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

def choose_landmarks(G: Graph, k: int, seed: int = 0) -> List[Node]:
    rnd = random.Random(seed)
    nodes = list(G.keys())
    if not nodes:
        return []
    lm = [rnd.choice(nodes)]
    # Farthest-point sampling over graph distances
    for _ in range(1, k):
        best_node, best_score = None, -1.0
        # distance from each candidate to nearest existing landmark
        ref_d = {p: dijkstra_all_nodes(G, p) for p in lm}
        for cand in nodes:
            # maximize min distance across chosen landmarks
            score = min(ref_d[p].get(cand, math.inf) for p in lm)
            if score > best_score:
                best_score, best_node = score, cand
        if best_node is not None:
            lm.append(best_node)
    return lm

def preprocess_alt(G: Graph, landmarks: List[Node]) -> Dict[Node, Dict[Node, float]]:
    # dists[ℓ][x] = distance(ℓ, x)
    return {lm: dijkstra_all_nodes(G, lm) for lm in landmarks}

def alt_heuristic_factory(dists: Dict[Node, Dict[Node, float]]) -> Heuristic:
    lms = list(dists.keys())
    def h(u: Node, v: Node) -> float:
        # ALT lower bound via triangle inequality
        # max over landmarks ℓ of |d(ℓ, v) - d(ℓ, u)|
        best = 0.0
        for lm in lms:
            du = dists[lm].get(u, math.inf)
            dv = dists[lm].get(v, math.inf)
            if math.isfinite(du) and math.isfinite(dv):
                lb = abs(dv - du)
                if lb > best: best = lb
        return best
    return h

# -----------------------------
# Bidirectional A* with ALT heuristic
# -----------------------------
def bidir_alt_spsp(
    G: Graph,
    s: Node,
    t: Node,
    landmarks: Optional[List[Node]] = None,
    lm_seed: int = 0,
    k_landmarks: int = 4,
) -> Tuple[float, List[Node], Stats]:
    if s == t:
        return 0.0, [s], Stats("Bidirectional ALT", 0.0, 0, 0, 0, 0, 0, 1, True)

    if landmarks is None:
        landmarks = choose_landmarks(G, k=k_landmarks, seed=lm_seed)
    lm_dists = preprocess_alt(G, landmarks)
    hf = alt_heuristic_factory(lm_dists)

    gF: Dict[Node, float] = {s: 0.0}
    gB: Dict[Node, float] = {t: 0.0}
    prevF: Dict[Node, Node] = {}
    prevB: Dict[Node, Node] = {}

    pqF: List[Tuple[float, Node]] = [(hf(s, t), s)]
    pqB: List[Tuple[float, Node]] = [(hf(t, s), t)]
    closedF: set = set()
    closedB: set = set()

    pops = pushes = relaxes = 0
    pq_peak = 2

    best = math.inf
    meet: Optional[Node] = None

    tracemalloc.start()
    t0 = time.perf_counter()

    # Expand alternately from the shallower frontier
    while pqF and pqB:
        # forward step
        fF, u = heapq.heappop(pqF); pops += 1
        if u in closedF:
            continue
        closedF.add(u)

        if u in closedB:
            cand = gF[u] + gB[u]
            if cand < best:
                best = cand; meet = u

        for v, w in G.get(u, []):
            relaxes += 1
            ng = gF[u] + w
            if ng < gF.get(v, math.inf):
                gF[v] = ng
                prevF[v] = u
                heapq.heappush(pqF, (ng + hf(v, t), v)); pushes += 1
        pq_peak = max(pq_peak, len(pqF) + len(pqB))

        # backward step
        fB, u = heapq.heappop(pqB); pops += 1
        if u in closedB:
            continue
        closedB.add(u)

        if u in closedF:
            cand = gF[u] + gB[u]
            if cand < best:
                best = cand; meet = u

        for v, w in G.get(u, []):
            relaxes += 1
            ng = gB[u] + w
            if ng < gB.get(v, math.inf):
                gB[v] = ng
                prevB[v] = u
                heapq.heappush(pqB, (ng + hf(v, s), v)); pushes += 1
        pq_peak = max(pq_peak, len(pqF) + len(pqB))

        # admissible stop: if best is established and both frontiers' next f >= best
        if math.isfinite(best):
            nextF = pqF[0][0] if pqF else math.inf
            nextB = pqB[0][0] if pqB else math.inf
            if nextF >= best and nextB >= best:
                break

    elapsed = time.perf_counter() - t0
    mem_peak_kb = tracemalloc.get_traced_memory()[1] // 1024
    tracemalloc.stop()

    if meet is None:
        return math.inf, [], Stats("Bidirectional ALT", math.inf, elapsed, pops, pushes, relaxes, pq_peak, mem_peak_kb, 0, False)

    # rebuild path: s -> meet, meet -> t
    pathF = []
    cur = meet
    while cur != s:
        pathF.append(cur)
        cur = prevF[cur]
    pathF.append(s); pathF.reverse()

    pathB = []
    cur = meet
    while cur != t and cur in prevB:
        cur = prevB[cur]
        pathB.append(cur)

    path = pathF + pathB
    return best, path, Stats(
        algo="Bidirectional ALT",
        distance=best, time_s=elapsed,
        pops=pops, pushes=pushes, relaxes=relaxes,
        pq_peak=pq_peak, mem_peak_kb=mem_peak_kb,
        path_len=len(path), ok=True
    )

# -----------------------------
# Graph generators (reproducible)
# -----------------------------
def gen_sparse_er(n: int, avg_deg: int, max_w: int = 10, seed: int = 42) -> Graph:
    rnd = random.Random(seed)
    G: Graph = defaultdict(list)
    m = n * avg_deg // 2
    edges = set()
    while len(edges) < m:
        u = rnd.randrange(n); v = rnd.randrange(n)
        if u == v or (u, v) in edges or (v, u) in edges:
            continue
        w = rnd.randint(1, max_w)
        G[u].append((v, w)); G[v].append((u, w))
        edges.add((u, v))
    return G

def gen_grid(w: int, h: int, max_w: int = 10, seed: int = 42) -> Tuple[Graph, Dict[Node, Tuple[int,int]]]:
    rnd = random.Random(seed)
    G: Graph = defaultdict(list)
    pos: Dict[Node, Tuple[int,int]] = {}
    def nid(x: int, y: int) -> int:
        return y * w + x
    for y in range(h):
        for x in range(w):
            u = nid(x, y); pos[u] = (x, y)
            if x + 1 < w:
                wgt = rnd.randint(1, max_w)
                G[u].append((nid(x+1, y), wgt)); G[nid(x+1,y)].append((u, wgt))
            if y + 1 < h:
                wgt = rnd.randint(1, max_w)
                G[u].append((nid(x, y+1), wgt)); G[nid(x,y+1)].append((u, wgt))
    return G, pos

# -----------------------------
# Benchmark harness (SPSP only)
# -----------------------------
def run_once(
    G: Graph,
    s: Node,
    t: Node,
    h: Optional[Heuristic] = None,
    use_bidir_alt: bool = True,
    k_landmarks: int = 4,
    lm_seed: int = 0,
) -> Tuple[Stats, Stats, Stats]:
    # Baseline (ground truth)
    D0, path0, st0 = dijkstra_spsp(G, s, t)

    # A* (if heuristic provided; otherwise falls back to Dijkstra heuristic=0)
    if h is None:
        h = lambda u, v: 0.0
    D1, path1, st1 = astar_spsp(G, s, t, h)
    assert math.isclose(D1, D0, rel_tol=0.0, abs_tol=1e-12), "A* distance mismatch vs Dijkstra"

    # Bidirectional ALT
    if use_bidir_alt:
        D2, path2, st2 = bidir_alt_spsp(G, s, t, landmarks=None, k_landmarks=k_landmarks, lm_seed=lm_seed)
        assert math.isclose(D2, D0, rel_tol=0.0, abs_tol=1e-12), "Bidirectional ALT distance mismatch vs Dijkstra"
    else:
        st2 = Stats("Bidirectional ALT", math.inf, 0.0, 0, 0, 0, 0, 0, 0, False)

    return st0, st1, st2

def summarize(rows: List[Stats]) -> Dict[str, float]:
    def mean(x): return statistics.mean(x) if x else float("nan")
    def stdev(x): return statistics.pstdev(x) if len(x) > 1 else 0.0
    return {
        "time_ms_mean": mean([r.time_s*1000 for r in rows]),
        "time_ms_std":  stdev([r.time_s*1000 for r in rows]),
        "pops_mean":    mean([r.pops for r in rows]),
        "relax_mean":   mean([r.relaxes for r in rows]),
        "push_mean":    mean([r.pushes for r in rows]),
        "pq_peak_mean": mean([r.pq_peak for r in rows]),
        "mem_kb_mean":  mean([r.mem_peak_kb for r in rows]),
        "path_len_mean":mean([r.path_len for r in rows]),
    }

def bench_suite(
    which: str = "grid",          # "grid" or "er"
    sizes: Iterable[int] = (10_000, 100_000, 250_000),
    trials: int = 3,
    seed: int = 42,
    avg_deg: int = 6,
    grid_aspect: float = 1.0,     # width/height ratio
    k_landmarks: int = 4,
):
    print("\n=== Single-Pair Shortest Path (SPSP) Benchmarks ===")
    print("All distances validated against baseline Dijkstra (early stop).")
    print("Metrics: time, pops (PQ pops), relaxes (edge relax), pushes, pq_peak, mem_peak.\n")

    for n in sizes:
        rows_dij: List[Stats] = []
        rows_ast: List[Stats] = []
        rows_alt: List[Stats] = []

        for t in range(trials):
            rnd_seed = seed + 1000*t + n

            if which == "grid":
                # choose w,h close to n with aspect
                h = int(math.sqrt(n / max(grid_aspect, 1e-9)))
                w = max(1, int(h * grid_aspect))
                G, pos = gen_grid(w, h, seed=rnd_seed)
                s, tnode = 0, w*h - 1
                hf = manhattan(pos)
            else:
                G = gen_sparse_er(n=n, avg_deg=avg_deg, seed=rnd_seed)
                nodes = list(G.keys())
                if not nodes:
                    print("Empty graph; skipping size", n); break
                rnd = random.Random(rnd_seed)
                s, tnode = rnd.choice(nodes), rnd.choice(nodes)
                while tnode == s:
                    tnode = rnd.choice(nodes)
                hf = (lambda u, v: 0.0)  # no geometry; A*≈Dijkstra

            st0, st1, st2 = run_once(
                G, s, tnode,
                h=hf,
                use_bidir_alt=True,
                k_landmarks=k_landmarks,
                lm_seed=rnd_seed
            )
            rows_dij.append(st0); rows_ast.append(st1)
            if st2.ok: rows_alt.append(st2)

        # summarize
        S0 = summarize(rows_dij)
        S1 = summarize(rows_ast)
        S2 = summarize(rows_alt) if rows_alt else None

        print(f"--- Graph: {which} | size≈{n} | trials={trials} ---")
        def line(name, S):
            print(f"{name:18s}  time(ms) {S['time_ms_mean']:.2f} ±{S['time_ms_std']:.2f} | "
                  f"pops {S['pops_mean']:.1f} | relax {S['relax_mean']:.1f} | "
                  f"push {S['push_mean']:.1f} | pq_peak {S['pq_peak_mean']:.1f} | "
                  f"mem_kb {S['mem_kb_mean']:.0f} | path_len {S['path_len_mean']:.1f}")
        line("Dijkstra", S0)
        line("A*", S1)
        if S2:
            line("Bidirectional ALT", S2)
        print()

# -----------------------------
# Import-friendly API for large_scale_tester.py
# -----------------------------
def run_algorithm(
    algo: str,
    G: Graph,
    s: Node,
    t: Node,
    pos: Optional[Dict[Node, Tuple[int,int]]] = None,
    k_landmarks: int = 4,
    lm_seed: int = 0,
) -> Stats:
    """Return Stats only (for easy integration with external testers)."""
    if algo.lower() == "dijkstra":
        _, _, st = dijkstra_spsp(G, s, t); return st
    elif algo.lower() in ("a*", "astar", "a-star"):
        h = manhattan(pos) if pos is not None else (lambda u, v: 0.0)
        D0, _, st0 = dijkstra_spsp(G, s, t)
        D1, _, st1 = astar_spsp(G, s, t, h)
        assert math.isclose(D0, D1, abs_tol=1e-12), "A* mismatch vs Dijkstra"
        return st1
    elif algo.lower() in ("bidirectional alt", "bidir alt", "alt"):
        D0, _, st0 = dijkstra_spsp(G, s, t)
        D2, _, st2 = bidir_alt_spsp(G, s, t, landmarks=None, k_landmarks=k_landmarks, lm_seed=lm_seed)
        assert math.isclose(D0, D2, abs_tol=1e-12), "Bidirectional ALT mismatch vs Dijkstra"
        return st2
    else:
        raise ValueError(f"Unknown algo: {algo}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    # Example: run a quick suite on grid and ER graphs
    bench_suite(which="grid",  sizes=(10_000, 40_000, 100_000), trials=3, seed=42, k_landmarks=4, grid_aspect=1.0)
    bench_suite(which="er",    sizes=(10_000, 40_000, 100_000), trials=3, seed=42, k_landmarks=4, avg_deg=6)
