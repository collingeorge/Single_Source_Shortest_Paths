# enhanced_sssp.py
"""
Enhanced SSSP with hybrid strategy and Numba-accelerated Dijkstra.
Provides:
 - dijkstra_reference(graph, source)
 - SSSSPSolver(use_improved=True/False, use_numba=False)
 - BenchmarkSuite.dijkstra_baseline(...)
 - bidirectional_alt(...) (single-pair shim)
"""

from typing import Dict, List, Tuple, Optional
import heapq
import time
import math
import numpy as np

# try to import numba; if unavailable we gracefully fall back
try:
    from numba import njit, int32, float64
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# -------------------------
# Utilities
# -------------------------
def _collect_all_nodes(graph: Dict[int, List[Tuple[int, float]]], source: int):
    all_nodes = set([source])
    for u, nbrs in graph.items():
        all_nodes.add(u)
        for v, _w in nbrs:
            all_nodes.add(v)
    return all_nodes

def _remap_graph_to_ints(graph: Dict[int, List[Tuple[int, float]]]):
    """
    Map arbitrary node labels to 0..n-1 and return CSR arrays.
    Returns: (n, id_to_index, index_to_id, offsets, neighbors, weights)
    offsets is length n+1
    neighbors, weights are flat arrays
    """
    # collect nodes
    nodes = sorted(list(_collect_all_nodes(graph, next(iter(graph)) if graph else 0)))
    id_to_index = {nid: i for i, nid in enumerate(nodes)}
    index_to_id = nodes
    n = len(nodes)

    # count edges and build adjacency lists
    neighbors_list = []
    weights_list = []
    offsets = np.zeros(n + 1, dtype=np.int64)

    idx = 0
    for i, nid in enumerate(nodes):
        nbrs = graph.get(nid, [])
        offsets[i] = idx
        for v, w in nbrs:
            neighbors_list.append(id_to_index[v])
            weights_list.append(float(w))
            idx += 1
    offsets[n] = idx

    neighbors = np.array(neighbors_list, dtype=np.int64)
    weights = np.array(weights_list, dtype=np.float64)

    return n, id_to_index, index_to_id, offsets, neighbors, weights

# -------------------------
# Python baseline Dijkstra
# -------------------------
def dijkstra_reference(graph: Dict[int, List[Tuple[int, float]]], source: int) -> Dict[int, float]:
    all_nodes = _collect_all_nodes(graph, source)
    dist = {node: float('inf') for node in all_nodes}
    dist[source] = 0.0
    pq = [(0.0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

def enhanced_dijkstra(graph: Dict[int, List[Tuple[int, float]]], source: int) -> Dict[int, float]:
    # same as reference but micro-optimized
    all_nodes = _collect_all_nodes(graph, source)
    dist = {node: float('inf') for node in all_nodes}
    dist[source] = 0.0
    pq = [(0.0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if d == float('inf'):
            break
        for v, w in graph.get(u, []):
            if v in visited:
                continue
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

# -------------------------
# Bidirectional Dijkstra (single pair) - Python
# -------------------------
def bidirectional_dijkstra_pair(graph: Dict[int, List[Tuple[int, float]]], source: int, target: int) -> float:
    """Return shortest distance source->target using bidirectional Dijkstra (float('inf') if unreachable)."""
    if source == target:
        return 0.0

    # forward and backward graphs
    Gf = graph
    Gb = {}
    for u, nbrs in graph.items():
        for v, w in nbrs:
            Gb.setdefault(v, []).append((u, w))

    dist_f = {source: 0.0}
    dist_b = {target: 0.0}
    pq_f = [(0.0, source)]
    pq_b = [(0.0, target)]
    seen_f = {}
    seen_b = {}
    best = float('inf')
    visited_f = set()
    visited_b = set()

    while pq_f or pq_b:
        if pq_f:
            d, u = heapq.heappop(pq_f)
            if d > dist_f.get(u, float('inf')):
                pass
            else:
                if u in visited_f:
                    pass
                visited_f.add(u)
                if u in dist_b:
                    best = min(best, d + dist_b[u])
                for v, w in Gf.get(u, []):
                    nd = d + w
                    if nd < dist_f.get(v, float('inf')):
                        dist_f[v] = nd
                        heapq.heappush(pq_f, (nd, v))
        if pq_b:
            d, u = heapq.heappop(pq_b)
            if d > dist_b.get(u, float('inf')):
                pass
            else:
                if u in visited_b:
                    pass
                visited_b.add(u)
                if u in dist_f:
                    best = min(best, d + dist_f[u])
                for v, w in Gb.get(u, []):
                    nd = d + w
                    if nd < dist_b.get(v, float('inf')):
                        dist_b[v] = nd
                        heapq.heappush(pq_b, (nd, v))
        # termination
        if pq_f and pq_b:
            if pq_f[0][0] + pq_b[0][0] >= best:
                break
    return best

# -------------------------
# Numba-accelerated Dijkstra on CSR arrays
# -------------------------
if NUMBA_AVAILABLE:
    @njit
    def _heap_push(keys, nodes, size, key, node):
        # keys, nodes are 1D arrays; size is int (current size)
        i = size
        keys[i] = key
        nodes[i] = node
        # sift up
        while i > 0:
            parent = (i - 1) // 2
            if keys[parent] <= keys[i]:
                break
            # swap
            tk = keys[parent]; tn = nodes[parent]
            keys[parent] = keys[i]; nodes[parent] = nodes[i]
            keys[i] = tk; nodes[i] = tn
            i = parent
        return size + 1

    @njit
    def _heap_pop(keys, nodes, size):
        if size == 0:
            return (1e300, -1, 0)  # sentinel
        min_key = keys[0]
        min_node = nodes[0]
        last = size - 1
        keys[0] = keys[last]
        nodes[0] = nodes[last]
        # clear last (not necessary)
        # sift down
        i = 0
        size -= 1
        while True:
            left = 2 * i + 1
            right = left + 1
            smallest = i
            if left < size and keys[left] < keys[smallest]:
                smallest = left
            if right < size and keys[right] < keys[smallest]:
                smallest = right
            if smallest == i:
                break
            # swap
            tk = keys[smallest]; tn = nodes[smallest]
            keys[smallest] = keys[i]; nodes[smallest] = nodes[i]
            keys[i] = tk; nodes[i] = tn
            i = smallest
        return (min_key, min_node, size)

    @njit
    def dijkstra_numba(n, offsets, neighbors, weights, source):
        # offsets: int64[n+1], neighbors:int64[m], weights:float64[m]
        INF = 1e300
        dist = np.full(n, INF)
        visited = np.zeros(n, dtype=np.uint8)
        # heap arrays sized n (worst-case)
        keys = np.full(n, INF)
        nodes = np.full(n, -1)
        heap_size = 0
        # push source
        heap_size = _heap_push(keys, nodes, heap_size, 0.0, source)
        dist[source] = 0.0

        while heap_size > 0:
            min_key, u, heap_size = _heap_pop(keys, nodes, heap_size)
            if u < 0:
                break
            if visited[u]:
                continue
            visited[u] = 1
            d = min_key
            # relax neighbors
            start = offsets[u]
            end = offsets[u + 1]
            for idx in range(start, end):
                v = neighbors[idx]
                w = weights[idx]
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heap_size = _heap_push(keys, nodes, heap_size, nd, v)
        return dist

else:
    # placeholder function signatures when numba not available
    def dijkstra_numba(*args, **kwargs):
        raise RuntimeError("Numba not available. Install numba to use dijkstra_numba.")

# -------------------------
# Graph-type detector
# -------------------------
def detect_graph_type(graph: Dict[int, List[Tuple[int, float]]]) -> str:
    """
    Heuristic classifier:
     - compute avg degree and max degree
     - if max_degree / max(1, avg_degree) > threshold -> 'scale_free'
     - else -> 'sparse_structured'
    """
    degrees = []
    for u, nbrs in graph.items():
        degrees.append(len(nbrs))
    if not degrees:
        return "sparse_structured"
    avg = sum(degrees) / len(degrees)
    mx = max(degrees)
    ratio = mx / max(1.0, avg)
    if ratio > 8.0:
        return "scale_free"
    return "sparse_structured"

# -------------------------
# Hybrid Solver & public API
# -------------------------
class SSSSPSolver:
    def __init__(self, use_improved: bool = True, use_numba: bool = False, prefer_alt: bool = False, experimental: bool = False):
        """
        use_improved: prefer optimized Dijkstra in Python
        use_numba: use numba CSR solver when appropriate
        prefer_alt: prefer bidirectional alt for single-pair (if target provided)
        experimental: enable BMSSP experimental (disabled by default)
        """
        self.use_improved = use_improved
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.prefer_alt = prefer_alt
        self.experimental = experimental
        self.stats = {'runtime': 0.0, 'nodes_processed': 0, 'method': None}

    def solve(self, graph: Dict[int, List[Tuple[int, float]]], source: int, n: Optional[int] = None, m: Optional[int] = None, target: Optional[int] = None, visualize: bool = False) -> Dict[int, float]:
        start = time.time()
        method = "dijkstra_python"

        # Validate graph node labels; if not 0..n-1 remap for numba
        can_use_numba = self.use_numba
        if self.use_numba:
            # verify node labels are numeric and dense; we remap if needed
            try:
                n_nodes, id_to_index, index_to_id, offsets, neighbors, weights = _remap_graph_to_ints(graph)
            except Exception:
                can_use_numba = False

        # Optionally handle single-pair with bidirectional
        if self.prefer_alt and target is not None:
            best = bidirectional_dijkstra_pair(graph, source, target)
            # build SSSP-like dict with known target and source distances (others leave inf)
            all_nodes = _collect_all_nodes(graph, source)
            dist = {node: float('inf') for node in all_nodes}
            dist[source] = 0.0
            if best < 1e299:
                dist[target] = best
            self.stats['runtime'] = time.time() - start
            self.stats['nodes_processed'] = sum(1 for v in dist.values() if v != float('inf'))
            self.stats['method'] = 'bidirectional_pair'
            return dist

        # detect graph type to pick method
        gtype = detect_graph_type(graph)
        if gtype == 'scale_free':
            # Hubs; pruning is less effective. Use robust python Dijkstra
            dist = enhanced_dijkstra(graph, source)
            method = 'dijkstra_python_scale_free'
        else:
            # structured/sparse: prefer numba if enabled & available & remapping succeeded
            if can_use_numba:
                # call numba version on CSR arrays
                dist_arr = dijkstra_numba(n_nodes, offsets, neighbors, weights, id_to_index.get(source, 0))
                # remap back
                dist = {}
                for i, orig in enumerate(index_to_id):
                    val = dist_arr[i]
                    if val >= 1e299:
                        dist[orig] = float('inf')
                    else:
                        dist[orig] = float(val)
                method = 'dijkstra_numba'
            else:
                dist = enhanced_dijkstra(graph, source)
                method = 'dijkstra_python_sparse'

        self.stats['runtime'] = time.time() - start
        self.stats['nodes_processed'] = sum(1 for v in dist.values() if v != float('inf'))
        self.stats['method'] = method
        return dist

# -------------------------
# BenchmarkSuite
# -------------------------
class BenchmarkSuite:
    @staticmethod
    def dijkstra_baseline(graph: Dict[int, List[Tuple[int, float]]], source: int) -> Dict[int, float]:
        return dijkstra_reference(graph, source)

    @staticmethod
    def compare_algorithms(graph: Dict[int, List[Tuple[int, float]]], source: int, n: Optional[int] = None, m: Optional[int] = None) -> Dict[str, Tuple[float, Dict[int, float]]]:
        results = {}
        t0 = time.time()
        dref = dijkstra_reference(graph, source)
        results['Dijkstra'] = (time.time() - t0, dref)

        solver = SSSSPSolver(use_improved=True, use_numba=True)
        d_imp = solver.solve(graph, source, n=n, m=m)
        results['SSSP Improved'] = (solver.stats['runtime'], d_imp)

        return results

# -------------------------
# bidirectional_alt shim (compat)
# -------------------------
def bidirectional_alt(graph: Dict[int, List[Tuple[int, float]]], source: int, target: Optional[int] = None, landmarks: Optional[List[int]] = None) -> Dict[int, float]:
    """
    Compatibility shim for large_scale_tester; if a target is provided, use bidirectional pair search,
    otherwise do a full SSSP via enhanced Dijkstra.
    """
    if target is None:
        return enhanced_dijkstra(graph, source)
    best = bidirectional_dijkstra_pair(graph, source, target)
    all_nodes = _collect_all_nodes(graph, source)
    dist = {node: float('inf') for node in all_nodes}
    dist[source] = 0.0
    if best < 1e299:
        dist[target] = best
    return dist

# exports
__all__ = [
    'dijkstra_reference',
    'enhanced_dijkstra',
    'SSSSPSolver',
    'BenchmarkSuite',
    'bidirectional_alt'
]
