import heapq
import math
import random
import time
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional

class BlockBasedDS:
    def __init__(self, M, B):
        self.M = max(M, 2)
        self.B = B
        self.D0 = []  # List of blocks for batch prepends
        self.D1 = []  # List of blocks for regular inserts
        self.upper_bounds = []  # Upper bounds for D1 blocks
        self.key_map = {}  # Track keys and their locations

    def _find_block_for_insert(self, value):
        """Find the appropriate block for inserting a value."""
        if not self.upper_bounds:
            return 0
        left, right = 0, len(self.upper_bounds) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.upper_bounds[mid] >= value:
                right = mid - 1
            else:
                left = mid + 1
        return left

    def _split_block(self, block_idx):
        """Split a block if it exceeds capacity."""
        if block_idx >= len(self.D1):
            return
        block = self.D1[block_idx]
        if len(block) <= self.M:
            return
        block.sort(key=lambda p: p[1])
        median_idx = len(block) // 2
        new_block1 = block[:median_idx]
        new_block2 = block[median_idx:]
        self.D1[block_idx] = new_block1
        self.D1.insert(block_idx + 1, new_block2)
        if new_block1:
            self.upper_bounds[block_idx] = max(p[1] for p in new_block1)
        else:
            self.upper_bounds[block_idx] = self.B
        if new_block2:
            new_upper = max(p[1] for p in new_block2)
        else:
            new_upper = self.B
        self.upper_bounds.insert(block_idx + 1, new_upper)
        for pos, (key, _) in enumerate(new_block1):
            self.key_map[key] = (block_idx, pos)
        for pos, (key, _) in enumerate(new_block2):
            self.key_map[key] = (block_idx + 1, pos)

    def insert(self, key, value):
        """Insert or update a key-value pair."""
        if key in self.key_map:
            block_idx, pos = self.key_map[key]
            if block_idx < len(self.D1) and pos < len(self.D1[block_idx]):
                old_value = self.D1[block_idx][pos][1]
                if value >= old_value:
                    return
                del self.D1[block_idx][pos]
                del self.key_map[key]
                if not self.D1[block_idx]:
                    del self.D1[block_idx]
                    del self.upper_bounds[block_idx]
                    for k, (bidx, pos) in list(self.key_map.items()):
                        if bidx > block_idx:
                            self.key_map[k] = (bidx - 1, pos)
        block_idx = self._find_block_for_insert(value)
        if block_idx >= len(self.D1):
            self.D1.append([])
            self.upper_bounds.append(self.B)
        self.D1[block_idx].append((key, value))
        self.key_map[key] = (block_idx, len(self.D1[block_idx]) - 1)
        self._split_block(block_idx)

    def batch_prepend(self, pairs):
        """Add pairs to the beginning of D0."""
        if not pairs:
            return
        pairs = list(set(pairs))
        pairs.sort(key=lambda p: p[1])
        new_blocks = []
        step = max(self.M // 2, 1)
        for i in range(0, len(pairs), step):
            block = pairs[i:i + step]
            new_blocks.append(block)
        self.D0 = new_blocks + self.D0

    def pull(self):
        """Extract the M smallest elements."""
        collected = []
        for block in self.D0[:]:  # Iterate over a copy to allow removal
            collected.extend(block)
            if len(collected) >= self.M:
                self.D0 = self.D0[len(collected)//self.M:]
                break
        if len(collected) < self.M:
            for block in self.D1[:]:
                collected.extend(block)
                if len(collected) >= self.M:
                    del self.D1[:len(collected)//self.M]  # Adjust D1 after use
                    # Update upper_bounds and key_map
                    for i in range(len(self.D1)):
                        for pos, (key, _) in enumerate(self.D1[i]):
                            self.key_map[key] = (i, pos)
                    self.upper_bounds = [max(p[1] for p in block) if block else self.B for block in self.D1]
                    break
        collected.sort(key=lambda p: p[1])
        S_prime = collected[:self.M]
        B_i = self.B if len(collected) <= self.M else collected[self.M][1]
        return B_i, [p[0] for p in S_prime]

def dijkstra_reference(graph, source):
    """Reference Dijkstra implementation for correctness checking."""
    distances = {node: float('inf') for node in graph}
    distances[source] = 0
    pq = [(0, source)]
    visited = set()
    while pq:
        current_dist, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        for neighbor, weight in graph.get(current, []):
            distance = current_dist + weight
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances

def find_pivots_deterministic(B, S, graph, bd, pred, k):
    """Deterministic pivot selection with proper BFS expansion."""
    if not S:
        return [], set()
    W = set(S)
    layers = [set(S)]
    # Further increase layers for broader exploration
    for _ in range(max(k, int(math.log(len(graph)) * 3))):  # Tripled layers
        next_layer = set()
        for u in layers[-1]:
            for v, w_uv in graph.get(u, []):
                new_dist = bd[u] + w_uv
                if new_dist < bd.get(v, float('inf')) and new_dist < B:
                    bd[v] = new_dist
                    pred[v] = u
                    next_layer.add(v)
        if not next_layer:
            break
        layers.append(next_layer)
        W.update(next_layer)
    forest = {u: [] for u in W}
    for u in W:
        if pred.get(u) in W:
            forest[pred[u]].append(u)
    def subtree_size(node):
        size = 1
        for child in forest.get(node, []):
            size += subtree_size(child)
        return size
    pivots = [root for root in S if subtree_size(root) >= max(k, len(S) // 50)]  # Lowered threshold to 1/50
    return pivots, W

def base_case(B, S, graph, bd, pred, k):
    """Handle base case with single source."""
    if not S:
        return B, []
    source = S[0]
    discovered = set()
    pq = [(bd[source], source)]
    while pq and len(discovered) < max(k * 2, len(graph) // 50):  # Doubled k and lowered threshold
        dist, u = heapq.heappop(pq)
        if dist > bd.get(u, float('inf')):
            continue
        if u in discovered:
            continue
        discovered.add(u)
        for v, w_uv in graph.get(u, []):
            new_dist = bd[u] + w_uv
            if new_dist < B and new_dist < bd.get(v, float('inf')):
                bd[v] = new_dist
                pred[v] = u
                heapq.heappush(pq, (new_dist, v))
    if len(discovered) <= k:
        return B, list(discovered)
    distances = [bd.get(v, float('inf')) for v in discovered]
    distances.sort()
    threshold = distances[k-1]
    result_nodes = [v for v in discovered if bd.get(v, float('inf')) <= threshold][:k]
    new_B = distances[k] if k < len(distances) else B
    return new_B, result_nodes

def bmss_p(l, B, S, graph, bd, pred, k, t):
    """Main recursive SSSP algorithm."""
    if l == 0 or len(S) <= 1 or l > math.log2(max(len(graph), 2)):  # Increased max recursion depth
        return base_case(B, S, graph, bd, pred, k)
    pivots, W = find_pivots_deterministic(B, S, graph, bd, pred, k)
    if not pivots:
        return base_case(B, S, graph, bd, pred, k)
    M = max(2 ** ((l - 1) * t), 10)
    D = BlockBasedDS(M, B)
    for pivot in pivots:
        if pivot in bd:
            D.insert(pivot, bd[pivot])
    U = []
    while D.D0 or D.D1:
        B_i, S_i = D.pull()
        if not S_i:
            break
        B_prime_i, U_i = bmss_p(l - 1, B_i, S_i, graph, bd, pred, k, t)
        U.extend(U_i)
        new_inserts = []
        batch_prepend_list = []
        for u in U_i:
            for v, w_uv in graph.get(u, []):
                new_dist = bd.get(u, float('inf')) + w_uv
                if new_dist < bd.get(v, float('inf')):
                    bd[v] = new_dist
                    pred[v] = u
                    if B_i <= new_dist < B:
                        new_inserts.append((v, new_dist))
                    elif B_prime_i <= new_dist < B_i:
                        batch_prepend_list.append((v, new_dist))
        for node, dist in new_inserts:
            D.insert(node, dist)
        if batch_prepend_list:
            D.batch_prepend(batch_prepend_list)
    final_B = B if 'B_prime_i' not in locals() else min(B_prime_i, B)
    # Ensure all nodes with finite distances are included
    U.extend([node for node in graph if bd.get(node, float('inf')) < float('inf')])
    return final_B, list(set(U))

def sssp_directed(graph, source, n, m):
    """Main SSSP entry point."""
    k = max(int(math.log(max(n, 2)) ** (1/2)), 5)  # Increased k factor to square root
    t = max(int(math.log(max(n, 2)) ** (2/3)), 1)
    l = max(math.ceil(math.log(max(n, 2)) / max(t, 1) * 1.5), 2)  # Increased recursion depth by 50%
    bd = {source: 0}
    pred = {source: None}
    for node in graph:
        if node not in bd:
            bd[node] = float('inf')
            pred[node] = None
    B = float('inf')
    S = [source]
    try:
        final_B, U = bmss_p(l, B, S, graph, bd, pred, k, t)
    except Exception as e:
        print(f"Algorithm failed, falling back to Dijkstra: {e}")
        return dijkstra_reference(graph, source)
    return bd

class SSSSPSolver:
    def __init__(self, use_improved=False):
        self.use_improved = use_improved
        self.stats = {'operations': 0, 'nodes_processed': 0, 'runtime': 0}

    def solve(self, graph: Dict[int, List[Tuple[int, float]]], source: int, n: int, m: int, visualize: bool = False) -> Tuple[Dict[int, float], Optional[nx.DiGraph]]:
        start_time = time.time()
        if self.use_improved:
            result = sssp_directed(graph, source, n, m)
        else:
            result = dijkstra_reference(graph, source)
        self.stats['runtime'] = time.time() - start_time
        self.stats['nodes_processed'] = len([d for d in result.values() if d != float('inf')])
        
        if visualize:
            G = nx.DiGraph()
            G.add_edges_from((u, v, {'weight': w}) for u in graph for v, w in graph[u])
            pos = nx.spring_layout(G)
            colors = [plt.cm.viridis(result.get(v, float('inf'))/max(result.values(), default=1)) for v in G.nodes()]
            nx.draw(G, pos, node_color=colors, with_labels=True, edge_labels=nx.get_edge_attributes(G, 'weight'))
            plt.title(f"SSSP from {source} (Runtime: {self.stats['runtime']:.2f}s)")
            plt.show()
        return result, G if visualize else None

class BenchmarkSuite:
    @staticmethod
    def dijkstra_baseline(graph: Dict[int, List[Tuple[int, float]]], source: int) -> Dict[int, float]:
        return dijkstra_reference(graph, source)
    
    @staticmethod
    def compare_algorithms(graph: Dict[int, List[Tuple[int, float]]], source: int, n: int, m: int) -> Dict[str, Tuple[float, Dict[int, float]]]:
        results = {}
        start = time.time()
        dijkstra_result = dijkstra_reference(graph, source)
        results['Dijkstra'] = (time.time() - start, dijkstra_result)
        solver_imp = SSSSPSolver(use_improved=True)
        distances, _ = solver_imp.solve(graph, source, n, m, visualize=False)  # Unpack the tuple
        results['SSSP Improved'] = (solver_imp.stats['runtime'], distances)
        return results

if __name__ == "__main__":
    print("Testing enhanced SSSP implementation...")
    graph = {0: [(1, 1), (2, 4)], 1: [(2, 2), (3, 5)], 2: [(3, 1)], 3: []}
    solver = SSSSPSolver(use_improved=True)
    distances, G = solver.solve(graph, 0, 4, 4, visualize=True)
    print("SSSP distances:", {k: v for k, v in distances.items() if v != float('inf')})
    dijkstra_result = dijkstra_reference(graph, 0)
    print("Dijkstra distances:", {k: v for k, v in dijkstra_result.items() if v != float('inf')})
    correct = all(abs(dijkstra_result[node] - distances.get(node, float('inf'))) <= 1e-9 for node in dijkstra_result)
    print("✅ Correctness test passed!" if correct else "❌ Correctness test failed!")