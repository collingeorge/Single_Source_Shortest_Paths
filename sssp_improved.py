import heapq
import math
import random  # For small perturbations to simulate unique paths

class BlockBasedDS:
    def __init__(self, M, B):
        self.M = max(M, 2)  # Ensure M >= 2 to avoid division by zero
        self.B = B
        self.D0 = []  # List of blocks for batch prepends (lists of (key, value))
        self.D1 = [[]]  # List of blocks for inserts
        self.upper_bounds = [B]  # Upper bounds for D1 blocks
        self.key_map = {}  # Track keys and their (block_idx, pos) in D1 for updates/deletes

    def _find_block_for_insert(self, value):
        left, right = 0, len(self.upper_bounds) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.upper_bounds[mid] >= value:
                right = mid - 1
            else:
                left = mid + 1
        return left

    def _split_block(self, block_idx):
        block = self.D1[block_idx]
        if len(block) <= self.M:
            return
        block.sort(key=lambda p: p[1])
        median_idx = len(block) // 2
        new_block1 = block[:median_idx]
        new_block2 = block[median_idx:]
        self.D1[block_idx] = new_block1
        self.D1.insert(block_idx + 1, new_block2)
        new_upper = max(p[1] for p in new_block2) if new_block2 else self.B
        self.upper_bounds[block_idx] = max(p[1] for p in new_block1) if new_block1 else self.B
        self.upper_bounds.insert(block_idx + 1, new_upper)
        for pos, (key, _) in enumerate(new_block1):
            self.key_map[key] = (block_idx, pos)
        for pos, (key, _) in enumerate(new_block2):
            self.key_map[key] = (block_idx + 1, pos)

    def insert(self, key, value):
        if key in self.key_map:
            block_idx, pos = self.key_map[key]
            old_value = self.D1[block_idx][pos][1]
            if value >= old_value:
                return
            del self.D1[block_idx][pos]
            del self.key_map[key]
            if not self.D1[block_idx]:
                del self.D1[block_idx]
                del self.upper_bounds[block_idx]
        block_idx = self._find_block_for_insert(value)
        if block_idx == len(self.D1):
            self.D1.append([])
            self.upper_bounds.append(self.B)
        self.D1[block_idx].append((key, value))
        self.key_map[key] = (block_idx, len(self.D1[block_idx]) - 1)
        self._split_block(block_idx)

    def batch_prepend(self, pairs):
        pairs = list(set(pairs))
        pairs.sort(key=lambda p: p[1])
        new_blocks = []
        step = max(self.M // 2, 1)  # Ensure step >= 1
        for i in range(0, len(pairs), step):
            block = pairs[i:i + step]
            new_blocks.insert(0, block)
        self.D0 = new_blocks + self.D0

    def pull(self):
        collected = []
        for block in self.D0:
            collected.extend(block)
            if len(collected) >= self.M:
                break
        if len(collected) < self.M:
            for block in self.D1:
                collected.extend(block)
                if len(collected) >= self.M:
                    break
        collected.sort(key=lambda p: p[1])
        S_prime = collected[:self.M]
        B_i = self.B if len(collected) <= self.M else collected[self.M][1]
        return B_i, [p[0] for p in S_prime]

def find_pivots_random(B, S, graph, bd, pred, k):
    if len(S) < k:
        return find_pivots_standard(B, S, graph, bd, pred, k)
    sample_size = int(math.sqrt(k) * len(S))
    P_sample = random.sample(S, min(sample_size, len(S)))
    W = set(P_sample)
    W_layers = [set(P_sample)]
    for _ in range(k):
        W_next = set()
        for u in W_layers[-1]:
            for v, w_uv in graph.get(u, []):
                new_dist = bd[u] + w_uv
                if new_dist <= bd[v]:
                    bd[v] = new_dist
                    pred[v] = u
                    if new_dist < B:
                        W_next.add(v)
        W_layers.append(W_next)
        W.update(W_next)
        if len(W) > k * len(P_sample):
            return P_sample, W
    F = {u: [] for u in W}
    for u in W:
        if pred.get(u) in W and bd[u] == bd[pred[u]] + graph[pred[u]][u]:
            F[pred[u]].append(u)
    P = [root for root in P_sample if dfs_tree_size(root, F) >= k]
    return P, W

def find_pivots_standard(B, S, graph, bd, pred, k):
    W = set(S)
    W_layers = [set(S)]
    for i in range(k):
        W_next = set()
        for u in W_layers[-1]:
            for v, w_uv in graph.get(u, []):
                new_dist = bd[u] + w_uv
                if new_dist <= bd[v]:
                    bd[v] = new_dist
                    pred[v] = u
                    if new_dist < B:
                        W_next.add(v)
        W_layers.append(W_next)
        W.update(W_next)
        if len(W) > k * len(S):
            return S, W
    F = {u: [] for u in W}
    for u in W:
        if pred.get(u) in W and bd[u] == bd[pred[u]] + graph[pred[u]][u]:
            F[pred[u]].append(u)
    P = [root for root in S if dfs_tree_size(root, F) >= k]
    return P, W

def dfs_tree_size(node, F):
    visited = set()
    stack = [node]
    size = 0
    while stack:
        curr = stack.pop()
        if curr in visited:
            continue
        visited.add(curr)
        size += 1
        stack.extend(F.get(curr, []))
    return size

def base_case(B, S, graph, bd, pred, k):
    assert len(S) == 1
    x = S[0]
    U0 = set(S)
    H = []
    heapq.heappush(H, (bd[x], x))
    while H and len(U0) < k + 1:
        dist_u, u = heapq.heappop(H)
        if dist_u > bd[u]: continue
        U0.add(u)
        for v, w_uv in graph.get(u, []):
            new_dist = bd[u] + w_uv
            if new_dist < B and new_dist < bd[v]:
                bd[v] = new_dist
                pred[v] = u
                heapq.heappush(H, (new_dist, v))
    if len(U0) <= k:
        return B, list(U0)
    max_dist = max(bd[v] for v in U0)
    U = [v for v in U0 if bd[v] < max_dist]
    return max_dist, U

def bmss_p(l, B, S, graph, bd, pred, k, t):
    if l == 0:
        return base_case(B, S, graph, bd, pred, k)
    P, W = find_pivots_random(B, S, graph, bd, pred, k)  # Use randomized version
    M = 2 ** ((l - 1) * t)
    D = BlockBasedDS(M, B)
    for x in P:
        D.insert(x, bd[x])
    i = 0
    B_prime_prev = min(bd[x] for x in P) if P else B
    U = []
    while len(U) < k * (2 ** (l * t)) and (D.D0 or D.D1):
        i += 1
        B_i, S_i = D.pull()
        B_prime_i, U_i = bmss_p(l - 1, B_i, S_i, graph, bd, pred, k, t)
        U.extend(U_i)
        K = []
        for u in U_i:
            for v, w_uv in graph.get(u, []):
                new_dist = bd[u] + w_uv
                if new_dist <= bd[v]:
                    bd[v] = new_dist
                    pred[v] = u
                    if B_i <= new_dist < B:
                        D.insert(v, new_dist)
                    elif B_prime_i <= new_dist < B_i:
                        K.append((v, new_dist))
        prepend_list = K + [(x, bd[x]) for x in S_i if B_prime_i <= bd[x] < B_i]
        D.batch_prepend(prepend_list)
    B_prime = min(B_prime_i, B) if 'B_prime_i' in locals() else B
    U.extend([x for x in W if bd[x] < B_prime])
    return B_prime, U

def sssp_directed(graph, s, n, m):
    k = int(math.log(n) ** (1/3))
    t = int(math.log(n) ** (2/3))
    l = math.ceil(math.log(n) / t)
    bd = {v: float('inf') for v in range(n)}
    bd[s] = 0
    pred = {v: None for v in range(n)}
    B = float('inf')
    S = [s]
    _, U = bmss_p(l, B, S, graph, bd, pred, k, t)
    return bd

# Generate a random directed graph with n=100, m=200
n = 100
m = 200
graph = {i: [] for i in range(n)}
edges = set()
while len(edges) < m:
    u = random.randint(0, n-1)
    v = random.randint(0, n-1)
    if u != v and (u, v) not in edges:
        edges.add((u, v))
        w = random.uniform(1, 10)
        graph[u].append((v, w))
s = 0
for u in graph:
    for i in range(len(graph[u])):
        v, w = graph[u][i]
        graph[u][i] = (v, w + random.uniform(0, 1e-9))

# Run the algorithm
distances = sssp_directed(graph, s, n, m)
print("Distances from source:", distances)