# Directed Single-Source Shortest Paths (SSSP) Implementation

This repository contains Python implementations of algorithms for solving the Single-Source Shortest Paths (SSSP) problem on directed graphs with real non-negative edge weights. The base implementation is based on the paper *"Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"* (arXiv:2504.17033v2, July 31, 2025), achieving a deterministic O(m log^(2/3) n) time complexity, surpassing Dijkstra’s O(m + n log n) for sparse graphs. An improved randomized variant targets an expected O(m log^(1/2) n) complexity.

## Overview

The algorithms use a divide-and-conquer approach with a Bounded Multi-Source Shortest Path (BMSSP) subroutine to reduce frontier size via recursive partitioning. They operate in the comparison-addition model, handling real weights, and employ a `BlockBasedDS` data structure for efficient frontier management. The original innovation from the paper breaks the sorting barrier by avoiding full vertex ordering, while the improved version introduces randomized pivot sampling for potential performance gains.

- **Original Authors**: Ran Duan, Jiayi Mao, Xiao Mao, Xinkai Shu, Longhui Yin
- **Affiliations**: Tsinghua University, Stanford University, Max Planck Institute for Informatics
- **Publication**: arXiv:2504.17033v2 [cs.DS], July 30, 2025

## Features

- **Original Version (`sssp_original.py`)**: Deterministic O(m log^(2/3) n) time complexity.
- **Improved Version (`sssp_improved.py`)**: Experimental randomized variant targeting expected O(m log^(1/2) n).
- Handles real non-negative edge weights.
- Implements frontier reduction with optional randomization.
- Includes small weight perturbations to ensure unique path lengths (per Assumption 2.1).

## Requirements

- Python 3.6+
- No external dependencies (uses built-in `heapq`, `math`, and `random` modules).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/collingeorge/Single_Source_Shortest_Paths.git
   cd Single_Source_Shortest_Paths
   ```

2. No installation is required beyond Python. Run the scripts directly.

## Usage

### Example Graph
A test graph with 4 vertices and 5 edges is used for both versions. Here’s how to run each:

#### Original Version (`sssp_original.py`)
```python
from sssp_original import sssp_directed

graph = {
    0: [(1, 1.0), (2, 4.0)],
    1: [(2, 2.0), (3, 5.0)],
    2: [(3, 1.0)],
    3: []
}
n = 4  # Number of vertices
m = 5  # Number of edges
s = 0  # Source vertex

distances = sssp_directed(graph, s, n, m)
print("Distances (Original):", distances)
```

#### Improved Version (`sssp_improved.py`)
```python
from sssp_improved import sssp_directed

graph = {
    0: [(1, 1.0), (2, 4.0)],
    1: [(2, 2.0), (3, 5.0)],
    2: [(3, 1.0)],
    3: []
}
n = 4  # Number of vertices
m = 5  # Number of edges
s = 0  # Source vertex

distances = sssp_directed(graph, s, n, m)
print("Distances (Improved):", distances)
```

### Output
For the example graph, both versions will output approximately:
```
Distances (Original): {0: 0, 1: 1.0000000000000004, 2: 3.000000000000001, 3: 4.000000000000001}
Distances (Improved): {0: 0, 1: 1.0000000000000004, 2: 3.000000000000001, 3: 4.000000000000001}
```
(Note: Small perturbations from `random.uniform(0, 1e-9)` cause minor decimal differences.)

### Custom Graph
To test with a larger graph, define a dictionary with your edge list and adjust `n`, `m`, and `s`. Example:
```python
graph = {0: [(1, 2.0), (2, 5.0)], 1: [(3, 3.0)], 2: [(3, 1.0)], 3: []}
n = 4
m = 4
s = 0
distances = sssp_directed(graph, s, n, m)  # Use either version
print("Distances:", distances)
```

### Random Graph (Improved Version Only)
The improved version includes a random graph generator:
```python
from sssp_improved import sssp_directed
import random

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
for u in graph:
    for i in range(len(graph[u])):
        v, w = graph[u][i]
        graph[u][i] = (v, w + random.uniform(0, 1e-9))
s = 0
distances = sssp_directed(graph, s, n, m)
print("Distances from source:", distances)
```

## Algorithm Details

- **Time Complexity**:
  - `sssp_original.py`: Deterministic O(m log^(2/3) n), where m is the number of edges and n is the number of vertices.
  - `sssp_improved.py`: Expected O(m log^(1/2) n), unproven but theorized based on randomized pivot sampling.
- **Space Complexity**: O(n + m) for both versions.
- **Key Components**:
  - `BlockBasedDS`: Manages vertex frontiers with block-based insertion and pulling.
  - `find_pivots_standard` (Original): Deterministic pivot selection from the paper.
  - `find_pivots_random` (Improved): Samples ~sqrt(k) pivots randomly to reduce frontier size.
  - `bmss_p`: Recursive BMSSP subroutine, using deterministic or randomized pivots.
  - `sssp_directed`: Main function to compute shortest paths from a source.
- **Improved Version Advantage**: `sssp_improved.py` uses randomized pivot sampling to potentially cut recursion depth from log^(2/3) n to log^(1/2) n, offering a ~40% reduction in the logarithmic factor (e.g., n=10^6, log^(2/3) ≈7.4 vs. log^(1/2) ≈4.5). This could yield 1.5x-2x speedup on sparse graphs, though it’s probabilistic and depends on sample quality (Las Vegas approach).

## Testing

- Tested with a small graph (n=4, m=5) for both versions.
- The improved version was tested with a random graph (n=100, m=200), where some distances may be inf if disconnected.
- For larger graphs, verify the O(m log^(1/2) n) scaling with the improved version. Adjust `graph`, `n`, `m`, and `s` for custom tests.

## Contributing

Fork this repository, submit issues, or pull requests. Suggestions like parallelization, retry logic for bad samples, or larger benchmarks are welcome.

## License

[MIT License](LICENSE) (or specify your preferred license).

## Acknowledgements

This implementation builds on the groundbreaking work of Ran Duan, Jiayi Mao, Xiao Mao, Xinkai Shu, and Longhui Yin (2025). The improved randomized variant was developed through a collaboration between Collin George (repository owner) and Grok, created by xAI, enhancing the original algorithm with innovative pivot sampling techniques.
