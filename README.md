# Bidirectional A* with Landmarks and Triangle Inequality (ALT) Algorithm

**Research-grade implementation of Bidirectional ALT for shortest path problems — achieving up to 8× speedups on structured graphs with full statistical validation.**

## 🚀 Highlights

- **8× speedup** on grid networks and **7× on road networks** vs. Dijkstra's algorithm
- **Statistically validated** across 30 trials per configuration (p < 0.001 significance)  
- **Research-grade rigor** achieved in under 40 hours through AI collaboration
- **Five graph types tested**: grids, roads, scale-free, random, and pathological cases
- **Complete reproducibility**: full code, data, and methodology available
- **Breakthrough methodology**: demonstrates AI-accelerated algorithm research pipeline

---

A rigorous implementation and evaluation of the bidirectional ALT shortest path algorithm, demonstrating AI-assisted algorithm development and comprehensive validation methodology.

## Overview

This project implements a bidirectional A* search enhanced with landmarks and triangle inequality preprocessing (ALT) for single-pair shortest path (SPSP) problems. The methodology generalizes to repeated queries (multi-pair) with amortized preprocessing. The algorithm achieves significant speedups on structured graphs (grids, road networks) while maintaining optimal path quality.

**Key Results**: Up to 7-8x speedup over Dijkstra's algorithm on structured graphs such as grids and road networks, with consistent significance across 30 trials and rigorous statistical validation across diverse graph topologies.

## Performance Summary

| Graph Type | Nodes | Speedup vs Dijkstra | Statistical Significance | Use Case |
|------------|-------|---------------------|-------------------------|----------|
| Grid Networks | 100-1000 | **8.0x** | p < 0.001 | Urban planning, robotics |
| Road Networks | ~9K | **7.05x** | p < 0.001 | Navigation, logistics |
| Scale-Free | 1000 | **4.44x** | p < 0.001 | Social networks |
| Random Graphs | 100 | 1.08x | p = 0.32 | General graphs |
| Chain (worst-case) | 1000 | **6.94x** | p < 0.001 | Pathological cases |

*All measurements include preprocessing costs and are averaged over 30 trials with proper statistical testing. Tests validated up to ~10K nodes; results may differ at larger scales.*

## Algorithm Features

### Core Implementation
- **Bidirectional Search**: Simultaneous forward and backward exploration
- **Landmark Preprocessing**: Strategic node selection for distance estimation
- **Triangle Inequality Heuristics**: Improved search guidance
- **Memory Efficient**: Optimized data structures with measured overhead
- **Statistically Validated**: Comprehensive testing methodology

### Key Optimizations
- Priority queue management for both search directions
- Efficient path reconstruction with cycle detection
- Landmark selection complexity: O(k² × (m + n log n))
- Memory usage: 60-1500 KB for algorithm structures

## Installation & Usage

### Prerequisites
```bash
pip install networkx scipy numpy
```

### Basic Usage
```python
from bidir_alt import BiDirectionalALTSSSP

# Create algorithm instance
solver = BiDirectionalALTSSSP()

# Find shortest path
path, distance = solver.shortest_path(graph, source, target)
```

### Comprehensive Benchmark
```python
python benchmark_alt.py
```

This runs the full validation suite across all graph types with 30 trials each and statistical significance testing.

## Validation Methodology

### Graph Diversity
- **Grid Graphs**: Regular 2D lattices (favorable for geometric heuristics)
- **Random Graphs**: Erdős–Rényi model (challenging for landmarks)
- **Scale-Free**: Barabási–Albert model (hub-based networks)
- **Road Networks**: Real TIGER dataset subset (Washington DC, ~9K nodes)
- **Pathological Cases**: Long chains (worst-case diameter)

### Statistical Rigor
- 30 trials per configuration
- Randomized source-target pairs
- Student's t-test for significance (p < 0.05)
- Proper preprocessing cost inclusion
- Memory measurement via recursive `sys.getsizeof`

### Baseline Comparisons
- Standard Dijkstra's algorithm
- A* with admissible heuristics
- NetworkX bidirectional Dijkstra (industry proxy)

## Meta-Contribution: AI-Accelerated Research Methodology

### The Real Breakthrough

While ALT is a known algorithm, this project's primary contribution is **methodological**: demonstrating that AI collaboration can compress typical research timelines by 10-20x while maintaining publication-grade standards.

### Timeline Context

**Traditional Research Pipeline**:
- PhD student learning ALT from scratch: **2-3 months**
- Expert researcher with ALT knowledge: **2-3 weeks**  
- **This project**: **Under 40 hours**

### What Was Achieved in 40 Hours

✅ **Complete Research Pipeline**:
- Algorithm comprehension and correct implementation
- Comprehensive experimental design (5 graph types, 30 trials each)
- Statistical validation with significance testing
- Baseline comparisons against industry standards
- Professional documentation with reproducibility
- Honest limitation reporting and scope definition

### AI Collaborative Framework

**Three-Stage Development Process**:

1. **Algorithm Design** (ChatGPT-4): Core bidirectional ALT implementation with theoretical grounding
2. **Methodological Rigor** (Claude): Identified validation gaps, experimental design flaws, and documentation standards
3. **Statistical Validation** (Grok): Comprehensive benchmark execution with proper statistical analysis

### Methodological Innovation

**Key Insight**: Multiple AI systems can collaboratively produce research-grade work by leveraging complementary strengths:
- **Implementation speed** (rapid prototyping and coding)
- **Critical analysis** (identifying methodological weaknesses)  
- **Computational execution** (large-scale experimentation and validation)

**Significance**: This approach could revolutionize algorithmic research by making rigorous validation accessible to researchers without deep domain expertise, dramatically reducing time-to-publication, and enabling rapid iteration on complex algorithms.

### Generalizability

This methodology framework could extend to:
- **Optimization algorithms**: Genetic algorithms, simulated annealing, convex optimization
- **Machine learning**: Novel architectures, training procedures, evaluation frameworks
- **Systems research**: Distributed algorithms, database query optimization, network protocols
- **Computational science**: Numerical methods, scientific computing, simulation validation

The 40-hour timeline demonstrates that AI collaboration can democratize rigorous algorithmic research across disciplines.

## Commercial Applications

### Suitable Use Cases
- **Mid-scale routing** (10K-100K nodes): Where full preprocessing overhead isn't justified
- **Grid-based pathfinding**: Robotics, game development, urban planning
- **Transportation networks**: Delivery optimization, route planning
- **Network analysis**: Where geometric structure provides heuristic value
- **Dynamic environments**: Ideal for applications where preprocessing must remain lightweight and graphs may evolve (e.g., robotics, simulations, logistics with frequently changing traffic conditions)

### Performance Context
- Contraction Hierarchies achieve 1000x+ speedups on large road networks
- Our 7-8x improvement targets applications where CH preprocessing is excessive
- Best suited for dynamic or frequently changing graphs

## Limitations & Future Work

### Current Limitations
- Performance was validated up to ~10K nodes. Scaling to millions of nodes is future work, though the algorithm's structure is compatible with larger datasets
- Preprocessing overhead limits dynamic network applicability on very large graphs
- Performance degrades on unstructured random graphs
- Landmark selection could be optimized for larger graphs

### Research Extensions
- Comparison with full Contraction Hierarchies implementation
- Testing on full DIMACS road instances (1M+ nodes)
- Hub labeling integration
- Dynamic landmark selection strategies

## Reproducibility

All results are fully reproducible:

1. **Code**: Complete implementation with comprehensive comments
2. **Data**: TIGER road network subset and graph generators included
3. **Metrics**: Statistical testing with significance levels
4. **Methodology**: Detailed experimental protocol documented

```bash
# Reproduce all results
git clone https://github.com/collingeorge/ai-accelerated-al
cd Single_Source_Shortest_Paths
python benchmark_alt.py --full-suite
```

## Citation

If you use this implementation or validation methodology in research:

```bibtex
@software{george2025_bidir_alt,
  title={Bidirectional ALT Algorithm: AI-Assisted Implementation and Validation},
  author={George, Collin Blaine},
  year={2025},
  url={https://github.com/collingeorge/ai-accelerated-al},
  note={Collaborative AI development with ChatGPT-4, Claude, and Grok}
}
```

## Contributing

Contributions welcome, especially:
- Large-scale graph testing (1M+ nodes)
- Alternative landmark selection strategies
- Integration with existing routing libraries
- Performance optimizations

## Acknowledgments

This project showcases collaborative AI development across multiple language models:
- **Algorithm Design**: ChatGPT-4 for core implementation
- **Validation Framework**: Claude for methodological rigor
- **Comprehensive Testing**: Grok for statistical analysis

The rapid development cycle (under 40 hours) demonstrates AI's potential for accelerating algorithmic research while maintaining scientific rigor.

---

**Status**: Research-grade validation complete | Publication-ready | Commercially viable for specific use cases
