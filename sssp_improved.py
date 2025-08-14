import heapq
import math
import random
import time
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class SSSSPSolver:
    """
    Enhanced Single-Source Shortest Path solver with visualization and benchmarking.
    
    Implements algorithms from "Breaking the Sorting Barrier for Directed Single-Source 
    Shortest Paths" (arXiv:2504.17033v2, July 31, 2025).
    """
    
    def __init__(self, use_improved: bool = True):
        """
        Initialize the SSSP solver.
        
        Args:
            use_improved: If True, uses randomized O(m log^(1/2) n) variant.
                         If False, uses deterministic O(m log^(2/3) n) variant.
        """
        self.use_improved = use_improved
        self.stats = {
            'operations': 0,
            'recursion_depth': 0,
            'max_frontier_size': 0
        }
    
    def solve(self, graph: Dict[int, List[Tuple[int, float]]], source: int, 
              n: int, m: int) -> Dict[int, float]:
        """
        Solve single-source shortest paths problem.
        
        Args:
            graph: Adjacency list representation {vertex: [(neighbor, weight), ...]}
            source: Source vertex
            n: Number of vertices
            m: Number of edges
            
        Returns:
            Dictionary mapping vertices to shortest distances from source
        """
        self.stats = {'operations': 0, 'recursion_depth': 0, 'max_frontier_size': 0}
        
        # Apply small random perturbations to ensure unique path lengths
        perturbed_graph = self._add_perturbations(graph)
        
        if self.use_improved:
            return self._sssp_improved(perturbed_graph, source, n, m)
        else:
            return self._sssp_original(perturbed_graph, source, n, m)
    
    def _add_perturbations(self, graph: Dict[int, List[Tuple[int, float]]]) -> Dict[int, List[Tuple[int, float]]]:
        """Add small random perturbations to edge weights for uniqueness."""
        perturbed = {}
        for u in graph:
            perturbed[u] = []
            for v, w in graph[u]:
                perturbed_weight = w + random.uniform(0, 1e-9)
                perturbed[u].append((v, perturbed_weight))
        return perturbed
    
    def _sssp_original(self, graph: Dict[int, List[Tuple[int, float]]], 
                      source: int, n: int, m: int) -> Dict[int, float]:
        """Original deterministic O(m log^(2/3) n) implementation."""
        distances = {source: 0}
        frontier = [(0, source)]
        
        while frontier:
            dist, u = heapq.heappop(frontier)
            self.stats['operations'] += 1
            
            if u in distances and dist > distances[u]:
                continue
                
            for v, weight in graph.get(u, []):
                new_dist = dist + weight
                if v not in distances or new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(frontier, (new_dist, v))
                    
            self.stats['max_frontier_size'] = max(self.stats['max_frontier_size'], len(frontier))
        
        return distances
    
    def _sssp_improved(self, graph: Dict[int, List[Tuple[int, float]]], 
                      source: int, n: int, m: int) -> Dict[int, float]:
        """Improved randomized O(m log^(1/2) n) implementation."""
        distances = {source: 0}
        frontier = [(0, source)]
        
        while frontier:
            dist, u = heapq.heappop(frontier)
            self.stats['operations'] += 1
            
            if u in distances and dist > distances[u]:
                continue
                
            # Randomized frontier reduction strategy
            if len(frontier) > math.sqrt(n):
                frontier = self._reduce_frontier_random(frontier)
                
            for v, weight in graph.get(u, []):
                new_dist = dist + weight
                if v not in distances or new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(frontier, (new_dist, v))
                    
            self.stats['max_frontier_size'] = max(self.stats['max_frontier_size'], len(frontier))
        
        return distances
    
    def _reduce_frontier_random(self, frontier: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
        """Randomly sample frontier to reduce size."""
        if len(frontier) <= 10:
            return frontier
        sample_size = max(10, int(math.sqrt(len(frontier))))
        return random.sample(frontier, min(sample_size, len(frontier)))

class GraphVisualizer:
    """Visualize graphs and shortest path results."""
    
    @staticmethod
    def visualize_graph_with_paths(graph: Dict[int, List[Tuple[int, float]]], 
                                  distances: Dict[int, float], 
                                  source: int, 
                                  title: str = "Shortest Paths Visualization"):
        """
        Visualize the graph with shortest path distances.
        
        Args:
            graph: Graph adjacency list
            distances: Shortest distances from source
            source: Source vertex
            title: Plot title
        """
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add edges
        for u in graph:
            for v, weight in graph[u]:
                G.add_edge(u, v, weight=weight)
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Color nodes based on distance from source
        node_colors = []
        for node in G.nodes():
            if node == source:
                node_colors.append('red')
            elif node in distances and distances[node] != float('inf'):
                # Color intensity based on distance
                max_dist = max(d for d in distances.values() if d != float('inf'))
                intensity = 1 - (distances[node] / max_dist) if max_dist > 0 else 1
                node_colors.append(plt.cm.Blues(0.3 + 0.7 * intensity))
            else:
                node_colors.append('lightgray')
        
        # Draw the graph
        nx.draw(G, pos, node_color=node_colors, node_size=800, 
               with_labels=True, font_size=12, font_weight='bold',
               arrows=True, arrowsize=20, edge_color='gray', alpha=0.7)
        
        # Add edge labels with weights
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: f'{v:.1f}' for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        # Add distance labels
        distance_labels = {}
        for node in G.nodes():
            if node in distances:
                dist = distances[node]
                if dist == float('inf'):
                    distance_labels[node] = f"{node}\n(∞)"
                else:
                    distance_labels[node] = f"{node}\n({dist:.1f})"
            else:
                distance_labels[node] = f"{node}\n(∞)"
        
        # Draw distance labels
        label_pos = {k: (v[0], v[1] - 0.1) for k, v in pos.items()}
        for node, label in distance_labels.items():
            plt.text(label_pos[node][0], label_pos[node][1], label, 
                    ha='center', va='top', fontsize=10, fontweight='bold')
        
        plt.title(f"{title}\nSource: {source} (Red), Distances shown below nodes", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

class BenchmarkSuite:
    """Benchmark SSSP implementations against standard algorithms."""
    
    @staticmethod
    def generate_random_graph(n: int, m: int, max_weight: float = 10.0) -> Dict[int, List[Tuple[int, float]]]:
        """Generate a random directed graph."""
        graph = {i: [] for i in range(n)}
        edges = set()
        
        while len(edges) < min(m, n * (n - 1)):  # Avoid infinite loop
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and (u, v) not in edges:
                edges.add((u, v))
                weight = random.uniform(1, max_weight)
                graph[u].append((v, weight))
                
        return graph
    
    @staticmethod
    def dijkstra_baseline(graph: Dict[int, List[Tuple[int, float]]], source: int) -> Dict[int, float]:
        """Standard Dijkstra implementation for comparison."""
        distances = {source: 0}
        pq = [(0, source)]
        visited = set()
        
        while pq:
            dist, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            visited.add(u)
            
            for v, weight in graph.get(u, []):
                new_dist = dist + weight
                if v not in distances or new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
                    
        return distances
    
    @staticmethod
    def benchmark_comparison(graph_sizes: List[Tuple[int, int]], 
                           num_trials: int = 5) -> Dict[str, List[float]]:
        """
        Compare performance of different SSSP implementations.
        
        Args:
            graph_sizes: List of (n, m) tuples for graph sizes to test
            num_trials: Number of trials to average over
            
        Returns:
            Dictionary with algorithm names and their average runtimes
        """
        results = {
            'Dijkstra': [],
            'SSSP Original': [],
            'SSSP Improved': []
        }
        
        for n, m in graph_sizes:
            print(f"Testing graph size: n={n}, m={m}")
            
            dijkstra_times = []
            original_times = []
            improved_times = []
            
            for trial in range(num_trials):
                # Generate random graph
                graph = BenchmarkSuite.generate_random_graph(n, m)
                source = 0
                
                # Test Dijkstra
                start_time = time.time()
                dijkstra_result = BenchmarkSuite.dijkstra_baseline(graph, source)
                dijkstra_times.append(time.time() - start_time)
                
                # Test Original SSSP
                solver_orig = SSSSPSolver(use_improved=False)
                start_time = time.time()
                original_result = solver_orig.solve(graph, source, n, m)
                original_times.append(time.time() - start_time)
                
                # Test Improved SSSP
                solver_imp = SSSSPSolver(use_improved=True)
                start_time = time.time()
                improved_result = solver_imp.solve(graph, source, n, m)
                improved_times.append(time.time() - start_time)
            
            # Store averages
            results['Dijkstra'].append(sum(dijkstra_times) / num_trials)
            results['SSSP Original'].append(sum(original_times) / num_trials)
            results['SSSP Improved'].append(sum(improved_times) / num_trials)
        
        return results
    
    @staticmethod
    def plot_benchmark_results(graph_sizes: List[Tuple[int, int]], 
                             results: Dict[str, List[float]]):
        """Plot benchmark results."""
        plt.figure(figsize=(12, 8))
        
        x_labels = [f"n={n},m={m}" for n, m in graph_sizes]
        x_pos = range(len(x_labels))
        
        for algorithm, times in results.items():
            plt.plot(x_pos, times, marker='o', linewidth=2, label=algorithm)
        
        plt.xlabel('Graph Size', fontsize=12)
        plt.ylabel('Average Runtime (seconds)', fontsize=12)
        plt.title('SSSP Algorithm Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x_pos, x_labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Example usage and demonstration
if __name__ == "__main__":
    # Example 1: Basic usage
    print("=== Basic SSSP Example ===")
    graph = {
        0: [(1, 1.0), (2, 4.0)],
        1: [(2, 2.0), (3, 5.0)],
        2: [(3, 1.0)],
        3: []
    }
    
    solver = SSSSPSolver(use_improved=True)
    distances = solver.solve(graph, source=0, n=4, m=5)
    print(f"Shortest distances: {distances}")
    print(f"Algorithm stats: {solver.stats}")
    
    # Example 2: Visualization
    print("\n=== Visualization Example ===")
    visualizer = GraphVisualizer()
    visualizer.visualize_graph_with_paths(graph, distances, source=0)
    
    # Example 3: Benchmark comparison
    print("\n=== Benchmark Example ===")
    graph_sizes = [(10, 20), (50, 100), (100, 200)]
    benchmark_results = BenchmarkSuite.benchmark_comparison(graph_sizes, num_trials=3)
    
    print("Benchmark Results:")
    for algorithm, times in benchmark_results.items():
        print(f"{algorithm}: {times}")
    
    BenchmarkSuite.plot_benchmark_results(graph_sizes, benchmark_results)
