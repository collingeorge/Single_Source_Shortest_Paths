#!/usr/bin/env python3
"""
Large Scale SSSP Testing Suite

Tests SSSP algorithms on industry-scale graphs to validate performance claims.
This is the make-or-break test for real-world applicability.
"""

import os
import time
import random
import math
import gc
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import psutil
import tracemalloc

try:
    from enhanced_sssp import SSSSPSolver, BenchmarkSuite
    HAS_ENHANCED = True
except ImportError:
    print("❌ enhanced_sssp.py not found!")
    exit(1)

class LargeScaleTester:
    """Test SSSP algorithms on industry-scale problems."""
    
    def __init__(self):
        self.results = {}
        self.memory_usage = {}
        
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def create_chip_routing_graph(self, n: int, avg_degree: int, 
                                 locality_bias: float = 0.8) -> Dict[int, List[Tuple[int, float]]]:
        """
        Create realistic chip routing graph with:
        - Local connectivity (most connections to nearby nodes)
        - Power-law degree distribution
        - Realistic edge weights
        """
        print(f"🏗️  Generating chip routing graph: {n:,} nodes, avg degree {avg_degree}")
        
        graph = {i: [] for i in range(n)}
        edges = set()
        total_edges = 0
        target_edges = n * avg_degree // 2
        
        # Create local connectivity clusters (simulates chip layout)
        cluster_size = int(math.sqrt(n))
        clusters = []
        for i in range(0, n, cluster_size):
            clusters.append(list(range(i, min(i + cluster_size, n))))
        
        # Phase 1: Local cluster connections
        for cluster in clusters:
            for node in cluster:
                degree = random.randint(max(1, avg_degree - 3), avg_degree + 3)
                connections = 0
                attempts = 0
                
                while connections < degree and attempts < degree * 20:
                    if random.random() < locality_bias:
                        # Local connection within cluster
                        target = random.choice(cluster)
                    else:
                        # Random long-distance connection
                        target = random.randint(0, n - 1)
                    
                    if (target != node and (node, target) not in edges and 
                        (target, node) not in edges and total_edges < target_edges):
                        
                        # Realistic weight based on "distance"
                        weight = random.uniform(0.1, 5.0)
                        if abs(node - target) > cluster_size:
                            weight *= random.uniform(1.5, 3.0)  # Long connections are more expensive
                        
                        graph[node].append((target, weight))
                        graph[target].append((node, weight))
                        edges.add((min(node, target), max(node, target)))
                        connections += 1
                        total_edges += 1
                    
                    attempts += 1
        
        print(f"✅ Generated graph with {total_edges:,} edges (avg degree: {2*total_edges/n:.1f})")
        return graph
    
    def create_scale_free_graph(self, n: int, m: int) -> Dict[int, List[Tuple[int, float]]]:
        """Create scale-free graph using preferential attachment (realistic for many networks)."""
        print(f"🌐 Generating scale-free graph: {n:,} nodes, {m} edges per new node")
        
        graph = defaultdict(list)
        degrees = defaultdict(int)
        
        # Start with a small complete graph
        initial_nodes = min(m + 1, 10)
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                weight = random.uniform(0.5, 2.0)
                graph[i].append((j, weight))
                graph[j].append((i, weight))
                degrees[i] += 1
                degrees[j] += 1
        
        # Add remaining nodes with preferential attachment
        for new_node in range(initial_nodes, n):
            # Select m nodes to connect to based on their degree
            total_degree = sum(degrees.values())
            if total_degree == 0:
                targets = list(range(min(new_node, m)))
            else:
                targets = []
                candidates = list(range(new_node))
                
                for _ in range(min(m, len(candidates))):
                    # Probabilistic selection based on degree
                    if random.random() < 0.7 and candidates:  # Preferential attachment
                        weights = [degrees[node] + 1 for node in candidates]
                        total_weight = sum(weights)
                        r = random.uniform(0, total_weight)
                        cumsum = 0
                        for i, weight in enumerate(weights):
                            cumsum += weight
                            if cumsum >= r:
                                targets.append(candidates.pop(i))
                                break
                    elif candidates:  # Random selection
                        targets.append(candidates.pop(random.randint(0, len(candidates) - 1)))
            
            # Connect to selected targets
            for target in targets:
                weight = random.uniform(0.1, 3.0)
                graph[new_node].append((target, weight))
                graph[target].append((new_node, weight))
                degrees[new_node] += 1
                degrees[target] += 1
        
        return dict(graph)
    
    def benchmark_single_graph(self, graph: Dict[int, List[Tuple[int, float]]], 
                             graph_name: str, source: int = 0) -> Dict[str, Dict]:
        """Benchmark all algorithms on a single graph."""
        n = len(graph)
        m = sum(len(adj_list) for adj_list in graph.values()) // 2
        
        print(f"\n🧪 Testing {graph_name}")
        print(f"   📊 Stats: {n:,} nodes, {m:,} edges, avg degree: {2*m/n:.1f}")
        
        results = {}
        
        # Test each algorithm
        algorithms = [
            ("Dijkstra Baseline", lambda: BenchmarkSuite.dijkstra_baseline(graph, source)),
            ("SSSP Original", lambda: SSSSPSolver(use_improved=False).solve(graph, source, n, m)[0]),  # Unpack tuple
            ("SSSP Improved", lambda: SSSSPSolver(use_improved=True).solve(graph, source, n, m)[0])   # Unpack tuple
        ]
        
        for alg_name, alg_func in algorithms:
            print(f"   🔄 Running {alg_name}...", end=" ")
            
            # Memory tracking
            gc.collect()
            memory_before = self.get_memory_usage()
            
            try:
                start_time = time.time()
                distances = alg_func()  # This should now be the dictionary
                end_time = time.time()
                
                memory_after = self.get_memory_usage()
                runtime = end_time - start_time
                reachable = len([d for d in distances.values() if d != float('inf')])
                
                results[alg_name] = {
                    'runtime': runtime,
                    'reachable_nodes': reachable,
                    'memory_mb': memory_after - memory_before,
                    'success': True
                }
                
                print(f"✅ {runtime:.3f}s ({reachable:,} nodes, {memory_after-memory_before:.1f}MB)")
                
            except Exception as e:
                print(f"❌ FAILED: {str(e)}")
                results[alg_name] = {
                    'runtime': float('inf'),
                    'reachable_nodes': 0,
                    'memory_mb': float('inf'),
                    'error': str(e),
                    'success': False
                }
        
        return results
    
    def run_scalability_test(self):
        """Test how algorithms scale with graph size."""
        print("🚀 LARGE SCALE SCALABILITY TEST")
        print("=" * 60)
        
        # Test sizes that simulate real chip routing problems
        test_cases = [
            # Format: (nodes, avg_degree, graph_type)
            (5_000, 6, "chip_routing"),      # Small chip
            (10_000, 8, "chip_routing"),     # Medium chip  
            (25_000, 10, "chip_routing"),    # Large chip
            (50_000, 12, "chip_routing"),    # Very large chip
            (100_000, 8, "chip_routing"),    # Massive chip (industry scale)
            
            # Scale-free networks (different connectivity pattern)
            (10_000, 5, "scale_free"),
            (25_000, 6, "scale_free"),
            (50_000, 7, "scale_free"),
        ]
        
        all_results = {}
        
        for n, avg_degree, graph_type in test_cases:
            graph_name = f"{graph_type}_{n//1000}k"
            
            print(f"\n{'='*20} {graph_name.upper()} {'='*20}")
            
            try:
                # Generate graph
                if graph_type == "chip_routing":
                    graph = self.create_chip_routing_graph(n, avg_degree)
                else:
                    graph = self.create_scale_free_graph(n, avg_degree // 2)
                
                # Run benchmark
                results = self.benchmark_single_graph(graph, graph_name, source=0)
                all_results[graph_name] = results
                
                # Quick analysis
                if 'SSSP Improved' in results and 'Dijkstra Baseline' in results:
                    improved_time = results['SSSP Improved']['runtime']
                    dijkstra_time = results['Dijkstra Baseline']['runtime']
                    
                    if improved_time > 0 and improved_time != float('inf'):
                        speedup = dijkstra_time / improved_time
                        if speedup > 1.1:
                            print(f"   🏆 SPEEDUP: {speedup:.2f}x faster than Dijkstra!")
                        elif speedup < 0.9:
                            print(f"   ⚠️  SLOWDOWN: {1/speedup:.2f}x slower than Dijkstra")
                        else:
                            print(f"   ➖ Similar performance to Dijkstra ({speedup:.2f}x)")
                
                # Memory cleanup
                del graph
                gc.collect()
                
            except MemoryError:
                print(f"❌ OUT OF MEMORY for {graph_name}")
                all_results[graph_name] = {"error": "Out of memory"}
            except Exception as e:
                print(f"❌ ERROR for {graph_name}: {e}")
                all_results[graph_name] = {"error": str(e)}
        
        return all_results
    
    def generate_summary_report(self, results: Dict) -> str:
        """Generate a comprehensive summary report."""
        report = []
        report.append("\n" + "="*80)
        report.append("🏁 LARGE SCALE PERFORMANCE ANALYSIS REPORT")
        report.append("="*80)
        
        successful_tests = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_tests:
            report.append("❌ NO SUCCESSFUL TESTS - All tests failed!")
            return "\n".join(report)
        
        # Performance analysis
        report.append("\n📈 PERFORMANCE ANALYSIS:")
        report.append("-" * 40)
        
        dijkstra_faster = 0
        improved_faster = 0
        similar_performance = 0
        
        speedups = []
        
        for graph_name, graph_results in successful_tests.items():
            if 'SSSP Improved' in graph_results and 'Dijkstra Baseline' in graph_results:
                improved = graph_results['SSSP Improved']
                dijkstra = graph_results['Dijkstra Baseline']
                
                if improved['success'] and dijkstra['success']:
                    speedup = dijkstra['runtime'] / improved['runtime']
                    speedups.append(speedup)
                    
                    if speedup > 1.1:
                        improved_faster += 1
                        status = f"🏆 {speedup:.2f}x FASTER"
                    elif speedup < 0.9:
                        dijkstra_faster += 1
                        status = f"⚠️  {1/speedup:.2f}x SLOWER"
                    else:
                        similar_performance += 1
                        status = f"➖ Similar ({speedup:.2f}x)"
                    
                    report.append(f"{graph_name:15} | {improved['runtime']:8.3f}s | {dijkstra['runtime']:8.3f}s | {status}")
        
        # Overall assessment
        report.append(f"\n🎯 OVERALL ASSESSMENT:")
        report.append(f"   Improved Algorithm Faster: {improved_faster} cases")
        report.append(f"   Dijkstra Faster:          {dijkstra_faster} cases") 
        report.append(f"   Similar Performance:      {similar_performance} cases")
        
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            max_speedup = max(speedups)
            min_speedup = min(speedups)
            
            report.append(f"\n📊 SPEEDUP STATISTICS:")
            report.append(f"   Average Speedup: {avg_speedup:.2f}x")
            report.append(f"   Best Speedup:    {max_speedup:.2f}x")
            report.append(f"   Worst Speedup:   {min_speedup:.2f}x")
            
            # Make the call
            if avg_speedup > 1.5:
                report.append(f"\n🎉 VERDICT: SIGNIFICANT PERFORMANCE IMPROVEMENT!")
                report.append(f"   Your algorithm shows clear advantages on large graphs.")
                report.append(f"   This is commercially viable performance improvement.")
            elif avg_speedup > 1.1:
                report.append(f"\n✅ VERDICT: MODERATE PERFORMANCE IMPROVEMENT")
                report.append(f"   Your algorithm shows consistent but modest improvements.")
                report.append(f"   Good academic result, potential commercial interest.")
            elif avg_speedup > 0.9:
                report.append(f"\n➖ VERDICT: SIMILAR PERFORMANCE")
                report.append(f"   Your algorithm performs comparably to Dijkstra.")
                report.append(f"   Good implementation but no clear advantage.")
            else:
                report.append(f"\n⚠️  VERDICT: PERFORMANCE REGRESSION")
                report.append(f"   Your algorithm is generally slower than Dijkstra.")
                report.append(f"   Implementation issues or algorithm limitations.")
        
        # Memory analysis
        report.append(f"\n💾 MEMORY USAGE ANALYSIS:")
        report.append("-" * 40)
        
        for graph_name, graph_results in successful_tests.items():
            memory_info = []
            for alg_name in ['Dijkstra Baseline', 'SSSP Original', 'SSSP Improved']:
                if alg_name in graph_results and graph_results[alg_name]['success']:
                    mem = graph_results[alg_name]['memory_mb']
                    memory_info.append(f"{alg_name}: {mem:.1f}MB")
            
            if memory_info:
                report.append(f"{graph_name:15} | {' | '.join(memory_info)}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)

def main():
    """Run the large scale test suite."""
    print("🔬 LARGE SCALE SSSP ALGORITHM VALIDATION")
    print("This is the make-or-break test for real-world performance claims!")
    print()
    
    # Check system resources
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"💻 System: {memory_gb:.1f}GB RAM available")
    
    if memory_gb < 8:
        print("⚠️  Warning: Less than 8GB RAM. Large tests may fail.")
    
    # Confirm before running
    response = input("🚨 This will test graphs up to 100K nodes. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return
    
    tester = LargeScaleTester()
    
    print("\n⏱️  Starting large scale tests (this may take several minutes)...")
    results = tester.run_scalability_test()
    
    # Generate and display report
    report = tester.generate_summary_report(results)
    print(report)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"large_scale_results_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("LARGE SCALE SSSP PERFORMANCE TEST RESULTS\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write("Raw Results:\n")
        f.write("-"*40 + "\n")
        
        for graph_name, graph_results in results.items():
            f.write(f"\n{graph_name}:\n")
            for alg_name, alg_data in graph_results.items():
                f.write(f"  {alg_name}: {alg_data}\n")
        
        f.write("\n\n" + report)
    
    print(f"\n💾 Full results saved to: {filename}")
    print("\n🎯 NEXT STEPS:")
    print("1. If performance is significantly better: Pursue commercial validation")
    print("2. If performance is similar: Focus on methodology and academic publication") 
    print("3. If performance is worse: Debug implementation or revisit algorithm")

if __name__ == "__main__":
    main()