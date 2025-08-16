#!/usr/bin/env python3
"""
Integration test to debug your SSSP algorithm using the existing framework.
"""

import os
import time
import random
import math
from typing import Dict, List, Tuple

# Import your existing code
try:
    from enhanced_sssp import SSSSPSolver, BenchmarkSuite, sssp_directed, dijkstra_reference
    HAS_ENHANCED = True
except ImportError:
    print("Could not import enhanced_sssp.py - make sure it's in the same directory")
    exit(1)

# Debugging instrumentation
class SSSSPDebugger:
    def __init__(self):
        self.stats = {
            'popped_count': 0,
            'relax_count': 0,
            'pruned_edges': 0,
            'pq_size_peak': 0,
            'stopping_reason': 'unknown',
            'stale_pops': 0,
            'bmss_recursions': 0,
            'base_case_calls': 0,
            'pivot_count': 0
        }
    
    def reset_stats(self):
        for key in self.stats:
            if key == 'stopping_reason':
                self.stats[key] = 'unknown'
            else:
                self.stats[key] = 0

def validate_exactness(graph: Dict[int, List[Tuple[int, float]]], source: int, 
                      test_distances: Dict[int, float], algorithm_name: str = "test"):
    """Validate algorithm results against Dijkstra."""
    ref_distances = dijkstra_reference(graph, source)
    
    errors = []
    for node in graph:
        ref_dist = ref_distances[node]
        test_dist = test_distances.get(node, float('inf'))
        
        if ref_dist < float('inf'):
            if abs(ref_dist - test_dist) > 1e-9:
                errors.append((node, ref_dist, test_dist))
    
    if errors:
        print(f"EXACTNESS FAILED for {algorithm_name}:")
        for i, (node, ref, test) in enumerate(errors[:10]):
            print(f"  Node {node}: Reference={ref:.6f}, Test={test:.6f}, Error={abs(ref-test):.6e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more errors")
        return False
    else:
        ref_reachable = sum(1 for d in ref_distances.values() if d < float('inf'))
        test_reachable = sum(1 for d in test_distances.values() if d < float('inf'))
        print(f"EXACTNESS TEST for {algorithm_name}: {test_reachable}/{ref_reachable} nodes reachable")
        
        if test_reachable < ref_reachable:
            print(f"  WARNING: Missing {ref_reachable - test_reachable} reachable nodes!")
            return False
        return True

def create_debug_graph(n: int, connectivity: str = "sparse") -> Dict[int, List[Tuple[int, float]]]:
    """Create test graphs similar to your chip routing graphs."""
    graph = {i: [] for i in range(n)}
    
    if connectivity == "sparse":
        # Create connected graph with low degree (like chip routing)
        for i in range(n-1):
            weight = random.uniform(0.5, 2.0)
            graph[i].append((i+1, weight))
            graph[i+1].append((i, weight))
        
        # Add some random edges for realism
        edges_added = 0
        target_edges = n * 3  # Average degree ~6
        
        while edges_added < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and v not in [x[0] for x in graph[u]]:
                weight = random.uniform(0.1, 3.0)
                graph[u].append((v, weight))
                edges_added += 1
    
    elif connectivity == "dense":
        # Higher connectivity
        for i in range(n):
            degree = random.randint(8, 15)
            targets = random.sample([j for j in range(n) if j != i], 
                                  min(degree, n-1))
            for j in targets:
                if j not in [x[0] for x in graph[i]]:
                    weight = random.uniform(0.1, 4.0)
                    graph[i].append((j, weight))
    
    return graph

def run_comprehensive_debug(graph_sizes: List[int] = [100, 500, 1000, 5000]):
    """Run comprehensive debugging on your algorithm."""
    print("COMPREHENSIVE SSSP DEBUGGING")
    print("=" * 60)
    
    random.seed(42)  # Reproducible results
    
    for n in graph_sizes:
        print(f"\nTesting graph size: {n} nodes")
        print("-" * 40)
        
        # Create test graph
        graph = create_debug_graph(n, "sparse")
        total_edges = sum(len(adj_list) for adj_list in graph.values())
        print(f"Generated graph: {n} nodes, {total_edges} edges (avg degree: {total_edges/n:.1f})")
        
        source = 0
        
        # Test 1: Reference Dijkstra
        start_time = time.time()
        ref_distances = dijkstra_reference(graph, source)
        ref_time = time.time() - start_time
        ref_reachable = sum(1 for d in ref_distances.values() if d < float('inf'))
        
        print(f"Reference Dijkstra: {ref_time:.4f}s, {ref_reachable} reachable nodes")
        
        # Test 2: Your "original" algorithm
        solver_orig = SSSSPSolver(use_improved=False)
        start_time = time.time()
        orig_distances = solver_orig.solve(graph, source, n, total_edges)
        orig_time = time.time() - start_time
        orig_reachable = sum(1 for d in orig_distances.values() if d < float('inf'))
        
        print(f"SSSP Original: {orig_time:.4f}s, {orig_reachable} reachable nodes")
        validate_exactness(graph, source, orig_distances, "SSSP Original")
        
        # Test 3: Your "improved" algorithm (the problematic one)
        solver_imp = SSSSPSolver(use_improved=True)
        start_time = time.time()
        imp_distances = solver_imp.solve(graph, source, n, total_edges)
        imp_time = time.time() - start_time
        imp_reachable = sum(1 for d in imp_distances.values() if d < float('inf'))
        
        print(f"SSSP Improved: {imp_time:.4f}s, {imp_reachable} reachable nodes")
        is_exact = validate_exactness(graph, source, imp_distances, "SSSP Improved")
        
        # Analysis
        if imp_reachable < ref_reachable:
            coverage = (imp_reachable / ref_reachable) * 100
            print(f"  PROBLEM: Only {coverage:.1f}% coverage!")
            print(f"  Missing {ref_reachable - imp_reachable} reachable nodes")
            
            # Find some missing nodes
            missing_nodes = []
            for node in graph:
                if (ref_distances[node] < float('inf') and 
                    imp_distances.get(node, float('inf')) == float('inf')):
                    missing_nodes.append((node, ref_distances[node]))
            
            if missing_nodes:
                print(f"  Sample missing nodes:")
                for node, dist in sorted(missing_nodes, key=lambda x: x[1])[:5]:
                    print(f"    Node {node} at distance {dist:.3f}")
        
        elif is_exact:
            speedup = ref_time / imp_time if imp_time > 0 else float('inf')
            print(f"  SUCCESS: Exact result with {speedup:.2f}x speedup!")
        
        print()

def test_specific_failure_cases():
    """Test cases that commonly expose SSSP bugs."""
    print("TESTING SPECIFIC FAILURE CASES")
    print("=" * 60)
    
    # Test 1: Single node
    graph1 = {0: []}
    print("Test 1: Single node graph")
    test_distances = sssp_directed(graph1, 0, 1, 0)
    validate_exactness(graph1, 0, test_distances, "Single node")
    
    # Test 2: Disconnected graph
    graph2 = {0: [(1, 1.0)], 1: [], 2: [(3, 1.0)], 3: []}
    print("\nTest 2: Disconnected graph")
    test_distances = sssp_directed(graph2, 0, 4, 2)
    validate_exactness(graph2, 0, test_distances, "Disconnected")
    
    # Test 3: Linear chain
    graph3 = {}
    n = 100
    for i in range(n):
        graph3[i] = []
        if i < n-1:
            graph3[i].append((i+1, 1.0))
    
    print(f"\nTest 3: Linear chain ({n} nodes)")
    test_distances = sssp_directed(graph3, 0, n, n-1)
    validate_exactness(graph3, 0, test_distances, "Linear chain")
    
    # Test 4: Star graph
    graph4 = {0: [(i, 1.0) for i in range(1, 50)]}
    for i in range(1, 50):
        graph4[i] = []
    
    print("\nTest 4: Star graph (50 nodes)")
    test_distances = sssp_directed(graph4, 0, 50, 49)
    validate_exactness(graph4, 0, test_distances, "Star graph")

def debug_your_algorithm():
    """Direct debugging of your sssp_directed function."""
    print("DIRECT ALGORITHM DEBUGGING")
    print("=" * 60)
    
    # Create a small graph where we can trace execution
    graph = {
        0: [(1, 1), (2, 4)],
        1: [(2, 2), (3, 5)],
        2: [(3, 1)],
        3: []
    }
    
    print("Test graph: 0->1(1), 0->2(4), 1->2(2), 1->3(5), 2->3(1)")
    print("Expected distances: {0:0, 1:1, 2:3, 3:4}")
    
    # Test your implementation
    result = sssp_directed(graph, 0, 4, 4)
    print(f"Your result: {result}")
    
    # Compare with reference
    reference = dijkstra_reference(graph, 0)
    print(f"Reference:   {reference}")
    
    # Check exactness
    validate_exactness(graph, 0, result, "sssp_directed")

if __name__ == "__main__":
    print("SSSP ALGORITHM DEBUGGING SUITE")
    print("Using your enhanced_sssp.py implementation")
    print("=" * 60)
    
    # Start with specific failure cases
    test_specific_failure_cases()
    
    # Debug your algorithm directly
    debug_your_algorithm()
    
    # Run comprehensive debugging
    run_comprehensive_debug([100, 500, 1000, 2000])
    
    print("\n" + "=" * 60)
    print("DEBUGGING COMPLETE")
    print("\nNext steps:")
    print("1. Look for patterns in the missing nodes")
    print("2. Add print statements to your bmss_p function to trace execution")
    print("3. Check if recursion depth limits are being hit")
    print("4. Verify that your BlockBasedDS is not losing data")