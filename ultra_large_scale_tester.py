#!/usr/bin/env python3
"""
Ultra Large Scale SSSP Testing Suite
Handles graphs up to millions of nodes with safe UTF-8 output.
"""

import os, time, random, math, gc
from collections import defaultdict
from typing import Dict, List, Tuple
import psutil

try:
    from enhanced_sssp import SSSSPSolver, BenchmarkSuite
    HAS_ENHANCED = True
except ImportError:
    print("❌ enhanced_sssp.py not found!")
    exit(1)

class UltraLargeScaleTester:
    """Test SSSP algorithms on ultra-large graphs."""
    
    def __init__(self):
        self.results = {}
    
    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    # --- Graph Generators ---
    def create_chip_routing_graph(self, n: int, avg_degree: int, locality_bias: float = 0.8):
        print(f"🏗️  Generating chip routing graph: {n:,} nodes, avg degree {avg_degree}")
        graph = {i: [] for i in range(n)}
        edges = set()
        total_edges = 0
        target_edges = n * avg_degree // 2
        cluster_size = int(math.sqrt(n))
        clusters = [list(range(i, min(i + cluster_size, n))) for i in range(0, n, cluster_size)]

        for cluster in clusters:
            for node in cluster:
                degree = random.randint(max(1, avg_degree - 3), avg_degree + 3)
                connections = 0
                attempts = 0
                while connections < degree and attempts < degree * 20:
                    target = random.choice(cluster) if random.random() < locality_bias else random.randint(0, n - 1)
                    if (target != node and (node, target) not in edges and (target, node) not in edges and total_edges < target_edges):
                        weight = random.uniform(0.1, 5.0)
                        if abs(node - target) > cluster_size:
                            weight *= random.uniform(1.5, 3.0)
                        graph[node].append((target, weight))
                        graph[target].append((node, weight))
                        edges.add((min(node, target), max(node, target)))
                        connections += 1
                        total_edges += 1
                    attempts += 1
        print(f"✅ Generated graph with {total_edges:,} edges (avg degree: {2*total_edges/n:.1f})")
        return graph

    def create_scale_free_graph(self, n: int, m: int):
        print(f"🌐 Generating scale-free graph: {n:,} nodes, {m} edges per new node")
        graph = defaultdict(list)
        degrees = defaultdict(int)
        initial_nodes = min(m + 1, 10)
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                w = random.uniform(0.5, 2.0)
                graph[i].append((j, w))
                graph[j].append((i, w))
                degrees[i] += 1
                degrees[j] += 1
        for new_node in range(initial_nodes, n):
            targets = []
            candidates = list(range(new_node))
            for _ in range(min(m, len(candidates))):
                if random.random() < 0.7 and candidates:
                    weights = [degrees[node] + 1 for node in candidates]
                    total_weight = sum(weights)
                    r = random.uniform(0, total_weight)
                    cumsum = 0
                    for i, w in enumerate(weights):
                        cumsum += w
                        if cumsum >= r:
                            targets.append(candidates.pop(i))
                            break
                elif candidates:
                    targets.append(candidates.pop(random.randint(0, len(candidates) - 1)))
            for t in targets:
                weight = random.uniform(0.1, 3.0)
                graph[new_node].append((t, weight))
                graph[t].append((new_node, weight))
                degrees[new_node] += 1
                degrees[t] += 1
        return dict(graph)

    # --- Benchmark ---
    def benchmark_single_graph(self, graph, name, source=0):
        n = len(graph)
        m = sum(len(adj) for adj in graph.values()) // 2
        print(f"\n🧪 Testing {name} | {n:,} nodes, {m:,} edges")
        results = {}
        algorithms = [
            ("Dijkstra Baseline", lambda: BenchmarkSuite.dijkstra_baseline(graph, source)),
            ("SSSP Original", lambda: SSSSPSolver(use_improved=False).solve(graph, source, n, m)[0]),
            ("SSSP Improved", lambda: SSSSPSolver(use_improved=True).solve(graph, source, n, m)[0])
        ]
        for alg_name, alg_func in algorithms:
            print(f"   🔄 Running {alg_name}...", end=" ")
            gc.collect()
            mem_before = self.get_memory_usage()
            try:
                start = time.time()
                distances = alg_func()
                end = time.time()
                mem_after = self.get_memory_usage()
                runtime = end - start
                reachable = len([d for d in distances.values() if d != float('inf')])
                results[alg_name] = {
                    "runtime": runtime,
                    "reachable_nodes": reachable,
                    "memory_mb": mem_after - mem_before,
                    "success": True
                }
                print(f"✅ {runtime:.3f}s ({reachable:,} nodes, {mem_after-mem_before:.1f}MB)")
            except Exception as e:
                print(f"❌ FAILED: {e}")
                results[alg_name] = {"runtime": float('inf'), "reachable_nodes": 0, "memory_mb": float('inf'), "error": str(e), "success": False}
        return results

    # --- Run tests ---
    def run_scalability_test(self):
        print("🚀 ULTRA LARGE SCALE TEST")
        test_cases = [
            (100_000, 8, "chip_routing"),
            (500_000, 6, "chip_routing"),
            (1_000_000, 5, "chip_routing"),
            (2_000_000, 4, "chip_routing"),
            (100_000, 5, "scale_free"),
            (500_000, 3, "scale_free"),
            (1_000_000, 2, "scale_free")
        ]
        all_results = {}
        for n, deg, typ in test_cases:
            graph_name = f"{typ}_{n//1000}k"
            print(f"\n{'='*20} {graph_name.upper()} {'='*20}")
            try:
                graph = self.create_chip_routing_graph(n, deg) if typ=="chip_routing" else self.create_scale_free_graph(n, deg)
                res = self.benchmark_single_graph(graph, graph_name)
                all_results[graph_name] = res
                del graph
                gc.collect()
            except MemoryError:
                print(f"❌ OUT OF MEMORY for {graph_name}")
                all_results[graph_name] = {"error": "Out of memory"}
            except Exception as e:
                print(f"❌ ERROR for {graph_name}: {e}")
                all_results[graph_name] = {"error": str(e)}
        return all_results

    # --- Summary ---
    def generate_summary_report(self, results):
        report = ["\n" + "="*80, "🏁 ULTRA LARGE SCALE PERFORMANCE REPORT", "="*80]
        for gname, gres in results.items():
            if 'error' in gres:
                report.append(f"{gname}: {gres['error']}")
                continue
            for alg in ["Dijkstra Baseline", "SSSP Original", "SSSP Improved"]:
                if alg in gres:
                    r = gres[alg]
                    report.append(f"{gname:15} | {alg:15} | {r['runtime']:.3f}s | {r['reachable_nodes']:,} nodes | {r['memory_mb']:.1f}MB")
        report.append("="*80)
        return "\n".join(report)

def main():
    print("🔬 ULTRA LARGE SCALE SSSP VALIDATION")
    memory_gb = psutil.virtual_memory().total / 1024**3
    print(f"💻 System: {memory_gb:.1f}GB RAM available")
    if memory_gb < 16:
        print("⚠️ Warning: Less than 16GB RAM. Very large tests may fail.")
    response = input("🚨 Continue with 500k+ node graphs? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled."); return
    tester = UltraLargeScaleTester()
    print("\n⏱️  Starting ultra-large scale tests (this may take a while)...")
    results = tester.run_scalability_test()
    report = tester.generate_summary_report(results)
    print(report)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"ultra_large_scale_results_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n💾 Full results saved to: {filename}")

if __name__ == "__main__":
    main()
