#!/usr/bin/env python3
"""
Simple debug test to verify the fixes work with debug_integration.py
"""

def test_return_formats():
    """Test that all functions return the expected formats."""
    try:
        from enhanced_sssp import SSSSPSolver, BenchmarkSuite, dijkstra_reference
        
        # Test graph
        graph = {0: [(1, 1), (2, 4)], 1: [(2, 2), (3, 5)], 2: [(3, 1)], 3: []}
        
        print("Testing return formats...")
        
        # Test dijkstra_reference
        result1 = dijkstra_reference(graph, 0)
        print(f"dijkstra_reference returns: {type(result1)} - {'✅' if isinstance(result1, dict) else '❌'}")
        
        # Test BenchmarkSuite.dijkstra_baseline
        result2 = BenchmarkSuite.dijkstra_baseline(graph, 0)
        print(f"BenchmarkSuite.dijkstra_baseline returns: {type(result2)} - {'✅' if isinstance(result2, dict) else '❌'}")
        
        # Test SSSSPSolver.solve
        solver = SSSSPSolver(use_improved=False)
        result3 = solver.solve(graph, 0, 4, 4, visualize=False)
        print(f"SSSSPSolver(improved=False).solve returns: {type(result3)} - {'✅' if isinstance(result3, dict) else '❌'}")
        
        solver = SSSSPSolver(use_improved=True)
        result4 = solver.solve(graph, 0, 4, 4, visualize=False)
        print(f"SSSSPSolver(improved=True).solve returns: {type(result4)} - {'✅' if isinstance(result4, dict) else '❌'}")
        
        # Test correctness
        expected = {0: 0, 1: 1, 2: 3, 3: 4}
        print(f"\nExpected distances: {expected}")
        print(f"dijkstra_reference:  {result1}")
        print(f"SSSP original:       {result3}")  
        print(f"SSSP improved:       {result4}")
        
        # Check if all match
        all_correct = True
        for results, name in [(result1, "dijkstra_reference"), (result3, "SSSP original"), (result4, "SSSP improved")]:
            for node in expected:
                if abs(results.get(node, float('inf')) - expected[node]) > 1e-9:
                    print(f"❌ {name} incorrect for node {node}: got {results.get(node)}, expected {expected[node]}")
                    all_correct = False
        
        if all_correct:
            print("✅ All algorithms return correct results!")
        else:
            print("❌ Some algorithms have incorrect results")
            
        print(f"\n{'✅ ALL TESTS PASSED' if all_correct else '❌ SOME TESTS FAILED'}")
        print("The debug_integration.py should now work correctly!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_large_chain():
    """Test the linear chain that was failing."""
    try:
        from enhanced_sssp import SSSSPSolver
        
        print("\n" + "="*50)
        print("Testing 100-node linear chain (the failing case)...")
        
        # Create 100-node chain: 0->1->2->...->99
        chain_graph = {}
        for i in range(100):
            if i < 99:
                chain_graph[i] = [(i+1, 1)]
            else:
                chain_graph[i] = []
        
        solver = SSSSPSolver(use_improved=True)
        result = solver.solve(chain_graph, 0, 100, 99, visualize=False)
        
        print(f"Chain algorithm returned {len(result)} nodes")
        reachable = len([d for d in result.values() if d != float('inf')])
        print(f"Reachable nodes: {reachable}/100")
        
        # Check specific nodes that were failing
        problem_nodes = [11, 12, 15, 20, 50, 99]
        print("\nChecking problem nodes:")
        for node in problem_nodes:
            expected = node
            actual = result.get(node, float('inf'))
            status = "✅" if abs(actual - expected) < 1e-9 else "❌"
            print(f"  Node {node:2d}: Expected={expected:2d}, Got={actual}, {status}")
        
        if reachable == 100:
            print("✅ Linear chain test PASSED!")
        else:
            print("❌ Linear chain test still failing")
            
    except Exception as e:
        print(f"❌ Error testing linear chain: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 TESTING FIXES FOR debug_integration.py COMPATIBILITY")
    print("="*60)
    
    test_return_formats()
    test_large_chain()
    
    print("\n" + "="*60)
    print("Now run: python debug_integration.py")
    print("It should work without the AttributeError!")