from mathematical_attractors import MathematicalAttractorExplorer, PrimeAttractor
from sympy import symbols, zeta

# Create the comprehensive explorer
explorer = MathematicalAttractorExplorer()

# Run the full analysis
print("Starting comprehensive mathematical attractor analysis...")
results = explorer.run_comprehensive_analysis()

# Additional analysis of specific attractors
print("\n" + "="*50)
print("DETAILED ATTRACTOR ANALYSIS")
print("="*50)

# Analyze specific attractors in detail
for attractor in explorer.engine.attractors:
    if attractor.name in ["riemann_zeta", "sphere", "exponential"]:
        print(f"\n--- {attractor.name.upper()} ---")
        print(f"Initial: {attractor.history[0]}")
        print(f"Final: {attractor.expr}")
        print(f"History length: {len(attractor.history)}")
        print(f"Curvature trace: {attractor.curvature_trace}")
        print(f"Symmetries: {attractor.symmetries}")

# Test prime attractor specifically
print("\n--- PRIME ATTRACTOR TEST ---")
s = symbols('s')
prime_attractor = PrimeAttractor(zeta(s), "test_prime")
prime_attractor.evaluate_at_primes(max_prime=10)
print("Prime evaluations:")
for prime, value in prime_attractor.prime_values:
    print(f"  ζ({prime}) = {value}")

# Summary statistics
print(f"\n--- SUMMARY STATISTICS ---")
print(f"Total attractors: {len(explorer.engine.attractors)}")
print(f"Fixed points found: {len(results['fixed_points'])}")
print(f"Critical points: {len(results['critical_points'])}")
print(f"Zero curvature points: {len(results['zero_curvature'])}")
print(f"Symmetry types: {len(results['symmetries'])}")

# Convergence analysis
convergence_types = {}
for behavior in results['convergence'].values():
    convergence_types[behavior] = convergence_types.get(behavior, 0) + 1

print(f"\nConvergence patterns:")
for behavior, count in convergence_types.items():
    print(f"  {behavior}: {count} attractors") 