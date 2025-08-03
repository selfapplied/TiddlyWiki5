from sympy import symbols, zeta, simplify, expand, sin, cos, pi, diff, exp, I
from riemann_attractor import RiemannEngine, rotation_generator, scaling_generator, translation_generator

# Initialize the Riemann engine
engine = RiemannEngine()

# Add Lie generators
engine.add_lie_generator(rotation_generator)
engine.add_lie_generator(scaling_generator)
engine.add_lie_generator(translation_generator)

# Add transforms
engine.add_transform(simplify)
engine.add_transform(expand)
engine.add_transform(lambda e: diff(e, symbols('x')))

# Seed with mathematical attractors
x, y = symbols('x y')
engine.seed(zeta(x), "zeta_function")
engine.seed(sin(x)**2 + cos(x)**2, "trig_identity")
engine.seed((x + 1)**3, "cubic_polynomial")
engine.seed(x**2 + y**2, "circle_equation")
engine.seed(exp(I * x), "complex_exponential")

# Run the engine
print("Running Riemann attractor search...")
engine.run(max_iters=5)

# Analyze results
print("\n=== FIXED POINTS ===")
for attractor in engine.get_fixed_points():
    print(f"{attractor.name}: {attractor.expr}")

print("\n=== ZERO CURVATURE POINTS ===")
zero_curvature = engine.get_zero_curvature_points()
for name, expr in zero_curvature:
    print(f"{name}: {expr}")

print("\n=== SYMMETRY ANALYSIS ===")
symmetries = engine.analyze_symmetries()
for symmetry_type, attractors in symmetries.items():
    print(f"{symmetry_type}: {attractors}")

print("\n=== CURVATURE TRACES ===")
for attractor in engine.attractors:
    print(f"\n{attractor.name}:")
    for i, curvature in enumerate(attractor.curvature_trace):
        print(f"  Step {i}: {curvature}")

# Test Lie expansion
print("\n=== LIE EXPANSION TEST ===")
test_expr = x**2 + y**2
expanded = engine.apply_lie_expansion(test_expr)
for i, expansion in enumerate(expanded):
    print(f"Lie expansion {i}: {expansion}") 