from sympy import symbols, zeta, simplify, expand, sin, cos, pi, diff
from symbolic_engine import SymbolicEngine

x = symbols('x')
engine = SymbolicEngine()

# Add transforms
engine.add_transform(simplify)
engine.add_transform(expand)
engine.add_transform(lambda e: diff(e, x))

# Seed with wild attractors
engine.seed(zeta(x))
engine.seed(sin(x)**2 + cos(x)**2)
engine.seed((x + 1)**3)

# Run it
engine.run()

# Show attractor fixed points
for attractor in engine.get_fixed_points():
    print("Fixed point found:", attractor.expr) 