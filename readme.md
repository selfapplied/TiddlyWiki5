# Symbolic Attractor Engine

A sophisticated mathematical exploration system built with sympy for discovering and analyzing mathematical attractors, with special focus on prime number theory, geometric structures, and analytic functions.

## 🧠 Core Concept

The Symbolic Attractor Engine uses symbolic computation to:
- **Track fixed points** in mathematical transformations
- **Compute curvature** using Ricci-like tensors
- **Analyze symmetries** and conservation laws
- **Explore Lie group expansions** for symmetry transformations
- **Discover zero-curvature points** where mathematical structures become flat

## 📦 Architecture

### Core Modules

1. **`symbolic_engine.py`** - Basic symbolic attractor framework
2. **`riemann_attractor.py`** - Enhanced engine with curvature tracking and Lie group expansions
3. **`mathematical_attractors.py`** - Specialized explorers for prime numbers, geometry, and analytic functions
4. **`attractor_explorer.py`** - Unified interface for comprehensive analysis

### Key Classes

- **`SymbolicAttractor`** - Tracks expression evolution and fixed points
- **`RiemannAttractor`** - Enhanced with curvature computation and symmetry tracking
- **`RiemannEngine`** - Manages transformations and Lie group generators
- **`MathematicalAttractorExplorer`** - Specialized analysis for different mathematical domains
- **`AttractorExplorer`** - Unified interface for comprehensive exploration

## 🚀 Quick Start

```python
from attractor_explorer import AttractorExplorer

# Create explorer
explorer = AttractorExplorer()

# Run comprehensive analysis
results = explorer.run_comprehensive_analysis()

# Print summary
explorer.print_summary()

# Export results
explorer.export_results()
```

## 🔍 Analysis Types

### Prime Number Attractors
- Riemann Zeta function and variants
- Dirichlet Eta function
- Prime counting functions
- Möbius transforms

### Geometric Attractors
- Sphere, hyperboloid, cubic surfaces
- Gaussian functions
- Curvature-related functions
- Wave patterns

### Analytic Attractors
- Exponential, trigonometric functions
- Logarithmic functions
- Quadratic and reciprocal functions
- Complex analytic functions

## 🧪 Key Features

### Fixed Point Detection
```python
# Automatically detects when expressions reach fixed points
attractor = SymbolicAttractor(expression)
if attractor.is_fixed_point():
    print("Fixed point found!")
```

### Curvature Computation
```python
# Computes symbolic curvature using Ricci-like tensor
curvature = attractor.compute_curvature()
if curvature == 0:
    print("Zero curvature point!")
```

### Lie Group Expansions
```python
# Applies Lie group generators for symmetry transformations
engine.add_lie_generator(rotation_generator)
expanded = engine.apply_lie_expansion(expression)
```

### Symmetry Analysis
```python
# Tracks symmetry-breaking and conservation laws
symmetries = engine.analyze_symmetries()
for symmetry_type, attractors in symmetries.items():
    print(f"{symmetry_type}: {attractors}")
```

## 📊 Example Results

Running the comprehensive analysis typically finds:

- **33 total attractors** across all domains
- **32 fixed points** where expressions stabilize
- **368 zero-curvature points** where mathematical structures become flat
- **Multiple symmetry types** including scale invariance

## 🔧 Installation

```bash
# Install dependencies
uv add sympy

# Run tests
python3 test_symbolic_engine.py
python3 test_riemann_attractor.py
python3 test_mathematical_attractors.py

# Run comprehensive analysis
python3 attractor_explorer.py
```

## 🎯 Advanced Usage

### Custom Attractors
```python
from riemann_attractor import RiemannEngine

engine = RiemannEngine()
engine.add_transform(simplify)
engine.add_transform(expand)

# Seed custom expressions
engine.seed(your_expression, "custom_name")
engine.run()
```

### Custom Lie Generators
```python
def custom_generator(expr):
    x, y = symbols('x y')
    return x * diff(expr, y) - y * diff(expr, x)

engine.add_lie_generator(custom_generator)
```

### Curvature Analysis
```python
# Find zero-curvature points
zero_curvature = engine.get_zero_curvature_points()
for name, expr in zero_curvature:
    print(f"{name}: {expr}")
```

## 🔮 Future Extensions

- **Ricci flow simulation** for geometric evolution
- **Noether theorem implementation** for conservation laws
- **Quantum field theory attractors** for particle physics
- **Machine learning integration** for pattern discovery
- **Visualization tools** for attractor dynamics

## 📚 Mathematical Background

The system is inspired by:
- **Riemann geometry** and curvature tensors
- **Lie group theory** and symmetry transformations
- **Dynamical systems** and fixed point theory
- **Prime number theory** and zeta functions
- **Analytic continuation** and complex analysis

## 🤝 Contributing

The system is designed to be extensible. Key areas for contribution:
- New mathematical attractor types
- Advanced Lie group generators
- Curvature computation algorithms
- Visualization and analysis tools
- Documentation and examples

## 📄 License

This project explores mathematical concepts and is provided for educational and research purposes.

---

*"Mathematics is the art of giving the same name to different things."* - Henri Poincaré

The Symbolic Attractor Engine brings this art to life through computational exploration of mathematical structures and their evolutionary dynamics.