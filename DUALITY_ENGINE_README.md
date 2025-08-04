# Symbolic Duality Engine

A comprehensive symbolic mathematics framework for studying duality transformations in mathematical functions, with special focus on zeta functions and their functional equations.

## 🎯 Overview

The Symbolic Duality Engine provides a complete toolkit for analyzing mathematical functions under the duality transformation `s ↔ 1-s`. This framework is particularly powerful for studying:

- **Riemann's functional equation** for the zeta function
- **Fixed points** and **attractor dynamics** under duality
- **Critical line behavior** of complex functions
- **Functional equation attractors** in mathematical function spaces

## 🏗️ Architecture

The system consists of three main components:

### 1. Basic Duality Engine (`symbolic_duality_engine.py`)
Core functionality for duality transformations and basic analysis.

### 2. Advanced Duality Analyzer (`advanced_duality_analysis.py`)
Sophisticated tools for fixed point analysis, attractor dynamics, and complex transformations.

### 3. Research Framework (`duality_research_framework.py`)
Comprehensive research platform that integrates all analysis tools.

## 🚀 Quick Start

```python
from symbolic_duality_engine import DualityEngine

# Create engine
engine = DualityEngine()

# Apply duality transformation
f = zeta(s)
f_dual = engine.dual_map(f)  # f(1-s)

# Check if function is a fixed point
is_fixed = engine.fixed_point_test(f)

# Study critical line behavior
critical_behavior = f.subs(s, S.Half + I*t)
```

## 📊 Core Features

### Duality Transformations
```python
# Basic duality map: f(s) → f(1-s)
dual_func = engine.dual_map(func)

# Test fixed points: f(s) = f(1-s)
is_fixed = engine.fixed_point_test(func)
```

### Critical Line Analysis
```python
# Check if point is on critical line Re(s) = 1/2
critical_point = S.Half + I*t
on_critical = engine.is_on_critical_line(critical_point)

# Study behavior on critical line
critical_behavior = func.subs(s, S.Half + I*t)
```

### Zeta Function Analysis
```python
# Riemann's functional equation
functional_eq = engine.zeta_functional_equation()
# Returns: ζ(s) = χ(s) · ζ(1-s)

# χ(s) factor
chi_factor = engine.chi_factor()
# Returns: 2^s · π^(s-1) · sin(πs/2) · Γ(1-s)
```

## 🔬 Advanced Analysis

### Fixed Point Analysis
```python
from advanced_duality_analysis import AdvancedDualityAnalyzer

analyzer = AdvancedDualityAnalyzer()

# Find fixed points under repeated duality
fixed_points = analyzer.find_duality_fixed_points(func)

# Analyze attractor dynamics
attractors = analyzer.analyze_attractor_dynamics(func)
```

### Critical Line Zeta Analysis
```python
# Specialized analysis for zeta function on critical line
critical_analysis = analyzer.critical_line_zeta_analysis()

# Access components:
zeta_critical = critical_analysis['critical_zeta']  # ζ(1/2 + it)
dual_critical = critical_analysis['dual_critical_zeta']  # ζ(1/2 - it)
chi_critical = critical_analysis['chi_critical']  # χ(1/2 + it)
```

### Transformation Analysis
```python
# Study various duality transformations
transformations = analyzer.create_duality_transformations()
results = analyzer.study_transformation_fixed_points(func)
```

## 🌌 Research Framework

### Comprehensive Function Study
```python
from duality_research_framework import DualityResearchFramework

framework = DualityResearchFramework()

# Complete analysis of a function
analysis = framework.comprehensive_function_study(func, "my_function")

# Access results:
basic = analysis['basic_analysis']
advanced = analysis['advanced_analysis']
```

### Zeta Function Research Suite
```python
# Complete zeta function analysis
zeta_suite = framework.create_zeta_research_suite()

# Includes:
# - Basic zeta function
# - Zeta family (scaled, shifted, powered, etc.)
# - Critical line analysis
```

### Function Space Research
```python
# Analyze entire function spaces
function_space = framework.create_function_space_research()

# Pattern analysis
patterns = framework.analyze_duality_patterns(results)
```

## 📋 Mathematical Insights

### 1. Zeta Function Duality
The engine confirms that `ζ(s) ≠ ζ(1-s)`, requiring Riemann's functional equation:
```
ζ(s) = χ(s) · ζ(1-s)
where χ(s) = 2^s · π^(s-1) · sin(πs/2) · Γ(1-s)
```

### 2. Critical Line Symmetry
On the critical line `Re(s) = 1/2`, the functional equation becomes:
```
ζ(1/2 + it) = χ(1/2 + it) · ζ(1/2 - it)
```

### 3. Fixed Point Classification
The engine can identify functions that are fixed points under duality:
- `f(s) = f(1-s)` (even under duality)
- `f(s) = -f(1-s)` (odd under duality)
- `f(s) = conjugate(f(1-s))` (conjugate symmetry)

### 4. Attractor Dynamics
Analysis of convergence behavior under repeated duality transformations.

## 🎯 Key Applications

### 1. Function Classification
```python
# Classify functions by duality behavior
functions = {
    'Fixed Point': s**2 - s + 1,
    'Not Fixed Point': exp(s),
    'Zeta Function': zeta(s)
}

for name, func in functions.items():
    is_fixed = engine.fixed_point_test(func)
    print(f"{name}: {'✓' if is_fixed else '✗'}")
```

### 2. Critical Line Behavior
```python
# Study behavior on critical line
critical_functions = [zeta(s), gamma(s), exp(s), sin(s)]

for func in critical_functions:
    critical_behavior = func.subs(s, S.Half + I*t)
    print(f"{func}: {critical_behavior}")
```

### 3. Functional Equation Verification
```python
# Verify Riemann's functional equation
functional_eq = engine.zeta_functional_equation()
print(f"Riemann's equation: {functional_eq}")
```

### 4. Duality Transformations
```python
# Apply various duality transformations
test_func = s**2 + 2*s + 1
dual_func = engine.dual_map(test_func)
print(f"Original: {test_func}")
print(f"Dual: {simplify(dual_func)}")
```

## 📊 Research Capabilities

### Pattern Analysis
- **Fixed Points**: Identify functions invariant under duality
- **Attractors**: Study convergence behavior
- **Symmetries**: Analyze various symmetry properties
- **Functional Equations**: Discover and verify functional equations

### Export and Reporting
```python
# Export results to JSON
framework.export_results(results, "research_results.json")

# Generate research reports
report = framework.generate_research_report(analysis_results)
```

## 🧪 Demonstration Scripts

### Basic Demo
```bash
python3 symbolic_duality_engine.py
```

### Advanced Analysis Demo
```bash
python3 advanced_duality_analysis.py
```

### Research Framework Demo
```bash
python3 duality_research_framework.py
```

### Complete Demonstration
```bash
python3 duality_engine_demo.py
```

## 📈 Example Output

```
🧠 Symbolic Duality Engine Demo
==================================================

📐 Phase 1: Foundation – Dual Map & Critical Line
Original: s**2 + 2*s + 1
Dual: -2*s + (1 - s)**2 + 3

📊 Phase 2: Zeta Function Analysis
Zeta function: zeta(s)
Zeta dual: zeta(1 - s)
Is fixed point: False

🔗 Phase 3: Functional Equation Attractor
χ(s) factor: 2**s*pi**(s - 1)*sin(pi*s/2)*gamma(1 - s)
Riemann's functional equation: Eq(zeta(s), χ(s)*zeta(1 - s))
```

## 🔬 Mathematical Foundation

### Duality Map
The core transformation is `s ↔ 1-s`, which maps:
- `s = 0` → `s = 1`
- `s = 1` → `s = 0`
- `s = 1/2` → `s = 1/2` (fixed point)

### Critical Line
The line `Re(s) = 1/2` is special because:
- It's invariant under duality: `1/2 + it` → `1/2 - it`
- It's the conjectured location of all non-trivial zeta zeros
- It's the axis of symmetry for the functional equation

### Functional Equations
For the zeta function:
```
ζ(s) = χ(s) · ζ(1-s)
where χ(s) = 2^s · π^(s-1) · sin(πs/2) · Γ(1-s)
```

## 🎯 Future Extensions

The framework is designed to be extensible for:

1. **Generalized Zeta Functions**: L-functions, Dirichlet L-functions
2. **Higher-Dimensional Duality**: Multi-variable transformations
3. **Numerical Analysis**: Integration with numerical methods
4. **Visualization**: Plotting capabilities for duality behavior
5. **Machine Learning**: Pattern recognition in duality spaces

## 📚 Dependencies

- **sympy**: Symbolic mathematics library
- **json**: Data export capabilities
- **typing**: Type hints for better code organization

## 🚀 Getting Started

1. Install dependencies:
```bash
pip install sympy
```

2. Run the basic demo:
```bash
python3 symbolic_duality_engine.py
```

3. Explore advanced features:
```bash
python3 advanced_duality_analysis.py
```

4. Run comprehensive research:
```bash
python3 duality_research_framework.py
```

## 🎉 Conclusion

The Symbolic Duality Engine provides a powerful framework for studying duality transformations in mathematical functions. It's particularly valuable for:

- **Research**: Comprehensive analysis of function spaces
- **Education**: Understanding duality concepts
- **Discovery**: Finding new patterns and relationships
- **Verification**: Confirming known mathematical results

The framework successfully demonstrates the duality structure of the zeta function and provides tools for exploring similar patterns in other mathematical functions. 