#!/usr/bin/env python3
"""
Symbolic Attractor Engine Demo
==============================

A comprehensive demonstration of the symbolic attractor engine capabilities,
showing how it can discover mathematical patterns, fixed points, and symmetries.
"""

from sympy import symbols, zeta, simplify, expand, sin, cos, pi, diff, exp, I, log, sqrt
from sympy.abc import s, n, p, k
from riemann_attractor import RiemannEngine, rotation_generator, scaling_generator, translation_generator
from mathematical_attractors import MathematicalAttractorExplorer
from attractor_explorer import AttractorExplorer
import json

def demo_basic_attractors():
    """Demonstrate basic attractor functionality"""
    print("🔬 DEMO 1: Basic Attractor Discovery")
    print("="*50)
    
    engine = RiemannEngine()
    engine.add_transform(simplify)
    engine.add_transform(expand)
    
    # Test some classic mathematical identities
    x = symbols('x')
    test_expressions = [
        (sin(x)**2 + cos(x)**2, "Trig Identity"),
        ((x + 1)**3, "Cubic Polynomial"),
        (exp(I * x), "Complex Exponential"),
        (zeta(x), "Riemann Zeta")
    ]
    
    for expr, name in test_expressions:
        engine.seed(expr, name)
    
    engine.run(max_iters=5)
    
    print("Fixed Points Found:")
    for attractor in engine.get_fixed_points():
        print(f"  {attractor.name}: {attractor.expr}")
    
    print(f"\nTotal attractors: {len(engine.attractors)}")
    print(f"Fixed points: {len(engine.get_fixed_points())}")

def demo_curvature_analysis():
    """Demonstrate curvature computation and analysis"""
    print("\n🔬 DEMO 2: Curvature Analysis")
    print("="*50)
    
    engine = RiemannEngine()
    engine.add_transform(simplify)
    
    # Test geometric functions with different curvature properties
    x, y = symbols('x y')
    geometric_functions = [
        (x**2 + y**2, "Circle - Constant Curvature"),
        (x*y, "Hyperbolic - Variable Curvature"),
        (exp(-(x**2 + y**2)), "Gaussian - Smooth Curvature"),
        (1/(1 + x**2 + y**2), "Decay Function - Asymptotic Curvature")
    ]
    
    for expr, name in geometric_functions:
        engine.seed(expr, name)
    
    engine.run(max_iters=3)
    
    print("Curvature Analysis:")
    for attractor in engine.attractors:
        if attractor.curvature_trace:
            print(f"\n{attractor.name}:")
            for i, curvature in enumerate(attractor.curvature_trace):
                print(f"  Step {i}: {curvature}")
    
    # Find zero curvature points
    zero_curv = engine.get_zero_curvature_points()
    print(f"\nZero curvature points found: {len(zero_curv)}")

def demo_lie_group_expansions():
    """Demonstrate Lie group expansions and symmetry analysis"""
    print("\n🔬 DEMO 3: Lie Group Expansions")
    print("="*50)
    
    engine = RiemannEngine()
    
    # Add Lie generators
    engine.add_lie_generator(rotation_generator)
    engine.add_lie_generator(scaling_generator)
    engine.add_lie_generator(translation_generator)
    
    # Test with a simple expression
    x, y = symbols('x y')
    test_expr = x**2 + y**2
    
    print(f"Original expression: {test_expr}")
    
    # Apply Lie expansions
    expanded = engine.apply_lie_expansion(test_expr)
    
    print("Lie group expansions:")
    for i, expansion in enumerate(expanded):
        print(f"  Expansion {i+1}: {expansion}")

def demo_prime_number_attractors():
    """Demonstrate prime number theory attractors"""
    print("\n🔬 DEMO 4: Prime Number Attractors")
    print("="*50)
    
    explorer = MathematicalAttractorExplorer()
    explorer.setup_prime_attractors()
    
    # Add transforms
    explorer.engine.add_transform(simplify)
    # Fix the scope issue by defining s locally
    s_local = symbols('s')
    explorer.engine.add_transform(lambda e: e.subs(s_local, s_local + 1))
    
    explorer.engine.run(max_iters=5)
    
    print("Prime-related attractors:")
    for attractor in explorer.engine.attractors:
        print(f"  {attractor.name}: {attractor.expr}")
    
    # Test prime evaluation
    s = symbols('s')
    prime_attractor = explorer.engine.attractors[0]  # Riemann zeta
    print(f"\nEvaluating {prime_attractor.name} at primes:")
    
    primes = [2, 3, 5, 7]
    for p in primes:
        try:
            value = prime_attractor.expr.subs(s, p)
            print(f"  ζ({p}) = {value}")
        except:
            print(f"  ζ({p}) = [complex evaluation]")

def demo_comprehensive_analysis():
    """Demonstrate the full comprehensive analysis"""
    print("\n🔬 DEMO 5: Comprehensive Analysis")
    print("="*50)
    
    explorer = AttractorExplorer()
    
    # Run a quick comprehensive analysis
    results = explorer.run_comprehensive_analysis(max_iters=5)
    
    print("Analysis Summary:")
    summary = results['summary']
    print(f"  Total attractors: {summary['total_attractors']}")
    print(f"  Fixed points: {summary['total_fixed_points']}")
    print(f"  Zero curvature points: {summary['total_zero_curvature_points']}")
    
    # Show some interesting fixed points
    print("\nInteresting Fixed Points:")
    for analysis_type, data in results.items():
        if analysis_type != 'summary':
            fixed_points = data.get('fixed_points', [])
            if fixed_points:
                print(f"\n{analysis_type.upper()}:")
                for name, expr in fixed_points[:2]:
                    print(f"  {name}: {expr[:60]}...")

def demo_custom_attractors():
    """Demonstrate creating custom attractors"""
    print("\n🔬 DEMO 6: Custom Attractor Creation")
    print("="*50)
    
    engine = RiemannEngine()
    engine.add_transform(simplify)
    engine.add_transform(expand)
    
    # Create custom mathematical expressions
    x, y, z = symbols('x y z')
    custom_expressions = [
        (x**3 + y**3 + z**3 - 3*x*y*z, "Cubic Form"),
        (sin(x)*sin(y)*sin(z), "Triple Sine"),
        (exp(x) + exp(y) + exp(z), "Triple Exponential"),
        (log(1 + x**2 + y**2 + z**2), "Logarithmic Form")
    ]
    
    for expr, name in custom_expressions:
        engine.seed(expr, name)
    
    engine.run(max_iters=4)
    
    print("Custom attractors:")
    for attractor in engine.attractors:
        print(f"  {attractor.name}: {attractor.expr}")
        if attractor.curvature_trace:
            final_curvature = attractor.curvature_trace[-1]
            print(f"    Final curvature: {final_curvature}")

def main():
    """Run all demonstrations"""
    print("🎯 SYMBOLIC ATTRACTOR ENGINE DEMONSTRATION")
    print("="*60)
    print("This demo showcases the power of symbolic computation")
    print("for discovering mathematical patterns and attractors.\n")
    
    # Run all demos
    demo_basic_attractors()
    demo_curvature_analysis()
    demo_lie_group_expansions()
    demo_prime_number_attractors()
    demo_comprehensive_analysis()
    demo_custom_attractors()
    
    print("\n" + "="*60)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nThe symbolic attractor engine successfully demonstrated:")
    print("  • Fixed point discovery in mathematical transformations")
    print("  • Curvature computation and zero-curvature detection")
    print("  • Lie group expansions for symmetry analysis")
    print("  • Prime number theory attractors")
    print("  • Comprehensive mathematical pattern analysis")
    print("  • Custom attractor creation and exploration")
    print("\nThis system provides a powerful framework for exploring")
    print("mathematical structures and their evolutionary dynamics.")

if __name__ == "__main__":
    main() 