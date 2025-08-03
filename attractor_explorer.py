#!/usr/bin/env python3
"""
Symbolic Attractor Explorer
==========================

A unified interface for exploring mathematical attractors using sympy.
Combines prime number theory, geometric structures, and analytic functions
into a comprehensive attractor analysis system.
"""

from sympy import symbols, zeta, simplify, expand, sin, cos, pi, diff, exp, I, log, sqrt
from sympy.abc import s, n, p, k
from riemann_attractor import RiemannEngine
from mathematical_attractors import MathematicalAttractorExplorer
from collections import defaultdict
import datetime

class AttractorExplorer:
    """Unified interface for exploring mathematical attractors"""
    
    def __init__(self):
        self.riemann_engine = RiemannEngine()
        self.math_explorer = MathematicalAttractorExplorer()
        self.results_cache = {}
        
    def explore_prime_attractors(self, max_iters=10):
        """Explore attractors related to prime number theory"""
        print("🔍 Exploring Prime Number Attractors...")
        
        # Setup prime-specific attractors
        s = symbols('s')
        attractors = [
            (zeta(s), "Riemann Zeta"),
            (zeta(s) * (1 - 2**(1-s)), "Dirichlet Eta"),
            (log(zeta(s)), "Log Zeta"),
            (diff(zeta(s), s), "Zeta Derivative"),
            (1/zeta(s), "Möbius Transform")
        ]
        
        for expr, name in attractors:
            self.riemann_engine.seed(expr, name)
            
        # Add prime-specific transforms
        self.riemann_engine.add_transform(simplify)
        self.riemann_engine.add_transform(lambda e: e.subs(s, s + 1))
        self.riemann_engine.add_transform(lambda e: e.subs(s, 1/s))
        
        self.riemann_engine.run(max_iters)
        
        return self._analyze_results("prime_attractors")
    
    def explore_geometric_attractors(self, max_iters=10):
        """Explore geometric and topological attractors"""
        print("🔍 Exploring Geometric Attractors...")
        
        x, y, z = symbols('x y z')
        attractors = [
            (x**2 + y**2 + z**2, "Sphere"),
            (x**2 + y**2 - z**2, "Hyperboloid"),
            (x*y*z, "Cubic Surface"),
            (exp(-(x**2 + y**2)), "Gaussian"),
            (1/(1 + x**2 + y**2), "Curvature Function"),
            (sin(x)*cos(y), "Wave Pattern")
        ]
        
        for expr, name in attractors:
            self.riemann_engine.seed(expr, name)
            
        # Add geometric transforms
        self.riemann_engine.add_transform(simplify)
        self.riemann_engine.add_transform(lambda e: e.subs(x, x + y))
        self.riemann_engine.add_transform(lambda e: e.subs(y, y + z))
        
        self.riemann_engine.run(max_iters)
        
        return self._analyze_results("geometric_attractors")
    
    def explore_analytic_attractors(self, max_iters=10):
        """Explore analytic functions and their attractors"""
        print("🔍 Exploring Analytic Attractors...")
        
        z = symbols('z')
        attractors = [
            (exp(z), "Exponential"),
            (sin(z), "Sine"),
            (cos(z), "Cosine"),
            (log(z), "Logarithm"),
            (z**2, "Quadratic"),
            (1/z, "Reciprocal")
        ]
        
        for expr, name in attractors:
            self.riemann_engine.seed(expr, name)
            
        # Add analytic transforms
        self.riemann_engine.add_transform(simplify)
        self.riemann_engine.add_transform(lambda e: e.subs(z, z + 1))
        self.riemann_engine.add_transform(lambda e: e.subs(z, I*z))
        
        self.riemann_engine.run(max_iters)
        
        return self._analyze_results("analytic_attractors")
    
    def run_comprehensive_analysis(self, max_iters=8):
        """Run comprehensive analysis of all attractor types"""
        print("🚀 Starting Comprehensive Attractor Analysis...")
        
        results = {}
        
        # Run all analyses
        results['prime'] = self.explore_prime_attractors(max_iters)
        results['geometric'] = self.explore_geometric_attractors(max_iters)
        results['analytic'] = self.explore_analytic_attractors(max_iters)
        
        # Generate summary
        summary = self._generate_summary(results)
        results['summary'] = summary
        
        self.results_cache = results
        return results
    
    def _analyze_results(self, analysis_type):
        """Analyze results from attractor exploration"""
        fixed_points = self.riemann_engine.get_fixed_points()
        zero_curvature = self.riemann_engine.get_zero_curvature_points()
        symmetries = self.riemann_engine.analyze_symmetries()
        
        return {
            'type': analysis_type,
            'fixed_points': [(a.name, str(a.expr)) for a in fixed_points],
            'zero_curvature_count': len(zero_curvature),
            'symmetries': symmetries,
            'total_attractors': len(self.riemann_engine.attractors)
        }
    
    def _generate_summary(self, results):
        """Generate summary statistics"""
        total_fixed = sum(len(r['fixed_points']) for r in results.values() if isinstance(r, dict))
        total_curvature = sum(r['zero_curvature_count'] for r in results.values() if isinstance(r, dict))
        total_attractors = sum(r['total_attractors'] for r in results.values() if isinstance(r, dict))
        
        return {
            'total_fixed_points': total_fixed,
            'total_zero_curvature_points': total_curvature,
            'total_attractors': total_attractors,
            'analysis_types': list(results.keys())
        }
    
    def export_results(self, filename="attractor_results.toml"):
        """Export results to TOML file"""
        if not self.results_cache:
            print("No results to export. Run analysis first.")
            return
            
        # Create TOML-formatted output
        with open(filename, 'w') as f:
            f.write("# Symbolic Attractor Engine Results\n")
            f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n\n")
            
            for section_name, section_data in self.results_cache.items():
                f.write(f"[{section_name}]\n")
                
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if isinstance(value, list):
                            f.write(f"{key} = [\n")
                            for item in value:
                                if isinstance(item, list):
                                    f.write(f"  [\"{item[0]}\", \"{item[1]}\"],\n")
                                else:
                                    f.write(f"  \"{item}\",\n")
                            f.write("]\n")
                        else:
                            f.write(f"{key} = \"{value}\"\n")
                else:
                    f.write(f"value = \"{section_data}\"\n")
                f.write("\n")
        
        print(f"Results exported to {filename}")
    
    def print_summary(self):
        """Print a formatted summary of results"""
        if not self.results_cache:
            print("No results available. Run analysis first.")
            return
            
        print("\n" + "="*60)
        print("🎯 SYMBOLIC ATTRACTOR EXPLORATION SUMMARY")
        print("="*60)
        
        summary = self.results_cache.get('summary', {})
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Attractors: {summary.get('total_attractors', 0)}")
        print(f"   Fixed Points Found: {summary.get('total_fixed_points', 0)}")
        print(f"   Zero Curvature Points: {summary.get('total_zero_curvature_points', 0)}")
        
        print(f"\n🔍 Analysis Types: {', '.join(summary.get('analysis_types', []))}")
        
        for analysis_type, results in self.results_cache.items():
            if analysis_type != 'summary' and isinstance(results, dict):
                print(f"\n📈 {analysis_type.upper()} ANALYSIS:")
                print(f"   Attractors: {results.get('total_attractors', 0)}")
                print(f"   Fixed Points: {len(results.get('fixed_points', []))}")
                print(f"   Zero Curvature: {results.get('zero_curvature_count', 0)}")
                
                # Show some example fixed points
                fixed_points = results.get('fixed_points', [])
                if fixed_points:
                    print(f"   Example Fixed Points:")
                    for name, expr in fixed_points[:3]:
                        print(f"     {name}: {expr[:50]}...")

def main():
    """Main function to demonstrate the attractor explorer"""
    explorer = AttractorExplorer()
    
    # Run comprehensive analysis
    results = explorer.run_comprehensive_analysis()
    
    # Print summary
    explorer.print_summary()
    
    # Export results
    explorer.export_results()
    
    print("\n✅ Analysis complete! Check attractor_results.toml for detailed results.")

if __name__ == "__main__":
    main() 