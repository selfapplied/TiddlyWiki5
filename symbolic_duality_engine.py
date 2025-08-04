"""
Symbolic Duality Engine for Mathematical Functions

This module implements a symbolic duality framework for studying functions
under the duality map s ↔ 1-s, with special focus on zeta functions and
their functional equations.
"""

from sympy import (
    symbols, Function, simplify, Eq, conjugate, re, im, I, pi, 
    gamma, exp, log, sin, cos, sqrt, oo, S, expand, factor
)
from sympy.functions.special.zeta_functions import zeta
from sympy.functions.special.gamma_functions import gamma
from sympy.core.relational import Equality
from sympy.solvers import solve
from sympy.series import series
import sympy as sp

# Define symbolic variables
s = symbols('s', complex=True)
t = symbols('t', real=True)
z = symbols('z', complex=True)

class DualityEngine:
    """
    A symbolic engine for studying duality transformations s ↔ 1-s
    and their effects on mathematical functions.
    """
    
    def __init__(self):
        self.s = s
        self.functions = {}
        
    def dual_map(self, f):
        """
        Apply the duality map: f(s) → f(1-s)
        
        Args:
            f: SymPy expression in variable s
            
        Returns:
            The dual function f(1-s)
        """
        return f.subs(s, 1 - s)
    
    def is_on_critical_line(self, expr):
        """
        Check if an expression represents a point on the critical line Re(s) = 1/2
        
        Args:
            expr: SymPy expression
            
        Returns:
            Boolean indicating if Re(expr) = 1/2
        """
        return simplify(re(expr)) == S.Half
    
    def critical_line_condition(self, expr):
        """
        Return the condition for being on the critical line
        
        Args:
            expr: SymPy expression
            
        Returns:
            The equation Re(expr) = 1/2
        """
        return Eq(re(expr), S.Half)
    
    def fixed_point_test(self, f):
        """
        Test if a function is a fixed point under duality: f(s) = f(1-s)
        
        Args:
            f: SymPy expression
            
        Returns:
            Boolean indicating if f is a fixed point
        """
        f_dual = self.dual_map(f)
        return simplify(f - f_dual) == 0
    
    def functional_equation_attractor(self, f, chi=None):
        """
        Define the functional equation attractor: f(s) = χ(s) * f(1-s)
        
        Args:
            f: The function to study
            chi: The factor χ(s) (if None, will be computed)
            
        Returns:
            The functional equation as an equality
        """
        f_dual = self.dual_map(f)
        
        if chi is None:
            # For zeta function, χ(s) = 2^s * π^(s-1) * sin(πs/2) * Γ(1-s)
            chi = (2**s * pi**(s-1) * sin(pi*s/2) * gamma(1-s))
        
        return Eq(f, chi * f_dual)
    
    def zeta_functional_equation(self):
        """
        Return Riemann's functional equation for the zeta function
        
        Returns:
            The functional equation ζ(s) = χ(s) * ζ(1-s)
        """
        return self.functional_equation_attractor(zeta(s))
    
    def chi_factor(self):
        """
        Compute the χ(s) factor in Riemann's functional equation
        
        Returns:
            The χ(s) factor: 2^s * π^(s-1) * sin(πs/2) * Γ(1-s)
        """
        return 2**s * pi**(s-1) * sin(pi*s/2) * gamma(1-s)
    
    def study_function(self, f, name="f"):
        """
        Comprehensive study of a function under duality
        
        Args:
            f: SymPy expression to study
            name: Name for the function
            
        Returns:
            Dictionary with analysis results
        """
        f_dual = self.dual_map(f)
        is_fixed = self.fixed_point_test(f)
        
        # Check behavior on critical line
        critical_line_expr = f.subs(s, S.Half + I*t)
        critical_line_dual = f_dual.subs(s, S.Half + I*t)
        
        # Try to simplify the critical line expressions
        try:
            critical_line_simplified = simplify(critical_line_expr)
            critical_line_dual_simplified = simplify(critical_line_dual)
        except:
            critical_line_simplified = critical_line_expr
            critical_line_dual_simplified = critical_line_dual
        
        return {
            'function': f,
            'dual': f_dual,
            'is_fixed_point': is_fixed,
            'critical_line_behavior': critical_line_simplified,
            'critical_line_dual_behavior': critical_line_dual_simplified,
            'functional_equation': self.functional_equation_attractor(f)
        }
    
    def create_function_space(self, base_functions=None):
        """
        Create a function space for studying duality
        
        Args:
            base_functions: List of base functions to include
            
        Returns:
            Dictionary of functions to study
        """
        if base_functions is None:
            base_functions = {
                'zeta': zeta(s),
                'gamma': gamma(s),
                'exp': exp(s),
                'log': log(s),
                'sin': sin(s),
                'cos': cos(s)
            }
        
        function_space = {}
        for name, func in base_functions.items():
            function_space[name] = self.study_function(func, name)
        
        return function_space
    
    def analyze_critical_zeros(self, f, max_terms=5):
        """
        Analyze the behavior of a function near critical zeros
        
        Args:
            f: Function to analyze
            max_terms: Maximum terms in series expansion
            
        Returns:
            Series expansion around s = 1/2
        """
        critical_point = S.Half
        try:
            # Use expand around the critical point
            series_expansion = f.series(s, critical_point, n=max_terms)
            return series_expansion
        except:
            return f"Could not expand {f} around s = 1/2"
    
    def duality_symmetry_analysis(self, f):
        """
        Analyze the symmetry properties under duality
        
        Args:
            f: Function to analyze
            
        Returns:
            Dictionary with symmetry analysis
        """
        f_dual = self.dual_map(f)
        
        # Check if f(s) = f(1-s) (even under duality)
        even_under_duality = self.fixed_point_test(f)
        
        # Check if f(s) = -f(1-s) (odd under duality)
        odd_under_duality = simplify(f + f_dual) == 0
        
        # Check if f(s) = conjugate(f(1-s)) (conjugate symmetry)
        conjugate_symmetry = simplify(f - conjugate(f_dual)) == 0
        
        return {
            'even_under_duality': even_under_duality,
            'odd_under_duality': odd_under_duality,
            'conjugate_symmetry': conjugate_symmetry,
            'functional_equation': self.functional_equation_attractor(f)
        }


def demo_duality_engine():
    """
    Demonstration of the duality engine capabilities
    """
    print("🧠 Symbolic Duality Engine Demo")
    print("=" * 50)
    
    engine = DualityEngine()
    
    # Phase 1: Foundation
    print("\n📐 Phase 1: Foundation – Dual Map & Critical Line")
    print("-" * 40)
    
    # Test duality map
    test_func = s**2 + 2*s + 1
    dual_func = engine.dual_map(test_func)
    print(f"Original: {test_func}")
    print(f"Dual: {dual_func}")
    
    # Test critical line
    critical_point = S.Half + I*t
    print(f"\nCritical line point: {critical_point}")
    print(f"Re(s) = 1/2: {engine.is_on_critical_line(critical_point)}")
    
    # Phase 2: Zeta Function Analysis
    print("\n📊 Phase 2: Zeta Function Analysis")
    print("-" * 40)
    
    zeta_analysis = engine.study_function(zeta(s), "zeta")
    print(f"Zeta function: {zeta_analysis['function']}")
    print(f"Zeta dual: {zeta_analysis['dual']}")
    print(f"Is fixed point: {zeta_analysis['is_fixed_point']}")
    
    # Phase 3: Functional Equation
    print("\n🔗 Phase 3: Functional Equation Attractor")
    print("-" * 40)
    
    chi = engine.chi_factor()
    print(f"χ(s) factor: {chi}")
    
    functional_eq = engine.zeta_functional_equation()
    print(f"Riemann's functional equation: {functional_eq}")
    
    # Phase 4: Function Space Analysis
    print("\n🌌 Phase 4: Function Space Analysis")
    print("-" * 40)
    
    function_space = engine.create_function_space()
    
    for name, analysis in function_space.items():
        print(f"\n{name.upper()} function:")
        print(f"  Fixed point: {analysis['is_fixed_point']}")
        print(f"  Critical line behavior: {analysis['critical_line_behavior']}")
    
    # Phase 5: Symmetry Analysis
    print("\n🔄 Phase 5: Symmetry Analysis")
    print("-" * 40)
    
    symmetry_analysis = engine.duality_symmetry_analysis(zeta(s))
    for property_name, value in symmetry_analysis.items():
        print(f"{property_name}: {value}")
    
    return engine


if __name__ == "__main__":
    engine = demo_duality_engine() 