#!/usr/bin/env python3
"""
🌟 Zeta meets Y: The Golden Bridge Between Worlds

This module implements the deep connection between:
- The Y combinator (recursive fixed point)
- The Riemann zeta function (spectral analysis)
- Prime factorization (descent through primes)
- Möbius inversion (symmetry)

The Fixed Point reveals itself as the Golden Bridge between worlds.
"""

import sympy as sp
from sympy import symbols, S, expand, factor, oo
from typing import Callable, List, Any, Optional
from functools import lru_cache

# ASCII plotting utilities
def create_ascii_plot(data, title, width=80, height=30):
    """Create ASCII art plot from data."""
    if not data:
        return f"\n{title}\n" + "─" * width + "\nNo data to plot\n"
    
    min_val = min(data)
    max_val = max(data)
    if max_val == min_val:
        max_val = min_val + 1
    
    # Normalize data to plot height
    normalized = [(val - min_val) / (max_val - min_val) * (height - 1) for val in data]
    
    # Create ASCII plot
    plot_lines = [f"\n{title}\n" + "─" * width]
    
    for i in range(height - 1, -1, -1):
        line = ""
        for j, val in enumerate(normalized):
            if j >= width:
                break
            if val >= i:
                line += ":"
            elif val >= i - 0.5:
                line += "_"
            elif val >= i - 1:
                line += "/"
            else:
                line += " "
        plot_lines.append(line)
    
    # Add axis labels
    plot_lines.append("─" * width)
    plot_lines.append(f"Min: {min_val:.3f} | Max: {max_val:.3f}")
    
    return "\n".join(plot_lines)

# Mathematical symbols
s = symbols('s', complex=True)

class ZetaYCombinator:
    """
    The Golden Bridge: Y combinator meets Riemann zeta through prime descent.
    
    This implements the recursive system where each descent (n → n-1) 
    is encoded as a prime spin, converging on the zeta function.
    """
    
    def __init__(self):
        self.primes = self._generate_primes(100)  # First 100 primes
        self.evolution_level = 0
        self.fixed_points = []
        
    def _generate_primes(self, n: int) -> List[int]:
        """Generate first n primes using sieve."""
        if n <= 0:
            return []
        
        # Simple prime generation
        primes = []
        num = 2
        while len(primes) < n:
            if all(num % p != 0 for p in primes):
                primes.append(num)
            num += 1
        return primes
    
    def y_combinator(self, f: Callable) -> Callable:
        """
        The Y combinator: λf.(λx.f(x x))(λx.f(x x))
        
        This creates the recursive fixed point without explicit recursion.
        """
        # Assertions for Y combinator
        assert callable(f), "Y combinator requires a callable function"
        
        result = (lambda x: f(lambda *args: x(x)(*args)))(
            lambda x: f(lambda *args: x(x)(*args))
        )
        
        # Assert the result is callable
        assert callable(result), "Y combinator must return a callable"
        
        return result
    
    def prime_spin(self, rec: Callable) -> Callable:
        """
        The golden Möbius twist: each prime is a mirror spin.
        
        This implements the Euler product form:
        ζ(s) = ∏(1 - p^(-s))^(-1)
        """
        # Assertions for prime_spin
        assert callable(rec), "prime_spin requires a callable recursive function"
        
        def wrapped(s_val, primes_list):
            # Assertions for wrapped function
            assert primes_list is not None, "primes_list cannot be None"
            assert isinstance(primes_list, (list, tuple)), f"primes_list must be list/tuple: {type(primes_list)}"
            
            if not primes_list:
                return 1  # Base case: empty product
            p, *rest = primes_list
            
            # Assertions for prime
            assert p > 1, f"Prime must be > 1: {p}"
            assert isinstance(p, int), f"Prime must be integer: {type(p)}"
            
            # The Möbius twist: (1 - p^(-s))^(-1)
            result = (1 - p**(-s_val))**(-1) * rec(s_val, rest)
            
            # Assertions for result
            assert result != 0, f"Prime spin result cannot be zero for prime {p}"
            
            return result
        
        # Assert the result is callable
        assert callable(wrapped), "prime_spin must return a callable"
        
        return wrapped
    
    def zeta_approximation(self, s_val, max_primes: Optional[int] = None) -> sp.Expr:
        """
        Approximate zeta function using Y combinator and prime descent.
        
        This is the concrete realization of the recursive system.
        """
        if max_primes is None:
            max_primes = len(self.primes)
        
        # Assertions for input validation
        assert max_primes > 0, f"max_primes must be positive: {max_primes}"
        assert max_primes <= len(self.primes), f"max_primes too large: {max_primes} > {len(self.primes)}"
        
        primes_subset = self.primes[:max_primes]
        
        # Apply Y combinator to prime_spin
        zeta_recursive = self.y_combinator(self.prime_spin)
        
        # Compute the approximation
        result = zeta_recursive(s_val, primes_subset)
        
        # Assertions for result validation
        assert result != 0, "Zeta approximation cannot be zero"
        assert len(str(result)) > 0, "Result must be a valid expression"
        
        # Track evolution
        self.evolution_level += 1
        self.fixed_points.append({
            'level': self.evolution_level,
            'primes_used': len(primes_subset),
            'result': result
        })
        
        return result
    
    def mobius_inversion(self, s_val, n: int) -> sp.Expr:
        """
        Möbius inversion: the symmetry that gives us the twist.
        
        μ(n) is the Möbius function, giving us the inversion formula.
        """
        def mobius_function(k):
            if k == 1:
                return 1
            factors = sp.factorint(k)
            if any(exp > 1 for exp in factors.values()):
                return 0  # Has square factor
            return (-1) ** len(factors)  # Number of distinct prime factors
        
        result = sp.S.Zero
        for k in range(1, n + 1):
            result += mobius_function(k) / (k ** s_val)
        
        return result
    
    def plot_convergence(self, s_points: List[complex], max_primes: int):
        """
        Visualize the convergence of our Y-combinator zeta approximation using ASCII art.
        """
        print("\n🌟 ASCII Convergence Plots")
        print("=" * 50)
        
        # Plot 1: Convergence with number of primes
        prime_counts = list(range(1, max_primes + 1))
        convergence_data = []
        
        for prime_count in prime_counts:
            approx = self.zeta_approximation(s, prime_count)
            # Evaluate at s = 2 for comparison with π²/6
            exact_value = sp.zeta(2)
            approx_value = approx.subs(s, 2)
            convergence_data.append(abs(approx_value - exact_value))
        
        # ASCII plot for convergence
        convergence_plot = create_ascii_plot(
            convergence_data, 
            "Y-combinator Zeta Convergence (Error vs Prime Count)"
        )
        print(convergence_plot)
        
        # Plot 2: Fixed points evolution
        if self.fixed_points:
            levels = [fp['level'] for fp in self.fixed_points]
            prime_counts = [fp['primes_used'] for fp in self.fixed_points]
            
            evolution_plot = create_ascii_plot(
                prime_counts,
                "Fixed Point Evolution (Primes Used vs Level)",
            )
            print(evolution_plot)
        
        # Text summary
        print(f"\n📊 Convergence Summary:")
        print(f"   Initial error: {convergence_data[0]:.6f}")
        print(f"   Final error: {convergence_data[-1]:.6f}")
        print(f"   Improvement: {convergence_data[0]/convergence_data[-1]:.2f}x")
        print(f"   Fixed points tracked: {len(self.fixed_points)}")
    
    def ascii_animation(self, s_val=2, steps=10):
        """
        ASCII animation of the recursive descent through primes.
        """
        print("🌟 Zeta meets Y: The Golden Bridge Animation")
        print("=" * 50)
        
        for i in range(1, steps + 1):
            approx = self.zeta_approximation(s, i)
            exact = sp.zeta(s_val)
            error = abs(approx.subs(s, s_val) - exact)
            
            print(f"Step {i:2d}: Primes={i:2d} | Error={error:.6f}")
            
            # ASCII art for the recursive structure
            indent = "  " * i
            print(f"{indent}└─ Prime {i}: {self.primes[i-1]}")
            print(f"{indent}   └─ Spin: (1 - {self.primes[i-1]}^(-s))^(-1)")
        
        print("\n✨ Fixed Point Achieved!")
        print(f"Final approximation: {self.zeta_approximation(s, steps)}")
        print(f"Exact value: {sp.zeta(s_val)}")
    
    def modular_conjugation(self, s_val, k):
        """
        Conjugate with modular forms: the spectral dance.
        
        This explores the connection to modular forms and the modular group.
        """
        # Basic modular form: Eisenstein series
        def eisenstein_series(s, k):
            result = sp.S.Zero
            for m in range(1, 10):  # Truncated for computation
                for n in range(1, 10):
                    if m != 0 or n != 0:
                        result += 1 / ((m * s + n) ** k)
            return result
        
        # Conjugate our zeta with modular forms
        zeta_approx = self.zeta_approximation(s, 10)
        modular_form = eisenstein_series(s, k)
        
        # The conjugation: zeta * modular_form
        conjugated = zeta_approx * modular_form
        
        return {
            'zeta_approx': zeta_approx,
            'modular_form': modular_form,
            'conjugated': conjugated
        }

def demo_golden_bridge():
    """
    Demonstrate the Golden Bridge between recursive and spectral worlds.
    """
    print("🌟 The Golden Bridge: Zeta meets Y")
    print("=" * 50)
    
    zeta_y = ZetaYCombinator()
    
    # 1. Basic Y-combinator zeta approximation
    print("\n1. Y-combinator Zeta Approximation:")
    s_val = 2
    approx = zeta_y.zeta_approximation(s, 10)
    exact = sp.zeta(s_val)
    approx_val = float(approx.subs(s, s_val).evalf())
    exact_val = float(exact.evalf())
    error = abs(approx_val - exact_val)
    print(f"Approximation: {approx}")
    print(f"Exact zeta({s_val}): {exact}")
    print(f"Numerical error: {error:.6f}")
    print(f"Approximation value: {approx_val:.6f}")
    print(f"Exact value: {exact_val:.6f}")
    
    # Assertions for mathematical correctness
    assert error < 0.1, f"Error too large: {error} (should be < 0.1)"
    assert approx_val > 0, f"Approximation must be positive: {approx_val}"
    assert exact_val > 0, f"Exact value must be positive: {exact_val}"
    assert abs(exact_val - sp.pi**2/6) < 1e-10, f"Exact value should be π²/6: {exact_val}"
    print("✅ Mathematical assertions passed!")
    
    # 2. ASCII animation
    print("\n2. ASCII Animation of Recursive Descent:")
    zeta_y.ascii_animation(s_val=2, steps=5)
    
    # Assertions for convergence
    convergence_errors = []
    for i in range(1, 6):
        approx = zeta_y.zeta_approximation(s, i)
        exact = sp.zeta(s_val)
        approx_val = float(approx.subs(s, s_val).evalf())
        exact_val = float(exact.evalf())
        error = abs(approx_val - exact_val)
        convergence_errors.append(error)
    
    # Assert convergence is monotonic (error should decrease)
    for i in range(1, len(convergence_errors)):
        assert convergence_errors[i] <= convergence_errors[i-1], f"Error should decrease: {convergence_errors[i]} > {convergence_errors[i-1]}"
    print("✅ Convergence assertions passed!")
    
    # 3. Möbius inversion
    print("\n3. Möbius Inversion (Symmetry):")
    mobius_result = zeta_y.mobius_inversion(s, 10)
    print(f"Möbius inversion: {mobius_result}")
    
    # Assertions for Möbius function properties
    mobius_val = float(mobius_result.subs(s, 2).evalf())
    assert mobius_val != 0, f"Möbius inversion should be non-zero: {mobius_val}"
    assert abs(mobius_val) < 10, f"Möbius inversion should be reasonable: {mobius_val}"
    print("✅ Möbius inversion assertions passed!")
    
    # 4. Modular conjugation
    print("\n4. Modular Conjugation (Spectral Dance):")
    conjugated = zeta_y.modular_conjugation(s, k=2)
    print(f"Conjugated result: {conjugated['conjugated']}")
    
    # 5. Plot convergence
    print("\n5. Generating convergence plot...")
    zeta_y.plot_convergence([2, 3, 4], max_primes=80)
    
    print("\n✨ The Fixed Point reveals itself as the Golden Bridge!")
    print("Recursive descent meets spectral symmetry in perfect harmony.")

if __name__ == "__main__":
    demo_golden_bridge() 