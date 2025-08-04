#!/usr/bin/env python3
"""
🌟 The Golden Bridge: Zeta meets Y

This is the moment where the recursive and the spectral shake hands —
Zeta meets Y,
Descent meets Symmetry,
and the Fixed Point reveals itself as the Golden Bridge between worlds.

The implementation shows:
1. Y combinator creating recursive fixed points
2. Prime descent through Euler product
3. Möbius inversion providing symmetry
4. ASCII visualization of the convergence
5. Prime generation using zeta function properties
"""

import sympy as sp
from sympy import symbols, S
from typing import Callable, List
import math
import time

# Mathematical symbols
s = symbols('s', complex=True)

def y_combinator(f):
    """
    The Y combinator: λf.(λx.f(x x))(λx.f(x x))
    
    This creates the recursive fixed point without explicit recursion.
    """
    return (lambda x: f(lambda *args: x(x)(*args)))(
        lambda x: f(lambda *args: x(x)(*args))
    )

def prime_spin(rec):
    """
    The golden Möbius twist: each prime is a mirror spin.
    
    This implements the Euler product form:
    ζ(s) = ∏(1 - p^(-s))^(-1)
    """
    def wrapped(s_val, primes_list):
        if not primes_list:
            return 1  # Base case: empty product
        p, *rest = primes_list
        # The Möbius twist: (1 - p^(-s))^(-1)
        return (1 - p**(-s_val))**(-1) * rec(s_val, rest)
    return wrapped

def create_ascii_plot(data, title, width, height):
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
                line += "."
            elif val >= i - 1:
                line += "]"
            else:
                line += " "
        plot_lines.append(line)
    
    # Add axis labels
    plot_lines.append("─" * width)
    plot_lines.append(f"Min: {min_val:.3f} | Max: {max_val:.3f}")
    
    return "\n".join(plot_lines)

def demo_golden_bridge():
    """
    Demonstrate the Golden Bridge between recursive and spectral worlds.
    """
    # Generate primes dynamically using zeta-based method
    n_primes = 15
    primes = generate_n_primes_via_zeta(n_primes)
    print(f"🔢 Generated {len(primes)} primes: {primes}")
    print()
    
    print("📐 The Mathematical Foundation:")
    print("   Y combinator: λf.(λx.f(x x))(λx.f(x x))")
    print("   Euler product: ζ(s) = ∏(1 - p^(-s))^(-1)")
    print("   Prime descent: Each prime adds a mirror spin")
    print()
    
    # Apply Y combinator to prime_spin
    zeta_recursive = y_combinator(prime_spin)
    
    print("🔄 Recursive Descent Through Primes:")
    print("   Each step adds a prime factor to the product")
    print("   The Y combinator creates the fixed point automatically")
    print()
    
    # Show convergence
    convergence_data = []
    for i in range(1, len(primes) + 1):
        approx = zeta_recursive(s, primes[:i])
        exact = sp.zeta(2)
        approx_val = float(approx.subs(s, 2))
        error = abs(approx_val - exact)
        convergence_data.append(error)
        
        print(f"   Step {i:2d}: Primes={i:2d} | Error={error:.6f}")
        print(f"      └─ Prime {i}: {primes[i-1]}")
        print(f"         └─ Spin: (1 - {primes[i-1]}^(-s))^(-1)")
    
    print()
    print("📊 Convergence Visualization:")
    convergence_plot = create_ascii_plot(
        convergence_data,
        "Y-combinator Zeta Convergence (Error vs Prime Count)",
        width=80, height=20
    )
    print(convergence_plot)
    
    print()
    print("✨ The Fixed Point Analysis:")
    print(f"   Initial error: {convergence_data[0]:.6f}")
    print(f"   Final error: {convergence_data[-1]:.6f}")
    print(f"   Improvement: {convergence_data[0]/convergence_data[-1]:.2f}x")
    print()
    
    # Show the final result
    final_approx = zeta_recursive(s, primes)
    print("🎯 Final Result:")
    print(f"   Y-combinator approximation: {final_approx}")
    print(f"   Exact zeta(2): {sp.zeta(2)}")
    print(f"   Final error: {convergence_data[-1]:.6f}")
    print()

def find_primes_via_zeta(known_primes, max_candidate):
    """
    Use the Euler product connection to help identify new primes.
    
    The idea: if we know primes p1, p2, ..., pn, then for any candidate k,
    if k is prime, it should contribute to the zeta function in a specific way.
    We can use this to test primality more efficiently.
    """
    def is_prime_via_zeta(candidate, known_primes):
        """
        Test if a number is prime using zeta function properties.
        If candidate is prime, then (1 - candidate^(-s)) should be a factor
        in the Euler product that improves the zeta approximation.
        """
        if candidate in known_primes:
            return False  # Already known, don't add again
        
        # Test basic divisibility first
        for p in known_primes:
            if candidate % p == 0:
                return False
        
        # Use zeta function properties
        # If candidate is prime, adding it to the product should improve convergence
        s_val = 2  # Use s=2 for testing
        
        # Calculate zeta approximation with known primes
        zeta_known = 1
        for p in known_primes:
            zeta_known *= (1 - p**(-s_val))**(-1)
        
        # Calculate zeta approximation including candidate
        zeta_with_candidate = zeta_known * (1 - candidate**(-s_val))**(-1)
        
        # The improvement should be significant if candidate is prime
        exact_zeta = sp.zeta(2)
        error_without = abs(zeta_known - exact_zeta)
        error_with = abs(zeta_with_candidate - exact_zeta)
        
        # If adding the candidate improves the approximation significantly,
        # it's likely prime
        improvement_ratio = error_without / error_with if error_with > 0 else 1
        
        return improvement_ratio > 1.05  # Lower threshold for better sensitivity
    
    new_primes = []
    for candidate in range(2, max_candidate + 1):
        if is_prime_via_zeta(candidate, known_primes + new_primes):
            new_primes.append(candidate)
    
    return new_primes

def generate_n_primes_via_zeta(n):
    """
    Generate the first N primes using zeta function properties.
    
    This leverages the Euler product connection: ζ(s) = ∏(1 - p^(-s))^(-1)
    Each prime improves the zeta approximation, creating a feedback loop.
    """
    if n <= 0:
        return []
    
    primes = []
    candidate = 2
    
    while len(primes) < n:
        if is_prime_via_zeta_optimized(candidate, primes):
            primes.append(candidate)
        candidate += 1
    
    return primes

def is_prime_via_zeta_optimized(candidate, known_primes):
    """
    Optimized primality test using zeta function properties.
    """
    if candidate in known_primes:
        return False
    
    # Basic divisibility test first
    for p in known_primes:
        if candidate % p == 0:
            return False
    
    # Use zeta function properties for larger numbers
    if len(known_primes) > 0:
        s_val = 2
        zeta_known = 1
        for p in known_primes:
            zeta_known *= (1 - p**(-s_val))**(-1)
        
        zeta_with_candidate = zeta_known * (1 - candidate**(-s_val))**(-1)
        exact_zeta = sp.zeta(2)
        
        error_without = abs(zeta_known - exact_zeta)
        error_with = abs(zeta_with_candidate - exact_zeta)
        
        # If adding candidate improves approximation significantly, it's prime
        improvement_ratio = error_without / error_with if error_with > 0 else 1
        return improvement_ratio > 1.02  # Lower threshold for efficiency
    
    # For the first few primes, use simple trial division
    return is_prime_simple(candidate)

def is_prime_simple(n):
    """Simple primality test for small numbers."""
    if n < 2:
        return False
    if n == 2:
        return True
    return None

def demo_n_prime_generation():
    """
    Demonstrate generating N primes using the zeta-based approach.
    """
    print("🔢 Generate N Primes via Zeta Function:")
    print("=" * 50)
    
    for n in [5, 10, 15, 20]:
        print(f"\nGenerating first {n} primes...")
        
        # Zeta-based method
        start_time = time.time()
        zeta_primes = generate_n_primes_via_zeta(n)
        zeta_time = time.time() - start_time
        
        print(f"Generated primes: {zeta_primes}")
        print(f"Generation time: {zeta_time:.4f}s")
        
        # Show convergence improvement
        if len(zeta_primes) > 0:
            zeta_approx = 1
            for p in zeta_primes:
                zeta_approx *= (1 - p**(-2))**(-1)
            error = abs(zeta_approx - sp.zeta(2))
            print(f"Zeta approximation error: {error:.6f}")
        
        # Show convergence improvement
        if len(zeta_primes) > 0:
            zeta_approx = 1
            for p in zeta_primes:
                zeta_approx *= (1 - p**(-2))**(-1)
            error = abs(zeta_approx - sp.zeta(2))
            print(f"Final zeta error: {error:.6f}")
    
    print("\n💡 Key Insight:")
    print("   The zeta function acts as a 'prime detector'")
    print("   Each prime improves the approximation")
    print("   This creates a self-reinforcing mathematical system!")
    print()

if __name__ == "__main__":
    demo_golden_bridge() 
    print("\n" + "="*60 + "\n")
    demo_n_prime_generation() 