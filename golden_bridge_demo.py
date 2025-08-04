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
"""

import sympy as sp
from sympy import symbols, S
from typing import Callable, List

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
    # Generate primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
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
    
if __name__ == "__main__":
    demo_golden_bridge() 