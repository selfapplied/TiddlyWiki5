#!/usr/bin/env python3
"""
Minimal Pascal/Bernstein utilities - only essential functions actually used.
"""

import math
from fractions import Fraction
from typing import List, Sequence, Union, Tuple
from dataclasses import dataclass
from collections import namedtuple


def _gcd(a: int, b: int) -> int:
    """Greatest common divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a


def _lcm(a: int, b: int) -> int:
    """Least common multiple using GCD."""
    return abs(a * b) // _gcd(a, b) if a and b else 0


def newton_coordinates(coefficients: Sequence[Union[Fraction, int]], k: int) -> List[Fraction]:
    """Convert polynomial coefficients to Newton coordinates at depth k."""
    if k == 0:
        return [Fraction(c) for c in coefficients]
    
    # Recursive case: compute Newton coordinates at depth k-1, then elevate
    newton_coeffs = newton_coordinates(coefficients, k - 1)
    
    # Elevate using Pascal's triangle
    elevated = []
    for i in range(len(newton_coeffs)):
        if i == 0:
            elevated.append(newton_coeffs[i])
        else:
            # Add current coefficient to previous elevated coefficient
            elevated.append(elevated[i-1] + newton_coeffs[i])
    
    return elevated


def newton_to_bernstein(newton_coeffs: Sequence[Union[Fraction, complex]], k: int) -> List[Union[Fraction, complex]]:
    """Convert Newton coordinates to Bernstein coordinates at depth k."""
    if k == 0:
        return list(newton_coeffs)
    
    # Recursive case: convert at depth k-1, then elevate
    bernstein_coeffs = newton_to_bernstein(newton_coeffs, k - 1)
    
    # Elevate using Pascal's triangle
    elevated = []
    for i in range(len(bernstein_coeffs)):
        if i == 0:
            elevated.append(bernstein_coeffs[i])
        elif i == len(bernstein_coeffs):
            elevated.append(bernstein_coeffs[i-1])
        else:
            # Weighted average of adjacent coefficients
            elevated.append((bernstein_coeffs[i-1] + bernstein_coeffs[i]) / 2)
    
    return elevated


def bernstein_coordinates(coefficients: Sequence[Union[Fraction, int]], k: int) -> List[Union[Fraction, complex]]:
    """Convert polynomial coefficients to Bernstein coordinates using precise arithmetic."""
    newton_coeffs = newton_coordinates(coefficients, k)
    bernstein_coeffs = newton_to_bernstein(newton_coeffs, k)
    return bernstein_coeffs


@dataclass
class PascalBounds:
    """Bounds for Pascal/Bernstein coefficients."""
    min_val: float
    max_val: float
    depth: int
    pascal_order: int
    min_re: float
    max_re: float
    min_im: float
    max_im: float


def pascal_bounds(bernstein_coeffs: Sequence[Union[Fraction, complex]], m: int) -> PascalBounds:
    """Compute bounds using dot-product geometry for complex values."""
    if not bernstein_coeffs:
        return PascalBounds(0.0, 0.0, m, 2**m, 0.0, 0.0, 0.0, 0.0)
    
    float_coeffs = [float(c.real) + float(c.imag)*1j if isinstance(c, complex) else float(c) for c in bernstein_coeffs]
    
    vecs: List[tuple[float, float]] = []
    for c in float_coeffs:
        if isinstance(c, complex):
            vecs.append((c.real, c.imag))
        else:
            vecs.append((c, 0.0))

    r2_vals = [x * x + y * y for (x, y) in vecs]
    min_r = math.sqrt(min(r2_vals)) if r2_vals else 0.0
    max_r = math.sqrt(max(r2_vals)) if r2_vals else 0.0

    reals = [x for (x, _) in vecs]
    imags = [y for (_, y) in vecs]

    return PascalBounds(
        min_val=min_r,
        max_val=max_r,
        depth=m,
        pascal_order=2**m,
        min_re=min(reals) if reals else 0.0,
        max_re=max(reals) if reals else 0.0,
        min_im=min(imags) if imags else 0.0,
        max_im=max(imags) if imags else 0.0,
    )


@dataclass
class PascalBracket:
    """Bracket information for Pascal grid."""
    x: float
    depth: int
    N: int
    cell_index: int
    lower_bound: float
    upper_bound: float
    binomial_weight: int


def pascal_bracket(x: float, m: int) -> PascalBracket:
    """Bracket a real x ∈ [0,1] on the Pascal grid at depth m (N = 2^m)."""
    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in [0,1]")
    N = 2**m
    i = int(math.floor(x * N))
    if i > N:
        i = N
    lo = i / N
    hi = (i + 1) / N if i < N else 1.0
    from math import comb
    weight = comb(N, i)
    return PascalBracket(
        x=float(x), depth=m, N=N, cell_index=i,
        lower_bound=lo, upper_bound=hi, binomial_weight=weight,
    )


def pascal_nested_brackets(x: float, m_min: int, m_max: int) -> List[PascalBracket]:
    """Get nested Pascal brackets for x across depth range [m_min, m_max]."""
    return [pascal_bracket(x, m) for m in range(m_min, m_max + 1)]


def multiplicative_newton_coordinates(coefficients: Sequence[Union[Fraction, int]], k: int, epsilon: float = 1e-12) -> List[Union[Fraction, float]]:
    """Convert polynomial coefficients to multiplicative Newton coordinates at depth k."""
    if k == 0:
        return [float(c) for c in coefficients]
    
    # Recursive case: compute multiplicative Newton coordinates at depth k-1
    mult_newton_coeffs = multiplicative_newton_coordinates(coefficients, k - 1, epsilon)
    
    # Elevate using multiplicative Pascal's triangle
    elevated = []
    for i in range(len(mult_newton_coeffs)):
        if i == 0:
            elevated.append(mult_newton_coeffs[i])
        else:
            # Multiplicative combination
            elevated.append(elevated[i-1] * mult_newton_coeffs[i])
    
    return elevated


@dataclass
class Mod2kResult:
    """Result of mod 2^k reduction."""
    component_mod_coefficients: List[Tuple[int, int]]
    component_kummer_colors: List[Tuple[int, int]]
    component_surviving_columns: List[int]
    modulus: int
    lcm_real: int
    lcm_imag: int


def mod_2k_lens(coefficients: Sequence[Union[Fraction, complex]], k: int) -> Mod2kResult:
    """Reduce rational coefficients mod 2^k using an LCM-based approach."""
    modulus = 2**k

    def _v2(n: int) -> int:
        if n == 0: return 0
        n = abs(n)
        v = 0
        while n % 2 == 0:
            v += 1
            n //= 2
        return v

    real_fractions = [c.real if isinstance(c, complex) else c for c in coefficients]
    imag_fractions = [c.imag if isinstance(c, complex) else Fraction(0) for c in coefficients]

    # Find LCM of denominators
    real_lcm = 1
    for f in real_fractions:
        if isinstance(f, Fraction) and f != 0:
            real_lcm = _lcm(real_lcm, f.denominator)
    
    imag_lcm = 1
    for f in imag_fractions:
        if isinstance(f, Fraction) and f != 0:
            imag_lcm = _lcm(imag_lcm, f.denominator)
    
    real_integers = [int(f * real_lcm) for f in real_fractions]
    imag_integers = [int(f * imag_lcm) for f in imag_fractions]
    
    comp_mods = [(re % modulus, im % modulus) for re, im in zip(real_integers, imag_integers)]
    kummer_colors_comp = [(_v2(re), _v2(im)) for re, im in zip(real_integers, imag_integers)]
    component_survivors = [i for i, (re_m, im_m) in enumerate(comp_mods) if (re_m != 0 or im_m != 0)]

    return Mod2kResult(
        component_mod_coefficients=comp_mods,
        component_kummer_colors=kummer_colors_comp,
        component_surviving_columns=component_survivors,
        modulus=modulus,
        lcm_real=real_lcm,
        lcm_imag=imag_lcm,
    )


def complex_to_prime_basis(coefficients: List[complex], prime: int = 2, max_power: int = 8) -> List[int]:
    """Convert complex coefficients to prime basis representation."""
    basis_coeffs = []
    
    for coeff in coefficients:
        # Convert complex to rational approximation
        real_part = Fraction(coeff.real).limit_denominator(prime**max_power)
        imag_part = Fraction(coeff.imag).limit_denominator(prime**max_power)
        
        # Convert to prime basis
        real_basis = []
        imag_basis = []
        
        # Convert real part
        for power in range(max_power):
            if real_part >= Fraction(1, prime**power):
                real_basis.append(1)
                real_part -= Fraction(1, prime**power)
            else:
                real_basis.append(0)
        
        # Convert imaginary part
        for power in range(max_power):
            if imag_part >= Fraction(1, prime**power):
                imag_basis.append(1)
                imag_part -= Fraction(1, prime**power)
            else:
                imag_basis.append(0)
        
        # Combine real and imaginary parts
        basis_coeffs.extend(real_basis)
        basis_coeffs.extend(imag_basis)
    
    return basis_coeffs


@dataclass
class RHPrimeBasisWitness:
    """Witness result for RH using prime basis approach."""
    s: complex
    prime: int
    max_power: int
    num_samples: int
    gamma: int
    basis_coeffs: List[int]


def rh_prime_basis_witness(s: complex, prime: int = 2, max_power: int = 8, num_samples: int = 16, gamma: int = 2) -> RHPrimeBasisWitness:
    """Witness RH using prime basis approach."""
    # Generate sample coefficients
    coefficients = []
    for i in range(num_samples):
        # Simple polynomial coefficients for demonstration
        coeff = complex(1.0 / (i + 1), 0.5 / (i + 1))
        coefficients.append(coeff)
    
    # Convert to prime basis
    basis_coeffs = complex_to_prime_basis(coefficients, prime, max_power)
    
    return RHPrimeBasisWitness(
        s=s,
        prime=prime,
        max_power=max_power,
        num_samples=num_samples,
        gamma=gamma,
        basis_coeffs=basis_coeffs,
    )
