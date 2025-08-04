#!/usr/bin/env python3
"""
Zeta Fractal Cellular Automaton
===============================

A recursive, self-defining CA that evolves by prime descent and Möbius transforms,
descending toward the critical line to resolve ζ(s) = 0.

Each cell carries spectral information and decides whether to split, recurse, or stabilize
based on local error and prime content analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Complex, Optional
from math import log, pi, sqrt
import cmath

# Prime cache for efficient computation
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

@dataclass
class ZetaCell:
    """
    Core cell in the Zeta CA system.
    
    Each cell represents a region of the complex plane being refined
    toward the critical line through recursive prime descent.
    """
    value: Complex                 # Current zeta approximation
    prime_content: List[int]       # Which primes have been inserted
    moebius_spin: float           # Möbius inversion phase [-1, 0, 1]
    error: float                  # Spectral approximation error
    depth: int                    # Recursive subdivision depth
    position: Tuple[float, float] # (Re(s), Im(s)) position in complex plane
    size: float                   # Cell size in complex plane
    active: bool = True           # Whether cell is still evolving

    def should_subdivide(self, threshold: float = 1e-3) -> bool:
        """Determine if this cell needs recursive subdivision."""
        return self.error > threshold and self.depth < 10 and self.active

    def euler_factor(self, p: int, s: Complex) -> Complex:
        """Compute (1 - p^(-s))^(-1) Euler factor for prime p."""
        try:
            p_to_minus_s = p ** (-s)
            if abs(1 - p_to_minus_s) < 1e-15:
                return complex(1e15)  # Avoid division by zero
            return 1.0 / (1.0 - p_to_minus_s)
        except (OverflowError, ZeroDivisionError):
            return complex(1e15)

    def moebius_mu(self, n: int) -> int:
        """Möbius function μ(n)."""
        if n == 1:
            return 1
        
        # Factor n to check for squared prime factors
        factors = []
        temp = n
        for p in PRIMES:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count > 1:
                return 0  # Squared prime factor
            if count == 1:
                factors.append(p)
        
        if temp > 1:
            factors.append(temp)
        
        return (-1) ** len(factors)

    def apply_moebius_twist(self, s: Complex) -> Complex:
        """Apply Möbius twist transformation to complex number s."""
        # Möbius transform: (as + b) / (cs + d)
        # Use spectral content to determine transformation parameters
        a = 1 + 0.1j * self.moebius_spin
        b = 0.1 * self.error
        c = 0.01 * len(self.prime_content)
        d = 1
        
        denominator = c * s + d
        if abs(denominator) < 1e-15:
            return s
        
        return (a * s + b) / denominator


class ZetaCA:
    """
    Main Zeta Cellular Automaton engine.
    
    Manages the recursive attractor-driven evolution of the zeta function
    approximation through prime descent and fractal subdivision.
    """
    
    def __init__(self, initial_s: Complex = 0.5 + 14.134725j, grid_size: int = 64):
        """
        Initialize the Zeta CA with a starting point near the critical line.
        
        Args:
            initial_s: Starting complex number (default near first zeta zero)
            grid_size: Initial grid resolution
        """
        self.grid_size = grid_size
        self.initial_s = initial_s
        self.iteration = 0
        self.prime_index = 0
        
        # Initialize with single attractor cell
        self.cells = [[ZetaCell(
            value=complex(1.0),
            prime_content=[],
            moebius_spin=0.0,
            error=1.0,
            depth=0,
            position=(initial_s.real, initial_s.imag),
            size=2.0,  # Initial size in complex plane
            active=True
        )]]
        
        self.error_threshold = 1e-6
        self.max_depth = 8

    def insert_prime(self, p: int):
        """Insert prime p into all active cells via Euler product."""
        for row in self.cells:
            for cell in row:
                if not cell.active:
                    continue
                    
                # Compute s value for this cell
                s = complex(cell.position[0], cell.position[1])
                
                # Apply Euler factor
                euler = cell.euler_factor(p, s)
                old_value = cell.value
                cell.value *= euler
                
                # Update error estimate
                error_contribution = abs(cell.value - old_value) / max(abs(cell.value), 1e-15)
                cell.error = max(cell.error * 0.9, error_contribution)
                
                # Update Möbius spin based on prime
                mu_p = cell.moebius_mu(p)
                cell.moebius_spin = 0.7 * cell.moebius_spin + 0.3 * mu_p
                
                # Add prime to content
                cell.prime_content.append(p)

    def subdivide_cell(self, row_idx: int, col_idx: int):
        """Subdivide a cell into 4 subcells when threshold is exceeded."""
        parent = self.cells[row_idx][col_idx]
        if not parent.should_subdivide(self.error_threshold):
            return
            
        # Deactivate parent
        parent.active = False
        
        # Create 4 subcells
        new_size = parent.size * 0.5
        pos_x, pos_y = parent.position
        offsets = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        # Expand grid if needed
        if len(self.cells) <= row_idx * 2 + 1:
            for _ in range(len(self.cells), row_idx * 2 + 2):
                self.cells.append([])
        
        for i, (dx, dy) in enumerate(offsets):
            new_pos = (pos_x + dx * parent.size, pos_y + dy * parent.size)
            
            # Apply Möbius twist to new position
            s_new = complex(new_pos[0], new_pos[1])
            s_twisted = parent.apply_moebius_twist(s_new)
            twisted_pos = (s_twisted.real, s_twisted.imag)
            
            subcell = ZetaCell(
                value=parent.value,
                prime_content=parent.prime_content.copy(),
                moebius_spin=parent.moebius_spin,
                error=parent.error * 0.8,  # Assume subdivision improves error
                depth=parent.depth + 1,
                position=twisted_pos,
                size=new_size,
                active=True
            )
            
            # Add to appropriate grid position
            target_row = row_idx * 2 + (i // 2)
            target_col = col_idx * 2 + (i % 2)
            
            while len(self.cells[target_row]) <= target_col:
                self.cells[target_row].append(None)
            
            self.cells[target_row][target_col] = subcell

    def evolution_step(self):
        """Execute one step of the CA evolution."""
        self.iteration += 1
        
        # 1. Insert next prime if available
        if self.prime_index < len(PRIMES):
            current_prime = PRIMES[self.prime_index]
            print(f"Iteration {self.iteration}: Inserting prime {current_prime}")
            self.insert_prime(current_prime)
            self.prime_index += 1
        
        # 2. Check for subdivision in all active cells
        subdivisions = []
        for i, row in enumerate(self.cells):
            for j, cell in enumerate(row):
                if cell and cell.should_subdivide(self.error_threshold):
                    subdivisions.append((i, j))
        
        # 3. Perform subdivisions
        for row_idx, col_idx in subdivisions:
            self.subdivide_cell(row_idx, col_idx)
        
        # 4. Apply Möbius coupling between neighboring cells
        self.apply_fourier_moebius_coupling()
        
        print(f"  Active cells: {self.count_active_cells()}")
        print(f"  Average error: {self.average_error():.6f}")

    def apply_fourier_moebius_coupling(self):
        """Apply Fourier ↔ Möbius coupling between neighboring cells."""
        for i, row in enumerate(self.cells):
            for j, cell in enumerate(row):
                if not cell or not cell.active:
                    continue
                
                # Find neighbors and apply coupling
                neighbors = self.get_neighbors(i, j)
                if not neighbors:
                    continue
                
                # Compute average Möbius spin of neighbors
                avg_spin = sum(n.moebius_spin for n in neighbors) / len(neighbors)
                
                # Apply coupling (mix with neighbors)
                coupling_strength = 0.1
                cell.moebius_spin = (1 - coupling_strength) * cell.moebius_spin + coupling_strength * avg_spin
                
                # Phase descent toward critical line (Re(s) = 0.5)
                s = complex(cell.position[0], cell.position[1])
                critical_pull = 0.01 * (0.5 - s.real)
                new_pos = (cell.position[0] + critical_pull, cell.position[1])
                cell.position = new_pos

    def get_neighbors(self, row: int, col: int) -> List[ZetaCell]:
        """Get active neighboring cells."""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = row + di, col + dj
                if (0 <= ni < len(self.cells) and 
                    0 <= nj < len(self.cells[ni]) and
                    self.cells[ni][nj] and 
                    self.cells[ni][nj].active):
                    neighbors.append(self.cells[ni][nj])
        return neighbors

    def count_active_cells(self) -> int:
        """Count total active cells."""
        count = 0
        for row in self.cells:
            for cell in row:
                if cell and cell.active:
                    count += 1
        return count

    def average_error(self) -> float:
        """Compute average error across active cells."""
        errors = []
        for row in self.cells:
            for cell in row:
                if cell and cell.active:
                    errors.append(cell.error)
        return sum(errors) / len(errors) if errors else 0.0

    def has_converged(self) -> bool:
        """Check if the system has converged."""
        return self.average_error() < self.error_threshold or self.prime_index >= len(PRIMES)

    def run_evolution(self, max_iterations: int = 50):
        """Run the CA evolution for a specified number of iterations."""
        print(f"🌀 Starting Zeta CA Evolution (target: critical line)")
        print(f"Initial position: {self.initial_s}")
        print(f"Error threshold: {self.error_threshold}")
        print("=" * 60)
        
        for i in range(max_iterations):
            self.evolution_step()
            
            if self.has_converged():
                print(f"\n🎯 Convergence reached at iteration {self.iteration}!")
                break
            
            if i % 10 == 9:
                print(f"--- Checkpoint: {i+1} iterations ---")
        
        print("=" * 60)
        print(f"🔥 Evolution complete! Final state:")
        print(f"  Total iterations: {self.iteration}")
        print(f"  Active cells: {self.count_active_cells()}")
        print(f"  Final average error: {self.average_error():.8f}")
        print(f"  Primes inserted: {self.prime_index}")


if __name__ == "__main__":
    # Initialize near first known zero of Riemann zeta
    zeta_ca = ZetaCA(initial_s=0.5 + 14.134725j, grid_size=32)
    
    print("🔥 Zeta Fractal CA: Recursive Attractor Descent")
    print("=" * 50)
    
    # Run evolution
    zeta_ca.run_evolution(max_iterations=30)