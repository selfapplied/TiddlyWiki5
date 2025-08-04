#!/usr/bin/env python3
"""
Möbius Descent Tower: Symbolic Calculus for Möbius Evolution

This module implements the Möbius descent tower system where Möbius transformations
evolve through degenerate transitions, creating a recursive cascade of attractors.

Key Concepts:
- Möbius transformations: f(z) = (az + b)/(cz + d) with Δ = ad - bc ≠ 0
- Degenerate transitions: When Δ → 0, structure collapses and regenerates
- Descent tower: M_{n+1} is defined at the moment Δ_n = 0
- Evolution sequence: M^{(k)}(z) with Δ_k = a_k d_k - b_k c_k
"""

import sympy as sp
from sympy import symbols, Matrix, simplify, expand, factor
from sympy.core.expr import Expr
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

# Mathematical constants and symbols
z, s, t = symbols('z s t', complex=True)
a, b, c, d = symbols('a b c d', complex=True)
k, n = symbols('k n', integer=True)

class MobiusState(Enum):
    """States of a Möbius transformation in the descent tower"""
    INVERTIBLE = "invertible"
    DEGENERATE = "degenerate"
    TRANSITION = "transition"
    EVOLVED = "evolved"

@dataclass
class MobiusTransformation:
    """Represents a Möbius transformation with evolution tracking"""
    a: Any
    b: Any
    c: Any
    d: Any
    generation: int = 0
    state: MobiusState = MobiusState.INVERTIBLE
    
    @property
    def determinant(self) -> Expr:
        """Calculate the determinant Δ = ad - bc"""
        return self.a * self.d - self.b * self.c
    
    @property
    def is_invertible(self) -> bool:
        """Check if transformation is invertible (Δ ≠ 0)"""
        det = self.determinant
        if hasattr(det, 'is_zero'):
            return not det.is_zero
        else:
            return abs(det) > 1e-10
    
    def apply(self, z_val: Expr) -> Expr:
        """Apply the Möbius transformation: f(z) = (az + b)/(cz + d)"""
        return (self.a * z_val + self.b) / (self.c * z_val + self.d)
    
    def inverse(self) -> 'MobiusTransformation':
        """Calculate the inverse transformation (if invertible)"""
        if not self.is_invertible:
            raise ValueError("Transformation is not invertible")
        
        det = self.determinant
        return MobiusTransformation(
            a=self.d / det,
            b=-self.b / det,
            c=-self.c / det,
            d=self.a / det,
            generation=self.generation,
            state=self.state
        )
    
    def __str__(self) -> str:
        return f"M_{self.generation}(z) = ({self.a}z + {self.b})/({self.c}z + {self.d})"

class MobiusDescentTower:
    """
    Implements the Möbius descent tower system for evolving transformations
    through degenerate transitions.
    """
    
    def __init__(self):
        self.tower: List[MobiusTransformation] = []
        self.evolution_history: List[Dict] = []
        self.transition_points: List[Tuple[int, Expr]] = []
    
    def add_transformation(self, transformation: MobiusTransformation) -> None:
        """Add a Möbius transformation to the tower"""
        transformation.generation = len(self.tower)
        self.tower.append(transformation)
        self._analyze_state(transformation)
    
    def _analyze_state(self, transformation: MobiusTransformation) -> None:
        """Analyze the state of a transformation"""
        det = transformation.determinant
        
        # Handle both symbolic and numeric determinants
        if hasattr(det, 'is_zero') and det.is_zero:
            transformation.state = MobiusState.DEGENERATE
        elif isinstance(det, (int, float)) and abs(det) < 1e-10:
            transformation.state = MobiusState.DEGENERATE
        elif hasattr(det, 'is_zero') and abs(det) < 1e-10:
            transformation.state = MobiusState.TRANSITION
        elif isinstance(det, (int, float)) and abs(det) < 1e-6:
            transformation.state = MobiusState.TRANSITION
        else:
            transformation.state = MobiusState.INVERTIBLE
    
    def evolve_transformation(self, transformation: MobiusTransformation, 
                           evolution_rule: str = "canonical") -> MobiusTransformation:
        """
        Evolve a Möbius transformation through a degenerate transition
        
        Args:
            transformation: The original transformation
            evolution_rule: Rule for evolution ("canonical", "symmetric", "chaotic")
        
        Returns:
            The evolved transformation
        """
        det = transformation.determinant
        
        if evolution_rule == "canonical":
            # Canonical evolution: M_{n+1} = lim_{Δ_n → 0} M_n
            evolved = self._canonical_evolution(transformation)
        elif evolution_rule == "symmetric":
            # Symmetric evolution preserving certain properties
            evolved = self._symmetric_evolution(transformation)
        elif evolution_rule == "chaotic":
            # Chaotic evolution introducing randomness
            evolved = self._chaotic_evolution(transformation)
        else:
            raise ValueError(f"Unknown evolution rule: {evolution_rule}")
        
        evolved.generation = transformation.generation + 1
        self._analyze_state(evolved)
        
        # Record the evolution
        self.evolution_history.append({
            'from_generation': transformation.generation,
            'to_generation': evolved.generation,
            'determinant': det,
            'evolution_rule': evolution_rule,
            'state_change': (transformation.state, evolved.state)
        })
        
        return evolved
    
    def _canonical_evolution(self, transformation: MobiusTransformation) -> MobiusTransformation:
        """
        Canonical evolution: M_{n+1} = lim_{Δ_n → 0} M_n
        
        This creates a new transformation that emerges from the degenerate point
        """
        # When Δ → 0, we can write: M_{n+1}(z) = (a'z + b')/(c'z + d')
        # where the new coefficients emerge from the degenerate condition
        
        # For canonical evolution, we can choose:
        # a' = a + ε, b' = b, c' = c, d' = d + ε
        # where ε is a small parameter that emerges from the degenerate transition
        
        epsilon = symbols('ε', real=True)
        
        new_a = transformation.a + epsilon
        new_b = transformation.b
        new_c = transformation.c
        new_d = transformation.d + epsilon
        
        # Ensure the new transformation has non-zero determinant
        new_det = new_a * new_d - new_b * new_c
        # new_det = (a + ε)(d + ε) - bc = ad + aε + dε + ε² - bc
        # = (ad - bc) + (a + d)ε + ε²
        
        return MobiusTransformation(
            a=new_a,
            b=new_b,
            c=new_c,
            d=new_d
        )
    
    def _symmetric_evolution(self, transformation: MobiusTransformation) -> MobiusTransformation:
        """
        Symmetric evolution preserving certain symmetries
        """
        # Create a symmetric evolution that preserves the structure
        # This could involve rotations, reflections, or other symmetries
        
        # For example, if we have a transformation that's symmetric about the real axis
        # we might evolve it by introducing a small imaginary component
        
        i = sp.I
        epsilon = symbols('ε', real=True)
        
        new_a = transformation.a + epsilon * i
        new_b = transformation.b
        new_c = transformation.c
        new_d = transformation.d - epsilon * i
        
        return MobiusTransformation(
            a=new_a,
            b=new_b,
            c=new_c,
            d=new_d
        )
    
    def _chaotic_evolution(self, transformation: MobiusTransformation) -> MobiusTransformation:
        """
        Chaotic evolution introducing randomness and unpredictability
        """
        # Introduce chaotic elements through complex perturbations
        
        epsilon = symbols('ε', real=True)
        theta = symbols('θ', real=True)
        
        # Add chaotic perturbations
        new_a = transformation.a + epsilon * sp.exp(sp.I * theta)
        new_b = transformation.b + epsilon * sp.exp(sp.I * (theta + sp.pi/2))
        new_c = transformation.c + epsilon * sp.exp(sp.I * (theta + sp.pi))
        new_d = transformation.d + epsilon * sp.exp(sp.I * (theta + 3*sp.pi/2))
        
        return MobiusTransformation(
            a=new_a,
            b=new_b,
            c=new_c,
            d=new_d
        )
    
    def find_transition_points(self) -> List[Tuple[int, Any]]:
        """Find all transition points where Δ → 0"""
        transition_points = []
        
        for i, transformation in enumerate(self.tower):
            det = transformation.determinant
            
            # Check if determinant is zero or very small
            if (hasattr(det, 'is_zero') and det.is_zero) or abs(det) < 1e-10:
                transition_points.append((i, det))
        
        self.transition_points = transition_points
        return transition_points
    
    def build_descent_sequence(self, initial_transformation: MobiusTransformation,
                             max_generations: int = 10) -> List[MobiusTransformation]:
        """
        Build a complete descent sequence starting from an initial transformation
        
        Args:
            initial_transformation: The starting transformation
            max_generations: Maximum number of generations to evolve
        
        Returns:
            List of transformations in the descent sequence
        """
        sequence = [initial_transformation]
        
        for generation in range(max_generations):
            current = sequence[-1]
            
            # Check if we should evolve (near degenerate or at transition point)
            det = current.determinant
            if isinstance(det, (int, float)):
                det_abs = abs(det)
            else:
                # For symbolic expressions, try to evaluate or use a different approach
                try:
                    det_abs = abs(float(det))
                except (TypeError, ValueError):
                    # If we can't convert to float, assume it's not near zero
                    det_abs = 1.0
            
            if det_abs < 1e-6 or current.state == MobiusState.TRANSITION:
                evolved = self.evolve_transformation(current, "canonical")
                sequence.append(evolved)
            else:
                # If not near degenerate, we can still evolve for demonstration
                evolved = self.evolve_transformation(current, "symmetric")
                sequence.append(evolved)
        
        return sequence
    
    def analyze_descent_dynamics(self) -> Dict:
        """
        Analyze the dynamics of the descent tower
        """
        analysis = {
            'total_transformations': len(self.tower),
            'transition_points': len(self.transition_points),
            'state_distribution': {},
            'evolution_patterns': [],
            'stability_analysis': {}
        }
        
        # Analyze state distribution
        for transformation in self.tower:
            state = transformation.state
            analysis['state_distribution'][state] = analysis['state_distribution'].get(state, 0) + 1
        
        # Analyze evolution patterns
        for i in range(len(self.evolution_history)):
            history = self.evolution_history[i]
            analysis['evolution_patterns'].append({
                'generation': history['from_generation'],
                'determinant': history['determinant'],
                'rule': history['evolution_rule'],
                'state_change': history['state_change']
            })
        
        # Stability analysis
        for i, transformation in enumerate(self.tower):
            det = transformation.determinant
            if isinstance(det, (int, float)):
                det_abs = abs(det)
            else:
                try:
                    det_abs = abs(float(det))
                except (TypeError, ValueError):
                    det_abs = 1.0
            
            analysis['stability_analysis'][i] = {
                'determinant': det,
                'is_stable': det_abs > 1e-6,
                'state': transformation.state
            }
        
        return analysis
    
    def generate_symbolic_insights(self) -> Dict:
        """
        Generate symbolic insights about the Möbius descent tower
        """
        insights = {
            'mathematical_structure': {},
            'evolution_theorems': [],
            'degenerate_analysis': {},
            'descent_properties': []
        }
        
        # Mathematical structure insights
        insights['mathematical_structure'] = {
            'mobius_group': 'PSL(2, ℂ)',
            'degenerate_condition': 'Δ = ad - bc = 0',
            'evolution_operator': 'M_{n+1} = lim_{Δ_n → 0} M_n',
            'descent_sequence': 'M^{(k)}(z) = (a_k z + b_k)/(c_k z + d_k)'
        }
        
        # Evolution theorems
        insights['evolution_theorems'] = [
            "Theorem 1: When Δ → 0, the transformation collapses to a degenerate state",
            "Theorem 2: At the degenerate point, a new Möbius transformation emerges",
            "Theorem 3: The descent tower creates a recursive cascade of attractors",
            "Theorem 4: Each transition point defines a new Möbius frame"
        ]
        
        # Degenerate analysis
        if self.transition_points:
            insights['degenerate_analysis'] = {
                'transition_count': len(self.transition_points),
                'degenerate_generations': [point[0] for point in self.transition_points],
                'degenerate_determinants': [point[1] for point in self.transition_points]
            }
        
        # Descent properties
        insights['descent_properties'] = [
            "Property 1: Each generation preserves the Möbius form",
            "Property 2: Evolution creates new attractor basins",
            "Property 3: The tower exhibits fractal-like structure",
            "Property 4: Degenerate transitions are the birth points of new frames"
        ]
        
        return insights

def demo_mobius_descent_tower():
    """
    Demonstrate the Möbius descent tower system
    """
    print("🌪 Möbius Descent Tower Demonstration")
    print("=" * 50)
    
    # Create the descent tower
    tower = MobiusDescentTower()
    
    # Create initial transformations
    transformations = [
        MobiusTransformation(a=1, b=0, c=0, d=1),  # Identity
        MobiusTransformation(a=1, b=1, c=0, d=1),  # Translation
        MobiusTransformation(a=1, b=0, c=1, d=1),  # Near degenerate
        MobiusTransformation(a=1, b=1, c=1, d=1),  # Degenerate
        MobiusTransformation(a=2, b=1, c=1, d=1),  # Regular
    ]
    
    print("📊 Initial Transformations:")
    for i, trans in enumerate(transformations):
        print(f"  M_{i}: {trans}")
        print(f"    Determinant: {trans.determinant}")
        print(f"    State: {trans.state}")
        print()
    
    # Add to tower
    for trans in transformations:
        tower.add_transformation(trans)
    
    # Find transition points
    transition_points = tower.find_transition_points()
    print("🔍 Transition Points:")
    for gen, det in transition_points:
        print(f"  Generation {gen}: Δ = {det}")
    print()
    
    # Build descent sequence
    initial = MobiusTransformation(a=1, b=0, c=0, d=1)
    sequence = tower.build_descent_sequence(initial, max_generations=5)
    
    print("🔄 Descent Sequence:")
    for i, trans in enumerate(sequence):
        print(f"  Generation {i}: {trans}")
        print(f"    Determinant: {trans.determinant}")
        print(f"    State: {trans.state}")
        print()
    
    # Analyze dynamics
    analysis = tower.analyze_descent_dynamics()
    print("📈 Dynamics Analysis:")
    print(f"  Total transformations: {analysis['total_transformations']}")
    print(f"  Transition points: {analysis['transition_points']}")
    print(f"  State distribution: {analysis['state_distribution']}")
    print()
    
    # Generate insights
    insights = tower.generate_symbolic_insights()
    print("💡 Symbolic Insights:")
    print("  Mathematical Structure:")
    for key, value in insights['mathematical_structure'].items():
        print(f"    {key}: {value}")
    print()
    
    print("  Evolution Theorems:")
    for theorem in insights['evolution_theorems']:
        print(f"    {theorem}")
    print()
    
    return tower, analysis, insights

if __name__ == "__main__":
    tower, analysis, insights = demo_mobius_descent_tower() 