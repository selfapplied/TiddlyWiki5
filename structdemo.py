#!/usr/bin/env python3
"""
Structural Meaning Demo
Shows how punctuation and spacing carry the mass of meaning
"""

from structure import StructuralSemanticParser
import json

def demo_structural_meaning():
    """Demonstrate structural meaning analysis"""
    
    # Sample text with various structural elements
    sample_text = """The Riemann Hypothesis???

What if we approach this differently...


Let's consider: $$\\zeta(s) = 0$$

This is significant!!! The zeros are critical.

Alternative approach:
  - Symbolic methods
  - Computational verification  
  - Fractal analysis???


Wait... perhaps there's a pattern here?

Mathematical structures as data compression---interesting idea."""

    parser = StructuralSemanticParser()
    
    print("🌊 STRUCTURAL MEANING PHYSICS ANALYSIS")
    print("=" * 60)
    print("Analyzing meaning particles in mathematical conversation...")
    print()
    
    # Parse meaning particles
    particles = parser.parse_meaning_particles(sample_text)
    structural_mass = parser.calculate_structural_mass(particles)
    
    # Show high-mass particles
    print("🔥 HIGH-MASS MEANING PARTICLES:")
    high_mass = [p for p in particles if p.mass > 2.0]
    
    for particle in high_mass:
        context_start = max(0, particle.position - 10)
        context_end = min(len(sample_text), particle.position + 10)
        context = sample_text[context_start:context_end].replace('\n', '\\n')
        
        print(f"  Pos {particle.position:3d}: {particle.particle_type:10s} "
              f"Mass={particle.mass:4.1f} "
              f"Radius={particle.influence_radius:2d} "
              f"'{particle.content}' in '{context}'")
    
    print()
    print("📊 STRUCTURAL MASS DISTRIBUTION:")
    print(f"  Whitespace Mass:  {structural_mass.whitespace_mass:6.1f}")
    print(f"  Punctuation Mass: {structural_mass.punctuation_mass:6.1f}")
    print(f"  Boundary Mass:    {structural_mass.boundary_mass:6.1f}")
    print(f"  Rhythm Mass:      {structural_mass.rhythm_mass:6.1f}")
    print(f"  Tension Mass:     {structural_mass.tension_mass:6.1f}")
    
    total_mass = (structural_mass.whitespace_mass + 
                  structural_mass.punctuation_mass + 
                  structural_mass.boundary_mass + 
                  structural_mass.rhythm_mass + 
                  structural_mass.tension_mass)
    
    print(f"  TOTAL MASS:       {total_mass:6.1f}")
    
    print()
    print("💫 PARTICLE CONNECTIONS:")
    connected_particles = [p for p in particles if p.connections and p.mass > 1.0]
    for particle in connected_particles[:5]:  # Show first 5
        print(f"  Particle at {particle.position} ({particle.content!r}) "
              f"connects to {len(particle.connections)} others")
    
    print()
    print("🎯 INSIGHTS:")
    print(f"• Total meaning particles detected: {len(particles)}")
    print(f"• High-mass particles (>2.0): {len(high_mass)}")
    print(f"• Connected particles: {len(connected_particles)}")
    print(f"• Meaning density: {total_mass / len(particles) if particles else 0:.2f} mass/particle")
    
    # Show the most significant structural elements
    print()
    print("⚡ MOST SIGNIFICANT STRUCTURES:")
    if structural_mass.tension_mass > 3.0:
        print("  📍 High tension - many questions/ellipses detected")
    if structural_mass.boundary_mass > 5.0:
        print("  🧮 Mathematical boundaries - heavy LaTeX usage")
    if structural_mass.whitespace_mass > 4.0:
        print("  📄 Structural organization - good paragraph breaks")
    if structural_mass.rhythm_mass > 2.0:
        print("  🎵 Rhythmic patterns - repeated characters for emphasis")

if __name__ == '__main__':
    demo_structural_meaning()