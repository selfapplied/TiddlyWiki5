#!/usr/bin/env python3
"""
Structural Meaning Parser
Based on the insight that meaning mass lives in structure: spaces, punctuation, formatting
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter

@dataclass
class StructuralMass:
    """Quantifies the semantic mass of structural elements"""
    whitespace_mass: float      # Density of meaning in spacing
    punctuation_mass: float     # Semantic weight of punctuation
    boundary_mass: float        # Mathematical/formatting boundaries
    rhythm_mass: float          # Pattern repetition and flow
    tension_mass: float         # Questions, ellipses, breaks

@dataclass
class MeaningParticle:
    """A fundamental unit of conversational meaning"""
    position: int               # Character position in text
    particle_type: str          # 'space', 'punct', 'boundary', 'rhythm'
    mass: float                 # Semantic mass/importance
    content: str                # The actual character(s)
    influence_radius: int       # How far this particle affects meaning
    connections: List[int]      # Indices of connected particles

class StructuralSemanticParser:
    """Parse conversations as physics of meaning particles"""
    
    def __init__(self):
        # Mass coefficients for different structural elements
        self.mass_coefficients = {
            # Whitespace mass (semantic boundaries)
            'single_space': 0.1,
            'double_newline': 2.0,      # Paragraph breaks = high mass
            'triple_newline': 5.0,      # Section breaks = very high mass
            'indent_space': 0.3,        # Indentation = structural mass
            
            # Punctuation mass (semantic signals)
            'period': 1.0,              # Completion
            'question': 3.0,            # Exploration branch (high mass!)
            'exclamation': 2.0,         # Emphasis
            'ellipsis': 4.0,            # Continuation/suspension (very high!)
            'colon': 1.5,               # Definition/explanation
            'semicolon': 1.2,           # Complex connection
            'comma': 0.5,               # Lightweight connection
            
            # Mathematical boundary mass
            'dollar_single': 2.0,       # Inline math
            'dollar_double': 4.0,       # Display math (high mass!)
            'latex_bracket': 3.0,       # LaTeX boundaries
            'backslash': 1.0,           # Command indicators
            
            # Rhythm and pattern mass
            'repeated_char': 1.5,       # ???, ---, ===
            'list_marker': 2.0,         # bullets, numbers
            'emphasis_marker': 1.8,     # *, _, **
        }
    
    def parse_meaning_particles(self, text: str) -> List[MeaningParticle]:
        """Extract meaning particles from text structure"""
        particles = []
        
        # Analyze character by character for structural meaning
        i = 0
        while i < len(text):
            particle = self._detect_particle_at_position(text, i)
            if particle:
                particles.append(particle)
                i += len(particle.content)
            else:
                i += 1
        
        # Calculate particle connections (gravitational influences)
        self._calculate_particle_connections(particles, text)
        
        return particles
    
    def _detect_particle_at_position(self, text: str, pos: int) -> MeaningParticle:
        """Detect meaning particle at specific position"""
        if pos >= len(text):
            return None
        
        char = text[pos]
        
        # Check for multi-character patterns first
        particle = self._check_multichar_patterns(text, pos)
        if particle:
            return particle
        
        # Single character structural elements
        if char == ' ':
            # Analyze space context
            return self._analyze_space_particle(text, pos)
        
        elif char in '.?!':
            return self._analyze_punctuation_particle(text, pos, char)
        
        elif char == '$':
            return self._analyze_math_boundary(text, pos)
        
        elif char in ':;,':
            mass = self.mass_coefficients.get(self._punct_name(char), 0.5)
            return MeaningParticle(
                position=pos,
                particle_type='punct',
                mass=mass,
                content=char,
                influence_radius=self._calculate_influence_radius(mass),
                connections=[]
            )
        
        return None
    
    def _check_multichar_patterns(self, text: str, pos: int) -> MeaningParticle:
        """Check for multi-character structural patterns"""
        
        # Triple newlines (section breaks - massive semantic mass!)
        if pos + 2 < len(text) and text[pos:pos+3] == '\n\n\n':
            return MeaningParticle(
                position=pos,
                particle_type='boundary',
                mass=self.mass_coefficients['triple_newline'],
                content='\n\n\n',
                influence_radius=50,  # Very wide influence
                connections=[]
            )
        
        # Double newlines (paragraph breaks)
        if pos + 1 < len(text) and text[pos:pos+2] == '\n\n':
            return MeaningParticle(
                position=pos,
                particle_type='boundary',
                mass=self.mass_coefficients['double_newline'],
                content='\n\n',
                influence_radius=20,
                connections=[]
            )
        
        # Ellipsis (continuation/suspension - high semantic mass!)
        if pos + 2 < len(text) and text[pos:pos+3] == '...':
            return MeaningParticle(
                position=pos,
                particle_type='tension',
                mass=self.mass_coefficients['ellipsis'],
                content='...',
                influence_radius=15,
                connections=[]
            )
        
        # Display math boundaries $$
        if pos + 1 < len(text) and text[pos:pos+2] == '$$':
            return MeaningParticle(
                position=pos,
                particle_type='boundary',
                mass=self.mass_coefficients['dollar_double'],
                content='$$',
                influence_radius=25,
                connections=[]
            )
        
        # LaTeX brackets \[ or \]
        if pos + 1 < len(text) and text[pos:pos+2] in ['\\[', '\\]']:
            return MeaningParticle(
                position=pos,
                particle_type='boundary',
                mass=self.mass_coefficients['latex_bracket'],
                content=text[pos:pos+2],
                influence_radius=20,
                connections=[]
            )
        
        # Repeated characters (???, ---, ===)
        char = text[pos]
        if char in '?-=*_':
            repeat_count = 1
            while (pos + repeat_count < len(text) and 
                   text[pos + repeat_count] == char):
                repeat_count += 1
            
            if repeat_count >= 2:  # At least 2 repetitions
                mass = self.mass_coefficients['repeated_char'] * (repeat_count / 2)
                return MeaningParticle(
                    position=pos,
                    particle_type='rhythm',
                    mass=mass,
                    content=char * repeat_count,
                    influence_radius=int(5 + repeat_count * 2),
                    connections=[]
                )
        
        return None
    
    def _analyze_space_particle(self, text: str, pos: int) -> MeaningParticle:
        """Analyze semantic mass of space character"""
        # Count consecutive spaces
        space_count = 0
        i = pos
        while i < len(text) and text[i] == ' ':
            space_count += 1
            i += 1
        
        # Determine if this is indentation (start of line)
        is_indent = pos == 0 or (pos > 0 and text[pos-1] == '\n')
        
        if is_indent:
            mass = self.mass_coefficients['indent_space'] * space_count
            particle_type = 'structure'
        else:
            mass = self.mass_coefficients['single_space'] * space_count
            particle_type = 'space'
        
        return MeaningParticle(
            position=pos,
            particle_type=particle_type,
            mass=mass,
            content=' ' * space_count,
            influence_radius=self._calculate_influence_radius(mass),
            connections=[]
        )
    
    def _analyze_punctuation_particle(self, text: str, pos: int, char: str) -> MeaningParticle:
        """Analyze semantic mass of punctuation"""
        punct_names = {'.': 'period', '?': 'question', '!': 'exclamation'}
        punct_name = punct_names.get(char, 'unknown')
        
        mass = self.mass_coefficients.get(punct_name, 1.0)
        
        # Questions have especially high mass - they're exploration branches!
        if char == '?':
            # Check for multiple question marks
            question_count = 1
            i = pos + 1
            while i < len(text) and text[i] == '?':
                question_count += 1
                i += 1
            
            mass *= question_count  # Multiple ??? = exponential mass increase
        
        return MeaningParticle(
            position=pos,
            particle_type='punct',
            mass=mass,
            content=char,
            influence_radius=self._calculate_influence_radius(mass),
            connections=[]
        )
    
    def _analyze_math_boundary(self, text: str, pos: int) -> MeaningParticle:
        """Analyze mathematical boundary markers"""
        if pos + 1 < len(text) and text[pos+1] == '$':
            # Already handled in multichar patterns
            return None
        
        # Single dollar sign - inline math
        return MeaningParticle(
            position=pos,
            particle_type='boundary',
            mass=self.mass_coefficients['dollar_single'],
            content='$',
            influence_radius=self._calculate_influence_radius(2.0),
            connections=[]
        )
    
    def _calculate_influence_radius(self, mass: float) -> int:
        """Calculate how far a particle's influence extends"""
        # Higher mass = wider influence radius
        return max(1, int(mass * 3))
    
    def _punct_name(self, char: str) -> str:
        """Get punctuation name for mass lookup"""
        names = {':': 'colon', ';': 'semicolon', ',': 'comma'}
        return names.get(char, 'unknown')
    
    def _calculate_particle_connections(self, particles: List[MeaningParticle], text: str):
        """Calculate gravitational connections between meaning particles"""
        for i, particle in enumerate(particles):
            for j, other in enumerate(particles):
                if i != j:
                    distance = abs(particle.position - other.position)
                    
                    # Connection strength based on mass and distance
                    connection_strength = (particle.mass * other.mass) / (distance + 1)
                    
                    # Connect if strength exceeds threshold
                    if connection_strength > 0.5:
                        particle.connections.append(j)
    
    def calculate_structural_mass(self, particles: List[MeaningParticle]) -> StructuralMass:
        """Calculate overall structural mass metrics"""
        
        # Categorize particles by type
        whitespace_particles = [p for p in particles if p.particle_type in ['space', 'structure']]
        punct_particles = [p for p in particles if p.particle_type == 'punct']
        boundary_particles = [p for p in particles if p.particle_type == 'boundary']
        rhythm_particles = [p for p in particles if p.particle_type == 'rhythm']
        tension_particles = [p for p in particles if p.particle_type == 'tension']
        
        return StructuralMass(
            whitespace_mass=sum(p.mass for p in whitespace_particles),
            punctuation_mass=sum(p.mass for p in punct_particles),
            boundary_mass=sum(p.mass for p in boundary_particles),
            rhythm_mass=sum(p.mass for p in rhythm_particles),
            tension_mass=sum(p.mass for p in tension_particles)
        )
    
    def export_meaning_physics(self, particles: List[MeaningParticle], 
                              structural_mass: StructuralMass) -> Dict:
        """Export the physics analysis of meaning"""
        return {
            'total_particles': len(particles),
            'structural_mass': {
                'whitespace': structural_mass.whitespace_mass,
                'punctuation': structural_mass.punctuation_mass,
                'boundaries': structural_mass.boundary_mass,
                'rhythm': structural_mass.rhythm_mass,
                'tension': structural_mass.tension_mass
            },
            'high_mass_particles': [
                {
                    'position': p.position,
                    'type': p.particle_type,
                    'mass': p.mass,
                    'content': repr(p.content),
                    'influence': p.influence_radius,
                    'connections': len(p.connections)
                }
                for p in particles if p.mass > 2.0
            ],
            'meaning_density': sum(p.mass for p in particles) / len(particles) if particles else 0
        }

def main():
    """Demonstrate structural meaning analysis"""
    parser = StructuralSemanticParser()
    
    print("🌊 STRUCTURAL MEANING PARSER")
    print("Based on the insight: Mass lives in structure, not words!")
    print()
    print("High-mass structural elements:")
    print("  ??? = Exploration branches (high mass)")
    print("  ... = Suspension/continuation (high mass)")  
    print("  \\n\\n\\n = Section breaks (massive!)")
    print("  $$ = Mathematical boundaries (high mass)")
    print()
    print("Ready to analyze the physics of meaning...")

if __name__ == '__main__':
    main()