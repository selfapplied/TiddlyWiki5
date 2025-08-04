#!/usr/bin/env python3
"""
Semantic Parser Demo
Shows how the parser analyzes mathematical conversations
"""

from textparse import SemanticTextParser, LineType
from pathlib import Path

def demo_analysis():
    """Demonstrate semantic analysis on sample mathematical text"""
    
    # Sample mathematical conversation (simulating DeepSeek format)
    sample_text = """The Riemann Hypothesis states that all non-trivial zeros of the zeta function have real part 1/2.

Let's consider the functional equation: $$\\zeta(s) = 2^s \\pi^{s-1} \\sin(\\frac{\\pi s}{2}) \\Gamma(1-s) \\zeta(1-s)$$

This suggests a deep symmetry in the distribution of prime numbers.


What if we approach this through symbolic differentiation? 

Consider the derivative: $\\frac{d}{ds} \\log \\zeta(s) = -\\frac{\\zeta'(s)}{\\zeta(s)}$

The zeros of $\\zeta(s)$ correspond to poles of this logarithmic derivative.


Theorem: If the Riemann Hypothesis is true, then the prime counting function $\\pi(x)$ satisfies certain bounds.

Proof: We can verify this computationally using symbolic calculus...

Alternative approach: What about using fractal analysis of the zero distribution?

Perhaps the zeros exhibit self-similar patterns at different scales?

This could connect to the libz compression idea - mathematical structures as compressible data."""

    # Create temporary file
    temp_file = Path('temp_conversation.txt')
    with open(temp_file, 'w') as f:
        f.write(sample_text)
    
    # Parse with semantic analyzer
    parser = SemanticTextParser()
    segments = parser.parse_conversation(temp_file)
    
    # Display analysis
    print("🔍 SEMANTIC ANALYSIS DEMO")
    print("=" * 50)
    
    for i, segment in enumerate(segments):
        print(f"\n📊 SEGMENT {i+1}: {segment.segment_type.upper()}")
        print(f"Lines {segment.start_line}-{segment.end_line}")
        print(f"Math Density: {segment.math_density:.2f}")
        print(f"Explore Branches: {segment.explore_branches}")
        print(f"Completion Boundary: {'✓' if segment.completion_boundary else '✗'}")
        
        print("\n📝 LINE ANNOTATIONS:")
        for annotation in segment.annotations:
            if annotation.line_type != LineType.WHITESPACE:
                line_preview = annotation.content[:60] + "..." if len(annotation.content) > 60 else annotation.content
                
                print(f"  L{annotation.line_num:2d}: {annotation.line_type.value:12s} "
                      f"[{annotation.meaning_score:.1f}] {line_preview}")
                
                if annotation.tags:
                    print(f"       Tags: {', '.join(annotation.tags)}")
                
                if annotation.math_boundaries:
                    boundaries = ', '.join(f"{k}:{v}" for k,v in annotation.math_boundaries.items())
                    print(f"       Math: {boundaries}")
                
                if annotation.explore_signals:
                    print(f"       Explore: {', '.join(annotation.explore_signals)}")
            
            elif annotation.whitespace_combo > 0:
                print(f"  L{annotation.line_num:2d}: {'whitespace':12s} "
                      f"[combo: {annotation.whitespace_combo}]")
    
    # Clean up
    temp_file.unlink()
    
    print("\n" + "=" * 50)
    print("🎯 ANALYSIS INSIGHTS:")
    print(f"• Total segments: {len(segments)}")
    print(f"• Mathematical segments: {sum(1 for s in segments if s.math_density > 0.3)}")
    print(f"• Exploration points: {sum(s.explore_branches for s in segments)}")
    print(f"• Completion boundaries: {sum(1 for s in segments if s.completion_boundary)}")

if __name__ == '__main__':
    demo_analysis()