#!/usr/bin/env python3
"""
Meaning Atoms Analyzer: Elementary Units of Meaning via Compression
==================================================================

Uses compression techniques to discover the fundamental "atoms" of meaning
in text and code. When we compress, we reveal the irreducible units of
information - the elementary particles of meaning itself.

This connects to:
- Kolmogorov Complexity: Shortest description = true complexity
- Information Theory: Compression reveals essential information content
- Semantic Atoms: Frequent patterns = fundamental meaning units
- Computational Linguistics: Meaning has atomic structure
"""

import zlib
import heapq
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Any
from dataclasses import dataclass
import json

@dataclass
class MeaningAtom:
    """An elementary unit of meaning discovered through compression."""
    pattern: str              # The actual pattern/string
    frequency: int            # How often it appears
    compression_ratio: float  # How much it compresses
    semantic_weight: float    # Computed semantic importance
    contexts: List[str]       # Where it appears
    meaning_type: str         # code, text, mixed, structural
    
    def __str__(self):
        return f"Atom['{self.pattern[:20]}...'] freq:{self.frequency} weight:{self.semantic_weight:.3f}"

class MeaningAtomAnalyzer:
    """
    Discovers elementary atoms of meaning through compression analysis.
    
    The key insight: compression reveals what's truly fundamental.
    Patterns that compress well are redundant (atoms of structure).
    Patterns that don't compress are irreducible (atoms of meaning).
    """
    
    def __init__(self):
        self.atoms = []
        self.pattern_frequencies = Counter()
        self.compression_cache = {}
        
        # Different pattern types for different kinds of meaning
        self.pattern_extractors = {
            'words': self._extract_words,
            'code_tokens': self._extract_code_tokens,  
            'n_grams': self._extract_n_grams,
            'structural': self._extract_structural_patterns,
            'semantic_units': self._extract_semantic_units
        }
    
    def analyze_text_for_atoms(self, text: str, text_type: str = "mixed") -> List[MeaningAtom]:
        """Extract meaning atoms from text using compression analysis."""
        print(f"🔬 Analyzing {len(text)} characters for meaning atoms...")
        
        # Extract all possible patterns
        all_patterns = {}
        for extractor_name, extractor_func in self.pattern_extractors.items():
            patterns = extractor_func(text)
            all_patterns[extractor_name] = patterns
            print(f"  {extractor_name}: {len(patterns)} patterns")
        
        # Analyze each pattern for compression characteristics
        atoms = []
        for pattern_type, patterns in all_patterns.items():
            for pattern in patterns:
                if len(pattern.strip()) < 2:  # Skip trivial patterns
                    continue
                    
                atom = self._analyze_pattern_as_atom(pattern, text, text_type, pattern_type)
                if atom and atom.semantic_weight > 0.1:  # Threshold for significance
                    atoms.append(atom)
        
        # Sort by semantic weight
        atoms.sort(key=lambda a: a.semantic_weight, reverse=True)
        self.atoms.extend(atoms)
        
        return atoms
    
    def _analyze_pattern_as_atom(self, pattern: str, full_text: str, text_type: str, pattern_type: str) -> MeaningAtom:
        """Analyze a pattern to determine if it's a fundamental meaning atom."""
        
        # Count frequency
        frequency = full_text.count(pattern)
        if frequency < 2:  # Must appear multiple times to be significant
            return None
        
        # Compute compression characteristics
        compression_ratio = self._compute_compression_ratio(pattern)
        
        # Find contexts where this pattern appears
        contexts = self._extract_contexts(pattern, full_text, context_size=50)
        
        # Compute semantic weight based on multiple factors
        semantic_weight = self._compute_semantic_weight(
            pattern, frequency, compression_ratio, contexts, pattern_type
        )
        
        return MeaningAtom(
            pattern=pattern,
            frequency=frequency,
            compression_ratio=compression_ratio,
            semantic_weight=semantic_weight,
            contexts=contexts[:5],  # Keep top 5 contexts
            meaning_type=self._classify_meaning_type(pattern, contexts)
        )
    
    def _compute_compression_ratio(self, text: str) -> float:
        """Compute compression ratio for a text pattern."""
        if text in self.compression_cache:
            return self.compression_cache[text]
        
        original_size = len(text.encode('utf-8'))
        if original_size == 0:
            return 0.0
        
        compressed_size = len(zlib.compress(text.encode('utf-8')))
        ratio = compressed_size / original_size
        
        self.compression_cache[text] = ratio
        return ratio
    
    def _compute_semantic_weight(self, pattern: str, frequency: int, compression_ratio: float, 
                                contexts: List[str], pattern_type: str) -> float:
        """
        Compute semantic weight of a pattern.
        
        Key insight: Meaning atoms have specific compression characteristics:
        - High frequency (appears often)
        - Moderate compression (not too redundant, not too random)
        - Rich contexts (appears in diverse situations)
        - Pattern complexity (not trivial)
        """
        
        # Frequency component (log scale to avoid dominance)
        freq_weight = min(1.0, frequency / 100.0)
        
        # Compression component (sweet spot around 0.4-0.8)
        # Too low = too redundant, too high = too random
        optimal_compression = 0.6
        compression_weight = 1.0 - abs(compression_ratio - optimal_compression)
        compression_weight = max(0.0, compression_weight)
        
        # Context diversity (more diverse contexts = higher weight)
        context_diversity = len(set(contexts)) / max(len(contexts), 1)
        
        # Pattern complexity (longer, more complex patterns get higher weight)
        complexity_weight = min(1.0, len(pattern) / 50.0)
        
        # Pattern type weights
        type_weights = {
            'words': 1.0,
            'code_tokens': 1.2,      # Code tokens are slightly more important
            'n_grams': 0.8,          # N-grams are less fundamental
            'structural': 1.5,       # Structural patterns are very important
            'semantic_units': 2.0    # Semantic units are most important
        }
        type_weight = type_weights.get(pattern_type, 1.0)
        
        # Combine all weights
        semantic_weight = (
            freq_weight * 0.3 +
            compression_weight * 0.3 +
            context_diversity * 0.2 +
            complexity_weight * 0.2
        ) * type_weight
        
        return semantic_weight
    
    def _extract_contexts(self, pattern: str, text: str, context_size: int = 50) -> List[str]:
        """Extract contexts where the pattern appears."""
        contexts = []
        start = 0
        
        while True:
            pos = text.find(pattern, start)
            if pos == -1:
                break
            
            # Extract context around the pattern
            context_start = max(0, pos - context_size)
            context_end = min(len(text), pos + len(pattern) + context_size)
            context = text[context_start:context_end].strip()
            
            if context and context not in contexts:
                contexts.append(context)
            
            start = pos + 1
        
        return contexts
    
    def _classify_meaning_type(self, pattern: str, contexts: List[str]) -> str:
        """Classify the type of meaning this atom represents."""
        
        # Check if it's code-like
        if re.search(r'[{}()\[\];=+\-*/><!]', pattern):
            return "code"
        
        # Check if it's structural (punctuation, formatting)
        if re.search(r'^[\s\n\t\.,;:!?\-_"\']+$', pattern):
            return "structural"
        
        # Check if it's pure text
        if re.search(r'^[a-zA-Z\s]+$', pattern):
            return "text"
        
        return "mixed"
    
    # Pattern extraction methods
    def _extract_words(self, text: str) -> List[str]:
        """Extract word-level patterns."""
        words = re.findall(r'\b\w+\b', text.lower())
        return list(set(words))
    
    def _extract_code_tokens(self, text: str) -> List[str]:
        """Extract code-like tokens and patterns."""
        # Common code patterns
        patterns = []
        
        # Function calls
        patterns.extend(re.findall(r'\w+\([^)]*\)', text))
        
        # Variable assignments
        patterns.extend(re.findall(r'\w+\s*=\s*[^;]+', text))
        
        # Import statements
        patterns.extend(re.findall(r'import\s+[\w.]+', text))
        patterns.extend(re.findall(r'from\s+[\w.]+\s+import\s+[\w,\s]+', text))
        
        # Control structures
        patterns.extend(re.findall(r'(if|for|while|def|class)\s+[^:]+:', text))
        
        return list(set(patterns))
    
    def _extract_n_grams(self, text: str, n: int = 3) -> List[str]:
        """Extract n-gram patterns."""
        words = text.split()
        n_grams = []
        
        for i in range(len(words) - n + 1):
            n_gram = ' '.join(words[i:i+n])
            n_grams.append(n_gram)
        
        return list(set(n_grams))
    
    def _extract_structural_patterns(self, text: str) -> List[str]:
        """Extract structural/formatting patterns."""
        patterns = []
        
        # Indentation patterns
        lines = text.split('\n')
        for line in lines:
            if line.strip():
                leading_whitespace = line[:len(line) - len(line.lstrip())]
                if leading_whitespace:
                    patterns.append(leading_whitespace)
        
        # Punctuation sequences
        patterns.extend(re.findall(r'[^\w\s]{2,}', text))
        
        # Common delimiters
        patterns.extend(re.findall(r'[{}()\[\]<>]', text))
        
        return list(set(patterns))
    
    def _extract_semantic_units(self, text: str) -> List[str]:
        """Extract semantic units using advanced pattern matching."""
        patterns = []
        
        # Multi-word concepts (title case)
        patterns.extend(re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text))
        
        # Technical terms (camelCase, snake_case)
        patterns.extend(re.findall(r'[a-z]+(?:[A-Z][a-z]*)+', text))  # camelCase
        patterns.extend(re.findall(r'\w+_\w+(?:_\w+)*', text))        # snake_case
        
        # Quoted strings (potential semantic units)
        patterns.extend(re.findall(r'"([^"]+)"', text))
        patterns.extend(re.findall(r"'([^']+)'", text))
        
        # Mathematical expressions
        patterns.extend(re.findall(r'[a-zA-Z]\s*[+\-*/=]\s*[a-zA-Z0-9]+', text))
        
        return list(set(patterns))
    
    def get_atomic_summary(self) -> Dict[str, Any]:
        """Get summary of discovered meaning atoms."""
        if not self.atoms:
            return {"error": "No atoms analyzed yet"}
        
        # Group atoms by type
        by_type = defaultdict(list)
        for atom in self.atoms:
            by_type[atom.meaning_type].append(atom)
        
        # Compute statistics
        total_atoms = len(self.atoms)
        avg_weight = sum(a.semantic_weight for a in self.atoms) / total_atoms
        
        top_atoms = sorted(self.atoms, key=lambda a: a.semantic_weight, reverse=True)[:10]
        
        return {
            "total_atoms": total_atoms,
            "average_semantic_weight": avg_weight,
            "atoms_by_type": {t: len(atoms) for t, atoms in by_type.items()},
            "top_atoms": [
                {
                    "pattern": atom.pattern[:50],
                    "weight": atom.semantic_weight,
                    "frequency": atom.frequency,
                    "type": atom.meaning_type
                } for atom in top_atoms
            ]
        }
    
    def find_atomic_relationships(self) -> List[Tuple[MeaningAtom, MeaningAtom, float]]:
        """Find relationships between meaning atoms."""
        relationships = []
        
        for i, atom1 in enumerate(self.atoms):
            for j, atom2 in enumerate(self.atoms[i+1:], i+1):
                # Compute relationship strength based on:
                # 1. Shared contexts
                # 2. Pattern similarity
                # 3. Frequency correlation
                
                shared_contexts = len(set(atom1.contexts) & set(atom2.contexts))
                context_similarity = shared_contexts / max(len(atom1.contexts), len(atom2.contexts), 1)
                
                # Pattern similarity (simple overlap for now)
                pattern_similarity = len(set(atom1.pattern.split()) & set(atom2.pattern.split())) / max(len(atom1.pattern.split()), len(atom2.pattern.split()), 1)
                
                # Frequency correlation
                freq_similarity = 1.0 - abs(atom1.frequency - atom2.frequency) / max(atom1.frequency, atom2.frequency)
                
                relationship_strength = (context_similarity + pattern_similarity + freq_similarity) / 3.0
                
                if relationship_strength > 0.3:  # Threshold for meaningful relationship
                    relationships.append((atom1, atom2, relationship_strength))
        
        return sorted(relationships, key=lambda r: r[2], reverse=True)

def demo_meaning_atoms():
    """Demonstrate meaning atoms discovery on various texts."""
    print("🔬 Meaning Atoms Analyzer: Elementary Units of Meaning")
    print("=" * 60)
    
    analyzer = MeaningAtomAnalyzer()
    
    # Test on different types of content
    test_texts = {
        "code": '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.processed = False
    
    def process(self):
        self.data = [x * 2 for x in self.data]
        self.processed = True
        return self.data
''',
        
        "text": '''
The LibZ Virtual Machine represents a revolutionary paradigm in computational thinking.
By using Huffman codes as opcodes and Y combinator automorphism discovery, we create
a universal computation model. This model bridges information theory, lambda calculus,
and group theory into a unified framework. The Y combinator serves as a universal
fixed-point finder, discovering the fundamental structure underlying all computation.
Information theory provides optimal encoding through Huffman trees. Group theory
captures the symmetries and automorphisms that define computational equivalence.
''',
        
        "mixed": '''
# LibZ VM Implementation
def Y(f):
    """Y Combinator: Y f = f (Y f)"""
    return (lambda x: f(lambda v: x(x)(v)))(lambda x: f(lambda v: x(x)(v)))

class HuffmanEncoder:
    def __init__(self):
        self.codes = {}
        self.frequencies = Counter()
    
    def analyze_frequencies(self, text):
        """Analyze operation frequencies for optimal encoding."""
        for char in text:
            self.frequencies[char] += 1
        return self.frequencies
'''
    }
    
    all_atoms = []
    
    for text_type, content in test_texts.items():
        print(f"\n📝 Analyzing {text_type} content...")
        atoms = analyzer.analyze_text_for_atoms(content, text_type)
        
        print(f"  Discovered {len(atoms)} meaning atoms")
        
        # Show top atoms
        print("  Top meaning atoms:")
        for atom in atoms[:5]:
            print(f"    {atom}")
        
        all_atoms.extend(atoms)
    
    # Overall analysis
    print(f"\n📊 Overall Analysis:")
    summary = analyzer.get_atomic_summary()
    print(f"  Total atoms discovered: {summary['total_atoms']}")
    print(f"  Average semantic weight: {summary['average_semantic_weight']:.3f}")
    print(f"  Atoms by type: {summary['atoms_by_type']}")
    
    print(f"\n🔗 Top overall meaning atoms:")
    for atom_info in summary['top_atoms']:
        print(f"    {atom_info['pattern'][:30]}... (weight: {atom_info['weight']:.3f}, freq: {atom_info['frequency']})")
    
    # Find relationships
    print(f"\n🔗 Atomic Relationships:")
    relationships = analyzer.find_atomic_relationships()
    for atom1, atom2, strength in relationships[:5]:
        print(f"    '{atom1.pattern[:20]}...' ↔ '{atom2.pattern[:20]}...' (strength: {strength:.3f})")
    
    # Export results
    export_data = {
        "atoms": [
            {
                "pattern": atom.pattern,
                "frequency": atom.frequency,
                "compression_ratio": atom.compression_ratio,
                "semantic_weight": atom.semantic_weight,
                "meaning_type": atom.meaning_type
            } for atom in all_atoms
        ],
        "summary": summary
    }
    
    with open('.out/meaning_atoms.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n💾 Results exported to meaning_atoms.json")
    print("\n🔥 Key Insight: Compression reveals the elementary atoms of meaning!")
    print("   • High-frequency patterns = structural atoms")
    print("   • Medium-compression patterns = semantic atoms") 
    print("   • Context diversity = meaning richness")
    print("   • Pattern relationships = conceptual networks")
    
    return analyzer

if __name__ == "__main__":
    analyzer = demo_meaning_atoms()
    
    print("\n🌀 Meaning Atoms: The fundamental particles of information!")
    print("   Just as physics has elementary particles,")
    print("   computation has elementary meaning atoms!")
    print("   Compression is the key to discovering them! 🔥")