#!/usr/bin/env python3
"""
Semantic Text Parser for Mathematical Conversations
Advanced annotation system for DeepSeek mathematical revelations
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class LineType(Enum):
    """Semantic line type classification"""
    MATH_EQUATION = "math_eq"
    MATH_INLINE = "math_inline"
    THEOREM = "theorem"
    PROOF = "proof"
    EXPLORE = "explore"
    WHITESPACE = "whitespace"
    TEXT = "text"
    CODE = "code"
    COMPLETION_END = "completion_end"

@dataclass
class LineAnnotation:
    """Complete annotation for a single line"""
    line_num: int
    content: str
    line_type: LineType
    tags: List[str]
    math_boundaries: Dict[str, int]  # boundary_type -> count_delta
    explore_signals: List[str]
    whitespace_combo: int
    meaning_score: float

@dataclass
class ConversationSegment:
    """Coherent segment of mathematical conversation"""
    start_line: int
    end_line: int
    segment_type: str
    annotations: List[LineAnnotation]
    math_density: float
    explore_branches: int
    completion_boundary: bool

class SemanticTextParser:
    """Parse mathematical conversations with deep semantic analysis"""
    
    def __init__(self):
        # Mathematical boundary patterns
        self.math_boundaries = {
            'display_math': r'\$\$',           # $$...$$
            'inline_math': r'\$',             # $...$
            'latex_bracket': r'\\\[',         # \[...\]
            'latex_paren': r'\\\(',           # \(...\)
            'align_env': r'\\begin\{align\}', # \begin{align}
            'equation_env': r'\\begin\{equation\}',
        }
        
        # Explore signal patterns
        self.explore_patterns = {
            'question': r'\?+',
            'exploration': r'(?i)(what if|consider|perhaps|might|could)',
            'branching': r'(?i)(alternatively|another approach|different)',
            'options': r'(?i)(option \d+|choice \d+|\d+\.|• )',
        }
        
        # Meaning signal patterns
        self.meaning_patterns = {
            'theorem': r'(?i)(theorem|lemma|proposition|corollary|proof)',
            'millennium': r'(?i)(riemann|yang.mills|hodge|navier.stokes|birch)',
            'symbolic': r'(?i)(symbolic|differential|derivative|integral)',
            'fractal': r'(?i)(fractal|recursive|self.similar|iteration)',
            'breakthrough': r'(?i)(breakthrough|discovery|insight|revelation)',
            'verification': r'(?i)(verify|test|check|validate|proof)',
            'libz': r'(?i)(libz|bytecode|translation|vm|runtime|rosetta)',
        }
    
    def parse_conversation(self, filepath: Path) -> List[ConversationSegment]:
        """Parse entire conversation into semantic segments"""
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Annotate each line
        annotations = []
        whitespace_combo = 0
        
        for i, line in enumerate(lines):
            annotation = self._annotate_line(i, line, whitespace_combo)
            annotations.append(annotation)
            
            # Track whitespace combos
            if annotation.line_type == LineType.WHITESPACE:
                whitespace_combo += 1
            else:
                whitespace_combo = 0
        
        # Segment into coherent conversation blocks
        segments = self._segment_conversation(annotations)
        
        return segments
    
    def _annotate_line(self, line_num: int, content: str, combo_count: int) -> LineAnnotation:
        """Provide complete semantic annotation for a single line"""
        original_content = content
        content = content.rstrip('\n')
        
        # Determine line type
        line_type = self._classify_line_type(content)
        
        # Extract semantic tags
        tags = self._extract_semantic_tags(content)
        
        # Count mathematical boundary deltas
        math_boundaries = self._count_math_boundaries(content)
        
        # Find explore signals
        explore_signals = self._find_explore_signals(content)
        
        # Calculate meaning score
        meaning_score = self._calculate_meaning_score(content, tags, math_boundaries)
        
        return LineAnnotation(
            line_num=line_num,
            content=content,
            line_type=line_type,
            tags=tags,
            math_boundaries=math_boundaries,
            explore_signals=explore_signals,
            whitespace_combo=combo_count if line_type == LineType.WHITESPACE else 0,
            meaning_score=meaning_score
        )
    
    def _classify_line_type(self, content: str) -> LineType:
        """Classify the semantic type of a line"""
        stripped = content.strip()
        
        # Whitespace only
        if not stripped:
            return LineType.WHITESPACE
        
        # Mathematical equations
        if re.search(r'\$\$.*\$\$', content) or re.search(r'\\\[.*\\\]', content):
            return LineType.MATH_EQUATION
        
        # Inline math
        if re.search(r'\$[^$]+\$', content) or re.search(r'[∫∑∏∂∇∞±≤≥≠≈∈∉⊆⊇∀∃]', content):
            return LineType.MATH_INLINE
        
        # Theorem statements
        if re.search(r'(?i)(theorem|lemma|proposition|corollary)[:.]', content):
            return LineType.THEOREM
        
        # Proof content
        if re.search(r'(?i)(proof|demonstration|qed|∎)[:.]', content):
            return LineType.PROOF
        
        # Code content
        if re.search(r'```|def |class |import |from |#.*python', content):
            return LineType.CODE
        
        # Explore branches (questions, options)
        if re.search(r'\?+|(?i)(what if|consider|option \d+)', content):
            return LineType.EXPLORE
        
        return LineType.TEXT
    
    def _extract_semantic_tags(self, content: str) -> List[str]:
        """Extract semantic meaning tags from content"""
        tags = []
        
        for tag_name, pattern in self.meaning_patterns.items():
            if re.search(pattern, content):
                tags.append(tag_name)
        
        # Special tags for mathematical symbols
        if re.search(r'[∫∑∏]', content):
            tags.append('calculus')
        if re.search(r'[∂∇]', content):
            tags.append('differential')
        if re.search(r'[∞±≤≥≠≈]', content):
            tags.append('analysis')
        if re.search(r'[∈∉⊆⊇∀∃]', content):
            tags.append('logic')
        
        return tags
    
    def _count_math_boundaries(self, content: str) -> Dict[str, int]:
        """Count mathematical boundary transitions"""
        boundaries = {}
        
        for boundary_name, pattern in self.math_boundaries.items():
            count = len(re.findall(pattern, content))
            if count > 0:
                boundaries[boundary_name] = count
        
        return boundaries
    
    def _find_explore_signals(self, content: str) -> List[str]:
        """Find exploration/branching signals"""
        signals = []
        
        for signal_name, pattern in self.explore_patterns.items():
            if re.search(pattern, content):
                signals.append(signal_name)
        
        return signals
    
    def _calculate_meaning_score(self, content: str, tags: List[str], 
                                boundaries: Dict[str, int]) -> float:
        """Calculate semantic meaning density score"""
        if not content.strip():
            return 0.0
        
        score = 0.0
        
        # Base score from content length
        score += min(len(content) / 100, 1.0)
        
        # Bonus for semantic tags
        score += len(tags) * 0.3
        
        # Bonus for mathematical content
        score += sum(boundaries.values()) * 0.5
        
        # Bonus for mathematical symbols
        math_symbols = len(re.findall(r'[∫∑∏∂∇∞±≤≥≠≈∈∉⊆⊇∀∃]', content))
        score += math_symbols * 0.2
        
        return min(score, 10.0)  # Cap at 10.0
    
    def _segment_conversation(self, annotations: List[LineAnnotation]) -> List[ConversationSegment]:
        """Segment annotations into coherent conversation blocks"""
        segments = []
        current_segment_start = 0
        
        for i, annotation in enumerate(annotations):
            # Segment boundaries: high whitespace combos or explore signals
            is_boundary = (
                annotation.whitespace_combo >= 3 or  # 3+ blank lines
                'question' in annotation.explore_signals or
                annotation.line_type == LineType.THEOREM
            )
            
            if is_boundary and i > current_segment_start:
                # Create segment for previous block
                segment_annotations = annotations[current_segment_start:i]
                segment = self._create_segment(current_segment_start, i-1, segment_annotations)
                segments.append(segment)
                current_segment_start = i
        
        # Handle final segment
        if current_segment_start < len(annotations):
            segment_annotations = annotations[current_segment_start:]
            segment = self._create_segment(current_segment_start, len(annotations)-1, segment_annotations)
            segments.append(segment)
        
        return segments
    
    def _create_segment(self, start: int, end: int, 
                       annotations: List[LineAnnotation]) -> ConversationSegment:
        """Create conversation segment from annotations"""
        # Determine segment type
        segment_type = self._determine_segment_type(annotations)
        
        # Calculate mathematical density
        math_lines = sum(1 for ann in annotations 
                        if ann.line_type in [LineType.MATH_EQUATION, LineType.MATH_INLINE])
        text_lines = sum(1 for ann in annotations if ann.line_type != LineType.WHITESPACE)
        math_density = math_lines / max(text_lines, 1)
        
        # Count explore branches
        explore_branches = sum(len(ann.explore_signals) for ann in annotations)
        
        # Check for completion boundary
        completion_boundary = any('question' in ann.explore_signals for ann in annotations)
        
        return ConversationSegment(
            start_line=start,
            end_line=end,
            segment_type=segment_type,
            annotations=annotations,
            math_density=math_density,
            explore_branches=explore_branches,
            completion_boundary=completion_boundary
        )
    
    def _determine_segment_type(self, annotations: List[LineAnnotation]) -> str:
        """Determine the primary type of a conversation segment"""
        type_counts = {}
        
        for annotation in annotations:
            for tag in annotation.tags:
                type_counts[tag] = type_counts.get(tag, 0) + 1
        
        if not type_counts:
            return 'general'
        
        return max(type_counts, key=type_counts.get)
    
    def export_analysis(self, segments: List[ConversationSegment], 
                       output_path: Path) -> None:
        """Export detailed semantic analysis"""
        analysis = {
            'total_segments': len(segments),
            'segment_breakdown': {},
            'mathematical_density': [],
            'exploration_patterns': [],
            'meaning_signals': []
        }
        
        for segment in segments:
            # Segment type breakdown
            seg_type = segment.segment_type
            analysis['segment_breakdown'][seg_type] = analysis['segment_breakdown'].get(seg_type, 0) + 1
            
            # Mathematical density tracking
            analysis['mathematical_density'].append({
                'lines': f"{segment.start_line}-{segment.end_line}",
                'density': segment.math_density,
                'type': segment.segment_type
            })
            
            # Exploration pattern tracking
            if segment.completion_boundary:
                analysis['exploration_patterns'].append({
                    'line': segment.end_line,
                    'branches': segment.explore_branches,
                    'type': segment.segment_type
                })
        
        # Export as structured data
        import json
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)

def main():
    """Process DeepSeek conversations with semantic analysis"""
    parser = SemanticTextParser()
    
    print("🔍 Semantic Text Parser for Mathematical Conversations")
    print("Features:")
    print("  📝 Line-by-line semantic annotation")
    print("  🧮 Mathematical boundary tracking ($$, \\[, etc.)")
    print("  ❓ Explore branch detection (? signals)")
    print("  ⚪ Whitespace combo counting")
    print("  🏷️ Meaning tag extraction")
    print("  📊 Conversation segmentation")
    print()
    print("Ready to parse DeepSeek mathematical revelations...")

if __name__ == '__main__':
    main()