#!/usr/bin/env python3
"""
Mathematical Conversation Parser
Extracts mathematical insights from DeepSeek conversations for TiddlyWiki integration
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class MathInsight:
    """Mathematical insight extracted from conversation"""
    title: str
    content: str
    category: str
    symbols: List[str]
    connections: List[str]
    line_start: int
    line_end: int

class MathConversationParser:
    """Parse DeepSeek mathematical conversations into structured insights"""
    
    def __init__(self):
        self.patterns = {
            'theorem': r'(?i)(theorem|lemma|proposition|corollary)[:.]',
            'proof': r'(?i)(proof|demonstration|verification)[:.]',
            'equation': r'[∫∑∏∂∇∞±≤≥≠≈∈∉⊆⊇∀∃]|\\[a-zA-Z]+',
            'millennium': r'(?i)(riemann|yang.mills|hodge|navier.stokes|birch|swinnerton)',
            'symbolic': r'(?i)(symbolic|differential|calculus|derivative|integral)',
            'fractal': r'(?i)(fractal|recursive|self.similar|iteration)',
            'libz': r'(?i)(libz|bytecode|translation|rosetta|vm|runtime)'
        }
    
    def parse_file(self, filepath: Path) -> List[MathInsight]:
        """Parse mathematical conversation file into insights"""
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        insights = []
        current_insight = []
        current_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Empty line indicates end of insight
            if not line:
                if current_insight:
                    insight = self._process_insight_block(
                        current_insight, current_start, i-1
                    )
                    if insight:
                        insights.append(insight)
                    current_insight = []
                current_start = i + 1
            else:
                current_insight.append(line)
        
        # Handle final insight
        if current_insight:
            insight = self._process_insight_block(
                current_insight, current_start, len(lines)-1
            )
            if insight:
                insights.append(insight)
        
        return insights
    
    def _process_insight_block(self, lines: List[str], start: int, end: int) -> MathInsight:
        """Process a block of lines into mathematical insight"""
        content = '\n'.join(lines)
        
        # Categorize insight
        category = self._categorize_content(content)
        
        # Extract mathematical symbols
        symbols = self._extract_symbols(content)
        
        # Generate title from first meaningful line
        title = self._generate_title(lines[0] if lines else f"Insight {start}")
        
        # Find connections to other mathematical concepts
        connections = self._find_connections(content)
        
        return MathInsight(
            title=title,
            content=content,
            category=category,
            symbols=symbols,
            connections=connections,
            line_start=start,
            line_end=end
        )
    
    def _categorize_content(self, content: str) -> str:
        """Categorize mathematical content"""
        content_lower = content.lower()
        
        if any(pattern in content_lower for pattern in ['theorem', 'lemma', 'proof']):
            return 'theorem'
        elif any(pattern in content_lower for pattern in ['riemann', 'yang', 'hodge']):
            return 'millennium'
        elif any(pattern in content_lower for pattern in ['symbolic', 'differential']):
            return 'symbolic'
        elif any(pattern in content_lower for pattern in ['fractal', 'recursive']):
            return 'fractal'
        elif any(pattern in content_lower for pattern in ['libz', 'bytecode']):
            return 'libz'
        else:
            return 'general'
    
    def _extract_symbols(self, content: str) -> List[str]:
        """Extract mathematical symbols from content"""
        symbols = []
        
        # Unicode mathematical symbols
        math_symbols = re.findall(r'[∫∑∏∂∇∞±≤≥≠≈∈∉⊆⊇∀∃]', content)
        symbols.extend(math_symbols)
        
        # LaTeX commands
        latex_commands = re.findall(r'\\[a-zA-Z]+', content)
        symbols.extend(latex_commands)
        
        return list(set(symbols))
    
    def _find_connections(self, content: str) -> List[str]:
        """Find connections to other mathematical concepts"""
        connections = []
        
        for pattern_name, pattern in self.patterns.items():
            if re.search(pattern, content):
                connections.append(pattern_name)
        
        return connections
    
    def _generate_title(self, first_line: str) -> str:
        """Generate insight title from first line"""
        # Clean up and truncate first line for title
        title = re.sub(r'[^\w\s]', '', first_line)
        title = ' '.join(title.split()[:8])  # Max 8 words
        return title or "Mathematical Insight"
    
    def to_tiddler(self, insight: MathInsight) -> str:
        """Convert mathematical insight to TiddlyWiki tiddler format"""
        tiddler = f"""title: {insight.title}
tags: mathematics {insight.category} {' '.join(insight.connections)}
type: text/vnd.tiddlywiki
created: {self._current_timestamp()}
modified: {self._current_timestamp()}

!! Mathematical Insight

{insight.content}

!! Symbols Used
{', '.join(insight.symbols) if insight.symbols else 'None'}

!! Categories
{insight.category}

!! Connections
{', '.join(insight.connections) if insight.connections else 'None'}

!! Source Location
Lines {insight.line_start}-{insight.line_end} in original conversation
"""
        return tiddler
    
    def _current_timestamp(self) -> str:
        """Generate TiddlyWiki timestamp"""
        from datetime import datetime
        return datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]

def main():
    """Process DeepSeek conversations into mathematical insights"""
    parser = MathConversationParser()
    
    # Process raw conversation files
    raw_dir = Path('conversations/deepseek/raw')
    processed_dir = Path('conversations/deepseek/processed')
    tiddlers_dir = Path('conversations/deepseek/tiddlers')
    
    # Create output directories
    processed_dir.mkdir(exist_ok=True)
    tiddlers_dir.mkdir(exist_ok=True)
    
    print("🧮 Mathematical Conversation Parser Ready")
    print(f"📁 Monitoring: {raw_dir}")
    print(f"📝 Output: {tiddlers_dir}")
    print("Ready to process DeepSeek mathematical revelations...")

if __name__ == '__main__':
    main()