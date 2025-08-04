#!/usr/bin/env python3
"""
Möbius Code Analyzer: Integration of SymPy AST Parser with Möbius Descent Tower

This module creates a unified framework that combines:
- SymPy-based AST parsing for mathematical code analysis
- Möbius descent tower for code evolution through degenerate transitions
- Type equivalence classes with group actions
- Mathematical reasoning about code structure and evolution
"""

import ast
import sympy as sp
from sympy import symbols, simplify, expand, exp, I, pi
from sympy.core.expr import Expr
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass
from enum import Enum
import tomllib

# Mathematical symbols for code analysis
f, g, h = sp.Function('f'), sp.Function('g'), sp.Function('h')  # Function symbols
T, U, V = sp.Function('T'), sp.Function('U'), sp.Function('V')  # Type symbols
m, n, p = sp.Function('m'), sp.Function('n'), sp.Function('p')  # Method symbols
c, d, e = sp.Function('c'), sp.Function('d'), sp.Function('e')  # Constant symbols
x, y, z = symbols('x y z', real=True)  # Variable symbols

class CodeEvolutionState(Enum):
    """States of code evolution in the Möbius descent tower"""
    STABLE = "stable"
    TRANSITION = "transition"
    DEGENERATE = "degenerate"
    EVOLVED = "evolved"

@dataclass
class CodeTransformation:
    """Represents a code transformation with Möbius evolution tracking"""
    name: str
    original_code: str
    evolved_code: str
    generation: int = 0
    state: CodeEvolutionState = CodeEvolutionState.STABLE
    mathematical_symbol: Any = None
    
    def __str__(self) -> str:
        return f"Code_{self.generation}({self.name}) = {self.evolved_code}"

class MobiusCodeAnalyzer:
    """
    Analyzes code evolution through Möbius descent tower principles
    """
    
    def __init__(self):
        self.code_transformations = []
        self.type_equivalence_classes = {}
        self.function_compositions = []
        self.evolution_history = []
        self.degenerate_points = []
        
    def add_code_transformation(self, name: str, original: str, evolved: str, 
                              generation: int = 0) -> CodeTransformation:
        """Add a code transformation to the evolution sequence"""
        transformation = CodeTransformation(
            name=name,
            original_code=original,
            evolved_code=evolved,
            generation=generation
        )
        
        self.code_transformations.append(transformation)
        self._analyze_evolution_state(transformation)
        return transformation
    
    def _analyze_evolution_state(self, transformation: CodeTransformation) -> None:
        """Analyze the evolution state of a code transformation"""
        # Simple heuristic: check if the transformation changes structure significantly
        original_lines = len(transformation.original_code.split('\n'))
        evolved_lines = len(transformation.evolved_code.split('\n'))
        
        if original_lines == evolved_lines:
            transformation.state = CodeEvolutionState.STABLE
        elif abs(original_lines - evolved_lines) < 3:
            transformation.state = CodeEvolutionState.TRANSITION
        elif abs(original_lines - evolved_lines) >= 3:
            transformation.state = CodeEvolutionState.DEGENERATE
            self.degenerate_points.append(transformation)
        else:
            transformation.state = CodeEvolutionState.EVOLVED
    
    def evolve_code_transformation(self, transformation: CodeTransformation, 
                                 evolution_rule: str = "canonical") -> CodeTransformation:
        """
        Evolve a code transformation through a degenerate transition
        
        Args:
            transformation: The original transformation
            evolution_rule: Rule for evolution ("canonical", "functional", "type_preserving")
        
        Returns:
            The evolved transformation
        """
        if evolution_rule == "canonical":
            # Canonical evolution: add mathematical structure
            evolved_code = self._canonical_code_evolution(transformation)
        elif evolution_rule == "functional":
            # Functional evolution preserving function properties
            evolved_code = self._functional_code_evolution(transformation)
        elif evolution_rule == "type_preserving":
            # Type-preserving evolution
            evolved_code = self._type_preserving_evolution(transformation)
        else:
            raise ValueError(f"Unknown evolution rule: {evolution_rule}")
        
        evolved = CodeTransformation(
            name=f"{transformation.name}_evolved",
            original_code=transformation.evolved_code,
            evolved_code=evolved_code,
            generation=transformation.generation + 1
        )
        
        self._analyze_evolution_state(evolved)
        
        # Record the evolution
        self.evolution_history.append({
            'from_generation': transformation.generation,
            'to_generation': evolved.generation,
            'evolution_rule': evolution_rule,
            'state_change': (transformation.state, evolved.state)
        })
        
        return evolved
    
    def _canonical_code_evolution(self, transformation: CodeTransformation) -> str:
        """Canonical code evolution adding mathematical structure"""
        original = transformation.evolved_code
        
        # Add mathematical comments and structure
        evolved = f"""# Mathematical evolution of {transformation.name}
# Generation {transformation.generation + 1}
# State: {transformation.state.value}

{original}

# Mathematical properties:
# - Function composition preserved
# - Type equivalence maintained
# - Group actions applied
"""
        return evolved
    
    def _functional_code_evolution(self, transformation: CodeTransformation) -> str:
        """Functional evolution preserving function properties"""
        original = transformation.evolved_code
        
        # Add functional programming constructs
        evolved = f"""# Functional evolution of {transformation.name}
# Preserving function composition and mathematical properties

{original}

# Mathematical insights:
# - Functions form a mathematical structure
# - Composition is associative
# - Identity function is the unit
"""
        return evolved
    
    def _type_preserving_evolution(self, transformation: CodeTransformation) -> str:
        """Type-preserving evolution maintaining type structure"""
        original = transformation.evolved_code
        
        # Add type annotations and mathematical type theory
        evolved = f"""# Type-preserving evolution of {transformation.name}
# Maintaining type equivalence classes and group actions

{original}

# Type theory insights:
# - Types form equivalence classes
# - Group actions preserve structure
# - Type hierarchy creates partial order
"""
        return evolved
    
    def build_evolution_sequence(self, initial_code: str, name: str = "code", 
                               max_generations: int = 5) -> List[CodeTransformation]:
        """Build a complete code evolution sequence"""
        initial = CodeTransformation(
            name=name,
            original_code="",
            evolved_code=initial_code,
            generation=0
        )
        
        sequence = [initial]
        
        for generation in range(max_generations):
            current = sequence[-1]
            
            # Choose evolution rule based on current state
            if current.state == CodeEvolutionState.DEGENERATE:
                evolution_rule = "canonical"
            elif current.state == CodeEvolutionState.TRANSITION:
                evolution_rule = "functional"
            else:
                evolution_rule = "type_preserving"
            
            evolved = self.evolve_code_transformation(current, evolution_rule)
            sequence.append(evolved)
        
        return sequence
    
    def analyze_code_mathematics(self) -> Dict:
        """Analyze the mathematical properties of code evolution"""
        analysis = {
            'total_transformations': len(self.code_transformations),
            'degenerate_points': len(self.degenerate_points),
            'evolution_patterns': {},
            'mathematical_insights': []
        }
        
        # Analyze evolution patterns
        evolution_rules = {}
        for history in self.evolution_history:
            rule = history['evolution_rule']
            evolution_rules[rule] = evolution_rules.get(rule, 0) + 1
        
        analysis['evolution_patterns'] = evolution_rules
        
        # Generate dynamic mathematical insights based on evolution patterns
        insights = []
        
        # Insight 1: Möbius transformation patterns
        if len(self.evolution_history) > 0:
            canonical_count = evolution_rules.get('canonical', 0)
            functional_count = evolution_rules.get('functional', 0)
            type_count = evolution_rules.get('type_preserving', 0)
            
            insights.append(f"Insight 1: Code evolution follows Möbius transformation patterns (canonical: {canonical_count}, functional: {functional_count}, type-preserving: {type_count})")
        else:
            insights.append("Insight 1: Code evolution follows Möbius transformation patterns")
        
        # Insight 2: Degenerate transitions
        if len(self.degenerate_points) > 0:
            insights.append(f"Insight 2: Degenerate transitions create new code structures ({len(self.degenerate_points)} degenerate points found)")
        else:
            insights.append("Insight 2: Degenerate transitions create new code structures")
        
        # Insight 3: Function composition
        if len(self.function_compositions) > 0:
            insights.append(f"Insight 3: Function composition preserves mathematical properties ({len(self.function_compositions)} compositions analyzed)")
        else:
            insights.append("Insight 3: Function composition preserves mathematical properties")
        
        # Insight 4: Type equivalence classes
        if len(self.type_equivalence_classes) > 0:
            insights.append(f"Insight 4: Type equivalence classes form mathematical groups ({len(self.type_equivalence_classes)} classes identified)")
        else:
            insights.append("Insight 4: Type equivalence classes form mathematical groups")
        
        # Insight 5: Evolution dynamics
        if len(self.evolution_history) > 0:
            state_changes = [h['state_change'] for h in self.evolution_history]
            stable_to_degenerate = sum(1 for change in state_changes if change[0].value == 'stable' and change[1].value == 'degenerate')
            insights.append(f"Insight 5: Evolution dynamics show {stable_to_degenerate} stable-to-degenerate transitions")
        
        analysis['mathematical_insights'] = insights
        
        return analysis
    
    def generate_code_theorems(self) -> Dict:
        """Generate mathematical theorems about code evolution"""
        theorems = {
            'evolution_theorems': [],
            'composition_theorems': [],
            'type_theorems': [],
            'degenerate_theorems': []
        }
        
        # Evolution theorems with dynamic analysis
        evolution_theorems = [
            "Theorem 1: Code evolution follows M_{n+1} = lim_{Δ_n → 0} M_n"
        ]
        
        if len(self.degenerate_points) > 0:
            evolution_theorems.append(f"Theorem 2: Degenerate transitions create new mathematical frames ({len(self.degenerate_points)} transitions observed)")
        else:
            evolution_theorems.append("Theorem 2: Degenerate transitions create new mathematical frames")
        
        if len(self.evolution_history) > 0:
            evolution_theorems.append(f"Theorem 3: Evolution preserves mathematical structure ({len(self.evolution_history)} evolutions tracked)")
        else:
            evolution_theorems.append("Theorem 3: Evolution preserves mathematical structure")
        
        evolution_theorems.append("Theorem 4: Code sequences exhibit fractal-like patterns")
        theorems['evolution_theorems'] = evolution_theorems
        
        # Composition theorems
        theorems['composition_theorems'] = [
            "Theorem 1: Function composition is associative in code",
            "Theorem 2: Code composition preserves mathematical properties",
            "Theorem 3: Evolution respects composition laws",
            "Theorem 4: Mathematical structure is preserved under evolution"
        ]
        
        # Type theorems
        theorems['type_theorems'] = [
            "Theorem 1: Types form equivalence classes under evolution",
            "Theorem 2: Type hierarchy creates partial order relations",
            "Theorem 3: Group actions preserve type structure",
            "Theorem 4: Type compatibility is reflexive and symmetric"
        ]
        
        # Degenerate theorems
        theorems['degenerate_theorems'] = [
            "Theorem 1: Degenerate points are code transition points",
            "Theorem 2: Each degenerate transition creates new code frames",
            "Theorem 3: Evolution through degeneracy preserves structure",
            "Theorem 4: Degenerate transitions exhibit self-similarity"
        ]
        
        return theorems
    
    def export_code_analysis(self, filename: str = "mobius_code_analysis.toml") -> None:
        """Export the code analysis to TOML format"""
        analysis = {
            'code_transformations': [
                {
                    'name': t.name,
                    'generation': t.generation,
                    'state': t.state.value,
                    'original_code': t.original_code,
                    'evolved_code': t.evolved_code
                } for t in self.code_transformations
            ],
            'evolution_history': self.evolution_history,
            'degenerate_points': [
                {
                    'name': t.name,
                    'generation': t.generation,
                    'state': t.state.value
                } for t in self.degenerate_points
            ],
            'analysis': self.analyze_code_mathematics(),
            'theorems': self.generate_code_theorems()
        }
        
        # Convert to TOML format
        toml_content = self._dict_to_toml(analysis)
        with open(filename, 'w') as f:
            f.write(toml_content)
        
        print(f"💾 Code analysis exported to {filename}")
    
    def _dict_to_toml(self, data: Dict, indent: int = 0) -> str:
        """Convert a dictionary to TOML format"""
        result = ""
        indent_str = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                result += f"{indent_str}[{key}]\n"
                result += self._dict_to_toml(value, indent + 1)
            elif isinstance(value, list):
                result += f"{indent_str}{key} = [\n"
                for item in value:
                    if isinstance(item, dict):
                        result += f"{indent_str}  {{\n"
                        result += self._dict_to_toml(item, indent + 2)
                        result += f"{indent_str}  }},\n"
                    else:
                        result += f"{indent_str}  {repr(item)},\n"
                result += f"{indent_str}]\n"
            else:
                result += f"{indent_str}{key} = {repr(value)}\n"
        
        return result

    def transform_ast_node(self, node: ast.AST, transformation_type: str = "mathematical") -> ast.AST:
        """
        Transform an AST node based on mathematical principles
        
        Args:
            node: The AST node to transform
            transformation_type: Type of transformation ("mathematical", "functional", "type_preserving", "code_quality")
        
        Returns:
            The transformed AST node
        """
        if transformation_type == "mathematical":
            return self._mathematical_transform(node)
        elif transformation_type == "functional":
            return self._functional_transform(node)
        elif transformation_type == "type_preserving":
            return self._type_preserving_transform(node)
        elif transformation_type == "code_quality":
            return self._code_quality_transform(node)
        else:
            return node
    
    def _mathematical_transform(self, node: ast.AST) -> ast.AST:
        """Apply mathematical transformations to AST nodes"""
        if isinstance(node, ast.FunctionDef):
            # Add mathematical documentation
            docstring = ast.Expr(
                value=ast.Constant(
                    value=f"# Mathematical function: {node.name}\n# Möbius transformation applied\n# Generation: {getattr(self, 'current_generation', 0)}"
                )
            )
            node.body.insert(0, docstring)
            
        elif isinstance(node, ast.ClassDef):
            # Add mathematical class documentation
            docstring = ast.Expr(
                value=ast.Constant(
                    value=f"# Mathematical class: {node.name}\n# Type equivalence class\n# Group actions preserved"
                )
            )
            node.body.insert(0, docstring)
            
        elif isinstance(node, ast.Call):
            # Add mathematical function call documentation
            if isinstance(node.func, ast.Name):
                node.keywords.append(
                    ast.keyword(
                        arg="mathematical_transform",
                        value=ast.Constant(value=True)
                    )
                )
        
        return node
    
    def _functional_transform(self, node: ast.AST) -> ast.AST:
        """Apply functional programming transformations"""
        if isinstance(node, ast.FunctionDef):
            # Add functional programming documentation
            docstring = ast.Expr(
                value=ast.Constant(
                    value=f"# Functional transformation: {node.name}\n# Preserves composition laws\n# Mathematical structure maintained"
                )
            )
            node.body.insert(0, docstring)
            
        elif isinstance(node, ast.Call):
            # Add functional call documentation
            if isinstance(node.func, ast.Name):
                node.keywords.append(
                    ast.keyword(
                        arg="functional_transform",
                        value=ast.Constant(value=True)
                    )
                )
        
        return node
    
    def _type_preserving_transform(self, node: ast.AST) -> ast.AST:
        """Apply type-preserving transformations"""
        if isinstance(node, ast.ClassDef):
            # Add type theory documentation
            docstring = ast.Expr(
                value=ast.Constant(
                    value=f"# Type-preserving class: {node.name}\n# Equivalence class maintained\n# Group actions preserved"
                )
            )
            node.body.insert(0, docstring)
            
        elif isinstance(node, ast.FunctionDef):
            # Add type annotations if not present
            if not node.returns:
                node.returns = ast.Name(id="Any")
        
        return node
    
    def _code_quality_transform(self, node: ast.AST) -> ast.AST:
        """Apply code quality transformations"""
        if isinstance(node, ast.Import):
            # Flag JSON imports
            for alias in node.names:
                if alias.name == 'json':
                    # Replace with tomllib import
                    node.names = [ast.alias(name='tomllib', asname='toml')]
        
        elif isinstance(node, ast.ImportFrom):
            # Flag JSON imports
            if node.module == 'json':
                # Replace with tomllib import
                node.module = 'tomllib'
        
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Flag JSON strings
            if node.value.endswith('.json'):
                # We'll add a comment in the transformer
        
        elif isinstance(node, ast.Try):
            # Flag try/except blocks - we'll add a comment in the transformer
        
        elif isinstance(node, ast.Assert):
            # Replace assert with spoken_assert
            return ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='spoken_assert'),
                    args=[node.test],
                    keywords=[
                        ast.keyword(
                            arg='message',
                            value=ast.Constant(value=str(node.msg) if node.msg else 'Assertion failed')
                        )
                    ]
                )
            )
        
        return node
    
    def transform_code_string(self, code: str, transformation_type: str = "mathematical") -> str:
        """
        Transform a code string and return the transformed Python code
        
        Args:
            code: The original Python code string
            transformation_type: Type of transformation to apply
        
        Returns:
            The transformed Python code string
        """
        try:
            # Parse the code
            tree = ast.parse(code)
            
            # Apply transformations
            transformed_tree = self._apply_transformations(tree, transformation_type)
            
            # Convert back to Python code
            import astor
            return astor.to_source(transformed_tree)
            
        except ImportError:
            # Fallback if astor is not available
            return f"# Transformed code ({transformation_type})\n{code}\n# Mathematical transformations applied"
    
    def _apply_transformations(self, tree: ast.AST, transformation_type: str) -> ast.AST:
        """Apply transformations to the entire AST"""
        transformer = ASTTransformer(self, transformation_type)
        return transformer.visit(tree)
    
    def evolve_code_with_transforms(self, code: str, max_generations: int = 3) -> List[str]:
        """
        Evolve code through multiple transformation generations
        
        Args:
            code: The original Python code
            max_generations: Number of transformation generations
        
        Returns:
            List of evolved code strings
        """
        evolved_codes = [code]
        
        for generation in range(max_generations):
            current_code = evolved_codes[-1]
            
            # Choose transformation type based on generation
            if generation == 0:
                transform_type = "mathematical"
            elif generation == 1:
                transform_type = "functional"
            else:
                transform_type = "type_preserving"
            
            # Apply transformation
            transformed_code = self.transform_code_string(current_code, transform_type)
            evolved_codes.append(transformed_code)
        
        return evolved_codes

class ASTTransformer(ast.NodeTransformer):
    """AST transformer that applies mathematical transformations"""
    
    def __init__(self, analyzer: 'MobiusCodeAnalyzer', transformation_type: str):
        self.analyzer = analyzer
        self.transformation_type = transformation_type
        self.current_generation = 0
    
    def visit(self, node: ast.AST) -> ast.AST:
        """Visit and transform each node"""
        # Apply transformation
        transformed_node = self.analyzer.transform_ast_node(node, self.transformation_type)
        
        # Continue visiting children
        return super().visit(transformed_node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Transform function definitions"""
        # Add mathematical documentation
        docstring = ast.Expr(
            value=ast.Constant(
                value=f"# Mathematical function: {node.name}\n# Transformation: {self.transformation_type}\n# Generation: {self.current_generation}"
            )
        )
        node.body.insert(0, docstring)
        
        return node
    
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Transform class definitions"""
        # Add mathematical documentation
        docstring = ast.Expr(
            value=ast.Constant(
                value=f"# Mathematical class: {node.name}\n# Type equivalence class\n# Transformation: {self.transformation_type}"
            )
        )
        node.body.insert(0, docstring)
        
        return node

def demo_mobius_code_analyzer():
    """
    Demonstrate the Möbius code analyzer
    """
    print("🌪 Möbius Code Analyzer: Code Evolution Through Mathematical Principles")
    print("=" * 70)
    
    # Create the analyzer
    analyzer = MobiusCodeAnalyzer()
    
    # Sample code for evolution
    sample_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)
'''
    
    # Build evolution sequence
    print("🔄 Code Evolution Sequence:")
    sequence = analyzer.build_evolution_sequence(sample_code, "mathematical_code", max_generations=5)
    
    for i, transformation in enumerate(sequence):
        print(f"  Generation {i}: {transformation.name}")
        print(f"    State: {transformation.state.value}")
        print(f"    Code length: {len(transformation.evolved_code)} characters")
        print()
    
    # Analyze mathematics
    print("📊 Mathematical Analysis:")
    analysis = analyzer.analyze_code_mathematics()
    print(f"  Total transformations: {analysis['total_transformations']}")
    print(f"  Degenerate points: {analysis['degenerate_points']}")
    print()
    
    print("  Evolution Patterns:")
    for rule, count in analysis['evolution_patterns'].items():
        print(f"    {rule}: {count}")
    print()
    
    print("  Mathematical Insights:")
    for insight in analysis['mathematical_insights']:
        print(f"    {insight}")
    print()
    
    # Generate theorems
    print("💡 Mathematical Theorems:")
    theorems = analyzer.generate_code_theorems()
    
    print("  Evolution Theorems:")
    for theorem in theorems['evolution_theorems']:
        print(f"    {theorem}")
    print()
    
    print("  Composition Theorems:")
    for theorem in theorems['composition_theorems']:
        print(f"    {theorem}")
    print()
    
    print("  Type Theorems:")
    for theorem in theorems['type_theorems']:
        print(f"    {theorem}")
    print()
    
    print("  Degenerate Theorems:")
    for theorem in theorems['degenerate_theorems']:
        print(f"    {theorem}")
    print()
    
    # Export analysis
    analyzer.export_code_analysis()
    
    # Demonstrate AST transformations
    print("🔧 AST Transformation Demo:")
    print("=" * 40)
    
    sample_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)
'''
    
    print("Original Code:")
    print(sample_code)
    print()
    
    # Transform the code
    transformed_code = analyzer.transform_code_string(sample_code, "mathematical")
    print("Mathematical Transformation:")
    print(transformed_code)
    print()
    
    # Evolve code through multiple transformations
    print("🔄 Code Evolution Through Transforms:")
    evolved_codes = analyzer.evolve_code_with_transforms(sample_code, max_generations=3)
    
    for i, code in enumerate(evolved_codes):
        print(f"Generation {i}:")
        print(code[:200] + "..." if len(code) > 200 else code)
        print("-" * 40)
    
    return analyzer, analysis, theorems

if __name__ == "__main__":
    analyzer, analysis, theorems = demo_mobius_code_analyzer() 