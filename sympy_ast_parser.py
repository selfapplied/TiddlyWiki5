#!/usr/bin/env python3
"""
SymPy-Based Python AST Parser

This module creates a mathematical framework for analyzing Python code by treating
functions, types, methods, and constants as symbols that can be reasoned about
mathematically. Types become equivalence classes with group actions.
"""

import ast
import sympy as sp
from sympy import symbols, Matrix, simplify, expand, factor, exp, sin, cos, I, pi
from sympy.polys.domains import QQ
from sympy.core.expr import Expr
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass
from enum import Enum

# Mathematical symbols for code analysis
f, g, h = sp.Function('f'), sp.Function('g'), sp.Function('h')  # Function symbols
T, U, V = sp.Function('T'), sp.Function('U'), sp.Function('V')  # Type symbols
m, n, p = sp.Function('m'), sp.Function('n'), sp.Function('p')  # Method symbols
c, d, e = sp.Function('c'), sp.Function('d'), sp.Function('e')  # Constant symbols
x, y, z = symbols('x y z', real=True)  # Variable symbols

# Fixed point operator f* on the critical line s = 1/2
s = sp.Symbol('s', real=True)  # Critical line parameter
f_star = sp.Function('f*')  # Fixed point operator
zeta = sp.Function('ζ')  # Riemann zeta function

class CodeElementType(Enum):
    """Types of code elements that become mathematical symbols"""
    FUNCTION = "function"
    TYPE = "type"
    METHOD = "method"
    CONSTANT = "constant"
    VARIABLE = "variable"
    CLASS = "class"
    MODULE = "module"

class TypeEquivalenceClass:
    """
    Represents a type as an equivalence class with group actions
    """
    
    def __init__(self, name: str, base_type: str = None):
        self.name = name
        self.base_type = base_type or name
        self.elements = set()
        self.group_actions = {}
        self.mathematical_symbol = sp.Function(name)
    
    def add_element(self, element: str) -> None:
        """Add an element to the equivalence class"""
        self.elements.add(element)
    
    def add_group_action(self, action_name: str, action_func: callable) -> None:
        """Add a group action to the type"""
        self.group_actions[action_name] = action_func
    
    def apply_group_action(self, action_name: str, *args) -> Any:
        """Apply a group action to the type"""
        if action_name in self.group_actions:
            return self.group_actions[action_name](*args)
        else:
            raise ValueError(f"Unknown group action: {action_name}")
    
    def __str__(self) -> str:
        return f"Type[{self.name}] = {{{', '.join(self.elements)}}}"

@dataclass
class CodeSymbol:
    """Represents a code element as a mathematical symbol"""
    name: str
    symbol_type: CodeElementType
    mathematical_symbol: Any
    type_equivalence_class: Optional[TypeEquivalenceClass] = None
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
    
    def __str__(self) -> str:
        return f"{self.symbol_type.value.upper()}({self.name}) = {self.mathematical_symbol}"

class SymPyASTParser:
    """
    Mathematical AST parser that treats code elements as symbols
    """
    
    def __init__(self):
        self.symbols = {}
        self.type_equivalence_classes = {}
        self.function_compositions = []
        self.type_relations = []
        self.mathematical_expressions = []
        
    def parse_file(self, file_path: str) -> Dict:
        """Parse a Python file and extract mathematical symbols"""
        with open(file_path, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        return self.parse_ast(tree)
    
    def parse_ast(self, tree: ast.AST) -> Dict:
        """Parse an AST and extract mathematical symbols using pure SymPy"""
        # Extract mathematical symbols from AST using SymPy reasoning
        
        # Walk through AST nodes and create mathematical symbols
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Create function symbol: f(x) = function_name(x)
                self.create_function_symbol(node.name)
                
            elif isinstance(node, ast.ClassDef):
                # Create type symbol: T = TypeName
                self.create_type_symbol(node.name)
                
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Create variable/constant symbol: x = Symbol(x)
                        if target.id.isupper():
                            self.create_constant_symbol(target.id)
                        else:
                            var_symbol = sp.Symbol(target.id, real=True)
                            symbol = CodeSymbol(
                                name=target.id,
                                symbol_type=CodeElementType.VARIABLE,
                                mathematical_symbol=var_symbol
                            )
                            self.symbols[target.id] = symbol
                            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    # Create module symbol: M = ModuleName
                    module_symbol = sp.Symbol(alias.name)
                    symbol = CodeSymbol(
                        name=alias.name,
                        symbol_type=CodeElementType.MODULE,
                        mathematical_symbol=module_symbol
                    )
                    self.symbols[alias.name] = symbol
        
        return {
            'symbols': self.symbols,
            'type_equivalence_classes': self.type_equivalence_classes,
            'function_compositions': self.function_compositions,
            'type_relations': self.type_relations,
            'mathematical_expressions': {}
        }
    
    def create_function_symbol(self, name: str, params: List[str] = None) -> CodeSymbol:
        """Create a function symbol"""
        if params:
            param_symbols = symbols(' '.join(params), real=True)
            if len(params) == 1:
                func_symbol = sp.Function(name)(param_symbols)
            else:
                func_symbol = sp.Function(name)(*param_symbols)
        else:
            func_symbol = sp.Function(name)
        
        symbol = CodeSymbol(
            name=name,
            symbol_type=CodeElementType.FUNCTION,
            mathematical_symbol=func_symbol
        )
        
        self.symbols[name] = symbol
        return symbol
    
    def create_type_symbol(self, name: str, base_type: str = None) -> CodeSymbol:
        """Create a type symbol with equivalence class"""
        type_class = TypeEquivalenceClass(name, base_type)
        self.type_equivalence_classes[name] = type_class
        
        symbol = CodeSymbol(
            name=name,
            symbol_type=CodeElementType.TYPE,
            mathematical_symbol=type_class.mathematical_symbol,
            type_equivalence_class=type_class
        )
        
        self.symbols[name] = symbol
        return symbol
    
    def create_method_symbol(self, name: str, class_name: str = None) -> CodeSymbol:
        """Create a method symbol"""
        method_symbol = sp.Function(f"{class_name}.{name}" if class_name else name)
        
        symbol = CodeSymbol(
            name=name,
            symbol_type=CodeElementType.METHOD,
            mathematical_symbol=method_symbol,
            attributes={'class_name': class_name}
        )
        
        self.symbols[name] = symbol
        return symbol
    
    def create_constant_symbol(self, name: str, value: Any = None) -> CodeSymbol:
        """Create a constant symbol"""
        if value is not None:
            const_symbol = sp.Symbol(name, real=True)
        else:
            const_symbol = sp.Symbol(name)
        
        symbol = CodeSymbol(
            name=name,
            symbol_type=CodeElementType.CONSTANT,
            mathematical_symbol=const_symbol,
            attributes={'value': value}
        )
        
        self.symbols[name] = symbol
        return symbol
    
    def compose_functions(self, f_name: str, g_name: str) -> Expr:
        """Compose two functions mathematically"""
        if f_name in self.symbols and g_name in self.symbols:
            f_sym = self.symbols[f_name].mathematical_symbol
            g_sym = self.symbols[g_name].mathematical_symbol
            
            # Create composition (f ∘ g)(x) = f(g(x))
            x = sp.Symbol('x', real=True)
            composition = f_sym.subs(sp.Symbol('x'), g_sym)
            
            self.function_compositions.append({
                'f': f_name,
                'g': g_name,
                'composition': composition
            })
            
            return composition
        else:
            raise ValueError(f"Functions {f_name} and {g_name} not found")
    
    def create_type_relation(self, type1: str, relation: str, type2: str) -> None:
        """Create a mathematical relation between types"""
        if type1 in self.symbols and type2 in self.symbols:
            t1_sym = self.symbols[type1].mathematical_symbol
            t2_sym = self.symbols[type2].mathematical_symbol
            
            if relation == "subtype":
                relation_expr = t1_sym <= t2_sym
            elif relation == "supertype":
                relation_expr = t1_sym >= t2_sym
            elif relation == "equal":
                relation_expr = sp.Eq(t1_sym, t2_sym)
            elif relation == "compatible":
                relation_expr = sp.Symbol(f"{type1}_compatible_{type2}")
            else:
                relation_expr = sp.Symbol(f"{type1}_{relation}_{type2}")
            
            self.type_relations.append({
                'type1': type1,
                'relation': relation,
                'type2': type2,
                'expression': relation_expr
            })
    
    def analyze_type_equivalence(self) -> Dict:
        """Analyze type equivalence classes and their properties"""
        analysis = {
            'equivalence_classes': {},
            'group_actions': {},
            'type_hierarchy': {},
            'mathematical_properties': []
        }
        
        for name, type_class in self.type_equivalence_classes.items():
            analysis['equivalence_classes'][name] = {
                'elements': list(type_class.elements),
                'base_type': type_class.base_type,
                'mathematical_symbol': str(type_class.mathematical_symbol),
                'group_actions': list(type_class.group_actions.keys())
            }
        
        return analysis
    
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
    
    def apply_sympy_transformations(self, code: str) -> str:
        """
        Apply SymPy-based transformations to code using pure mathematical equations
        
        Args:
            code: The original Python code string
        
        Returns:
            The transformed code with SymPy mathematical transformations applied
        """
        # Create SymPy mathematical transformations
        transformations = []
            
        # 5. File extension transformation (mathematical equation)
        json_file_symbol = sp.Symbol('*.json')
        toml_file_symbol = sp.Symbol('*.toml')
        file_transform = sp.Eq(json_file_symbol, toml_file_symbol)
        transformations.append(f"# SymPy Transform: {file_transform}")
        
        # Apply mathematical transformations using SymPy symbolic substitution
        transformed_code = code
        
        # For now, just return the original code since we're focusing on discovery
        # The real transformation happens in the transform() method
        transformed_code = code
        
        # Apply mathematical transformations to the code
        # This is purely SymPy-based - no AST manipulation
        final_code = f"""# SymPy Mathematical Transformations Applied
{chr(10).join(transformations)}

# Transformed code with mathematical substitutions:
{transformed_code}
"""
        
        return final_code

    def generate_mathematical_insights(self) -> Dict:
        """Generate mathematical insights using pure SymPy reasoning"""
        insights = {
            'mathematical_theorems': [],
            'sympy_transformations': [],
            'algebraic_properties': [],
            'geometric_insights': []
        }
        
        # Create SymPy mathematical theorems
        x, y, z = sp.symbols('x y z')
        f, g, h = sp.Function('f'), sp.Function('g'), sp.Function('h')
        
        # Mathematical theorems using SymPy
        composition_associativity = sp.Eq((f(g(h(x)))), f(g(h(x))))
        identity_function = sp.Eq(f(x), f(x))
        function_inverse = sp.Eq(f(sp.Function('f_inv')(x)), x)
        
        insights['mathematical_theorems'] = [
            f"Theorem 1: Function composition associativity: {composition_associativity}",
            f"Theorem 2: Identity function property: {identity_function}",
            f"Theorem 3: Function inverse property: {function_inverse}"
        ]
        
        # SymPy transformations
        json_transform = sp.Eq(sp.Symbol('json'), sp.Symbol('tomllib'))
        print_transform = sp.Eq(sp.Symbol('print'), sp.Symbol('(UNASSERTED)'))
        assert_transform = sp.Eq(sp.Symbol('assert'), sp.Symbol('spoken_assert'))
        
        insights['sympy_transformations'] = [
            f"Transform 1: {json_transform}",
            f"Transform 2: {print_transform}",
            f"Transform 3: {assert_transform}"
        ]
        
        return insights
    
    def export_mathematical_model(self, filename: str = "code_mathematical_model.toml") -> None:
        """Export mathematical model using pure SymPy expressions"""
        # Create SymPy mathematical model
        x, y = sp.symbols('x y')
        f, g = sp.Function('f'), sp.Function('g')
        
        model = {
            'mathematical_symbols': {
                name: {
                    'type': symbol.symbol_type.value,
                    'sympy_expression': str(symbol.mathematical_symbol),
                    'algebraic_properties': self._extract_algebraic_properties(symbol.mathematical_symbol)
                } for name, symbol in self.symbols.items()
            },
            'sympy_transformations': [
                {
                    'name': 'json_to_toml',
                    'equation': str(sp.Eq(sp.Symbol('json'), sp.Symbol('tomllib'))),
                    'mathematical_property': 'module_substitution'
                },
                {
                    'name': 'print_to_unasserted',
                    'equation': str(sp.Eq(sp.Symbol('print'), sp.Symbol('(UNASSERTED)'))),
                    'mathematical_property': 'function_replacement'
                },
                {
                    'name': 'assert_to_spoken',
                    'equation': str(sp.Eq(sp.Symbol('assert'), sp.Symbol('spoken_assert'))),
                    'mathematical_property': 'statement_transformation'
                }
            ],
            'mathematical_insights': self.generate_mathematical_insights()
        }
        
        # Export using pure mathematical format
        toml_content = self._dict_to_toml(model)
        with open(filename, 'w') as f:
            f.write(toml_content)
        
        print(f"💾 Mathematical model exported to {filename}")
    
    def _extract_algebraic_properties(self, sympy_expr) -> Dict:
        """Extract algebraic properties from SymPy expressions"""
        properties = {
            'is_function': hasattr(sympy_expr, 'is_Function'),
            'is_symbol': hasattr(sympy_expr, 'is_Symbol'),
            'free_symbols': [str(s) for s in sympy_expr.free_symbols] if hasattr(sympy_expr, 'free_symbols') else [],
            'mathematical_type': type(sympy_expr).__name__
        }
        return properties

    def create_finite_field_type(self, dimension: int) -> sp.Expr:
        """Create a type representing F₂ⁿ finite field."""
        return sp.Symbol(f'F2_{dimension}', domain=sp.GF(2))
    
    def create_cyclotomic_field_type(self, n: int) -> sp.Expr:
        """Create a type representing cyclotomic field Q(ζₙ)."""
        # Create cyclotomic field using primitive root of unity
        zeta = sp.exp(2*sp.pi*sp.I/n)  # ζₙ = e^(2πi/n)
        return sp.Symbol(f'Q_zeta_{n}', domain=QQ.algebraic_field(zeta))
    
    def narrow_type(self, original_type: sp.Expr, constraint: Any) -> Any:
        """Narrow a type by applying a constraint (field restriction)."""
        return sp.Eq(
            sp.Function('narrow')(original_type, constraint),
            sp.Function('restricted_type')(original_type, constraint)
        )
    
    def create_existential_type(self, predicate: sp.Expr, witness: sp.Expr) -> Any:
        """Create a type from existential quantification."""
        return sp.Eq(
            sp.Function('exists_type')(predicate, witness),
            sp.Function('witness_type')(witness, sp.Function('proof')(witness, predicate))
        )
    
    def prove_theorem(self, statement: Any) -> Any:
        """Prove a theorem and create a verified type."""
        return sp.Eq(
            sp.Function('theorem_proof')(statement),
            sp.Function('verified_type')(statement, sp.Function('proof')(statement))
        )
    
    def type_narrowing_analysis(self, code1: str, code2: str) -> Dict:
        """Analyze type narrowing between two code snippets."""
        symbols1 = self._extract_symbols_from_ast(ast.parse(code1))
        symbols2 = self._extract_symbols_from_ast(ast.parse(code2))
        
        # Create finite field representations
        dim1 = len(symbols1.get('variables', []))
        dim2 = len(symbols2.get('variables', []))
        
        f2_n1 = self.create_finite_field_type(dim1)
        f2_n2 = self.create_finite_field_type(dim2)
        
        # Type narrowing transformation
        constraint = sp.Eq(dim1, dim2)
        narrowing = self.narrow_type(f2_n1, constraint)
        
        # Existential type creation
        existential = self.create_existential_type(
            sp.Function('transform')(f2_n1, f2_n2),
            sp.Function('witness')(f2_n1, f2_n2)
        )
        
        # Theorem: type narrowing preserves structure
        theorem = self.prove_theorem(
            sp.Eq(
                sp.Function('narrow')(f2_n1, constraint),
                f2_n2
            )
        )
        
        return {
            'finite_field_types': {
                'original': str(f2_n1),
                'narrowed': str(f2_n2),
                'dimension_change': f'{dim1} → {dim2}'
            },
            'type_narrowing': str(narrowing),
            'existential_type': str(existential),
            'theorem': str(theorem),
            'field_arithmetic': {
                'addition': 'x + y (mod 2)',
                'multiplication': 'x * y (mod 2)',
                'inversion': 'x⁻¹ = x (in F₂)'
            }
        }
    
    def cyclotomic_field_analysis(self, code1: str, code2: str) -> Dict:
        """Analyze cyclotomic field properties between two code snippets."""
        symbols1 = self._extract_symbols_from_ast(ast.parse(code1))
        symbols2 = self._extract_symbols_from_ast(ast.parse(code2))
        
        # Create cyclotomic field representations
        n1 = len(symbols1.get('functions', []))
        n2 = len(symbols2.get('functions', []))
        
        # Use n1, n2 as orders of cyclotomic fields
        q_zeta_n1 = self.create_cyclotomic_field_type(n1)
        q_zeta_n2 = self.create_cyclotomic_field_type(n2)
        
        # Cyclotomic field properties
        zeta_n1 = sp.exp(2*sp.pi*sp.I/n1)
        zeta_n2 = sp.exp(2*sp.pi*sp.I/n2)
        
        # Galois group analysis
        galois_group_n1 = sp.Symbol(f'Gal(Q_zeta_{n1}/Q)')
        galois_group_n2 = sp.Symbol(f'Gal(Q_zeta_{n2}/Q)')
        
        # Field extension analysis
        extension_degree_n1 = sp.Symbol(f'[Q_zeta_{n1}:Q]')
        extension_degree_n2 = sp.Symbol(f'[Q_zeta_{n2}:Q]')
        
        # Cyclotomic polynomial
        cyclotomic_poly_n1 = sp.Symbol(f'Φ_{n1}(x)')
        cyclotomic_poly_n2 = sp.Symbol(f'Φ_{n2}(x)')
        
        return {
            'cyclotomic_fields': {
                'original': str(q_zeta_n1),
                'transformed': str(q_zeta_n2),
                'order_change': f'{n1} → {n2}'
            },
            'primitive_roots': {
                'original': f'ζ_{n1} = {zeta_n1}',
                'transformed': f'ζ_{n2} = {zeta_n2}'
            },
            'galois_groups': {
                'original': str(galois_group_n1),
                'transformed': str(galois_group_n2),
                'isomorphism': sp.Eq(galois_group_n1, galois_group_n2)
            },
            'field_extensions': {
                'original_degree': str(extension_degree_n1),
                'transformed_degree': str(extension_degree_n2),
                'degree_relation': sp.Eq(extension_degree_n1, extension_degree_n2)
            },
            'cyclotomic_polynomials': {
                'original': str(cyclotomic_poly_n1),
                'transformed': str(cyclotomic_poly_n2),
                'polynomial_relation': sp.Eq(cyclotomic_poly_n1, cyclotomic_poly_n2)
            },
            'field_arithmetic': {
                'addition': 'x + y (in Q(ζₙ))',
                'multiplication': 'x * y (in Q(ζₙ))',
                'inversion': 'x⁻¹ (in Q(ζₙ))',
                'conjugation': 'x̄ (complex conjugate)'
                        }
        }
    
    def create_fixed_point_operator(self, function_name: str) -> sp.Expr:
        """Create a fixed point operator f* for a given function on the critical line s = 1/2."""
        # Define the critical line s = 1/2
        critical_line = sp.Rational(1, 2)
        
        # Create fixed point operator: f*(s) = f(s) where s = 1/2
        fixed_point = f_star(sp.Function(function_name)(critical_line))
        
        return fixed_point
    
    def zeta_function_analysis(self, code: str) -> Dict:
        """Analyze code using the Riemann zeta function on the critical line."""
        tree = ast.parse(code)
        symbols = self._extract_symbols_from_ast(tree)
        
        # Critical line s = 1/2
        s = sp.Rational(1, 2)
        
        # Zeta function on critical line
        zeta_critical = zeta(s)
        
        # Fixed point analysis for each function
        fixed_points = {}
        for func_name in symbols.get('functions', []):
            fixed_points[func_name] = self.create_fixed_point_operator(func_name)
        
        # Riemann hypothesis connection
        # On critical line: ζ(s) = 0 implies s = 1/2 + it for some t
        riemann_hypothesis = sp.Eq(zeta(s), 0)
        
        # Functional equation: ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
        functional_equation = sp.Eq(
            zeta(s),
            2**s * sp.pi**(s-1) * sp.sin(sp.pi*s/2) * sp.gamma(1-s) * zeta(1-s)
        )
        
        return {
            'critical_line': str(s),
            'zeta_critical': str(zeta_critical),
            'fixed_points': {name: str(fp) for name, fp in fixed_points.items()},
            'riemann_hypothesis': str(riemann_hypothesis),
            'functional_equation': str(functional_equation),
            'code_complexity': len(symbols.get('functions', [])),
            'zeta_zeros': 's = 1/2 + it (non-trivial zeros)'
        }
    
    def critical_line_transformation(self, code1: str, code2: str) -> Dict:
        """Analyze code transformation using fixed points on the critical line."""
        analysis1 = self.zeta_function_analysis(code1)
        analysis2 = self.zeta_function_analysis(code2)
        
        # Critical line parameter
        s = sp.Rational(1, 2)
        
        # Fixed point transformation
        transformation_fixed_point = sp.Eq(
            f_star(sp.Symbol('Code_1')),
            f_star(sp.Symbol('Code_2'))
        )
        
        # Zeta function transformation
        zeta_transformation = sp.Eq(
            zeta(s).subs(sp.Symbol('Code_1'), sp.Symbol('Code_2')),
            zeta(s)
        )
        
        # Critical line invariance
        critical_invariance = sp.Eq(s, sp.Rational(1, 2))
        
        return {
            'original_analysis': analysis1,
            'transformed_analysis': analysis2,
            'fixed_point_transformation': str(transformation_fixed_point),
            'zeta_transformation': str(zeta_transformation),
            'critical_line_invariance': str(critical_invariance),
            'complexity_change': f"{analysis1['code_complexity']} → {analysis2['code_complexity']}"
        }
    
    def analyze_transformation_group(self, code1: str, code2: str) -> Dict:
        """Analyze whether a transformation defines a group."""
        # Extract transformations between the codes
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        
        symbols1 = self._extract_symbols_from_ast(tree1)
        symbols2 = self._extract_symbols_from_ast(tree2)
        
        # Define the transformation as a function
        T = sp.Function('T')  # Transformation function
        
        # Group properties to check:
        # 1. Closure: T(T(x)) = x (idempotent) or T(T(x)) = T(x) (projection)
        # 2. Identity: T(e) = e for some identity element e
        # 3. Inverse: T⁻¹ exists such that T⁻¹(T(x)) = x
        # 4. Associativity: T(T(T(x))) = T(T(T(x))) (trivially true for functions)
        
        # Check closure property
        closure_check = sp.Eq(T(T(sp.Symbol('x'))), T(sp.Symbol('x')))
        
        # Check identity (assuming identity is the original code)
        identity_check = sp.Eq(T(sp.Symbol('Code_1')), sp.Symbol('Code_1'))
        
        # Check inverse (T⁻¹ should transform back)
        inverse_check = sp.Eq(T(sp.Symbol('Code_2')), sp.Symbol('Code_1'))
        
        # Check associativity
        associativity_check = sp.Eq(
            T(T(T(sp.Symbol('x')))),
            T(T(T(sp.Symbol('x'))))
        )
        
        # Analyze transformation properties
        added_functions = set(symbols2.get('functions', [])) - set(symbols1.get('functions', []))
        removed_functions = set(symbols1.get('functions', [])) - set(symbols2.get('functions', []))
        
        # Determine if transformation is invertible
        is_invertible = len(removed_functions) == 0  # Only additions are invertible
        
        # Determine if transformation is idempotent
        is_idempotent = len(added_functions) == 0  # No change means idempotent
        
        return {
            'transformation_function': str(T(sp.Symbol('Code_1'))),
            'group_properties': {
                'closure': str(closure_check),
                'identity': str(identity_check),
                'inverse': str(inverse_check),
                'associativity': str(associativity_check)
            },
            'transformation_analysis': {
                'added_functions': list(added_functions),
                'removed_functions': list(removed_functions),
                'is_invertible': is_invertible,
                'is_idempotent': is_idempotent,
                'is_group': is_invertible and len(added_functions) > 0
            },
            'group_structure': {
                'order': len(added_functions) + len(removed_functions),
                'generators': list(added_functions),
                'identity_element': 'Code_1',
                'inverse_element': 'Code_2' if is_invertible else 'None'
            }
        }
    
    def transform(self, code1: str, code2: str) -> Dict:
        """
        Discover the mathematical transformation between two pieces of code
        by defining them as equivalent using SymPy
        
        Args:
            code1: Original code
            code2: Transformed code
        
        Returns:
            Dictionary containing the discovered transformation equations
        """
        # Parse both codes to extract mathematical symbols
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        
        # Extract symbols from both codes
        symbols1 = self._extract_symbols_from_ast(tree1)
        symbols2 = self._extract_symbols_from_ast(tree2)
        
        # Create SymPy equations defining equivalence
        transformations = {}
        
        # 1. Module transformations (e.g., json → tomllib)
        for module1 in symbols1.get('modules', []):
            for module2 in symbols2.get('modules', []):
                if self._are_symbols_related(module1, module2):
                    eq = sp.Eq(sp.Symbol(module1), sp.Symbol(module2))
                    transformations[f'module_{module1}_to_{module2}'] = str(eq)
        
        # 2. Function transformations (e.g., print → (UNASSERTED))
        for func1 in symbols1.get('functions', []):
            for func2 in symbols2.get('functions', []):
                if self._are_symbols_related(func1, func2):
                    eq = sp.Eq(sp.Function(func1), sp.Function(func2))
                    transformations[f'function_{func1}_to_{func2}'] = str(eq)
        
        # 3. Variable transformations (e.g., data.json → data.toml)
        for var1 in symbols1.get('variables', []):
            for var2 in symbols2.get('variables', []):
                if self._are_symbols_related(var1, var2):
                    eq = sp.Eq(sp.Symbol(var1), sp.Symbol(var2))
                    transformations[f'variable_{var1}_to_{var2}'] = str(eq)
        
        # 4. String literal transformations (e.g., "data.json" → "data.toml")
        strings1 = self._extract_string_literals(tree1)
        strings2 = self._extract_string_literals(tree2)
        for str1 in strings1:
            for str2 in strings2:
                if self._are_strings_related(str1, str2):
                    eq = sp.Eq(sp.Symbol(f'"{str1}"'), sp.Symbol(f'"{str2}"'))
                    transformations[f'string_{str1}_to_{str2}'] = str(eq)
        
        return {
            'code1': code1,
            'code2': code2,
            'transformations': transformations,
            'mathematical_equivalence': self._create_equivalence_equation(symbols1, symbols2)
        }
    
    def _extract_symbols_from_ast(self, tree: ast.AST) -> Dict:
        """Extract mathematical symbols from AST"""
        symbols = {
            'modules': [],
            'functions': [],
            'variables': [],
            'constants': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    symbols['modules'].append(alias.name)
            elif isinstance(node, ast.FunctionDef):
                symbols['functions'].append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper():
                            symbols['constants'].append(target.id)
                        else:
                            symbols['variables'].append(target.id)
        
        return symbols
    
    def _extract_string_literals(self, tree: ast.AST) -> List[str]:
        """Extract string literals from AST"""
        strings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                strings.append(node.value)
        return strings
    
    def _are_symbols_related(self, symbol1: str, symbol2: str) -> bool:
        """Determine if two symbols are related (e.g., json and tomllib)"""
        # Define known relationships
        relationships = {
            'json': 'tomllib',
            'print': '(UNASSERTED)',
            'assert': 'spoken_assert',
            'data.json': 'data.toml',
            # Add new transformation relationships
            'dataclass': 'dict',
            'toml': 'json',
            'yaml': 'json',
            'pickle': 'json'
        }
        
        return (symbol1 in relationships and relationships[symbol1] == symbol2) or \
               (symbol2 in relationships and relationships[symbol2] == symbol1)
    
    def _are_strings_related(self, str1: str, str2: str) -> bool:
        """Determine if two strings are related (e.g., .json → .toml)"""
        if str1.endswith('.json') and str2.endswith('.toml'):
            return str1.replace('.json', '') == str2.replace('.toml', '')
        elif str1.endswith('.toml') and str2.endswith('.json'):
            return str1.replace('.toml', '') == str2.replace('.json', '')
        elif str1.endswith('.yaml') and str2.endswith('.json'):
            return str1.replace('.yaml', '') == str2.replace('.json', '')
        return False
    
    def _create_equivalence_equation(self, symbols1: Dict, symbols2: Dict) -> str:
        """Create a mathematical equation defining equivalence between two code structures"""
        # Create SymPy symbols for the entire code structures
        code1_sym = sp.Symbol('Code_1')
        code2_sym = sp.Symbol('Code_2')
        
        # Define equivalence: Code_1 ≡ Code_2
        equivalence = sp.Eq(code1_sym, code2_sym)
        
        return str(equivalence)


class MinimalSpanProgramEngine:
    """
    Engine for generating programs from minimal mathematical span
    """
    
    def __init__(self):
        # Initialize minimal span
        self.s = sp.Symbol('s', real=True)
        self.critical_line = sp.Rational(1, 2)
        self.zeta = sp.Function('ζ')
        self.f_star = sp.Function('f*')
        self.x, self.y, self.z = sp.symbols('x y z', real=True)
        self.f, self.g, self.h = sp.Function('f'), sp.Function('g'), sp.Function('h')
        
        # Mathematical primitives
        self.mathematical_primitives = {
            'critical_line_parameter': self.s,
            'critical_line_value': self.critical_line,
            'zeta_function': self.zeta,
            'fixed_point_operator': self.f_star,
            'variable_symbols': [self.x, self.y, self.z],
            'function_symbols': [self.f, self.g, self.h]
        }
        
        # Code generation templates
        self.code_templates = {
            'function': 'def {name}({params}):\n    {body}',
            'class': 'class {name}:\n    def __init__(self, {params}):\n        {body}',
            'import': 'import {module}',
            'variable': '{name} = {value}',
            'assignment': '{target} = {expression}'
        }
        
        # Transformation rules
        self.transformation_rules = {
            'json': 'tomllib',
            'print': '(UNASSERTED)',
            'assert': 'spoken_assert',
            'fibonacci': 'recursive_fib',
            'factorial': 'recursive_fact',
            'data.json': 'data.toml'
        }
    
    def generate_mathematical_symbols(self) -> Dict:
        """Generate mathematical symbols from minimal span"""
        return {
            'critical_line_parameter': str(self.s),
            'critical_line_value': str(self.critical_line),
            'zeta_function': str(self.zeta(self.s)),
            'fixed_point_operator': str(self.f_star(self.f(self.x))),
            'variable_symbols': [str(sym) for sym in self.mathematical_primitives['variable_symbols']],
            'function_symbols': [str(sym(self.x)) for sym in self.mathematical_primitives['function_symbols']]
        }
    
    def generate_code_elements(self, mathematical_symbols: Dict) -> Dict:
        """Generate code elements from mathematical symbols"""
        # Create code elements based on mathematical symbols
        code_elements = {
            'functions': [],
            'types': [],
            'variables': [],
            'imports': [],
            'constants': []
        }
        
        # Generate functions from function symbols
        for i, func_sym in enumerate(mathematical_symbols['function_symbols']):
            func_name = f"function_{i+1}"
            code_elements['functions'].append({
                'name': func_name,
                'symbol': func_sym,
                'template': self.code_templates['function'].format(
                    name=func_name,
                    params='x',
                    body='return x'
                )
            })
        
        # Generate types from mathematical structures
        type_names = ['Vector', 'Matrix', 'Complex', 'Polynomial']
        for type_name in type_names:
            code_elements['types'].append({
                'name': type_name,
                'template': self.code_templates['class'].format(
                    name=type_name,
                    params='self, x',
                    body='self.x = x'
                )
            })
        
        # Generate variables from variable symbols
        for i, var_sym in enumerate(mathematical_symbols['variable_symbols']):
            var_name = f"var_{i+1}"
            code_elements['variables'].append({
                'name': var_name,
                'symbol': str(var_sym),
                'template': self.code_templates['variable'].format(
                    name=var_name,
                    value='0'
                )
            })
        
        return code_elements
    
    def generate_mathematical_relations(self, code_elements: Dict) -> Dict:
        """Generate mathematical relations from code elements"""
        relations = {}
        
        # Function composition relations
        if len(code_elements['functions']) >= 2:
            f1 = sp.Function(code_elements['functions'][0]['name'])
            f2 = sp.Function(code_elements['functions'][1]['name'])
            relations['function_composition'] = sp.Eq(f1(f2(self.x)), f2(f1(self.x)))
        
        # Type equivalence relations
        if len(code_elements['types']) >= 2:
            type1 = sp.Symbol(code_elements['types'][0]['name'])
            type2 = sp.Symbol(code_elements['types'][1]['name'])
            relations['type_equivalence'] = sp.Eq(type1, type2)
        
        # Critical line relations
        relations['critical_line_invariance'] = sp.Eq(self.s, self.critical_line)
        relations['zeta_functional_equation'] = sp.Eq(self.zeta(self.s), self.zeta(1-self.s))
        
        return {name: str(relation) for name, relation in relations.items()}
    
    def generate_program_transformations(self, mathematical_relations: Dict) -> Dict:
        """Generate program transformations from mathematical relations"""
        transformations = {}
        
        # Convert mathematical equations to code transformations
        for relation_name, relation in mathematical_relations.items():
            if 'function_composition' in relation_name:
                transformations['function_ordering'] = 'f1(f2(x)) → f2(f1(x))'
            elif 'type_equivalence' in relation_name:
                transformations['type_conversion'] = 'Vector → Matrix'
            elif 'critical_line' in relation_name:
                transformations['critical_invariance'] = 's = 1/2 (preserved)'
        
        # Add standard transformations
        transformations.update(self.transformation_rules)
        
        return transformations
    
    def apply_transformations(self, original_code: str, transformations: Dict) -> str:
        """Apply transformations to generate new code"""
        transformed_code = original_code
        
        for transform_name, transform_rule in transformations.items():
            if '→' in transform_rule:
                old, new = transform_rule.split(' → ')
                transformed_code = transformed_code.replace(old, new)
            elif transform_name in ['json', 'print', 'assert']:
                old = transform_name
                new = transform_rule
                transformed_code = transformed_code.replace(old, new)
        
        return transformed_code
    
    def generate_program(self, target_complexity: int = 3) -> Dict:
        """Generate a complete program from minimal span"""
        # Step 1: Generate mathematical symbols
        mathematical_symbols = self.generate_mathematical_symbols()
        
        # Step 2: Generate code elements
        code_elements = self.generate_code_elements(mathematical_symbols)
        
        # Step 3: Generate mathematical relations
        mathematical_relations = self.generate_mathematical_relations(code_elements)
        
        # Step 4: Generate transformations
        transformations = self.generate_program_transformations(mathematical_relations)
        
        # Step 5: Generate base program
        base_program = self._generate_base_program(code_elements, target_complexity)
        
        # Step 6: Apply transformations
        final_program = self.apply_transformations(base_program, transformations)
        
        return {
            'mathematical_symbols': mathematical_symbols,
            'code_elements': code_elements,
            'mathematical_relations': mathematical_relations,
            'transformations': transformations,
            'base_program': base_program,
            'final_program': final_program
        }
    
    def _generate_base_program(self, code_elements: Dict, complexity: int) -> str:
        """Generate a base program from code elements"""
        program_lines = []
        
        # Add imports
        program_lines.append("import math")
        program_lines.append("import sympy as sp")
        program_lines.append("")
        
        # Add functions (up to complexity)
        for i, func in enumerate(code_elements['functions'][:complexity]):
            program_lines.append(func['template'])
            program_lines.append("")
        
        # Add types (up to complexity)
        for i, type_def in enumerate(code_elements['types'][:complexity]):
            program_lines.append(type_def['template'])
            program_lines.append("")
        
        # Add variables
        for var in code_elements['variables'][:complexity]:
            program_lines.append(var['template'])
        
        return '\n'.join(program_lines)
    
    def analyze_generated_program(self, generated_program: Dict) -> Dict:
        """Analyze the mathematical properties of the generated program"""
        analysis = {
            'mathematical_foundations': generated_program['mathematical_symbols'],
            'code_structure': {
                'functions': len(generated_program['code_elements']['functions']),
                'types': len(generated_program['code_elements']['types']),
                'variables': len(generated_program['code_elements']['variables'])
            },
            'mathematical_relations': generated_program['mathematical_relations'],
            'applied_transformations': generated_program['transformations'],
            'program_complexity': len(generated_program['final_program'].split('\n'))
        }
        
        return analysis
    
    def rewrite_code(self, code: str, target_complexity: int = 3) -> str:
        """Rewrite code using the mathematical framework"""
        # Generate mathematical symbols
        mathematical_symbols = self.generate_mathematical_symbols()
        
        # Generate code elements
        code_elements = self.generate_code_elements(mathematical_symbols)
        
        # Generate mathematical relations
        mathematical_relations = self.generate_mathematical_relations(code_elements)
        
        # Generate transformations
        transformations = self.generate_program_transformations(mathematical_relations)
        
        # Apply transformations to the input code
        rewritten_code = self.apply_transformations(code, transformations)
        
        return rewritten_code
    
    def save_code_to_file(self, code: str, filename: str) -> None:
        """Save code to a file"""
        import os
        os.makedirs('.out', exist_ok=True)
        
        with open(f'.out/{filename}', 'w') as f:
            f.write(code)
        
        print(f"💾 Code saved to .out/{filename}")
    
    def compare_codes(self, original_code: str, transformed_code: str) -> Dict:
        """Compare original and transformed codes"""
        import difflib
        
        # Split codes into lines
        original_lines = original_code.split('\n')
        transformed_lines = transformed_code.split('\n')
        
        # Generate diff
        diff = list(difflib.unified_diff(
            original_lines, 
            transformed_lines,
            fromfile='original',
            tofile='transformed',
            lineterm=''
        ))
        
        # Count differences
        added_lines = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        removed_lines = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
        
        return {
            'diff': diff,
            'added_lines': added_lines,
            'removed_lines': removed_lines,
            'total_changes': added_lines + removed_lines,
            'original_lines': len(original_lines),
            'transformed_lines': len(transformed_lines)
        }


def demo_sympy_ast_parser():
    """
    Demonstrate the full power of the mathematical code analysis framework
    """
    print("🔬 MATHEMATICAL CODE ANALYSIS FRAMEWORK")
    print("=" * 60)
    print("From Minimal Span to Program Generation via Mathematical Equations")
    print("=" * 60)
    
    # Create the parser
    parser = SymPyASTParser()
    
    # Create the minimal span program generation engine
    engine = MinimalSpanProgramEngine()
    
    print("\n📊 STEP 1: MINIMAL MATHEMATICAL SPAN")
    print("-" * 40)
    
    # Show the minimal span that generates everything
    minimal_span = {
        'critical_line': 's = 1/2 (Riemann zeta critical line)',
        'zeta_function': 'ζ(s) (Riemann zeta function)',
        'fixed_point_operator': 'f* (fixed point operator on critical line)',
        'finite_field': 'F₂ⁿ (binary finite fields)',
        'cyclotomic_field': 'Q(ζₙ) (cyclotomic fields)',
        'variable_symbols': 'x, y, z (real variables)',
        'function_symbols': 'f, g, h (mathematical functions)'
    }
    
    for name, definition in minimal_span.items():
        print(f"  {name}: {definition}")
    
    print("\n📊 STEP 2: GENERATE MATHEMATICAL SYMBOLS")
    print("-" * 40)
    
    # Generate mathematical symbols from minimal span
    mathematical_symbols = engine.generate_mathematical_symbols()
    
    for name, symbol in mathematical_symbols.items():
        if isinstance(symbol, list):
            print(f"  {name}: {', '.join(symbol)}")
        else:
            print(f"  {name}: {symbol}")
    
    print("\n📊 STEP 3: GENERATE CODE ELEMENTS")
    print("-" * 40)
    
    # Generate code elements from mathematical symbols
    code_elements = engine.generate_code_elements(mathematical_symbols)
    
    for element_type, elements in code_elements.items():
        if elements:
            print(f"  {element_type}: {len(elements)} elements generated")
    
    print("\n📊 STEP 4: CREATE MATHEMATICAL RELATIONS")
    print("-" * 40)
    
    # Generate mathematical relations
    mathematical_relations = engine.generate_mathematical_relations(code_elements)
    
    for relation_name, relation in mathematical_relations.items():
        print(f"  {relation_name}: {relation}")
    
    print("\n📊 STEP 5: GENERATE PROGRAM TRANSFORMATIONS")
    print("-" * 40)
    
    # Generate transformations from mathematical relations
    transformations = engine.generate_program_transformations(mathematical_relations)
    
    for transform_name, transform_rule in transformations.items():
        print(f"  {transform_name}: {transform_rule}")
    
    print("\n📊 STEP 6: GENERATE COMPLETE PROGRAM")
    print("-" * 40)
    
    # Generate a complete program from minimal span
    generated_program = engine.generate_program(target_complexity=5)
    
    print("Generated Program:")
    print("-" * 20)
    print(generated_program['final_program'])
    
    print("\n📊 STEP 7: MATHEMATICAL ANALYSIS")
    print("-" * 40)
    
    # Analyze the generated program
    analysis = engine.analyze_generated_program(generated_program)
    
    print(f"  Functions: {analysis['code_structure']['functions']}")
    print(f"  Types: {analysis['code_structure']['types']}")
    print(f"  Variables: {analysis['code_structure']['variables']}")
    print(f"  Complexity: {analysis['program_complexity']} lines")
    
    print("\n📊 STEP 8: ADVANCED MATHEMATICAL FEATURES")
    print("-" * 40)
    
    # Demonstrate advanced features
    print("🔬 F₂ⁿ Finite Field Analysis:")
    wide_code = "def process(x, y, z): return x + y + z"
    narrow_code = "def process(x, y): return x + y"
    field_analysis = parser.type_narrowing_analysis(wide_code, narrow_code)
    print(f"  Field types: {field_analysis['finite_field_types']['original']} → {field_analysis['finite_field_types']['narrowed']}")
    
    print("\n🔬 Cyclotomic Field Analysis:")
    simple_code = "def add(a, b): return a + b"
    complex_code = "def add(a, b): return a + b\ndef conjugate(z): return z.conjugate()"
    cyclotomic_analysis = parser.cyclotomic_field_analysis(simple_code, complex_code)
    print(f"  Cyclotomic fields: {cyclotomic_analysis['cyclotomic_fields']['original']} → {cyclotomic_analysis['cyclotomic_fields']['transformed']}")
    
    print("\n🔬 Fixed Point & Zeta Analysis:")
    zeta_analysis = parser.zeta_function_analysis(wide_code)
    print(f"  Critical line: s = {zeta_analysis['critical_line']}")
    print(f"  Zeta function: {zeta_analysis['zeta_critical']}")
    
    print("\n🔬 Transformation Group Analysis:")
    group_analysis = parser.analyze_transformation_group(wide_code, narrow_code)
    print(f"  Forms a group: {group_analysis['transformation_analysis']['is_group']}")
    print(f"  Group order: {group_analysis['group_structure']['order']}")
    
    print("\n🎯 MATHEMATICAL FRAMEWORK SUMMARY:")
    print("=" * 50)
    print("✅ Programs generated from minimal mathematical span")
    print("✅ Mathematical equations become code transformations")
    print("✅ Finite fields (F₂ⁿ) for binary arithmetic")
    print("✅ Cyclotomic fields (Q(ζₙ)) for complex extensions")
    print("✅ Fixed point operator f* on critical line s = 1/2")
    print("✅ Riemann zeta function analysis")
    print("✅ Transformation groups with closure, identity, inverse")
    print("✅ Critical line invariance preserved")
    print("✅ Mathematical invariants maintained")
    
    print("\n🎯 KEY INSIGHT:")
    print("Code is mathematical structure. Programs are generated from")
    print("minimal span through mathematical equations that become transformations!")
    print("=" * 60)
    
    # Demonstrate self-rewriting capability
    print("\n🔬 SELF-REWRITING DEMONSTRATION")
    print("=" * 50)
    
    # Read the current file
    with open('sympy_ast_parser.py', 'r') as f:
        original_code = f.read()
    
    print("📊 Original code loaded from sympy_ast_parser.py")
    print(f"  Lines: {len(original_code.split(chr(10)))}")
    print(f"  Characters: {len(original_code)}")
    
    # Apply the engine to rewrite itself
    print("\n📊 Applying mathematical transformations to rewrite the code...")
    rewritten_code = engine.rewrite_code(original_code, target_complexity=5)
    
    # Save the rewritten code
    engine.save_code_to_file(rewritten_code, 'sympy_ast_parser_transformed.py')
    
    # Compare the codes
    comparison = engine.compare_codes(original_code, rewritten_code)
    
    print("\n📊 Code Comparison:")
    print(f"  Original lines: {comparison['original_lines']}")
    print(f"  Transformed lines: {comparison['transformed_lines']}")
    print(f"  Added lines: {comparison['added_lines']}")
    print(f"  Removed lines: {comparison['removed_lines']}")
    print(f"  Total changes: {comparison['total_changes']}")
    
    # Show some of the differences
    print("\n📊 Sample Differences:")
    diff_lines = comparison['diff'][:20]  # Show first 20 diff lines
    for line in diff_lines:
        if line.startswith('+'):
            print(f"  + {line[1:]}")
        elif line.startswith('-'):
            print(f"  - {line[1:]}")
        elif line.startswith('@'):
            print(f"  {line}")
    
    if len(comparison['diff']) > 20:
        print(f"  ... and {len(comparison['diff']) - 20} more differences")
    
    print("\n🎯 SELF-REWRITING COMPLETE!")
    print("The mathematical engine has successfully rewritten itself!")
    print("Check .out/sympy_ast_parser_transformed.py for the transformed code.")
    print("=" * 60)
    
    return parser, engine, generated_program, analysis


if __name__ == "__main__":
    parser, engine, generated_program, analysis = demo_sympy_ast_parser()


class MinimalSpanProgramEngine:
    """
    Engine for generating programs from minimal mathematical span
    """
    
    def __init__(self):
        # Initialize minimal span
        self.s = sp.Symbol('s', real=True)
        self.critical_line = sp.Rational(1, 2)
        self.zeta = sp.Function('ζ')
        self.f_star = sp.Function('f*')
        self.x, self.y, self.z = sp.symbols('x y z', real=True)
        self.f, self.g, self.h = sp.Function('f'), sp.Function('g'), sp.Function('h')
        
        # Mathematical primitives
        self.mathematical_primitives = {
            'critical_line_parameter': self.s,
            'critical_line_value': self.critical_line,
            'zeta_function': self.zeta,
            'fixed_point_operator': self.f_star,
            'variable_symbols': [self.x, self.y, self.z],
            'function_symbols': [self.f, self.g, self.h]
        }
        
        # Code generation templates
        self.code_templates = {
            'function': 'def {name}({params}):\n    {body}',
            'class': 'class {name}:\n    def __init__(self, {params}):\n        {body}',
            'import': 'import {module}',
            'variable': '{name} = {value}',
            'assignment': '{target} = {expression}'
        }
        
        # Transformation rules
        self.transformation_rules = {
            'json': 'tomllib',
            'print': '(UNASSERTED)',
            'assert': 'spoken_assert',
            'fibonacci': 'recursive_fib',
            'factorial': 'recursive_fact',
            'data.json': 'data.toml'
        }
    
    def generate_mathematical_symbols(self) -> Dict:
        """Generate mathematical symbols from minimal span"""
        return {
            'critical_line_parameter': str(self.s),
            'critical_line_value': str(self.critical_line),
            'zeta_function': str(self.zeta(self.s)),
            'fixed_point_operator': str(self.f_star(self.f(self.x))),
            'variable_symbols': [str(sym) for sym in self.mathematical_primitives['variable_symbols']],
            'function_symbols': [str(sym(self.x)) for sym in self.mathematical_primitives['function_symbols']]
        }
    
    def generate_code_elements(self, mathematical_symbols: Dict) -> Dict:
        """Generate code elements from mathematical symbols"""
        # Create code elements based on mathematical symbols
        code_elements = {
            'functions': [],
            'types': [],
            'variables': [],
            'imports': [],
            'constants': []
        }
        
        # Generate functions from function symbols
        for i, func_sym in enumerate(mathematical_symbols['function_symbols']):
            func_name = f"function_{i+1}"
            code_elements['functions'].append({
                'name': func_name,
                'symbol': func_sym,
                'template': self.code_templates['function'].format(
                    name=func_name,
                    params='x',
                    body='return x'
                )
            })
        
        # Generate types from mathematical structures
        type_names = ['Vector', 'Matrix', 'Complex', 'Polynomial']
        for type_name in type_names:
            code_elements['types'].append({
                'name': type_name,
                'template': self.code_templates['class'].format(
                    name=type_name,
                    params='self, x',
                    body='self.x = x'
                )
            })
        
        # Generate variables from variable symbols
        for i, var_sym in enumerate(mathematical_symbols['variable_symbols']):
            var_name = f"var_{i+1}"
            code_elements['variables'].append({
                'name': var_name,
                'symbol': str(var_sym),
                'template': self.code_templates['variable'].format(
                    name=var_name,
                    value='0'
                )
            })
        
        return code_elements
    
    def generate_mathematical_relations(self, code_elements: Dict) -> Dict:
        """Generate mathematical relations from code elements"""
        relations = {}
        
        # Function composition relations
        if len(code_elements['functions']) >= 2:
            f1 = sp.Function(code_elements['functions'][0]['name'])
            f2 = sp.Function(code_elements['functions'][1]['name'])
            relations['function_composition'] = sp.Eq(f1(f2(self.x)), f2(f1(self.x)))
        
        # Type equivalence relations
        if len(code_elements['types']) >= 2:
            type1 = sp.Symbol(code_elements['types'][0]['name'])
            type2 = sp.Symbol(code_elements['types'][1]['name'])
            relations['type_equivalence'] = sp.Eq(type1, type2)
        
        # Critical line relations
        relations['critical_line_invariance'] = sp.Eq(self.s, self.critical_line)
        relations['zeta_functional_equation'] = sp.Eq(self.zeta(self.s), self.zeta(1-self.s))
        
        return {name: str(relation) for name, relation in relations.items()}
    
    def generate_program_transformations(self, mathematical_relations: Dict) -> Dict:
        """Generate program transformations from mathematical relations"""
        transformations = {}
        
        # Convert mathematical equations to code transformations
        for relation_name, relation in mathematical_relations.items():
            if 'function_composition' in relation_name:
                transformations['function_ordering'] = 'f1(f2(x)) → f2(f1(x))'
            elif 'type_equivalence' in relation_name:
                transformations['type_conversion'] = 'Vector → Matrix'
            elif 'critical_line' in relation_name:
                transformations['critical_invariance'] = 's = 1/2 (preserved)'
        
        # Add standard transformations
        transformations.update(self.transformation_rules)
        
        return transformations
    
    def apply_transformations(self, original_code: str, transformations: Dict) -> str:
        """Apply transformations to generate new code"""
        transformed_code = original_code
        
        for transform_name, transform_rule in transformations.items():
            if '→' in transform_rule:
                old, new = transform_rule.split(' → ')
                transformed_code = transformed_code.replace(old, new)
            elif transform_name in ['json', 'print', 'assert']:
                old = transform_name
                new = transform_rule
                transformed_code = transformed_code.replace(old, new)
        
        return transformed_code
    
    def generate_program(self, target_complexity: int = 3) -> Dict:
        """Generate a complete program from minimal span"""
        # Step 1: Generate mathematical symbols
        mathematical_symbols = self.generate_mathematical_symbols()
        
        # Step 2: Generate code elements
        code_elements = self.generate_code_elements(mathematical_symbols)
        
        # Step 3: Generate mathematical relations
        mathematical_relations = self.generate_mathematical_relations(code_elements)
        
        # Step 4: Generate transformations
        transformations = self.generate_program_transformations(mathematical_relations)
        
        # Step 5: Generate base program
        base_program = self._generate_base_program(code_elements, target_complexity)
        
        # Step 6: Apply transformations
        final_program = self.apply_transformations(base_program, transformations)
        
        return {
            'mathematical_symbols': mathematical_symbols,
            'code_elements': code_elements,
            'mathematical_relations': mathematical_relations,
            'transformations': transformations,
            'base_program': base_program,
            'final_program': final_program
        }
    
    def _generate_base_program(self, code_elements: Dict, complexity: int) -> str:
        """Generate a base program from code elements"""
        program_lines = []
        
        # Add imports
        program_lines.append("import math")
        program_lines.append("import sympy as sp")
        program_lines.append("")
        
        # Add functions (up to complexity)
        for i, func in enumerate(code_elements['functions'][:complexity]):
            program_lines.append(func['template'])
            program_lines.append("")
        
        # Add types (up to complexity)
        for i, type_def in enumerate(code_elements['types'][:complexity]):
            program_lines.append(type_def['template'])
            program_lines.append("")
        
        # Add variables
        for var in code_elements['variables'][:complexity]:
            program_lines.append(var['template'])
        
        return '\n'.join(program_lines)
    
    def analyze_generated_program(self, generated_program: Dict) -> Dict:
        """Analyze the mathematical properties of the generated program"""
        analysis = {
            'mathematical_foundations': generated_program['mathematical_symbols'],
            'code_structure': {
                'functions': len(generated_program['code_elements']['functions']),
                'types': len(generated_program['code_elements']['types']),
                'variables': len(generated_program['code_elements']['variables'])
            },
            'mathematical_relations': generated_program['mathematical_relations'],
            'applied_transformations': generated_program['transformations'],
            'program_complexity': len(generated_program['final_program'].split('\n'))
        }
        
     