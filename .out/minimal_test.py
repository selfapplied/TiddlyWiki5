#!/usr/bin/env python3
"""Mathematical code transformer - compact version"""

import argparse, tomllib, re, ast, sympy as sp
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any
import sys
from sympy import symbols, Function, Eq
from pathlib import Path

# Dynamic enums that will be populated from TOML
class T(Enum): pass  # Will be populated from TOML
class P(Enum): pass  # Will be populated from TOML

@dataclass
class R:
    name: str; pattern: str; replacement: str; type: str; props: List[str]; desc: str; weight: float = 1.0; conds: List[str] = None
    def __post_init__(self): self.conds = self.conds or []

class M:
    def __init__(self):
        self.x, self.y, self.z = sp.symbols('x y z', real=True)
        self.f, self.g, self.h = sp.Function('f'), sp.Function('g'), sp.Function('h')
        self.props = {}; self.transformation_types = {}; self.mathematical_properties = {}
        self.rules = []; self.symbols = {}; self.history = []; self.actions = {}; self.current_action = None
        self.transformations = {}; self.composite_transformations = {}; self.lie_groups = {}; self.operator_classes = {}
        self.composition_symbol_table = {}; self.composition_compatibility = {}; self.projection_operators = {}
        self.fundamental_types = {}; self.generation_rules = {}
    
    def extract_symbols(self, code: str) -> Dict[str, Any]:
        symbols = {}
        for func in re.findall(r'def\s+(\w+)\s*\(', code): symbols[func] = sp.Function(func)
        for var in re.findall(r'(\w+)\s*=', code): 
            if var not in ['def', 'class', 'import', 'from', 'if', 'for', 'while']: symbols[var] = sp.Symbol(var)
        for imp in re.findall(r'import\s+(\w+)', code): symbols[imp] = sp.Symbol(imp)
        return symbols
    
    def transform(self, code: str) -> str:
        symbols = self.extract_symbols(code); self.symbols.update(symbols)
        result = code; self.history = []
        for rule in sorted(self.rules, key=lambda r: r.weight, reverse=True):
            if self._check_conditions(rule.conds, symbols):
                new_code = result.replace(rule.pattern, rule.replacement)
                if new_code != result: result = new_code; self.history.append({'rule': rule.name, 'pattern': rule.pattern, 'replacement': rule.replacement, 'type': rule.type})
        return result
    
    def _check_conditions(self, conditions: List[str], symbols: Dict[str, Any]) -> bool:
        if not conditions: return True
        for condition in conditions:
            if not self._evaluate_condition(condition, symbols):
                return False
        return True
    
    def _evaluate_condition(self, condition: str, symbols: Dict[str, Any]) -> bool:
        """Evaluate a condition based on symbols."""
        if condition.startswith('has_symbol:'):
            symbol_name = condition.split(':', 1)[1]
            return symbol_name in symbols
        return True
    
    def analyze(self, original: str, transformed: str) -> Dict:
        orig_syms = self.extract_symbols(original); trans_syms = self.extract_symbols(transformed)
        return {
            'original_lines': len(original.split('\n')),
            'transformed_lines': len(transformed.split('\n')),
            'rules_applied': len(self.history),
            'original_symbols': len(orig_syms),
            'transformed_symbols': len(trans_syms),
            'symbol_difference': len(orig_syms) - len(trans_syms),
            'transformation_count': len(self.history),
            'transformation_equations': [],
            'mathematical_properties_preserved': True
        }
    
    def load_toml(self, config_file: str, action: str = None) -> None:
        try:
            with open(config_file, 'r') as f: config = tomllib.loads(f.read())
            
            # Load transformation types and mathematical properties
            self.transformation_types = config.get('transformation_types', {})
            self.mathematical_properties = config.get('mathematical_properties', {})
            
            # Load transformations and composite transformations
            self.transformations = config.get('transformations', {})
            self.composite_transformations = {k: v for k, v in self.transformations.items() if v.get('type') == 'composite'}
            
            # Load Lie groups
            self.lie_groups = config.get('lie_groups', {})
            
            # Load operator classes
            self.operator_classes = config.get('operator_classes', {})
            
            # Load symbol table and compatibility matrix
            self.composition_symbol_table = config.get('composition_symbol_table', {})
            self.composition_compatibility = config.get('composition_compatibility', {})
            
            # Load projection operators
            self.projection_operators = config.get('projection_operators', {})
            
            # Load fundamental types and generation rules
            self.fundamental_types = config.get('fundamental_types', {})
            self.generation_rules = config.get('generation_rules', {})
            
            # Populate dynamic enums
            global T, P
            T = Enum('T', {k.upper(): k for k in self.transformation_types.keys()})
            P = Enum('P', {k.upper(): k for k in self.mathematical_properties.keys()})
            
            self.rules = []; self.actions = config.get('actions', {})
            self.action_combinations = config.get('action_combinations', {})
            self.action_intersections = config.get('action_intersections', {})
            
            if 'rules' in config:
                all_rules = []
                for r in config['rules']:
                    rule = R(name=r['name'], pattern=r['pattern'], replacement=r['replacement'], type=r['type'], props=r['preserved_properties'], desc=r['description'], weight=r.get('weight', 1.0), conds=r.get('conditions', []))
                    all_rules.append(rule)
                
                # Handle action combinations and intersections
                if action:
                    if action in self.action_combinations:
                        # Union of rules from multiple actions
                        self.current_action = action
                        action_names = self.action_combinations[action]['actions']
                        combined_rules = set()
                        for action_name in action_names:
                            if action_name in self.transformations:
                                action_rules = self.transformations[action_name].get('rules', [])
                                combined_rules.update(action_rules)
                        self.rules = [r for r in all_rules if r.name in combined_rules]
                        print(f"📋 Loaded {len(self.rules)} rules for combination '{action}' from {config_file}")
                        
                    elif action in self.action_intersections:
                        # Intersection of rules from multiple actions
                        self.current_action = action
                        action_names = self.action_intersections[action]['actions']
                        intersection_rules = None
                        for action_name in action_names:
                            if action_name in self.transformations:
                                action_rules = set(self.transformations[action_name].get('rules', []))
                                if intersection_rules is None:
                                    intersection_rules = action_rules
                                else:
                                    intersection_rules &= action_rules
                        self.rules = [r for r in all_rules if r.name in (intersection_rules or set())]
                        print(f"📋 Loaded {len(self.rules)} rules for intersection '{action}' from {config_file}")
                        
                    elif action in self.transformations:
                        # Single transformation
                        self.current_action = action
                        action_rules = self.transformations[action].get('rules', [])
                        self.rules = [r for r in all_rules if r.name in action_rules]
                        print(f"📋 Loaded {len(self.rules)} rules for transformation '{action}' from {config_file}")
                    else:
                        # Unknown action, use all rules
                        self.rules = all_rules
                        print(f"📋 Unknown action '{action}', loaded {len(self.rules)} rules from {config_file}")
                else:
                    self.rules = all_rules
                    print(f"📋 Loaded {len(self.rules)} rules from {config_file}")
        except FileNotFoundError: print(f"Warning: Config file {config_file} not found, using default rules")
        except Exception as e: print(f"Error loading config: {e}")

    def analyze_transformation_complexity(self, action: str) -> Dict[str, Any]:
        """Analyze the complexity of a transformation action."""
        if action in self.transformations:
            transform = self.transformations[action]
            return {
                'action': action,
                'type': transform.get('type', 'unknown'),
                'complexity': transform.get('complexity', 'O(1)'),
                'input_type': transform.get('input', 'unknown'),
                'output_type': transform.get('output', 'unknown'),
                'properties': transform.get('properties', []),
                'rule_count': len(transform.get('rules', [])),
                'description': transform.get('description', '')
            }
        return {'error': f"Transformation '{action}' not found"}
    
    def analyze_transformation_arity(self, action: str) -> Dict[str, Any]:
        """Analyze the arity of a transformation action."""
        if action in self.transformations:
            transform = self.transformations[action]
            
            # Determine arity based on transformation structure
            if 'transformations' in transform:
                # Composite transformation
                sub_transforms = transform['transformations']
                arity = len(sub_transforms)
                arity_type = f"arity_{arity}" if arity <= 2 else "arity_n"
            elif 'rules' in transform:
                # Single transformation with rules
                arity = 1
                arity_type = "arity_1"
            else:
                # Atomic rule
                arity = 0
                arity_type = "arity_0"
            
            return {
                'action': action,
                'arity': arity,
                'arity_type': arity_type,
                'type': transform.get('type', 'unknown'),
                'complexity': transform.get('complexity', 'O(1)'),
                'properties': transform.get('properties', []),
                'description': transform.get('description', ''),
                'is_base_case': arity == 0,
                'is_composite': arity > 1
            }
        return {'error': f"Transformation '{action}' not found"}
    
    def generalize_by_arity(self) -> Dict[str, Any]:
        """Generalize the transformation concept using arity."""
        arity_analysis = {}
        
        for action in self.transformations:
            analysis = self.analyze_transformation_arity(action)
            if 'error' not in analysis:
                arity_type = analysis['arity_type']
                if arity_type not in arity_analysis:
                    arity_analysis[arity_type] = []
                arity_analysis[arity_type].append(analysis)
        
        return {
            'generalization': 'Transformations generalized by arity',
            'base_case': 'Rules (arity 0) are atomic transformations',
            'induction': 'Higher arity = more complex composition',
            'arity_distribution': arity_analysis,
            'mathematical_structure': 'Forms a graded algebra over transformations'
        }
    
    def compose_transformations(self, action: str) -> Dict[str, Any]:
        """Compose multiple transformations for composite actions."""
        if action in self.composite_transformations:
            composite = self.composite_transformations[action]
            sub_transformations = composite.get('transformations', [])
            return {
                'action': action,
                'type': 'composite',
                'sub_transformations': sub_transformations,
                'properties': composite.get('properties', []),
                'complexity': composite.get('complexity', 'O(n)'),
                'description': composite.get('description', '')
            }
        return {'error': f"Composite transformation '{action}' not found"}

    def analyze_lie_group(self, group_name: str) -> Dict[str, Any]:
        """Analyze a Lie group transformation."""
        if group_name in self.lie_groups:
            group = self.lie_groups[group_name]
            return {
                'group': group_name,
                'description': group.get('description', ''),
                'generators': group.get('generators', []),
                'invariants': group.get('invariants', []),
                'orbit': group.get('orbit', ''),
                'dimension': group.get('dimension', 'unknown'),
                'complexity': group.get('complexity', 'O(1)'),
                'group_type': self._classify_lie_group(group_name, group),
                'properties': self._analyze_group_properties(group)
            }
        return {'error': f"Lie group '{group_name}' not found"}
    
    def _classify_lie_group(self, name: str, group: Dict) -> str:
        """Classify the type of Lie group."""
        if 'translation' in name:
            return "Abelian (additive)"
        elif 'scaling' in name:
            return "Abelian (multiplicative)"
        elif 'rotation' in name:
            return "SO(2) - Special Orthogonal"
        elif 'symmetry' in name:
            return "S_n - Symmetric Group"
        elif 'gauge' in name:
            return "Gauge Group (local)"
        elif 'conformal' in name:
            return "Conformal Group"
        else:
            return "Unknown"
    
    def _analyze_group_properties(self, group: Dict) -> Dict[str, Any]:
        """Analyze mathematical properties of the group."""
        dimension = group.get('dimension', 'unknown')
        complexity = group.get('complexity', 'O(1)')
        
        return {
            'is_abelian': 'translation' in group.get('description', '') or 'scaling' in group.get('description', ''),
            'is_finite': dimension != 'infinite' and dimension != 'n!',
            'is_compact': dimension in ['1', 'finite'],
            'is_simple': 'rotation' in group.get('description', '') or 'symmetry' in group.get('description', ''),
            'dimension_type': 'finite' if dimension in ['1', 'finite'] else 'infinite',
            'complexity_class': complexity
        }
    
    def compose_lie_groups(self, group1: str, group2: str) -> Dict[str, Any]:
        """Compose two Lie group transformations."""
        if group1 in self.lie_groups and group2 in self.lie_groups:
            g1 = self.lie_groups[group1]
            g2 = self.lie_groups[group2]
            
            # Group composition properties
            is_abelian = self._analyze_group_properties(g1)['is_abelian'] and self._analyze_group_properties(g2)['is_abelian']
            
            return {
                'composition': f"{group1} × {group2}",
                'is_abelian': is_abelian,
                'generators': g1.get('generators', []) + g2.get('generators', []),
                'invariants': list(set(g1.get('invariants', []) + g2.get('invariants', []))),
                'dimension': f"{g1.get('dimension', '1')} × {g2.get('dimension', '1')}",
                'complexity': self._combine_complexity(g1.get('complexity', 'O(1)'), g2.get('complexity', 'O(1)'))
            }
        return {'error': f"One or both Lie groups not found"}
    
    def _combine_complexity(self, c1: str, c2: str) -> str:
        """Combine complexity classes."""
        if 'O(1)' in [c1, c2]:
            return max(c1, c2, key=lambda x: len(x))
        elif 'O(n!)' in [c1, c2]:
            return 'O(n!)'
        elif 'O(n^2)' in [c1, c2]:
            return 'O(n^2)'
        else:
            return 'O(n log n)'

    def analyze_generalized_operators(self, operator_class: str = None) -> Dict[str, Any]:
        """Analyze any class of operators in a generalized framework."""
        if operator_class and operator_class in self.operator_classes:
            # Analyze specific operator class
            op_class = self.operator_classes[operator_class]
            return {
                'class': operator_class,
                'description': op_class.get('description', ''),
                'examples': op_class.get('examples', []),
                'properties': op_class.get('properties', []),
                'composition': op_class.get('composition', ''),
                'analysis': self._analyze_operator_class_properties(op_class)
            }
        else:
            # Analyze all operator classes
            all_analyses = {}
            for class_name, op_class in self.operator_classes.items():
                all_analyses[class_name] = {
                    'description': op_class.get('description', ''),
                    'examples': op_class.get('examples', []),
                    'properties': op_class.get('properties', []),
                    'composition': op_class.get('composition', ''),
                    'analysis': self._analyze_operator_class_properties(op_class)
                }
            
            return {
                'framework': 'Generalized Operator Analysis',
                'classes': all_analyses,
                'generalization': 'Any operator class can be analyzed using this framework',
                'mathematical_structure': 'Forms a category of operator classes'
            }
    
    def _analyze_operator_class_properties(self, op_class: Dict) -> Dict[str, Any]:
        """Analyze properties of any operator class."""
        composition = op_class.get('composition', '')
        properties = op_class.get('properties', [])
        
        return {
            'is_associative': 'associativity' in properties,
            'is_commutative': composition in ['algebraic', 'functional'],
            'has_identity': 'identity' in properties,
            'has_inverse': 'inverse' in properties,
            'is_closed': 'closure' in properties,
            'preserves_structure': 'structure_preservation' in properties,
            'preserves_semantics': 'semantic_equivalence' in properties,
            'preserves_truth': 'truth_preservation' in properties,
            'complexity_aware': 'complexity' in properties,
            'composition_type': composition
        }
    
    def classify_operator(self, rule_name: str) -> Dict[str, Any]:
        """Classify a rule into operator classes."""
        classifications = {}
        
        for class_name, op_class in self.operator_classes.items():
            examples = op_class.get('examples', [])
            if rule_name in examples:
                classifications[class_name] = {
                    'description': op_class.get('description', ''),
                    'properties': op_class.get('properties', []),
                    'composition': op_class.get('composition', '')
                }
        
        return {
            'rule': rule_name,
            'classifications': classifications,
            'primary_class': list(classifications.keys())[0] if classifications else 'unknown'
        }
    
    def compose_operator_classes(self, class1: str, class2: str) -> Dict[str, Any]:
        """Compose two operator classes."""
        if class1 in self.operator_classes and class2 in self.operator_classes:
            op1 = self.operator_classes[class1]
            op2 = self.operator_classes[class2]
            
            # Cross-class composition
            combined_properties = list(set(op1.get('properties', []) + op2.get('properties', [])))
            combined_examples = list(set(op1.get('examples', []) + op2.get('examples', [])))
            
            return {
                'composition': f"{class1} × {class2}",
                'combined_properties': combined_properties,
                'combined_examples': combined_examples,
                'composition_type': self._determine_composition_type(op1, op2),
                'is_well_defined': self._check_compatibility(op1.get('composition', ''), op2.get('composition', ''))
            }
        return {'error': f"One or both operator classes not found"}
    
    def _determine_composition_type(self, op1: Dict, op2: Dict) -> str:
        """General operator that uses symbol table for composition classification."""
        comp1 = op1.get('composition', '')
        comp2 = op2.get('composition', '')
        
        # Use symbol table for classification
        for classification_name, classification in self.composition_symbol_table.items():
            pattern = classification.get('pattern', '')
            
            # Evaluate pattern against current composition types
            if pattern == "comp1 == comp2" and comp1 == comp2:
                result = classification.get('result', 'Unknown')
                return result.format(comp1=comp1) if '{comp1}' in result else result
                
            elif pattern == "comp1 in ['algebraic', 'functional'] and comp2 in ['algebraic', 'functional']":
                if comp1 in ['algebraic', 'functional'] and comp2 in ['algebraic', 'functional']:
                    return classification.get('result', 'Algebraic-Functional')
                    
            elif pattern == "comp1 in ['logical', 'semantic'] and comp2 in ['logical', 'semantic']":
                if comp1 in ['logical', 'semantic'] and comp2 in ['logical', 'semantic']:
                    return classification.get('result', 'Logical-Semantic')
                    
            elif pattern == "comp1 in ['structural', 'computational'] and comp2 in ['structural', 'computational']":
                if comp1 in ['structural', 'computational'] and comp2 in ['structural', 'computational']:
                    return classification.get('result', 'Structural-Computational')
        
        # Fallback to heterogeneous
        return self.composition_symbol_table.get('heterogeneous', {}).get('result', 'Heterogeneous')
    
    def classify_composition_general(self, comp1: str, comp2: str) -> Dict[str, Any]:
        """General composition classifier using symbol table."""
        classification_result = self._determine_composition_type({'composition': comp1}, {'composition': comp2})
        
        # Find the matching classification in symbol table
        matching_classification = None
        for name, classification in self.composition_symbol_table.items():
            if classification.get('result', '') == classification_result:
                matching_classification = classification
                break
        
        return {
            'composition': f"{comp1} × {comp2}",
            'classification': classification_result,
            'description': matching_classification.get('description', '') if matching_classification else '',
            'examples': matching_classification.get('examples', []) if matching_classification else [],
            'is_compatible': self._check_compatibility(comp1, comp2),
            'symbol_table_used': True
        }
    
    def _check_compatibility(self, comp1: str, comp2: str) -> bool:
        """Check if two composition types are compatible using the compatibility matrix."""
        if comp1 in self.composition_compatibility:
            return comp2 in self.composition_compatibility[comp1]
        return False

    def generate_projection_operators(self, operator_class: str) -> Dict[str, Any]:
        """Generate projection operators for a given operator class."""
        if operator_class in self.operator_classes:
            op_class = self.operator_classes[operator_class]
            properties = op_class.get('properties', [])
            composition = op_class.get('composition', '')
            
            # Generate check operator
            check_operator = f"is_{operator_class}"
            check_operation = {
                'description': f"Check if operator is {operator_class}",
                'projection': operator_class,
                'check_operation': check_operator,
                'transform_operation': f"to_{operator_class}",
                'properties': properties
            }
            
            # Generate transform operator
            transform_operator = f"to_{operator_class}"
            transform_operation = {
                'description': f"Transform to {operator_class} form",
                'projection': operator_class,
                'check_operation': check_operator,
                'transform_operation': transform_operator,
                'target_properties': properties
            }
            
            return {
                'operator_class': operator_class,
                'check_operator': check_operation,
                'transform_operator': transform_operation,
                'properties': properties,
                'composition': composition,
                'projection_generated': True
            }
        return {'error': f"Operator class '{operator_class}' not found"}
    
    def check_operator_projection(self, operator_name: str, target_class: str) -> Dict[str, Any]:
        """Check if an operator projects onto a target class."""
        if target_class in self.operator_classes:
            target_properties = self.operator_classes[target_class].get('properties', [])
            
            # Check if operator has the required properties
            operator_classifications = self.classify_operator(operator_name)
            is_projection = target_class in operator_classifications.get('classifications', {})
            
            return {
                'operator': operator_name,
                'target_class': target_class,
                'is_projection': is_projection,
                'required_properties': target_properties,
                'check_operation': f"is_{target_class}",
                'transform_operation': f"to_{target_class}"
            }
        return {'error': f"Target class '{target_class}' not found"}
    
    def transform_operator_projection(self, operator_name: str, target_class: str) -> Dict[str, Any]:
        """Transform an operator to project onto a target class."""
        check_result = self.check_operator_projection(operator_name, target_class)
        
        if 'error' not in check_result:
            target_properties = self.operator_classes[target_class].get('properties', [])
            
            return {
                'operator': operator_name,
                'target_class': target_class,
                'transformation_applied': check_result['is_projection'],
                'target_properties': target_properties,
                'transform_operation': f"to_{target_class}",
                'projection_successful': check_result['is_projection']
            }
        return check_result
    
    def apply_projection_chain(self, operator_name: str, projection_chain: List[str]) -> Dict[str, Any]:
        """Apply a chain of projections to an operator."""
        chain_results = []
        current_operator = operator_name
        
        for target_class in projection_chain:
            transform_result = self.transform_operator_projection(current_operator, target_class)
            chain_results.append(transform_result)
            
            if transform_result.get('projection_successful', False):
                current_operator = f"{current_operator}_as_{target_class}"
        
        return {
            'original_operator': operator_name,
            'projection_chain': projection_chain,
            'chain_results': chain_results,
            'final_operator': current_operator,
            'all_projections_successful': all(r.get('projection_successful', False) for r in chain_results)
        }

    def generate_from_minimal_span(self, fundamental_type: str, components: List[str]) -> Dict[str, Any]:
        """Generate higher-level constructs from fundamental types."""
        if fundamental_type in self.fundamental_types:
            base_type = self.fundamental_types[fundamental_type]
            
            # Generate based on fundamental type
            if fundamental_type == "pattern":
                return self._generate_pattern_rule(components)
            elif fundamental_type == "replacement":
                return self._generate_replacement_rule(components)
            elif fundamental_type == "condition":
                return self._generate_conditional_rule(components)
            elif fundamental_type == "composition":
                return self._generate_composition_rule(components)
            elif fundamental_type == "projection":
                return self._generate_projection_rule(components)
            else:
                return {'error': f"Unknown fundamental type: {fundamental_type}"}
        return {'error': f"Fundamental type '{fundamental_type}' not found"}
    
    def _generate_pattern_rule(self, components: List[str]) -> Dict[str, Any]:
        """Generate a pattern-based rule."""
        return {
            'type': 'pattern_rule',
            'components': components,
            'generation': 'pattern → rule',
            'description': 'Pattern matching rule',
            'properties': ['matching', 'extraction']
        }
    
    def _generate_replacement_rule(self, components: List[str]) -> Dict[str, Any]:
        """Generate a replacement-based rule."""
        return {
            'type': 'replacement_rule',
            'components': components,
            'generation': 'replacement → rule',
            'description': 'Replacement transformation rule',
            'properties': ['substitution', 'transformation']
        }
    
    def _generate_conditional_rule(self, components: List[str]) -> Dict[str, Any]:
        """Generate a conditional rule."""
        return {
            'type': 'conditional_rule',
            'components': components,
            'generation': 'condition → rule',
            'description': 'Conditional transformation rule',
            'properties': ['evaluation', 'branching']
        }
    
    def _generate_composition_rule(self, components: List[str]) -> Dict[str, Any]:
        """Generate a composition rule."""
        return {
            'type': 'composition_rule',
            'components': components,
            'generation': 'composition → rule',
            'description': 'Composition transformation rule',
            'properties': ['combination', 'ordering']
        }
    
    def _generate_projection_rule(self, components: List[str]) -> Dict[str, Any]:
        """Generate a projection rule."""
        return {
            'type': 'projection_rule',
            'components': components,
            'generation': 'projection → rule',
            'description': 'Projection transformation rule',
            'properties': ['mapping', 'preservation']
        }
    
    def apply_generation_rule(self, rule_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a generation rule to create higher-level constructs."""
        if rule_name in self.generation_rules:
            rule = self.generation_rules[rule_name]
            components = rule.get('components', [])
            
            # Check if all required components are provided
            missing_components = [comp for comp in components if comp not in inputs]
            if missing_components:
                return {'error': f"Missing components: {missing_components}"}
            
            return {
                'generated_type': rule_name,
                'description': rule.get('description', ''),
                'generation': rule.get('generation', ''),
                'components_used': components,
                'inputs': inputs,
                'generation_successful': True
            }
        return {'error': f"Generation rule '{rule_name}' not found"}
    
    def generate_rule_from_pattern_replacement(self, pattern: str, replacement: str) -> Dict[str, Any]:
        """Generate a rule from pattern and replacement using minimal span."""
        return self.apply_generation_rule('rule', {
            'pattern': pattern,
            'replacement': replacement
        })
    
    def generate_transformation_from_rules_condition(self, rules: List[str], condition: str) -> Dict[str, Any]:
        """Generate a transformation from rules and condition using minimal span."""
        return self.apply_generation_rule('transformation', {
            'rules': rules,
            'condition': condition
        })
    
    def generate_operator_from_transformations_composition(self, transformations: List[str], composition: str) -> Dict[str, Any]:
        """Generate an operator from transformations and composition using minimal span."""
        return self.apply_generation_rule('operator', {
            'transformations': transformations,
            'composition': composition
        })
    
    def generate_action_from_operators_projection(self, operators: List[str], projection: str) -> Dict[str, Any]:
        """Generate an action from operators and projection using minimal span."""
        return self.apply_generation_rule('action', {
            'operators': operators,
            'projection': projection
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform a Python file using mathematical code transformer.")
    parser.add_argument("input_file", nargs="?", default="rewriter.py", help="Input Python file to transform (default: rewriter.py)")
    parser.add_argument("-c", "--config", default="rewriter.toml", help="TOML config file (default: rewriter.toml)")
    parser.add_argument("-o", "--output", help="Output file path (default: .out/input_file)")
    parser.add_argument("-a", "--action", help="Transformation action (dynamically loaded from config)")
    parser.add_argument("--arity", help="Analyze transformation arity for given action")
    parser.add_argument("--generalize", action="store_true", help="Show arity generalization analysis")
    parser.add_argument("--lie-group", help="Analyze Lie group transformation")
    parser.add_argument("--compose-groups", nargs=2, metavar=('GROUP1', 'GROUP2'), help="Compose two Lie groups")
    
    args = parser.parse_args()
    
    # Load actions dynamically from the specified config file
    available_actions = []
    try:
        with open(args.config, 'r') as f: config = tomllib.loads(f.read())
        actions = config.get('actions', {})
        action_combinations = config.get('action_combinations', {})
        action_intersections = config.get('action_intersections', {})
        available_actions = list(actions.keys()) + list(action_combinations.keys()) + list(action_intersections.keys())
    except:
        available_actions = ["simplify", "optimize", "refactor", "minimal", "full"]  # fallback
    
    # Update the action argument with dynamic choices
    if args.action and args.action not in available_actions:
        print(f"Error: Action '{args.action}' not found in {args.config}")
        print(f"Available actions: {', '.join(available_actions)}")
        sys.exit(1)
    
    try:
        with open(args.input_file, 'r') as f: original_code = f.read()
        
        transformer = M()
        transformer.load_toml(args.config, args.action)
        
        # Handle arity analysis
        if args.arity:
            analysis = transformer.analyze_transformation_arity(args.arity)
            if 'error' in analysis:
                print(f"❌ {analysis['error']}")
            else:
                print(f"🔬 Arity Analysis: {args.arity}")
                print(f"   Arity: {analysis['arity']} ({analysis['arity_type']})")
                print(f"   Type: {analysis['type']}")
                print(f"   Complexity: {analysis['complexity']}")
                print(f"   Properties: {', '.join(analysis['properties'])}")
                print(f"   Base Case: {analysis['is_base_case']}")
                print(f"   Composite: {analysis['is_composite']}")
                print(f"   Description: {analysis['description']}")
            sys.exit(0)
        
        # Handle generalization analysis
        if args.generalize:
            generalization = transformer.generalize_by_arity()
            print(f"📊 {generalization['generalization']}")
            print(f"   Base Case: {generalization['base_case']}")
            print(f"   Induction: {generalization['induction']}")
            print(f"   Structure: {generalization['mathematical_structure']}")
            print("\n📈 Arity Distribution:")
            for arity_type, actions in generalization['arity_distribution'].items():
                print(f"   {arity_type}: {len(actions)} transformations")
                for action in actions:
                    print(f"     - {action['action']} (arity {action['arity']})")
            sys.exit(0)
        
        # Normal transformation
        transformed_code = transformer.transform(original_code)
        
        if args.output:
            output_file = args.output
        else:
            import os
            os.makedirs('.out', exist_ok=True)
            output_file = f".out/{args.input_file}"
        
        with open(output_file, 'w') as f: f.write(transformed_code)
        print(f"💾 Saved: {output_file}")
        
        analysis = transformer.analyze(original_code, transformed_code)
        print(f"📊 Analysis: {analysis['original_lines']} → {analysis['transformed_lines']} lines, {analysis['rules_applied']} rules applied")
        print("🎯 TRANSFORMATION COMPLETE!")
        
    except Exception as e: print(f"Error: {e}")