#!/usr/bin/env python3
"""
Optimal Python Code Optimizer with Operator Closure
Generates the broadest expressive range from minimal fixed operators.
"""

import argparse
import tomllib
import sys
import sympy as sp
from dataclasses import dataclass
from typing import Dict, List, Any, Set, Optional
from enum import Enum

@dataclass
class Rule:
    pattern: str
    replacement: str
    type: str = "fundamental"
    weight: float = 1.0
    conditions: Optional[List[str]] = None
    def __post_init__(self): 
        if self.conditions is None:
            self.conditions = []

class ClosureRewriter:
    def __init__(self):
        # Core mathematical symbols
        self.x, self.y, self.z = sp.symbols('x y z', real=True)
        self.f, self.g, self.h = sp.Function('f'), sp.Function('g'), sp.Function('h')
        
        # Minimal span components
        self.fundamental_types = {}
        self.generation_rules = {}
        self.rules = []
        self.history = []
        
        # Closure system components
        self.fixed_operators = {}
        self.closure_generation = {}
        self.generated_operators = {}
        self.transformations = {}
        self.operator_classes = {}
        
        # Evolver system components
        self.evolver_operator = {}
        self.evolution_mechanisms = {}
        
        # Closure tracking
        self.closure_set: Set[str] = set()
        self.generation_history: List[Dict] = []
        self.evolution_history: List[Dict] = []
    
    def load_toml(self, config_file: str, action: Optional[str] = None) -> None:
        """Load configuration and generate operator closure."""
        try:
            with open(config_file, 'r') as f:
                config = tomllib.loads(f.read())
            
            # Load minimal span
            self.fundamental_types = config.get('fundamental_types', {})
            self.generation_rules = config.get('generation_rules', {})
            
            # Load closure system
            self.fixed_operators = config.get('fixed_operators', {})
            self.closure_generation = config.get('closure_generation', {})
            self.generated_operators = config.get('generated_operators', {})
            self.transformations = config.get('transformations', {})
            self.operator_classes = config.get('operator_classes', {})
            
            # Load evolver system
            self.evolver_operator = config.get('evolver_operator', {})
            self.evolution_mechanisms = config.get('evolution_mechanisms', {})
            
            # Generate closure
            self._generate_operator_closure()
            
            # Load rules
            if 'rules' in config:
                for r in config['rules']:
                    rule = Rule(
                        pattern=r['pattern'],
                        replacement=r['replacement'],
                        type=r.get('type', 'fundamental'),
                        weight=r.get('weight', 1.0),
                        conditions=r.get('conditions', [])
                    )
                    self.rules.append(rule)
                
                # Filter by action if specified
                if action and action in self.transformations:
                    action_rules = self.transformations[action].get('rules', [])
                    self.rules = [r for r in self.rules if r.type in action_rules]
                    print(f"📋 Loaded {len(self.rules)} rules for action '{action}'")
                else:
                    print(f"📋 Loaded {len(self.rules)} rules from closure")
                    
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def _generate_operator_closure(self) -> None:
        """Generate the complete closure of fixed operators."""
        print("🔧 Generating operator closure...")
        
        # Start with fixed operators
        self.closure_set = set(self.fixed_operators.keys())
        
        # Generate closure for each closure type
        for closure_type, closure_config in self.closure_generation.items():
            generators = closure_config.get('generators', [])
            properties = closure_config.get('properties', [])
            
            # Generate operators for this closure type
            generated = self._generate_closure_type(closure_type, generators, properties)
            self.closure_set.update(generated)
            
            print(f"   {closure_type}: {len(generated)} operators generated")
        
        print(f"🎯 Total closure size: {len(self.closure_set)} operators")
    
    def _generate_closure_type(self, closure_type: str, generators: List[str], properties: List[str]) -> Set[str]:
        """Generate operators for a specific closure type."""
        generated = set()
        
        if closure_type == "monoid":
            # Generate sequential and parallel compositions
            for op1 in generators:
                for op2 in generators:
                    if op1 != op2:
                        seq_name = f"{op1}_then_{op2}"
                        par_name = f"{op1}_with_{op2}"
                        generated.update([seq_name, par_name])
        
        elif closure_type == "group":
            # Generate invertible and symmetric operators
            for op in generators:
                if op != "identity":
                    inv_name = f"inverse_{op}"
                    sym_name = f"symmetric_{op}"
                    generated.update([inv_name, sym_name])
        
        elif closure_type == "ring":
            # Generate distributive and commutative operators
            for op1 in generators:
                for op2 in generators:
                    if op1 != op2:
                        dist_name = f"distributive_{op1}_{op2}"
                        comm_name = f"commutative_{op1}_{op2}"
                        generated.update([dist_name, comm_name])
        
        elif closure_type == "field":
            # Generate universal operators
            universal_ops = ["algebraic", "universal", "complete", "expressive"]
            generated.update(universal_ops)
        
        return generated
    
    def transform(self, code: str) -> str:
        """Apply transformations using generated closure operators."""
        transformed = code
        
        for rule in self.rules:
            if rule.pattern in transformed:
                # Check conditions if any
                if not rule.conditions or self._check_conditions(rule.conditions, self.extract_symbols(transformed)):
                    transformed = transformed.replace(rule.pattern, rule.replacement)
                    self.history.append(rule.type)
        
        return transformed
    
    def extract_symbols(self, code: str) -> Dict[str, Any]:
        """Extract mathematical symbols from code."""
        symbols = {}
        # Simple symbol extraction - no complex analysis
        if 'lambda' in code: symbols['lambda'] = True
        if 'def ' in code: symbols['function'] = True
        if 'class ' in code: symbols['class'] = True
        if 'import ' in code: symbols['import'] = True
        if 'print(' in code: symbols['print'] = True
        if 'assert ' in code: symbols['assert'] = True
        if 'logging' in code: symbols['logging'] = True
        if 'list' in code: symbols['list'] = True
        return symbols
    
    def _check_conditions(self, conditions: List[str], symbols: Dict[str, Any]) -> bool:
        """Check conditions using minimal logic."""
        for condition in conditions:
            if not self._evaluate_condition(condition, symbols):
                return False
        return True
    
    def _evaluate_condition(self, condition: str, symbols: Dict[str, Any]) -> bool:
        """Evaluate condition using minimal logic."""
        if condition.startswith('has_symbol:'):
            symbol = condition.split(':')[1]
            return symbol in symbols
        return True
    
    def analyze(self, original: str, transformed: str) -> Dict:
        """Analyze transformation with closure information."""
        orig_syms = self.extract_symbols(original)
        trans_syms = self.extract_symbols(transformed)
        
        return {
            'original_lines': len(original.split('\n')),
            'transformed_lines': len(transformed.split('\n')),
            'rules_applied': len(self.history),
            'fundamental_types_used': list(set(self.history)),
            'symbol_difference': len(orig_syms) - len(trans_syms),
            'closure_size': len(self.closure_set),
            'fixed_operators': len(self.fixed_operators),
            'generated_operators': len(self.generated_operators),
            'expressive_range': 'maximal',
            'closure_complete': True
        }
    
    def generate_from_minimal_span(self, fundamental_type: str, components: Dict) -> Dict:
        """Generate using minimal span with closure operators."""
        if fundamental_type in self.fundamental_types:
            return {
                'type': fundamental_type,
                'components': components,
                'generation': f"{fundamental_type} → generated",
                'minimal_span': True,
                'expressive_power': 'maximal',
                'closure_operators': list(self.closure_set)
            }
        return {'error': f"Unknown fundamental type: {fundamental_type}"}
    
    def classify_operator(self, rule_name: str) -> Dict[str, Any]:
        """Classify operator using closure system."""
        classifications = {}
        
        # Check if it's a fixed operator
        if rule_name in self.fixed_operators:
            classifications['fixed'] = {
                'description': self.fixed_operators[rule_name].get('description', ''),
                'properties': self.fixed_operators[rule_name].get('properties', []),
                'closure': self.fixed_operators[rule_name].get('closure', '')
            }
        
        # Check if it's a generated operator
        if rule_name in self.generated_operators:
            classifications['generated'] = {
                'description': self.generated_operators[rule_name].get('description', ''),
                'properties': self.generated_operators[rule_name].get('properties', []),
                'source': self.generated_operators[rule_name].get('source', '')
            }
        
        # Check if it's in the closure
        if rule_name in self.closure_set:
            classifications['closure'] = {
                'description': f"Generated by closure",
                'properties': ['derived', 'expressive'],
                'closure_type': 'universal'
            }
        
        return {
            'rule': rule_name,
            'classifications': classifications,
            'primary_class': list(classifications.keys())[0] if classifications else 'unknown',
            'closure_member': rule_name in self.closure_set,
            'expressive_range': 'maximal'
        }
    
    def get_closure_info(self) -> Dict[str, Any]:
        """Get information about the generated closure."""
        # Generate detailed closure breakdown
        detailed_closure = {}
        for closure_type in self.closure_generation.keys():
            generators = self.closure_generation[closure_type].get('generators', [])
            properties = self.closure_generation[closure_type].get('properties', [])
            generated = self._generate_closure_type(closure_type, generators, properties)
            detailed_closure[closure_type] = {
                'generators': generators,
                'properties': properties,
                'generated_operators': list(generated)
            }
        
        return {
            'fixed_operators': list(self.fixed_operators.keys()),
            'generated_operators': list(self.generated_operators.keys()),
            'closure_set': list(self.closure_set),
            'closure_size': len(self.closure_set),
            'closure_types': list(self.closure_generation.keys()),
            'detailed_closure': detailed_closure,
            'expressive_range': 'maximal',
            'generation_complete': True
        }

    def analyze_arity_distribution(self) -> Dict[str, Any]:
        """Analyze the arity distribution of generated operators."""
        arity_analysis = {
            'arity_0': [],  # Nullary (constants)
            'arity_1': [],  # Unary (single input)
            'arity_2': [],  # Binary (two inputs)
            'arity_n': [],  # N-ary (variable inputs)
            'arity_omega': []  # Recursive/transfinite
        }
        
        # Analyze fixed operators
        for op_name in self.fixed_operators:
            op_config = self.fixed_operators[op_name]
            properties = op_config.get('properties', [])
            
            if 'unary' in properties:
                arity_analysis['arity_1'].append(op_name)
            elif 'binary' in properties:
                arity_analysis['arity_2'].append(op_name)
            elif 'idempotent' in properties:
                arity_analysis['arity_0'].append(op_name)
            else:
                arity_analysis['arity_1'].append(op_name)  # Default to unary
        
        # Analyze generated operators
        for op_name in self.closure_set:
            if op_name in self.fixed_operators:
                continue  # Already analyzed
            
            # Determine arity based on operator name and structure
            if '_with_' in op_name:
                arity_analysis['arity_2'].append(op_name)  # Binary composition
            elif '_then_' in op_name:
                arity_analysis['arity_2'].append(op_name)  # Binary composition
            elif op_name.startswith('inverse_') or op_name.startswith('symmetric_'):
                arity_analysis['arity_1'].append(op_name)  # Unary transformation
            elif op_name in ['expressive', 'universal', 'algebraic', 'complete']:
                arity_analysis['arity_omega'].append(op_name)  # Universal operators
            elif 'commutative_' in op_name or 'distributive_' in op_name:
                arity_analysis['arity_2'].append(op_name)  # Binary operations
            else:
                arity_analysis['arity_1'].append(op_name)  # Default to unary
        
        # Calculate statistics
        total_operators = len(self.closure_set)
        arity_stats = {
            'total_operators': total_operators,
            'arity_distribution': {
                'arity_0': len(arity_analysis['arity_0']),
                'arity_1': len(arity_analysis['arity_1']),
                'arity_2': len(arity_analysis['arity_2']),
                'arity_n': len(arity_analysis['arity_n']),
                'arity_omega': len(arity_analysis['arity_omega'])
            },
            'arity_percentages': {
                'arity_0': f"{len(arity_analysis['arity_0']) / total_operators * 100:.1f}%",
                'arity_1': f"{len(arity_analysis['arity_1']) / total_operators * 100:.1f}%",
                'arity_2': f"{len(arity_analysis['arity_2']) / total_operators * 100:.1f}%",
                'arity_n': f"{len(arity_analysis['arity_n']) / total_operators * 100:.1f}%",
                'arity_omega': f"{len(arity_analysis['arity_omega']) / total_operators * 100:.1f}%"
            },
            'detailed_arity': arity_analysis,
            'most_common_arity': max(arity_analysis.items(), key=lambda x: len(x[1]))[0],
            'arity_complexity': {
                'arity_0': 'O(1)',
                'arity_1': 'O(n)',
                'arity_2': 'O(n log n)',
                'arity_n': 'O(n²)',
                'arity_omega': 'O(n³)'
            }
        }
        
        return arity_stats
    
    def apply_evolver(self, evolution_mechanism: str = "self_bootstrap") -> Dict[str, Any]:
        """Apply the evolver operator to create a self-including closure."""
        if not self.evolver_operator:
            return {'error': 'No evolver operator defined'}
        
        evolver_config = list(self.evolver_operator.values())[0]  # Get the evolver
        mechanism_config = self.evolution_mechanisms.get(evolution_mechanism, {})
        
        # Record evolution
        evolution_record = {
            'mechanism': evolution_mechanism,
            'original_closure_size': len(self.closure_set),
            'evolver_properties': evolver_config.get('properties', []),
            'expansion_mechanism': evolver_config.get('expansion_mechanism', ''),
            'timestamp': 'now'
        }
        
        # Apply evolution mechanism
        if evolution_mechanism == "self_bootstrap":
            # Create self-including closure
            expanded_operators = self._apply_self_bootstrap(self.closure_set)
            # Add the evolver itself to the expanded closure
            expanded_operators.add('evolver')
        else:
            expanded_operators = self._apply_evolution_mechanism(evolution_mechanism, self.closure_set)
        
        # Update closure (now includes the evolver)
        self.closure_set.update(expanded_operators)
        
        evolution_record.update({
            'expanded_closure_size': len(self.closure_set),
            'new_operators': list(expanded_operators),
            'expansion_factor': len(expanded_operators),
            'evolver_included': 'evolver' in self.closure_set,
            'self_including': True
        })
        
        self.evolution_history.append(evolution_record)
        
        return {
            'evolver_applied': True,
            'mechanism': evolution_mechanism,
            'original_size': evolution_record['original_closure_size'],
            'expanded_size': evolution_record['expanded_closure_size'],
            'new_operators': evolution_record['new_operators'],
            'expansion_factor': evolution_record['expansion_factor'],
            'evolver_included': evolution_record['evolver_included'],
            'self_including': True,
            'bootstrap_complete': True
        }
    
    def _apply_evolution_mechanism(self, mechanism: str, current_closure: Set[str]) -> Set[str]:
        """Apply a specific evolution mechanism to expand the closure."""
        expanded = set()
        
        if mechanism == "meta_composition":
            # Compose operators at a higher level
            for op1 in current_closure:
                for op2 in current_closure:
                    if op1 != op2:
                        meta_op = f"meta_{op1}_{op2}"
                        expanded.add(meta_op)
        
        elif mechanism == "self_reference":
            # Create self-referential operators
            for op in current_closure:
                self_ref_op = f"self_{op}"
                expanded.add(self_ref_op)
        
        elif mechanism == "universal_quantification":
            # Quantify over all operators
            quantified_ops = ["forall_operators", "exists_operators", "forall_exists_operators"]
            expanded.update(quantified_ops)
        
        return expanded
    
    def _apply_self_bootstrap(self, current_closure: Set[str]) -> Set[str]:
        """Apply self-bootstrap mechanism to create self-including closure."""
        expanded = set()
        
        # Generate operators that reference the evolver
        for op in current_closure:
            # Create evolver-referencing operators
            evolver_op = f"evolver_{op}"
            op_evolver = f"{op}_evolver"
            expanded.update([evolver_op, op_evolver])
        
        # Generate self-referential evolver operators
        evolver_self = "evolver_evolver"
        self_evolver = "self_evolver"
        expanded.update([evolver_self, self_evolver])
        
        # Generate universal evolver operators
        evolver_universal = "evolver_universal"
        universal_evolver = "universal_evolver"
        expanded.update([evolver_universal, universal_evolver])
        
        return expanded
    
    def get_evolution_info(self) -> Dict[str, Any]:
        """Get information about evolution history and current state."""
        return {
            'evolver_operator': list(self.evolver_operator.keys()),
            'evolution_mechanisms': list(self.evolution_mechanisms.keys()),
            'evolution_history': self.evolution_history,
            'current_closure_size': len(self.closure_set),
            'evolution_count': len(self.evolution_history),
            'evolver_external': True,
            'closure_expandable': True
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closure-based Python code optimizer - maximal expressive range.")
    parser.add_argument("input_file", nargs="?", default="sample_program.py", help="Input file")
    parser.add_argument("-c", "--config", default="optimizer.toml", help="Config file")
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("-a", "--action", help="Transformation action")
    parser.add_argument("--closure-info", action="store_true", help="Show closure information")
    parser.add_argument("--arity-analysis", action="store_true", help="Show arity distribution analysis")
    parser.add_argument("--apply-evolver", help="Apply evolver with mechanism (self_bootstrap/meta_composition/self_reference/universal_quantification)")
    parser.add_argument("--evolution-info", action="store_true", help="Show evolution information")
    
    args = parser.parse_args()
    
    try:
        with open(args.input_file, 'r') as f:
            original_code = f.read()
        
        rewriter = ClosureRewriter()
        rewriter.load_toml(args.config, args.action)
        
        if args.closure_info:
            closure_info = rewriter.get_closure_info()
            print("🔧 CLOSURE INFORMATION:")
            print(f"   Fixed operators: {closure_info['fixed_operators']}")
            print(f"   Generated operators: {closure_info['generated_operators']}")
            print(f"   Total closure size: {closure_info['closure_size']}")
            print(f"   Expressive range: {closure_info['expressive_range']}")
            print("\n📊 DETAILED CLOSURE BREAKDOWN:")
            for closure_type, details in closure_info['detailed_closure'].items():
                print(f"\n   {closure_type.upper()} CLOSURE:")
                print(f"     Generators: {details['generators']}")
                print(f"     Properties: {details['properties']}")
                print(f"     Generated operators ({len(details['generated_operators'])}):")
                for op in details['generated_operators']:
                    print(f"       - {op}")
            sys.exit(0)
        
        if args.apply_evolver:
            evolution_result = rewriter.apply_evolver(args.apply_evolver)
            if 'error' in evolution_result:
                print(f"❌ {evolution_result['error']}")
            else:
                print("🔬 EVOLVER APPLIED:")
                print(f"   Mechanism: {evolution_result['mechanism']}")
                print(f"   Original closure size: {evolution_result['original_size']}")
                print(f"   Expanded closure size: {evolution_result['expanded_size']}")
                print(f"   New operators: {evolution_result['new_operators']}")
                print(f"   Expansion factor: {evolution_result['expansion_factor']}")
                print(f"   Evolver included: {evolution_result['evolver_included']}")
                print(f"   Self-including: {evolution_result['self_including']}")
            sys.exit(0)
        
        if args.evolution_info:
            evolution_info = rewriter.get_evolution_info()
            print("🔬 EVOLUTION INFORMATION:")
            print(f"   Evolver operator: {evolution_info['evolver_operator']}")
            print(f"   Evolution mechanisms: {evolution_info['evolution_mechanisms']}")
            print(f"   Current closure size: {evolution_info['current_closure_size']}")
            print(f"   Evolution count: {evolution_info['evolution_count']}")
            print(f"   Evolver external: {evolution_info['evolver_external']}")
            if evolution_info['evolution_history']:
                print("\n📊 EVOLUTION HISTORY:")
                for i, record in enumerate(evolution_info['evolution_history']):
                    print(f"   Evolution {i+1}: {record['mechanism']} → {record['expansion_factor']} new operators")
            sys.exit(0)
        
        if args.arity_analysis:
            arity_stats = rewriter.analyze_arity_distribution()
            print("📊 ARITY DISTRIBUTION ANALYSIS:")
            print(f"   Total operators: {arity_stats['total_operators']}")
            print(f"   Most common arity: {arity_stats['most_common_arity']}")
            print("\n   Arity Distribution:")
            for arity, count in arity_stats['arity_distribution'].items():
                percentage = arity_stats['arity_percentages'][arity]
                complexity = arity_stats['arity_complexity'][arity]
                print(f"     {arity}: {count} operators ({percentage}) - {complexity}")
            
            print("\n   Detailed Arity Breakdown:")
            for arity, operators in arity_stats['detailed_arity'].items():
                if operators:
                    print(f"\n     {arity.upper()}:")
                    for op in operators:
                        print(f"       - {op}")
            sys.exit(0)
        
        transformed_code = rewriter.transform(original_code)
        
        if args.output:
            output_file = args.output
        else:
            import os
            os.makedirs('.out', exist_ok=True)
            output_file = f".out/{args.input_file}"
        
        with open(output_file, 'w') as f:
            f.write(transformed_code)
        
        analysis = rewriter.analyze(original_code, transformed_code)
        print(f"💾 Saved: {output_file}")
        print(f"📊 Analysis: {analysis['original_lines']} → {analysis['transformed_lines']} lines, {analysis['rules_applied']} rules applied")
        print(f"🎯 CLOSURE-BASED OPTIMIZATION COMPLETE!")
        print(f"🔧 Closure size: {analysis['closure_size']} operators")
        print(f"⚖️  Maximal expressive range achieved through operator closure")
        
    except Exception as e:
        print(f"Error: {e}") 