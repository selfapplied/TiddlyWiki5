#!/usr/bin/env python3
"""
Minimal Python Code Rewriter
Uses minimal span of fundamental types to generate all transformations.
"""

import argparse
import tomllib
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class Rule:
    pattern: str
    replacement: str
    type: str = "fundamental"
    weight: float = 1.0

class MinimalRewriter:
    def __init__(self):
        self.rules = []
        self.fundamental_types = {}
        self.generation_rules = {}
        self.history = []
    
    def load_toml(self, config_file: str) -> None:
        """Load minimal configuration."""
        try:
            with open(config_file, 'r') as f:
                config = tomllib.loads(f.read())
            
            self.fundamental_types = config.get('fundamental_types', {})
            self.generation_rules = config.get('generation_rules', {})
            
            # Load rules using minimal span generation
            if 'rules' in config:
                for r in config['rules']:
                    rule = Rule(
                        pattern=r['pattern'],
                        replacement=r['replacement'],
                        type=r.get('type', 'fundamental'),
                        weight=r.get('weight', 1.0)
                    )
                    self.rules.append(rule)
                print(f"📋 Loaded {len(self.rules)} rules from minimal span")
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def transform(self, code: str) -> str:
        """Apply transformations using minimal span."""
        transformed = code
        
        for rule in self.rules:
            if rule.pattern in transformed:
                transformed = transformed.replace(rule.pattern, rule.replacement)
                self.history.append(rule.type)
        
        return transformed
    
    def analyze(self, original: str, transformed: str) -> Dict:
        """Minimal analysis."""
        return {
            'original_lines': len(original.split('\n')),
            'transformed_lines': len(transformed.split('\n')),
            'rules_applied': len(self.history),
            'fundamental_types_used': list(set(self.history))
        }
    
    def generate_from_minimal_span(self, fundamental_type: str, components: Dict) -> Dict:
        """Generate using minimal span."""
        if fundamental_type in self.fundamental_types:
            return {
                'type': fundamental_type,
                'components': components,
                'generation': f"{fundamental_type} → generated",
                'minimal_span': True
            }
        return {'error': f"Unknown fundamental type: {fundamental_type}"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Python code rewriter using fundamental types.")
    parser.add_argument("input_file", nargs="?", default="rewriter.py", help="Input file")
    parser.add_argument("-c", "--config", default="rewriter.toml", help="Config file")
    parser.add_argument("-o", "--output", help="Output file")
    
    args = parser.parse_args()
    
    try:
        with open(args.input_file, 'r') as f:
            original_code = f.read()
        
        rewriter = MinimalRewriter()
        rewriter.load_toml(args.config)
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
        print("🎯 MINIMAL TRANSFORMATION COMPLETE!")
        
    except Exception as e:
        print(f"Error: {e}") 