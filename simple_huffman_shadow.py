#!/usr/bin/env python3
"""
Simple Huffman Shadow System
A self-modifying code system using Huffman compression
"""

import heapq

import hashlib
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

def table() -> List[str]:
    return []

@dataclass
class HuffmanNode:
    """Represents a node in the Huffman tree"""
    char: str
    freq: int
    left: Optional['HuffmanNode'] = None
    right: Optional['HuffmanNode'] = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data: str) -> HuffmanNode:
    """Build Huffman tree from input data"""
    # Count frequencies
    freq = {}
    for char in data:
        freq[char] = freq.get(char, 0) + 1
        
    # Create leaf nodes
    heap = [HuffmanNode(char, freq) for char, freq in freq.items()]
    heapq.heapify(heap)
    
    # Build tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        internal = HuffmanNode('', left.freq + right.freq, left, right)
        heapq.heappush(heap, internal)
        
    return heap[0] if heap else HuffmanNode('', 0)

def generate_codes(root: Optional[HuffmanNode], code: str = "", codes: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Generate Huffman codes from tree"""
    if codes is None:
        codes = {}
        
    if root is None:
        return codes
        
    if root.left is None and root.right is None:
        codes[root.char] = code
        return codes
        
    generate_codes(root.left, code + "0", codes)
    generate_codes(root.right, code + "1", codes)
    
    return codes

def compress_data(data: str) -> Tuple[str, Dict[str, str]]:
    """Compress data using Huffman coding"""
    if not data:
        return "", {}
        
    tree = build_huffman_tree(data)
    codes = generate_codes(tree)
    
    # Compress data
    compressed = ''.join(codes[char] for char in data)
    
    return compressed, codes


def decompress():
    return decompress_data(*table())

def decompress_data(compressed_data: str, codes: Dict[str, str]) -> str:
    reverse_codes = {v: k for k, v in codes.items()}
    result = ""
    current_code = ""
    
    for bit in compressed_data:
        current_code += bit
        if current_code in reverse_codes:
            result += reverse_codes[current_code]
            current_code = ""
    
    return result


def generate_python_file(data: str, filename: str, generation: int = 0) -> str:
    """Return a *fully self-hosting* Python script.

    We read *our own* source file, update the `table()` definition to reflect
    `data`, and return the modified source.  Because we copy the entire file,
    every generation carries forward *all* regeneration logic automatically.
    """
    import re

    template = Path(__file__).read_text()

    # Build constant list for the new table
    _, codes = compress_data(data)
    content_hash = hashlib.md5(data.encode()).hexdigest()[:8]
    constants = []
    for char, code in codes.items():
        symbol_name = f"SYMBOL_{content_hash}_{ord(char):03d}"
        constants.append(f"{symbol_name} = '{code}'")

    new_table_block = "def table():\n    return " + str(constants)

    # Replace the first occurrence of `def table(): ... return ...` in template
    new_source = re.sub(
        r"def table\(\):[\s\S]*?return .*?\n",
        new_table_block + "\n",
        template,
        count=1,
    )

    return new_source

def run_regeneration_cycle(main_file: str, new_data: str, generation: int):
    """Run a regeneration cycle"""
    print(f"Shadow Monitor - Generation {{generation}}")
    print(f"Regenerating {{main_file}} with new data...")
    
    # Import helper to build the new source
    from simple_huffman_shadow import generate_python_file
    import py_compile, sys

    # Pre-compute expected metadata for sanity-check
    _, codes = compress_data(new_data)
    expected_table_size = len(codes)

    # Generate new file content
    new_content = generate_python_file(new_data, main_file, generation + 1)

    # Write the new file
    with open(main_file, 'w') as f:
        f.write(new_content)

    # ---- Step 2: ensure the newly generated code compiles ----
    try:
        pyc_path = py_compile.compile(main_file, doraise=True)
        assert pyc_path is not None
    except py_compile.PyCompileError as err:
        print("[shadow] Newly generated code failed to compile", file=sys.stderr)
        print(err, file=sys.stderr)
        return

    # ---- Step 3: load the compiled bytecode and verify table size ----
    import importlib.machinery, types
    loader = importlib.machinery.SourcelessFileLoader("generated_module", pyc_path)
    module = types.ModuleType(loader.name)
    loader.exec_module(module)

    actual_size = len(module.table())
    if actual_size != expected_table_size:
        print("[shadow] Sanity check failed: table size mismatch", file=sys.stderr)
        return

    print("[shadow] Sanity checks passed; generation", generation + 1, "is valid.")

    # Add a parameter to main that accepts the previous shadow file
    # Check that file compiles to bytecode
    # Spawn a new process to run the file
    # If the comes back with the new symbol table, we exit and set child as parent


def header():
    return f'#!/usr/bin/env python3\n"""\n{globals().__doc__}\n"""\n\n'

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 {{sys.argv[0]}} <new_data> [generation]", file=sys.stderr)
        sys.exit(1)

    new_data = sys.argv[1]
    generation = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run_regeneration_cycle(__file__, new_data, generation)

if __name__ == "__main__":
    main()
