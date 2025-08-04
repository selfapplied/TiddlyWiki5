#!/usr/bin/env python3
"""
Sample program demonstrating various Python patterns
that can be optimized by the closure-based rewriter.
"""

import math
import random

# Lambda functions that can be converted to def statements
square = lambda x: x ** 2
cube = lambda x: x ** 3
is_even = lambda x: x % 2 == 0

# List comprehensions that can be optimized
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squares = [x**2 for x in numbers]
evens = [x for x in numbers if x % 2 == 0]

# Print statements that can be converted to logging
print("Starting optimization demo")
print(f"Numbers: {numbers}")
print(f"Squares: {squares}")

# Assert statements that can be converted to proper validation
assert len(numbers) > 0, "Numbers list cannot be empty"
assert all(isinstance(x, int) for x in numbers), "All numbers must be integers"

# Inefficient loops that can be vectorized
def calculate_fibonacci(n):
    """Calculate Fibonacci numbers inefficiently."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Unused imports
import os  # This import is not used
import sys  # This import is not used

# Complex nested functions that can be simplified
def process_data(data):
    """Process data with nested complexity."""
    result = []
    for item in data:
        if isinstance(item, (int, float)):
            if item > 0:
                result.append(math.sqrt(item))
            else:
                result.append(0)
        else:
            result.append(None)
    return result

# Main execution
if __name__ == "__main__":
    print("Processing data...")
    processed = process_data(numbers)
    print(f"Processed: {processed}")
    
    # More lambda functions
    double = lambda x: x * 2
    triple = lambda x: x * 3
    
    # More assertions
    assert processed is not None
    assert len(processed) == len(numbers)
    
    print("Demo complete!") 