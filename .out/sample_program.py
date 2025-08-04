#!/usr/bin/env python3
"""
Sample program demonstrating various Python patterns
that can be optimized by the closure-based rewriter.
"""

import math
import logging
import random

# Lambda functions that can be converted to def statements
square = def anonymous(x): return x ** 2
cube = def anonymous(x): return x ** 3
is_even = def anonymous(x): return x % 2 == 0

# List comprehensions that can be optimized
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squares = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

# Print statements that can be converted to logging
logging.info("Starting optimization demo")
logging.info(f"Numbers: {numbers}")
logging.info(f"Squares: {squares}")

# Assert statements that can be converted to proper validation
if not len(numbers) > 0, "Numbers list cannot be empty"
if not all(isinstance(x, int) for x in numbers), "All numbers must be integers"

# Inefficient loops that can be vectorized
def calculate_fibonacci(n):
    """Calculate Fibonacci numbers inefficiently."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Unused imports



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
    logging.basicConfig(level=logging.INFO)
    logging.info("Processing data...")
    processed = process_data(numbers)
    logging.info(f"Processed: {processed}")
    
    # More lambda functions
    double = def anonymous(x): return x * 2
    triple = def anonymous(x): return x * 3
    
    # More assertions
    if not processed is not None
    if not len(processed) == len(numbers)
    
    logging.info("Demo complete!") 