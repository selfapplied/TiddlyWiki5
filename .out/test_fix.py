#!/usr/bin/env python3
"""
Test file for mathematical measurement transformations:
- Order bounds (for i in range)
- Min/max analysis (max function)
- Countable sets (len function)
- Numerical validation (if x > 0)
- Metric spaces (abs function)
"""

# import json  # Unused import removed
def process_data(data):
    print("Processing data...")
    assert data is not None
    return [x * 2 for x in data if x > 0]

def analyze_measurements():
    # Order bounds
    for i in range(0, 10):
        print(f"Processing item {i}")
    
    # Min/max analysis
    values = [1, 2, 3, 4, 5]
    maximum = max(values)
    print(f"Maximum value: {maximum}")
    
    # Countable sets
    data_length = len(values)
    print(f"Data length: {data_length}")
    
    # Numerical validation
    x = 5
    if x > 0 and isinstance(x, (int, float)):
        print("Positive number")
    
    # Metric spaces
    distance = abs(10 - 5)
    print(f"Distance: {distance}")

def lambda_example():
    return def anonymous(x): return x + 1

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    result = process_data(data)
    print(f"Result: {result}")
    
    analyze_measurements() 