#!/usr/bin/env python3
"""Test zeta function analysis to see RH progress"""

import numpy as np
from rh import RHIntegerAnalyzer, PascalKernel, GyroscopeLoss, AngularVelocity, QuantitativeGapAnalyzer

def test_zeta_behavior():
    """Test zeta function behavior on and off critical line"""
    print("🔬 Testing Zeta Function Behavior")
    print("=" * 50)
    
    # Known RH zeros (first few)
    known_zeros = [
        complex(0.5, 14.134725),
        complex(0.5, 21.022040),
        complex(0.5, 25.010858),
        complex(0.5, 30.424876),
        complex(0.5, 32.935062)
    ]
    
    print(f"Using {len(known_zeros)} known RH zeros")
    
    # Test points on critical line
    print("\n1. Testing ON Critical Line (Re(s) = 0.5):")
    critical_points = [
        complex(0.5, 14.0),
        complex(0.5, 20.0),
        complex(0.5, 25.0),
        complex(0.5, 30.0)
    ]
    
    for s in critical_points:
        try:
            # Create analyzer
            analyzer = RHIntegerAnalyzer(depth=3)  # N = 9
            
            # Test coefficients (simulate zeta behavior)
            coeffs = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
            
            result = analyzer.analyze_point(s, coeffs)
            
            print(f"   Point {s}:")
            print(f"     - Locked: {result['is_locked']}")
            print(f"     - Gap: {result['gap']}")
            print(f"     - Best action: {result['best_action']}")
            
        except Exception as e:
            print(f"   Point {s}: Error - {e}")
    
    # Test points OFF critical line
    print("\n2. Testing OFF Critical Line:")
    off_critical_points = [
        complex(0.3, 14.0),  # Re(s) = 0.3
        complex(0.7, 14.0),  # Re(s) = 0.7
        complex(0.4, 20.0),  # Re(s) = 0.4
        complex(0.6, 20.0)   # Re(s) = 0.6
    ]
    
    for s in off_critical_points:
        try:
            analyzer = RHIntegerAnalyzer(depth=3)
            coeffs = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
            
            result = analyzer.analyze_point(s, coeffs)
            
            print(f"   Point {s}:")
            print(f"     - Locked: {result['is_locked']}")
            print(f"     - Gap: {result['gap']}")
            print(f"     - Best action: {result['best_action']}")
            
        except Exception as e:
            print(f"   Point {s}: Error - {e}")
    
    # Test gap scaling analysis
    print("\n3. Testing Gap Scaling Analysis:")
    try:
        d_values = [0.01, 0.05, 0.1, 0.15, 0.2]  # Distance from critical line
        t = 14.0
        
        for N in [17, 33, 65]:  # Different depths
            print(f"   N = {N} (depth = {int(np.log2(N-1))}):")
            
            # Test linear gap fitting
            alpha, A_hat, residual = QuantitativeGapAnalyzer.fit_linear_gap(
                N, t, d_values, known_zeros
            )
            
            print(f"     - Linear fit: α = {alpha:.3f}, A_hat = {A_hat:.6f}")
            print(f"     - Residual: {residual:.6f}")
            
            # Test quadratic gap fitting
            alpha, A_hat, residual = QuantitativeGapAnalyzer.fit_quadratic_gap(
                N, t, d_values, known_zeros
            )
            
            print(f"     - Quadratic fit: α = {alpha:.3f}, A_hat = {A_hat:.6f}")
            print(f"     - Residual: {residual:.6f}")
            
    except Exception as e:
        print(f"   Gap scaling analysis failed: {e}")
    
    # Test gyroscope loss comparison
    print("\n4. Testing Gyroscope Loss Comparison:")
    try:
        kernel = PascalKernel(17, 4)
        
        # On critical line
        sigma_critical = 0.5
        t = 14.0
        loss_critical = GyroscopeLoss.compute_gyro_loss(sigma_critical, t, known_zeros, kernel)
        
        # Off critical line
        sigma_off = 0.3
        loss_off = GyroscopeLoss.compute_gyro_loss(sigma_off, t, known_zeros, kernel)
        
        print(f"   Critical line (σ = {sigma_critical}): loss = {loss_critical:.6f}")
        print(f"   Off critical (σ = {sigma_off}): loss = {loss_off:.6f}")
        print(f"   Ratio (off/critical): {loss_off/loss_critical:.2f}x")
        
    except Exception as e:
        print(f"   Gyroscope loss comparison failed: {e}")

if __name__ == "__main__":
    test_zeta_behavior()
