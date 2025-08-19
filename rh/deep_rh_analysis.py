#!/usr/bin/env python3
"""Deep RH analysis to understand current state"""

import numpy as np
from rh import RHIntegerAnalyzer, PascalKernel, GyroscopeLoss, AngularVelocity, QuantitativeGapAnalyzer

def analyze_critical_line_behavior():
    """Analyze behavior specifically on the critical line"""
    print("🎯 Critical Line Analysis")
    print("=" * 40)
    
    # Test with actual zeta-like behavior
    known_zeros = [
        complex(0.5, 14.134725),
        complex(0.5, 21.022040),
        complex(0.5, 25.010858)
    ]
    
    print(f"Using {len(known_zeros)} known RH zeros")
    
    # Test different depths
    for depth in [2, 3, 4]:
        N = 2**depth + 1
        print(f"\nDepth {depth}, N = {N}:")
        
        analyzer = RHIntegerAnalyzer(depth=depth)
        
        # Test points near known zeros
        for zero in known_zeros:
            t = zero.imag
            s_critical = complex(0.5, t)
            s_off = complex(0.4, t)  # Slightly off critical line
            
            try:
                # Create more realistic coefficients based on distance from zero
                coeffs = []
                for i in range(N):
                    # Simulate zeta-like decay with zero influence
                    dist_to_zero = abs(s_critical - zero)
                    base_coeff = 1.0 / (i + 1)
                    zero_influence = 1.0 / (dist_to_zero + 0.1)
                    coeffs.append(base_coeff * zero_influence)
                
                # Analyze critical line point
                result_critical = analyzer.analyze_point(s_critical, coeffs)
                
                # Analyze off-critical point
                result_off = analyzer.analyze_point(s_off, coeffs)
                
                print(f"  Zero at t = {t:.1f}:")
                print(f"    Critical line: locked={result_critical['is_locked']}, gap={result_critical['gap']}")
                print(f"    Off critical:  locked={result_off['is_locked']}, gap={result_off['gap']}")
                
                # Check if we can distinguish them
                if result_critical['gap'] != result_off['gap']:
                    print(f"    ✓ Gaps differ: {result_critical['gap']} vs {result_off['gap']}")
                else:
                    print(f"    ✗ Gaps are identical: {result_critical['gap']}")
                
            except Exception as e:
                print(f"  Error at t = {t:.1f}: {e}")

def test_gap_scaling_robust():
    """Test gap scaling with more robust analysis"""
    print("\n📊 Robust Gap Scaling Analysis")
    print("=" * 40)
    
    known_zeros = [complex(0.5, 14.134725)]
    
    # Test with very small distances
    d_values = [0.001, 0.005, 0.01, 0.02, 0.05]
    t = 14.0
    
    for N in [17, 33]:
        depth = int(np.log2(N - 1))
        print(f"\nN = {N} (depth = {depth}):")
        
        try:
            # Test linear scaling
            alpha, A_hat, residual = QuantitativeGapAnalyzer.fit_linear_gap(
                N, t, d_values, known_zeros
            )
            
            print(f"  Linear scaling:")
            print(f"    α = {alpha:.3f} (should be close to 1.0)")
            print(f"    A_hat = {A_hat:.6f}")
            print(f"    Residual = {residual:.6f}")
            
            # Test if scaling is actually linear
            if abs(alpha - 1.0) < 0.2:
                print(f"    ✓ Linear scaling confirmed (α ≈ 1.0)")
            else:
                print(f"    ✗ Non-linear scaling (α = {alpha:.3f})")
            
        except Exception as e:
            print(f"  Linear analysis failed: {e}")

def test_dihedral_discrimination():
    """Test if dihedral actions can discriminate critical vs off-critical"""
    print("\n🔄 Dihedral Action Discrimination Test")
    print("=" * 40)
    
    # Create a more sophisticated test
    depth = 3
    N = 2**depth + 1  # 9
    
    analyzer = RHIntegerAnalyzer(depth=depth)
    
    # Test points
    test_points = [
        (complex(0.5, 14.0), "Critical line"),
        (complex(0.4, 14.0), "Off critical (left)"),
        (complex(0.6, 14.0), "Off critical (right)")
    ]
    
    for s, description in test_points:
        print(f"\n{description} at {s}:")
        
        try:
            # Create coefficients that vary with position
            coeffs = []
            for i in range(N):
                # Create pattern that depends on position and s
                base = 1.0 / (i + 1)
                position_factor = np.sin(2 * np.pi * i / N)
                s_factor = abs(s - 0.5)  # Distance from critical line
                coeffs.append(base * (1 + 0.1 * position_factor * s_factor))
            
            result = analyzer.analyze_point(s, coeffs)
            
            print(f"  Mask: {result['mask']}")
            print(f"  Best action: {result['best_action']}")
            print(f"  Locked: {result['is_locked']}")
            print(f"  Gap: {result['gap']}")
            
            # Check if this point has unique characteristics
            if result['is_locked']:
                print(f"  ✓ Point is locked with gap {result['gap']}")
            else:
                print(f"  ✗ Point is not locked (gap = {result['gap']})")
                
        except Exception as e:
            print(f"  Error: {e}")

def main():
    """Run all analyses"""
    print("🚀 Deep RH Analysis Suite")
    print("=" * 50)
    
    analyze_critical_line_behavior()
    test_gap_scaling_robust()
    test_dihedral_discrimination()
    
    print("\n" + "=" * 50)
    print("📋 Analysis Summary:")
    print("• The system is operational but gaps are currently 0")
    print("• Need to improve coefficient generation for better discrimination")
    print("• Dihedral actions are being computed but not providing sufficient contrast")
    print("• Gap scaling analysis shows some structure but needs refinement")

if __name__ == "__main__":
    main()
