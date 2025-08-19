#!/usr/bin/env python3
"""Simple test of RH system components"""

import numpy as np
from rh import RHIntegerAnalyzer, PascalKernel, GyroscopeLoss, AngularVelocity

def test_basic_components():
    """Test basic components work"""
    print("🧪 Testing Basic RH Components")
    print("=" * 40)
    
    # Test PascalKernel
    print("\n1. Testing PascalKernel:")
    try:
        kernel = PascalKernel(17, 4)  # N = 2^4 + 1 = 17
        print(f"   ✓ Created kernel with N={kernel.N}, depth={kernel.depth}")
        print(f"   ✓ Kernel row: {kernel.get_kernel_row()}")
        print(f"   ✓ Normalized: {[f'{x:.3f}' for x in kernel.get_normalized_kernel()]}")
        print(f"   ✓ Variance: {kernel.get_variance():.6f}")
        print(f"   ✓ Scaling factor: {kernel.get_scaling_factor():.6f}")
    except Exception as e:
        print(f"   ✗ PascalKernel failed: {e}")
    
    # Test RHIntegerAnalyzer
    print("\n2. Testing RHIntegerAnalyzer:")
    try:
        analyzer = RHIntegerAnalyzer(depth=2)
        print(f"   ✓ Created analyzer with depth={analyzer.depth}, N={analyzer.N}")
        
        # Test with a simple point
        s = complex(0.5, 14.134725)  # First RH zero
        test_coeffs = [1.0, 0.5, 0.25, 0.125, 0.0625]
        
        result = analyzer.analyze_point(s, test_coeffs)
        print(f"   ✓ Analysis successful:")
        print(f"     - Point: {result['s']}")
        print(f"     - Mask: {result['mask']}")
        print(f"     - Best action: {result['best_action']}")
        print(f"     - Locked: {result['is_locked']}")
        print(f"     - Gap: {result['gap']}")
        
    except Exception as e:
        print(f"   ✗ RHIntegerAnalyzer failed: {e}")
    
    # Test AngularVelocity
    print("\n3. Testing AngularVelocity:")
    try:
        # Create some mock zeros
        zeros = [complex(0.5, 14.134725), complex(0.5, 21.022040)]
        
        sigma, t = 0.5, 14.0
        omega = AngularVelocity.get_angular_velocity(sigma, t, zeros)
        print(f"   ✓ Angular velocity computed:")
        print(f"     - σ={sigma}, t={t}")
        print(f"     - ω = {omega}")
        print(f"     - ||ω|| = {omega.norm():.6f}")
        
    except Exception as e:
        print(f"   ✗ AngularVelocity failed: {e}")
    
    # Test GyroscopeLoss
    print("\n4. Testing GyroscopeLoss:")
    try:
        kernel = PascalKernel(17, 4)
        zeros = [complex(0.5, 14.134725), complex(0.5, 21.022040)]
        
        sigma, t = 0.5, 14.0
        E_N = GyroscopeLoss.smooth_omega_sigma(sigma, t, zeros, kernel)
        loss = GyroscopeLoss.compute_gyro_loss(sigma, t, zeros, kernel)
        
        print(f"   ✓ Gyroscope loss computed:")
        print(f"     - E_N = {E_N:.6f}")
        print(f"   ✓ Loss = {loss:.6f}")
        
    except Exception as e:
        print(f"   ✗ GyroscopeLoss failed: {e}")

if __name__ == "__main__":
    test_basic_components()
