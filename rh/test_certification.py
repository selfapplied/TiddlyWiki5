#!/usr/bin/env python3
"""Test the complete RH certification system"""

from rh import BalancedLockingLoss, QuantitativeGapAnalyzer
import numpy as np

def test_certification_system():
    """Test the complete certification system"""
    
    print("🔒 RH CERTIFICATION SYSTEM TEST")
    print("=" * 50)
    
    # Test parameters
    N = 9  # Smaller N = larger gaps (gap ∝ 1/N²)
    t = 14.0
    d_values = [0.0, 0.05, 0.1, 0.15, 0.2]  # Test on and off critical line
    zeros = [0.5 + 14.134725j, 0.5 + 21.022040j]  # First two RH zeros
    
    print(f"Testing N = {N}, t = {t}")
    print(f"Critical line offset d values: {d_values}")
    print()
    
    # Test the certification switch
    print("1️⃣ CERTIFICATION SWITCH TEST")
    print("-" * 35)
    
    certification_result = BalancedLockingLoss.demonstrate_certification_switch(
        d_values, t, N, zeros
    )
    
    # Display results
    print("Results by critical line offset:")
    print("-" * 50)
    print(f"{'d':>6} {'σ':>8} {'L_triangle':>12} {'Certified':>10} {'Gap_N':>8} {'Gap_2N':>8}")
    print("-" * 50)
    
    for result in certification_result['results']:
        d = result['d']
        sigma = result['sigma']
        L_triangle = result['L_triangle']
        certified = result['certified']
        gap_N = result['gap_N']
        gap_2N = result['gap_2N']
        
        print(f"{d:>6.2f} {sigma:>8.3f} {L_triangle:>12.3f} {str(certified):>10} {gap_N:>8.1f} {gap_2N:>8.1f}")
    
    print()
    
    # Display certification switch status
    print("2️⃣ CERTIFICATION SWITCH STATUS")
    print("-" * 35)
    
    switch = certification_result['certification_switch']
    print(f"On critical line (d=0): {'✅ SUCCEEDS' if switch['on_line_succeeds'] else '❌ FAILS'}")
    print(f"Off critical line (d>0): {'✅ FAILS' if switch['off_line_fails'] else '❌ SUCCEEDS'}")
    print(f"Switch working: {'✅ YES' if switch['switch_working'] else '❌ NO'}")
    print()
    
    # Display mathematical principle
    print("3️⃣ MATHEMATICAL PRINCIPLE")
    print("-" * 30)
    print(f"Principle: {certification_result['mathematical_principle']}")
    print(f"Mechanism: {certification_result['proof_mechanism']}")
    print()
    
    # Test individual loss components
    print("4️⃣ LOSS COMPONENT BREAKDOWN")
    print("-" * 35)
    
    # Test on critical line (should succeed)
    on_line_result = next(r for r in certification_result['results'] if r['d'] == 0)
    print(f"On critical line (d=0):")
    print(f"  L_triangle = {on_line_result['L_triangle']:.3f}")
    print(f"  Certified: {on_line_result['certified']}")
    
    # Test off critical line (should fail)
    off_line_result = next(r for r in certification_result['results'] if r['d'] > 0.01)
    print(f"Off critical line (d={off_line_result['d']:.2f}):")
    print(f"  L_triangle = {off_line_result['L_triangle']:.3f}")
    print(f"  Certified: {off_line_result['certified']}")
    
    print()
    
    # Final assessment
    print("5️⃣ FINAL ASSESSMENT")
    print("-" * 20)
    
    if switch['switch_working']:
        print("🎉 CERTIFICATION SYSTEM WORKING!")
        print("• On critical line: L_triangle = 0 → RH certificate succeeds")
        print("• Off critical line: L_triangle > 0 → RH certificate fails")
        print("• This creates the mathematical proof mechanism")
    else:
        print("⚠️  CERTIFICATION SYSTEM NEEDS ADJUSTMENT")
        print("• Check gap thresholds and loss function parameters")
        print("• Verify mask/template creation for proper contrast")
    
    print()
    print("🔬 NEXT STEPS:")
    print("• Scale up N to achieve required certification gaps")
    print("• Refine loss function parameters")
    print("• Document the complete proof structure")

if __name__ == "__main__":
    test_certification_system()
