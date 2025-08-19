"""
Riemann Hypothesis: Integer-Only Analysis Framework

Main entrypoint that demonstrates the clean integer-only approach.
"""

import numpy as np

from rh import RHIntegerAnalyzer, demonstrate_integer_analysis
from pascal import pascal_nested_brackets


def main():
    """Main execution function for integer-only RH analysis"""
    try:
        print("🌌 Riemann Hypothesis Analysis: Integer-Only Framework")
        print("=" * 60)

        # Run the main demonstration
        print("\n🔍 Running Integer-Only Analysis Demo:")
        result, two_adic_result = demonstrate_integer_analysis()
        
        # Show detailed results
        print(f"\n📊 Analysis Results:")
        print(f"  • Point analyzed: {result['s']}")
        print(f"  • Depth: {result['depth']}, N: {result['N']}")
        print(f"  • Mask: {result['mask']}")
        print(f"  • Best action: {result['best_action']}")
        print(f"  • Locked: {result['is_locked']}")
        print(f"  • Gap: {result['gap']}")
        print(f"  • Lock reason: {result['lock_reason']}")
        
        # Show 2-adic pyramid results
        print(f"\n🔺 2-Adic Pyramid Results:")
        if "error" not in two_adic_result:
            print(f"  • Method: {two_adic_result['method']}")
            print(f"  • N: {two_adic_result['N']} (padded to power of 2)")
            print(f"  • Pyramid levels: {two_adic_result['pyramid_levels']}")
            print(f"  • Candidates after culling: {two_adic_result['candidates_after_culling']}")
            print(f"  • Best shift: {two_adic_result['best_shift']}, score: {two_adic_result['best_score']}")
            print(f"  • Gap: {two_adic_result['gap']}")
            print(f"  ✅ 2-Adic pyramid analysis successful!")
        else:
            print(f"  ❌ 2-Adic pyramid system not available: {two_adic_result['error']}")

        # Test with different depths
        print("\n📊 Testing Different Depths:")
        for depth in [1, 2, 3]:
            analyzer = RHIntegerAnalyzer(depth=depth)
            N = 2**depth + 1 if depth > 0 else 2
            print(f"  Depth {depth}: N = {N}")
            
            # Test with a simple point
            s = complex(0.5, 14.134725)
            test_coeffs = [1.0, 0.5, 0.25, 0.125][:N]
            
            try:
                result = analyzer.analyze_point(s, test_coeffs)
                print(f"    ✓ Analysis successful: locked={result['is_locked']}")
                print(f"      Mask: {result['mask']}, Best action: {result['best_action']}, Gap: {result['gap']}")
            except Exception as e:
                print(f"    ✗ Analysis failed: {e}")
        
        # Test 2-adic pyramid at different depths
        print("\n🔺 Testing 2-Adic Pyramid at Different Depths:")
        for depth in [1, 2]:
            analyzer = RHIntegerAnalyzer(depth=depth)
            s = complex(0.5, 14.134725)
            test_coeffs = [1.0, 0.5, 0.25, 0.125]
            
            try:
                two_adic_result = analyzer.analyze_with_two_adic_pyramid(s, test_coeffs)
                if "error" not in two_adic_result:
                    print(f"  Depth {depth}: ✓ 2-Adic analysis successful, gap={two_adic_result['gap']}")
                else:
                    print(f"  Depth {depth}: ❌ {two_adic_result['error']}")
            except Exception as e:
                print(f"  Depth {depth}: ✗ Analysis failed: {e}")

        # Pascal-space demo
        print("\n📐 Pascal-Space Bracket Demo:")
        x = 0.5  # Simple test value
        brackets = pascal_nested_brackets(x, 2, 4)
        if brackets:
            b = brackets[-1]
            print(f"  Bracket for x={x:.6f} at depth={b.depth}:")
            print(f"    Index: {b.cell_index}")
            print(f"    Interval: [{b.lower_bound:.6f}, {b.upper_bound:.6f}]")
            print(f"    Weight: C({b.N},{b.cell_index})")

        print("\n✅ Integer-Only Analysis Complete!")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
