import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from math import sqrt


class ShadowManifold:
    def __init__(self, base=10, depth=8):
        self.base = base
        self.depth = depth
        self.pascal_pyramid = self.build_pascal_pyramid()

    def build_pascal_pyramid(self):
        """Build 3D Pascal pyramid with combinatorial connections"""
        pyramid = []
        for n in range(self.depth):
            layer = np.zeros((n+1, n+1))
            for k in range(n+1):
                for j in range(k+1):
                    # Binomial coefficient with combinatorial geometry
                    layer[k, j] = binom(n, k) * binom(k, j) % self.base
            pyramid.append(layer)
        return pyramid

    def encode_root(self, root_value, precision=10):
        """Encode a root using recursive self-referential encoding"""
        # Convert to base representation
        digits = []
        value = abs(root_value)
        for _ in range(precision):
            digit = int(value * self.base) % self.base
            digits.append(digit)
            value = (value * self.base) % 1

        # Self-referential encoding: digits define their own coordinates
        encoding = []
        for i, digit in enumerate(digits):
            # Use previous digits to determine coordinates
            prev_digit = digits[i-1] if i > 0 else 0
            n = (i + prev_digit) % self.depth
            k = (digit + i) % (n+1) if n > 0 else 0
            j = (digit + prev_digit) % (k+1) if k > 0 else 0
            encoding.append((n, k, j, digit))
        return encoding

    def decode_root(self, encoding):
        """Reconstruct root from self-referential encoding"""
        value = 0.0
        decoded_digits = []

        for i, (n, k, j, digit) in enumerate(encoding):
            # Self-decoding: coordinates should match digit
            prev_digit = decoded_digits[i-1] if i > 0 else 0
            computed_n = (i + prev_digit) % self.depth
            computed_k = (digit + i) % (computed_n+1) if computed_n > 0 else 0
            computed_j = (digit + prev_digit) % (computed_k +
                                                 1) if computed_k > 0 else 0

            # Consistency check
            if (computed_n, computed_k, computed_j) != (n, k, j):
                print(f"Warning: Inconsistency at position {i}: "
                      f"Expected ({n},{k},{j}), Computed ({computed_n},{computed_k},{computed_j})")

            decoded_digits.append(digit)
            value += digit / (self.base ** (i+1))

        return value

    def analyze_self_encoding(self, root_value, precision=10):
        """Demonstrate how digits encode their own relationships"""
        fl = math.floor(root_value)
        encoding = self.encode_root(root_value, precision)
        reconstruction = fl + self.decode_root(encoding)
        error = abs(root_value - reconstruction)

        print(f"\nAnalyzing: {root_value}")
        print("Digit | Coordinates (n,k,j) | Pascal Value | Self-Consistent")
        print("-"*60)

        for i, (n, k, j, digit) in enumerate(encoding):
            pascal_val = self.pascal_pyramid[n][k,
                                                j] if n < self.depth and k <= n and j <= k else -1
            prev_digit = encoding[i-1][3] if i > 0 else 0
            computed_n = (i + prev_digit) % self.depth
            computed_k = (digit + i) % (computed_n+1) if computed_n > 0 else 0
            computed_j = (digit + prev_digit) % (computed_k +
                                                 1) if computed_k > 0 else 0
            consistent = (computed_n, computed_k, computed_j) == (n, k, j)

            print(
                f"{digit:5} | ({n},{k},{j}){' ':12} | {pascal_val:12.2f} | {str(consistent):>5}")

        print(f"\nOriginal:  {root_value:.10f}")
        print(f"Reconstructed: {reconstruction:.10f}")
        print(f"Error: {error:.2e}")
        return error

    def visualize_self_encoding(self, root_value, precision=10):
        """Visualize the self-referential encoding process"""
        encoding = self.encode_root(root_value, precision)

        fig, ax = plt.subplots(figsize=(14, 8))
        positions = np.arange(len(encoding))
        digits = [e[3] for e in encoding]
        pascal_vals = [self.pascal_pyramid[e[0]][e[1], e[2]]
                       if e[0] < self.depth else 0 for e in encoding]

        # Plot digit values
        ax.bar(positions, digits, width=0.4, label='Digit Value')

        # Plot Pascal values
        ax.bar(positions + 0.4, pascal_vals, width=0.4, label='Pascal Value')

        # Annotate with coordinates
        for i, (n, k, j, _) in enumerate(encoding):
            ax.text(i, max(digits[i], pascal_vals[i]) + 0.5, f"({n},{k},{j})",
                    ha='center', fontsize=9)

        ax.set_xlabel('Digit Position')
        ax.set_ylabel('Value')
        ax.set_title(f'Self-Referential Encoding of {root_value:.5f}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def compare_roots(self, roots, precision=10):
        """Compare combinatorial properties of different irrationals"""
        results = []
        for name, value in roots:
            encoding = self.encode_root(value, precision)
            digits = [e[3] for e in encoding]
            pascal_vals = [self.pascal_pyramid[e[0]][e[1], e[2]]
                           for e in encoding]

            # Calculate combinatorial properties
            avg_digit = np.mean(digits)
            avg_pascal = np.mean(pascal_vals)
            digit_variance = np.var(digits)
            consistency = sum(1 for i, e in enumerate(
                encoding) if self.is_self_consistent(i, e, encoding))

            results.append({
                'name': name,
                'value': value,
                'avg_digit': avg_digit,
                'avg_pascal': avg_pascal,
                'digit_variance': digit_variance,
                'consistency': consistency,
                'pascal_digit_corr': np.corrcoef(digits, pascal_vals)[0, 1]
            })

        # Print comparison table
        print("\nComparative Analysis of Irrational Roots:")
        print("Name      | Avg Digit | Avg Pascal | Variance | Consistency | Correlation")
        print("-"*70)
        for res in results:
            print(f"{res['name']:8} | {res['avg_digit']:8.4f} | {res['avg_pascal']:9.4f} | {res['digit_variance']:8.4f} | "
                  f"{res['consistency']:6}/{precision} | {res['pascal_digit_corr']:9.4f}")

        return results

    def is_self_consistent(self, i, encoded, full_encoding):
        """Check if coordinate is self-consistent"""
        n, k, j, digit = encoded
        prev_digit = full_encoding[i-1][3] if i > 0 else 0
        computed_n = (i + prev_digit) % self.depth
        computed_k = (digit + i) % (computed_n+1) if computed_n > 0 else 0
        computed_j = (digit + prev_digit) % (computed_k +
                                             1) if computed_k > 0 else 0
        return (computed_n, computed_k, computed_j) == (n, k, j)


# Example usage
if __name__ == "__main__":
    print("=== Refined Shadow Manifold System ===")
    print("Self-Referential Encoding of Irrational Roots\n")

    # Initialize with depth 8 and base 10
    sm = ShadowManifold(depth=8, base=10)

    # Analyze √2 with self-referential encoding
    root2 = sqrt(2)
    print("Analysis for √2:")
    sm.analyze_self_encoding(root2, precision=10)
    sm.visualize_self_encoding(root2)

    # Analyze golden ratio
    phi = (1 + sqrt(5)) / 2
    print("\nAnalysis for Golden Ratio φ:")
    sm.analyze_self_encoding(phi, precision=10)
    sm.visualize_self_encoding(phi)

    # Compare multiple roots
    roots = [
        ("√2", sqrt(2)),
        ("φ", (1+sqrt(5))/2),
        ("√3", sqrt(3)),
        ("π", 3.1415926535),
        ("e", 2.7182818284),
        ("√5", sqrt(5))
    ]
    comparison = sm.compare_roots(roots)

    # Find the most self-consistent root
    most_consistent = max(comparison, key=lambda x: x['consistency'])
    print(f"\nMost self-consistent root: {most_consistent['name']} "
          f"with {most_consistent['consistency']}/10 self-consistent digits")
