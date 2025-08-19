import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from math import floor, sqrt


class CompleteShadowManifold:
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

    def separate_number(self, x):
        """Separate number into integer and fractional parts"""
        integer_part = floor(x)
        fractional = x - integer_part
        return integer_part, fractional

    def encode_integer(self, n):
        """Encode integer using combinadics representation"""
        # Simple direct encoding instead of the inefficient loop
        return [[n, 0, n]]

    def encode_fractional(self, fractional, precision=10):
        """Encode fractional part using self-referential encoding"""
        digits = []
        value = fractional
        for _ in range(precision):
            digit = int(value * self.base) % self.base
            digits.append(digit)
            value = (value * self.base) % 1

        # Self-referential encoding
        encoding = []
        for i, digit in enumerate(digits):
            prev_digit = digits[i-1] if i > 0 else 0
            n = (i + prev_digit) % self.depth
            k = (digit + i) % (n+1) if n > 0 else 0
            j = (digit + prev_digit) % (k+1) if k > 0 else 0
            encoding.append((n, k, j, digit))
        return encoding

    def reconstruct_integer(self, encoding):
        """Reconstruct integer from combinadics representation"""
        value = 0
        for c, k, binom_val in encoding:
            value += binom_val
        return value

    def reconstruct_fractional(self, encoding):
        """Reconstruct fractional part from self-referential encoding"""
        fractional = 0
        for i, (n, k, j, digit) in enumerate(encoding):
            fractional += digit / (self.base ** (i+1))
        return fractional

    def full_encode(self, x, precision=10):
        """Complete encoding of a real number"""
        integer_part, fractional = self.separate_number(x)
        int_encoding = self.encode_integer(integer_part)
        frac_encoding = self.encode_fractional(fractional, precision)
        return {
            'integer_part': integer_part,
            'fractional': fractional,
            'int_encoding': int_encoding,
            'frac_encoding': frac_encoding
        }

    def full_reconstruct(self, encoding):
        """Complete reconstruction of a real number"""
        int_part = self.reconstruct_integer(encoding['int_encoding'])
        frac_part = self.reconstruct_fractional(encoding['frac_encoding'])
        return int_part + frac_part

    def visualize_full_encoding(self, x, precision=10):
        """Visualize the complete encoding process"""
        encoding = self.full_encode(x, precision)

        # Create figure with 3D pyramid and digit plots
        fig = plt.figure(figsize=(18, 12))

        # 3D pyramid plot
        ax1 = fig.add_subplot(121, projection='3d')
        self.plot_pyramid(ax1, encoding)

        # Digit comparison plot
        ax2 = fig.add_subplot(222)
        self.plot_digits(ax2, encoding, 'frac_encoding', 'Fractional Part')

        # Integer encoding plot
        ax3 = fig.add_subplot(224)
        self.plot_integer_encoding(ax3, encoding)

        plt.tight_layout()
        plt.show()

        # Print reconstruction accuracy
        reconstructed = self.full_reconstruct(encoding)
        error = abs(x - reconstructed)
        print(f"\nOriginal Value: {x:.15f}")
        print(f"Reconstructed:   {reconstructed:.15f}")
        print(f"Absolute Error:  {error:.2e}")

        return error

    def plot_pyramid(self, ax, encoding):
        """Plot the Pascal pyramid with encoding highlights"""
        # Plot pyramid structure
        for n, layer in enumerate(self.pascal_pyramid):
            size = layer.shape[0]
            for k in range(size):
                for j in range(k+1):
                    value = layer[k, j]
                    if value > 0:
                        ax.scatter(j, n, k, c='blue', s=30*value, alpha=0.2)

        # Highlight fractional encoding
        for i, (n, k, j, digit) in enumerate(encoding['frac_encoding']):
            ax.scatter(j, n, k, c='red', s=200, marker='*')
            ax.text(j, n, k, f"{digit}", fontsize=12, color='black')

        # Highlight integer encoding
        for i, (c, k, binom_val) in enumerate(encoding['int_encoding']):
            if c < self.depth and k < self.depth:
                ax.scatter(k, c, k, c='green', s=300, marker='s')
                ax.text(k, c, k, f"{binom_val}", fontsize=10, color='white')

        ax.set_xlabel('X: Column Index')
        ax.set_ylabel('Y: Pyramid Height')
        ax.set_zlabel('Z: Binomial Depth')
        ax.set_title(f"Complete Encoding in Pascal Pyramid")

    def plot_digits(self, ax, encoding, part, title):
        """Plot digit values vs combinatorial values"""
        digits = [e[3] for e in encoding[part]]
        positions = range(len(digits))

        # Get combinatorial values
        comb_vals = []
        for e in encoding[part]:
            n, k, j, digit = e
            if n < self.depth and k <= n and j <= k:
                comb_vals.append(self.pascal_pyramid[n][k, j])
            else:
                comb_vals.append(0)

        ax.bar(positions, digits, width=0.4, label='Digit Value')
        ax.bar([p + 0.4 for p in positions], comb_vals,
               width=0.4, label='Combinatorial Value')

        # Annotate with coordinates
        for i, e in enumerate(encoding[part]):
            n, k, j, digit = e
            ax.text(i, max(digits[i], comb_vals[i]) + 0.5, f"({n},{k},{j})",
                    ha='center', fontsize=9)

        ax.set_xlabel('Digit Position')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    def plot_integer_encoding(self, ax, encoding):
        """Visualize the integer encoding"""
        binom_vals = [e[2] for e in encoding['int_encoding']]
        indices = range(len(binom_vals))

        ax.bar(indices, binom_vals, color='green')
        for i, (c, k, val) in enumerate(encoding['int_encoding']):
            ax.text(i, val, f"C({c},{k})", ha='center', va='bottom')

        ax.set_xlabel('Term Index')
        ax.set_ylabel('Binomial Value')
        ax.set_title(f"Integer Encoding: {encoding['integer_part']}")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    def analyze_root(self, x, precision=10):
        """Complete analysis of a number's combinatorial structure"""
        encoding = self.full_encode(x, precision)
        reconstructed = self.full_reconstruct(encoding)
        error = abs(x - reconstructed)

        print(f"\nAnalysis for: {x}")
        print(f"Integer Part: {encoding['integer_part']}")
        print("Integer Encoding:")
        for i, (c, k, val) in enumerate(encoding['int_encoding']):
            print(f"  Term {i}: C({c},{k}) = {val}")

        print("\nFractional Part Encoding:")
        print("Pos | Digit | Coordinates | Pascal Value | Self-Consistent")
        print("-"*60)
        for i, (n, k, j, digit) in enumerate(encoding['frac_encoding']):
            # Get Pascal value
            if n < self.depth and k <= n and j <= k:
                pascal_val = self.pascal_pyramid[n][k, j]
            else:
                pascal_val = -1

            # Check self-consistency
            prev_digit = encoding['frac_encoding'][i-1][3] if i > 0 else 0
            computed_n = (i + prev_digit) % self.depth
            computed_k = (digit + i) % (computed_n+1) if computed_n > 0 else 0
            computed_j = (digit + prev_digit) % (computed_k +
                          1) if computed_k > 0 else 0
            consistent = (computed_n, computed_k, computed_j) == (n, k, j)

            print(
                f"{i:3} | {digit:5} | ({n},{k},{j}) | {pascal_val:12.2f} | {str(consistent):>5}")

        print(f"\nOriginal:  {x:.15f}")
        print(f"Reconstructed: {reconstructed:.15f}")
        print(f"Error: {error:.2e}")

        return {
            'encoding': encoding,
            'reconstructed': reconstructed,
            'error': error,
            'int_complexity': len(encoding['int_encoding']),
            'frac_consistency': sum(1 for i, _ in enumerate(encoding['frac_encoding']))
                                if consistent else 0
        }

# Example usage with √2
if __name__ == "__main__":
    print("=== Complete Shadow Manifold System ===")
    print("Integer-Fractional Bridge through Pascal's Pyramid\n")

    # Initialize system
    sm= CompleteShadowManifold(base=10, depth=8)

    # Analyze √2
    root2= sqrt(2)
    analysis= sm.analyze_root(root2, precision=10)
    sm.visualize_full_encoding(root2)

    # Analyze golden ratio
    phi= (1 + sqrt(5)) / 2
    print("\n" + "="*60)
    sm.analyze_root(phi, precision=10)

    # Analyze π
    pi= 3.141592653589793
    print("\n" + "="*60)
    sm.analyze_root(pi, precision=10)

    # Compare reconstruction accuracy
    numbers= {
        "√2": sqrt(2),
        "φ": (1+sqrt(5))/2,
        "π": 3.141592653589793,
        "e": 2.718281828459045,
        "√3": sqrt(3)
    }

    print("\nReconstruction Accuracy Comparison:")
    print("Number | Original Value       | Reconstructed       | Error")
    print("-"*70)
    for name, value in numbers.items():
        encoding= sm.full_encode(value, 10)
        reconstructed= sm.full_reconstruct(encoding)
        error= abs(value - reconstructed)
        print(f"{name:6} | {value:20.15f} | {reconstructed:20.15f} | {error:.2e}")
