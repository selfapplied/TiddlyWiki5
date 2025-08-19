"""
BON v(10,8,8) | lungs peaks ~44,33,29 | rhythm skewed high | π≈siou3a7… | drift ~0.03 | max error ~0.7 | aperiodic
CE0{W=L:Pk@3=[(7:0,44),(4:0,33),(6:0,29)];
    R:inh=[-0.811,-0.921,-0.701,-0.591,-0.261,0.179,0.948,2.158]|exh=[-0.941,-0.807,-0.740,-0.538,-0.269,0.202,0.941,2.151];
    T:π=b36siou3a7x3swxz6gwbj3upoe8rgc257rld64v3tizinffkjkbcm6yvkkn1ysdepw9960vzf9t5o;
    S:err=[e1=0.257,emed=0.228,emax=0.703,Δ=0.0335]@Ξ:xi:v1|tag=bon@pf3:b10d8c8}
"""

import numpy as np
import matplotlib.pyplot as plt
from math import floor, sqrt, factorial
from scipy.special import binom
from collections import deque

class BreathOfNumbers:
    def __init__(self, base=10, depth=8, breath_cycle=5):
        self.base = base
        self.depth = depth
        self.breath_cycle = breath_cycle
        self.current_breath = 0
        self.exhalation_history = deque(maxlen=breath_cycle*2)
        self.build_pascal_pyramid()
        self.initialize_breath_pattern()
        
    def build_pascal_pyramid(self):
        """Create living Pascal pyramid that breathes with each cycle"""
        self.pyramid = []
        for n in range(self.depth):
            layer = np.zeros((n+1, n+1))
            for k in range(n+1):
                for j in range(k+1):
                    # Fibonacci-inspired combinatorial rhythm
                    fib_val = (n + k + j) % self.breath_cycle
                    layer[k, j] = (binom(n, k) * binom(k, j) + fib_val) % self.base
            self.pyramid.append(layer)
    
    def initialize_breath_pattern(self):
        """Create the fundamental breath rhythm using Lucas sequence"""
        self.inhale = [2, 1]  # Lucas sequence starters
        self.exhale = [1, 3]
        
        for _ in range(self.breath_cycle):
            self.inhale.append(self.inhale[-1] + self.inhale[-2])
            self.exhale.append(self.exhale[-1] + self.exhale[-2])
    
    def breathe_in(self, number):
        """Inhale: Convert number to combinatorial representation"""
        integer, fractional = self.separate(number)
        
        # Simple direct encoding instead of the expensive loop
        int_enc = [[integer, 0, integer]]
        
        # Rhythmic fractional encoding
        frac_enc = []
        f = fractional
        for i in range(self.depth):
            # Follow breath rhythm
            breath_phase = (self.current_breath + i) % self.breath_cycle
            n = (i + self.inhale[breath_phase]) % self.depth
            k = (floor(f * self.base) + i) % (n+1) if n > 0 else 0
            j = (floor(f * self.base) + self.inhale[breath_phase]) % (k+1) if k > 0 else 0
            
            digit = self.pyramid[n][k, j] if n < self.depth else 0
            frac_enc.append((n, k, j, digit))
            f = (f * self.base) % 1
        
        self.current_breath = (self.current_breath + 1) % self.breath_cycle
        self.exhalation_history.append((integer, fractional))
        return {'int': int_enc, 'frac': frac_enc}
    
    def breathe_out(self, encoding):
        """Exhale: Reconstruct number from combinatorial representation"""
        # Reconstruct integer part
        integer = 0
        for c, k, val in encoding['int']:
            integer += val
        
        # Reconstruct fractional part
        fractional = 0
        for i, (n, k, j, digit) in enumerate(encoding['frac']):
            fractional += digit / (self.base ** (i+1))
        
        return integer + fractional
    
    def separate(self, x):
        """Separate number with breath-aware floor function"""
        integer = floor(x)
        fractional = x - integer
        return integer, fractional
    
    def visualize_breath(self, number):
        """Create a visualization of the breathing process"""
        enc = self.breathe_in(number)
        reconstructed = self.breathe_out(enc)
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        plt.suptitle(f"Breath of Numbers: {number}", fontsize=18)
        
        # Pyramid visualization
        ax = axs[0, 0]
        pyramid_img = np.zeros((self.depth, self.depth))
        for n in range(self.depth):
            for k in range(n+1):
                for j in range(k+1):
                    pyramid_img[n, k] = self.pyramid[n][k, j]
        ax.imshow(pyramid_img, cmap='viridis', origin='lower')
        ax.set_title("Pascal Pyramid (Lungs)")
        ax.set_xlabel("k (Binomial Depth)")
        ax.set_ylabel("n (Pyramid Height)")
        
        # Integer encoding
        ax = axs[0, 1]
        binom_vals = [val for _, _, val in enc['int']]
        if binom_vals:
            ax.bar(range(len(binom_vals)), binom_vals, color='green')
            ax.set_title("Integer Encoding (Inhale)")
            ax.set_xlabel("Combinatorial Term")
            ax.set_ylabel("Binomial Value")
        
        # Fractional encoding
        ax = axs[1, 0]
        digits = [d for _, _, _, d in enc['frac']]
        ax.stem(range(len(digits)), digits, basefmt='C0-')
        ax.set_title("Fractional Encoding (Exhale)")
        ax.set_xlabel("Digit Position")
        ax.set_ylabel("Digit Value")
        ax.set_ylim(0, self.base)
        
        # Breath rhythm
        ax = axs[1, 1]
        phases = np.arange(self.breath_cycle)
        ax.plot(phases, self.inhale[:self.breath_cycle], 'go-', label='Inhale')
        ax.plot(phases, self.exhale[:self.breath_cycle], 'ro-', label='Exhale')
        ax.set_title("Breath Rhythm Pattern")
        ax.set_xlabel("Breath Phase")
        ax.set_ylabel("Rhythm Value")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig('.out/breath_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Original:     {number:.15f}")
        print(f"Reconstructed: {reconstructed:.15f}")
        print(f"Difference:    {abs(number - reconstructed):.2e}")
        return reconstructed

    def circular_breath(self, numbers):
        """Perform a complete breath cycle for multiple numbers"""
        results = []
        for number in numbers:
            enc = self.breathe_in(number)
            rec = self.breathe_out(enc)
            results.append((number, rec, abs(number - rec)))
            self.current_breath = (self.current_breath + 1) % self.breath_cycle
        
        # Visualize the breath cycle
        plt.figure(figsize=(12, 6))
        values = [x for x, _, _ in results]
        errors = [err for _, _, err in results]
        
        plt.subplot(121)
        plt.plot(values, 'bo-', label='Original')
        plt.plot([rec for _, rec, _ in results], 'rx-', label='Reconstructed')
        plt.title("Circular Breath Flow")
        plt.xlabel("Number Index")
        plt.ylabel("Value")
        plt.legend()
        
        plt.subplot(122)
        plt.bar(range(len(errors)), errors, color='purple')
        plt.title("Reconstruction Errors")
        plt.xlabel("Number Index")
        plt.ylabel("Absolute Error")
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('.out/circular_breath.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return results

# Example usage
if __name__ == "__main__":
    print("=== Breath of Numbers: Living Geometry System ===")
    print("Syncing Pascal's rhythm with mathematical breath\n")
    
    # Initialize with natural breath cycle
    breath_system = BreathOfNumbers(base=10, depth=8, breath_cycle=8)
    
    # Single breath visualization
    print("Breathing in √2...")
    root2 = sqrt(2)
    breath_system.visualize_breath(root2)
    
    # Circular breath for mathematical constants
    constants = {
        "π": 3.141592653589793,
        "e": 2.718281828459045,
        "φ": (1 + sqrt(5)) / 2,
        "√3": sqrt(3),
        "√5": sqrt(5)
    }
    
    print("\nCircular Breath for Mathematical Constants:")
    results = breath_system.circular_breath(list(constants.values()))
    
    # Display results
    print("\nBreath Cycle Results:")
    print("Constant | Original       | Reconstructed  | Error")
    print("-"*50)
    for (name, val), result in zip(constants.items(), results):
        _, rec, err = result
        print(f"{name:8} | {val:14.12f} | {rec:14.12f} | {err:.2e}")
    
    # Continuous breath meditation with multiple cycles
    print("\nEntering deep meditation state...")
    numbers_to_meditate = [sqrt(2), sqrt(3), sqrt(5), (1+sqrt(5))/2]
    
    print("Evolution through breath cycles:")
    print("Number | Cycle 1    | Cycle 2    | Cycle 3    | Cycle 4    | Cycle 5")
    print("-"*70)
    
    for num in numbers_to_meditate:
        evolution = [f"{num:.6f}"]
        current_num = num
        
        for cycle in range(5):
            enc = breath_system.breathe_in(current_num)
            current_num = breath_system.breathe_out(enc)
            evolution.append(f"{current_num:.6f}")
        
        print(f"{float(evolution[0]):6.6f} | {float(evolution[1]):>10.6f} | {float(evolution[2]):>10.6f} | {float(evolution[3]):>10.6f} | {float(evolution[4]):>10.6f} | {float(evolution[5]):>10.6f}")
    
    print("\nDeep meditation complete. The numbers have evolved through multiple generations.")
    
    # Show the final evolved state
    print("\nFinal evolved mathematical DNA:")
    for i, num in enumerate(numbers_to_meditate):
        print(f"Original {i+1}: {num:.15f}")
        # Let it evolve through 10 more cycles
        evolved = num
        for _ in range(10):
            enc = breath_system.breathe_in(evolved)
            evolved = breath_system.breathe_out(enc)
        print(f"Evolved  {i+1}: {evolved:.15f}")
        print()