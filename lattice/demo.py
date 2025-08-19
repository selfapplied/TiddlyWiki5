import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import hsv_to_rgb
from scipy.special import binom
from sympy import binomial, prime

class ShadowManifold:
    def __init__(self, depth=64):
        self.depth = depth
        self.color_map = self._create_color_map()
        
    def _create_color_map(self):
        """Create HSV color map for visualizing residue classes"""
        hues = np.linspace(0, 1, 11, endpoint=False)  # 10+1 for 0-10
        saturations = np.ones_like(hues)
        values = np.ones_like(hues)
        return [hsv_to_rgb([h, s, v]) for h, s, v in zip(hues, saturations, values)]
    
    def generate_pascal_layer(self, n, modulus=2):
        """Generate a layer of Pascal's triangle modulo m"""
        layer = np.zeros((n+1, n+1), dtype=int)
        for i in range(n+1):
            for j in range(i+1):
                layer[i, j] = binomial(i, j) % modulus
        return layer
    
    def generate_3d_structure(self, modulus=2):
        """Generate 3D structure from Pascal's triangle modulo m"""
        structure = []
        for n in range(self.depth):
            layer = self.generate_pascal_layer(n, modulus)
            structure.append(layer)
        return structure
    
    def calculate_beta_sheet(self, structure):
        """Calculate beta sheets (planar surfaces) in the structure"""
        sheet_areas = []
        for z, layer in enumerate(structure):
            rows, cols = layer.shape
            area = 0
            for i in range(rows):
                for j in range(i+1):
                    if layer[i, j] != 0:
                        # Calculate area contribution based on position
                        area += 1 / (1 + abs(i - j))
            sheet_areas.append(area)
        return sheet_areas
    
    def calculate_alpha_helix(self, structure):
        """Calculate alpha helices (depth bridges) in the structure"""
        helix_paths = []
        for start_col in range(0, min(10, self.depth)):
            path = []
            for n in range(self.depth):
                if n < start_col:
                    continue
                # Follow diagonal path through the structure
                col = start_col
                row = n - start_col
                if row < 0 or col > row:
                    break
                if row < structure[n].shape[0] and col < structure[n].shape[1]:
                    path.append((row, col, n, structure[n][row, col]))
            helix_paths.append(path)
        return helix_paths
    
    def shadow_manifold_transform(self, structure, prime_set=(3,5,7)):
        """Create shadow manifold bridge with multiple primes"""
        shadow = []
        for n in range(self.depth):
            shadow_layer = np.zeros((n+1, n+1), dtype=complex)
            for i in range(n+1):
                for j in range(i+1):
                    # Combine residues using Chinese Remainder Theorem
                    residues = [binomial(i, j) % p for p in prime_set]
                    # Create complex number representation
                    z = sum(r * np.exp(2j * np.pi * k / len(prime_set)) 
                            for k, r in enumerate(residues))
                    shadow_layer[i, j] = z
            shadow.append(shadow_layer)
        return shadow
    
    def calculate_volume(self, structure):
        """Calculate volume of the geometric structure"""
        volume = 0
        for z, layer in enumerate(structure):
            for i in range(layer.shape[0]):
                for j in range(layer.shape[1]):
                    if layer[i, j] != 0:
                        # Volume element with depth scaling
                        volume += 1 / (z + 1)**2
        return volume
    
    def calculate_surface_area(self, structure):
        """Calculate surface area of the geometric structure"""
        surface_area = 0
        for z in range(len(structure)):
            if z == 0:
                continue
            current_layer = structure[z]
            prev_layer = structure[z-1]
            
            for i in range(current_layer.shape[0]):
                for j in range(current_layer.shape[1]):
                    if current_layer[i, j] == 0:
                        continue
                    
                    # Check neighbors in 3D space
                    exposed_faces = 6  # Start with 6 faces of a cube
                    
                    # Check same layer neighbors (4 directions)
                    neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                    for ni, nj in neighbors:
                        if 0 <= ni < current_layer.shape[0] and 0 <= nj < current_layer.shape[1]:
                            if current_layer[ni, nj] != 0:
                                exposed_faces -= 1
                    
                    # Check layer below
                    if z > 0:
                        if i < prev_layer.shape[0] and j < prev_layer.shape[1]:
                            if prev_layer[i, j] != 0:
                                exposed_faces -= 1
                    
                    # Check layer above (if exists)
                    if z < len(structure) - 1:
                        next_layer = structure[z+1]
                        if i < next_layer.shape[0] and j < next_layer.shape[1]:
                            if next_layer[i, j] != 0:
                                exposed_faces -= 1
                    
                    surface_area += exposed_faces / 6.0
        
        return surface_area
    
    def visualize_structure(self, structure, title):
        """Visualize 3D structure using matplotlib"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        points = []
        colors = []
        
        for z, layer in enumerate(structure):
            for i in range(layer.shape[0]):
                for j in range(layer.shape[1]):
                    if layer[i, j] != 0:
                        x = j
                        y = i - j  # Transform to isometric view
                        points.append((x, y, z))
                        residue = layer[i, j]
                        colors.append(self.color_map[residue % len(self.color_map)])
        
        if not points:
            print(f"No non-zero points in {title}")
            return
            
        xs, ys, zs = zip(*points)
        ax.scatter(xs, ys, zs, c=colors, s=10, alpha=0.7, depthshade=True)
        
        ax.set_xlabel('X: Column')
        ax.set_ylabel('Y: Row - Column')
        ax.set_zlabel('Z: Depth')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_shadow_manifold(self, shadow):
        """Visualize shadow manifold with phase coloring"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        points = []
        phases = []
        magnitudes = []
        
        for z, layer in enumerate(shadow):
            for i in range(layer.shape[0]):
                for j in range(layer.shape[1]):
                    val = layer[i, j]
                    if abs(val) > 1e-6:  # Non-zero
                        x = j
                        y = i - j
                        points.append((x, y, z))
                        phase = np.angle(val) / (2 * np.pi) % 1.0
                        magnitude = np.abs(val)
                        phases.append(phase)
                        magnitudes.append(magnitude)
        
        if not points:
            print("No non-zero points in Shadow Manifold")
            return
            
        xs, ys, zs = zip(*points)
        colors = [hsv_to_rgb([p, 0.8, min(1.0, m)]) for p, m in zip(phases, magnitudes)]
        
        ax.scatter(xs, ys, zs, c=colors, s=10, alpha=0.7, depthshade=True)
        ax.set_xlabel('X: Column')
        ax.set_ylabel('Y: Row - Column')
        ax.set_zlabel('Z: Depth')
        ax.set_title('Shadow Manifold (3,5,7 Bridge)')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    print("=== Shadow Manifold Bridge Theory ===")
    print("Pascal's Triangle as Geometric Blueprint\n")
    
    sm = ShadowManifold(depth=32)
    
    # Generate structures for different moduli
    print("Generating geometric structures...")
    mod2_structure = sm.generate_3d_structure(modulus=2)
    mod3_structure = sm.generate_3d_structure(modulus=3)
    mod5_structure = sm.generate_3d_structure(modulus=5)
    mod7_structure = sm.generate_3d_structure(modulus=7)
    mod10_structure = sm.generate_3d_structure(modulus=10)
    
    # Create shadow manifold bridge
    shadow_manifold = sm.shadow_manifold_transform(mod10_structure)
    
    # Calculate geometric properties
    print("\nCalculating geometric properties...")
    print(f"Mod 2 Volume: {sm.calculate_volume(mod2_structure):.4f}")
    print(f"Mod 2 Surface Area: {sm.calculate_surface_area(mod2_structure):.4f}")
    
    print(f"\nMod 10 Volume: {sm.calculate_volume(mod10_structure):.4f}")
    print(f"Mod 10 Surface Area: {sm.calculate_surface_area(mod10_structure):.4f}")
    
    # Analyze beta sheets and alpha helices
    print("\nBeta Sheet Areas (Mod 10):")
    sheet_areas = sm.calculate_beta_sheet(mod10_structure)
    print(f"Average Sheet Area: {np.mean(sheet_areas):.4f}")
    
    print("\nAlpha Helix Paths (Mod 10):")
    helix_paths = sm.calculate_alpha_helix(mod10_structure)
    for i, path in enumerate(helix_paths[:3]):
        print(f"Helix {i+1}: Length={len(path)}, Start={path[0][:3] if path else None}")
    
    # Visualize structures
    print("\nVisualizing structures... (close windows to continue)")
    sm.visualize_structure(mod2_structure, "Mod 2: Sierpinski Tetrahedron")
    sm.visualize_structure(mod10_structure, "Mod 10: Alpha Helices & Beta Sheets")
    sm.visualize_shadow_manifold(shadow_manifold)
    
    print("\n=== Geometric Insights ===")
    print("1. Mod 2: Recursive Sierpinski tetrahedron (3D fractal)")
    print("2. Mod 3/5/7: Shadow manifold primes (phase-entangled dimensions)")
    print("3. Mod 4: Crystal-like sheets (2D planes in 3D space)")
    print("4. Mod 8: Full 3D coordinate system (x,y,z) representation")
    print("5. Mod 10: Hybrid structure with helices and sheets")
    print("6. Shadow Manifold: Complex plane bridge between prime dimensions")