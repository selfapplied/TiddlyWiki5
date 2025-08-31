import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import math

class SeedKernel:
    """Each seed is a computer that defines its own operational parameters"""
    
    def __init__(self, position, seed_value):
        self.position = position
        self.seed_value = seed_value
        
        # Each seed defines its own computational parameters
        self.define_kernels()
        self.define_cuts()
        self.define_timing()
        self.define_shapes()
    
    def define_kernels(self):
        """Kernel functions defined by the seed's computational signature"""
        # Use seed to determine kernel characteristics
        s = self.seed_value
        
        # Convolution kernels - each seed has its own filter bank
        self.conv_kernels = {
            'edge': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) * s,
            'blur': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * (1 - s) / 16,
            'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) * s,
            'emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]) * (s - 0.5)
        }
        
        # Transformation kernels
        self.transform_matrix = np.array([
            [np.cos(s * 2 * np.pi), -np.sin(s * 2 * np.pi)],
            [np.sin(s * 2 * np.pi), np.cos(s * 2 * np.pi)]
        ])
        
        # Frequency kernels for spectral operations
        self.frequencies = {
            'low': s * 5.0,
            'mid': s * 15.0 + 10.0,
            'high': s * 30.0 + 20.0
        }
    
    def define_cuts(self):
        """Branch cuts and discontinuities defined by seed"""
        s = self.seed_value
        
        # Each seed defines where it creates cuts in the mathematical fabric
        self.cut_angles = [
            s * 2 * np.pi,                    # Primary cut
            (s + 0.618) * 2 * np.pi,          # Golden ratio cut
            (s * 3) % (2 * np.pi),            # Harmonic cut
            (s * s) * 2 * np.pi               # Nonlinear cut
        ]
        
        # Cut strengths - how severely they disrupt the field
        self.cut_strengths = [
            s,
            (1 - s) * 0.618,
            s * (1 - s) * 2,
            s * s
        ]
        
        # Cut types define the nature of the discontinuity
        self.cut_types = [
            'reflection',  # Mirror across the cut
            'inversion',   # Invert values across the cut
            'rotation',    # Rotate phase across the cut
            'scaling'      # Scale magnitude across the cut
        ]
    
    def define_timing(self):
        """Temporal evolution parameters defined by seed"""
        s = self.seed_value
        
        # Each seed has its own clock and evolution rate
        self.clock_frequency = s * 10.0 + 1.0
        self.phase_offset = s * 2 * np.pi
        self.evolution_rate = s * 0.1 + 0.01
        
        # Timing functions for different processes
        self.timing_functions = {
            'heartbeat': lambda t: np.sin(t * self.clock_frequency + self.phase_offset),
            'breathing': lambda t: np.cos(t * self.clock_frequency * 0.5 + self.phase_offset),
            'growth': lambda t: np.exp(-t * self.evolution_rate) * np.sin(t * self.clock_frequency),
            'decay': lambda t: np.exp(t * self.evolution_rate * 0.1) * np.cos(t * self.clock_frequency * 2)
        }
        
        # Synchronization with other seeds
        self.sync_range = s * 5.0  # How far this seed's timing influences others
        self.sync_strength = (1 - s) * 0.3  # How strongly it synchronizes
    
    def define_shapes(self):
        """Geometric shapes and patterns defined by seed"""
        s = self.seed_value
        
        # Each seed generates its own geometric vocabulary
        self.primary_shape = self.generate_primary_shape(s)
        self.secondary_shapes = self.generate_secondary_shapes(s)
        
        # Shape transformation parameters
        self.shape_scale = s * 2.0 + 0.5
        self.shape_rotation = s * 2 * np.pi
        self.shape_skew = (s - 0.5) * 0.5
        
        # Fractal parameters for recursive shape generation
        self.fractal_depth = int(s * 5) + 1
        self.fractal_ratio = s * 0.8 + 0.2
        self.fractal_angle = s * np.pi / 3
    
    def generate_primary_shape(self, s):
        """Generate the fundamental shape signature of this seed"""
        # Use seed to determine basic shape type
        shape_type = int(s * 6)
        
        if shape_type == 0:  # Circle
            return lambda r, theta: 1.0
        elif shape_type == 1:  # Triangle
            return lambda r, theta: np.abs(np.sin(3 * theta))
        elif shape_type == 2:  # Square
            return lambda r, theta: np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
        elif shape_type == 3:  # Pentagon
            return lambda r, theta: np.abs(np.sin(5 * theta))
        elif shape_type == 4:  # Star
            return lambda r, theta: np.abs(np.sin(7 * theta)) * (0.5 + 0.5 * np.cos(2 * theta))
        else:  # Complex spiral
            return lambda r, theta: np.exp(-r * s) * np.sin(theta * (1/s + 1))
    
    def generate_secondary_shapes(self, s):
        """Generate harmonics and variations of the primary shape"""
        harmonics = []
        for i in range(1, 4):  # Generate 3 harmonics
            harmonic_freq = i * (s * 5 + 1)
            harmonic_amp = (1 - s) / i
            harmonics.append(lambda r, theta, f=harmonic_freq, a=harmonic_amp: 
                           a * np.sin(f * theta))
        return harmonics
    
    def compute_local_field(self, query_points, time=0.0):
        """Compute the field values at query points using this seed's computer"""
        # Transform query points to seed's local coordinate system
        relative_points = query_points - self.position
        distances = np.linalg.norm(relative_points, axis=-1)
        angles = np.arctan2(relative_points[..., 1], relative_points[..., 0])
        
        # Apply timing functions
        time_factor = self.timing_functions['heartbeat'](time)
        
        # Compute primary shape influence
        shape_value = self.primary_shape(distances, angles)
        
        # Add harmonic contributions
        for harmonic in self.secondary_shapes:
            shape_value += harmonic(distances, angles) * time_factor
        
        # Apply cuts - check if points cross any branch cuts
        field_value = shape_value
        for cut_angle, cut_strength, cut_type in zip(self.cut_angles, self.cut_strengths, self.cut_types):
            cut_mask = self.apply_cut(angles, cut_angle, cut_strength, cut_type)
            field_value = field_value * cut_mask
        
        # Apply distance-based falloff
        influence = np.exp(-distances / (self.shape_scale * 2))
        
        return field_value * influence
    
    def apply_cut(self, angles, cut_angle, strength, cut_type):
        """Apply a branch cut to the field"""
        # Determine which points are affected by the cut
        angle_diff = np.abs(angles - cut_angle)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        
        cut_width = 0.1  # Width of the cut region
        cut_mask = np.ones_like(angles)
        
        # Apply different types of cuts
        in_cut = angle_diff < cut_width
        
        if cut_type == 'reflection':
            cut_mask[in_cut] = -cut_mask[in_cut] * strength
        elif cut_type == 'inversion':
            cut_mask[in_cut] = (1.0 / (cut_mask[in_cut] + 1e-10)) * strength
        elif cut_type == 'rotation':
            cut_mask[in_cut] = cut_mask[in_cut] * np.exp(1j * strength * np.pi).real
        else:  # scaling
            cut_mask[in_cut] = cut_mask[in_cut] * (1 + strength)
        
        return cut_mask

class CE1KernelComputer:
    """The main CE1 system where each point is computed by multiple seed kernels"""
    
    def __init__(self, width=1024, height=1024, num_seeds=64):
        self.width = width
        self.height = height
        self.num_seeds = num_seeds
        
        # Create coordinate grid
        x = np.linspace(-2, 2, width)
        y = np.linspace(-2, 2, height)
        self.X, self.Y = np.meshgrid(x, y)
        self.coordinates = np.stack([self.X, self.Y], axis=-1)
        
        # Generate seed computers distributed across the space
        self.seeds = self.generate_seed_computers()
    
    def generate_seed_computers(self):
        """Generate a collection of seed computers with different characteristics"""
        seeds = []
        
        # Distribute seeds across the space using various patterns
        for i in range(self.num_seeds):
            # Use different distribution strategies
            if i < self.num_seeds // 4:  # Grid pattern
                grid_size = int(np.sqrt(self.num_seeds // 4))
                row = i // grid_size
                col = i % grid_size
                x = -1.5 + 3.0 * col / (grid_size - 1)
                y = -1.5 + 3.0 * row / (grid_size - 1)
            elif i < self.num_seeds // 2:  # Random distribution
                x = np.random.uniform(-2, 2)
                y = np.random.uniform(-2, 2)
            elif i < 3 * self.num_seeds // 4:  # Spiral pattern
                angle = i * 2 * np.pi / (self.num_seeds // 4)
                radius = (i - self.num_seeds // 2) / (self.num_seeds // 4) * 1.5
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
            else:  # Fibonacci spiral
                golden_angle = np.pi * (3 - np.sqrt(5))
                angle = i * golden_angle
                radius = np.sqrt(i - 3 * self.num_seeds // 4) * 0.3
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
            
            position = np.array([x, y])
            seed_value = (i / self.num_seeds + np.random.random() * 0.1) % 1.0
            
            seeds.append(SeedKernel(position, seed_value))
        
        return seeds
    
    def compute_global_field(self, time=0.0):
        """Compute the global field by combining all seed computers"""
        print(f"Computing global field with {len(self.seeds)} seed computers...")
        
        # Initialize field components
        field_real = np.zeros((self.height, self.width))
        field_imag = np.zeros((self.height, self.width))
        field_magnitude = np.zeros((self.height, self.width))
        field_phase = np.zeros((self.height, self.width))
        
        # Each seed contributes to the global field
        for i, seed in enumerate(self.seeds):
            if i % 10 == 0:
                print(f"  Processing seed {i}/{len(self.seeds)}")
            
            # Get this seed's contribution
            local_field = seed.compute_local_field(self.coordinates, time)
            
            # Accumulate different field components
            field_real += local_field.real if hasattr(local_field, 'real') else local_field
            field_imag += local_field.imag if hasattr(local_field, 'imag') else np.sin(local_field)
            field_magnitude += np.abs(local_field)
            field_phase += np.angle(local_field + 1j * np.sin(local_field))
        
        # Normalize fields safely
        def safe_normalize(field):
            field_min, field_max = field.min(), field.max()
            if field_max - field_min > 1e-10:
                return (field - field_min) / (field_max - field_min)
            else:
                return np.ones_like(field) * 0.5
        
        field_real = safe_normalize(field_real)
        field_imag = safe_normalize(field_imag)
        field_magnitude = safe_normalize(field_magnitude)
        field_phase = safe_normalize(field_phase)
        
        return {
            'real': field_real,
            'imag': field_imag,
            'magnitude': field_magnitude,
            'phase': field_phase
        }
    
    def generate_pattern(self, time=0.0, color_scheme='quantum'):
        """Generate the final kaleidoscopic pattern"""
        # Compute the global field from all seed computers
        fields = self.compute_global_field(time)
        
        # Create color mapping based on field components
        if color_scheme == 'quantum':
            # Quantum-inspired coloring using field components
            hue = fields['phase']
            saturation = fields['magnitude']
            value = 0.3 + 0.7 * fields['real']
        elif color_scheme == 'interference':
            # Interference pattern coloring
            hue = np.fmod(fields['real'] + fields['imag'], 1.0)
            saturation = 0.8 + 0.2 * fields['magnitude']
            value = 0.5 + 0.5 * np.sin(fields['phase'] * 2 * np.pi)
        elif color_scheme == 'computational':
            # Show the computational structure
            hue = fields['real']
            saturation = fields['imag']
            value = fields['magnitude']
        else:  # 'holographic'
            # Holographic-style coloring
            hue = fields['phase']
            saturation = np.full_like(fields['phase'], 0.9)
            value = 0.2 + 0.8 * fields['magnitude'] * fields['real']
        
        # Convert HSV to RGB
        hsv_image = np.stack([hue, saturation, value], axis=-1)
        rgb_image = hsv_to_rgb(hsv_image)
        
        return rgb_image
    
    def render_kernel_map(self):
        """Visualize the seed kernel locations and their influence"""
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        
        # Show the background field
        background = self.generate_pattern(color_scheme='computational')
        ax.imshow(background, extent=(-2, 2, -2, 2), alpha=0.3)
        
        # Plot seed locations
        for i, seed in enumerate(self.seeds):
            x, y = seed.position
            # Color code by seed value
            color = plt.cm.viridis(seed.seed_value)
            ax.scatter(x, y, c=[color], s=100, alpha=0.8, edgecolors='white', linewidth=1)
            
            # Show influence radius
            circle = plt.Circle((x, y), seed.shape_scale, fill=False, 
                              color=color, alpha=0.3, linewidth=1)
            ax.add_patch(circle)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_title('CE1 Seed Computer Distribution', color='white', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """Generate CE1 patterns using the kernel computer architecture"""
    print("🚀 CE1 Kernel Computer System")
    print("=" * 50)
    print("Each seed is a computer defining kernels, cuts, timing, and shapes")
    
    # Create the kernel computer system
    ce1_system = CE1KernelComputer(width=1024, height=1024, num_seeds=64)
    
    # Show the kernel distribution
    print("\n📍 Visualizing seed computer distribution...")
    ce1_system.render_kernel_map()
    
    # Generate patterns with different color schemes
    color_schemes = ['quantum', 'interference', 'computational', 'holographic']
    
    for scheme in color_schemes:
        print(f"\n🎨 Generating {scheme} pattern...")
        pattern = ce1_system.generate_pattern(color_scheme=scheme)
        
        # Display pattern
        plt.figure(figsize=(12, 12), facecolor='black')
        plt.imshow(pattern, extent=(-2, 2, -2, 2))
        plt.title(f'CE1 Kernel Computer - {scheme.title()} Scheme', 
                 color='white', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Save pattern
        filename = f'.in/ce1_kernel_{scheme}.png'
        plt.imsave(filename, pattern)
        print(f"Saved: {filename}")
    
    # Generate a time evolution sequence
    print("\n⏰ Generating time evolution sequence...")
    time_steps = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), facecolor='black')
    fig.suptitle('CE1 Kernel Computer - Time Evolution', color='white', fontsize=20)
    
    for i, t in enumerate(time_steps):
        pattern = ce1_system.generate_pattern(time=t, color_scheme='quantum')
        axes[i].imshow(pattern, extent=(-2, 2, -2, 2))
        axes[i].set_title(f't = {t}', color='white', fontsize=14)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n✨ Kernel computer pattern generation complete!")
    print("Each pattern emerges from the collective computation of distributed seed kernels.")

if __name__ == "__main__":
    main()
