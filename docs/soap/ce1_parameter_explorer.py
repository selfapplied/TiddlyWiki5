import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import math

def generate_kaleidoscope_variation(angle_deg, translation_x, shadow_lambda, 
                                   lattice_scale=1.0, color_scheme='psychedelic'):
    """Generate a kaleidoscopic pattern with custom parameters"""
    
    width, height = 1024, 1024
    x = np.linspace(-2, 2, width)
    y = np.linspace(-2, 2, height)
    X, Y = np.meshgrid(x, y)
    
    # Lattice parameters with scaling
    a = np.array([1.0 * lattice_scale, 0.0])
    b = np.array([0.5 * lattice_scale, math.sqrt(3)/2 * lattice_scale])
    
    # Transformation parameters
    theta0 = np.radians(angle_deg)
    
    # Apply transformation
    X_rot = X * np.cos(theta0) - Y * np.sin(theta0)
    Y_rot = X * np.sin(theta0) + Y * np.cos(theta0)
    
    R = np.sqrt(X_rot**2 + Y_rot**2) + 1e-10
    Phi = np.arctan2(Y_rot, X_rot)
    
    logR = np.log(R)
    Zx = logR * np.cos(Phi)
    Zy = logR * np.sin(Phi)
    
    # Apply translation
    Zx += translation_x
    
    # Lattice projections
    v = np.stack([Zx, Zy], axis=-1)
    proj_a = (v[..., 0] * a[0] + v[..., 1] * a[1])
    proj_b = (v[..., 0] * b[0] + v[..., 1] * b[1])
    
    # Channels
    S_channel = np.fmod(proj_a * 12.345 + proj_b * 67.891, 1.0)
    M_channel = (np.fmod(np.floor(proj_a) + np.floor(proj_b), 2) >= 1).astype(float)
    L_channel = np.abs(proj_a - 0.5) + np.abs(proj_b - 0.5)
    C_channel = R
    
    # Normalize
    S_norm = (S_channel - S_channel.min()) / (S_channel.max() - S_channel.min())
    M_norm = M_channel
    L_norm = (L_channel - L_channel.min()) / (L_channel.max() - L_channel.min())
    C_norm = (C_channel - C_channel.min()) / (C_channel.max() - C_channel.min())
    
    # Color schemes
    if color_scheme == 'psychedelic':
        Hue = np.fmod(S_norm + L_norm, 1.0)
        Saturation = 0.5 + 0.5 * M_norm
        Value = 0.8 * C_norm + 0.2 * (1 - L_norm)
    elif color_scheme == 'cosmic':
        Hue = np.fmod(0.6 + 0.3 * S_norm, 1.0)
        Saturation = 0.7 + 0.3 * L_norm
        Value = 0.3 + 0.7 * C_norm
    elif color_scheme == 'fire':
        Hue = np.fmod(0.05 + 0.15 * L_norm, 1.0)
        Saturation = 0.8 + 0.2 * M_norm
        Value = 0.4 + 0.6 * S_norm
    else:  # monochrome
        intensity = 0.3 * S_norm + 0.3 * L_norm + 0.4 * C_norm
        return np.stack([intensity, intensity, intensity], axis=-1)
    
    HSV_Image = np.stack([Hue, Saturation, Value], axis=-1)
    RGB_Image = hsv_to_rgb(HSV_Image)
    
    # Shadow effect
    shadow_intensity = np.clip(1 - shadow_lambda * L_norm, 0, 1)
    RGB_Image = RGB_Image * shadow_intensity[..., np.newaxis]
    
    return RGB_Image

def explore_parameter_space():
    """Generate a grid of parameter variations"""
    print("🎨 Exploring CE1 parameter space...")
    
    # Parameter ranges
    angles = [0, 15, 30, 45, 60, 75, 90]
    translations = [0.0, 0.1, 0.2, 0.3]
    shadow_lambdas = [0.1, 0.18, 0.3, 0.5]
    lattice_scales = [0.5, 1.0, 1.5, 2.0]
    
    # Create a comprehensive grid
    fig, axes = plt.subplots(4, 4, figsize=(20, 20), facddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddecolor='black')
    fig.suptitle('CE1 Parameter Space Exploration', color='white', fontsize=20, y=0.95)
    
    # Generate variations
    for i, angle in enumerate(angles[:4]):
        for j, trans in enumerate(translations):
            pattern = generate_kaleidoscope_variation(
                angle, trans, 0.18, 1.0, 'psychedelic'
            )
            
            axes[i, j].imshow(pattern, extent=(-2, 2, -2, 2))
            axes[i, j].set_title(f'θ={angle}°, σ={trans}', color='white', fontsize=10)
            axes[i, j].axis('off')
    
    plt.tight_layout(pad=2)
    plt.savefig('.in/ce1_parameter_exploration.mp4')
    
    # Save individual variations
    print("Saving parameter variations...")
    for angle in angles:
        for trans in translations:
            for shadow in shadow_lambdas:
                pattern = generate_kaleidoscope_variation(
                    angle, trans, shadow, 1.0, 'psychedelic'
                )
                filename = f'.in/ce1_exploration_θ{angle}_σ{trans}_λ{shadow}.png'
                plt.imsave(filename, pattern)
                print(f"Saved: {filename}")

def generate_special_patterns():
    """Generate some mathematically interesting special cases"""
    print("🔬 Generating special mathematical patterns...")
    
    special_cases = [
        # Golden ratio inspired
        {'angle': 137.5, 'trans': 0.618, 'shadow': 0.382, 'scale': 1.618, 'name': 'golden'},
        # Pi inspired
        {'angle': 180, 'trans': 3.14159, 'shadow': 0.314, 'scale': 1.0, 'name': 'pi'},
        # Fibonacci inspired
        {'angle': 144, 'trans': 1.618, 'shadow': 0.236, 'scale': 2.618, 'name': 'fibonacci'},
        # Chaos theory inspired
        {'angle': 45, 'trans': 3.5699, 'shadow': 0.5, 'scale': 0.5, 'name': 'chaos'},
    ]
    
    for case in special_cases:
        pattern = generate_kaleidoscope_variation(
            case['angle'], case['trans'], case['shadow'], 
            case['scale'], 'psychedelic'
        )
        filename = f'.in/ce1_special_{case["name"]}.png'
        plt.imsave(filename, pattern)
        print(f"Saved special case: {filename}")

if __name__ == "__main__":
    print("🚀 CE1 Parameter Explorer")
    print("=" * 50)
    
    # Generate parameter space exploration
    explore_parameter_space()
    
    # Generate special mathematical patterns
    generate_special_patterns()
    
    print("\n✨ Parameter exploration complete!")
    print("Check the .in/ directory for all generated variations.")
