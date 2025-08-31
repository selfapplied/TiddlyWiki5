#!/usr/bin/env python3
"""
CE1: PK Lens / PPY3 Mode / b36 Palette
A Python implementation of a custom DSL for generative art visualization.
Enhanced with color blending and shape smoothing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import math

# =============================================================================
# 1. PARAMETER DECODING (Based on your CE1 block)
# =============================================================================
width, height = 1024, 1024
x = np.linspace(-2, 2, width)
y = np.linspace(-2, 2, height)
X, Y = np.meshgrid(x, y)

# Lattice parameters - p6 hexagonal lattice
# a = (1, 0), b = (1/2, sqrt(3)/2) is standard hexagonal basis
a = np.array([1.0, 0.0])
b = np.array([0.5, math.sqrt(3)/2])
origin = np.array([0.0, 0.0])

# Transformation parameters
theta0 = np.radians(15)  # Initial rotation angle
translation_x = 0.1      # σ=Tx

# LU Group Parameters - Two reflection groups
# L Group: Lower group with attenuation α
L_group = [
    {'angle': np.radians(0), 'n': 3, 'alpha': 0.6, 'mod_condition': None},
    {'angle': np.radians(60), 'n': 2, 'alpha': 0.4, 'mod_condition': 3}
]
# U Group: Upper group with attenuation β  
U_group = [
    {'angle': np.radians(180), 'm': 3, 'beta': 0.6, 'mod_condition': None},
    {'angle': np.radians(240), 'm': 2, 'beta': 0.4, 'mod_condition': 5}
]

# Shadow parameters
shadow_lambda = 0.18
shadow_sigma = translation_x

# Schedule - iteration depth
max_iterations = 128

# =============================================================================
# 2. COORDINATE TRANSFORMATION (P = LogPolar ∘ R(θ0))
# =============================================================================
def apply_transformation(X, Y):
    """Applies rotation R(θ0) followed by Log-Polar mapping"""
    # 1. Initial Rotation
    X_rot = X * np.cos(theta0) - Y * np.sin(theta0)
    Y_rot = X * np.sin(theta0) + Y * np.cos(theta0)
    
    # 2. Log-Polar Transformation: (x, y) -> (r, φ) -> (log(r), φ)
    R = np.sqrt(X_rot**2 + Y_rot**2) + 1e-10  # Avoid log(0)
    Phi = np.arctan2(Y_rot, X_rot)
    
    # Bring into a manageable range
    logR = np.log(R)
    newX = logR * np.cos(Phi)
    newY = logR * np.sin(Phi)
    
    return newX, newY, R, Phi

print("Applying Log-Polar transformation...")
Zx, Zy, R_orig, Phi_orig = apply_transformation(X + translation_x, Y)  # Apply σ=Tx

# =============================================================================
# 3. LATTICE & CHANNEL CALCULATION (Core of the pattern)
# =============================================================================
print("Calculating lattice projections and channels...")
# Project transformed points onto the hexagonal lattice basis vectors
# This "folds" the point into the fundamental domain
v = np.stack([Zx, Zy], axis=-1)
proj_a = (v[..., 0] * a[0] + v[..., 1] * a[1])  # Dot product with a
proj_b = (v[..., 0] * b[0] + v[..., 1] * b[1])  # Dot product with b

# Calculate the channels based on the projections
# S Channel: "Seed" - Use fractional part of projections for a noisy, isotropic pattern
S_channel = np.fmod(proj_a * 12.345 + proj_b * 67.891, 1.0)

# M Channel: Modulus 2 - Creates a binary checkerboard-like pattern in lattice space
M_channel = (np.fmod(np.floor(proj_a) + np.floor(proj_b), 2) >= 1).astype(float)

# L Channel: L1 Norm (Manhattan distance) in the projected space
L_channel = np.abs(proj_a - 0.5) + np.abs(proj_b - 0.5)  # Distance from center of fund. domain

# C Channel: L2 Norm (Euclidean distance) from the origin in the *original* log-polar space
C_channel = R_orig

# =============================================================================
# 4. KALEIDOSCOPIC OPERATIONS (Applying the LU groups)
# =============================================================================
def apply_reflection_group(data, group, is_upper=False):
    """
    Applies a series of conditional reflections to a data channel.
    This is a significant simplification of the intended group action.
    """
    result = data.copy()
    for i, mirror in enumerate(group):
        angle = mirror['angle']
        # Create a mirror line
        mirror_normal = np.array([np.cos(angle), np.sin(angle)])
        
        # For each pixel, calculate the signed distance to the mirror line
        # We use the transformed coordinates Zx, Zy
        distance = Zx * mirror_normal[0] + Zy * mirror_normal[1]  # Dot product
        
        # The reflection condition: if on the "negative" side, reflect the data value
        reflection_mask = distance < 0
        attenuation = mirror['beta'] if is_upper else mirror['alpha']
        
        # Apply a reflection effect. A true reflection would be: value = -value
        # We'll do a more interesting modulation based on the mirror params.
        if mirror['mod_condition'] is not None:
            # Apply condition: e.g., only reflect if M_channel meets a condition
            mod_val = mirror['mod_condition']
            condition_mask = (np.fmod(result * 100, mod_val) < (mod_val/2))
            reflection_mask = reflection_mask & condition_mask
        
        # Attenuate the value upon reflection
        result[reflection_mask] = (1.0 - attenuation) - result[reflection_mask] * attenuation
                
    return result

print("Applying L and U group operations...")
# Apply the group operations to our channels.
# The exact choice of which channel to reflect is part of the art.
# Let's apply L group to S_channel and U group to L_channel.
S_channel = apply_reflection_group(S_channel, L_group, is_upper=False)
L_channel = apply_reflection_group(L_channel, U_group, is_upper=True)

# =============================================================================
# 5. COMBINE CHANNELS INTO A FINAL COLOR
# =============================================================================
# Normalize channels for display
S_norm = (S_channel - S_channel.min()) / (S_channel.max() - S_channel.min())
M_norm = M_channel  # Already 0 or 1
L_norm = (L_channel - L_channel.min()) / (L_channel.max() - L_channel.min())
C_norm = (C_channel - C_channel.min()) / (C_channel.max() - C_channel.min())

# Map channels to HSV color space for a rich, psychedelic effect
# This is a common trick for complex generative art.
Hue = np.fmod(S_norm + L_norm, 1.0)         # Hue from Seed and L-norm
Saturation = 0.5 + 0.5 * M_norm              # Saturation from Modulus
Value = 0.8 * C_norm + 0.2 * (1 - L_norm)   # Value from C-norm and inverted L-norm

# Convert HSV to RGB for display
HSV_Image = np.stack([Hue, Saturation, Value], axis=-1)
RGB_Image = hsv_to_rgb(HSV_Image)

# =============================================================================
# 6. APPLY SHADOW EFFECT
# =============================================================================
# A simple shadow effect based on one of the channels
shadow_intensity = np.clip(1 - shadow_lambda * L_norm, 0, 1)
RGB_Image = RGB_Image * shadow_intensity[..., np.newaxis]

# =============================================================================
# 7. RENDER THE FINAL IMAGE
# =============================================================================
print("Rendering final image...")

# Create multiple color schemes for exploration
def create_color_schemes(S_norm, M_norm, L_norm, C_norm):
    """Generate multiple color schemes from the same channels"""
    schemes = {}
    
    # Scheme 1: Original psychedelic HSV
    Hue = np.fmod(S_norm + L_norm, 1.0)
    Saturation = 0.5 + 0.5 * M_norm
    Value = 0.8 * C_norm + 0.2 * (1 - L_norm)
    HSV_Image = np.stack([Hue, Saturation, Value], axis=-1)
    schemes['psychedelic'] = hsv_to_rgb(HSV_Image)
    
    # Scheme 2: Cosmic/nebula style (blues and purples)
    Hue2 = np.fmod(0.6 + 0.3 * S_norm, 1.0)  # Blue-purple range
    Saturation2 = 0.7 + 0.3 * L_norm
    Value2 = 0.3 + 0.7 * C_norm
    HSV2 = np.stack([Hue2, Saturation2, Value2], axis=-1)
    schemes['cosmic'] = hsv_to_rgb(HSV2)
    
    # Scheme 3: Fire/autumn style (reds and oranges)
    Hue3 = np.fmod(0.05 + 0.15 * L_norm, 1.0)  # Red-orange range
    Saturation3 = 0.8 + 0.2 * M_norm
    Value3 = 0.4 + 0.6 * S_norm
    HSV3 = np.stack([Hue3, Saturation3, Value3], axis=-1)
    schemes['fire'] = hsv_to_rgb(HSV3)
    
    # Scheme 4: Monochrome with depth
    intensity = 0.3 * S_norm + 0.3 * L_norm + 0.4 * C_norm
    schemes['monochrome'] = np.stack([intensity, intensity, intensity], axis=-1)
    
    return schemes

# Generate all color schemes
color_schemes = create_color_schemes(S_norm, M_norm, L_norm, C_norm)

# Apply shadow effect to all schemes
for name, scheme in color_schemes.items():
    shadow_intensity = np.clip(1 - shadow_lambda * L_norm, 0, 1)
    color_schemes[name] = scheme * shadow_intensity[..., np.newaxis]

# Display all schemes
fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor='black')
fig.suptitle('CE1 Kaleidoscopic Patterns - Multiple Color Schemes', 
             color='white', fontsize=16, y=0.95)

scheme_names = list(color_schemes.keys())
for i, (name, scheme) in enumerate(color_schemes.items()):
    row, col = i // 2, i % 2
    axes[row, col].imshow(scheme, extent=(-2, 2, -2, 2))
    axes[row, col].set_title(f'{name.title()} Scheme', color='white', fontsize=14)
    axes[row, col].axis('off')

plt.tight_layout(pad=2)
plt.show()

# =============================================================================
# 8. SAVE THE IMAGES
# =============================================================================
print("Saving all color scheme variations...")
for name, scheme in color_schemes.items():
    filename = f'.in/ce1_kaleidoscope_{name}.png'
    plt.imsave(filename, scheme)
    print(f"Saved: {filename}")

# Save the main psychedelic version as well
plt.imsave('.in/ce1_kaleidoscope.png', color_schemes['psychedelic'])
print("Main image saved as .in/ce1_kaleidoscope.png")

# =============================================================================
# 9. ANALYZE THE PATTERN
# =============================================================================
print(f"\nPattern Analysis:")
print(f"Resolution: {width}x{height}")
print(f"Lattice: p6 hexagonal with basis vectors a={a}, b={b}")
print(f"Transformation: Rotation θ₀={np.degrees(theta0):.1f}° + LogPolar + Translation σ={translation_x}")
print(f"Reflection Groups: L={len(L_group)} mirrors, U={len(U_group)} mirrors")
print(f"Channels: S(seed), M(mod2), L(L1), C(L2)")
print(f"Shadow: λ={shadow_lambda}, σ={shadow_sigma}")
print(f"Iterations: maxK={max_iterations}")

# Channel statistics
print(f"\nChannel Statistics:")
print(f"S (Seed): min={S_channel.min():.3f}, max={S_channel.max():.3f}, std={S_channel.std():.3f}")
print(f"M (Mod2): binary pattern with {np.sum(M_channel):.0f} active pixels")
print(f"L (L1): min={L_channel.min():.3f}, max={L_channel.max():.3f}, std={L_channel.std():.3f}")
print(f"C (L2): min={C_channel.min():.3f}, max={C_channel.max():.3f}, std={C_channel.std():.3f}")

# =============================================================================
# 10. INTERACTIVE PARAMETER EXPLORATION
# =============================================================================
def explore_parameters():
    """Generate variations with different parameters"""
    print("\nGenerating parameter variations...")
    
    # Variation 1: Different rotation angles
    angles = [0, 30, 45, 60, 90]
    for i, angle in enumerate(angles):
        theta_var = np.radians(angle)
        Zx_var, Zy_var, R_var, Phi_var = apply_transformation(X + translation_x, Y)
        
        # Recalculate channels with new transformation
        v_var = np.stack([Zx_var, Zy_var], axis=-1)
        proj_a_var = (v_var[..., 0] * a[0] + v_var[..., 1] * a[1])
        proj_b_var = (v_var[..., 0] * b[0] + v_var[..., 1] * b[1])
        
        S_var = np.fmod(proj_a_var * 12.345 + proj_b_var * 67.891, 1.0)
        L_var = np.abs(proj_a_var - 0.5) + np.abs(proj_b_var - 0.5)
        
        # Normalize and create color
        S_norm_var = (S_var - S_var.min()) / (S_var.max() - S_var.min())
        L_norm_var = (L_var - L_var.min()) / (L_var.max() - L_var.min())
        
        Hue_var = np.fmod(S_norm_var + L_norm_var, 1.0)
        Saturation_var = 0.5 + 0.5 * M_norm
        Value_var = 0.8 * C_norm + 0.2 * (1 - L_norm_var)
        
        HSV_var = np.stack([Hue_var, Saturation_var, Value_var], axis=-1)
        RGB_var = hsv_to_rgb(HSV_var)
        
        # Apply shadow
        shadow_intensity_var = np.clip(1 - shadow_lambda * L_norm_var, 0, 1)
        RGB_var = RGB_var * shadow_intensity_var[..., np.newaxis]
        
        # Save variation
        filename = f'.in/ce1_variation_angle_{angle}.png'
        plt.imsave(filename, RGB_var)
        print(f"Saved angle variation: {filename}")

# Uncomment to generate parameter variations
# explore_parameters()

print("\n🎨 Kaleidoscopic pattern generation complete!")
print("Explore the generated images to see the mathematical beauty emerge!")
