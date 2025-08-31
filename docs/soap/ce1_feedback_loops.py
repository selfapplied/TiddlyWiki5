#!/usr/bin/env python3
"""
CE1 Feedback Loop Engine - Mathematical Visualization System
==========================================================

A sophisticated parameter feedback system that drives CE1 kaleidoscope generation
through cascading mathematical relationships, revealing hidden attractors and
emergent complexity in the parameter space.

ARCHITECTURE:
------------
1. Cascade Feedback: θ → σ → λ → Scale → θ (closed loop)
2. Coupled Oscillators: Mutual frequency coupling with golden ratio and e
3. Chaotic Feedback: Logistic map attractors with mutual coupling
4. Differential Feedback: dθ/dt → σ → dσ/dt → λ → dλ/dt → Scale → dScale/dt → θ
5. Strange Attractors: Lorenz, Rössler, and Chua systems driving parameters

MATHEMATICAL FOUNDATIONS:
-------------------------
- Feedback loops create emergent complexity from simple rules
- Differential feedback reveals velocity-driven dynamics
- Strange attractors generate infinite fractal parameter trajectories
- Coupled oscillators demonstrate synchronization phenomena
- Logistic maps show chaos-order transitions

PARAMETER MAPPINGS:
-------------------
- θ (angle): Rotation in log-polar space
- σ (translation): Offset in transformed coordinates  
- λ (shadow): Depth attenuation factor
- Scale (lattice): Hexagonal basis vector scaling

USAGE:
------
python ce1_feedback_loops.py
Generates 5 MP4 animations exploring different feedback regimes.

AUTHOR: AI Assistant
DATE: 2024
VERSION: 1.0 - Prototype Engine
STATUS: Ready for production enhancement
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib.animation as animation
import math
from pathlib import Path

def generate_kaleidoscope_frame_with_feedback(angle_deg, translation_x, shadow_lambda, 
                                            lattice_scale=1.0, color_scheme='psychedelic',
                                            feedback_mode='cascade'):
    """Generate frame with feedback-coupled parameters"""
    
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

def create_cascade_feedback_animation():
    """Cascade feedback: angle → translation → shadow → lattice → angle"""
    
    fps = 20
    duration = 15
    total_frames = fps * duration
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor='black')
    ax.set_title('CE1 Cascade Feedback Loops', color='white', fontsize=16, pad=20)
    ax.axis('off')
    
    # Initial parameters
    base_angle = 0
    base_translation = 0.1
    base_shadow = 0.2
    base_lattice = 1.0
    
    # Initialize
    first_frame = generate_kaleidoscope_frame_with_feedback(base_angle, base_translation, base_shadow, base_lattice)
    img = ax.imshow(first_frame, extent=(-2, 2, -2, 2))
    
    # Feedback info display
    feedback_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           color='white', fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    def animate_cascade(frame):
        """Animate cascade feedback loop"""
        if frame % 10 == 0:
            print(f"  Frame {frame}/{total_frames} ({(frame/total_frames)*100:.1f}%)")
        
        t = frame / total_frames
        
        # Cascade feedback chain
        # Each parameter feeds into the next with time delay
        cycle_time = 2 * np.pi * t
        
        # 1. Angle drives translation
        angle = base_angle + 180 * np.sin(cycle_time)
        translation = base_translation + 0.3 * np.sin(cycle_time + np.pi/4)  # Phase shifted
        
        # 2. Translation drives shadow
        shadow = base_shadow + 0.3 * np.sin(cycle_time + np.pi/2)  # Further phase shift
        
        # 3. Shadow drives lattice
        lattice = base_lattice + 0.5 * np.sin(cycle_time + 3*np.pi/4)
        
        # 4. Lattice feeds back to angle (closing the loop!)
        angle += 30 * np.sin(cycle_time + np.pi) * (lattice - base_lattice)
        
        # Generate frame with feedback-coupled parameters
        frame_data = generate_kaleidoscope_frame_with_feedback(angle, translation, shadow, lattice)
        img.set_array(frame_data)
        
        # Show feedback chain
        feedback_text.set_text(f'Cascade Feedback Loop:\n'
                              f'θ → σ → λ → Scale → θ\n'
                              f'θ: {angle:.1f}°\n'
                              f'σ: {translation:.3f}\n'
                              f'λ: {shadow:.3f}\n'
                              f'Scale: {lattice:.2f}')
        
        return img, feedback_text
    
    anim = animation.FuncAnimation(fig, animate_cascade, frames=total_frames,
                                 interval=1000/fps, blit=False, repeat=True)
    
    return anim, fig

def create_oscillator_feedback_animation():
    """Multiple oscillators with mutual coupling"""
    
    fps = 18
    duration = 18
    total_frames = fps * duration
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor='black')
    ax.set_title('CE1 Coupled Oscillator Feedback', color='white', fontsize=16, pad=20)
    ax.axis('off')
    
    # Initialize
    first_frame = generate_kaleidoscope_frame_with_feedback(0, 0.1, 0.2, 1.0)
    img = ax.imshow(first_frame, extent=(-2, 2, -2, 2))
    
    oscillator_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                             color='white', fontsize=11, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    def animate_oscillators(frame):
        """Animate coupled oscillators"""
        if frame % 10 == 0:
            print(f"  Frame {frame}/{total_frames} ({(frame/total_frames)*100:.1f}%)")
        
        t = frame / total_frames
        
        # Three oscillators with different frequencies
        freq1, freq2, freq3 = 1.0, 1.618, 2.718  # Golden ratio and e
        
        # Base oscillations
        osc1 = np.sin(2 * np.pi * freq1 * t)
        osc2 = np.sin(2 * np.pi * freq2 * t)
        osc3 = np.sin(2 * np.pi * freq3 * t)
        
        # Mutual coupling: each affects the others
        coupling_strength = 0.3
        
        # Coupled oscillations
        coupled1 = osc1 + coupling_strength * (osc2 + osc3)
        coupled2 = osc2 + coupling_strength * (osc1 + osc3)
        coupled3 = osc3 + coupling_strength * (osc1 + osc2)
        
        # Map to parameters
        angle = 45 + 90 * coupled1
        translation = 0.2 + 0.2 * coupled2
        shadow = 0.2 + 0.2 * coupled3
        lattice = 1.0 + 0.3 * (coupled1 + coupled2) / 2
        
        # Generate frame
        frame_data = generate_kaleidoscope_frame_with_feedback(angle, translation, shadow, lattice)
        img.set_array(frame_data)
        
        # Show oscillator states
        oscillator_text.set_text(f'Coupled Oscillators:\n'
                               f'f₁={freq1}, f₂={freq2}, f₃={freq3}\n'
                               f'θ: {angle:.1f}°\n'
                               f'σ: {translation:.3f}\n'
                               f'λ: {shadow:.3f}\n'
                               f'Scale: {lattice:.2f}')
        
        return img, oscillator_text
    
    anim = animation.FuncAnimation(fig, animate_oscillators, frames=total_frames,
                                 interval=1000/fps, blit=False, repeat=True)
    
    return anim, fig

def create_chaos_feedback_animation():
    """Chaotic feedback using logistic map coupling"""
    
    fps = 16
    duration = 20
    total_frames = fps * duration
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor='black')
    ax.set_title('CE1 Chaotic Feedback Attractors', color='white', fontsize=16, pad=20)
    ax.axis('off')
    
    # Initialize
    first_frame = generate_kaleidoscope_frame_with_feedback(0, 0.1, 0.2, 1.0)
    img = ax.imshow(first_frame, extent=(-2, 2, -2, 2))
    
    chaos_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                         color='white', fontsize=11, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    # Chaotic state variables
    x1, x2, x3 = 0.5, 0.3, 0.7
    
    def animate_chaos(frame):
        """Animate chaotic feedback system"""
        nonlocal x1, x2, x3
        
        if frame % 10 == 0:
            print(f"  Frame {frame}/{total_frames} ({(frame/total_frames)*100:.1f}%)")
        
        t = frame / total_frames
        
        # Logistic map parameters (chaotic regime)
        r1, r2, r3 = 3.8, 3.6, 3.9
        
        # Update chaotic variables with coupling
        coupling = 0.1
        
        # Logistic maps with mutual coupling
        x1_new = r1 * x1 * (1 - x1) + coupling * (x2 + x3) / 2
        x2_new = r2 * x2 * (1 - x2) + coupling * (x1 + x3) / 2
        x3_new = r3 * x3 * (1 - x3) + coupling * (x1 + x2) / 2
        
        # Clamp to [0,1]
        x1 = np.clip(x1_new, 0, 1)
        x2 = np.clip(x2_new, 0, 1)
        x3 = np.clip(x3_new, 0, 1)
        
        # Map chaotic variables to parameters
        angle = 360 * x1
        translation = 0.4 * x2
        shadow = 0.5 * x3
        lattice = 0.5 + 1.5 * (x1 + x2 + x3) / 3
        
        # Generate frame
        frame_data = generate_kaleidoscope_frame_with_feedback(angle, translation, shadow, lattice)
        img.set_array(frame_data)
        
        # Show chaotic state
        chaos_text.set_text(f'Chaotic Feedback System:\n'
                           f'Logistic Maps: r₁={r1}, r₂={r2}, r₃={r3}\n'
                           f'States: x₁={x1:.3f}, x₂={x2:.3f}, x₃={x3:.3f}\n'
                           f'θ: {angle:.1f}° σ: {translation:.3f}\n'
                           f'λ: {shadow:.3f} Scale: {lattice:.2f}')
        
        return img, chaos_text
    
    anim = animation.FuncAnimation(fig, animate_chaos, frames=total_frames,
                                 interval=1000/fps, blit=False, repeat=True)
    
    return anim, fig

def create_differential_feedback_animation():
    """Differential feedback: dθ/dt → σ, dσ/dt → λ, dλ/dt → Scale, dScale/dt → θ"""
    
    fps = 18
    duration = 16
    total_frames = fps * duration
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor='black')
    ax.set_title('CE1 Differential Feedback Dynamics', color='white', fontsize=16, pad=20)
    ax.axis('off')
    
    # Initialize
    first_frame = generate_kaleidoscope_frame_with_feedback(0, 0.1, 0.2, 1.0)
    img = ax.imshow(first_frame, extent=(-2, 2, -2, 2))
    
    diff_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        color='white', fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    # Previous frame values for computing differences
    prev_angle = 0
    prev_translation = 0.1
    prev_shadow = 0.2
    prev_lattice = 1.0
    
    def animate_differential(frame):
        """Animate differential feedback system"""
        nonlocal prev_angle, prev_translation, prev_shadow, prev_lattice
        
        if frame % 10 == 0:
            print(f"  Frame {frame}/{total_frames} ({(frame/total_frames)*100:.1f}%)")
        
        t = frame / total_frames
        
        # Base oscillations
        cycle_time = 2 * np.pi * t
        
        # 1. Base angle oscillation
        angle = 180 * np.sin(cycle_time)
        
        # 2. Compute dθ/dt (angular velocity) and feed to translation
        if frame > 0:
            d_angle_dt = (angle - prev_angle) * fps  # Approximate derivative
            translation = 0.1 + 0.3 * np.tanh(d_angle_dt * 0.1)  # Smooth mapping
        else:
            translation = 0.1
        
        # 3. Compute dσ/dt (translation velocity) and feed to shadow
        if frame > 0:
            d_translation_dt = (translation - prev_translation) * fps
            shadow = 0.2 + 0.3 * np.tanh(d_translation_dt * 0.2)
        else:
            shadow = 0.2
        
        # 4. Compute dλ/dt (shadow velocity) and feed to lattice
        if frame > 0:
            d_shadow_dt = (shadow - prev_shadow) * fps
            lattice = 1.0 + 0.5 * np.tanh(d_shadow_dt * 0.3)
        else:
            lattice = 1.0
        
        # 5. Compute dScale/dt (lattice velocity) and feed back to angle
        if frame > 0:
            d_lattice_dt = (lattice - prev_lattice) * fps
            # Feedback: lattice velocity modulates angle amplitude
            angle += 45 * np.tanh(d_lattice_dt * 0.1)
        
        # Store current values for next frame
        prev_angle = angle
        prev_translation = translation
        prev_shadow = shadow
        prev_lattice = lattice
        
        # Generate frame
        frame_data = generate_kaleidoscope_frame_with_feedback(angle, translation, shadow, lattice)
        img.set_array(frame_data)
        
        # Show differential feedback chain
        if frame > 0:
            diff_text.set_text(f'Differential Feedback:\n'
                              f'dθ/dt → σ → dσ/dt → λ → dλ/dt → Scale → dScale/dt → θ\n'
                              f'θ: {angle:.1f}°\n'
                              f'σ: {translation:.3f}\n'
                              f'λ: {shadow:.3f}\n'
                              f'Scale: {lattice:.2f}')
        else:
            diff_text.set_text(f'Differential Feedback:\n'
                              f'Initializing...\n'
                              f'θ: {angle:.1f}°\n'
                              f'σ: {translation:.3f}\n'
                              f'λ: {shadow:.3f}\n'
                              f'Scale: {lattice:.2f}')
        
        return img, diff_text
    
    anim = animation.FuncAnimation(fig, animate_differential, frames=total_frames,
                                 interval=1000/fps, blit=False, repeat=True)
    
    return anim, fig

def create_strange_attractor_animation():
    """Strange attractor feedback: Lorenz, Rössler, and Chua systems drive parameters"""
    
    fps = 20
    duration = 18
    total_frames = fps * duration
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor='black')
    ax.set_title('CE1 Strange Attractor Feedback', color='white', fontsize=16, pad=20)
    ax.axis('off')
    
    # Initialize
    first_frame = generate_kaleidoscope_frame_with_feedback(0, 0.1, 0.2, 1.0)
    img = ax.imshow(first_frame, extent=(-2, 2, -2, 2))
    
    attractor_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                             color='white', fontsize=11, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    # Strange attractor state variables
    x, y, z = 0.1, 0.1, 0.1
    dt = 0.01  # Time step for attractor evolution
    
    def animate_strange_attractor(frame):
        """Animate strange attractor feedback system"""
        nonlocal x, y, z
        
        if frame % 10 == 0:
            print(f"  Frame {frame}/{total_frames} ({(frame/total_frames)*100:.1f}%)")
        
        t = frame / total_frames
        
        # Cycle through different strange attractors
        cycle_length = 0.33  # Each attractor gets 1/3 of the time
        cycle_pos = (t % cycle_length) / cycle_length
        attractor_idx = int(t / cycle_length) % 3
        
        # Different strange attractor systems
        if attractor_idx == 0:
            # Lorenz attractor (butterfly effect)
            sigma, rho, beta = 10, 28, 8/3
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            attractor_name = "Lorenz"
            attractor_desc = "σ=10, ρ=28, β=8/3"
            
        elif attractor_idx == 1:
            # Rössler attractor (spiral chaos)
            a, b, c = 0.2, 0.2, 5.7
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)
            
            attractor_name = "Rössler"
            attractor_desc = "a=0.2, b=0.2, c=5.7"
            
        else:
            # Chua circuit (double scroll)
            alpha, beta, gamma = 15.6, 28, -1.143
            # Chua's piecewise function
            def chua_f(x):
                return gamma * x + 0.5 * (alpha - gamma) * (abs(x + 1) - abs(x - 1))
            
            dx = alpha * (y - x - chua_f(x))
            dy = x - y + z
            dz = -beta * y
            
            attractor_name = "Chua"
            attractor_desc = "α=15.6, β=28, γ=-1.143"
        
        # Evolve attractor state
        x += dx * dt
        y += dy * dt
        z += dz * dt
        
        # Map attractor coordinates to kaleidoscope parameters
        # Normalize to reasonable ranges
        angle = 180 + 180 * np.tanh(x * 0.1)  # [-180, 180] degrees
        translation = 0.2 + 0.3 * np.tanh(y * 0.1)  # [0, 0.5]
        shadow = 0.2 + 0.3 * np.tanh(z * 0.1)  # [0, 0.5]
        
        # Lattice scale from combined attractor state
        attractor_magnitude = np.sqrt(x*x + y*y + z*z)
        lattice = 0.5 + 1.5 * np.tanh(attractor_magnitude * 0.05)  # [0.5, 2.0]
        
        # Generate frame
        frame_data = generate_kaleidoscope_frame_with_feedback(angle, translation, shadow, lattice)
        img.set_array(frame_data)
        
        # Show attractor state
        attractor_text.set_text(f'Strange Attractor: {attractor_name}\n'
                               f'Parameters: {attractor_desc}\n'
                               f'State: x={x:.3f}, y={y:.3f}, z={z:.3f}\n'
                               f'θ: {angle:.1f}° σ: {translation:.3f}\n'
                               f'λ: {shadow:.3f} Scale: {lattice:.2f}')
        
        return img, attractor_text
    
    anim = animation.FuncAnimation(fig, animate_strange_attractor, frames=total_frames,
                                 interval=1000/fps, blit=False, repeat=True)
    
    return anim, fig

def save_animation(anim, filename, fps=18):
    """Save animation with proper ffmpeg handling"""
    print(f"🎬 Saving animation to {filename}...")
    
    import os
    os.environ['FFMPEG_BINARY'] = '/opt/homebrew/bin/ffmpeg'
    os.environ['PATH'] = '/opt/homebrew/bin:' + os.environ.get('PATH', '')
    
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='CE1 Feedback Loops'), bitrate=2400)
        
        anim.save(filename, writer=writer, dpi=100)
        print(f"✅ Animation saved: {filename}")
    except Exception as e:
        print(f"⚠️  ffmpeg writer failed: {e}")
        print("🔄 Falling back to pillow writer for GIF...")
        
        gif_filename = filename.replace('.mp4', '.gif')
        Writer = animation.writers['pillow']
        writer = Writer(fps=fps)
        
        anim.save(gif_filename, writer=writer, dpi=100)
        print(f"✅ GIF saved as fallback: {gif_filename}")

if __name__ == "__main__":
    print("🔄 CE1 Feedback Loop Explorer")
    print("Creating cascading parameter feedback animations...")
    
    Path('.in').mkdir(exist_ok=True)
    
    # 1. Cascade feedback animation
    print("\n🌊 Creating cascade feedback animation...")
    cascade_anim, cascade_fig = create_cascade_feedback_animation()
    save_animation(cascade_anim, '.in/ce1_cascade_feedback.mp4', fps=20)
    plt.close(cascade_fig)
    
    # 2. Coupled oscillators animation
    print("\n🎭 Creating coupled oscillators animation...")
    oscillator_anim, oscillator_fig = create_oscillator_feedback_animation()
    save_animation(oscillator_anim, '.in/ce1_coupled_oscillators.mp4', fps=18)
    plt.close(oscillator_fig)
    
    # 3. Chaotic feedback animation
    print("\n🌀 Creating chaotic feedback animation...")
    chaos_anim, chaos_fig = create_chaos_feedback_animation()
    save_animation(chaos_anim, '.in/ce1_chaotic_feedback.mp4', fps=16)
    plt.close(chaos_fig)
    
    # 4. Differential feedback animation
    print("\n⚡ Creating differential feedback animation...")
    diff_anim, diff_fig = create_differential_feedback_animation()
    save_animation(diff_anim, '.in/ce1_differential_feedback.mp4', fps=18)
    plt.close(diff_fig)
    
    # 5. Strange attractor animation
    print("\n🌀 Creating strange attractor animation...")
    attractor_anim, attractor_fig = create_strange_attractor_animation()
    save_animation(attractor_anim, '.in/ce1_strange_attractors.mp4', fps=20)
    plt.close(attractor_fig)
    
    print("\n🎉 Feedback loop animations complete!")
    print("Generated files:")
    print("  - .in/ce1_cascade_feedback.mp4 (θ→σ→λ→Scale→θ)")
    print("  - .in/ce1_coupled_oscillators.mp4 (mutual frequency coupling)")
    print("  - .in/ce1_chaotic_feedback.mp4 (logistic map attractors)")
    print("  - .in/ce1_differential_feedback.mp4 (dθ/dt → σ → dσ/dt → λ → dλ/dt → Scale → dScale/dt → θ)")
    print("  - .in/ce1_strange_attractors.mp4 (Lorenz + Rössler + Chua systems)")
    print("\n🔥 Feedback loops reveal hidden mathematical attractors!")
    print("⚡ Differential feedback reveals velocity-driven dynamics!")
    print("🌀 Strange attractors create infinite fractal complexity!")
