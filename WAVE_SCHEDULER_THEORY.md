# Wave Scheduler: Theoretical Foundation

## Abstract

The Wave Scheduler is a new programming primitive that transforms scheduling from a procedural sequence of events into a dynamical system evolving under an operator. This document explores the mathematical foundations, theoretical implications, and connections to the broader CE Tower operator algebra framework.

## 1. From Loops to Dynamical Systems

### Traditional Scheduling

In conventional programming, scheduling is implemented as:
- **Loops**: `for (i=0; i<n; i++)`
- **Timers**: `setTimeout(fn, delay)`
- **Event queues**: Explicit lists of scheduled events

All share a common limitation: they are **sequences**, not **laws**.

### Wave Scheduling

A wave scheduler replaces sequences with evolution:

```
state[n+1] = E(state[n])
event_time[n] = f(state[n])
```

This is fundamentally a **discrete dynamical system** where:
- `E: State → State` is the evolution operator
- `state[n]` is the system state at generation n
- `f: State → ℝ` is the observation/sampling function

The schedule emerges from **sampling the orbit** of E.

## 2. Linear Recurrence Relations as Operators

### The Fibonacci Example

Consider the Fibonacci recurrence:
```
y_n = y_{n-1} + y_{n-2}
```

This is a 2nd-order linear recurrence, which can be written as a matrix operator:

```
[y_n    ]   [1 1] [y_{n-1}]
[y_{n-1}] = [1 0] [y_{n-2}]
```

Or simply: `v_n = E v_{n-1}` where E is the Fibonacci operator.

### General k-Order Recurrence

Any k-order linear recurrence:
```
y_n = a_1·y_{n-1} + a_2·y_{n-2} + ... + a_k·y_{n-k}
```

Becomes a k×k companion matrix:

```
E = [a_1  a_2  a_3  ...  a_k  ]
    [1    0    0    ...  0    ]
    [0    1    0    ...  0    ]
    [⋮    ⋮    ⋮    ⋱   ⋮    ]
    [0    0    0    ...  0    ]
```

### Spectral Decomposition

Every such operator E has an eigenbasis:
```
E = P Λ P⁻¹
```

Where:
- Λ is diagonal with eigenvalues λ_1, ..., λ_k
- P contains the corresponding eigenvectors

This means the evolution can be written as:
```
v_n = Σᵢ cᵢ λᵢⁿ vᵢ
```

Each eigenvalue λᵢ contributes a mode with:
- **Growth/decay rate**: |λᵢ|
- **Frequency**: arg(λᵢ) for complex λᵢ
- **Phase**: Determined by initial conditions

### The Golden Ratio

For Fibonacci, the eigenvalues are:
```
λ₁ = φ = (1 + √5)/2 ≈ 1.618  (golden ratio)
λ₂ = ψ = (1 - √5)/2 ≈ -0.618
```

The Fibonacci sequence is dominated by φⁿ as n → ∞.

This means **Fibonacci scheduling is exponential growth at golden ratio rate**.

## 3. Harmonic Oscillators

### Rotation Operators

A 2D rotation by angle θ:

```
E = [cos(θ)  -sin(θ)]
    [sin(θ)   cos(θ)]
```

Has complex eigenvalues: λ = e^(±iθ)

The evolution is circular motion:
```
[x_n]   [cos(nθ)  -sin(nθ)] [x_0]
[y_n] = [sin(nθ)   cos(nθ)] [y_0]
```

Sampling x_n gives a pure sinusoid with period 2π/θ.

### Physical Interpretation

A harmonic scheduler is equivalent to:
- A mass on a spring (in discrete time)
- Simple harmonic motion
- Pure periodic behavior without growth or decay

This is the **fundamental oscillator** - the building block of all periodic patterns.

## 4. Damped Oscillators and Spring Physics

### Continuous Differential Equation

The damped harmonic oscillator satisfies:
```
d²x/dt² + 2ζω₀(dx/dt) + ω₀²x = 0
```

Where:
- ω₀ is natural frequency
- ζ is damping ratio

### Discretization

Discretizing with Euler method:
```
v_{n+1} = v_n - kx_n - γv_n
x_{n+1} = x_n + v_n Δt
```

This becomes a 2D linear operator:
```
E = [1-γ   -k]
    [Δt    1 ]
```

The eigenvalues depend on k and γ:
- **Underdamped** (ζ < 1): Complex eigenvalues → oscillation with decay
- **Critically damped** (ζ = 1): Real repeated eigenvalue → fastest settling
- **Overdamped** (ζ > 1): Two real eigenvalues → slow approach

### Applications

Damped oscillators give natural settling behavior for:
- Animation easing
- UI transitions
- Control systems
- Convergence to steady state

## 5. Nonlinear Operators and Chaos

### Beyond Linear Recurrence

Wave schedulers need not be linear. Any function E: State → State works.

Examples:
- **Logistic map**: `x_{n+1} = rx_n(1-x_n)`
- **Neural activation**: `x_{n+1} = σ(Wx_n + b)`
- **Discrete PDEs**: Evolution of field distributions

### Chaotic Scheduling

Some nonlinear operators exhibit chaos:
- Sensitive dependence on initial conditions
- Aperiodic behavior
- Deterministic but unpredictable

This could be used for:
- Randomized timing (without RNG)
- Breaking regularity to avoid resonance
- Natural jitter in systems

## 6. Composition and Superposition

### Linear Superposition

For linear operators, multiple waves combine:
```
state = Σᵢ wᵢ stateᵢ
```

This is the **principle of superposition** from physics.

### Interference Patterns

Two harmonic schedulers with frequencies ω₁, ω₂ create:
- **Beat frequency**: |ω₁ - ω₂|
- **Carrier frequency**: (ω₁ + ω₂)/2

This gives natural amplitude modulation.

### Mode Coupling

In composite schedulers, different modes can interact:
- Fibonacci + Harmonic = Modulated exponential growth
- Multiple harmonics = Complex rhythms
- Damped + Harmonic = Decaying oscillation

## 7. Phase Space and Trajectories

### State Space Geometry

The evolution v_{n+1} = E(v_n) traces a trajectory in state space.

For a k-order recurrence, this is a curve in ℝᵏ.

Key concepts:
- **Fixed points**: States where E(v) = v
- **Periodic orbits**: Cycles E^k(v) = v
- **Attractors**: States the system tends toward
- **Basin of attraction**: Initial conditions leading to an attractor

### Phase Portraits

Visualizing the state space reveals:
- Spiral trajectories (complex eigenvalues)
- Straight-line trajectories (real eigenvalues)
- Limit cycles (periodic behavior)
- Strange attractors (chaos)

### Initial Conditions as Phase

The "phase" in wave scheduler is the initial condition v_0.

Different phases → different trajectories in the same dynamical system.

This is analogous to:
- Starting angle in a pendulum
- Initial displacement in a spring
- Phase offset in a sine wave

## 8. Connection to CE Tower Architecture

### CE1: Compositional Grammar

The operator E defines the **compositional structure**.

For linear recurrence:
```
y_n = Σᵢ aᵢ y_{n-i}
```

This is a weighted composition of past states.

The coefficients {aᵢ} define the grammar - the rule for combining history.

### CE2: Guardian-Mediated Dynamics

Guardian operators (φ, ∂, ℛ) can modulate E:

```
E' = E + φ·δE_semantic + ∂·δE_structural + ℛ·δE_invariant
```

The scheduler adapts based on:
- **φ**: Semantic compatibility of scheduled events
- **∂**: Structural coherence of timing
- **ℛ**: Preservation of system invariants

This makes scheduling **context-aware**.

### CE3: Self-Evolving Grammar

The operator itself can evolve:
```
E_{n+1} = 𝔈(E_n, state_n)
```

Where 𝔈 is the error-lift operator.

The scheduler **learns** from:
- Pattern detection in event sequences
- Optimization of timing based on outcomes
- Adaptation to changing system dynamics

### Full CE Scheduler

A CE-based wave scheduler has state:
```
{
  ce1: grammar_phase,    // Base compositional rhythm
  ce2: guardian_phase,   // Dynamic modulation
  ce3: evolution_phase   // Adaptive component
}
```

Evolution:
```
ce1_{n+1} = E_1(ce1_n)
ce2_{n+1} = E_2(ce2_n, ce1_{n+1})
ce3_{n+1} = E_3(ce3_n, ce1_{n+1}, ce2_{n+1})
```

Sample:
```
time_n = ce1_n + β·ce2_n + γ·ce3_n
```

This creates **multi-level adaptive scheduling** where:
- CE1 provides base rhythm
- CE2 modulates based on current context
- CE3 evolves long-term patterns

## 9. Bracket Calculus Representation

### Operator Brackets

In bracket notation:
```
E = [operator]
p = <phase>
f = (sample)
```

A scheduler is:
```
S = {[E], <p>, (f)}
```

### Evolution Expression

The nth generation is:
```
state_n = [E]^n <p>
```

And the event time is:
```
time_n = (f) [E]^n <p>
```

### Composition

Multiple schedulers compose as:
```
S_composite = Σᵢ wᵢ {[Eᵢ], <pᵢ>, (fᵢ)}
```

This is naturally expressed in tensor notation.

## 10. Applications Beyond Scheduling

### Differential Equations as Operators

Any ODE can be discretized:
```
dx/dt = F(x, t)
→
x_{n+1} = x_n + F(x_n, n)·Δt
```

This is a wave scheduler with:
- Operator: Euler step
- Phase: Initial condition x_0
- Sample: Any projection of state

### Signal Processing

A wave scheduler is a **discrete filter**:
- Linear recurrence = IIR filter
- Harmonic = oscillator bank
- Composite = filter cascade

### Control Systems

PID controller as wave scheduler:
```
state = {error, integral, derivative}
E = PID operator
sample = control signal
```

### Machine Learning

Recurrent neural networks are wave schedulers:
- Operator: RNN cell
- Phase: Hidden state h_0
- Sample: Output projection

## 11. Computational Complexity

### Time Complexity

Each step is O(k) for k-dimensional state.

This is constant time regardless of how many events have been scheduled.

### Space Complexity

State space is O(k), not O(n) for n events.

History is optional and capped.

### Comparison with Event Queues

| Aspect | Event Queue | Wave Scheduler |
|--------|-------------|----------------|
| Insert | O(log n) | O(1) |
| Pop | O(log n) | O(1) |
| Memory | O(n) | O(k) |
| Modification | Rebuild | Change operator |

## 12. Future Directions

### Continuous Time

Extend to continuous dynamical systems:
```
dx/dt = F(x)
```

Integrate numerically (RK4, adaptive step size).

Sample at desired resolution.

### Stochastic Waves

Add random perturbations:
```
state_{n+1} = E(state_n) + σ·ε_n
```

Where ε_n is noise.

This gives stochastic scheduling while maintaining wave structure.

### Quantum Schedulers

Represent state as superposition:
```
|ψ⟩ = Σᵢ αᵢ |state_i⟩
```

Evolution is unitary:
```
|ψ_{n+1}⟩ = U|ψ_n⟩
```

Measurement collapses to event time.

### Learning Operators

Use gradient descent to optimize E:
```
E* = argmin_{E} L(schedule generated by E)
```

This creates **optimal schedulers** for specific objectives.

### PDEs and Field Scheduling

Schedule based on field evolution:
```
∂u/∂t = ∇²u + f(u)
```

Events occur at field extrema, zero crossings, or threshold crossings.

## 13. Philosophical Implications

### Declarative vs Imperative

Wave scheduling is **declarative**:
- You specify the law (operator)
- The schedule emerges naturally

Traditional scheduling is **imperative**:
- You explicitly list each event
- Manual construction required

### Natural vs Artificial

Wave patterns are ubiquitous in nature:
- Heartbeats (damped oscillator)
- Circadian rhythms (harmonic)
- Population dynamics (nonlinear recurrence)

Wave scheduling aligns with natural processes rather than fighting them.

### Compositionality

Operators compose naturally. This enables:
- Modular design
- Hierarchical scheduling
- Algebraic reasoning about time

### Resilience

Dynamical systems are inherently resilient:
- Perturbations decay (for stable operators)
- Trajectories return to attractor
- Phase can be adjusted continuously

## 14. Mathematical Theorems

### Spectral Theorem (Linear Case)

For any linear operator E with real coefficients:
```
E = PΛP⁻¹
```

Where Λ is diagonal (possibly complex).

This guarantees decomposition into pure exponential modes.

### Perron-Frobenius Theorem

For non-negative operators, the largest eigenvalue is real and positive.

This means there's always a dominant growth mode.

### Stability Criterion

The system is stable (bounded) iff all eigenvalues satisfy:
```
|λᵢ| ≤ 1
```

This gives a simple test for whether schedules remain bounded.

### Poincaré Recurrence Theorem

For bounded state spaces, almost all trajectories return arbitrarily close to initial condition.

This guarantees periodic or quasi-periodic behavior for bounded schedulers.

## 15. Implementation Notes

### Numerical Stability

For large n, direct computation of E^n can overflow.

Use eigenvalue decomposition:
```
E^n = P Λ^n P⁻¹
```

And compute Λ^n as diagonal matrix of λᵢⁿ.

### Precision

For rational coefficients, exact arithmetic is possible using rational number types.

For irrational eigenvalues (e.g., φ), floating point suffices for practical n.

### Caching

State transitions can be memoized:
```
cache[state] → next_state
```

But typically not needed - evolution is cheap.

## Conclusion

The Wave Scheduler transforms scheduling from manual event lists into the natural evolution of dynamical systems. By grounding scheduling in operator theory, spectral analysis, and phase space geometry, we gain:

- **Mathematical elegance**: Schedules are orbits in state space
- **Composability**: Operators compose algebraically
- **Adaptability**: Phase and operator can be adjusted
- **Naturalness**: Aligns with physical and biological systems
- **Efficiency**: O(k) time and space, not O(n)

This is not just a better way to schedule. It's a **new primitive** - as fundamental as functions and types - for expressing rhythmic, periodic, and evolutionary behavior in software.

The wave scheduler sits at the intersection of:
- Dynamical systems theory
- Operator algebra
- Signal processing
- Control theory
- Compositional learning (CE Tower)

It realizes the vision of programming as **specifying laws of motion** rather than enumerating steps.

## References

1. **Dynamical Systems**: Strogatz, "Nonlinear Dynamics and Chaos"
2. **Spectral Theory**: Horn & Johnson, "Matrix Analysis"
3. **CE Tower**: antclock project, compositional learning architecture
4. **Recurrence Relations**: Graham, Knuth, Patashnik, "Concrete Mathematics"
5. **Operator Algebra**: Reed & Simon, "Functional Analysis"

---

*This document provides the theoretical foundation for the Wave Scheduler implementation in TiddlyWiki5.*
