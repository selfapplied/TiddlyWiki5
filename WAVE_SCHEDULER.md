# Wave Scheduler: Operator-Based Scheduling Primitive

## Overview

The Wave Scheduler is a new programming primitive that fundamentally reimagines how scheduling works. Instead of loops, timers, or queues, it uses **operator evolution** and **wave dynamics** to generate event schedules.

## Philosophy

Traditional scheduling:
```
t[n] = t[n-1] + Δ
```

Wave scheduling:
```
state[n+1] = E(state[n])
event_time[n] = f(state[n])
```

The event times are **samples of the orbit of the operator E**.

This transforms scheduling from a handmade sequence into **a law of motion acting in a small state-space**.

## Why Wave Scheduling?

Wave scheduling makes schedules:
- **Self-similar**: Natural patterns emerge from the operator
- **Resilient**: State evolution handles disruptions gracefully  
- **Rhythmic**: Schedules have natural periodicities, not brittle fixed intervals
- **Phase-adjustable**: Initial conditions determine the specific trajectory
- **Mathematically composable**: Multiple schedules can be combined algebraically
- **Declarative**: Describe the rule of motion, not the sequence

## Core Concepts

### The Scheduler Primitive

A wave scheduler consists of three elements:

1. **Operator E**: `State → State` - The rule of motion
2. **Phase p**: Initial state vector - Where the wave begins
3. **Sample f**: `State → Time` - How to extract event times from state

```javascript
var scheduler = new $tw.utils.WaveScheduler(operator, phase, sample);
var nextTime = scheduler.next();
```

### Example: Fibonacci Scheduling

The Fibonacci sequence is just a 2D linear wave:

```
y_n = y_{n-1} + y_{n-2}
```

As a matrix operator:
```
E = [[1, 1],
     [1, 0]]
```

In code:
```javascript
var fib = $tw.utils.WaveScheduler.createFibonacci(1, 1, 1);
fib.next(); // 2
fib.next(); // 3
fib.next(); // 5
fib.next(); // 8
```

The Fibonacci times are the "crest points" of the wave.

## Built-in Schedulers

### 1. Fibonacci Scheduler

Classic exponential growth with golden ratio periodicity.

```javascript
var fib = $tw.utils.WaveScheduler.createFibonacci(y0, y1, scale);
```

**Use cases**: 
- Exponentially increasing timeouts
- Natural growth patterns
- Golden ratio spacing

### 2. Linear Recurrence Scheduler

Generalized k-order linear recurrence:
```
y_n = a₁·y_{n-1} + a₂·y_{n-2} + ... + aₖ·y_{n-k}
```

```javascript
// Tribonacci: y_n = y_{n-1} + y_{n-2} + y_{n-3}
var trib = $tw.utils.WaveScheduler.createLinearRecurrence(
  [1, 1, 1],    // coefficients
  [1, 1, 1],    // initial values
  1             // scale
);
```

**Use cases**:
- Custom growth patterns
- Multi-term dependencies
- Polynomial-style scheduling

### 3. Harmonic Scheduler

Circular motion in 2D - pure oscillation.

```javascript
var harmonic = $tw.utils.WaveScheduler.createHarmonic(
  period,        // oscillation period
  amplitude,     // oscillation amplitude
  initialPhase   // phase offset (0 to 2π)
);
```

**Use cases**:
- Animation frames
- Heartbeat intervals
- Periodic refresh cycles
- Circadian rhythms

### 4. Exponential Backoff Scheduler

For retries and rate limiting.

```javascript
var backoff = $tw.utils.WaveScheduler.createExponentialBackoff(
  initialDelay,  // starting delay (ms)
  multiplier,    // growth factor (e.g., 2)
  maxDelay       // maximum delay cap
);
```

**Use cases**:
- Network retry logic
- API rate limiting
- Throttling
- Error recovery

### 5. Damped Oscillator Scheduler

Spring physics with decay.

```javascript
var spring = $tw.utils.WaveScheduler.createDampedOscillator(
  stiffness,         // spring constant
  damping,           // damping factor (0-1)
  initialPosition,   // starting position
  initialVelocity    // starting velocity
);
```

**Use cases**:
- Animation easing
- UI spring physics
- Smooth convergence
- Settling behaviors

### 6. Composite Scheduler

Superposition of multiple schedulers.

```javascript
var composite = $tw.utils.WaveScheduler.createComposite([
  {scheduler: fib, weight: 1},
  {scheduler: harmonic, weight: 0.5}
], "sum");  // or "product", "max", "min"
```

**Use cases**:
- Complex rhythms
- Multi-modal scheduling
- Interference patterns
- Adaptive timing

### 7. CE Bracket-Based Scheduler

Integrates with CE Tower operator algebra.

```javascript
var ce = $tw.utils.WaveScheduler.createCEScheduler(
  ce1Operator,  // Grammar level evolution
  ce2Operator,  // Guardian-mediated dynamics
  ce3Operator,  // Self-evolving patterns
  initialState
);
```

**Use cases**:
- Compositional learning schedules
- Multi-level adaptive timing
- Evolution-based scheduling
- Phase-locked operations

## API Reference

### Core Methods

```javascript
// Create custom scheduler
var scheduler = new $tw.utils.WaveScheduler(operator, phase, sample);

// Get next event time (advances state)
var time = scheduler.next();

// Peek ahead without advancing
var futureTimes = scheduler.peek(n);

// Reset to initial phase
scheduler.reset();

// Get current state
var state = scheduler.getState();
```

### Custom Operators

You can create any scheduler by defining an operator:

```javascript
// Double each time
var doubling = new $tw.utils.WaveScheduler(
  function(state) { return state * 2; },
  1,
  function(state) { return state; }
);

// Complex state evolution
var complex = new $tw.utils.WaveScheduler(
  function(state) {
    return {
      counter: state.counter + 1,
      energy: state.energy * 0.95
    };
  },
  {counter: 0, energy: 100},
  function(state) {
    return state.counter * state.energy;
  }
);
```

## Practical Examples

### Adaptive Refresh Rate

```javascript
// Start fast, slow down, then speed up again
var harmonic = $tw.utils.WaveScheduler.createHarmonic(20, 500, 0);
var baseline = 1000; // 1 second baseline

function scheduleRefresh() {
  var delay = baseline + harmonic.next();
  setTimeout(function() {
    refresh();
    scheduleRefresh();
  }, delay);
}
```

### Resilient Retry Logic

```javascript
var backoff = $tw.utils.WaveScheduler.createExponentialBackoff(100, 2, 30000);

function attemptOperation() {
  operation().catch(function(error) {
    var delay = backoff.next();
    setTimeout(attemptOperation, delay);
  });
}
```

### Animation Timeline

```javascript
// Fibonacci-spaced keyframes for organic feel
var fib = $tw.utils.WaveScheduler.createFibonacci(16, 16, 1);
var times = fib.peek(10); // Next 10 keyframe times

times.forEach(function(time, i) {
  setTimeout(function() {
    animate(i);
  }, time);
});
```

### Composite Attention System

```javascript
// Fast heartbeat + slow breath
var heartbeat = $tw.utils.WaveScheduler.createHarmonic(1, 100, 0);
var breath = $tw.utils.WaveScheduler.createHarmonic(5, 200, 0);

var attention = $tw.utils.WaveScheduler.createComposite([
  {scheduler: heartbeat, weight: 0.3},
  {scheduler: breath, weight: 0.7}
], "sum");

function checkAttention() {
  var delay = 1000 + attention.next();
  setTimeout(function() {
    updateUI();
    checkAttention();
  }, delay);
}
```

## Mathematical Background

Every k-order linear recurrence:
```
y_n = a₁·y_{n-1} + ... + aₖ·y_{n-k}
```

Becomes a k-dimensional linear operator E, which has an eigenbasis decomposition:
```
E = PΛP⁻¹
```

Where Λ contains eigenvalues λᵢ. Each eigenvalue contributes a mode:
- Real λ > 0: Exponential growth/decay
- Real λ < 0: Alternating exponential
- Complex λ: Oscillatory (spiral) motion

This means **every wave scheduler is fundamentally a set of oscillators**, each with amplitude and phase, evolving under a linear (or nonlinear) operator.

## Integration with CE Tower

The wave scheduler fits naturally into the CE Tower architecture:

- **CE1 (Grammar)**: The operator E defines compositional structure
- **CE2 (Dynamics)**: Phase p and guardian-mediated evolution
- **CE3 (Evolution)**: Operators can evolve based on pattern detection

A CE scheduler evolves through all three levels:
1. Base rhythm from CE1
2. Guardian modulation from CE2  
3. Self-evolving patterns from CE3

This creates **scheduling that learns and adapts** rather than following fixed rules.

## Performance Considerations

Wave schedulers are:
- **Lightweight**: Simple state evolution, no event queues
- **Predictable**: O(1) per next() call
- **Cache-friendly**: Small state vectors
- **Composable**: Multiple schedulers combine efficiently

The `peek()` method allows planning ahead without side effects.

History is capped at `maxHistory` (default 100) to prevent unbounded growth.

## Comparison with Traditional Scheduling

| Aspect | Traditional | Wave Scheduler |
|--------|------------|----------------|
| Representation | List of times or intervals | Operator + phase |
| Modification | Rewrite sequence | Adjust operator/phase |
| Composition | Merge/interleave lists | Operator algebra |
| Resilience | Brittle to disruption | Self-correcting |
| Expressiveness | Procedural | Declarative |
| Memory | O(n) events | O(k) state dimension |

## Future Directions

Potential extensions:
- **Adaptive operators**: E that changes based on system state
- **Learning schedulers**: Operators that optimize from feedback
- **Stochastic waves**: Random perturbations to deterministic evolution
- **PDE-based scheduling**: Continuous wave equations
- **Bracket calculus**: Full CE Tower operator expressions

## References

This implementation is inspired by:
- The CE Tower compositional learning architecture
- Dynamical systems theory and phase space evolution
- The antclock experiential time system
- Operator algebra and spectral theory

## License

Part of TiddlyWiki5, same license applies (BSD).
