# Wave Scheduler - Quick Start Guide

## What is this?

The Wave Scheduler is a new programming primitive for TiddlyWiki that reimagines scheduling. Instead of writing loops and timers, you define **operators** that evolve state.

Think of it as:
- **Not a timer** → A dynamical system
- **Not a loop** → A wave equation
- **Not a queue** → Phase space evolution

## 30-Second Example

```javascript
// Traditional scheduling
var delay = 100;
setTimeout(function() { doThing1(); }, delay);
delay *= 2;
setTimeout(function() { doThing2(); }, delay);
delay *= 2;
setTimeout(function() { doThing3(); }, delay);

// Wave scheduling
var scheduler = $tw.utils.WaveScheduler.createExponentialBackoff(100, 2, 10000);
setTimeout(function() { doThing1(); }, scheduler.next());
setTimeout(function() { doThing2(); }, scheduler.next());
setTimeout(function() { doThing3(); }, scheduler.next());
```

## Why Use It?

**Traditional approach:**
```javascript
// Brittle - have to manually calculate each time
var times = [100, 200, 400, 800, 1600];
```

**Wave scheduler:**
```javascript
// Resilient - pattern emerges from operator
var scheduler = $tw.utils.WaveScheduler.createExponentialBackoff(100, 2);
scheduler.next(); // 200
scheduler.next(); // 400
scheduler.next(); // 800
```

Change the pattern? Just change the operator, not every line of code.

## Built-In Schedulers

### 1. Fibonacci (Exponential Growth)
```javascript
var fib = $tw.utils.WaveScheduler.createFibonacci(1, 1, 100);
fib.next(); // 200ms
fib.next(); // 300ms
fib.next(); // 500ms
fib.next(); // 800ms
```

**Use for:** Natural growth patterns, progressive delays

### 2. Harmonic (Oscillation)
```javascript
var pulse = $tw.utils.WaveScheduler.createHarmonic(10, 500, 0);
pulse.next(); // ~500ms
pulse.next(); // ~300ms
pulse.next(); // ~0ms
pulse.next(); // ~-300ms
```

**Use for:** Heartbeats, periodic checks, rhythmic updates

### 3. Exponential Backoff
```javascript
var retry = $tw.utils.WaveScheduler.createExponentialBackoff(100, 2, 5000);
retry.next(); // 200ms
retry.next(); // 400ms
retry.next(); // 800ms
retry.next(); // 1600ms
```

**Use for:** Network retries, rate limiting, error recovery

### 4. Damped Oscillator (Spring Physics)
```javascript
var spring = $tw.utils.WaveScheduler.createDampedOscillator(0.15, 0.9, 10, 0);
spring.next(); // Bounces toward zero
spring.next(); // with natural decay
spring.next(); // settling smoothly
```

**Use for:** Animation easing, smooth convergence

### 5. Composite (Multiple Rhythms)
```javascript
var fast = $tw.utils.WaveScheduler.createHarmonic(5, 100, 0);
var slow = $tw.utils.WaveScheduler.createHarmonic(20, 200, 0);

var composite = $tw.utils.WaveScheduler.createComposite([
    {scheduler: fast, weight: 0.5},
    {scheduler: slow, weight: 0.5}
], "sum");

composite.next(); // Combined rhythm
```

**Use for:** Complex timing patterns, multi-modal systems

## Common Patterns

### Auto-Save with Backoff
```javascript
var autoSave = $tw.utils.WaveScheduler.createFibonacci(1, 1, 1000);
var isDirty = false;

function scheduleNext() {
    var delay = autoSave.next();
    setTimeout(function() {
        if(isDirty) {
            save();
            isDirty = false;
            autoSave.reset(); // Start fresh after save
        }
        scheduleNext();
    }, delay);
}

$tw.wiki.addEventListener("change", function() {
    isDirty = true;
});
```

### Smart Retry
```javascript
var backoff = $tw.utils.WaveScheduler.createExponentialBackoff(100, 2, 30000);

function attemptRequest() {
    makeRequest().catch(function(error) {
        var delay = backoff.next();
        console.log("Retrying in " + delay + "ms");
        setTimeout(attemptRequest, delay);
    });
}
```

### Adaptive Refresh Rate
```javascript
var refresh = $tw.utils.WaveScheduler.createHarmonic(20, 500, 0);

function scheduleRefresh() {
    var delay = 1000 + refresh.next();
    setTimeout(function() {
        updateUI();
        scheduleRefresh();
    }, Math.max(delay, 100));
}
```

## API Reference

### Creating Schedulers

```javascript
// Custom scheduler
var scheduler = new $tw.utils.WaveScheduler(
    operator,  // function: state -> new_state
    phase,     // initial state
    sample     // function: state -> time (optional)
);

// Built-in factories
WaveScheduler.createFibonacci(y0, y1, scale);
WaveScheduler.createLinearRecurrence(coefficients, initial, scale);
WaveScheduler.createHarmonic(period, amplitude, initialPhase);
WaveScheduler.createExponentialBackoff(initialDelay, multiplier, maxDelay);
WaveScheduler.createDampedOscillator(stiffness, damping, position, velocity);
WaveScheduler.createComposite(modes, combination);
WaveScheduler.createCEScheduler(ce1Op, ce2Op, ce3Op, initialState);
```

### Using Schedulers

```javascript
// Get next event time (advances state)
var time = scheduler.next();

// Preview future without advancing
var future = scheduler.peek(5); // Next 5 times

// Reset to beginning
scheduler.reset();

// Get current state
var state = scheduler.getState();
```

## When NOT to Use

- **Simple one-time delays**: Just use `setTimeout(fn, 1000)`
- **Fixed regular intervals**: Use `setInterval(fn, 1000)`
- **Event-driven logic**: Use event listeners
- **Complex dependencies**: May need explicit state machine

Wave schedulers shine when you have **patterns** in your timing, not one-off delays.

## Learning Path

1. **Start here:** Read this README
2. **See examples:** `WAVE_SCHEDULER_EXAMPLES.js` - 10 practical examples
3. **Understand theory:** `WAVE_SCHEDULER.md` - User guide with use cases
4. **Deep dive:** `WAVE_SCHEDULER_THEORY.md` - Mathematical foundations

## Key Concepts

### Operator
The rule that transforms state: `state[n+1] = E(state[n])`

Think: "How does my timing pattern evolve?"

### Phase
The initial conditions: where the wave starts

Think: "What are my starting values?"

### Sample
How to extract a time from state: `time[n] = f(state[n])`

Think: "Which part of the state is the actual delay?"

### Evolution
Each call to `next()` advances the state one generation

Think: "Take one step forward in the pattern"

## Philosophy

Traditional: "Give me times for events 1, 2, 3..."
Wave: "Here's the rule. Times emerge naturally."

Traditional: Procedural (what to do)
Wave: Declarative (law of motion)

Traditional: Brittle (breaks if disrupted)
Wave: Resilient (self-correcting)

## Integration with CE Tower

Wave schedulers integrate naturally with the CE Tower compositional learning architecture:

- **CE1**: Operator defines compositional structure
- **CE2**: Guardian-mediated state evolution
- **CE3**: Self-evolving scheduling patterns

See `WaveScheduler.createCEScheduler()` for multi-level adaptive scheduling.

## Performance

- **Time per step**: O(1) - constant time
- **Memory**: O(k) - state dimension, not event count
- **Overhead**: Minimal - simple arithmetic

Wave schedulers are **more efficient** than event queues for repetitive patterns.

## Testing

34 comprehensive tests covering all scheduler types.

Run tests:
```bash
npm test
```

All tests in: `editions/test/tiddlers/tests/test-wave-scheduler.js`

## Files

- `core/modules/utils/wave-scheduler.js` - Core implementation (435 lines)
- `WAVE_SCHEDULER.md` - User guide and API reference
- `WAVE_SCHEDULER_THEORY.md` - Mathematical foundations
- `WAVE_SCHEDULER_EXAMPLES.js` - Practical examples
- `editions/test/tiddlers/tests/test-wave-scheduler.js` - Test suite

## Support

Questions? Issues?
- Check the examples in `WAVE_SCHEDULER_EXAMPLES.js`
- Read the theory in `WAVE_SCHEDULER_THEORY.md`
- Open an issue on GitHub

## License

BSD License (same as TiddlyWiki5)

---

**Quick tip:** Start with `createExponentialBackoff` or `createFibonacci`. They're the most practical for everyday use.

The wave scheduler is not just a library - it's a new way of thinking about time in software. 🌊
