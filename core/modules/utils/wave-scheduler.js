/*\
module-type: utils
title: $:/core/modules/utils/wave-scheduler.js
type: application/javascript

A generalized wave scheduler - an operator-based scheduling primitive.

Instead of loops, timers, or queues, scheduling is done through wave evolution:
  state[n+1] = E(state[n])
  event_time[n] = f(state[n])

This makes schedules:
- Self-similar
- Resilient  
- Rhythmic instead of brittle
- Phase-adjustable
- Mathematically composable

\*/

"use strict";

/*
WaveScheduler: The core scheduling primitive

A wave scheduler evolves a state through an operator and samples it to
produce event times.

Parameters:
  operator: Function (state -> state) that evolves the state
  phase: Initial state vector
  sample: Function (state -> time) that extracts event time from state
*/
function WaveScheduler(operator, phase, sample) {
	if(typeof operator !== "function") {
		throw new Error("WaveScheduler operator must be a function");
	}
	if(typeof sample !== "function") {
		throw new Error("WaveScheduler sample function must be a function");
	}
	
	this.operator = operator;
	this.phase = phase;
	this.sample = sample || defaultSample;
	this.state = this.phase;
	this.generation = 0;
	this.history = [];
	this.maxHistory = 100; // Keep last 100 states
}

// Default sampling function: extract first component
function defaultSample(state) {
	if(Array.isArray(state)) {
		return state[0];
	}
	if(typeof state === "object" && state !== null) {
		return state.time || state.value || Object.values(state)[0];
	}
	return state;
}

/*
Advance the wave by one generation
Returns the next event time
*/
WaveScheduler.prototype.next = function() {
	// Evolve state
	this.state = this.operator(this.state, this.generation);
	this.generation++;
	
	// Store in history (ring buffer)
	if(this.history.length >= this.maxHistory) {
		this.history.shift();
	}
	this.history.push(this.state);
	
	// Sample to get event time
	return this.sample(this.state, this.generation - 1);
};

/*
Get the next N event times without actually advancing the scheduler
Useful for preview/planning
*/
WaveScheduler.prototype.peek = function(n) {
	var tempState = this.state;
	var tempGen = this.generation;
	var times = [];
	
	for(var i = 0; i < n; i++) {
		tempState = this.operator(tempState, tempGen + i);
		times.push(this.sample(tempState, tempGen + i));
	}
	
	return times;
};

/*
Reset the scheduler to its initial phase
*/
WaveScheduler.prototype.reset = function() {
	this.state = this.phase;
	this.generation = 0;
	this.history = [];
};

/*
Get the current state
*/
WaveScheduler.prototype.getState = function() {
	return this.state;
};

/*
Create a Fibonacci scheduler
This is the canonical example - a 2D linear wave

y_n = y_{n-1} + y_{n-2}
*/
WaveScheduler.createFibonacci = function(y0, y1, scale) {
	y0 = y0 || 1;
	y1 = y1 || 1;
	scale = scale || 1;
	
	var operator = function(state) {
		var y_n = state[0] + state[1];
		return [y_n, state[0]]; // [y_n, y_{n-1}]
	};
	
	var phase = [y1, y0]; // [y_1, y_0]
	
	var sample = function(state) {
		return state[0] * scale;
	};
	
	return new WaveScheduler(operator, phase, sample);
};

/*
Create a linear recurrence scheduler
y_n = a_1*y_{n-1} + a_2*y_{n-2} + ... + a_k*y_{n-k}

coefficients: array [a_1, a_2, ..., a_k]
initial: array [y_1, y_2, ..., y_k] (most recent first)
*/
WaveScheduler.createLinearRecurrence = function(coefficients, initial, scale) {
	if(!Array.isArray(coefficients) || coefficients.length === 0) {
		throw new Error("Coefficients must be a non-empty array");
	}
	if(!Array.isArray(initial) || initial.length !== coefficients.length) {
		throw new Error("Initial values must match coefficient count");
	}
	scale = scale || 1;
	
	var k = coefficients.length;
	
	var operator = function(state) {
		// Compute y_n = sum(a_i * y_{n-i})
		var y_n = 0;
		for(var i = 0; i < k; i++) {
			y_n += coefficients[i] * state[i];
		}
		
		// Shift state: [y_n, y_{n-1}, ..., y_{n-k+1}]
		var newState = [y_n];
		for(var i = 0; i < k - 1; i++) {
			newState.push(state[i]);
		}
		return newState;
	};
	
	var sample = function(state) {
		return state[0] * scale;
	};
	
	return new WaveScheduler(operator, initial, sample);
};

/*
Create a harmonic (oscillatory) scheduler
Based on rotation in 2D: rotation by angle theta

state = [x, y]
x_n = x*cos(theta) - y*sin(theta)
y_n = x*sin(theta) + y*cos(theta)

Parameters:
  period: oscillation period
  amplitude: oscillation amplitude
  initialPhase: initial phase offset (0 to 2*pi)
*/
WaveScheduler.createHarmonic = function(period, amplitude, initialPhase) {
	period = period || 10;
	amplitude = amplitude || 1;
	initialPhase = initialPhase || 0;
	
	var theta = 2 * Math.PI / period;
	var cosTheta = Math.cos(theta);
	var sinTheta = Math.sin(theta);
	
	var operator = function(state) {
		var x = state[0];
		var y = state[1];
		return [
			x * cosTheta - y * sinTheta,
			x * sinTheta + y * cosTheta
		];
	};
	
	var phase = [
		amplitude * Math.cos(initialPhase),
		amplitude * Math.sin(initialPhase)
	];
	
	var sample = function(state) {
		return state[0]; // Sample x-coordinate
	};
	
	return new WaveScheduler(operator, phase, sample);
};

/*
Create an exponential backoff scheduler
Commonly used for retries, but expressed as wave evolution

state = [current_delay, multiplier, max_delay]
*/
WaveScheduler.createExponentialBackoff = function(initialDelay, multiplier, maxDelay) {
	initialDelay = initialDelay || 100;
	multiplier = multiplier || 2;
	maxDelay = maxDelay || 60000;
	
	var operator = function(state) {
		var currentDelay = state[0];
		var mult = state[1];
		var max = state[2];
		var nextDelay = Math.min(currentDelay * mult, max);
		return [nextDelay, mult, max];
	};
	
	var phase = [initialDelay, multiplier, maxDelay];
	
	var sample = function(state) {
		return state[0];
	};
	
	return new WaveScheduler(operator, phase, sample);
};

/*
Create a damped oscillator scheduler
Useful for animation easing, spring physics

state = [position, velocity]
Evolution: 
  v_{n+1} = v_n * damping - position * stiffness
  x_{n+1} = x_n + v_{n+1} * dt
*/
WaveScheduler.createDampedOscillator = function(stiffness, damping, initialPosition, initialVelocity, dt) {
	stiffness = stiffness !== undefined ? stiffness : 0.1;
	damping = damping !== undefined ? damping : 0.9;
	initialPosition = initialPosition !== undefined ? initialPosition : 1;
	initialVelocity = initialVelocity !== undefined ? initialVelocity : 0;
	dt = dt || 1;
	
	var operator = function(state) {
		var position = state[0];
		var velocity = state[1];
		var newVelocity = velocity * damping - position * stiffness;
		var newPosition = position + newVelocity * dt;
		return [newPosition, newVelocity];
	};
	
	var phase = [initialPosition, initialVelocity];
	
	var sample = function(state) {
		return state[0];
	};
	
	return new WaveScheduler(operator, phase, sample);
};

/*
Create a composite scheduler by superposition
Combines multiple schedulers into one

modes: array of {scheduler, weight} objects
combination: "sum" | "product" | "max" | "min"
*/
WaveScheduler.createComposite = function(modes, combination) {
	if(!Array.isArray(modes) || modes.length === 0) {
		throw new Error("Modes must be a non-empty array");
	}
	combination = combination || "sum";
	
	var operator = function(compositeState) {
		// Evolve each sub-scheduler
		var newStates = compositeState.map(function(subState, i) {
			var scheduler = modes[i].scheduler;
			return scheduler.operator(subState);
		});
		return newStates;
	};
	
	var phase = modes.map(function(mode) {
		return mode.scheduler.phase;
	});
	
	var sample = function(compositeState) {
		// Sample each mode
		var samples = compositeState.map(function(subState, i) {
			var mode = modes[i];
			var value = mode.scheduler.sample(subState);
			var weight = mode.weight !== undefined ? mode.weight : 1;
			return value * weight;
		});
		
		// Combine samples
		switch(combination) {
			case "sum":
				return samples.reduce(function(a, b) { return a + b; }, 0);
			case "product":
				return samples.reduce(function(a, b) { return a * b; }, 1);
			case "max":
				return Math.max.apply(null, samples);
			case "min":
				return Math.min.apply(null, samples);
			default:
				return samples[0];
		}
	};
	
	return new WaveScheduler(operator, phase, sample);
};

/*
Create a CE bracket-based scheduler
Integrates with the CE Tower operator algebra

state = {ce1: grammar_phase, ce2: guardian_phase, ce3: evolution_phase}
*/
WaveScheduler.createCEScheduler = function(ce1Operator, ce2Operator, ce3Operator, initialState) {
	initialState = initialState || {
		ce1: 1,
		ce2: 0,
		ce3: 0
	};
	
	var operator = function(state) {
		// CE1: Compositional structure evolution
		var ce1Next = ce1Operator ? ce1Operator(state.ce1) : state.ce1;
		
		// CE2: Guardian-mediated dynamics (with CE1 influence)
		var ce2Next = ce2Operator ? ce2Operator(state.ce2, ce1Next) : state.ce2;
		
		// CE3: Self-evolving pattern (with CE1 and CE2 influence)
		var ce3Next = ce3Operator ? ce3Operator(state.ce3, ce1Next, ce2Next) : state.ce3;
		
		return {
			ce1: ce1Next,
			ce2: ce2Next,
			ce3: ce3Next
		};
	};
	
	var sample = function(state) {
		// Sample is a weighted combination of all three levels
		// CE1 provides base rhythm, CE2 modulates, CE3 evolves
		return state.ce1 + state.ce2 * 0.1 + state.ce3 * 0.01;
	};
	
	return new WaveScheduler(operator, initialState, sample);
};

// Export the scheduler
exports.WaveScheduler = WaveScheduler;
