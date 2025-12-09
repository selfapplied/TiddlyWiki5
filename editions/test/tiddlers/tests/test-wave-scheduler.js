/*\
title: test-wave-scheduler.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests the WaveScheduler class - operator-based scheduling primitive.

\*/

"use strict";

describe("WaveScheduler class tests", function() {

	var WaveScheduler = $tw.utils.WaveScheduler;

	it("should create a basic scheduler with custom operator", function() {
		// Simple doubling operator: state -> state * 2
		var operator = function(state) {
			return state * 2;
		};
		var scheduler = new WaveScheduler(operator, 1, function(s) { return s; });
		
		expect(scheduler.next()).toBe(2);
		expect(scheduler.next()).toBe(4);
		expect(scheduler.next()).toBe(8);
		expect(scheduler.generation).toBe(3);
	});

	it("should require operator to be a function", function() {
		expect(function() {
			new WaveScheduler("not a function", 1);
		}).toThrow();
	});

	it("should require sample to be a function if provided", function() {
		var operator = function(s) { return s + 1; };
		expect(function() {
			new WaveScheduler(operator, 1, "not a function");
		}).toThrow();
		
		// Should allow undefined sample
		expect(function() {
			new WaveScheduler(operator, 1);
		}).not.toThrow();
	});

	it("should reset to initial phase", function() {
		var operator = function(state) {
			return state + 10;
		};
		var scheduler = new WaveScheduler(operator, 5, function(s) { return s; });
		
		scheduler.next();
		scheduler.next();
		expect(scheduler.getState()).toBe(25);
		
		scheduler.reset();
		expect(scheduler.getState()).toBe(5);
		expect(scheduler.generation).toBe(0);
	});

	it("should peek ahead without advancing", function() {
		var operator = function(state) {
			return state + 1;
		};
		var scheduler = new WaveScheduler(operator, 0, function(s) { return s; });
		
		var future = scheduler.peek(5);
		expect(future).toEqual([1, 2, 3, 4, 5]);
		expect(scheduler.generation).toBe(0); // Should not advance
		expect(scheduler.getState()).toBe(0); // State unchanged
	});

	it("should store history up to maxHistory", function() {
		var operator = function(state) {
			return state + 1;
		};
		var scheduler = new WaveScheduler(operator, 0, function(s) { return s; });
		scheduler.maxHistory = 3;
		
		scheduler.next(); // 1
		scheduler.next(); // 2
		scheduler.next(); // 3
		scheduler.next(); // 4
		
		expect(scheduler.history.length).toBe(3);
		expect(scheduler.history).toEqual([2, 3, 4]);
	});

	describe("Fibonacci scheduler", function() {
		it("should generate classic Fibonacci sequence", function() {
			var fib = WaveScheduler.createFibonacci(1, 1, 1);
			
			expect(fib.next()).toBe(2);
			expect(fib.next()).toBe(3);
			expect(fib.next()).toBe(5);
			expect(fib.next()).toBe(8);
			expect(fib.next()).toBe(13);
			expect(fib.next()).toBe(21);
		});

		it("should respect custom initial conditions", function() {
			var fib = WaveScheduler.createFibonacci(2, 3, 1);
			
			expect(fib.next()).toBe(5);
			expect(fib.next()).toBe(8);
			expect(fib.next()).toBe(13);
		});

		it("should scale output", function() {
			var fib = WaveScheduler.createFibonacci(1, 1, 10);
			
			expect(fib.next()).toBe(20);
			expect(fib.next()).toBe(30);
			expect(fib.next()).toBe(50);
		});

		it("should maintain state structure", function() {
			var fib = WaveScheduler.createFibonacci(1, 1, 1);
			fib.next();
			
			var state = fib.getState();
			expect(Array.isArray(state)).toBe(true);
			expect(state.length).toBe(2);
		});
	});

	describe("Linear recurrence scheduler", function() {
		it("should reproduce Fibonacci with [1,1] coefficients", function() {
			var lr = WaveScheduler.createLinearRecurrence([1, 1], [1, 1], 1);
			
			expect(lr.next()).toBe(2);
			expect(lr.next()).toBe(3);
			expect(lr.next()).toBe(5);
		});

		it("should handle 3rd order recurrence", function() {
			// Tribonacci: y_n = y_{n-1} + y_{n-2} + y_{n-3}
			var trib = WaveScheduler.createLinearRecurrence([1, 1, 1], [1, 1, 1], 1);
			
			expect(trib.next()).toBe(3);  // 1+1+1
			expect(trib.next()).toBe(5);  // 3+1+1
			expect(trib.next()).toBe(9);  // 5+3+1
			expect(trib.next()).toBe(17); // 9+5+3
		});

		it("should handle weighted recurrence", function() {
			// y_n = 2*y_{n-1} + 1*y_{n-2}
			var lr = WaveScheduler.createLinearRecurrence([2, 1], [1, 1], 1);
			
			expect(lr.next()).toBe(3);  // 2*1 + 1*1
			expect(lr.next()).toBe(7);  // 2*3 + 1*1
			expect(lr.next()).toBe(17); // 2*7 + 1*3
		});

		it("should reject invalid inputs", function() {
			expect(function() {
				WaveScheduler.createLinearRecurrence([], [1, 1]);
			}).toThrow();

			expect(function() {
				WaveScheduler.createLinearRecurrence([1, 1], [1]); // Mismatched length
			}).toThrow();
		});
	});

	describe("Harmonic scheduler", function() {
		it("should oscillate with correct period", function() {
			var period = 4;
			var harmonic = WaveScheduler.createHarmonic(period, 1, 0);
			
			// Sample a full period
			var samples = [];
			for(var i = 0; i < period; i++) {
				samples.push(harmonic.next());
			}
			
			// Should return close to initial after one period
			var finalState = harmonic.getState();
			expect(Math.abs(finalState[0] - 1)).toBeLessThan(0.01);
			expect(Math.abs(finalState[1])).toBeLessThan(0.01);
		});

		it("should respect amplitude", function() {
			var amplitude = 5;
			var harmonic = WaveScheduler.createHarmonic(10, amplitude, 0);
			
			// Initial sample should be at amplitude
			var first = harmonic.next();
			expect(Math.abs(first) <= amplitude + 0.01).toBe(true);
		});

		it("should handle initial phase", function() {
			var harmonic1 = WaveScheduler.createHarmonic(10, 1, 0);
			var harmonic2 = WaveScheduler.createHarmonic(10, 1, Math.PI);
			
			var s1 = harmonic1.next();
			var s2 = harmonic2.next();
			
			// π phase shift should invert the signal
			expect(Math.abs(s1 + s2)).toBeLessThan(0.1);
		});
	});

	describe("Exponential backoff scheduler", function() {
		it("should exponentially increase delays", function() {
			var backoff = WaveScheduler.createExponentialBackoff(100, 2, 10000);
			
			expect(backoff.next()).toBe(200);
			expect(backoff.next()).toBe(400);
			expect(backoff.next()).toBe(800);
			expect(backoff.next()).toBe(1600);
		});

		it("should cap at maximum delay", function() {
			var backoff = WaveScheduler.createExponentialBackoff(100, 2, 500);
			
			backoff.next(); // 200
			backoff.next(); // 400
			var capped = backoff.next(); // Should be 500, not 800
			expect(capped).toBe(500);
			
			var stillCapped = backoff.next();
			expect(stillCapped).toBe(500);
		});

		it("should use default parameters", function() {
			var backoff = WaveScheduler.createExponentialBackoff();
			var first = backoff.next();
			expect(first).toBe(200); // Default 100 * 2
		});
	});

	describe("Damped oscillator scheduler", function() {
		it("should decay over time", function() {
			var osc = WaveScheduler.createDampedOscillator(0.1, 0.9, 10, 0);
			
			var samples = [];
			for(var i = 0; i < 20; i++) {
				samples.push(osc.next());
			}
			
			// Energy should decrease
			var early = Math.abs(samples[0]);
			var late = Math.abs(samples[19]);
			expect(late).toBeLessThan(early);
		});

		it("should oscillate around zero", function() {
			var osc = WaveScheduler.createDampedOscillator(0.2, 0.95, 5, 0);
			
			var samples = [];
			for(var i = 0; i < 10; i++) {
				samples.push(osc.next());
			}
			
			// Should have both positive and negative values
			var hasPositive = samples.some(function(s) { return s > 0; });
			var hasNegative = samples.some(function(s) { return s < 0; });
			expect(hasPositive && hasNegative).toBe(true);
		});
	});

	describe("Composite scheduler", function() {
		it("should sum multiple schedulers", function() {
			var s1 = WaveScheduler.createFibonacci(1, 1, 1);
			var s2 = WaveScheduler.createFibonacci(1, 1, 1);
			
			var composite = WaveScheduler.createComposite([
				{scheduler: s1, weight: 1},
				{scheduler: s2, weight: 1}
			], "sum");
			
			// Should be 2x Fibonacci
			expect(composite.next()).toBe(4);  // 2+2
			expect(composite.next()).toBe(6);  // 3+3
			expect(composite.next()).toBe(10); // 5+5
		});

		it("should handle different combination modes", function() {
			var s1 = new WaveScheduler(function(s) { return s + 1; }, 2, function(s) { return s; });
			var s2 = new WaveScheduler(function(s) { return s + 1; }, 3, function(s) { return s; });
			
			var sumComposite = WaveScheduler.createComposite([
				{scheduler: s1, weight: 1},
				{scheduler: s2, weight: 1}
			], "sum");
			expect(sumComposite.next()).toBe(7); // 3+4
			
			var maxComposite = WaveScheduler.createComposite([
				{scheduler: s1, weight: 1},
				{scheduler: s2, weight: 1}
			], "max");
			expect(maxComposite.next()).toBe(4); // max(3,4)
		});

		it("should reject invalid inputs", function() {
			expect(function() {
				WaveScheduler.createComposite([]);
			}).toThrow();
		});
	});

	describe("CE bracket-based scheduler", function() {
		it("should create with default operators", function() {
			var ce = WaveScheduler.createCEScheduler();
			
			var first = ce.next();
			expect(typeof first).toBe("number");
		});

		it("should validate CE operator types", function() {
			expect(function() {
				WaveScheduler.createCEScheduler("not a function");
			}).toThrow();

			expect(function() {
				WaveScheduler.createCEScheduler(
					function(x) { return x; },
					"not a function"
				);
			}).toThrow();

			expect(function() {
				WaveScheduler.createCEScheduler(
					function(x) { return x; },
					function(x) { return x; },
					"not a function"
				);
			}).toThrow();
		});

		it("should evolve through CE levels", function() {
			var ce1Op = function(ce1) { return ce1 + 1; };
			var ce2Op = function(ce2, ce1) { return ce2 + ce1 * 0.1; };
			var ce3Op = function(ce3, ce1, ce2) { return ce3 + (ce1 + ce2) * 0.01; };
			
			var ce = WaveScheduler.createCEScheduler(ce1Op, ce2Op, ce3Op, {
				ce1: 1,
				ce2: 0,
				ce3: 0
			});
			
			ce.next();
			var state = ce.getState();
			
			expect(state.ce1).toBe(2); // 1+1
			expect(state.ce2).toBeCloseTo(0.2, 5); // 0 + 2*0.1
			expect(state.ce3).toBeCloseTo(0.022, 5); // 0 + (2+0.2)*0.01
		});

		it("should maintain CE state structure", function() {
			var ce = WaveScheduler.createCEScheduler();
			ce.next();
			
			var state = ce.getState();
			expect(typeof state).toBe("object");
			expect(state.ce1).toBeDefined();
			expect(state.ce2).toBeDefined();
			expect(state.ce3).toBeDefined();
		});
	});

	describe("Integration tests", function() {
		it("should be composable: Fibonacci + Harmonic", function() {
			var fib = WaveScheduler.createFibonacci(1, 1, 1);
			var harmonic = WaveScheduler.createHarmonic(8, 2, 0);
			
			var composite = WaveScheduler.createComposite([
				{scheduler: fib, weight: 1},
				{scheduler: harmonic, weight: 0.5}
			], "sum");
			
			var samples = [];
			for(var i = 0; i < 5; i++) {
				samples.push(composite.next());
			}
			
			// Should have values influenced by both
			expect(samples.length).toBe(5);
			expect(samples[0]).toBeGreaterThan(0);
		});

		it("should handle complex state evolution", function() {
			// Custom operator with complex state
			var operator = function(state) {
				return {
					counter: state.counter + 1,
					sum: state.sum + state.counter,
					product: state.product * 2
				};
			};
			
			var sample = function(state) {
				return state.sum;
			};
			
			var scheduler = new WaveScheduler(operator, {
				counter: 0,
				sum: 0,
				product: 1
			}, sample);
			
			expect(scheduler.next()).toBe(0);  // sum = 0+0
			expect(scheduler.next()).toBe(1);  // sum = 0+1
			expect(scheduler.next()).toBe(3);  // sum = 1+2
			expect(scheduler.next()).toBe(6);  // sum = 3+3
		});
	});

	describe("Edge cases", function() {
		it("should handle zero initial state", function() {
			var scheduler = new WaveScheduler(
				function(s) { return s + 1; },
				0,
				function(s) { return s; }
			);
			
			expect(scheduler.next()).toBe(1);
		});

		it("should handle negative values", function() {
			var scheduler = new WaveScheduler(
				function(s) { return s - 2; },
				10,
				function(s) { return s; }
			);
			
			expect(scheduler.next()).toBe(8);
			expect(scheduler.next()).toBe(6);
			expect(scheduler.next()).toBe(4);
		});

		it("should handle array state", function() {
			var scheduler = new WaveScheduler(
				function(state) {
					return [state[0] + 1, state[1] * 2];
				},
				[0, 1],
				function(state) {
					return state[0] + state[1];
				}
			);
			
			expect(scheduler.next()).toBe(3);  // 1+2
			expect(scheduler.next()).toBe(6);  // 2+4
			expect(scheduler.next()).toBe(11); // 3+8
		});
	});
});
