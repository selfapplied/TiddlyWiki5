/**
 * Wave Scheduler Examples for TiddlyWiki
 * 
 * Practical demonstrations of wave-based scheduling in action.
 * These examples show how to replace traditional loops, timers, and queues
 * with operator-based scheduling.
 */

"use strict";

// ============================================================================
// Example 1: Adaptive Auto-Save with Fibonacci Backoff
// ============================================================================

/**
 * Auto-save that starts aggressive, then backs off exponentially.
 * Uses Fibonacci spacing for natural, organic timing.
 */
function createAdaptiveAutoSave($tw) {
	var fib = $tw.utils.WaveScheduler.createFibonacci(1, 2, 1000);
	var isDirty = false;
	var timeoutId = null;
	
	function scheduleNext() {
		if(timeoutId) {
			clearTimeout(timeoutId);
		}
		
		var delay = fib.next();
		console.log("Next auto-save in " + delay + "ms");
		
		timeoutId = setTimeout(function() {
			if(isDirty) {
				save();
				isDirty = false;
				// Reset to aggressive timing on save
				fib.reset();
			}
			scheduleNext();
		}, delay);
	}
	
	function save() {
		console.log("Saving...");
		// Actual save logic here
	}
	
	function markDirty() {
		isDirty = true;
	}
	
	return {
		start: scheduleNext,
		markDirty: markDirty,
		stop: function() {
			if(timeoutId) {
				clearTimeout(timeoutId);
			}
		}
	};
}

// ============================================================================
// Example 2: Heartbeat Monitor with Harmonic Oscillation
// ============================================================================

/**
 * UI heartbeat that pulses at natural frequency.
 * Can detect when last pulse happened using phase information.
 */
function createHeartbeatMonitor($tw) {
	// 10-second period, amplitude varies between 0-1000ms
	var harmonic = $tw.utils.WaveScheduler.createHarmonic(10, 500, 0);
	var baseInterval = 2000;
	var listeners = [];
	var running = false;
	
	function pulse() {
		if(!running) return;
		
		var offset = harmonic.next();
		var interval = baseInterval + offset;
		
		// Notify all listeners
		listeners.forEach(function(listener) {
			listener.onPulse(interval);
		});
		
		setTimeout(pulse, interval);
	}
	
	return {
		start: function() {
			running = true;
			pulse();
		},
		stop: function() {
			running = false;
		},
		subscribe: function(listener) {
			listeners.push(listener);
		}
	};
}

// ============================================================================
// Example 3: Smart Retry with Exponential Backoff
// ============================================================================

/**
 * Network request retry with capped exponential backoff.
 * Wave-based approach makes it declarative and composable.
 */
function createSmartRetry($tw, operation, maxRetries) {
	maxRetries = maxRetries || 5;
	var backoff = $tw.utils.WaveScheduler.createExponentialBackoff(100, 2, 30000);
	var attemptCount = 0;
	
	function attempt() {
		return operation().catch(function(error) {
			attemptCount++;
			
			if(attemptCount >= maxRetries) {
				throw new Error("Max retries exceeded: " + error.message);
			}
			
			var delay = backoff.next();
			console.log("Retry #" + attemptCount + " in " + delay + "ms");
			
			return new Promise(function(resolve) {
				setTimeout(function() {
					resolve(attempt());
				}, delay);
			});
		});
	}
	
	return attempt;
}

// ============================================================================
// Example 4: Animation Timeline with Damped Oscillator
// ============================================================================

/**
 * Spring-physics based animation.
 * Element "settles" into position with natural spring motion.
 */
function createSpringAnimation($tw, element, targetPosition) {
	var currentPos = 0;
	var spring = $tw.utils.WaveScheduler.createDampedOscillator(
		0.15,  // stiffness
		0.9,   // damping
		targetPosition - currentPos,  // initial displacement
		0      // initial velocity
	);
	
	var frameCount = 0;
	var maxFrames = 60;
	
	function animate() {
		if(frameCount >= maxFrames) {
			element.style.left = targetPosition + "px";
			return;
		}
		
		var displacement = spring.next();
		currentPos = targetPosition - displacement;
		element.style.left = currentPos + "px";
		
		frameCount++;
		requestAnimationFrame(animate);
	}
	
	return {
		start: animate,
		peek: function(n) {
			return spring.peek(n);
		}
	};
}

// ============================================================================
// Example 5: Composite Scheduling - Multiple Rhythms
// ============================================================================

/**
 * Attention system with multiple overlapping rhythms.
 * Combines fast local checks with slow global scans.
 */
function createAttentionScheduler($tw) {
	// Fast local rhythm: check immediate context
	var local = $tw.utils.WaveScheduler.createHarmonic(8, 200, 0);
	
	// Slow global rhythm: check whole system
	var global = $tw.utils.WaveScheduler.createHarmonic(30, 500, Math.PI/2);
	
	// Fibonacci exploration: occasional deep dives
	var exploration = $tw.utils.WaveScheduler.createFibonacci(100, 100, 10);
	
	var composite = $tw.utils.WaveScheduler.createComposite([
		{scheduler: local, weight: 0.5},
		{scheduler: global, weight: 0.3},
		{scheduler: exploration, weight: 0.2}
	], "sum");
	
	var baseInterval = 1000;
	var running = false;
	
	function tick() {
		if(!running) return;
		
		var offset = composite.next();
		var interval = baseInterval + offset;
		
		// Determine which rhythm is dominant
		var localContrib = local.getState()[0] * 0.5;
		var globalContrib = global.getState()[0] * 0.3;
		
		if(Math.abs(localContrib) > Math.abs(globalContrib)) {
			checkLocal();
		} else {
			checkGlobal();
		}
		
		setTimeout(tick, Math.max(interval, 100));
	}
	
	function checkLocal() {
		console.log("Local attention check");
		// Check immediate context
	}
	
	function checkGlobal() {
		console.log("Global attention scan");
		// Full system scan
	}
	
	return {
		start: function() {
			running = true;
			tick();
		},
		stop: function() {
			running = false;
		}
	};
}

// ============================================================================
// Example 6: CE-Based Scheduler for Learning Systems
// ============================================================================

/**
 * Adaptive scheduler that learns from system behavior.
 * Uses CE Tower levels: CE1 (structure), CE2 (dynamics), CE3 (evolution).
 */
function createLearningScheduler($tw) {
	// CE1: Base compositional rhythm
	var ce1Op = function(ce1) {
		return ce1 * 1.1; // Exponential base
	};
	
	// CE2: Guardian-mediated adjustments based on system load
	var systemLoad = 0.5; // 0-1 scale
	var ce2Op = function(ce2, ce1) {
		// Slow down when load is high
		return ce2 + (1 - systemLoad) * ce1 * 0.1;
	};
	
	// CE3: Pattern detection and evolution
	var recentErrors = [];
	var ce3Op = function(ce3, ce1, ce2) {
		// Learn from error patterns
		var errorRate = recentErrors.length / 10;
		return ce3 + errorRate * (ce1 + ce2) * 0.05;
	};
	
	var ce = $tw.utils.WaveScheduler.createCEScheduler(ce1Op, ce2Op, ce3Op, {
		ce1: 100,   // Base interval in ms
		ce2: 0,     // Dynamic adjustment
		ce3: 0      // Evolutionary component
	});
	
	return {
		next: function() {
			return ce.next();
		},
		updateLoad: function(load) {
			systemLoad = Math.max(0, Math.min(1, load));
		},
		recordError: function() {
			recentErrors.push(Date.now());
			// Keep only last 10
			if(recentErrors.length > 10) {
				recentErrors.shift();
			}
		},
		getState: function() {
			return ce.getState();
		}
	};
}

// ============================================================================
// Example 7: Custom Wave Scheduler - Polynomial Growth
// ============================================================================

/**
 * Custom polynomial growth: t_n = n^2
 * Demonstrates how to create any scheduling pattern.
 */
function createPolynomialScheduler($tw, degree, scale) {
	degree = degree || 2;
	scale = scale || 1;
	
	var operator = function(state) {
		return state + 1; // Just count generations
	};
	
	var sample = function(state, generation) {
		return Math.pow(generation + 1, degree) * scale;
	};
	
	return new $tw.utils.WaveScheduler(operator, 0, sample);
}

// ============================================================================
// Example 8: Self-Adjusting Refresh Rate
// ============================================================================

/**
 * Refresh rate that adapts to content change frequency.
 * Fast when content changes rapidly, slow when stable.
 */
function createAdaptiveRefreshRate($tw) {
	var lastChangeTime = Date.now();
	var dampedOsc = $tw.utils.WaveScheduler.createDampedOscillator(
		0.1,   // stiffness
		0.95,  // high damping for smooth settling
		10,    // initial displacement
		0      // no initial velocity
	);
	
	function refresh() {
		// Actual refresh logic
		console.log("Refreshing...");
	}
	
	function scheduleNext() {
		var displacement = dampedOsc.next();
		var interval = 1000 + Math.abs(displacement) * 100;
		
		setTimeout(function() {
			refresh();
			scheduleNext();
		}, interval);
	}
	
	function recordChange() {
		var now = Date.now();
		var timeSinceLastChange = now - lastChangeTime;
		
		// Reset oscillator on rapid changes
		if(timeSinceLastChange < 500) {
			dampedOsc.reset();
		}
		
		lastChangeTime = now;
	}
	
	return {
		start: scheduleNext,
		onChange: recordChange
	};
}

// ============================================================================
// Example 9: Progressive Loading with Fibonacci Stages
// ============================================================================

/**
 * Load resources in Fibonacci-spaced stages.
 * Natural progression: immediate, then increasingly patient.
 */
function createProgressiveLoader($tw, resources) {
	var fib = $tw.utils.WaveScheduler.createFibonacci(100, 100, 1);
	var index = 0;
	
	function loadNext() {
		if(index >= resources.length) {
			console.log("All resources loaded");
			return;
		}
		
		var resource = resources[index];
		console.log("Loading " + resource + "...");
		
		// Simulate loading
		loadResource(resource).then(function() {
			index++;
			var delay = fib.next();
			console.log("Next load in " + delay + "ms");
			setTimeout(loadNext, delay);
		});
	}
	
	function loadResource() {
		return new Promise(function(resolve) {
			setTimeout(resolve, 100);
		});
	}
	
	return {
		start: loadNext,
		getProgress: function() {
			return index / resources.length;
		}
	};
}

// ============================================================================
// Example 10: Multi-Phase Scheduler - Startup/Runtime/Shutdown
// ============================================================================

/**
 * Different scheduling behaviors for different application phases.
 * Each phase has its own wave pattern.
 */
function createMultiPhaseScheduler($tw) {
	var phase = "startup";
	
	// Startup: aggressive Fibonacci
	var startupScheduler = $tw.utils.WaveScheduler.createFibonacci(50, 50, 1);
	
	// Runtime: gentle harmonic
	var runtimeScheduler = $tw.utils.WaveScheduler.createHarmonic(20, 500, 0);
	
	// Shutdown: rapid linear
	var shutdownScheduler = $tw.utils.WaveScheduler.createLinearRecurrence(
		[1], [100], 1
	);
	
	var baseInterval = 1000;
	
	function getScheduler() {
		switch(phase) {
			case "startup": return startupScheduler;
			case "runtime": return runtimeScheduler;
			case "shutdown": return shutdownScheduler;
			default: return runtimeScheduler;
		}
	}
	
	return {
		next: function() {
			var scheduler = getScheduler();
			return baseInterval + scheduler.next();
		},
		setPhase: function(newPhase) {
			phase = newPhase;
		},
		getPhase: function() {
			return phase;
		}
	};
}

// ============================================================================
// Usage in TiddlyWiki
// ============================================================================

/*
// In a TiddlyWiki plugin or module:

exports.startup = function() {
	// Create an adaptive auto-save
	var autoSave = createAdaptiveAutoSave($tw);
	autoSave.start();
	
	// Track changes
	$tw.wiki.addEventListener("change", function(changes) {
		autoSave.markDirty();
	});
	
	// Create heartbeat monitor
	var heartbeat = createHeartbeatMonitor($tw);
	heartbeat.subscribe({
		onPulse: function(interval) {
			$tw.rootWidget.refresh();
		}
	});
	heartbeat.start();
	
	// Smart retry for network operations
	var retryOperation = createSmartRetry($tw, function() {
		return $tw.utils.httpRequest({
			url: "http://example.com/api",
			callback: function(err, data) {
				if(err) throw new Error(err);
				return data;
			}
		});
	}, 5);
};
*/

// Export examples for documentation
if(typeof exports !== "undefined") {
	exports.createAdaptiveAutoSave = createAdaptiveAutoSave;
	exports.createHeartbeatMonitor = createHeartbeatMonitor;
	exports.createSmartRetry = createSmartRetry;
	exports.createSpringAnimation = createSpringAnimation;
	exports.createAttentionScheduler = createAttentionScheduler;
	exports.createLearningScheduler = createLearningScheduler;
	exports.createPolynomialScheduler = createPolynomialScheduler;
	exports.createAdaptiveRefreshRate = createAdaptiveRefreshRate;
	exports.createProgressiveLoader = createProgressiveLoader;
	exports.createMultiPhaseScheduler = createMultiPhaseScheduler;
}
