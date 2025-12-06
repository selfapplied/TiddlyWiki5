/*\
title: $:/core/modules/utils/antclock-engine.js
type: application/javascript
module-type: utils

Antclock Engine - Proper implementation of CE2 antclock concepts

Antclock measures time in semantic transition units, not clock ticks.
It tracks meaningful state changes and calculates clock rates.

Based on: R(x) = χ_FEG · κ_d(x) · (1 + Q_9/11(x))
Where:
- R(x): Clock rate (significance of change)
- χ_FEG: Transform quality measure ≈ 0.638
- κ_d(x): Discrete curvature (change magnitude)
- Q_9/11(x): Modular correction (fine structure)

Reference: https://github.com/selfapplied/antclock/blob/main/arXiv/working.md
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

// Import the existing antclock utilities for clock rate calculation
var antclockUtils = require("$:/core/modules/utils/antclock.js");

// Constants from antclock research
var SIGNIFICANCE_THRESHOLD = 0.1; // Minimum clock rate to record a tick

/**
 * Antclock Engine - Proper implementation
 * 
 * Antclock is about experiential time: time measured by semantic transitions,
 * not clock ticks. It tracks meaningful state changes and calculates clock rates.
 * 
 * Key concepts:
 * - Clock Rate R(x): Significance of a semantic change
 * - Experiential Time: Cumulative clock rate over semantic transitions
 * - Semantic Transitions: Meaningful state changes, not trivial edits
 * - Self-Visibility: System observes its own semantic evolution
 */
function AntclockEngine(options) {
	options = options || {};
	this.name = options.name || "antclock";
	this.logging = options.logging !== false;
	
	// Experiential time tracking (the core of antclock)
	this.experientialTime = 0; // Cumulative clock rate
	this.semanticTransitions = 0; // Number of significant transitions
	this.transitionHistory = [];
	
	// Clock rate tracking
	this.lastClockRate = 0;
	this.averageClockRate = 0;
	
	// Self-visibility: observers of semantic transitions
	this.observers = [];
	
	// Guardian boundaries for semantic integrity
	this.guardians = [];
	
	// Configuration
	this.significanceThreshold = options.significanceThreshold || SIGNIFICANCE_THRESHOLD;
}

/**
 * Record a semantic transition with clock rate
 * This is the core antclock operation: advancing experiential time by clock rate
 * 
 * @param {Number} clockRate - The significance of the change (R(x))
 * @param {Object} details - Details about the semantic change
 * @returns {Object} Transition record
 */
AntclockEngine.prototype.recordTransition = function(clockRate, details) {
	details = details || {};
	
	// Only record if clock rate exceeds significance threshold
	if(clockRate < this.significanceThreshold) {
		return null; // Trivial change, no transition
	}
	
	// Advance experiential time by clock rate
	this.experientialTime += clockRate;
	this.semanticTransitions++;
	
	// Record transition
	var transition = {
		experientialTime: this.experientialTime,
		clockRate: clockRate,
		semanticTransitions: this.semanticTransitions,
		clockTime: Date.now(),
		details: details
	};
	
	this.transitionHistory.push(transition);
	
	// Keep history bounded (last 100 transitions)
	if(this.transitionHistory.length > 100) {
		this.transitionHistory.shift();
	}
	
	// Update average clock rate
	this.updateAverageClockRate();
	
	// Self-visibility: notify observers
	this.notifyObservers(transition);
	
	return transition;
};

/**
 * Calculate clock rate for a semantic change
 * Uses the antclock formula: R(x) = χ_FEG · κ_d(x) · (1 + Q_9/11(x))
 * 
 * @param {String} oldContent - Previous semantic state
 * @param {String} newContent - New semantic state
 * @param {Object} options - Calculation options
 * @returns {Number} Clock rate (0-1)
 */
AntclockEngine.prototype.calculateClockRate = function(oldContent, newContent, options) {
	// Use the existing antclock utility for calculation
	return antclockUtils.calculateClockRate(oldContent, newContent, options);
};

/**
 * Process a semantic change and record transition if significant
 * 
 * @param {String} oldContent - Previous content
 * @param {String} newContent - New content
 * @param {Object} metadata - Additional metadata about the change
 * @returns {Object|null} Transition record if significant, null otherwise
 */
AntclockEngine.prototype.processChange = function(oldContent, newContent, metadata) {
	metadata = metadata || {};
	
	// Calculate clock rate for this change
	var clockRate = this.calculateClockRate(oldContent, newContent);
	this.lastClockRate = clockRate;
	
	// Record transition if significant
	if(clockRate >= this.significanceThreshold) {
		var details = {
			structural: metadata.structural,
			semantic: metadata.semantic,
			coherence: metadata.coherence,
			source: metadata.source || "unknown"
		};
		
		return this.recordTransition(clockRate, details);
	}
	
	return null; // Change not significant enough
};

/**
 * Update average clock rate from recent transitions
 */
AntclockEngine.prototype.updateAverageClockRate = function() {
	if(this.transitionHistory.length === 0) {
		this.averageClockRate = 0;
		return;
	}
	
	// Calculate average from recent transitions (last 10)
	var recent = this.transitionHistory.slice(-10);
	var sum = recent.reduce(function(acc, t) {
		return acc + (t.clockRate || 0);
	}, 0);
	
	this.averageClockRate = sum / recent.length;
};

/**
 * Get experiential age (cumulative clock rate)
 */
AntclockEngine.prototype.getExperientialAge = function() {
	return this.experientialTime;
};

/**
 * Get number of semantic transitions
 */
AntclockEngine.prototype.getTransitionCount = function() {
	return this.semanticTransitions;
};

/**
 * Get recent activity rate
 */
AntclockEngine.prototype.getRecentActivityRate = function(windowSize) {
	windowSize = windowSize || 10;
	var recent = this.transitionHistory.slice(-windowSize);
	if(recent.length === 0) {
		return 0;
	}
	var sum = recent.reduce(function(acc, t) {
		return acc + (t.clockRate || 0);
	}, 0);
	return sum / recent.length;
};

/**
 * Self-Visibility: Add observer for semantic transitions
 */
AntclockEngine.prototype.observe = function(observer) {
	if(typeof observer === "function") {
		this.observers.push(observer);
		var self = this;
		return function() {
			var index = self.observers.indexOf(observer);
			if(index !== -1) {
				self.observers.splice(index, 1);
			}
		};
	}
};

/**
 * Notify observers of semantic transitions
 */
AntclockEngine.prototype.notifyObservers = function(transition) {
	var self = this;
	this.observers.forEach(function(observer) {
		try {
			observer(transition, self.experientialTime, self.semanticTransitions);
		} catch(e) {
			if(self.logging && $tw && $tw.log) {
				$tw.log("antclock-engine", "Observer error:", e);
			}
		}
	});
};

/**
 * Guardian Boundaries: Add semantic boundary check
 */
AntclockEngine.prototype.addGuardian = function(name, checkFunction) {
	if(typeof checkFunction === "function") {
		this.guardians.push({
			name: name,
			check: checkFunction
		});
	}
};

/**
 * Check guardian boundaries
 */
AntclockEngine.prototype.checkGuardians = function() {
	var violations = [];
	var self = this;
	this.guardians.forEach(function(guardian) {
		try {
			if(!guardian.check()) {
				violations.push(guardian.name);
			}
		} catch(e) {
			if(self.logging && $tw && $tw.log) {
				$tw.log("antclock-engine", "Guardian check error:", e);
			}
		}
	});
	return violations;
};

/**
 * Get engine statistics
 */
AntclockEngine.prototype.getStats = function() {
	return {
		experientialTime: this.experientialTime,
		semanticTransitions: this.semanticTransitions,
		lastClockRate: this.lastClockRate,
		averageClockRate: this.averageClockRate,
		recentActivityRate: this.getRecentActivityRate(),
		observers: this.observers.length,
		guardians: this.guardians.length,
		transitionHistorySize: this.transitionHistory.length
	};
};

/**
 * Get transition history
 */
AntclockEngine.prototype.getHistory = function(limit) {
	limit = limit || this.transitionHistory.length;
	return this.transitionHistory.slice(-limit);
};

/**
 * Observe Phase State Transitions
 * In antclock/CE2, phase (θ) represents semantic direction/topic
 * Phase states represent different semantic phases the system moves through
 * 
 * @param {Function} checkPhaseState - Returns current phase state (e.g., "initializing", "loading", "ready")
 * @param {Function} onPhaseTransition - Called when phase transitions to target phase
 * @param {Object} options - Configuration
 */
AntclockEngine.prototype.observePhaseState = function(checkPhaseState, onPhaseTransition, options) {
	options = options || {};
	var self = this;
	var targetPhase = options.targetPhase || "ready";
	var lastPhase = null;
	var checkInterval = options.checkInterval || 50;
	var maxChecks = options.maxChecks || 400; // 20 seconds at 50ms
	var checkCount = 0;
	var timerId = null;
	
	// Self-timing: check phase state recursively
	var checkPhase = function() {
		checkCount++;
		
		// Grammar adjusting: safe phase check
		var currentPhase;
		try {
			currentPhase = checkPhaseState();
		} catch(e) {
			if(self.logging && $tw && $tw.log) {
				$tw.log("antclock-engine", "Phase check error:", e);
			}
			currentPhase = "unknown";
		}
		
		// Phase transition detected (experiential time - meaningful change)
		if(currentPhase !== lastPhase) {
			// Phase transition occurred - this is a semantic transition
			if(self.logging && $tw && $tw.log) {
				$tw.log("antclock-engine", "Phase transition:", lastPhase, "->", currentPhase);
			}
			
			// Record phase transition as a semantic transition
			// Phase transitions have minimal clock rate (they're state observations, not content changes)
			var phaseTransition = {
				experientialTime: self.experientialTime,
				clockRate: 0.01, // Minimal clock rate for phase transition
				semanticTransitions: self.semanticTransitions,
				clockTime: Date.now(),
				details: { 
					type: "phase-transition", 
					fromPhase: lastPhase,
					toPhase: currentPhase 
				}
			};
			
			lastPhase = currentPhase;
			
			// If we reached target phase, this is a meaningful semantic transition
			if(currentPhase === targetPhase) {
				// Notify observers of phase transition
				self.notifyObservers(phaseTransition);
				
				// Call the phase transition handler
				if(onPhaseTransition) {
					try {
						onPhaseTransition(phaseTransition, self.experientialTime);
					} catch(e) {
						if(self.logging && $tw && $tw.log) {
							$tw.log("antclock-engine", "Phase transition handler error:", e);
						}
					}
				}
				
				// Stop checking
				if(timerId) {
					clearTimeout(timerId);
				}
				return;
			}
		}
		
		// Continue self-timing if not at target phase and checks remain
		if(currentPhase !== targetPhase && checkCount < maxChecks) {
			// Timing based on experiential time (not clock time)
			// More checks = more "experience" = adjust timing
			var delay = Math.min(
				checkInterval * Math.pow(1.1, Math.floor(checkCount / 10)),
				500 // Max 500ms
			);
			
			timerId = setTimeout(checkPhase, delay);
		} else if(checkCount >= maxChecks) {
			// Max checks reached
			if(options.onTimeout) {
				try {
					options.onTimeout(currentPhase, checkCount);
				} catch(e) {
					if(self.logging && $tw && $tw.log) {
						$tw.log("antclock-engine", "Timeout handler error:", e);
					}
				}
			}
		}
	};
	
	// Start observing phase state
	try {
		checkPhase();
	} catch(e) {
		if(self.logging && $tw && $tw.log) {
			$tw.log("antclock-engine", "Failed to start phase observation:", e);
		}
	}
	
	// Return cleanup function
	return function() {
		if(timerId) {
			clearTimeout(timerId);
		}
	};
};

/**
 * Reactive Phase State Observation
 * Combines self-timing phase observation with event-driven reactivity
 * Phase resonance: synchronize with system events
 */
AntclockEngine.prototype.observePhaseStateReactive = function(checkPhaseState, onPhaseTransition, options) {
	options = options || {};
	var self = this;
	var targetPhase = options.targetPhase || "ready";
	
	// Start self-timing phase observation
	var cleanup = this.observePhaseState(checkPhaseState, onPhaseTransition, options);
	
	// Also listen for wiki changes (phase resonance - event-driven phase transitions)
	if($tw && $tw.wiki && $tw.wiki.addEventListener) {
		var listener = function(changes) {
			try {
				var currentPhase = checkPhaseState();
				if(currentPhase === targetPhase) {
					// Phase transition occurred via event (phase resonance)
					var phaseTransition = {
						experientialTime: self.experientialTime,
						clockRate: 0.01,
						semanticTransitions: self.semanticTransitions,
						clockTime: Date.now(),
						details: { 
							type: "event-driven-phase-transition", 
							phase: currentPhase,
							via: "phase-resonance"
						}
					};
					
					self.notifyObservers(phaseTransition);
					
					if(onPhaseTransition) {
						try {
							onPhaseTransition(phaseTransition, self.experientialTime);
						} catch(e) {
							if(self.logging && $tw && $tw.log) {
								$tw.log("antclock-engine", "Event-driven phase transition handler error:", e);
							}
						}
					}
					
					// Cleanup
					if(cleanup) cleanup();
					if(options.removeListenerOnReady) {
						$tw.wiki.removeEventListener("change", listener);
					}
				}
			} catch(e) {
				if(self.logging && $tw && $tw.log) {
					$tw.log("antclock-engine", "Phase resonance listener error:", e);
				}
			}
		};
		
		try {
			$tw.wiki.addEventListener("change", listener);
			
			// Return combined cleanup
			return function() {
				if(cleanup) cleanup();
				if($tw && $tw.wiki && $tw.wiki.removeEventListener) {
					$tw.wiki.removeEventListener("change", listener);
				}
			};
		} catch(e) {
			if(self.logging && $tw && $tw.log) {
				$tw.log("antclock-engine", "Failed to add phase resonance listener:", e);
			}
			return cleanup;
		}
	}
	
	return cleanup;
};

// Export singleton instance
var defaultEngine = new AntclockEngine({name: "default"});

// Export both class and instance
exports.AntclockEngine = AntclockEngine;
exports.default = defaultEngine;
exports.create = function(options) {
	return new AntclockEngine(options);
};

})();
