/*\
title: $:/core/modules/utils/antclock-timing.js
type: application/javascript
module-type: utils

Antclock Self-Timing Utility - Experiential time-based dependency resolution

Based on CE2 antclock concepts: tracks meaningful state changes rather than clock time.
Useful for coordinating async initialization, widget availability, and module dependencies.
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

/**
 * Antclock self-timing: wait for a condition using experiential time (state changes)
 * rather than positional time (clock ticks).
 * 
 * @param {Function} checkCondition - Function that returns current state: "ready", "pending", or "missing"
 * @param {Function} onReady - Callback when condition becomes ready
 * @param {Object} options - Configuration options
 * @param {Number} options.maxAttempts - Maximum self-timing attempts (default: 20)
 * @param {Number} options.baseDelay - Base delay in ms for exponential backoff (default: 50)
 * @param {Number} options.maxDelay - Maximum delay between attempts (default: 500)
 * @param {Function} options.onStateChange - Callback when state transitions (state, previousState)
 * @param {Function} options.onTimeout - Callback if max attempts reached without success
 */
exports.waitForCondition = function(checkCondition, onReady, options) {
	// Grammar adjusting: normalize inputs
	if(typeof checkCondition !== "function") {
		if(onReady) onReady("error", 0);
		return;
	}
	options = options || {};
	var maxAttempts = Math.max(1, parseInt(options.maxAttempts, 10) || 20);
	var baseDelay = Math.max(10, parseInt(options.baseDelay, 10) || 50);
	var maxDelay = Math.max(baseDelay, parseInt(options.maxDelay, 10) || 500);
	
	var selfTiming = function(attempt, lastState) {
		// Grammar adjusting: normalize state
		attempt = Math.max(0, parseInt(attempt, 10) || 0);
		lastState = lastState || "unknown";
		
		// Grammar adjusting: safe condition check with error handling
		var currentState;
		try {
			currentState = checkCondition();
			// Normalize state to expected values
			if(currentState !== "ready" && currentState !== "pending" && currentState !== "missing") {
				currentState = "pending"; // Default to pending for unknown states
			}
		} catch(e) {
			// Grammar adjusting: handle errors gracefully
			currentState = "pending";
			if(options.onError) {
				options.onError(e, attempt);
			}
		}
		
		// State transition detected (experiential time - meaningful change)
		if(currentState !== lastState && options.onStateChange) {
			try {
				options.onStateChange(currentState, lastState);
			} catch(e) {
				// Grammar adjusting: continue even if callback fails
			}
		}
		
		// Ready state reached
		if(currentState === "ready") {
			if(onReady) {
				try {
					onReady(currentState, attempt);
				} catch(e) {
					// Grammar adjusting: log but don't break
					if($tw && $tw.log) {
						$tw.log("antclock-timing", "onReady callback error:", e);
					}
				}
			}
			return; // Success - stop self-timing
		}
		
		// Continue self-timing if not ready and attempts remain
		if(currentState !== "ready" && attempt < maxAttempts) {
			// Exponential backoff with jitter (experiential time scaling)
			// Grammar adjusting: ensure delay is valid
			var delay = Math.min(
				Math.max(10, baseDelay * Math.pow(1.5, attempt) + Math.random() * 20),
				maxDelay
			);
			try {
				setTimeout(function() {
					selfTiming(attempt + 1, currentState);
				}, delay);
			} catch(e) {
				// Grammar adjusting: if setTimeout fails, try immediate retry
				if(attempt < maxAttempts - 1) {
					selfTiming(attempt + 1, currentState);
				}
			}
		} else if(attempt >= maxAttempts) {
			// Timeout reached
			if(options.onTimeout) {
				try {
					options.onTimeout(currentState, attempt);
				} catch(e) {
					// Grammar adjusting: continue even if timeout handler fails
				}
			}
		}
	};
	
	// Start self-timing immediately
	try {
		selfTiming(0, "initializing");
	} catch(e) {
		// Grammar adjusting: if initial call fails, try once more
		if(onReady) {
			setTimeout(function() {
				try {
					selfTiming(0, "initializing");
				} catch(e2) {
					if(onReady) onReady("error", 0);
				}
			}, 10);
		}
	}
};

/**
 * Wait for a tiddler to be available and have a specific field value
 */
exports.waitForTiddler = function(tiddlerTitle, fieldName, expectedValue, onReady, options) {
	// Grammar adjusting: normalize inputs
	if(!tiddlerTitle || typeof tiddlerTitle !== "string") {
		if(onReady) onReady();
		return;
	}
	if(!$tw || !$tw.wiki) {
		if(onReady) onReady();
		return;
	}
	
	exports.waitForCondition(function() {
		try {
			var tiddler = $tw.wiki.getTiddler(tiddlerTitle);
			if(!tiddler) {
				return "missing";
			}
			if(fieldName) {
				var value = tiddler.fields && tiddler.fields[fieldName];
				if(expectedValue !== undefined && value !== expectedValue) {
					return "pending";
				}
			}
			return "ready";
		} catch(e) {
			return "pending"; // Grammar adjusting: retry on error
		}
	}, onReady, options);
};

/**
 * Wait for a macro/widget to be registered
 */
exports.waitForMacro = function(macroName, onReady, options) {
	// Grammar adjusting: normalize inputs
	if(!macroName || typeof macroName !== "string") {
		if(onReady) onReady();
		return;
	}
	if(!$tw || !$tw.wiki) {
		if(onReady) onReady();
		return;
	}
	
	exports.waitForCondition(function() {
		try {
			var tiddler = $tw.wiki.getTiddler(macroName);
			if(!tiddler) {
				return "missing";
			}
			if(!$tw.macros || !$tw.utils || !$tw.utils.hop($tw.macros, macroName)) {
				return "pending";
			}
			return "ready";
		} catch(e) {
			return "pending"; // Grammar adjusting: retry on error
		}
	}, function() {
		// Grammar adjusting: ensure modules are defined when ready, with error handling
		try {
			if($tw && $tw.wiki && $tw.wiki.defineTiddlerModules) {
				$tw.wiki.defineTiddlerModules();
			}
			if($tw && $tw.wiki && $tw.wiki.initParsers) {
				$tw.wiki.initParsers();
			}
		} catch(e) {
			// Grammar adjusting: continue even if module definition fails
		}
		if(onReady) {
			try {
				onReady();
			} catch(e) {
				// Grammar adjusting: don't break if callback fails
			}
		}
	}, options);
};

/**
 * Wait for a plugin to be loaded
 */
exports.waitForPlugin = function(pluginTitle, onReady, options) {
	exports.waitForCondition(function() {
		var plugin = $tw.wiki.getTiddler(pluginTitle);
		if(!plugin) {
			return "missing";
		}
		// Check if plugin tiddlers are unpacked
		var pluginInfo = $tw.wiki.getPluginInfo(pluginTitle);
		if(!pluginInfo) {
			return "pending";
		}
		return "ready";
	}, onReady, options);
};

/**
 * Wait for DOM element to be available
 */
exports.waitForElement = function(selector, onReady, options) {
	if(!$tw.browser) {
		if(onReady) onReady(null);
		return;
	}
	
	exports.waitForCondition(function() {
		var element = document.querySelector(selector);
		if(!element) {
			return "missing";
		}
		return "ready";
	}, function() {
		if(onReady) {
			onReady(document.querySelector(selector));
		}
	}, options);
};

/**
 * Reactive listener: watch for state changes and react immediately
 * Combines self-timing with event-driven reactivity
 */
exports.reactiveWait = function(checkCondition, onReady, options) {
	// Grammar adjusting: normalize inputs
	if(typeof checkCondition !== "function") {
		if(onReady) onReady("error", 0);
		return;
	}
	options = options || {};
	
	// Start self-timing
	exports.waitForCondition(checkCondition, onReady, options);
	
	// Also listen for wiki changes (reactivity)
	// Grammar adjusting: safe event listener setup
	if($tw && $tw.wiki && $tw.wiki.addEventListener) {
		var listener = function(changes) {
			try {
				var currentState = checkCondition();
				if(currentState === "ready") {
					if(onReady) {
						try {
							onReady(currentState, 0);
						} catch(e) {
							// Grammar adjusting: continue even if callback fails
						}
					}
					// Optionally remove listener after success
					if(options.removeListenerOnReady && $tw && $tw.wiki && $tw.wiki.removeEventListener) {
						try {
							$tw.wiki.removeEventListener("change", listener);
						} catch(e) {
							// Grammar adjusting: continue even if removal fails
						}
					}
				}
			} catch(e) {
				// Grammar adjusting: continue even if check fails
			}
		};
		try {
			$tw.wiki.addEventListener("change", listener);
			
			// Return cleanup function
			return function() {
				try {
					if($tw && $tw.wiki && $tw.wiki.removeEventListener) {
						$tw.wiki.removeEventListener("change", listener);
					}
				} catch(e) {
					// Grammar adjusting: continue even if cleanup fails
				}
			};
		} catch(e) {
			// Grammar adjusting: if event listener setup fails, continue without it
		}
	}
};

/**
 * Wait for widget class to be available
 * Useful in widget initialization
 */
exports.waitForWidget = function(widgetName, onReady, options) {
	var Widget = require("$:/core/modules/widgets/widget.js").widget;
	exports.waitForCondition(function() {
		if(!Widget || !Widget.prototype || !Widget.prototype.widgetClasses) {
			return "pending";
		}
		var widgetClasses = Widget.prototype.widgetClasses;
		if(!widgetClasses || !$tw.utils.hop(widgetClasses, widgetName)) {
			return "pending";
		}
		return "ready";
	}, onReady, options);
};

/**
 * Wait for story view to be available
 * Useful in story view initialization
 */
exports.waitForStoryView = function(storyViewName, onReady, options) {
	exports.waitForCondition(function() {
		if(!$tw.modules) {
			return "pending";
		}
		var storyViews = {};
		$tw.modules.applyMethods("storyview", storyViews);
		if(!storyViews || !$tw.utils.hop(storyViews, storyViewName)) {
			return "pending";
		}
		return "ready";
	}, function() {
		if(onReady) {
			var storyViews = {};
			$tw.modules.applyMethods("storyview", storyViews);
			onReady(storyViews[storyViewName]);
		}
	}, options);
};

/**
 * Wait for theme to be loaded and active
 * Useful in theme initialization
 */
exports.waitForTheme = function(themeTitle, onReady, options) {
	exports.waitForCondition(function() {
		var themeTiddler = $tw.wiki.getTiddler(themeTitle);
		if(!themeTiddler) {
			return "missing";
		}
		var currentTheme = $tw.wiki.getTiddlerText("$:/theme");
		if(currentTheme !== themeTitle) {
			return "pending";
		}
		// Check if theme modules are loaded
		if(!$tw.wiki.getPluginInfo(themeTitle)) {
			return "pending";
		}
		return "ready";
	}, onReady, options);
};

/**
 * Wait for plugin dependencies to be ready
 * Useful in plugin initialization
 */
exports.waitForPluginDependencies = function(pluginTitle, onReady, options) {
	exports.waitForCondition(function() {
		var plugin = $tw.wiki.getTiddler(pluginTitle);
		if(!plugin) {
			return "missing";
		}
		var pluginInfo = $tw.wiki.getPluginInfo(pluginTitle);
		if(!pluginInfo) {
			return "pending";
		}
		// Check dependencies
		var dependents = pluginInfo.dependents || [];
		var allReady = true;
		$tw.utils.each(dependents, function(depTitle) {
			var depInfo = $tw.wiki.getPluginInfo(depTitle);
			if(!depInfo) {
				allReady = false;
			}
		});
		return allReady ? "ready" : "pending";
	}, onReady, options);
};

/**
 * Widget initialization helper with antclock timing
 * Use in widget render() or execute() methods
 */
exports.widgetInit = function(widget, dependencies, onReady, options) {
	dependencies = dependencies || [];
	var readyCount = 0;
	var totalDeps = dependencies.length;
	
	if(totalDeps === 0) {
		if(onReady) onReady();
		return;
	}
	
	$tw.utils.each(dependencies, function(dep) {
		var depType = dep.type || "macro";
		var depName = dep.name;
		var checkFn;
		
		if(depType === "macro") {
			checkFn = function(callback) {
				exports.waitForMacro(depName, callback, {maxAttempts: 10});
			};
		} else if(depType === "widget") {
			checkFn = function(callback) {
				exports.waitForWidget(depName, callback, {maxAttempts: 10});
			};
		} else if(depType === "tiddler") {
			checkFn = function(callback) {
				exports.waitForTiddler(depName, dep.field, dep.value, callback, {maxAttempts: 10});
			};
		} else if(depType === "plugin") {
			checkFn = function(callback) {
				exports.waitForPlugin(depName, callback, {maxAttempts: 10});
			};
		} else if(depType === "element") {
			checkFn = function(callback) {
				exports.waitForElement(depName, callback, {maxAttempts: 10});
			};
		} else if(depType === "custom") {
			checkFn = dep.check || function(callback) { callback(); };
		}
		
		if(checkFn) {
			checkFn(function() {
				readyCount++;
				if(readyCount === totalDeps && onReady) {
					onReady();
				}
			});
		}
	});
};

})();

