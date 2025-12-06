/*\
title: $:/themes/awildsort/base/dark-mode-toggle-timing.js
type: application/javascript
module-type: startup

Antclock self-timing for widget registration - uses antclock engine semantic state observation
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

var antclockEngine = require("$:/core/modules/utils/antclock-engine.js").default;

exports.name = "dark-mode-toggle-timing";
exports.platforms = ["browser"];
exports.after = ["startup"];
exports.synchronous = false; // Async to allow self-timing

exports.startup = function() {
	// Grammar adjusting: safe browser check
	if(!$tw || !$tw.browser) {
		return;
	}
	
	// Grammar adjusting: ensure antclock engine is available
	if(!antclockEngine || typeof antclockEngine.observePhaseStateReactive !== "function") {
		// Fallback: try again after a delay
		setTimeout(function() {
			if(antclockEngine && typeof antclockEngine.observePhaseStateReactive === "function") {
				exports.startup();
			}
		}, 100);
		return;
	}
	
	// Use antclock phase state observation - the antclock way
	// Track widget availability as phase transitions: missing -> loading -> ready
	try {
		antclockEngine.observePhaseStateReactive(
			// Check current phase state
			function() {
				try {
					if(!$tw || !$tw.wiki) {
						return "initializing"; // Phase: system initializing
					}
					var widgetTiddler = $tw.wiki.getTiddler("$:/themes/awildsort/base/dark-mode-toggle");
					if(!widgetTiddler) {
						return "missing"; // Phase: widget definition not found
					}
					if(!$tw.macros || !$tw.utils || !$tw.utils.hop($tw.macros, "$dark-mode-toggle")) {
						return "loading"; // Phase: widget definition exists but not yet registered
					}
					return "ready"; // Phase: widget is available and registered
				} catch(e) {
					return "error"; // Phase: error state
				}
			},
			// On phase transition to "ready" phase
			function(phaseTransition, experientialTime) {
				try {
					// Ensure widget is properly registered
					if($tw && $tw.wiki && $tw.wiki.defineTiddlerModules) {
						$tw.wiki.defineTiddlerModules();
					}
					if($tw && $tw.wiki && $tw.wiki.initParsers) {
						$tw.wiki.initParsers();
					}
					// Trigger refresh if root widget exists
					if($tw && $tw.rootWidget && typeof $tw.rootWidget.refreshChildren === "function") {
						$tw.rootWidget.refreshChildren();
					}
				} catch(e) {
					// Grammar adjusting: continue even if initialization fails
					if($tw && $tw.log) {
						$tw.log("dark-mode-toggle-timing", "Initialization error:", e);
					}
				}
			},
			{
				targetPhase: "ready", // Target phase to transition to
				checkInterval: 50,
				maxChecks: 400, // ~20 seconds
				removeListenerOnReady: false // Keep listening for phase changes
			}
		);
	} catch(e) {
		// Grammar adjusting: if phase observation fails, log and continue
		if($tw && $tw.log) {
			$tw.log("dark-mode-toggle-timing", "Failed to observe phase state:", e);
		}
	}
	
	// Optional: log engine stats for debugging
	if($tw && $tw.wiki && $tw.wiki.getTiddlerText("$:/config/Debug/Antclock", "no") === "yes") {
		setTimeout(function() {
			var stats = antclockEngine.getStats();
			if($tw && $tw.log) {
				$tw.log("antclock-engine", "Stats:", stats);
			}
		}, 1000);
	}
};

})();

