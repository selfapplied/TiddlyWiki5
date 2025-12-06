/*\
title: $:/themes/awildsort/base/theme-init-timing.js
type: application/javascript
module-type: startup

Theme initialization with antclock timing for dependencies
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

var antclockTiming = require("$:/core/modules/utils/antclock-timing.js");

exports.name = "awildsort-theme-init-timing";
exports.platforms = ["browser"];
exports.after = ["startup"];
exports.synchronous = false;

exports.startup = function() {
	if($tw.browser) {
		// Wait for theme to be active
		antclockTiming.waitForTheme(
			"$:/themes/awildsort/base",
			function() {
				// Theme is active, initialize theme features
				initializeThemeFeatures();
			},
			{
				onStateChange: function(state, previousState) {
					// Log state transitions for debugging
					if($tw.perf) {
						$tw.perf.log("Theme state:", previousState, "->", state);
					}
				}
			}
		);
		
		// Also wait for required plugins
		antclockTiming.waitForPluginDependencies(
			"$:/themes/awildsort/base",
			function() {
				// All theme dependencies loaded
				initializeThemePlugins();
			}
		);
	}
};

function initializeThemeFeatures() {
	// Theme-specific initialization
	// e.g., setup scroll handlers, initialize components
}

function initializeThemePlugins() {
	// Plugin-specific initialization
	// e.g., ensure fractal background is ready
}

})();


