/*\
title: $:/themes/awildsort/base/dark-mode-startup.js
type: application/javascript
module-type: startup

Initialize dark mode from saved preference
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

exports.name = "awildsort-dark-mode-startup";
exports.platforms = ["browser"];
exports.after = ["startup"];
exports.synchronous = true;

exports.startup = function() {
	if($tw.browser) {
		var toggleState = "$:/state/awildsort/lightmode";
		var savedMode = null;
		
		// Try to load from localStorage
		try {
			savedMode = localStorage.getItem("awildsort-theme-mode");
		} catch(e) {
			// Ignore localStorage errors
		}
		
		// Check system preference if no saved preference
		if(!savedMode && window.matchMedia) {
			var prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
			savedMode = prefersLight ? "light" : "dark";
		}
		
		// Apply mode
		if(savedMode === "light") {
			document.body.classList.add("light-mode");
			if(!$tw.wiki.getTiddler(toggleState)) {
				$tw.wiki.setText(toggleState, "text", "", "yes");
			}
		} else {
			document.body.classList.remove("light-mode");
			if(!$tw.wiki.getTiddler(toggleState)) {
				$tw.wiki.setText(toggleState, "text", "", "no");
			}
		}
		
		// Sync body class with state
		var currentState = $tw.wiki.getTiddlerText(toggleState) === "yes";
		if(currentState && !document.body.classList.contains("light-mode")) {
			document.body.classList.add("light-mode");
		} else if(!currentState && document.body.classList.contains("light-mode")) {
			document.body.classList.remove("light-mode");
		}
		
		// Listen for system preference changes
		if(window.matchMedia) {
			var mediaQuery = window.matchMedia("(prefers-color-scheme: light)");
			mediaQuery.addEventListener("change", function(e) {
				// Only auto-switch if user hasn't manually set a preference
				var manualPreference = null;
				try {
					manualPreference = localStorage.getItem("awildsort-theme-mode");
				} catch(err) {
					// Ignore
				}
				
				if(!manualPreference) {
					if(e.matches) {
						document.body.classList.add("light-mode");
						$tw.wiki.setText(toggleState, "text", "", "yes");
					} else {
						document.body.classList.remove("light-mode");
						$tw.wiki.setText(toggleState, "text", "", "no");
					}
				}
			});
		}
	}
};

})();

