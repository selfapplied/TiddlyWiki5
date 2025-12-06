/*\
title: $:/core/modules/startup/fractal-parallax.js
type: application/javascript
module-type: startup

Fractal background parallax effect handling

\*/

"use strict";

// Export name and synchronous status
exports.name = "fractal-parallax";
exports.platforms = ["browser"];
exports.after = ["startup"];
exports.synchronous = true;

// Settings tiddlers
var PARALLAX_ENABLED = "$:/themes/tiddlywiki/vanilla/options/fractalparallax";
var FRACTAL_BACKGROUND = "$:/themes/tiddlywiki/vanilla/settings/fractalbackground";

var scrollHandler = null;
var isEnabled = false;
var rafId = null;
var ticking = false;

function updateParallax() {
	var scrollY = window.pageYOffset || document.documentElement.scrollTop;
	var translateY = scrollY * 0.5; // Parallax factor
	
	// Update CSS custom property that can be used in stylesheets
	document.documentElement.style.setProperty('--fractal-parallax-y', translateY + 'px');
	ticking = false;
}

function requestParallaxUpdate() {
	if (!ticking) {
		rafId = window.requestAnimationFrame(updateParallax);
		ticking = true;
	}
}

function enableParallax() {
	if (!isEnabled && typeof window !== "undefined") {
		scrollHandler = requestParallaxUpdate;
		window.addEventListener('scroll', scrollHandler, { passive: true });
		isEnabled = true;
		updateParallax();
	}
}

function disableParallax() {
	if (isEnabled && scrollHandler) {
		window.removeEventListener('scroll', scrollHandler);
		scrollHandler = null;
		isEnabled = false;
		if (rafId) {
			window.cancelAnimationFrame(rafId);
			rafId = null;
		}
		ticking = false;
		if (document.documentElement) {
			document.documentElement.style.removeProperty('--fractal-parallax-y');
		}
	}
}

function checkSettings() {
	var parallaxEnabled = $tw.wiki.getTiddlerText(PARALLAX_ENABLED, "no");
	var fractalBackground = $tw.wiki.getTiddlerText(FRACTAL_BACKGROUND, "");
	
	if (parallaxEnabled === "yes" && fractalBackground !== "") {
		enableParallax();
	} else {
		disableParallax();
	}
}

exports.startup = function() {
	if (typeof window === "undefined") {
		return;
	}
	
	// Initial check
	checkSettings();
	
	// Monitor changes to settings
	$tw.wiki.addEventListener("change", function(changes) {
		if ($tw.utils.hop(changes, PARALLAX_ENABLED) || $tw.utils.hop(changes, FRACTAL_BACKGROUND)) {
			checkSettings();
		}
	});
};
