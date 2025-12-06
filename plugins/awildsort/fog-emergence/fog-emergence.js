/*\
title: $:/plugins/awildsort/fog-emergence/fog-emergence.js
type: application/javascript
module-type: startup

Declarative scroll-based fog emergence - minimal JS, mostly CSS
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

exports.name = "fog-emergence";
exports.platforms = ["browser"];
exports.after = ["startup"];
exports.synchronous = true;

exports.startup = function() {
	if($tw.browser) {
		// Just set up scroll-based CSS variables - everything else is declarative CSS
		window.addEventListener("scroll", function() {
			var scrollY = window.pageYOffset || document.documentElement.scrollTop;
			var scrollPercent = Math.min(scrollY / (document.documentElement.scrollHeight - window.innerHeight), 1);
			document.documentElement.style.setProperty("--aws-scroll-progress", scrollPercent);
		}, { passive: true });
		
		// Initial scroll value
		var scrollY = window.pageYOffset || document.documentElement.scrollTop;
		var scrollPercent = Math.min(scrollY / (document.documentElement.scrollHeight - window.innerHeight), 1);
		document.documentElement.style.setProperty("--aws-scroll-progress", scrollPercent);
	}
};

})();

