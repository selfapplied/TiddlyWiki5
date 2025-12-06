/*\
title: $:/themes/awildsort/scroll-handler.js
type: application/javascript
module-type: startup

Handle scroll effects for navigation
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

exports.name = "awildsort-scroll-handler";
exports.platforms = ["browser"];
exports.after = ["startup"];
exports.synchronous = true;

exports.startup = function() {
	if($tw.browser) {
		var topbar = document.querySelector(".tc-topbar");
		if(topbar) {
			var lastScroll = 0;
			window.addEventListener("scroll", function() {
				var currentScroll = window.pageYOffset || document.documentElement.scrollTop;
				if(currentScroll > 50) {
					topbar.classList.add("scrolled");
				} else {
					topbar.classList.remove("scrolled");
				}
				lastScroll = currentScroll;
			});
		}
	}
};

})();

