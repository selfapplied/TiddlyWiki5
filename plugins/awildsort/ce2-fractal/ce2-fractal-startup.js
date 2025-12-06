/*\
title: $:/plugins/awildsort/ce2-fractal/ce2-fractal-startup.js
type: application/javascript
module-type: startup

Inject declarative CE2 fractal SVG into page
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

exports.name = "ce2-fractal-startup";
exports.platforms = ["browser"];
exports.after = ["startup"];
exports.synchronous = true;

exports.startup = function() {
	if($tw.browser) {
		// Get the SVG tiddler content and render it
		var svgTiddler = $tw.wiki.getTiddler("$:/plugins/awildsort/ce2-fractal/ce2-fractal-svg");
		if(svgTiddler) {
			var parser = $tw.wiki.parseText("text/vnd.tiddlywiki", svgTiddler.fields.text);
			var widgetNode = $tw.wiki.makeWidget(parser);
			var container = $tw.fakeDocument.createElement("div");
			widgetNode.render(container, null);
			
			// Move the rendered content to the real document body
			while(container.firstChild) {
				document.body.appendChild(container.firstChild);
			}
		}
	}
};

})();

