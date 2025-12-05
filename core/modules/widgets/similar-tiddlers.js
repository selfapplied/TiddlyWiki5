/*\
title: $:/core/modules/widgets/similar-tiddlers.js
type: application/javascript
module-type: widget

Similar Tiddlers Widget

Displays tiddlers similar to the current one using witness fingerprint analysis.

Example usage:
<$similar-tiddlers tiddler="CurrentTiddler" threshold="0.3" max="5"/>

Attributes:
- tiddler: The target tiddler to find similarities for (defaults to current tiddler)
- threshold: Similarity threshold 0-1 (defaults to 0.3, lower = more similar required)
- max: Maximum number of results (defaults to 10)
- template: Optional template for rendering results

\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

var Widget = require("$:/core/modules/widgets/widget.js").widget;

var SimilarTiddlersWidget = function(parseTreeNode,options) {
	this.initialise(parseTreeNode,options);
};

/*
Inherit from the base widget class
*/
SimilarTiddlersWidget.prototype = new Widget();

/*
Render this widget into the DOM
*/
SimilarTiddlersWidget.prototype.render = function(parent,nextSibling) {
	this.parentDomNode = parent;
	this.computeAttributes();
	this.execute();
	
	// Create container
	var containerNode = this.document.createElement("div");
	containerNode.className = "tc-similar-tiddlers";
	parent.insertBefore(containerNode, nextSibling);
	this.domNodes.push(containerNode);
	
	// Find similar tiddlers
	var targetTiddler = this.wiki.getTiddler(this.tiddlerTitle);
	
	if(!targetTiddler) {
		containerNode.appendChild(this.document.createTextNode("No tiddler specified"));
		return;
	}
	
	// Get witness fingerprint utilities
	var witnessUtils = require("$:/core/modules/utils/witness-fingerprint.js");
	
	// Find similar tiddlers
	var similarTiddlers = witnessUtils.findSimilarTiddlers(targetTiddler, this.wiki, {
		threshold: this.threshold,
		maxResults: this.maxResults
	});
	
	if(similarTiddlers.length === 0) {
		var noResultsNode = this.document.createElement("p");
		noResultsNode.className = "tc-similar-tiddlers-none";
		noResultsNode.appendChild(this.document.createTextNode("No similar tiddlers found"));
		containerNode.appendChild(noResultsNode);
		return;
	}
	
	// Create list of similar tiddlers
	var listNode = this.document.createElement("ul");
	listNode.className = "tc-similar-tiddlers-list";
	
	for(var i = 0; i < similarTiddlers.length; i++) {
		var result = similarTiddlers[i];
		var itemNode = this.document.createElement("li");
		itemNode.className = "tc-similar-tiddlers-item";
		
		// Create link using TiddlyWiki's link handling
		var linkNode = this.document.createElement("a");
		linkNode.className = "tc-tiddlylink tc-tiddlylink-resolves";
		linkNode.setAttribute("href", "#" + encodeURIComponent(result.title));
		linkNode.onclick = function(event) {
			// Use TiddlyWiki's navigation mechanism
			event.preventDefault();
			new $tw.Story().navigateTiddler(result.title);
			return false;
		};
		linkNode.appendChild(this.document.createTextNode(result.title));
		
		// Add similarity score
		var scoreNode = this.document.createElement("span");
		scoreNode.className = "tc-similar-tiddlers-score";
		var similarityPercent = Math.round(result.similarity * 100);
		scoreNode.appendChild(this.document.createTextNode(" (" + similarityPercent + "% similar)"));
		
		itemNode.appendChild(linkNode);
		itemNode.appendChild(scoreNode);
		listNode.appendChild(itemNode);
	}
	
	containerNode.appendChild(listNode);
};

/*
Compute the internal state of the widget
*/
SimilarTiddlersWidget.prototype.execute = function() {
	// Get attributes
	this.tiddlerTitle = this.getAttribute("tiddler", this.getVariable("currentTiddler"));
	this.threshold = parseFloat(this.getAttribute("threshold", "0.3"));
	this.maxResults = parseInt(this.getAttribute("max", "10"));
	this.template = this.getAttribute("template");
};

/*
Selectively refreshes the widget if needed. Returns true if the widget or any of its children needed re-rendering
*/
SimilarTiddlersWidget.prototype.refresh = function(changedTiddlers) {
	var changedAttributes = this.computeAttributes();
	
	// Refresh if target tiddler or any attribute changed
	if(changedAttributes.tiddler || changedAttributes.threshold || changedAttributes.max || 
	   changedTiddlers[this.tiddlerTitle]) {
		this.refreshSelf();
		return true;
	}
	
	return false;
};

exports["similar-tiddlers"] = SimilarTiddlersWidget;

})();
