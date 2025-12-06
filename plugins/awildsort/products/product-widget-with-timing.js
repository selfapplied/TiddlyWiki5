/*\
title: $:/plugins/awildsort/products/product-widget-with-timing.js
type: application/javascript
module-type: widget

Product widget with antclock timing for dependency resolution
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

var Widget = require("$:/core/modules/widgets/widget.js").widget;
var antclockTiming = require("$:/core/modules/utils/antclock-timing.js");

var ProductWidget = function(parseTreeNode,options) {
	this.initialise(parseTreeNode,options);
};

ProductWidget.prototype = new Widget();

ProductWidget.prototype.execute = function() {
	this.productTitle = this.getAttribute("title", "");
	this.productImage = this.getAttribute("image", "");
	this.productDescription = this.getAttribute("description", "");
	this.productPrice = this.getAttribute("price", "");
	this.makeChildWidgets();
};

ProductWidget.prototype.render = function(parent,nextSibling) {
	var self = this;
	
	// Use antclock timing to wait for dependencies
	antclockTiming.widgetInit(this, [
		// Wait for any required macros/widgets
		// {type: "macro", name: "$some-helper"},
		// Wait for config tiddler
		// {type: "tiddler", name: "$:/config/Products/Enabled", field: "text", value: "yes"},
		// Wait for DOM container
		// {type: "element", name: ".product-container"}
	], function() {
		// All dependencies ready - render widget
		self.parentDomNode = parent;
		self.computeAttributes();
		self.execute();
		
		var domNode = self.document.createElement("div");
		domNode.className = "aws-card";
		// ... render widget content ...
		
		parent.insertBefore(domNode,nextSibling);
		self.domNodes.push(domNode);
	});
};

exports.product = ProductWidget;

})();


