/*\
title: $:/plugins/awildsort/popout-icon/popout-icon-widget.js
type: application/javascript
module-type: widget

Pop-out icon widget for tiddlers
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

var Widget = require("$:/core/modules/widgets/widget.js").widget;

var PopoutIconWidget = function(parseTreeNode,options) {
	this.initialise(parseTreeNode,options);
};

PopoutIconWidget.prototype = new Widget();

PopoutIconWidget.prototype.execute = function() {
	this.iconText = this.getAttribute("text", "!");
	this.iconSize = this.getAttribute("size", "medium"); // small, medium, large
	this.position = this.getAttribute("position", "top-right"); // top-right, top-left, bottom-right, bottom-left
	this.color = this.getAttribute("color", "");
	this.animation = this.getAttribute("animation", "pulse"); // pulse, bounce, rotate, none
	this.tooltip = this.getAttribute("tooltip", "");
	this.action = this.getAttribute("action", ""); // navigate, popup, message
	this.actionTarget = this.getAttribute("actionTarget", "");
	this.makeChildWidgets();
};

PopoutIconWidget.prototype.render = function(parent,nextSibling) {
	if(!$tw.browser) {
		return;
	}
	
	this.parentDomNode = parent;
	this.computeAttributes();
	this.execute();
	
	var self = this;
	
	// Create icon container
	var domNode = this.document.createElement("span");
	domNode.className = "aws-popout-icon aws-popout-icon-" + this.iconSize;
	domNode.className += " aws-popout-icon-" + this.position;
	domNode.className += " aws-popout-icon-" + this.animation;
	
	// Set color if provided
	if(this.color) {
		domNode.style.color = this.color;
		domNode.style.borderColor = this.color;
	}
	
	// Set tooltip
	if(this.tooltip) {
		domNode.setAttribute("title", this.tooltip);
	}
	
	// Icon text
	domNode.textContent = this.iconText;
	
	// Add click handler
	if(this.action) {
		domNode.style.cursor = "pointer";
		domNode.addEventListener("click", function(e) {
			e.stopPropagation();
			
			if(self.action === "navigate" && self.actionTarget) {
				$tw.wiki.setTextReference("$:/StoryList", "", "", self.actionTarget);
			} else if(self.action === "popup" && self.actionTarget) {
				var popupState = "$:/state/popup/" + self.actionTarget;
				var currentState = $tw.wiki.getTiddlerText(popupState);
				$tw.wiki.setText(popupState, "text", "", currentState === "yes" ? "no" : "yes");
			} else if(self.action === "message" && self.actionTarget) {
				$tw.rootWidget.dispatchEvent({
					type: self.actionTarget,
					param: self.getAttribute("actionParam", "")
				});
			}
		});
	}
	
	parent.insertBefore(domNode,nextSibling);
	this.domNodes.push(domNode);
};

exports["popout-icon"] = PopoutIconWidget;

})();


