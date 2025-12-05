/*\
title: $:/plugins/awildsort/products/product-grid-widget.js
type: application/javascript
module-type: widget

Product grid widget that displays products in a responsive grid
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

var Widget = require("$:/core/modules/widgets/widget.js").widget;

var ProductGridWidget = function(parseTreeNode,options) {
	this.initialise(parseTreeNode,options);
};

ProductGridWidget.prototype = new Widget();

ProductGridWidget.prototype.render = function(parent,nextSibling) {
	this.parentDomNode = parent;
	this.computeAttributes();
	this.execute();
	
	var domNode = this.document.createElement("div");
	domNode.className = "aws-card-grid";
	
	// Get filter for products
	var filter = this.getAttribute("filter", "[tag[Product]]");
	var products = this.wiki.filterTiddlers(filter);
	
	var self = this;
	$tw.utils.each(products, function(title) {
		var tiddler = self.wiki.getTiddler(title);
		if(tiddler) {
			var productCard = self.document.createElement("div");
			productCard.className = "aws-card";
			
			// Image
			if(tiddler.fields.image) {
				var img = self.document.createElement("img");
				img.src = tiddler.fields.image;
				img.style.width = "100%";
				img.style.borderRadius = "8px";
				img.style.marginBottom = "1rem";
				img.alt = tiddler.fields.title;
				productCard.appendChild(img);
			}
			
			// Title
			var titleEl = self.document.createElement("h3");
			titleEl.textContent = tiddler.fields.title;
			productCard.appendChild(titleEl);
			
			// Category
			if(tiddler.fields.category) {
				var category = self.document.createElement("span");
				category.textContent = tiddler.fields.category;
				category.style.display = "block";
				category.style.fontSize = "0.875rem";
				category.style.opacity = "0.7";
				category.style.marginBottom = "0.5rem";
				productCard.appendChild(category);
			}
			
			// Description
			if(tiddler.fields.description) {
				var desc = self.document.createElement("p");
				desc.textContent = tiddler.fields.description;
				desc.style.marginBottom = "1rem";
				productCard.appendChild(desc);
			}
			
			// Price
			if(tiddler.fields.price) {
				var price = self.document.createElement("div");
				price.textContent = tiddler.fields.price;
				price.style.fontSize = "1.25rem";
				price.style.fontWeight = "600";
				price.style.marginBottom = "1rem";
				productCard.appendChild(price);
			}
			
			// Link
			if(tiddler.fields.link) {
				var cta = self.document.createElement("a");
				cta.href = tiddler.fields.link;
				cta.className = "tc-btn";
				cta.textContent = tiddler.fields.cta || "Learn More";
				cta.style.display = "inline-block";
				productCard.appendChild(cta);
			}
			
			domNode.appendChild(productCard);
		}
	});
	
	parent.insertBefore(domNode,nextSibling);
	this.domNodes.push(domNode);
};

exports["product-grid"] = ProductGridWidget;

})();


