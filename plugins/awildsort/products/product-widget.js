/*\
title: $:/plugins/awildsort/products/product-widget.js
type: application/javascript
module-type: widget

Product card widget
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

var Widget = require("$:/core/modules/widgets/widget.js").widget;

var ProductWidget = function(parseTreeNode,options) {
	this.initialise(parseTreeNode,options);
};

ProductWidget.prototype = new Widget();

ProductWidget.prototype.execute = function() {
	this.productTitle = this.getAttribute("title", "");
	this.productImage = this.getAttribute("image", "");
	this.productDescription = this.getAttribute("description", "");
	this.productFullDescription = this.getAttribute("fulldescription", "");
	this.productPrice = this.getAttribute("price", "");
	this.productCategory = this.getAttribute("category", "");
	this.productLink = this.getAttribute("link", "");
	this.productCta = this.getAttribute("cta", "Learn More");
	this.makeChildWidgets();
};

ProductWidget.prototype.render = function(parent,nextSibling) {
	this.parentDomNode = parent;
	this.computeAttributes();
	this.execute();
	
	var domNode = this.document.createElement("div");
	domNode.className = "aws-card";
	
	// Image
	if(this.productImage) {
		var img = this.document.createElement("img");
		img.src = this.productImage;
		img.style.width = "100%";
		img.style.borderRadius = "8px";
		img.style.marginBottom = "1rem";
		img.alt = this.productTitle;
		domNode.appendChild(img);
	}
	
	// Title
	if(this.productTitle) {
		var title = this.document.createElement("h3");
		title.className = "aws-card-title";
		title.textContent = this.productTitle;
		domNode.appendChild(title);
	}
	
	// Category
	if(this.productCategory) {
		var category = this.document.createElement("span");
		category.className = "aws-card-category";
		category.textContent = this.productCategory;
		category.style.display = "block";
		category.style.fontSize = "0.875rem";
		category.style.opacity = "0.7";
		category.style.marginBottom = "0.5rem";
		domNode.appendChild(category);
	}
	
	// Description
	if(this.productDescription) {
		var desc = this.document.createElement("p");
		desc.className = "aws-card-description";
		desc.textContent = this.productDescription;
		desc.style.marginBottom = "1rem";
		domNode.appendChild(desc);
	}
	
	// Price
	if(this.productPrice) {
		var price = this.document.createElement("div");
		price.className = "aws-card-price";
		price.textContent = this.productPrice;
		price.style.fontSize = "1.25rem";
		price.style.fontWeight = "600";
		price.style.marginBottom = "1rem";
		price.style.color = "var(--aws-starlight)";
		domNode.appendChild(price);
	}
	
	// CTA Button
	if(this.productLink) {
		var cta = this.document.createElement("a");
		cta.href = this.productLink;
		cta.className = "tc-btn";
		cta.textContent = this.productCta;
		cta.style.display = "inline-block";
		domNode.appendChild(cta);
	}
	
	// Full description (hidden, expandable)
	if(this.productFullDescription) {
		var fullDesc = this.document.createElement("div");
		fullDesc.className = "aws-card-full-description";
		fullDesc.style.display = "none";
		fullDesc.textContent = this.productFullDescription;
		domNode.appendChild(fullDesc);
		
		var toggleBtn = this.document.createElement("button");
		toggleBtn.textContent = "Read More";
		toggleBtn.className = "tc-btn";
		toggleBtn.style.marginTop = "0.5rem";
		toggleBtn.style.fontSize = "0.875rem";
		toggleBtn.onclick = function() {
			if(fullDesc.style.display === "none") {
				fullDesc.style.display = "block";
				toggleBtn.textContent = "Read Less";
			} else {
				fullDesc.style.display = "none";
				toggleBtn.textContent = "Read More";
			}
		};
		domNode.appendChild(toggleBtn);
	}
	
	parent.insertBefore(domNode,nextSibling);
	this.domNodes.push(domNode);
};

exports.product = ProductWidget;

})();

