/*\
title: $:/plugins/awildsort/gallery/gallery-widget.js
type: application/javascript
module-type: widget

Art gallery widget with lightbox
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

var Widget = require("$:/core/modules/widgets/widget.js").widget;

var GalleryWidget = function(parseTreeNode,options) {
	this.initialise(parseTreeNode,options);
};

GalleryWidget.prototype = new Widget();

GalleryWidget.prototype.render = function(parent,nextSibling) {
	this.parentDomNode = parent;
	this.computeAttributes();
	this.execute();
	
	var layout = this.getAttribute("layout", "masonry"); // masonry, grid, carousel
	var filter = this.getAttribute("filter", "[tag[Gallery]]");
	
	var domNode = this.document.createElement("div");
	domNode.className = "aws-gallery aws-gallery-" + layout;
	
	if(layout === "masonry") {
		domNode.style.columnCount = "3";
		domNode.style.columnGap = "1rem";
	}
	
	if(layout === "grid") {
		domNode.style.display = "grid";
		domNode.style.gridTemplateColumns = "repeat(auto-fill, minmax(250px, 1fr))";
		domNode.style.gap = "1rem";
	}
	
	var items = this.wiki.filterTiddlers(filter);
	var self = this;
	
	$tw.utils.each(items, function(title) {
		var tiddler = self.wiki.getTiddler(title);
		if(tiddler) {
			var item = self.document.createElement("div");
			item.className = "aws-gallery-item";
			item.style.marginBottom = "1rem";
			item.style.cursor = "pointer";
			item.style.transition = "transform 300ms, opacity 300ms";
			
			item.onmouseenter = function() {
				item.style.transform = "scale(1.02)";
				item.style.opacity = "0.9";
			};
			item.onmouseleave = function() {
				item.style.transform = "scale(1)";
				item.style.opacity = "1";
			};
			
			// Image or video
			if(tiddler.fields.image) {
				var img = self.document.createElement("img");
				img.src = tiddler.fields.image;
				img.style.width = "100%";
				img.style.borderRadius = "8px";
				img.alt = tiddler.fields.title || "";
				item.appendChild(img);
			} else if(tiddler.fields.video) {
				var video = self.document.createElement("video");
				video.src = tiddler.fields.video;
				video.controls = true;
				video.style.width = "100%";
				video.style.borderRadius = "8px";
				item.appendChild(video);
			}
			
			// Title
			if(tiddler.fields.title) {
				var titleEl = self.document.createElement("div");
				titleEl.textContent = tiddler.fields.title;
				titleEl.style.marginTop = "0.5rem";
				titleEl.style.fontSize = "0.875rem";
				titleEl.style.opacity = "0.8";
				item.appendChild(titleEl);
			}
			
			// Lightbox click
			item.onclick = function() {
				self.showLightbox(tiddler);
			};
			
			domNode.appendChild(item);
		}
	});
	
	parent.insertBefore(domNode,nextSibling);
	this.domNodes.push(domNode);
};

GalleryWidget.prototype.showLightbox = function(tiddler) {
	var lightbox = this.document.createElement("div");
	lightbox.className = "aws-lightbox";
	lightbox.style.position = "fixed";
	lightbox.style.top = "0";
	lightbox.style.left = "0";
	lightbox.style.width = "100%";
	lightbox.style.height = "100%";
	lightbox.style.backgroundColor = "rgba(0, 0, 0, 0.9)";
	lightbox.style.zIndex = "10000";
	lightbox.style.display = "flex";
	lightbox.style.alignItems = "center";
	lightbox.style.justifyContent = "center";
	lightbox.style.opacity = "0";
	lightbox.style.transition = "opacity 400ms";
	
	var content = this.document.createElement("div");
	content.style.position = "relative";
	content.style.maxWidth = "90%";
	content.style.maxHeight = "90%";
	
	if(tiddler.fields.image) {
		var img = this.document.createElement("img");
		img.src = tiddler.fields.image;
		img.style.maxWidth = "100%";
		img.style.maxHeight = "90vh";
		img.style.borderRadius = "8px";
		content.appendChild(img);
	} else if(tiddler.fields.video) {
		var video = this.document.createElement("video");
		video.src = tiddler.fields.video;
		video.controls = true;
		video.style.maxWidth = "100%";
		video.style.maxHeight = "90vh";
		content.appendChild(video);
	}
	
	// Close button
	var closeBtn = this.document.createElement("button");
	closeBtn.textContent = "×";
	closeBtn.style.position = "absolute";
	closeBtn.style.top = "-40px";
	closeBtn.style.right = "0";
	closeBtn.style.background = "transparent";
	closeBtn.style.border = "none";
	closeBtn.style.color = "white";
	closeBtn.style.fontSize = "2rem";
	closeBtn.style.cursor = "pointer";
	closeBtn.onclick = function() {
		lightbox.style.opacity = "0";
		setTimeout(function() {
			lightbox.remove();
		}, 400);
	};
	content.appendChild(closeBtn);
	
	// Title and notes
	if(tiddler.fields.title || tiddler.fields.notes) {
		var info = this.document.createElement("div");
		info.style.marginTop = "1rem";
		info.style.color = "white";
		info.style.textAlign = "center";
		
		if(tiddler.fields.title) {
			var title = this.document.createElement("h3");
			title.textContent = tiddler.fields.title;
			title.style.marginBottom = "0.5rem";
			info.appendChild(title);
		}
		
		if(tiddler.fields.notes) {
			var notes = this.document.createElement("p");
			notes.textContent = tiddler.fields.notes;
			notes.style.opacity = "0.8";
			info.appendChild(notes);
		}
		
		content.appendChild(info);
	}
	
	lightbox.appendChild(content);
	this.document.body.appendChild(lightbox);
	
	// Fade in
	setTimeout(function() {
		lightbox.style.opacity = "1";
	}, 10);
	
	// Close on click outside
	lightbox.onclick = function(e) {
		if(e.target === lightbox) {
			lightbox.style.opacity = "0";
			setTimeout(function() {
				lightbox.remove();
			}, 400);
		}
	};
	
	// Close on Escape
	var escapeHandler = function(e) {
		if(e.key === "Escape") {
			lightbox.style.opacity = "0";
			setTimeout(function() {
				lightbox.remove();
			}, 400);
			document.removeEventListener("keydown", escapeHandler);
		}
	};
	document.addEventListener("keydown", escapeHandler);
};

exports.gallery = GalleryWidget;

})();


