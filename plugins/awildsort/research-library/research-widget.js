/*\
title: $:/plugins/awildsort/research-library/research-widget.js
type: application/javascript
module-type: widget

Research library widget with filtering
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

var Widget = require("$:/core/modules/widgets/widget.js").widget;

var ResearchWidget = function(parseTreeNode,options) {
	this.initialise(parseTreeNode,options);
};

ResearchWidget.prototype = new Widget();

ResearchWidget.prototype.render = function(parent,nextSibling) {
	this.parentDomNode = parent;
	this.computeAttributes();
	this.execute();
	
	var filter = this.getAttribute("filter", "[tag[Research]]");
	var showFilters = this.getAttribute("showFilters", "yes") === "yes";
	
	var container = this.document.createElement("div");
	container.className = "aws-research-library";
	
	// Filter controls
	if(showFilters) {
		var filterContainer = this.document.createElement("div");
		filterContainer.className = "aws-research-filters";
		filterContainer.style.marginBottom = "2rem";
		filterContainer.style.display = "flex";
		filterContainer.style.gap = "1rem";
		filterContainer.style.flexWrap = "wrap";
		
		var topics = ["chaos", "renormalization", "symbolic-dynamics", "CE1", "field-equations", "fractals"];
		var self = this;
		
		$tw.utils.each(topics, function(topic) {
			var btn = self.document.createElement("button");
			btn.textContent = topic;
			btn.className = "tc-btn";
			btn.dataset.topic = topic;
			btn.onclick = function() {
				// Toggle active state
				if(btn.classList.contains("active")) {
					btn.classList.remove("active");
					btn.style.background = "var(--aws-glass-bg)";
				} else {
					btn.classList.add("active");
					btn.style.background = "rgba(255, 255, 255, 0.2)";
				}
				self.filterPapers();
			};
			filterContainer.appendChild(btn);
		});
		
		container.appendChild(filterContainer);
	}
	
	// Papers container
	var papersContainer = this.document.createElement("div");
	papersContainer.className = "aws-research-papers";
	this.papersContainer = papersContainer;
	this.filter = filter;
	this.wiki = this.wiki;
	
	container.appendChild(papersContainer);
	this.renderPapers();
	
	parent.insertBefore(container,nextSibling);
	this.domNodes.push(container);
};

ResearchWidget.prototype.renderPapers = function() {
	var self = this;
	var papers = this.wiki.filterTiddlers(this.filter);
	
	// Clear container
	while(this.papersContainer.firstChild) {
		this.papersContainer.removeChild(this.papersContainer.firstChild);
	}
	
	$tw.utils.each(papers, function(title) {
		var tiddler = self.wiki.getTiddler(title);
		if(tiddler) {
			var paper = self.document.createElement("div");
			paper.className = "aws-card";
			
			// Title
			var titleEl = self.document.createElement("h3");
			titleEl.textContent = tiddler.fields.title;
			paper.appendChild(titleEl);
			
			// Authors
			if(tiddler.fields.authors) {
				var authors = self.document.createElement("div");
				authors.textContent = tiddler.fields.authors;
				authors.style.marginBottom = "0.5rem";
				authors.style.opacity = "0.8";
				authors.style.fontSize = "0.875rem";
				paper.appendChild(authors);
			}
			
			// Abstract
			if(tiddler.fields.abstract) {
				var abstract = self.document.createElement("div");
				abstract.className = "aws-research-abstract";
				abstract.textContent = tiddler.fields.abstract;
				abstract.style.marginBottom = "1rem";
				abstract.style.padding = "1rem";
				abstract.style.background = "rgba(255, 255, 255, 0.05)";
				abstract.style.borderRadius = "8px";
				abstract.style.fontStyle = "italic";
				paper.appendChild(abstract);
			}
			
			// Topics/Tags
			if(tiddler.fields.tags) {
				var tags = self.document.createElement("div");
				tags.style.display = "flex";
				tags.style.gap = "0.5rem";
				tags.style.flexWrap = "wrap";
				tags.style.marginBottom = "1rem";
				
				var tagList = tiddler.fields.tags;
				$tw.utils.each(tagList, function(tag) {
					if(tag !== "Research") {
						var tagEl = self.document.createElement("span");
						tagEl.textContent = tag;
						tagEl.style.padding = "0.25rem 0.5rem";
						tagEl.style.background = "var(--aws-glass-bg)";
						tagEl.style.borderRadius = "4px";
						tagEl.style.fontSize = "0.75rem";
						tags.appendChild(tagEl);
					}
				});
				
				paper.appendChild(tags);
			}
			
			// Links
			var links = self.document.createElement("div");
			links.style.display = "flex";
			links.style.gap = "0.5rem";
			
			if(tiddler.fields.pdf) {
				var pdfLink = self.document.createElement("a");
				pdfLink.href = tiddler.fields.pdf;
				pdfLink.textContent = "PDF";
				pdfLink.className = "tc-btn";
				pdfLink.target = "_blank";
				links.appendChild(pdfLink);
			}
			
			if(tiddler.fields.arxiv) {
				var arxivLink = self.document.createElement("a");
				arxivLink.href = tiddler.fields.arxiv;
				arxivLink.textContent = "arXiv";
				arxivLink.className = "tc-btn";
				arxivLink.target = "_blank";
				links.appendChild(arxivLink);
			}
			
			if(tiddler.fields.link) {
				var link = self.document.createElement("a");
				link.href = tiddler.fields.link;
				link.textContent = "Read";
				link.className = "tc-btn";
				link.target = "_blank";
				links.appendChild(link);
			}
			
			// View full paper link
			var viewLink = self.document.createElement("a");
			viewLink.href = "#" + title;
			viewLink.textContent = "View Full Paper";
			viewLink.className = "tc-btn";
			viewLink.onclick = function(e) {
				e.preventDefault();
				$tw.wiki.setTextReference(title, "", "", "");
				$tw.wiki.setTextReference(title, "text", "", "");
				$tw.wiki.setTextReference(title, "text", "", tiddler.fields.text || "");
				window.location.hash = title;
			};
			links.appendChild(viewLink);
			
			paper.appendChild(links);
			
			self.papersContainer.appendChild(paper);
		}
	});
};

ResearchWidget.prototype.filterPapers = function() {
	// Get active filters
	var activeFilters = [];
	var filterButtons = this.domNodes[0].querySelectorAll(".aws-research-filters .active");
	$tw.utils.each(filterButtons, function(btn) {
		activeFilters.push(btn.dataset.topic);
	});
	
	// Filter papers
	var papers = this.papersContainer.querySelectorAll(".aws-card");
	$tw.utils.each(papers, function(paper) {
		var shouldShow = activeFilters.length === 0;
		
		if(!shouldShow) {
			var tags = paper.querySelectorAll("span");
			$tw.utils.each(tags, function(tag) {
				if(activeFilters.indexOf(tag.textContent) !== -1) {
					shouldShow = true;
				}
			});
		}
		
		paper.style.display = shouldShow ? "block" : "none";
	});
};

exports["research-library"] = ResearchWidget;

})();


