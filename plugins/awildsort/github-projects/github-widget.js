/*\
title: $:/plugins/awildsort/github-projects/github-widget.js
type: application/javascript
module-type: widget

GitHub projects browser widget
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

var Widget = require("$:/core/modules/widgets/widget.js").widget;

var GitHubWidget = function(parseTreeNode,options) {
	this.initialise(parseTreeNode,options);
};

GitHubWidget.prototype = new Widget();

GitHubWidget.prototype.render = function(parent,nextSibling) {
	this.parentDomNode = parent;
	this.computeAttributes();
	this.execute();
	
	var org = this.getAttribute("org", "selfapplied");
	var domNode = this.document.createElement("div");
	domNode.className = "aws-card-grid";
	
	var self = this;
	
	// Fetch GitHub repos
	var url = "https://api.github.com/orgs/" + org + "/repos?sort=updated&per_page=20";
	
	fetch(url)
		.then(function(response) {
			return response.json();
		})
		.then(function(repos) {
			$tw.utils.each(repos, function(repo) {
				var card = self.document.createElement("div");
				card.className = "aws-card";
				
				// Title
				var title = self.document.createElement("h3");
				title.textContent = repo.name;
				card.appendChild(title);
				
				// Description
				if(repo.description) {
					var desc = self.document.createElement("p");
					desc.textContent = repo.description;
					desc.style.marginBottom = "1rem";
					desc.style.opacity = "0.9";
					card.appendChild(desc);
				}
				
				// Stats
				var stats = self.document.createElement("div");
				stats.style.display = "flex";
				stats.style.gap = "1rem";
				stats.style.marginBottom = "1rem";
				stats.style.fontSize = "0.875rem";
				stats.style.opacity = "0.7";
				
				var stars = self.document.createElement("span");
				stars.textContent = "⭐ " + repo.stargazers_count;
				stats.appendChild(stars);
				
				var updated = self.document.createElement("span");
				var date = new Date(repo.updated_at);
				updated.textContent = "Updated: " + date.toLocaleDateString();
				stats.appendChild(updated);
				
				card.appendChild(stats);
				
				// Language
				if(repo.language) {
					var lang = self.document.createElement("span");
					lang.textContent = repo.language;
					lang.style.display = "inline-block";
					lang.style.padding = "0.25rem 0.5rem";
					lang.style.background = "var(--aws-glass-bg)";
					lang.style.borderRadius = "4px";
					lang.style.fontSize = "0.75rem";
					lang.style.marginBottom = "1rem";
					card.appendChild(lang);
				}
				
				// Link
				var link = self.document.createElement("a");
				link.href = repo.html_url;
				link.target = "_blank";
				link.rel = "noopener noreferrer";
				link.className = "tc-btn";
				link.textContent = "View on GitHub";
				link.style.display = "inline-block";
				card.appendChild(link);
				
				domNode.appendChild(card);
			});
		})
		.catch(function(error) {
			var errorMsg = self.document.createElement("div");
			errorMsg.className = "aws-card";
			errorMsg.textContent = "Error loading GitHub projects: " + error.message;
			errorMsg.style.color = "#ff6b6b";
			domNode.appendChild(errorMsg);
		});
	
	parent.insertBefore(domNode,nextSibling);
	this.domNodes.push(domNode);
};

exports["github-projects"] = GitHubWidget;

})();


