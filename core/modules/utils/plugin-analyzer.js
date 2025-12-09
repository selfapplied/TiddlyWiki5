/*\
title: $:/core/modules/utils/plugin-analyzer.js
type: application/javascript
module-type: utils

Plugin Analyzer - Extract feature vectors and analyze plugin compatibility using ZP35

This module analyzes TiddlyWiki plugins to extract:
- Structural features (depth, hooks, fields)
- Semantic features (sector, statefulness)
- Temporal features (lifecycle phases)
- Topological features (tiddler interactions)

\*/

(function(){

	"use strict";

	var zp35 = require("$:/core/modules/utils/zp35-golden-operator.js");

	/**
 * Analyze a plugin tiddler and extract its feature vector
 * @param {Object} wiki - The wiki object
 * @param {string} pluginTitle - Title of the plugin tiddler
 * @returns {Object} Plugin entity suitable for ZP35 analysis
 */
	function analyzePlugin(wiki, pluginTitle) {
		var pluginTiddler = wiki.getTiddler(pluginTitle);
	
		if(!pluginTiddler) {
			return null;
		}
	
		// Extract plugin metadata
		var pluginInfo = wiki.getPluginInfo(pluginTitle);
		var shadowTiddlers = pluginInfo ? pluginInfo.tiddlers : {};
	
		var entity = {
			title: pluginTitle,
			type: pluginTiddler.fields.type || "application/json",
			pluginType: pluginTiddler.fields["plugin-type"] || "plugin",
		
			// Structural analysis
			transclusions: [],
			macros: [],
			widgets: [],
			filters: [],
			hooks: [],
			fieldModifications: [],
			fields: {},
		
			// Semantic analysis
			sector: null,
			statefulness: "pure",
		
			// Temporal analysis
			startup: false,
			render: false,
			onChange: false,
		
			// Topological analysis
			tiddlerCount: Object.keys(shadowTiddlers).length
		};
	
		// Analyze shadow tiddlers
		Object.keys(shadowTiddlers).forEach(function(shadowTitle) {
			var shadowTiddler = shadowTiddlers[shadowTitle];
			var moduleType = shadowTiddler["module-type"];
		
			// Detect module types
			if(moduleType) {
				if(moduleType.indexOf("macro") !== -1) {
					entity.macros.push(shadowTitle);
				}
				if(moduleType.indexOf("widget") !== -1) {
					entity.widgets.push(shadowTitle);
				}
				if(moduleType.indexOf("filteroperator") !== -1) {
					entity.filters.push(shadowTitle);
				}
				if(moduleType.indexOf("startup") !== -1) {
					entity.startup = true;
				}
				if(moduleType.indexOf("global") !== -1) {
					entity.hooks.push(shadowTitle);
					entity.statefulness = "impure";
				}
			}
		
			// Analyze text content for patterns
			var text = shadowTiddler.text || "";
		
			// Detect transclusions
			var transclusionMatches = text.match(/\{\{[^}]+\}\}/g);
			if(transclusionMatches) {
				entity.transclusions = entity.transclusions.concat(transclusionMatches);
			}
		
			// Detect field modifications using regex patterns
			// Matches: addTiddler, setTiddlerData, deleteTiddler, setText, setFieldData
			var fieldModRegex = /\.(addTiddler|setTiddlerData|deleteTiddler|setText|setFieldData)\s*\(/;
			if(fieldModRegex.test(text)) {
				entity.fieldModifications.push(shadowTitle);
				entity.statefulness = "impure";
			}
		
			// Detect render-time behavior
			if(text.indexOf("render") !== -1 || text.indexOf("refreshSelf") !== -1) {
				entity.render = true;
			}
		
			// Detect change listeners
			if(text.indexOf("addEventListener") !== -1 || text.indexOf("onChange") !== -1) {
				entity.onChange = true;
			}
		});
	
		// Determine sector based on plugin type and content
		entity.sector = determineSector(entity, pluginInfo);
	
		return entity;
	}

	/**
 * Determine the primary sector of a plugin
 * @param {Object} entity - Plugin entity
 * @param {Object} pluginInfo - Plugin metadata
 * @returns {string} Sector classification
 */
	function determineSector(entity, pluginInfo) {
		var name = pluginInfo ? (pluginInfo.title || "") : "";
		var description = pluginInfo ? (pluginInfo.description || "") : "";
		var combined = (name + " " + description).toLowerCase();
	
		// Editor-related
		if(/editor|edit|text|input/i.test(combined) || (entity.widgets && entity.widgets.some(function(w) {
			return /editor|edit/i.test(w);
		}))) {
			return "editor";
		}
	
		// View/render-related
		if(/view|render|display|format|present/i.test(combined) || entity.render) {
			return "view";
		}
	
		// Storage-related
		if(/storage|saver|save|persist/i.test(combined)) {
			return "storage";
		}
	
		// Sync-related
		if(/sync|server|network|http/i.test(combined)) {
			return "sync";
		}
	
		// Theme-related
		if(/theme|style|color|palette/i.test(combined) || entity.pluginType === "theme") {
			return "theme";
		}
	
		// Visualization
		if(/chart|graph|diagram|viz|visual/i.test(combined)) {
			return "viz";
		}
	
		// Tool/utility
		if(/tool|util|helper/i.test(combined)) {
			return "tool";
		}
	
		return "unknown";
	}

	/**
 * Check compatibility between two plugins
 * @param {Object} wiki - The wiki object
 * @param {string} pluginTitleA - First plugin title
 * @param {string} pluginTitleB - Second plugin title
 * @returns {Object} Compatibility assessment
 */
	function checkPluginCompatibility(wiki, pluginTitleA, pluginTitleB) {
		var entityA = analyzePlugin(wiki, pluginTitleA);
		var entityB = analyzePlugin(wiki, pluginTitleB);
	
		if(!entityA || !entityB) {
			return {
				compatible: false,
				mode: "error",
				message: "One or both plugins not found"
			};
		}
	
		return zp35.calculateCompatibility(entityA, entityB);
	}

	/**
 * Build compatibility graph for all plugins in wiki
 * @param {Object} wiki - The wiki object
 * @param {Array} pluginTitles - Array of plugin titles to analyze
 * @returns {Object} Compatibility graph with nodes and edges
 */
	function buildCompatibilityGraph(wiki, pluginTitles) {
		var nodes = {};
		var edges = [];
	
		// Analyze each plugin
		pluginTitles.forEach(function(title) {
			var entity = analyzePlugin(wiki, title);
			if(entity) {
				var coord = zp35.applyGoldenOperator(entity);
				var features = zp35.extractFeatureVector(entity);
			
				nodes[title] = {
					title: title,
					coordinate: coord,
					sector: features.sector,
					depth: features.depth,
					statefulness: features.statefulness,
					entity: entity
				};
			}
		});
	
		// Calculate pairwise compatibility
		var titles = Object.keys(nodes);
		for(var i = 0; i < titles.length; i++) {
			for(var j = i + 1; j < titles.length; j++) {
				var titleA = titles[i];
				var titleB = titles[j];
				var entityA = nodes[titleA].entity;
				var entityB = nodes[titleB].entity;
			
				var compatibility = zp35.calculateCompatibility(entityA, entityB);
			
				edges.push({
					source: titleA,
					target: titleB,
					strength: compatibility.edgeStrength,
					mode: compatibility.mode,
					compatible: compatibility.compatible,
					confidence: compatibility.confidence,
					phi: compatibility.phi,
					delta: compatibility.delta,
					r: compatibility.r
				});
			}
		}
	
		return {
			nodes: nodes,
			edges: edges
		};
	}

	/**
 * Find potential conflicts in a set of plugins
 * @param {Object} wiki - The wiki object
 * @param {Array} pluginTitles - Array of plugin titles
 * @returns {Array} Array of conflict descriptions
 */
	function findPluginConflicts(wiki, pluginTitles) {
		var conflicts = [];
		var graph = buildCompatibilityGraph(wiki, pluginTitles);
	
		// Find incompatible edges
		graph.edges.forEach(function(edge) {
			if(!edge.compatible || edge.mode === "caution") {
				var bridge = zp35.findBridgeMorphism(
					graph.nodes[edge.source].entity,
					graph.nodes[edge.target].entity
				);
			
				conflicts.push({
					pluginA: edge.source,
					pluginB: edge.target,
					mode: edge.mode,
					strength: edge.strength,
					compatible: edge.compatible,
					confidence: edge.confidence,
					bridge: bridge,
					recommendation: generateRecommendation(edge, bridge)
				});
			}
		});
	
		return conflicts;
	}

	/**
 * Generate recommendation for resolving a conflict
 * @param {Object} edge - Compatibility edge
 * @param {Object} bridge - Bridge morphism
 * @returns {string} Recommendation text
 */
	function generateRecommendation(edge, bridge) {
		if(edge.mode === "safe") {
			return "No action needed - plugins are compatible";
		}
	
		if(edge.mode === "caution") {
			if(bridge.exists) {
				return "Consider creating an adapter plugin with: " + 
				bridge.adaptations.map(function(a) { return a.description; }).join(", ");
			}
			return "Review both plugins carefully - manual intervention may be needed";
		}
	
		if(edge.mode === "blocked") {
			if(bridge.exists) {
				return "Bridge morphism possible but complex. Adaptations needed: " +
				bridge.adaptations.map(function(a) { return a.description; }).join("; ");
			}
			return "Plugins are fundamentally incompatible - avoid using together";
		}
	
		return "Unknown compatibility status";
	}

	/**
 * Get plugin recommendations based on currently installed plugins
 * @param {Object} wiki - The wiki object
 * @param {Array} installedPlugins - Currently installed plugin titles
 * @param {Array} candidatePlugins - Candidate plugins to check
 * @returns {Array} Array of recommendations
 */
	function getPluginRecommendations(wiki, installedPlugins, candidatePlugins) {
		var recommendations = [];
	
		candidatePlugins.forEach(function(candidate) {
			var scores = [];
		
			installedPlugins.forEach(function(installed) {
				var compat = checkPluginCompatibility(wiki, installed, candidate);
				scores.push({
					with: installed,
					score: compat.confidence,
					mode: compat.mode
				});
			});
		
			// Calculate overall compatibility score
			var avgScore = scores.reduce(function(sum, s) {
				return sum + s.score;
			}, 0) / Math.max(1, scores.length);
		
			var conflicts = scores.filter(function(s) {
				return s.mode === "blocked" || s.mode === "caution";
			});
		
			recommendations.push({
				plugin: candidate,
				overallScore: avgScore,
				compatible: conflicts.length === 0,
				conflictCount: conflicts.length,
				conflicts: conflicts
			});
		});
	
		// Sort by overall score
		recommendations.sort(function(a, b) {
			return b.overallScore - a.overallScore;
		});
	
		return recommendations;
	}

	// Export functions
	exports.analyzePlugin = analyzePlugin;
	exports.determineSector = determineSector;
	exports.checkPluginCompatibility = checkPluginCompatibility;
	exports.buildCompatibilityGraph = buildCompatibilityGraph;
	exports.findPluginConflicts = findPluginConflicts;
	exports.getPluginRecommendations = getPluginRecommendations;

})();
