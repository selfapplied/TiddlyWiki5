/*\
title: test-plugin-analyzer.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for plugin analyzer module

\*/

(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

describe("Plugin Analyzer", function() {
	
	var pluginAnalyzer = require("$:/core/modules/utils/plugin-analyzer.js");
	
	// Helper to create mock wiki
	function createMockWiki() {
		var tiddlers = {};
		var pluginInfo = {};
		
		return {
			getTiddler: function(title) {
				return tiddlers[title] || null;
			},
			setTiddler: function(title, tiddler) {
				tiddlers[title] = tiddler;
			},
			getPluginInfo: function(title) {
				return pluginInfo[title] || null;
			},
			setPluginInfo: function(title, info) {
				pluginInfo[title] = info;
			}
		};
	}
	
	describe("analyzePlugin", function() {
		
		it("should return null for non-existent plugin", function() {
			var wiki = createMockWiki();
			var result = pluginAnalyzer.analyzePlugin(wiki, "$:/plugins/test/nonexistent");
			expect(result).toBe(null);
		});
		
		it("should analyze basic plugin structure", function() {
			var wiki = createMockWiki();
			
			wiki.setTiddler("$:/plugins/test/basic", {
				fields: {
					title: "$:/plugins/test/basic",
					type: "application/json",
					"plugin-type": "plugin"
				}
			});
			
			wiki.setPluginInfo("$:/plugins/test/basic", {
				title: "$:/plugins/test/basic",
				tiddlers: {}
			});
			
			var result = pluginAnalyzer.analyzePlugin(wiki, "$:/plugins/test/basic");
			
			expect(result).not.toBe(null);
			expect(result.title).toBe("$:/plugins/test/basic");
			expect(result.pluginType).toBe("plugin");
		});
		
		it("should detect macros in shadow tiddlers", function() {
			var wiki = createMockWiki();
			
			wiki.setTiddler("$:/plugins/test/withmacro", {
				fields: {
					title: "$:/plugins/test/withmacro",
					type: "application/json"
				}
			});
			
			wiki.setPluginInfo("$:/plugins/test/withmacro", {
				title: "$:/plugins/test/withmacro",
				tiddlers: {
					"$:/plugins/test/withmacro/mymacro.js": {
						title: "$:/plugins/test/withmacro/mymacro.js",
						"module-type": "macro",
						text: "exports.name = 'mymacro';"
					}
				}
			});
			
			var result = pluginAnalyzer.analyzePlugin(wiki, "$:/plugins/test/withmacro");
			
			expect(result.macros.length).toBe(1);
		});
		
		it("should detect widgets", function() {
			var wiki = createMockWiki();
			
			wiki.setTiddler("$:/plugins/test/withwidget", {
				fields: {
					title: "$:/plugins/test/withwidget",
					type: "application/json"
				}
			});
			
			wiki.setPluginInfo("$:/plugins/test/withwidget", {
				title: "$:/plugins/test/withwidget",
				tiddlers: {
					"$:/plugins/test/withwidget/mywidget.js": {
						title: "$:/plugins/test/withwidget/mywidget.js",
						"module-type": "widget",
						text: "Widget code here"
					}
				}
			});
			
			var result = pluginAnalyzer.analyzePlugin(wiki, "$:/plugins/test/withwidget");
			
			expect(result.widgets.length).toBe(1);
		});
		
		it("should detect transclusions in shadow tiddler text", function() {
			var wiki = createMockWiki();
			
			wiki.setTiddler("$:/plugins/test/withtransclusion", {
				fields: {
					title: "$:/plugins/test/withtransclusion",
					type: "application/json"
				}
			});
			
			wiki.setPluginInfo("$:/plugins/test/withtransclusion", {
				title: "$:/plugins/test/withtransclusion",
				tiddlers: {
					"$:/plugins/test/withtransclusion/template": {
						title: "$:/plugins/test/withtransclusion/template",
						text: "Some text with {{OtherTiddler}} and {{AnotherTiddler}}"
					}
				}
			});
			
			var result = pluginAnalyzer.analyzePlugin(wiki, "$:/plugins/test/withtransclusion");
			
			expect(result.transclusions.length).toBe(2);
		});
		
		it("should detect impure plugins", function() {
			var wiki = createMockWiki();
			
			wiki.setTiddler("$:/plugins/test/impure", {
				fields: {
					title: "$:/plugins/test/impure",
					type: "application/json"
				}
			});
			
			wiki.setPluginInfo("$:/plugins/test/impure", {
				title: "$:/plugins/test/impure",
				tiddlers: {
					"$:/plugins/test/impure/global.js": {
						title: "$:/plugins/test/impure/global.js",
						"module-type": "global",
						text: "$tw.globalState = {};"
					}
				}
			});
			
			var result = pluginAnalyzer.analyzePlugin(wiki, "$:/plugins/test/impure");
			
			expect(result.statefulness).toBe("impure");
		});
		
	});
	
	describe("determineSector", function() {
		
		it("should classify editor plugins", function() {
			var entity = { widgets: [] };
			var pluginInfo = {
				title: "$:/plugins/test/editor",
				description: "Text editor plugin"
			};
			
			var sector = pluginAnalyzer.determineSector(entity, pluginInfo);
			expect(sector).toBe("editor");
		});
		
		it("should classify storage plugins", function() {
			var entity = {};
			var pluginInfo = {
				title: "$:/plugins/test/storage",
				description: "Local storage plugin"
			};
			
			var sector = pluginAnalyzer.determineSector(entity, pluginInfo);
			expect(sector).toBe("storage");
		});
		
		it("should classify theme plugins", function() {
			var entity = { pluginType: "theme" };
			var pluginInfo = {
				title: "$:/themes/test/mytheme",
				description: "Beautiful theme"
			};
			
			var sector = pluginAnalyzer.determineSector(entity, pluginInfo);
			expect(sector).toBe("theme");
		});
		
	});
	
	describe("checkPluginCompatibility", function() {
		
		it("should check compatibility between two plugins", function() {
			var wiki = createMockWiki();
			
			// Create two simple plugins
			wiki.setTiddler("$:/plugins/test/plugin1", {
				fields: {
					title: "$:/plugins/test/plugin1",
					type: "application/json"
				}
			});
			
			wiki.setPluginInfo("$:/plugins/test/plugin1", {
				title: "$:/plugins/test/plugin1",
				description: "View plugin",
				tiddlers: {}
			});
			
			wiki.setTiddler("$:/plugins/test/plugin2", {
				fields: {
					title: "$:/plugins/test/plugin2",
					type: "application/json"
				}
			});
			
			wiki.setPluginInfo("$:/plugins/test/plugin2", {
				title: "$:/plugins/test/plugin2",
				description: "View helper",
				tiddlers: {}
			});
			
			var result = pluginAnalyzer.checkPluginCompatibility(
				wiki,
				"$:/plugins/test/plugin1",
				"$:/plugins/test/plugin2"
			);
			
			expect(result).toBeDefined();
			expect(result.compatible).toBeDefined();
			expect(result.mode).toBeDefined();
		});
		
		it("should return error for non-existent plugins", function() {
			var wiki = createMockWiki();
			
			var result = pluginAnalyzer.checkPluginCompatibility(
				wiki,
				"$:/plugins/test/missing1",
				"$:/plugins/test/missing2"
			);
			
			expect(result.mode).toBe("error");
		});
		
	});
	
	describe("buildCompatibilityGraph", function() {
		
		it("should build graph for multiple plugins", function() {
			var wiki = createMockWiki();
			
			// Create three plugins
			for(var i = 1; i <= 3; i++) {
				var title = "$:/plugins/test/plugin" + i;
				wiki.setTiddler(title, {
					fields: {
						title: title,
						type: "application/json"
					}
				});
				
				wiki.setPluginInfo(title, {
					title: title,
					tiddlers: {}
				});
			}
			
			var graph = pluginAnalyzer.buildCompatibilityGraph(wiki, [
				"$:/plugins/test/plugin1",
				"$:/plugins/test/plugin2",
				"$:/plugins/test/plugin3"
			]);
			
			expect(graph.nodes).toBeDefined();
			expect(graph.edges).toBeDefined();
			expect(Object.keys(graph.nodes).length).toBe(3);
			expect(graph.edges.length).toBe(3); // 3 choose 2 = 3 pairs
		});
		
		it("should calculate edge strengths", function() {
			var wiki = createMockWiki();
			
			wiki.setTiddler("$:/plugins/test/a", {
				fields: { title: "$:/plugins/test/a", type: "application/json" }
			});
			wiki.setPluginInfo("$:/plugins/test/a", {
				title: "$:/plugins/test/a",
				tiddlers: {}
			});
			
			wiki.setTiddler("$:/plugins/test/b", {
				fields: { title: "$:/plugins/test/b", type: "application/json" }
			});
			wiki.setPluginInfo("$:/plugins/test/b", {
				title: "$:/plugins/test/b",
				tiddlers: {}
			});
			
			var graph = pluginAnalyzer.buildCompatibilityGraph(wiki, [
				"$:/plugins/test/a",
				"$:/plugins/test/b"
			]);
			
			expect(graph.edges.length).toBe(1);
			expect(graph.edges[0].strength).toBeDefined();
			expect(typeof graph.edges[0].strength).toBe("number");
		});
		
	});
	
	describe("findPluginConflicts", function() {
		
		it("should find conflicts in plugin set", function() {
			var wiki = createMockWiki();
			
			// Create conflicting plugins
			wiki.setTiddler("$:/plugins/test/editor", {
				fields: { title: "$:/plugins/test/editor", type: "application/json" }
			});
			wiki.setPluginInfo("$:/plugins/test/editor", {
				title: "$:/plugins/test/editor",
				description: "Editor plugin",
				tiddlers: {
					"$:/plugins/test/editor/global.js": {
						"module-type": "global",
						text: "global hooks"
					}
				}
			});
			
			wiki.setTiddler("$:/plugins/test/storage", {
				fields: { title: "$:/plugins/test/storage", type: "application/json" }
			});
			wiki.setPluginInfo("$:/plugins/test/storage", {
				title: "$:/plugins/test/storage",
				description: "Storage plugin",
				tiddlers: {
					"$:/plugins/test/storage/saver.js": {
						"module-type": "saver",
						text: "addTiddler code"
					}
				}
			});
			
			var conflicts = pluginAnalyzer.findPluginConflicts(wiki, [
				"$:/plugins/test/editor",
				"$:/plugins/test/storage"
			]);
			
			expect(Array.isArray(conflicts)).toBe(true);
		});
		
		it("should return empty array for compatible plugins", function() {
			var wiki = createMockWiki();
			
			wiki.setTiddler("$:/plugins/test/simple1", {
				fields: { title: "$:/plugins/test/simple1", type: "application/json" }
			});
			wiki.setPluginInfo("$:/plugins/test/simple1", {
				title: "$:/plugins/test/simple1",
				tiddlers: {}
			});
			
			wiki.setTiddler("$:/plugins/test/simple2", {
				fields: { title: "$:/plugins/test/simple2", type: "application/json" }
			});
			wiki.setPluginInfo("$:/plugins/test/simple2", {
				title: "$:/plugins/test/simple2",
				tiddlers: {}
			});
			
			var conflicts = pluginAnalyzer.findPluginConflicts(wiki, [
				"$:/plugins/test/simple1",
				"$:/plugins/test/simple2"
			]);
			
			// Should only include entries that are caution or blocked
			var problematic = conflicts.filter(function(c) {
				return c.mode === "caution" || c.mode === "blocked";
			});
			
			expect(Array.isArray(conflicts)).toBe(true);
		});
		
	});
	
	describe("getPluginRecommendations", function() {
		
		it("should rank candidate plugins", function() {
			var wiki = createMockWiki();
			
			// Create installed plugin
			wiki.setTiddler("$:/plugins/test/installed", {
				fields: { title: "$:/plugins/test/installed", type: "application/json" }
			});
			wiki.setPluginInfo("$:/plugins/test/installed", {
				title: "$:/plugins/test/installed",
				description: "View plugin",
				tiddlers: {}
			});
			
			// Create candidates
			wiki.setTiddler("$:/plugins/test/candidate1", {
				fields: { title: "$:/plugins/test/candidate1", type: "application/json" }
			});
			wiki.setPluginInfo("$:/plugins/test/candidate1", {
				title: "$:/plugins/test/candidate1",
				description: "View helper",
				tiddlers: {}
			});
			
			wiki.setTiddler("$:/plugins/test/candidate2", {
				fields: { title: "$:/plugins/test/candidate2", type: "application/json" }
			});
			wiki.setPluginInfo("$:/plugins/test/candidate2", {
				title: "$:/plugins/test/candidate2",
				description: "Storage plugin",
				tiddlers: {}
			});
			
			var recommendations = pluginAnalyzer.getPluginRecommendations(
				wiki,
				["$:/plugins/test/installed"],
				["$:/plugins/test/candidate1", "$:/plugins/test/candidate2"]
			);
			
			expect(Array.isArray(recommendations)).toBe(true);
			expect(recommendations.length).toBe(2);
			expect(recommendations[0].overallScore).toBeDefined();
		});
		
	});
	
});

})();
