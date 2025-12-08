# ZP35 Plugin Compatibility System

**Document Version:** 1.0  
**Date:** December 7, 2024  
**Purpose:** Guide for plugin developers on using the ZP35 plugin compatibility system  
**Status:** Implementation Guide

---

## Executive Summary

The ZP35 Plugin Compatibility System uses mathematical principles from the golden operator framework to automatically detect and resolve conflicts between TiddlyWiki plugins. This document explains how to use the system and how to make your plugins compatible with it.

**Key Features:**
- Automatic plugin compatibility analysis
- Detection of structural and semantic conflicts
- Suggested bridge morphisms for incompatible plugins
- Guardian threshold (κ = 0.35) for safe composition
- Recommendations for plugin installation

---

## 1. Understanding the Golden Operator

The golden operator maps plugins to a fractal coordinate space that preserves four key invariants:

1. **Ordering** - More complex plugins map to higher values
2. **Clustering** - Similar plugins stay close together
3. **Coherence** - The κ = 0.35 threshold marks safe composition
4. **Self-similarity** - Patterns repeat across scales

### 1.1 Feature Vector Extraction

Each plugin is analyzed to extract:

**Structural Features:**
- Compositional depth (transclusions, macros, widgets, filters)
- Global hooks and monkey patches
- Field modifications

**Semantic Features:**
- Sector: `editor`, `view`, `storage`, `sync`, `theme`, `viz`, `tool`
- Statefulness: `pure` vs `impure`
- Idempotence and commutativity

**Temporal Features:**
- Lifecycle phases: `startup`, `render`, `change`
- Event dependencies

**Topological Features:**
- Number of shadow tiddlers
- Interaction patterns

---

## 2. Using the Plugin Analyzer

### 2.1 Basic Usage

```javascript
// Load the plugin analyzer
var pluginAnalyzer = require("$:/core/modules/utils/plugin-analyzer.js");

// Analyze a single plugin
var entity = pluginAnalyzer.analyzePlugin(
	$tw.wiki,
	"$:/plugins/tiddlywiki/codemirror"
);

console.log("Plugin sector:", entity.sector);
console.log("Plugin depth:", entity.depth);
console.log("Statefulness:", entity.statefulness);
```

### 2.2 Checking Compatibility

```javascript
// Check if two plugins are compatible
var compatibility = pluginAnalyzer.checkPluginCompatibility(
	$tw.wiki,
	"$:/plugins/tiddlywiki/codemirror",
	"$:/plugins/tiddlywiki/markdown"
);

console.log("Compatible:", compatibility.compatible);
console.log("Mode:", compatibility.mode); // "safe", "caution", or "blocked"
console.log("Confidence:", compatibility.confidence); // 0.0 to 1.0
console.log("Edge strength:", compatibility.edgeStrength);
```

### 2.3 Building Compatibility Graph

```javascript
// Get all installed plugins
var plugins = $tw.wiki.filterTiddlers("[is[plugin]]");

// Build compatibility graph
var graph = pluginAnalyzer.buildCompatibilityGraph($tw.wiki, plugins);

console.log("Nodes:", Object.keys(graph.nodes).length);
console.log("Edges:", graph.edges.length);

// Find problematic edges
var problems = graph.edges.filter(function(edge) {
	return edge.mode === "caution" || edge.mode === "blocked";
});

console.log("Potential conflicts:", problems.length);
```

### 2.4 Finding Conflicts

```javascript
// Find all conflicts in current plugin set
var conflicts = pluginAnalyzer.findPluginConflicts($tw.wiki, plugins);

conflicts.forEach(function(conflict) {
	console.log("Conflict between:", conflict.pluginA, "and", conflict.pluginB);
	console.log("  Mode:", conflict.mode);
	console.log("  Strength:", conflict.strength);
	console.log("  Recommendation:", conflict.recommendation);
	
	if(conflict.bridge && conflict.bridge.exists) {
		console.log("  Bridge possible with adaptations:");
		conflict.bridge.adaptations.forEach(function(adapt) {
			console.log("    -", adapt.description);
		});
	}
});
```

---

## 3. The Guardian Triad

The system uses three guardian functions to measure compatibility:

### 3.1 Guardian ϕ (Phi) - Semantic Compatibility

Measures semantic distance between plugins:
- Different sectors (editor vs storage)
- Different statefulness (pure vs impure)
- Non-overlapping lifecycles

**Range:** [0, 1]  
**Threshold:** < κ/2 = 0.175 is safe

### 3.2 Guardian ∂ (Delta) - Structural Compatibility

Measures structural conflicts:
- Depth mismatches
- Conflicting global hooks
- Field write conflicts

**Range:** [0, 1]  
**Threshold:** < κ/2 = 0.175 is safe

### 3.3 Guardian ℛ (R) - Invariant Preservation

Measures coordinate distance in golden space:
- Normalized by κ
- Reflects compositional level differences

**Range:** [0, ∞)  
**Threshold:** < 1.0 is safe

### 3.4 Edge Strength Formula

The overall compatibility is measured by edge strength:

```
E(A,B) = √(ϕ² + ∂² + ℛ²)
```

**Interpretation:**
- E < κ (0.35): **Safe** - Plugins are compatible
- κ ≤ E < 2κ (0.70): **Caution** - Review recommended
- E ≥ 2κ: **Blocked** - Likely conflict

---

## 4. Bridge Morphisms

When plugins are incompatible, the system can suggest bridge morphisms - adapters that preserve invariants while enabling composition.

### 4.1 Finding Bridges

```javascript
var zp35 = require("$:/core/modules/utils/zp35-golden-operator.js");

var entityA = pluginAnalyzer.analyzePlugin($tw.wiki, pluginA);
var entityB = pluginAnalyzer.analyzePlugin($tw.wiki, pluginB);

var bridge = zp35.findBridgeMorphism(entityA, entityB);

if(bridge.exists) {
	console.log("Bridge coordinate:", bridge.coordinate);
	console.log("Distortion:", bridge.distortion);
	console.log("Required adaptations:");
	
	bridge.adaptations.forEach(function(adapt) {
		console.log("Type:", adapt.type);
		console.log("Description:", adapt.description);
		console.log("Code template:", adapt.code);
	});
}
```

### 4.2 Types of Adaptations

**Field Mapping:**
- Translates field names between plugins
- Example: `status` ↔ `state`

**Lifecycle Adapter:**
- Forwards events between different lifecycles
- Example: `tiddler` change → `story` change

**State Guard:**
- Isolates conflicting state modifications
- Prevents mutual interference

---

## 5. Making Your Plugin Compatible

### 5.1 Plugin Structure Best Practices

**Minimize Global State:**
```javascript
// Bad: Modifies global state
$tw.globalConfig = {...};

// Good: Uses tiddler storage
$tw.wiki.setTiddlerData("$:/config/myplugin", {...});
```

**Declare Dependencies:**
```json
{
  "title": "$:/plugins/author/myplugin",
  "dependents": ["$:/plugins/tiddlywiki/plugin-a"],
  "sector": "view",
  "statefulness": "pure"
}
```

**Use Proper Module Types:**
```javascript
/*\
title: $:/plugins/author/myplugin/widget.js
type: application/javascript
module-type: widget
\*/
```

### 5.2 Sector Classification

Help the analyzer by using descriptive names and metadata:

```json
{
  "title": "$:/plugins/author/editor-extension",
  "description": "Advanced text editor features",
  "sector-hint": "editor"
}
```

### 5.3 Idempotence and Commutativity

Design operations to be repeatable and order-independent when possible:

```javascript
// Idempotent: Can be called multiple times
function setConfig(key, value) {
  $tw.wiki.setTiddlerData(configTiddler, key, value);
}

// Non-idempotent: Accumulates
function addToList(item) {
  list.push(item); // Each call adds another item
}
```

---

## 6. Advanced Usage

### 6.1 Custom Sector Definition

Add custom sectors for specialized plugins:

```javascript
// In your plugin's code
var determineSector = pluginAnalyzer.determineSector;

pluginAnalyzer.determineSector = function(entity, pluginInfo) {
	// Check for your custom sector first
	if(/ai|gpt|llm/i.test(pluginInfo.description)) {
		return "ai-assistant";
	}
	
	// Fall back to default logic
	return determineSector(entity, pluginInfo);
};
```

### 6.2 Plugin Recommendations

Use the system to recommend compatible plugins to users:

```javascript
var installed = $tw.wiki.filterTiddlers("[is[plugin]]");
var candidates = [...]; // List of candidate plugins to evaluate

var recommendations = pluginAnalyzer.getPluginRecommendations(
	$tw.wiki,
	installed,
	candidates
);

// Sort by compatibility score
recommendations.forEach(function(rec) {
	console.log(rec.plugin);
	console.log("  Score:", rec.overallScore);
	console.log("  Compatible:", rec.compatible);
	console.log("  Conflicts:", rec.conflictCount);
});
```

### 6.3 Continuous Monitoring

Monitor plugin compatibility as users install new plugins:

```javascript
// Listen for plugin installations
$tw.wiki.addEventListener("change", function(changes) {
	var pluginChanges = Object.keys(changes).filter(function(title) {
		return /^\$:\/plugins\//.test(title);
	});
	
	if(pluginChanges.length > 0) {
		// Re-analyze compatibility
		var plugins = $tw.wiki.filterTiddlers("[is[plugin]]");
		var conflicts = pluginAnalyzer.findPluginConflicts($tw.wiki, plugins);
		
		if(conflicts.length > 0) {
			// Notify user
			$tw.notifier.display("$:/core/ui/Notifications/plugin-conflicts", {
				conflicts: conflicts
			});
		}
	}
});
```

---

## 7. Troubleshooting

### 7.1 False Positives

If the system incorrectly flags compatible plugins:

1. Check if plugins have proper metadata
2. Verify sector classification
3. Consider adding compatibility hints
4. Report issue with details

### 7.2 Missing Conflicts

If the system misses actual conflicts:

1. Check plugin's global state modifications
2. Verify lifecycle phase detection
3. Look for monkey patching
4. Report with reproduction steps

### 7.3 Performance

For wikis with many plugins:

1. Cache analysis results
2. Use incremental updates
3. Limit graph depth
4. Consider background processing

---

## 8. API Reference

### 8.1 Plugin Analyzer API

```javascript
pluginAnalyzer.analyzePlugin(wiki, pluginTitle)
  → entity | null

pluginAnalyzer.checkPluginCompatibility(wiki, titleA, titleB)
  → { compatible, mode, edgeStrength, confidence, ... }

pluginAnalyzer.buildCompatibilityGraph(wiki, pluginTitles)
  → { nodes, edges }

pluginAnalyzer.findPluginConflicts(wiki, pluginTitles)
  → [{ pluginA, pluginB, mode, bridge, recommendation }]

pluginAnalyzer.getPluginRecommendations(wiki, installed, candidates)
  → [{ plugin, overallScore, compatible, conflicts }]
```

### 8.2 ZP35 Golden Operator API

```javascript
zp35.applyGoldenOperator(entity)
  → coordinate [0,1]

zp35.extractFeatureVector(entity)
  → { depth, sector, statefulness, lifecycle, ... }

zp35.calculateCompatibility(entityA, entityB)
  → { edgeStrength, phi, delta, r, compatible, mode, ... }

zp35.findBridgeMorphism(entityA, entityB)
  → { exists, coordinate, distortion, adaptations }
```

---

## 9. Examples

### 9.1 Check Before Installing

```javascript
function checkBeforeInstall(newPluginTitle) {
	var installed = $tw.wiki.filterTiddlers("[is[plugin]]");
	var conflicts = [];
	
	installed.forEach(function(existingPlugin) {
		var compat = pluginAnalyzer.checkPluginCompatibility(
			$tw.wiki,
			existingPlugin,
			newPluginTitle
		);
		
		if(compat.mode === "caution" || compat.mode === "blocked") {
			conflicts.push({
				with: existingPlugin,
				mode: compat.mode,
				message: compat.message
			});
		}
	});
	
	if(conflicts.length > 0) {
		console.warn("Potential conflicts detected:");
		conflicts.forEach(function(c) {
			console.log(" -", c.with, ":", c.message);
		});
		return false;
	}
	
	return true;
}
```

### 9.2 Visualize Compatibility Graph

```javascript
function generateGraphViz() {
	var plugins = $tw.wiki.filterTiddlers("[is[plugin]]");
	var graph = pluginAnalyzer.buildCompatibilityGraph($tw.wiki, plugins);
	
	var dot = "digraph G {\n";
	
	// Add nodes
	Object.keys(graph.nodes).forEach(function(title) {
		var node = graph.nodes[title];
		var shortTitle = title.replace(/^\$:\/plugins\//, "");
		dot += '  "' + shortTitle + '" [label="' + shortTitle + 
		       '\\n' + node.sector + '"];\n';
	});
	
	// Add edges (only problematic ones)
	graph.edges.forEach(function(edge) {
		if(edge.mode !== "safe") {
			var color = edge.mode === "caution" ? "orange" : "red";
			var source = edge.source.replace(/^\$:\/plugins\//, "");
			var target = edge.target.replace(/^\$:\/plugins\//, "");
			dot += '  "' + source + '" -> "' + target + 
			       '" [color=' + color + '];\n';
		}
	});
	
	dot += "}\n";
	return dot;
}
```

---

## 10. Future Enhancements

Planned improvements:

1. **Machine Learning Integration** - Learn from user feedback
2. **Automatic Bridge Generation** - Generate adapter plugins
3. **Plugin Marketplace Integration** - Show compatibility before download
4. **Version Compatibility** - Track compatibility across plugin versions
5. **Community Ratings** - Crowdsource compatibility data

---

## 11. Mathematical Foundations

For the detailed mathematical theory behind the golden operator, see:
- `ZP35_GOLDEN_OPERATOR.md` - Mathematical foundations
- `ZP35_TIDDLYWIKI_ENHANCEMENTS.md` - Practical applications
- `ANTCLOCK_RECOMMENDATIONS.md` - CE Tower architecture

---

## 12. Contributing

To improve the plugin compatibility system:

1. Report false positives/negatives with examples
2. Suggest new adaptation types
3. Contribute sector classifications
4. Add test cases for edge cases
5. Improve bridge morphism generation

**Contact:** See repository contributors

---

## License

This system is part of TiddlyWiki5 and follows the same BSD-3-Clause license.
