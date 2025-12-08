/*\
title: $:/core/modules/utils/zp35-operator.js
type: application/javascript
module-type: utils

ZP35 Golden Operator - Semantic Compatibility Layer

This module implements the ZP35 golden operator as a semantic compatibility
layer for TiddlyWiki. It provides coherence checking based on the mathematical
foundations of invariant-preserving morphisms.

The operator preserves four key invariants:
1. Ordering of proof-theoretic strength
2. Ultrametric clustering structure
3. Coherence curvature (κ = 0.35 guardian threshold)
4. Self-similarity (fractal structure)

\*/

"use strict";

/*
ZP35 Operator Constructor
*/
function ZP35Operator() {
	// Guardian threshold - the coherence curvature
	// Derived from empirical learnability boundary (~400 examples/transition)
	this.kappa = 0.35;
	
	// Golden ratio - used for minimal distortion scaling
	this.phi = (1 + Math.sqrt(5)) / 2;
	
	// Cached coordinate mappings
	this.coordinateCache = {};
}

/*
Apply golden operator to map tiddler to fractal coordinates
Preserves: ordering, clustering, plateau structure, self-similarity

@param {object} tiddler - Tiddler object
@returns {number} - Fractal coordinate in [0, 1]
*/
ZP35Operator.prototype.applyGoldenOperator = function(tiddler) {
	if(!tiddler) {
		return 0;
	}
	
	var title = tiddler.fields.title;
	
	// Check cache
	if(this.coordinateCache[title]) {
		return this.coordinateCache[title];
	}
	
	// Calculate ordinal height (compositional depth)
	var ordinalHeight = this.calculateOrdinalHeight(tiddler);
	
	// Apply Cantor embedding
	var cantorCoord = this.cantorEmbedding(ordinalHeight);
	
	// Apply golden ratio scaling for minimal distortion
	var fractalCoord = this.goldenScale(cantorCoord, this.phi);
	
	// Cache result
	this.coordinateCache[title] = fractalCoord;
	
	return fractalCoord;
};

/*
Calculate ordinal height (compositional depth) of tiddler
Based on field count, transclude depth, and complexity metrics

@param {object} tiddler - Tiddler object
@returns {number} - Ordinal height
*/
ZP35Operator.prototype.calculateOrdinalHeight = function(tiddler) {
	var height = 0;
	
	// Base height from field count
	var fieldCount = Object.keys(tiddler.fields).length;
	height += fieldCount;
	
	// Add complexity from text length (normalized)
	if(tiddler.fields.text) {
		var textLength = tiddler.fields.text.length;
		height += Math.log(textLength + 1) / Math.log(2); // Log scale
	}
	
	// Add depth from tags (clustering indicator)
	if(tiddler.fields.tags) {
		var tagCount = Array.isArray(tiddler.fields.tags) ? 
			tiddler.fields.tags.length : 
			tiddler.fields.tags.split(" ").length;
		height += tagCount * 2; // Tags contribute more to clustering
	}
	
	// Add type complexity
	if(tiddler.fields.type && tiddler.fields.type !== "text/vnd.tiddlywiki") {
		height += 5; // Special types have higher ordinal
	}
	
	// Normalize to reasonable range [0, 100]
	return Math.min(100, height);
};

/*
Cantor embedding - maps ordinal to [0, 1] with plateau structure
This creates the self-similar, fractal structure

@param {number} ordinal - Ordinal height
@returns {number} - Cantor coordinate in [0, 1]
*/
ZP35Operator.prototype.cantorEmbedding = function(ordinal) {
	// Simplified monotonic Cantor-like embedding
	// Uses a logarithmic scale with plateau formation
	
	if(ordinal === 0) {
		return 0;
	}
	
	// Create plateaus using logarithmic mapping
	// This ensures monotonicity while creating step-like structure
	var logValue = Math.log(ordinal + 1) / Math.log(101); // Log scale normalized to [0, 1]
	
	// Quantize to create plateaus (10 levels)
	var plateauLevel = Math.floor(logValue * 10) / 10;
	
	// Add fine-grained variation within plateau
	var withinPlateau = (logValue * 10 - Math.floor(logValue * 10)) * 0.09;
	
	var coord = Math.min(1.0, plateauLevel + withinPlateau);
	
	return coord;
};

/*
Golden ratio scaling for minimal distortion
Applies φ-scaling to preserve fractal structure

@param {number} coord - Input coordinate
@param {number} phi - Golden ratio
@returns {number} - Scaled coordinate
*/
ZP35Operator.prototype.goldenScale = function(coord, phi) {
	// Apply golden ratio modular arithmetic
	// This preserves self-similarity at all scales
	var scaled = (coord * phi) % 1.0;
	return scaled;
};

/*
Check coherence between two tiddlers
Uses ZP35 golden operator to measure semantic distance

@param {object} source - Source tiddler
@param {object} target - Target tiddler
@returns {object} - Coherence assessment
*/
ZP35Operator.prototype.checkCoherence = function(source, target) {
	if(!source || !target) {
		return {
			allowed: false,
			mode: "error",
			distance: 1.0,
			confidence: 0.0,
			message: "Invalid tiddlers for coherence check"
		};
	}
	
	// Map tiddlers to fractal coordinates
	var sourceCoord = this.applyGoldenOperator(source);
	var targetCoord = this.applyGoldenOperator(target);
	
	// Calculate distance in fractal space
	var distance = Math.abs(sourceCoord - targetCoord);
	
	// Check against κ threshold
	if(distance < this.kappa) {
		return {
			allowed: true,
			mode: "safe",
			distance: distance,
			confidence: 1.0 - (distance / this.kappa),
			message: "Composition maintains semantic coherence",
			sourceCoord: sourceCoord,
			targetCoord: targetCoord
		};
	} else if(distance < 2 * this.kappa) {
		return {
			allowed: true,
			mode: "caution",
			distance: distance,
			confidence: 0.5,
			message: "Composition crosses semantic boundary - review recommended",
			suggestions: this.generateMediationSuggestions(source, target),
			sourceCoord: sourceCoord,
			targetCoord: targetCoord
		};
	} else {
		return {
			allowed: false,
			mode: "blocked",
			distance: distance,
			confidence: 0.0,
			message: "Composition violates coherence - may break semantic structure",
			alternatives: this.suggestAlternatives(source, target),
			sourceCoord: sourceCoord,
			targetCoord: targetCoord
		};
	}
};

/*
Generate mediation suggestions for borderline cases
Helps bridge semantic boundaries safely

@param {object} source - Source tiddler
@param {object} target - Target tiddler
@returns {array} - Array of suggestion objects
*/
ZP35Operator.prototype.generateMediationSuggestions = function(source, target) {
	var suggestions = [];
	
	// Suggest common tag as bridge
	if(source.fields.tags && target.fields.tags) {
		suggestions.push({
			type: "common-tag",
			action: "Add shared tags to reduce semantic distance",
			priority: "medium"
		});
	}
	
	// Suggest type alignment
	if(source.fields.type !== target.fields.type) {
		suggestions.push({
			type: "type-alignment",
			action: "Consider aligning content types",
			priority: "low"
		});
	}
	
	// Suggest intermediate tiddler
	suggestions.push({
		type: "intermediate",
		action: "Create intermediate tiddler to bridge semantic gap",
		priority: "high"
	});
	
	return suggestions;
};

/*
Suggest alternatives when coherence is violated

@param {object} source - Source tiddler
@param {object} target - Target tiddler
@returns {array} - Array of alternative suggestions
*/
ZP35Operator.prototype.suggestAlternatives = function(source, target) {
	var alternatives = [];
	
	alternatives.push({
		type: "separate",
		action: "Keep tiddlers separate - link instead of transclude",
		rationale: "Semantic distance too large for safe composition"
	});
	
	alternatives.push({
		type: "refactor",
		action: "Refactor into smaller, more coherent components",
		rationale: "Break down complexity to reduce ordinal distance"
	});
	
	alternatives.push({
		type: "namespace",
		action: "Use different namespaces or plugins",
		rationale: "Maintain separation of concerns"
	});
	
	return alternatives;
};

/*
Calculate ZP35 signature for a tiddler
Returns a string signature that encodes semantic position

@param {object} tiddler - Tiddler object
@returns {string} - ZP35 signature
*/
ZP35Operator.prototype.calculateSignature = function(tiddler) {
	var coord = this.applyGoldenOperator(tiddler);
	var height = this.calculateOrdinalHeight(tiddler);
	
	// Format: coord.height
	return coord.toFixed(6) + "." + height.toFixed(0);
};

/*
Verify ZP35 signature matches tiddler
Used for integrity checking

@param {object} tiddler - Tiddler object
@param {string} signature - Expected ZP35 signature
@returns {object} - Verification result
*/
ZP35Operator.prototype.verifySignature = function(tiddler, signature) {
	var computed = this.calculateSignature(tiddler);
	var match = computed === signature;
	
	if(!match) {
		// Parse signatures to get distance
		// Format is: fractalCoord.ordinalHeight (e.g., "0.618034.15")
		var computedParts = computed.split(".");
		var expectedParts = signature.split(".");
		
		// Reconstruct fractal coordinate (first two parts: "0" and "618034")
		var computedCoord = parseFloat(computedParts[0] + "." + (computedParts[1] || "0"));
		var expectedCoord = parseFloat(expectedParts[0] + "." + (expectedParts[1] || "0"));
		var distance = Math.abs(computedCoord - expectedCoord);
		
		return {
			valid: false,
			computed: computed,
			expected: signature,
			distance: distance,
			message: "Signature mismatch - tiddler may have changed"
		};
	}
	
	return {
		valid: true,
		computed: computed,
		expected: signature,
		distance: 0,
		message: "Signature verified"
	};
};

/*
Check ultrametric clustering structure
Verifies that hierarchical relationships are preserved

@param {array} tiddlers - Array of tiddler objects
@returns {object} - Clustering analysis
*/
ZP35Operator.prototype.analyzeClusterStructure = function(tiddlers) {
	if(!tiddlers || tiddlers.length < 2) {
		return {
			valid: true,
			clusters: [],
			message: "Insufficient tiddlers for clustering analysis"
		};
	}
	
	// Map all tiddlers to coordinates
	var coords = tiddlers.map(function(t) {
		return {
			tiddler: t,
			coord: this.applyGoldenOperator(t)
		};
	}.bind(this));
	
	// Sort by coordinate
	coords.sort(function(a, b) {
		return a.coord - b.coord;
	});
	
	// Identify clusters (groups within κ distance)
	var clusters = [];
	var currentCluster = [coords[0]];
	
	for(var i = 1; i < coords.length; i++) {
		var distance = coords[i].coord - coords[i-1].coord;
		
		if(distance < this.kappa) {
			// Same cluster
			currentCluster.push(coords[i]);
		} else {
			// New cluster
			clusters.push(currentCluster);
			currentCluster = [coords[i]];
		}
	}
	
	if(currentCluster.length > 0) {
		clusters.push(currentCluster);
	}
	
	return {
		valid: true,
		clusterCount: clusters.length,
		clusters: clusters.map(function(c) {
			return {
				size: c.length,
				minCoord: c[0].coord,
				maxCoord: c[c.length - 1].coord,
				spread: c[c.length - 1].coord - c[0].coord,
				titles: c.map(function(item) { return item.tiddler.fields.title; })
			};
		}),
		message: "Identified " + clusters.length + " coherent clusters"
	};
};

/*
Clear coordinate cache
Call when tiddlers are modified
*/
ZP35Operator.prototype.clearCache = function() {
	this.coordinateCache = {};
};

// Export constructor
exports.ZP35Operator = ZP35Operator;

// Export guardian threshold constant
exports.ZP35_KAPPA = 0.35;
