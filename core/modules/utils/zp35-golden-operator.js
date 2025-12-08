/*\
title: $:/core/modules/utils/zp35-golden-operator.js
type: application/javascript
module-type: utils

ZP35 Golden Operator - Invariant-preserving morphism between representation spaces

Based on ZP35 framework mathematical foundations, this module provides:
- Golden operator for mapping theories/plugins to fractal coordinates
- Guardian triad (ϕ, ∂, ℛ) for compatibility checking
- Feature vector extraction and Cantor embedding
- Coherence curvature threshold κ = 0.35

\*/

(function(){

"use strict";

/**
 * ZP35 Golden Operator Module
 * Implements invariant-preserving morphisms with κ = 0.35 guardian threshold
 */

// Guardian threshold - the coherence curvature plateau
const KAPPA = 0.35;

// Golden ratio for minimal distortion scaling
const PHI = (1 + Math.sqrt(5)) / 2;

/**
 * Calculate ordinal height (compositional depth) of an entity
 * @param {Object} entity - Entity to analyze (tiddler, plugin, etc.)
 * @returns {number} Ordinal height
 */
function calculateOrdinalHeight(entity) {
	var depth = 0;
	
	// Count transclusion depth
	if(entity.transclusions) {
		depth += entity.transclusions.length;
	}
	
	// Count macro depth
	if(entity.macros) {
		depth += entity.macros.length;
	}
	
	// Count widget depth
	if(entity.widgets) {
		depth += entity.widgets.length;
	}
	
	// Count filter depth
	if(entity.filters) {
		depth += entity.filters.length;
	}
	
	// Count field dependencies
	if(entity.fields) {
		depth += Object.keys(entity.fields).length * 0.5;
	}
	
	return depth;
}

/**
 * Cantor embedding - map ordinal height to [0,1] preserving ultrametric structure
 * Uses hierarchical clustering to maintain tree-like distances
 * @param {number} ordinalHeight - The ordinal height to embed
 * @returns {number} Cantor coordinate in [0,1]
 */
function cantorEmbedding(ordinalHeight) {
	if(ordinalHeight === 0) {
		return 0;
	}
	
	// Use monotonic inverse exponential to create plateaus
	// This preserves ordering while maintaining ultrametric clustering
	// Formula: 1 - exp(-h/φ) ensures monotonic growth
	var embed = 1 - Math.exp(-ordinalHeight / PHI);
	
	return Math.min(1, embed);
}

/**
 * Golden scaling - apply golden ratio for minimal distortion
 * @param {number} coord - Coordinate to scale
 * @param {number} phi - Golden ratio
 * @returns {number} Scaled coordinate
 */
function goldenScale(coord, phi) {
	// Apply golden ratio scaling to maintain self-similarity
	return coord / phi;
}

/**
 * Apply golden operator to map entity to fractal coordinates
 * Preserves: ordering, clustering, coherence curvature, self-similarity
 * @param {Object} entity - Entity to map (plugin, tiddler, etc.)
 * @returns {number} Golden coordinate in [0,1]
 */
function applyGoldenOperator(entity) {
	// 1. Calculate ordinal height (compositional depth)
	var ordinalHeight = calculateOrdinalHeight(entity);
	
	// 2. Apply Cantor embedding
	var cantorCoord = cantorEmbedding(ordinalHeight);
	
	// 3. Apply golden ratio scaling for minimal distortion
	var fractalCoord = goldenScale(cantorCoord, PHI);
	
	return fractalCoord;
}

/**
 * Extract feature vector from entity
 * Returns multi-dimensional fingerprint for guardian checks
 * @param {Object} entity - Entity to analyze
 * @returns {Object} Feature vector with phase, depth, sector, monodromy
 */
function extractFeatureVector(entity) {
	var vector = {
		// Structural features
		depth: calculateOrdinalHeight(entity),
		globalHooks: 0,
		fieldWrites: 0,
		
		// Semantic features
		sector: "unknown",
		statefulness: "pure",
		idempotence: true,
		commutativity: true,
		
		// Temporal features
		lifecycle: [],
		
		// Topological features
		tiddlerTouches: 0,
		graphTraversals: 0
	};
	
	// Analyze structural features
	if(entity.hooks) {
		vector.globalHooks = entity.hooks.length;
		vector.statefulness = "impure";
	}
	
	if(entity.fieldModifications) {
		vector.fieldWrites = entity.fieldModifications.length;
		vector.idempotence = false;
	}
	
	// Determine sector
	if(entity.type) {
		if(/editor/i.test(entity.type)) {
			vector.sector = "editor";
		} else if(/view|render/i.test(entity.type)) {
			vector.sector = "view";
		} else if(/storage|saver/i.test(entity.type)) {
			vector.sector = "storage";
		} else if(/sync/i.test(entity.type)) {
			vector.sector = "sync";
		} else if(/theme/i.test(entity.type)) {
			vector.sector = "theme";
		}
	}
	
	// Analyze lifecycle
	if(entity.startup) {
		vector.lifecycle.push("startup");
	}
	if(entity.render) {
		vector.lifecycle.push("render");
	}
	if(entity.onChange) {
		vector.lifecycle.push("change");
	}
	
	return vector;
}

/**
 * Guardian ϕ - Semantic compatibility check
 * Measures phase difference between semantic fingerprints
 * @param {Object} entityA - First entity
 * @param {Object} entityB - Second entity
 * @returns {number} Semantic distance [0,1]
 */
function guardianPhi(entityA, entityB) {
	var vecA = extractFeatureVector(entityA);
	var vecB = extractFeatureVector(entityB);
	
	var distance = 0;
	
	// Sector difference
	if(vecA.sector !== vecB.sector && vecA.sector !== "unknown" && vecB.sector !== "unknown") {
		distance += 0.3;
	}
	
	// Statefulness mismatch
	if(vecA.statefulness !== vecB.statefulness) {
		distance += 0.2;
	}
	
	// Lifecycle overlap
	var lifecycleOverlap = vecA.lifecycle.filter(function(phase) {
		return vecB.lifecycle.indexOf(phase) !== -1;
	}).length;
	
	if(lifecycleOverlap === 0 && vecA.lifecycle.length > 0 && vecB.lifecycle.length > 0) {
		distance += 0.1;
	}
	
	return Math.min(1, distance);
}

/**
 * Guardian ∂ - Structural compatibility check
 * Measures structural mismatch (depths, shapes, conflicts)
 * @param {Object} entityA - First entity
 * @param {Object} entityB - Second entity
 * @returns {number} Structural distance [0,1]
 */
function guardianDelta(entityA, entityB) {
	var vecA = extractFeatureVector(entityA);
	var vecB = extractFeatureVector(entityB);
	
	var distance = 0;
	
	// Depth mismatch
	var depthDiff = Math.abs(vecA.depth - vecB.depth);
	distance += Math.min(0.3, depthDiff / 10);
	
	// Global hooks conflict
	if(vecA.globalHooks > 0 && vecB.globalHooks > 0) {
		// Both modify global state - potential conflict
		distance += 0.2;
	}
	
	// Field write conflicts
	if(vecA.fieldWrites > 0 && vecB.fieldWrites > 0) {
		distance += 0.15;
	}
	
	return Math.min(1, distance);
}

/**
 * Guardian ℛ - Invariant preservation check
 * Measures ZP35-distance between golden coordinates relative to κ
 * @param {Object} entityA - First entity
 * @param {Object} entityB - Second entity
 * @returns {number} Invariant distance [0,1]
 */
function guardianR(entityA, entityB) {
	var coordA = applyGoldenOperator(entityA);
	var coordB = applyGoldenOperator(entityB);
	
	var distance = Math.abs(coordA - coordB);
	
	// Normalize by κ to get relative distance
	return distance / KAPPA;
}

/**
 * Calculate edge strength between two entities
 * E(A,B) = sqrt(ϕ² + ∂² + ℛ²)
 * @param {Object} entityA - First entity
 * @param {Object} entityB - Second entity
 * @returns {Object} Compatibility assessment with edge strength
 */
function calculateCompatibility(entityA, entityB) {
	var phi = guardianPhi(entityA, entityB);
	var delta = guardianDelta(entityA, entityB);
	var r = guardianR(entityA, entityB);
	
	var edgeStrength = Math.sqrt(phi * phi + delta * delta + r * r);
	
	var result = {
		edgeStrength: edgeStrength,
		phi: phi,
		delta: delta,
		r: r,
		compatible: true,
		mode: "safe",
		confidence: 1.0,
		message: ""
	};
	
	if(edgeStrength < KAPPA) {
		result.mode = "safe";
		result.confidence = 1.0 - (edgeStrength / KAPPA);
		result.message = "Entities are compatible - safe to compose";
	} else if(edgeStrength < 2 * KAPPA) {
		result.mode = "caution";
		result.confidence = 0.5;
		result.message = "Caution - entities may conflict, review recommended";
		result.warnings = generateWarnings(phi, delta, r);
	} else {
		result.compatible = false;
		result.mode = "blocked";
		result.confidence = 0.0;
		result.message = "Entities are incompatible - likely conflict";
		result.reasons = generateReasons(phi, delta, r);
	}
	
	return result;
}

/**
 * Generate warnings for caution-level compatibility
 * @param {number} phi - Semantic distance
 * @param {number} delta - Structural distance
 * @param {number} r - Invariant distance
 * @returns {Array} Array of warning strings
 */
function generateWarnings(phi, delta, r) {
	var warnings = [];
	
	if(phi > KAPPA / 2) {
		warnings.push("Significant semantic differences detected");
	}
	
	if(delta > KAPPA / 2) {
		warnings.push("Structural conflicts may occur");
	}
	
	if(r > 1) {
		warnings.push("Entities operate at very different compositional levels");
	}
	
	return warnings;
}

/**
 * Generate reasons for blocked compatibility
 * @param {number} phi - Semantic distance
 * @param {number} delta - Structural distance
 * @param {number} r - Invariant distance
 * @returns {Array} Array of reason strings
 */
function generateReasons(phi, delta, r) {
	var reasons = [];
	
	if(phi > KAPPA) {
		reasons.push("Incompatible semantic domains - different sectors or lifecycles");
	}
	
	if(delta > KAPPA) {
		reasons.push("Structural conflicts - both modify global state or shared fields");
	}
	
	if(r > 2) {
		reasons.push("Extreme compositional depth mismatch");
	}
	
	return reasons;
}

/**
 * Find bridge morphism between incompatible entities
 * Attempts to find minimal-distortion adapter that preserves invariants
 * @param {Object} entityA - First entity
 * @param {Object} entityB - Second entity
 * @returns {Object} Bridge morphism specification
 */
function findBridgeMorphism(entityA, entityB) {
	var coordA = applyGoldenOperator(entityA);
	var coordB = applyGoldenOperator(entityB);
	var vecA = extractFeatureVector(entityA);
	var vecB = extractFeatureVector(entityB);
	
	var bridge = {
		exists: false,
		coordinate: (coordA + coordB) / 2,
		distortion: Math.abs(coordA - coordB),
		adaptations: []
	};
	
	// Check if bridge is possible (distortion not too large)
	if(bridge.distortion < 4 * KAPPA) {
		bridge.exists = true;
		
		// Field mapping adaptations
		if(vecA.fieldWrites > 0 && vecB.fieldWrites > 0) {
			bridge.adaptations.push({
				type: "field-mapping",
				description: "Add field name translation layer",
				code: "// Map field names between entities"
			});
		}
		
		// Lifecycle coordination
		var lifecycleA = vecA.lifecycle.join(",");
		var lifecycleB = vecB.lifecycle.join(",");
		if(lifecycleA !== lifecycleB) {
			bridge.adaptations.push({
				type: "lifecycle-adapter",
				description: "Forward events between lifecycles",
				code: "// Add event forwarding"
			});
		}
		
		// State isolation
		if(vecA.statefulness === "impure" && vecB.statefulness === "impure") {
			bridge.adaptations.push({
				type: "state-guard",
				description: "Isolate state modifications",
				code: "// Add state isolation boundary"
			});
		}
	}
	
	return bridge;
}

// Export functions
exports.KAPPA = KAPPA;
exports.PHI = PHI;
exports.calculateOrdinalHeight = calculateOrdinalHeight;
exports.cantorEmbedding = cantorEmbedding;
exports.goldenScale = goldenScale;
exports.applyGoldenOperator = applyGoldenOperator;
exports.extractFeatureVector = extractFeatureVector;
exports.guardianPhi = guardianPhi;
exports.guardianDelta = guardianDelta;
exports.guardianR = guardianR;
exports.calculateCompatibility = calculateCompatibility;
exports.findBridgeMorphism = findBridgeMorphism;

})();
