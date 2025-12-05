/*\
title: $:/core/modules/utils/witness-fingerprint.js
type: application/javascript
module-type: utils

Witness Fingerprint System for Semantic Tiddler Analysis

Inspired by the CE Tower witness operator from antclock research:
https://github.com/selfapplied/antclock/blob/main/arXiv/working.md

The witness operator <>g extracts self-describing invariant signatures
as 4D fingerprints: (phase θ, depth l, sector s, monodromy m)

\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

/**
 * Calculate witness fingerprint for a tiddler
 * Returns a semantic signature capturing key structural properties
 */
exports.calculateWitnessFingerprint = function(tiddler, wiki) {
	if(!tiddler) {
		return null;
	}
	
	wiki = wiki || $tw.wiki;
	
	var fingerprint = {
		// Phase: Semantic direction based on text content analysis
		phase: calculateSemanticPhase(tiddler, wiki),
		
		// Depth: Hierarchical position in knowledge graph
		depth: calculateHierarchyDepth(tiddler, wiki),
		
		// Sector: Domain classification based on tags
		sector: identifySector(tiddler, wiki),
		
		// Monodromy: Cyclic reference patterns
		monodromy: calculateCyclicPatterns(tiddler, wiki),
		
		// Additional metrics
		linkDensity: calculateLinkDensity(tiddler, wiki),
		transclusionComplexity: calculateTransclusionComplexity(tiddler, wiki),
		fieldComplexity: calculateFieldComplexity(tiddler)
	};
	
	return fingerprint;
};

/**
 * Calculate semantic phase (0 to 2π) based on content analysis
 * Maps content to angular position in semantic space
 */
function calculateSemanticPhase(tiddler, wiki) {
	var text = tiddler.fields.text || "";
	var title = tiddler.fields.title || "";
	
	// Simple hash-based phase calculation
	// In production, would use more sophisticated NLP
	var hash = 0;
	var combined = title + text;
	
	for(var i = 0; i < combined.length; i++) {
		var char = combined.charCodeAt(i);
		hash = ((hash << 5) - hash) + char;
		hash = hash & hash; // Convert to 32-bit integer
	}
	
	// Normalize to [0, 2π]
	return (Math.abs(hash) % 628318) / 100000;
}

/**
 * Calculate hierarchy depth based on tag relationships
 */
function calculateHierarchyDepth(tiddler, wiki) {
	var tags = tiddler.fields.tags || [];
	if(tags.length === 0) {
		return 0;
	}
	
	var maxDepth = 0;
	
	// Calculate depth by following tag chains
	$tw.utils.each(tags, function(tag) {
		var tagDepth = getTagDepth(tag, wiki, 0, {});
		maxDepth = Math.max(maxDepth, tagDepth);
	});
	
	return maxDepth;
}

/**
 * Recursively calculate tag depth
 */
function getTagDepth(tag, wiki, currentDepth, visited) {
	// Prevent infinite loops
	if(visited[tag] || currentDepth > 20) {
		return currentDepth;
	}
	
	visited[tag] = true;
	
	var tagTiddler = wiki.getTiddler(tag);
	if(!tagTiddler) {
		return currentDepth;
	}
	
	var parentTags = tagTiddler.fields.tags || [];
	if(parentTags.length === 0) {
		return currentDepth;
	}
	
	var maxParentDepth = currentDepth;
	$tw.utils.each(parentTags, function(parentTag) {
		var parentDepth = getTagDepth(parentTag, wiki, currentDepth + 1, visited);
		maxParentDepth = Math.max(maxParentDepth, parentDepth);
	});
	
	return maxParentDepth;
}

/**
 * Identify sector (domain classification) based on tags
 * Returns integer sector ID
 */
function identifySector(tiddler, wiki) {
	var tags = tiddler.fields.tags || [];
	
	if(tags.length === 0) {
		return 0;
	}
	
	// Hash tags to consistent sector
	var sectorHash = 0;
	$tw.utils.each(tags, function(tag) {
		for(var i = 0; i < tag.length; i++) {
			sectorHash = ((sectorHash << 3) + sectorHash) + tag.charCodeAt(i);
		}
	});
	
	// Return sector in range [0, 99]
	return Math.abs(sectorHash) % 100;
}

/**
 * Calculate monodromy (cyclic reference patterns)
 * Measures how references loop back
 */
function calculateCyclicPatterns(tiddler, wiki) {
	var links = tiddler.fields.links || [];
	if(links.length === 0) {
		return 0;
	}
	
	// Check for bidirectional links (simplest cycle)
	var cycleCount = 0;
	var title = tiddler.fields.title;
	
	$tw.utils.each(links, function(link) {
		var linkedTiddler = wiki.getTiddler(link);
		if(linkedTiddler) {
			var linkedLinks = linkedTiddler.fields.links || [];
			if(linkedLinks.indexOf(title) !== -1) {
				cycleCount++;
			}
		}
	});
	
	// Normalize by total links
	return links.length > 0 ? cycleCount / links.length : 0;
}

/**
 * Calculate link density
 */
function calculateLinkDensity(tiddler, wiki) {
	var text = tiddler.fields.text || "";
	var links = tiddler.fields.links || [];
	
	if(text.length === 0) {
		return 0;
	}
	
	// Links per 1000 characters
	return (links.length / text.length) * 1000;
}

/**
 * Calculate transclusion complexity
 */
function calculateTransclusionComplexity(tiddler, wiki) {
	var text = tiddler.fields.text || "";
	
	// Count transclusion patterns
	var transclusionCount = (text.match(/\{\{[^}]+\}\}/g) || []).length;
	var macroCount = (text.match(/<<[^>]+>>/g) || []).length;
	
	return transclusionCount + macroCount;
}

/**
 * Calculate field complexity
 */
function calculateFieldComplexity(tiddler) {
	var fieldCount = Object.keys(tiddler.fields).length;
	
	// Standard fields don't count toward complexity
	var standardFields = ["title", "text", "created", "modified", "tags", "type"];
	var customFieldCount = 0;
	
	$tw.utils.each(tiddler.fields, function(value, key) {
		if(standardFields.indexOf(key) === -1) {
			customFieldCount++;
		}
	});
	
	return customFieldCount;
}

/**
 * Calculate fingerprint distance between two tiddlers
 * Returns normalized distance in [0, 1]
 */
exports.calculateFingerprintDistance = function(fp1, fp2) {
	if(!fp1 || !fp2) {
		return 1.0; // Maximum distance for invalid fingerprints
	}
	
	// Calculate component distances
	var phaseDist = circularDistance(fp1.phase, fp2.phase, Math.PI * 2);
	var depthDist = Math.abs(fp1.depth - fp2.depth) / 20; // Normalize by max depth
	var sectorDist = fp1.sector === fp2.sector ? 0 : 1;
	var monodromyDist = Math.abs(fp1.monodromy - fp2.monodromy);
	var linkDensityDist = Math.abs(fp1.linkDensity - fp2.linkDensity) / 100; // Normalize
	
	// Weighted Euclidean distance
	var distance = Math.sqrt(
		0.25 * phaseDist * phaseDist +
		0.20 * depthDist * depthDist +
		0.20 * sectorDist * sectorDist +
		0.15 * monodromyDist * monodromyDist +
		0.20 * linkDensityDist * linkDensityDist
	);
	
	// Normalize to [0, 1]
	return Math.min(distance, 1.0);
};

/**
 * Calculate circular distance (for phase)
 */
function circularDistance(a, b, period) {
	var diff = Math.abs(a - b);
	return Math.min(diff, period - diff) / period;
}

/**
 * Find similar tiddlers based on fingerprint distance
 */
exports.findSimilarTiddlers = function(targetTiddler, wiki, options) {
	options = options || {};
	var threshold = options.threshold || 0.3;
	var maxResults = options.maxResults || 10;
	
	wiki = wiki || $tw.wiki;
	
	var targetFingerprint = exports.calculateWitnessFingerprint(targetTiddler, wiki);
	if(!targetFingerprint) {
		return [];
	}
	
	var results = [];
	var allTiddlers = wiki.getTiddlers();
	
	$tw.utils.each(allTiddlers, function(title) {
		// Skip self
		if(title === targetTiddler.fields.title) {
			return;
		}
		
		var tiddler = wiki.getTiddler(title);
		if(!tiddler) {
			return;
		}
		
		var fingerprint = exports.calculateWitnessFingerprint(tiddler, wiki);
		if(!fingerprint) {
			return;
		}
		
		var distance = exports.calculateFingerprintDistance(targetFingerprint, fingerprint);
		
		if(distance < threshold) {
			results.push({
				title: title,
				distance: distance,
				similarity: 1 - distance, // Convert to similarity score
				fingerprint: fingerprint
			});
		}
	});
	
	// Sort by distance (ascending)
	results.sort(function(a, b) {
		return a.distance - b.distance;
	});
	
	// Limit results
	return results.slice(0, maxResults);
};

/**
 * Calculate semantic resonance between two tiddlers
 * Returns resonance coefficient [0, 1]
 */
exports.calculateResonance = function(tiddler1, tiddler2, wiki) {
	wiki = wiki || $tw.wiki;
	
	var fp1 = exports.calculateWitnessFingerprint(tiddler1, wiki);
	var fp2 = exports.calculateWitnessFingerprint(tiddler2, wiki);
	
	if(!fp1 || !fp2) {
		return 0;
	}
	
	// Resonance is inverse of distance
	var distance = exports.calculateFingerprintDistance(fp1, fp2);
	return 1 - distance;
};

})();
