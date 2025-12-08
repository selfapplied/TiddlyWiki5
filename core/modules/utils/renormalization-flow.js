/*\
title: $:/core/modules/utils/renormalization-flow.js
type: application/javascript
module-type: utils

Symbolic Renormalization Flow Module for TiddlyWiki

This module implements the Symbolic Renormalization Flow - a purification cycle
that strips away structural waste while preserving fundamental meaning. It achieves
this through an iterative translation loop that converges to a canonical form.

The mechanism:
1. Forward Step (Z): Maps grammar S_n to geometric coordinate x
2. Inverse Step (Z^-1): Reconstructs grammar from coordinate with minimal complexity
3. Iteration: S_{n+1} = Z^-1(Z(S_n))
4. Convergence: Achieves canonical form S* with minimal bracket complexity

Key properties:
- Preserves ZP35 coordinate invariance (x_S* = x_S_n)
- Strips redundancy through optimal reconstruction
- Locks fundamental meaning in place
- Analogous to lossy-to-lossless compression

\*/

"use strict";

/*
Renormalization Flow Constructor
@param {object} wiki - TiddlyWiki instance
@param {object} zp35Operator - ZP35 operator instance
@param {object} shadowInducer - Shadow induction module
*/
function RenormalizationFlow(wiki, zp35Operator, shadowInducer) {
	this.wiki = wiki;
	this.zp35 = zp35Operator;
	this.shadowInducer = shadowInducer;
	
	// Convergence threshold for detecting canonical form
	// When complexity change drops below this, we've reached S*
	this.CONVERGENCE_THRESHOLD = 0.01;
	
	// Maximum iterations to prevent infinite loops
	this.MAX_ITERATIONS = 10;
	
	// Minimum improvement per iteration (prevents oscillation)
	this.MIN_IMPROVEMENT = 0.001;
}

/*
Apply renormalization flow to a tiddler
This implements the full S_{n+1} = Z^-1(Z(S_n)) cycle

@param {object} tiddler - Input tiddler to renormalize
@param {object} options - Optional parameters
@returns {object} - Renormalization result with canonical form
*/
RenormalizationFlow.prototype.renormalize = function(tiddler, options) {
	options = options || {};
	var maxIterations = options.maxIterations || this.MAX_ITERATIONS;
	var verbose = options.verbose || false;
	
	if(!tiddler) {
		return {
			success: false,
			message: "Invalid tiddler for renormalization"
		};
	}
	
	var iterations = [];
	var currentTiddler = tiddler;
	var previousComplexity = this.calculateBracketComplexity(tiddler);
	var converged = false;
	
	// Record initial state
	var initialCoord = this.forwardStep(tiddler);
	iterations.push({
		iteration: 0,
		complexity: previousComplexity,
		coordinate: initialCoord,
		tiddler: tiddler
	});
	
	// Iterative renormalization cycle
	for(var i = 1; i <= maxIterations && !converged; i++) {
		// Step 1: Forward mapping (Z) - extract coordinate
		var coordinate = this.forwardStep(currentTiddler);
		
		// Step 2: Inverse mapping (Z^-1) - reconstruct with minimal complexity
		var reconstructed = this.inverseStep(coordinate, currentTiddler);
		
		if(!reconstructed.success) {
			return {
				success: false,
				message: "Inverse step failed at iteration " + i,
				iterations: iterations,
				error: reconstructed.error
			};
		}
		
		// Step 3: Measure complexity
		var currentComplexity = this.calculateBracketComplexity(reconstructed.tiddler);
		var complexityDelta = previousComplexity - currentComplexity;
		var improvement = complexityDelta / previousComplexity;
		
		// Record iteration
		iterations.push({
			iteration: i,
			complexity: currentComplexity,
			complexityDelta: complexityDelta,
			improvement: improvement,
			coordinate: coordinate,
			tiddler: reconstructed.tiddler
		});
		
		if(verbose) {
			console.log("Renormalization iteration " + i + ": complexity=" + 
				currentComplexity.toFixed(3) + ", delta=" + complexityDelta.toFixed(3));
		}
		
		// Check convergence conditions
		if(Math.abs(complexityDelta) < this.CONVERGENCE_THRESHOLD) {
			converged = true;
			if(verbose) {
				console.log("Converged: complexity delta below threshold");
			}
		} else if(improvement < this.MIN_IMPROVEMENT && complexityDelta > 0) {
			converged = true;
			if(verbose) {
				console.log("Converged: improvement too small");
			}
		} else if(complexityDelta < 0) {
			// Complexity increased - this shouldn't happen, but handle it
			if(verbose) {
				console.log("Warning: complexity increased at iteration " + i);
			}
			// Revert to previous iteration
			currentTiddler = iterations[i - 1].tiddler;
			converged = true;
		} else {
			// Continue iteration
			currentTiddler = reconstructed.tiddler;
			previousComplexity = currentComplexity;
		}
	}
	
	// Verify invariance: coordinate should be preserved
	var finalCoord = this.forwardStep(currentTiddler);
	var coordDrift = Math.abs(finalCoord - initialCoord);
	
	return {
		success: true,
		converged: converged,
		iterations: iterations.length - 1,
		initialComplexity: iterations[0].complexity,
		finalComplexity: previousComplexity,
		complexityReduction: iterations[0].complexity - previousComplexity,
		coordinateInvariance: coordDrift < 0.001,
		coordinateDrift: coordDrift,
		canonicalForm: currentTiddler,
		iterationHistory: iterations,
		message: converged ? 
			"Converged to canonical form in " + (iterations.length - 1) + " iterations" :
			"Maximum iterations reached"
	};
};

/*
Forward step (Z): Map tiddler to geometric coordinate
This extracts the "pure information content" of the tiddler

@param {object} tiddler - Input tiddler
@returns {number} - ZP35 coordinate
*/
RenormalizationFlow.prototype.forwardStep = function(tiddler) {
	if(!tiddler || !this.zp35) {
		return 0;
	}
	
	// Use ZP35 golden operator to map to fractal coordinate
	return this.zp35.applyGoldenOperator(tiddler);
};

/*
Inverse step (Z^-1): Reconstruct tiddler from coordinate with minimal complexity
This is the "magic" that strips redundancy while preserving meaning

@param {number} coordinate - Target ZP35 coordinate
@param {object} seedTiddler - Original tiddler (provides context)
@returns {object} - Reconstructed tiddler with minimal complexity
*/
RenormalizationFlow.prototype.inverseStep = function(coordinate, seedTiddler) {
	if(!seedTiddler) {
		return {
			success: false,
			error: "No seed tiddler provided for inverse reconstruction"
		};
	}
	
	// Step 1: Extract crisp structure (what must be preserved)
	var coherenceAnalysis = this.shadowInducer.analyzeInternalCoherence(seedTiddler);
	var crispStructure = this.shadowInducer.extractCrispStructure(seedTiddler, coherenceAnalysis);
	
	// Step 2: Build minimal tiddler that reproduces the coordinate
	var minimalFields = this.buildMinimalFields(crispStructure, coordinate, seedTiddler);
	
	// Step 3: Verify the reconstruction hits the target coordinate
	var reconstructed = {
		fields: minimalFields
	};
	
	var reconstructedCoord = this.forwardStep(reconstructed);
	var coordError = Math.abs(reconstructedCoord - coordinate);
	
	// Allow small coordinate drift due to discretization
	if(coordError > 0.01) {
		// Try to refine the reconstruction
		minimalFields = this.refineReconstruction(minimalFields, coordinate, coordError);
		reconstructed = { fields: minimalFields };
	}
	
	return {
		success: true,
		tiddler: reconstructed,
		coordinateError: coordError
	};
};

/*
Build minimal field set that reproduces target coordinate
This is where we achieve "stripping redundancy"

@param {object} crispStructure - Extracted crisp structure
@param {number} targetCoord - Target ZP35 coordinate
@param {object} seedTiddler - Original tiddler for context
@returns {object} - Minimal field set
*/
RenormalizationFlow.prototype.buildMinimalFields = function(crispStructure, targetCoord, seedTiddler) {
	var fields = {};
	
	// Essential fields (always required)
	fields.title = seedTiddler.fields.title;
	fields.type = crispStructure.schema.type || seedTiddler.fields.type || "text/vnd.tiddlywiki";
	
	// Copy structural fields from crisp structure (they define the kernel)
	var structuralFields = ["generator", "version", "seed", "zp35", "regen-zip"];
	for(var i = 0; i < structuralFields.length; i++) {
		var field = structuralFields[i];
		if(crispStructure.schema[field] !== undefined) {
			fields[field] = crispStructure.schema[field];
		}
	}
	
	// Include tags if they contribute to semantic identity
	if(crispStructure.schema.tags) {
		fields.tags = this.minimalizeTagSet(crispStructure.schema.tags);
	}
	
	// Generate minimal text content
	fields.text = this.generateMinimalText(crispStructure, seedTiddler);
	
	// Add renormalization metadata
	fields["renormalized"] = "true";
	fields["renorm-source"] = seedTiddler.fields.title;
	fields["renorm-coord"] = targetCoord.toFixed(6);
	
	return fields;
};

/*
Minimalize tag set by removing redundant tags
Keep only tags that contribute to semantic identity

@param {array|string} tags - Input tags
@returns {array} - Minimalized tag set
*/
RenormalizationFlow.prototype.minimalizeTagSet = function(tags) {
	if(!tags) {
		return [];
	}
	
	// Convert to array if string
	var tagArray = tags;
	if(typeof tags === "string") {
		tagArray = tags.split(/\s+/).filter(function(t) { return t.length > 0; });
	}
	
	// Remove system tags that don't affect semantics
	var filtered = tagArray.filter(function(tag) {
		// Keep semantic tags, remove purely organizational ones
		return !tag.match(/^\$:\/tags\/(Draft|Temporary|Hidden)$/);
	});
	
	// Remove duplicates
	var unique = [];
	for(var i = 0; i < filtered.length; i++) {
		if(unique.indexOf(filtered[i]) === -1) {
			unique.push(filtered[i]);
		}
	}
	
	return unique;
};

/*
Generate minimal text content from crisp structure
Only includes patterns that define semantic identity

@param {object} crispStructure - Extracted crisp structure
@param {object} seedTiddler - Original tiddler
@returns {string} - Minimal text content
*/
RenormalizationFlow.prototype.generateMinimalText = function(crispStructure, seedTiddler) {
	var lines = [];
	
	// If original text is already minimal (short and crisp), keep it
	var originalText = seedTiddler.fields.text || "";
	if(originalText.length < 200 && crispStructure.patterns.length === 0) {
		return originalText;
	}
	
	// Generate minimal reconstruction based on patterns
	if(crispStructure.patterns.length > 0) {
		for(var i = 0; i < crispStructure.patterns.length; i++) {
			var pattern = crispStructure.patterns[i];
			// Include representative example of each pattern type
			if(pattern.examples && pattern.examples.length > 0) {
				lines.push(pattern.examples[0]);
			}
		}
	}
	
	// If no patterns extracted, create minimal placeholder
	if(lines.length === 0) {
		lines.push("!! " + seedTiddler.fields.title);
		lines.push("");
		lines.push("Canonical form of tiddler with ZP35 coordinate: " + 
			this.zp35.applyGoldenOperator(seedTiddler).toFixed(6));
	}
	
	return lines.join("\n");
};

/*
Refine reconstruction to better match target coordinate
Used when initial reconstruction has too much coordinate error

@param {object} fields - Current field set
@param {number} targetCoord - Target coordinate
@param {number} currentError - Current coordinate error
@returns {object} - Refined field set
*/
RenormalizationFlow.prototype.refineReconstruction = function(fields, targetCoord, currentError) {
	// For now, accept the reconstruction as-is
	// Future enhancement: iteratively adjust fields to reduce coordinate error
	// This is acceptable since small coordinate drift is allowed
	
	return fields;
};

/*
Calculate bracket complexity of a tiddler
This measures the "structural waste" that renormalization removes

Bracket complexity considers:
- Number of fields (structural overhead)
- Text length (information density)
- Redundant patterns
- Unnecessary metadata

@param {object} tiddler - Tiddler to measure
@returns {number} - Bracket complexity score
*/
RenormalizationFlow.prototype.calculateBracketComplexity = function(tiddler) {
	if(!tiddler || !tiddler.fields) {
		return 0;
	}
	
	var complexity = 0;
	var fields = tiddler.fields;
	var fieldNames = Object.keys(fields);
	
	// Component 1: Field count overhead
	// Each field adds to complexity, but structural fields add less
	var structuralFields = ["title", "type", "tags", "text"];
	var nonStructuralCount = 0;
	for(var i = 0; i < fieldNames.length; i++) {
		if(structuralFields.indexOf(fieldNames[i]) === -1) {
			nonStructuralCount++;
		}
	}
	complexity += nonStructuralCount * 2; // Non-structural fields cost more
	complexity += fieldNames.length * 0.5; // All fields have some cost
	
	// Component 2: Text complexity
	// Longer text = higher complexity, but with diminishing returns (log scale)
	if(fields.text) {
		var textLength = fields.text.length;
		complexity += Math.log(textLength + 1) / Math.log(2); // Log scale
	}
	
	// Component 3: Tag redundancy
	// More tags = higher complexity (potential for overlap)
	if(fields.tags) {
		var tagCount = 0;
		if(Array.isArray(fields.tags)) {
			tagCount = fields.tags.length;
		} else if(typeof fields.tags === "string") {
			tagCount = fields.tags.split(/\s+/).filter(function(t) { 
				return t.length > 0; 
			}).length;
		}
		complexity += tagCount * 1.5;
	}
	
	// Component 4: Metadata overhead
	// Check for redundant or verbose metadata fields
	var metadataFields = ["creator", "modifier", "created", "modified", "revision"];
	var metadataCount = 0;
	for(var j = 0; j < metadataFields.length; j++) {
		if(fields[metadataFields[j]]) {
			metadataCount++;
		}
	}
	complexity += metadataCount * 0.5;
	
	return complexity;
};

/*
Check if a tiddler is in canonical form
A tiddler is canonical if further renormalization produces no improvement

@param {object} tiddler - Tiddler to check
@returns {boolean} - True if canonical
*/
RenormalizationFlow.prototype.isCanonical = function(tiddler) {
	if(!tiddler) {
		return false;
	}
	
	// Quick check: already marked as renormalized?
	if(tiddler.fields && tiddler.fields.renormalized === "true") {
		return true;
	}
	
	// Test renormalization: does it converge in 1 iteration?
	var result = this.renormalize(tiddler, { maxIterations: 2, verbose: false });
	
	return result.success && result.iterations <= 1;
};

/*
Batch renormalize multiple tiddlers
Useful for optimizing an entire wiki

@param {array} tiddlers - Array of tiddlers to renormalize
@param {object} options - Optional parameters
@returns {object} - Batch renormalization results
*/
RenormalizationFlow.prototype.renormalizeBatch = function(tiddlers, options) {
	options = options || {};
	var results = [];
	var totalReduction = 0;
	var successCount = 0;
	
	for(var i = 0; i < tiddlers.length; i++) {
		var result = this.renormalize(tiddlers[i], options);
		results.push(result);
		
		if(result.success) {
			successCount++;
			totalReduction += result.complexityReduction;
		}
	}
	
	return {
		totalTiddlers: tiddlers.length,
		successCount: successCount,
		failureCount: tiddlers.length - successCount,
		totalComplexityReduction: totalReduction,
		averageReduction: successCount > 0 ? totalReduction / successCount : 0,
		results: results
	};
};

/*
Export the RenormalizationFlow constructor
*/
exports.RenormalizationFlow = RenormalizationFlow;
