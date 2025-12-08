/*\
title: $:/core/modules/utils/induce-shadow.js
type: application/javascript
module-type: utils

Shadow Induction - Bootstrap Compilers from Single Examples

This module implements "shadow induction" - the ability for a tiddler to analyze
its own structure and bootstrap a compiler from a single example. This is the
evolutionary step that transforms TiddlyWiki from a static semantic system into
a living, self-organizing computational substrate.

Key concepts:
- **Crisp fields**: High coherence (> 0.65), structural, stable
- **Chaotic fields**: Low coherence (< 0.35), high-entropy, variable
- **Curvature coefficient**: 1.0 - (crispFields/totalFields)
- **Shadow compiler**: Induced kernel that can process similar patterns

The process:
1. Analyze tiddler fields for coherence
2. Separate crisp (structural) from chaotic (content) components
3. Calculate curvature (how much chaos vs structure)
4. Extract kernel from crisp structure
5. Generate shadow compiler that can process similar tiddlers

This allows tiddlers to become self-hosting - they carry their own language,
compiler, kernel, invariants, and regeneration mechanism.

\*/

"use strict";

/*
Thresholds for field coherence classification
*/
var COHERENCE_THRESHOLDS = {
	CRISP: 0.65,      // Fields above this are structural/stable
	CHAOTIC: 0.35,    // Fields below this are high-entropy/variable
	KAPPA: 0.35       // Guardian threshold for safety
};

/*
Field types and their default coherence scores
These are heuristics based on how stable/structural each field type tends to be
*/
var FIELD_COHERENCE_SCORES = {
	// Structural fields (high coherence)
	"title": 0.95,
	"type": 0.90,
	"tags": 0.85,
	"created": 0.90,
	"modified": 0.60,  // Changes frequently
	"generator": 0.95,
	"version": 0.90,
	"bag": 0.85,
	"revision": 0.70,
	
	// Semantic fields (medium coherence)
	"caption": 0.70,
	"description": 0.65,
	"list": 0.60,
	"plugin-type": 0.85,
	"module-type": 0.85,
	
	// Content fields (low coherence, high entropy)
	"text": 0.30,
	"seed": 0.25,
	"params": 0.20,
	"regen-zip": 0.40,
	
	// ZP35 metadata (high coherence)
	"zp35": 0.95,
	"zp35-coord": 0.90,
	"zp35-height": 0.85,
	
	// Default for unknown fields
	"_default": 0.50
};

/*
Shadow Inducer Constructor
@param {object} wiki - TiddlyWiki instance
@param {object} zp35Operator - ZP35 operator instance
*/
function ShadowInducer(wiki, zp35Operator) {
	this.wiki = wiki;
	this.zp35 = zp35Operator;
	
	// Cache of induced shadow compilers
	this.shadowCompilers = {};
	
	// Statistics
	this.stats = {
		inductionCount: 0,
		successCount: 0,
		failureCount: 0
	};
}

/*
Analyze field coherence for a tiddler
Separates fields into crisp (structural) and chaotic (content) categories

@param {object} tiddler - Tiddler to analyze
@returns {object} - Analysis result with crisp/chaotic field lists
*/
ShadowInducer.prototype.analyzeFieldCoherence = function(tiddler) {
	if(!tiddler || !tiddler.fields) {
		return {
			crispFields: [],
			chaoticFields: [],
			intermediateFields: [],
			curvature: 0,
			totalFields: 0
		};
	}
	
	var fields = tiddler.fields;
	var fieldNames = Object.keys(fields);
	var crispFields = [];
	var chaoticFields = [];
	var intermediateFields = [];
	
	// Analyze each field
	for(var i = 0; i < fieldNames.length; i++) {
		var fieldName = fieldNames[i];
		var fieldValue = fields[fieldName];
		
		// Get coherence score for this field
		var coherence = this.calculateFieldCoherence(fieldName, fieldValue);
		
		// Classify field
		if(coherence >= COHERENCE_THRESHOLDS.CRISP) {
			crispFields.push({
				name: fieldName,
				value: fieldValue,
				coherence: coherence
			});
		} else if(coherence <= COHERENCE_THRESHOLDS.CHAOTIC) {
			chaoticFields.push({
				name: fieldName,
				value: fieldValue,
				coherence: coherence
			});
		} else {
			intermediateFields.push({
				name: fieldName,
				value: fieldValue,
				coherence: coherence
			});
		}
	}
	
	// Calculate curvature coefficient
	// Curvature = 1.0 - (crispFields / totalFields)
	// High curvature means more chaos (fewer crisp fields)
	// Low curvature means more structure (more crisp fields)
	var totalFields = fieldNames.length;
	var curvature = totalFields > 0 ? (1.0 - (crispFields.length / totalFields)) : 0;
	
	return {
		crispFields: crispFields,
		chaoticFields: chaoticFields,
		intermediateFields: intermediateFields,
		curvature: curvature,
		totalFields: totalFields,
		crispRatio: totalFields > 0 ? (crispFields.length / totalFields) : 0,
		chaoticRatio: totalFields > 0 ? (chaoticFields.length / totalFields) : 0
	};
};

/*
Calculate coherence score for a specific field
Combines base score with content analysis

@param {string} fieldName - Name of the field
@param {any} fieldValue - Value of the field
@returns {number} - Coherence score [0, 1]
*/
ShadowInducer.prototype.calculateFieldCoherence = function(fieldName, fieldValue) {
	// Get base score from field type
	var baseScore = FIELD_COHERENCE_SCORES[fieldName] || FIELD_COHERENCE_SCORES._default;
	
	// Adjust based on content characteristics
	var contentScore = this.analyzeContentCoherence(fieldValue);
	
	// Weighted combination: 70% base, 30% content
	var finalScore = (baseScore * 0.7) + (contentScore * 0.3);
	
	return Math.max(0, Math.min(1, finalScore));
};

/*
Analyze content coherence based on value characteristics
High coherence = structured, predictable, low entropy
Low coherence = random, high entropy, chaotic

@param {any} value - Field value to analyze
@returns {number} - Content coherence score [0, 1]
*/
ShadowInducer.prototype.analyzeContentCoherence = function(value) {
	if(value === null || value === undefined) {
		return 0.5; // Neutral
	}
	
	// Convert to string for analysis
	var str = String(value);
	var length = str.length;
	
	if(length === 0) {
		return 0.3; // Empty is somewhat chaotic
	}
	
	// Short values tend to be more structured
	if(length < 10) {
		return 0.8;
	}
	
	// Calculate entropy-like metric
	var uniqueChars = {};
	for(var i = 0; i < length; i++) {
		uniqueChars[str[i]] = true;
	}
	var uniqueCount = Object.keys(uniqueChars).length;
	
	// High unique/total ratio = high entropy = low coherence
	var entropyRatio = uniqueCount / length;
	
	// Invert: high entropy -> low coherence
	var coherence = 1.0 - entropyRatio;
	
	// Normalize to reasonable range
	coherence = 0.3 + (coherence * 0.4); // Range [0.3, 0.7]
	
	return coherence;
};

/*
Induce a shadow compiler from a single tiddler
Extracts the structural kernel and creates a compiler that can process
similar patterns

@param {object} tiddler - Source tiddler to induce from
@param {object} options - Induction options
@returns {object} - Induced shadow compiler tiddler
*/
ShadowInducer.prototype.induceShadowCompiler = function(tiddler, options) {
	options = options || {};
	
	this.stats.inductionCount++;
	
	try {
		// Analyze field coherence
		var analysis = this.analyzeFieldCoherence(tiddler);
		
		// Validate that we have enough structure to induce from
		if(analysis.crispFields.length === 0) {
			throw new Error("Cannot induce shadow compiler: no crisp fields found");
		}
		
		if(analysis.curvature > 0.85) {
			console.warn("High curvature (" + analysis.curvature.toFixed(3) + 
				") - induced compiler may be unstable");
		}
		
		// Extract kernel from crisp fields
		var kernel = this.extractKernel(analysis.crispFields);
		
		// Generate shadow compiler tiddler
		var shadowCompiler = this.generateShadowCompiler(tiddler, kernel, analysis, options);
		
		// Cache the shadow compiler
		var shadowId = this.getShadowId(tiddler);
		this.shadowCompilers[shadowId] = {
			compiler: shadowCompiler,
			source: tiddler.fields.title,
			analysis: analysis,
			timestamp: new Date().toISOString()
		};
		
		this.stats.successCount++;
		
		return {
			success: true,
			compiler: shadowCompiler,
			analysis: analysis,
			kernel: kernel,
			shadowId: shadowId
		};
		
	} catch(e) {
		this.stats.failureCount++;
		
		return {
			success: false,
			error: e.message,
			analysis: null
		};
	}
};

/*
Extract structural kernel from crisp fields
The kernel represents the invariant pattern that defines this type of tiddler

@param {array} crispFields - Array of crisp field objects
@returns {object} - Extracted kernel
*/
ShadowInducer.prototype.extractKernel = function(crispFields) {
	var kernel = {
		requiredFields: [],
		fieldTypes: {},
		structuralPattern: {}
	};
	
	// Extract required fields and their types
	for(var i = 0; i < crispFields.length; i++) {
		var field = crispFields[i];
		
		kernel.requiredFields.push(field.name);
		kernel.fieldTypes[field.name] = typeof field.value;
		
		// Store structural pattern (abstracted value)
		if(field.name === "type" || field.name === "plugin-type" || 
		   field.name === "module-type" || field.name === "generator") {
			// These define the semantic type - preserve exactly
			kernel.structuralPattern[field.name] = field.value;
		} else if(field.name === "tags") {
			// Tags define clustering - preserve for routing
			kernel.structuralPattern[field.name] = field.value;
		} else if(field.name === "version" || field.name === "zp35") {
			// Metadata fields - preserve for compatibility
			kernel.structuralPattern[field.name] = field.value;
		}
	}
	
	return kernel;
};

/*
Generate a shadow compiler tiddler from kernel and analysis
The shadow compiler can process tiddlers with similar structural patterns

@param {object} sourceTiddler - Original tiddler that was induced from
@param {object} kernel - Extracted structural kernel
@param {object} analysis - Field coherence analysis
@param {object} options - Generation options
@returns {object} - Shadow compiler tiddler
*/
ShadowInducer.prototype.generateShadowCompiler = function(sourceTiddler, kernel, analysis, options) {
	var sourceTitle = sourceTiddler.fields.title;
	var shadowTitle = options.shadowTitle || ("$:/shadow/compiler/" + this.sanitizeTitle(sourceTitle));
	
	// Build shadow compiler fields
	var compilerFields = {
		title: shadowTitle,
		tags: ["$:/tags/shadow-compiler", "compiler"],
		type: sourceTiddler.fields.type || "application/x-tiddler-compiler",
		"shadow-source": sourceTitle,
		"shadow-induced": new Date().toISOString(),
		"shadow-curvature": analysis.curvature.toFixed(4),
		"shadow-kernel": JSON.stringify(kernel),
		caption: "Shadow compiler induced from: " + sourceTitle,
		text: "This is a shadow compiler automatically induced from the structural " +
		      "analysis of tiddler '" + sourceTitle + "'.\n\n" +
		      "Curvature: " + analysis.curvature.toFixed(3) + "\n" +
		      "Crisp fields: " + analysis.crispFields.length + "\n" +
		      "Chaotic fields: " + analysis.chaoticFields.length + "\n\n" +
		      "This compiler can process tiddlers with similar structural patterns."
	};
	
	// Inherit generator if present in source
	if(sourceTiddler.fields.generator) {
		compilerFields.generator = sourceTiddler.fields.generator;
	}
	
	// Inherit regen-zip if present
	if(sourceTiddler.fields["regen-zip"]) {
		compilerFields["regen-zip"] = sourceTiddler.fields["regen-zip"];
	}
	
	// Calculate and set ZP35 coordinate for the shadow compiler
	// Shadow compilers should have high coherence (they're structural)
	if(this.zp35) {
		var coord = this.zp35.applyGoldenOperator(sourceTiddler);
		compilerFields["zp35-coord"] = coord.toFixed(6);
	}
	
	// Create tiddler object
	var shadowCompiler = {
		fields: compilerFields
	};
	
	return shadowCompiler;
};

/*
Get unique shadow ID for a tiddler
Used for caching and tracking induced shadows

@param {object} tiddler - Tiddler to get shadow ID for
@returns {string} - Shadow ID
*/
ShadowInducer.prototype.getShadowId = function(tiddler) {
	var title = tiddler.fields.title;
	var type = tiddler.fields.type || "default";
	
	// Simple hash-like ID
	return type + "::" + title;
};

/*
Sanitize title for use in shadow compiler naming
Removes special characters and spaces

@param {string} title - Original title
@returns {string} - Sanitized title
*/
ShadowInducer.prototype.sanitizeTitle = function(title) {
	return title.replace(/[^a-zA-Z0-9]/g, "_");
};

/*
Check if a shadow compiler exists for a given tiddler

@param {object} tiddler - Tiddler to check
@returns {boolean} - True if shadow exists
*/
ShadowInducer.prototype.hasShadowCompiler = function(tiddler) {
	var shadowId = this.getShadowId(tiddler);
	return !!this.shadowCompilers[shadowId];
};

/*
Get cached shadow compiler for a tiddler

@param {object} tiddler - Tiddler to get shadow for
@returns {object|null} - Shadow compiler entry or null
*/
ShadowInducer.prototype.getShadowCompiler = function(tiddler) {
	var shadowId = this.getShadowId(tiddler);
	return this.shadowCompilers[shadowId] || null;
};

/*
Get statistics about shadow induction

@returns {object} - Statistics object
*/
ShadowInducer.prototype.getStatistics = function() {
	return {
		inductionCount: this.stats.inductionCount,
		successCount: this.stats.successCount,
		failureCount: this.stats.failureCount,
		successRate: this.stats.inductionCount > 0 ? 
			(this.stats.successCount / this.stats.inductionCount) : 0,
		cachedShadows: Object.keys(this.shadowCompilers).length
	};
};

/*
Clear shadow compiler cache

@returns {void}
*/
ShadowInducer.prototype.clearCache = function() {
	this.shadowCompilers = {};
};

/*
Export the constructor
*/
exports.ShadowInducer = ShadowInducer;
exports.COHERENCE_THRESHOLDS = COHERENCE_THRESHOLDS;
exports.FIELD_COHERENCE_SCORES = FIELD_COHERENCE_SCORES;
