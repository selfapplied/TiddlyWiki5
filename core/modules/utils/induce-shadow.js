/*\
title: $:/core/modules/utils/induce-shadow.js
type: application/javascript
module-type: utils

Shadow Induction Module for TiddlyWiki

This module implements shadow induction - the ability for any tiddler to generate
its own shadow compiler when no semantic neighbor exists. This is done through
extraction of the tiddler's intrinsic structure rather than through clustering
or search.

The process:
1. Compute ZP35 signature (geometric anchor)
2. Compute internal coherence (crisp vs chaotic separation)
3. Extract crisp structure (field schema, stable tokens, repeatable patterns)
4. Generate shadow compiler tiddler with crisp subset
5. Mark original as self-hosted program

This gives each tiddler:
- A personal interpreter
- A personal dialect
- A personal curvature bound
- A way to evolve safely
- A way to compose with others on equal footing

\*/

"use strict";

/*
Shadow Induction Constructor
@param {object} wiki - TiddlyWiki instance
@param {object} zp35Operator - ZP35 operator instance
*/
function ShadowInducer(wiki, zp35Operator) {
	this.wiki = wiki;
	this.zp35 = zp35Operator;
	
	// Guardian threshold - inherited from ZP35
	// κ = 0.35 is the coherence curvature for composition safety
	// Derived from empirical learnability limits (~400 examples/transition)
	// See ZP35_GOLDEN_OPERATOR.md for mathematical foundations
	this.kappa = 0.35;
	
	// Coherence thresholds for crisp/chaotic separation
	this.CRISP_THRESHOLD = 0.65;      // Above this = crisp/stable
	this.CHAOTIC_THRESHOLD = 0.35;    // Below this = chaotic/noise
}

/*
Induce shadow compiler from a tiddler
This extracts the crisp core and generates a shadow compiler

@param {object} tiddler - Tiddler to induce shadow from
@returns {object} - Result with shadow compiler and modified tiddler
*/
ShadowInducer.prototype.induceShadow = function(tiddler) {
	if(!tiddler) {
		return {
			success: false,
			message: "Invalid tiddler for shadow induction"
		};
	}
	
	// Step 1: Compute ZP35 signature (geometric anchor)
	var signature = this.zp35.calculateSignature(tiddler);
	var coord = this.zp35.applyGoldenOperator(tiddler);
	var height = this.zp35.calculateOrdinalHeight(tiddler);
	
	// Step 2: Compute internal coherence (crisp vs chaotic separation)
	var coherenceAnalysis = this.analyzeInternalCoherence(tiddler);
	
	// Step 3: Extract crisp structure
	var crispStructure = this.extractCrispStructure(tiddler, coherenceAnalysis);
	
	// Step 4: Generate shadow compiler tiddler
	var shadowCompiler = this.generateShadowCompiler(tiddler, crispStructure, signature, coord, height);
	
	// Step 5: Mark original tiddler as self-hosted program
	var selfHostedProgram = this.markAsSelfHosted(tiddler, shadowCompiler);
	
	return {
		success: true,
		shadowCompiler: shadowCompiler,
		selfHostedProgram: selfHostedProgram,
		coherenceAnalysis: coherenceAnalysis,
		crispStructure: crispStructure,
		signature: signature,
		message: "Shadow compiler induced successfully"
	};
};

/*
Analyze internal coherence of a tiddler
Separates crisp (regularities) from chaotic (idiosyncrasies)

@param {object} tiddler - Tiddler to analyze
@returns {object} - Coherence analysis with crisp/chaotic separation
*/
ShadowInducer.prototype.analyzeInternalCoherence = function(tiddler) {
	var fields = tiddler.fields;
	var analysis = {
		crispFields: [],
		chaoticFields: [],
		patterns: [],
		curvatureCoefficient: 0
	};
	
	// Analyze each field for stability/coherence
	var fieldNames = Object.keys(fields);
	
	for(var i = 0; i < fieldNames.length; i++) {
		var fieldName = fieldNames[i];
		var fieldValue = fields[fieldName];
		var fieldCoherence = this.analyzeFieldCoherence(fieldName, fieldValue);
		
		if(fieldCoherence.score >= this.CRISP_THRESHOLD) {
			analysis.crispFields.push({
				name: fieldName,
				value: fieldValue,
				coherence: fieldCoherence
			});
		} else if(fieldCoherence.score <= this.CHAOTIC_THRESHOLD) {
			analysis.chaoticFields.push({
				name: fieldName,
				value: fieldValue,
				coherence: fieldCoherence
			});
		} else {
			// Intermediate - decision based on context
			if(this.isStructuralField(fieldName)) {
				analysis.crispFields.push({
					name: fieldName,
					value: fieldValue,
					coherence: fieldCoherence
				});
			} else {
				analysis.chaoticFields.push({
					name: fieldName,
					value: fieldValue,
					coherence: fieldCoherence
				});
			}
		}
	}
	
	// Extract patterns from text content
	if(fields.text) {
		analysis.patterns = this.extractPatterns(fields.text);
	}
	
	// Calculate curvature coefficient (semantic flexibility band)
	analysis.curvatureCoefficient = this.calculateCurvatureCoefficient(analysis);
	
	return analysis;
};

/*
Analyze coherence of a specific field
Determines if field is stable/crisp or chaotic/noisy

@param {string} fieldName - Name of the field
@param {*} fieldValue - Value of the field
@returns {object} - Field coherence assessment
*/
ShadowInducer.prototype.analyzeFieldCoherence = function(fieldName, fieldValue) {
	var score = 0.5; // Default intermediate
	var reasons = [];
	
	// Structural fields are inherently crisp
	if(this.isStructuralField(fieldName)) {
		score += 0.3;
		reasons.push("structural-field");
	}
	
	// Type indicators increase crispness
	if(fieldName === "type" || fieldName === "generator") {
		score += 0.2;
		reasons.push("type-indicator");
	}
	
	// Version fields are crisp
	if(fieldName === "version" || fieldName === "seed") {
		score += 0.2;
		reasons.push("versioned-field");
	}
	
	// Tags increase crispness (categorization)
	if(fieldName === "tags") {
		score += 0.15;
		reasons.push("categorization");
	}
	
	// Analyze value stability
	if(typeof fieldValue === "string") {
		// Short, stable strings are crisp
		if(fieldValue.length < 50) {
			score += 0.1;
			reasons.push("short-stable");
		}
		
		// Pattern-heavy strings are chaotic
		if(this.hasHighEntropy(fieldValue)) {
			score -= 0.2;
			reasons.push("high-entropy");
		}
	}
	
	// Normalize score to [0, 1]
	score = Math.max(0, Math.min(1, score));
	
	return {
		score: score,
		reasons: reasons
	};
};

/*
Check if field is structural (part of schema)
*/
ShadowInducer.prototype.isStructuralField = function(fieldName) {
	var structuralFields = [
		"title", "type", "generator", "version", "seed",
		"tags", "modifier", "modified", "creator", "created",
		"zp35", "regen-zip"
	];
	
	return structuralFields.indexOf(fieldName) !== -1;
};

/*
Check if string has high entropy (chaotic/random)
*/
ShadowInducer.prototype.hasHighEntropy = function(str) {
	if(!str || str.length < 10) {
		return false;
	}
	
	// Calculate character frequency distribution
	var freq = {};
	for(var i = 0; i < str.length; i++) {
		var char = str[i];
		freq[char] = (freq[char] || 0) + 1;
	}
	
	// Calculate Shannon entropy
	var entropy = 0;
	var len = str.length;
	var chars = Object.keys(freq);
	var log2 = Math.log(2);
	
	for(var j = 0; j < chars.length; j++) {
		var p = freq[chars[j]] / len;
		// Use natural log and convert to log2
		entropy -= p * (Math.log(p) / log2);
	}
	
	// High entropy threshold (close to maximum for ASCII)
	return entropy > 4.5;
};

/*
Extract repeatable patterns from text
*/
ShadowInducer.prototype.extractPatterns = function(text) {
	var patterns = [];
	
	if(!text || text.length === 0) {
		return patterns;
	}
	
	// Extract markdown-style patterns
	var markdownPatterns = [
		{ pattern: /^#{1,6}\s+/gm, type: "heading" },
		{ pattern: /\*\*[^*]+\*\*/g, type: "bold" },
		{ pattern: /\*[^*]+\*/g, type: "italic" },
		{ pattern: /`[^`]+`/g, type: "code" },
		{ pattern: /\[\[([^\]]+)\]\]/g, type: "link" },
		{ pattern: /{{([^}]+)}}/g, type: "transclusion" }
	];
	
	for(var i = 0; i < markdownPatterns.length; i++) {
		var mp = markdownPatterns[i];
		var matches = text.match(mp.pattern);
		if(matches && matches.length > 0) {
			patterns.push({
				type: mp.type,
				count: matches.length,
				examples: matches.slice(0, 3) // Keep first 3 examples
			});
		}
	}
	
	return patterns;
};

/*
Calculate curvature coefficient (semantic flexibility band)
This determines how much deviation the tiddler tolerates before losing identity

@param {object} analysis - Coherence analysis
@returns {number} - Curvature coefficient in [0, 1]
*/
ShadowInducer.prototype.calculateCurvatureCoefficient = function(analysis) {
	var crispRatio = analysis.crispFields.length / 
		(analysis.crispFields.length + analysis.chaoticFields.length);
	
	// More crisp fields = lower curvature (more rigid structure)
	// More chaotic fields = higher curvature (more flexible)
	var curvature = 1.0 - crispRatio;
	
	// Clamp to safe range around kappa
	curvature = Math.max(this.kappa * 0.5, Math.min(this.kappa * 2, curvature));
	
	return curvature;
};

/*
Extract crisp structure from tiddler
Returns fields and patterns that form the stable core

@param {object} tiddler - Tiddler to extract from
@param {object} coherenceAnalysis - Coherence analysis result
@returns {object} - Crisp structure
*/
ShadowInducer.prototype.extractCrispStructure = function(tiddler, coherenceAnalysis) {
	var structure = {
		schema: {},
		stableTokens: [],
		patterns: coherenceAnalysis.patterns
	};
	
	// Extract crisp fields into schema
	for(var i = 0; i < coherenceAnalysis.crispFields.length; i++) {
		var crispField = coherenceAnalysis.crispFields[i];
		structure.schema[crispField.name] = crispField.value;
	}
	
	// Extract stable tokens from patterns
	for(var j = 0; j < coherenceAnalysis.patterns.length; j++) {
		var pattern = coherenceAnalysis.patterns[j];
		structure.stableTokens.push({
			type: pattern.type,
			count: pattern.count
		});
	}
	
	return structure;
};

/*
Generate shadow compiler tiddler from crisp structure

@param {object} tiddler - Original tiddler
@param {object} crispStructure - Extracted crisp structure
@param {string} signature - ZP35 signature
@param {number} coord - Fractal coordinate
@param {number} height - Ordinal height
@returns {object} - Shadow compiler tiddler
*/
ShadowInducer.prototype.generateShadowCompiler = function(tiddler, crispStructure, signature, coord, height) {
	var originalTitle = tiddler.fields.title;
	var shadowTitle = originalTitle + "-shadow";
	
	var shadowFields = {
		title: shadowTitle,
		type: crispStructure.schema.type || "application/x-tiddler-shadow-compiler",
		generator: crispStructure.schema.generator || "shadow-compiler",
		version: "1.0.0",
		zp35: signature,
		seed: crispStructure.schema.seed || this.generateSeed(originalTitle),
		"shadow-source": originalTitle,
		"shadow-type": "induced",
		tags: ["$:/tags/ShadowCompiler"]
	};
	
	// Copy other crisp fields
	var schemaKeys = Object.keys(crispStructure.schema);
	for(var i = 0; i < schemaKeys.length; i++) {
		var key = schemaKeys[i];
		if(!shadowFields[key] && key !== "title" && key !== "text") {
			shadowFields[key] = crispStructure.schema[key];
		}
	}
	
	// Generate schema description in text field
	shadowFields.text = this.generateShadowCompilerText(tiddler, crispStructure);
	
	return {
		fields: shadowFields
	};
};

/*
Generate text content for shadow compiler
Describes the extracted structure and patterns

@param {object} tiddler - Original tiddler
@param {object} crispStructure - Extracted crisp structure
@returns {string} - Shadow compiler text
*/
ShadowInducer.prototype.generateShadowCompilerText = function(tiddler, crispStructure) {
	var lines = [];
	
	lines.push("!! Shadow Compiler");
	lines.push("");
	lines.push("This shadow compiler was auto-generated through induction from:");
	lines.push("`" + tiddler.fields.title + "`");
	lines.push("");
	
	lines.push("!!! Extracted Schema");
	lines.push("");
	var schemaKeys = Object.keys(crispStructure.schema);
	for(var i = 0; i < schemaKeys.length; i++) {
		var key = schemaKeys[i];
		lines.push("* `" + key + "`: " + this.formatFieldValue(crispStructure.schema[key]));
	}
	lines.push("");
	
	if(crispStructure.patterns.length > 0) {
		lines.push("!!! Detected Patterns");
		lines.push("");
		for(var j = 0; j < crispStructure.patterns.length; j++) {
			var pattern = crispStructure.patterns[j];
			lines.push("* " + pattern.type + " (count: " + pattern.count + ")");
		}
		lines.push("");
	}
	
	lines.push("!!! Usage");
	lines.push("");
	lines.push("This compiler defines the semantic kernel for interpreting the original tiddler.");
	lines.push("The original tiddler is now a program written in its own induced language.");
	
	return lines.join("\n");
};

/*
Format field value for display
*/
ShadowInducer.prototype.formatFieldValue = function(value) {
	if(typeof value === "string") {
		if(value.length > 50) {
			return value.substring(0, 47) + "...";
		}
		return value;
	} else if(Array.isArray(value)) {
		return "[" + value.join(", ") + "]";
	} else {
		return String(value);
	}
};

/*
Mark original tiddler as self-hosted program

@param {object} tiddler - Original tiddler
@param {object} shadowCompiler - Generated shadow compiler
@returns {object} - Modified tiddler fields
*/
ShadowInducer.prototype.markAsSelfHosted = function(tiddler, shadowCompiler) {
	// Shallow copy of fields - nested objects would be shared references
	// This is acceptable since we only modify top-level field values
	var modifiedFields = Object.assign({}, tiddler.fields);
	
	// Add compiler reference
	modifiedFields.compiler = shadowCompiler.fields.title;
	modifiedFields["program-mode"] = "self-hosted";
	modifiedFields["shadow-compiler"] = shadowCompiler.fields.title;
	
	// Add tag to indicate self-hosted status
	var tags = modifiedFields.tags || [];
	if(typeof tags === "string") {
		tags = tags.split(/\s+/).filter(function(t) { return t.length > 0; });
	}
	if(tags.indexOf("$:/tags/SelfHostedProgram") === -1) {
		tags.push("$:/tags/SelfHostedProgram");
	}
	modifiedFields.tags = tags;
	
	return {
		fields: modifiedFields
	};
};

/*
Generate seed for shadow compiler
Based on original tiddler title

@param {string} title - Original tiddler title
@returns {string} - Generated seed
*/
ShadowInducer.prototype.generateSeed = function(title) {
	// Simple hash-based seed generation
	var hash = 0;
	for(var i = 0; i < title.length; i++) {
		var char = title.charCodeAt(i);
		hash = ((hash << 5) - hash) + char;
		hash = hash | 0; // Convert to 32-bit integer
	}
	
	// Return hex representation with prefix
	return "shadow-" + Math.abs(hash).toString(16);
};

/*
Check if tiddler needs shadow induction
A tiddler needs shadow induction if it:
- Has no existing compiler reference
- Has enough structure to extract
- Is not already a compiler itself

@param {object} tiddler - Tiddler to check
@returns {boolean} - True if needs shadow induction
*/
ShadowInducer.prototype.needsShadowInduction = function(tiddler) {
	if(!tiddler || !tiddler.fields) {
		return false;
	}
	
	var fields = tiddler.fields;
	
	// Already has a compiler reference
	if(fields.compiler || fields["shadow-compiler"]) {
		return false;
	}
	
	// Is already a compiler or shadow
	if(fields.generator || fields["shadow-type"]) {
		return false;
	}
	
	// System tiddlers generally don't need shadow induction
	if(fields.title && fields.title.startsWith("$:/")) {
		return false;
	}
	
	// Needs at least some fields to extract structure from
	var fieldCount = Object.keys(fields).length;
	if(fieldCount < 2) {
		return false;
	}
	
	return true;
};

/*
Export constructor
*/
exports.ShadowInducer = ShadowInducer;
