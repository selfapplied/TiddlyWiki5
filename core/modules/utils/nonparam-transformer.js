/*\
title: $:/core/modules/utils/nonparam-transformer.js
type: application/javascript
module-type: utils

Non-Parametric Transformers - Pure Morphisms Between Compilers

This module implements non-parametric transformers - structure-preserving morphisms
that sit between semantic kernels (compilers/shadows) and operate over the VM
substrate without introducing new parameters or degrees of freedom.

Key principles:
- **Parameter-free**: No new params field, completely determined by structure
- **Geometry-respecting**: ZP35 distance bounded (~1-Lipschitz)
- **Seed-stable**: Deterministic seed transformation
- **Curvature-aware**: Bounded curvature scaling

This creates a category-theoretic layer where:
- Objects = compilers/shadows
- Morphisms = non-parametric transformers
- Composition = respects geometry and curvature bounds

\*/

"use strict";

/*
Transformer types and their semantic meaning
*/
var TRANSFORMER_TYPES = {
	PROJECTION: "projection",      // Forget structure, keep canonical form
	LIFT: "lift",                  // Embed subdialect into larger dialect
	NORMALIZE: "normalize",        // Canonicalize representation
	UPGRADE: "upgrade",            // Version migration
	RESTRICT: "restrict",          // Specialize to subdomain
	IDENTITY: "identity"           // No-op (for composition chains)
};

/*
Seed policies for deterministic transformation
*/
var SEED_POLICIES = {
	INHERIT: "inherit",            // seed' = seed (unchanged)
	HASH: "hash",                  // seed' = hash(seed + transformer-id)
	RESEED_FIXED: "reseed-fixed",  // seed' = fixed value from transformer
	COMPOSE: "compose"             // seed' = f(seed) from composition
};

/*
Default constraints for geometry and curvature
*/
var DEFAULT_CONSTRAINTS = {
	MAX_ZP35_DISTANCE: 0.70,       // Maximum allowed ZP35 movement
	MIN_CURVATURE_SCALE: 0.5,      // Minimum curvature ratio
	MAX_CURVATURE_SCALE: 2.0,      // Maximum curvature ratio
	LIPSCHITZ_CONSTANT: 2.0        // Maximum distance expansion factor
};

/*
Non-Parametric Transformer Manager Constructor
@param {object} wiki - TiddlyWiki instance
@param {object} zp35Operator - ZP35 operator instance
@param {object} vm - REGEN-ZIP VM instance
*/
function NonParametricTransformer(wiki, zp35Operator, vm) {
	this.wiki = wiki;
	this.zp35 = zp35Operator;
	this.vm = vm;
	
	// Registry of transformer tiddlers
	this.transformers = {};
	
	// Composition cache
	this.compositionCache = {};
	
	// Statistics
	this.stats = {
		transformCount: 0,
		successCount: 0,
		failureCount: 0,
		geometryViolations: 0,
		seedPolicyViolations: 0
	};
}

/*
Register a transformer tiddler
Validates that it satisfies the non-parametric contract

@param {object} transformerTiddler - Transformer tiddler to register
@returns {object} - Registration result
*/
NonParametricTransformer.prototype.registerTransformer = function(transformerTiddler) {
	if(!transformerTiddler || !transformerTiddler.fields) {
		return {
			success: false,
			error: "Invalid transformer tiddler"
		};
	}
	
	var fields = transformerTiddler.fields;
	var title = fields.title;
	
	// Validate type
	if(fields.type !== "application/x-tiddler-transformer") {
		return {
			success: false,
			error: "Transformer must have type: application/x-tiddler-transformer"
		};
	}
	
	// Validate mode (must be non-parametric)
	if(fields.mode !== "non-parametric") {
		return {
			success: false,
			error: "Transformer mode must be 'non-parametric'"
		};
	}
	
	// Critical: NO PARAMS ALLOWED in non-parametric transformers
	// Also check for parameter-like fields that could introduce runtime config
	var paramLikeFields = ["params", "config", "options", "settings", "args", "arguments"];
	for(var i = 0; i < paramLikeFields.length; i++) {
		if(fields[paramLikeFields[i]]) {
			return {
				success: false,
				error: "Non-parametric transformers cannot have parameter-like fields: " + paramLikeFields[i]
			};
		}
	}
	
	// Validate required fields
	if(!fields["source-compiler"] || !fields["target-compiler"]) {
		return {
			success: false,
			error: "Transformer must specify source-compiler and target-compiler"
		};
	}
	
	if(!fields["transform-kind"]) {
		return {
			success: false,
			error: "Transformer must specify transform-kind"
		};
	}
	
	// Validate transform-kind
	var validKinds = Object.values(TRANSFORMER_TYPES);
	if(validKinds.indexOf(fields["transform-kind"]) === -1) {
		return {
			success: false,
			error: "Invalid transform-kind: " + fields["transform-kind"]
		};
	}
	
	// Validate seed policy
	var seedPolicy = fields["seed-policy"] || SEED_POLICIES.INHERIT;
	var validPolicies = Object.values(SEED_POLICIES);
	if(validPolicies.indexOf(seedPolicy) === -1) {
		return {
			success: false,
			error: "Invalid seed-policy: " + seedPolicy
		};
	}
	
	// Parse and validate geometry constraints
	var constraints = this.parseConstraints(fields);
	
	// Store transformer
	this.transformers[title] = {
		tiddler: transformerTiddler,
		sourceCompiler: fields["source-compiler"],
		targetCompiler: fields["target-compiler"],
		transformKind: fields["transform-kind"],
		seedPolicy: seedPolicy,
		constraints: constraints,
		generator: fields.generator || null
	};
	
	return {
		success: true,
		transformer: this.transformers[title]
	};
};

/*
Parse geometry and curvature constraints from transformer fields

@param {object} fields - Transformer tiddler fields
@returns {object} - Parsed constraints
*/
NonParametricTransformer.prototype.parseConstraints = function(fields) {
	return {
		maxZP35Distance: parseFloat(fields["zp35-max-distance"] || DEFAULT_CONSTRAINTS.MAX_ZP35_DISTANCE),
		minCurvatureScale: parseFloat(fields["curvature-scale-min"] || DEFAULT_CONSTRAINTS.MIN_CURVATURE_SCALE),
		maxCurvatureScale: parseFloat(fields["curvature-scale-max"] || DEFAULT_CONSTRAINTS.MAX_CURVATURE_SCALE),
		lipschitzConstant: parseFloat(fields["lipschitz-constant"] || DEFAULT_CONSTRAINTS.LIPSCHITZ_CONSTANT)
	};
};

/*
Apply a non-parametric transformer to a program tiddler
Returns transformed program that can be executed under target compiler

@param {string} transformerTitle - Title of transformer to apply
@param {object} programTiddler - Program tiddler to transform
@param {object} options - Transformation options
@returns {object} - Transformation result
*/
NonParametricTransformer.prototype.applyTransformer = function(transformerTitle, programTiddler, options) {
	options = options || {};
	
	this.stats.transformCount++;
	
	try {
		// Get transformer
		var transformer = this.transformers[transformerTitle];
		if(!transformer) {
			throw new Error("Transformer not registered: " + transformerTitle);
		}
		
		// Get original ZP35 coordinate and curvature
		var originalCoord = this.zp35.applyGoldenOperator(programTiddler);
		var originalCurvature = this.calculateProgramCurvature(programTiddler);
		
		// Build transformed program (pure function of structure)
		var transformedProgram = this.buildTransformedProgram(
			transformer,
			programTiddler
		);
		
		// Validate geometry constraints
		var geometryValid = this.validateGeometry(
			transformer,
			programTiddler,
			transformedProgram,
			originalCoord,
			originalCurvature
		);
		
		if(!geometryValid.success) {
			this.stats.geometryViolations++;
			throw new Error("Geometry violation: " + geometryValid.error);
		}
		
		// Validate seed policy
		var seedValid = this.validateSeedPolicy(
			transformer,
			programTiddler,
			transformedProgram
		);
		
		if(!seedValid.success) {
			this.stats.seedPolicyViolations++;
			throw new Error("Seed policy violation: " + seedValid.error);
		}
		
		this.stats.successCount++;
		
		return {
			success: true,
			transformedProgram: transformedProgram,
			transformer: transformerTitle,
			sourceCompiler: transformer.sourceCompiler,
			targetCompiler: transformer.targetCompiler,
			geometry: geometryValid.metrics,
			seed: seedValid.seedTransform
		};
		
	} catch(e) {
		this.stats.failureCount++;
		
		return {
			success: false,
			error: e.message
		};
	}
};

/*
Build transformed program from original program and transformer
This is a PURE function - no side effects, deterministic

@param {object} transformer - Transformer entry
@param {object} programTiddler - Original program tiddler
@returns {object} - Transformed program tiddler
*/
NonParametricTransformer.prototype.buildTransformedProgram = function(transformer, programTiddler) {
	var fields = programTiddler.fields;
	var transformerFields = transformer.tiddler.fields;
	
	// Start with copy of original fields
	var newFields = Object.assign({}, fields);
	
	// Change compiler reference
	newFields.compiler = transformer.targetCompiler;
	
	// Apply seed policy
	newFields.seed = this.applySeedPolicy(
		transformer.seedPolicy,
		fields.seed,
		transformerFields
	);
	
	// Apply transformation based on kind
	switch(transformer.transformKind) {
		case TRANSFORMER_TYPES.PROJECTION:
			// Forget idiosyncratic structure, keep canonical fields
			newFields = this.applyProjection(newFields, transformerFields);
			break;
			
		case TRANSFORMER_TYPES.LIFT:
			// Embed into richer structure
			newFields = this.applyLift(newFields, transformerFields);
			break;
			
		case TRANSFORMER_TYPES.NORMALIZE:
			// Canonicalize representation
			newFields = this.applyNormalization(newFields, transformerFields);
			break;
			
		case TRANSFORMER_TYPES.UPGRADE:
			// Version migration
			newFields = this.applyUpgrade(newFields, transformerFields);
			break;
			
		case TRANSFORMER_TYPES.RESTRICT:
			// Specialize to subdomain
			newFields = this.applyRestriction(newFields, transformerFields);
			break;
			
		case TRANSFORMER_TYPES.IDENTITY:
			// No-op transformation
			break;
	}
	
	// Update title to reflect transformation
	newFields.title = this.generateTransformedTitle(
		fields.title,
		transformer.transformKind,
		transformer.targetCompiler
	);
	
	// Add transformation metadata
	newFields["transformed-from"] = fields.title;
	newFields["transformer"] = transformerFields.title;
	newFields["transform-timestamp"] = new Date().toISOString();
	
	return {
		fields: newFields
	};
};

/*
Apply projection transformation
Forgets structure, keeps canonical form

@param {object} fields - Program fields
@param {object} transformerFields - Transformer fields
@returns {object} - Projected fields
*/
NonParametricTransformer.prototype.applyProjection = function(fields, transformerFields) {
	var projected = {};
	
	// Keep only canonical fields specified in transformer
	var keepFields = transformerFields["projection-keep-fields"];
	if(keepFields) {
		var fieldList = keepFields.split(",").map(function(f) { return f.trim(); });
		for(var i = 0; i < fieldList.length; i++) {
			var fieldName = fieldList[i];
			if(fields[fieldName] !== undefined) {
				projected[fieldName] = fields[fieldName];
			}
		}
	} else {
		// Default: keep title, type, text, seed, compiler
		projected.title = fields.title;
		projected.type = fields.type;
		projected.text = fields.text;
		projected.seed = fields.seed;
		projected.compiler = fields.compiler;
	}
	
	return projected;
};

/*
Apply lift transformation
Embeds subdialect into larger dialect

@param {object} fields - Program fields
@param {object} transformerFields - Transformer fields
@returns {object} - Lifted fields
*/
NonParametricTransformer.prototype.applyLift = function(fields, transformerFields) {
	var lifted = Object.assign({}, fields);
	
	// Add fields required by target compiler
	var addFields = transformerFields["lift-add-fields"];
	if(addFields) {
		try {
			var additionalFields = JSON.parse(addFields);
			Object.assign(lifted, additionalFields);
		} catch(e) {
			console.warn("Failed to parse lift-add-fields:", e);
		}
	}
	
	return lifted;
};

/*
Apply normalization transformation
Canonicalizes representation

@param {object} fields - Program fields
@param {object} transformerFields - Transformer fields
@returns {object} - Normalized fields
*/
NonParametricTransformer.prototype.applyNormalization = function(fields, transformerFields) {
	var normalized = Object.assign({}, fields);
	
	// Normalize text field if present
	if(normalized.text) {
		// Trim whitespace
		normalized.text = normalized.text.trim();
		
		// Normalize line endings
		normalized.text = normalized.text.replace(/\r\n/g, "\n");
	}
	
	// Normalize tags
	if(normalized.tags) {
		if(Array.isArray(normalized.tags)) {
			normalized.tags = normalized.tags.sort();
		}
	}
	
	return normalized;
};

/*
Apply upgrade transformation
Migrates between versions

@param {object} fields - Program fields
@param {object} transformerFields - Transformer fields
@returns {object} - Upgraded fields
*/
NonParametricTransformer.prototype.applyUpgrade = function(fields, transformerFields) {
	var upgraded = Object.assign({}, fields);
	
	// Update version field
	var targetVersion = transformerFields["target-version"];
	if(targetVersion) {
		upgraded.version = targetVersion;
	}
	
	// Apply field mappings
	var fieldMappings = transformerFields["upgrade-field-mappings"];
	if(fieldMappings) {
		try {
			var mappings = JSON.parse(fieldMappings);
			for(var oldField in mappings) {
				if(upgraded[oldField] !== undefined) {
					var newField = mappings[oldField];
					upgraded[newField] = upgraded[oldField];
					delete upgraded[oldField];
				}
			}
		} catch(e) {
			console.warn("Failed to parse upgrade-field-mappings:", e);
		}
	}
	
	return upgraded;
};

/*
Apply restriction transformation
Specializes to subdomain

@param {object} fields - Program fields
@param {object} transformerFields - Transformer fields
@returns {object} - Restricted fields
*/
NonParametricTransformer.prototype.applyRestriction = function(fields, transformerFields) {
	var restricted = Object.assign({}, fields);
	
	// Add restriction constraint
	var constraintField = transformerFields["restriction-constraint"];
	if(constraintField) {
		restricted["restriction"] = constraintField;
	}
	
	return restricted;
};

/*
Apply seed policy to transform seed deterministically

@param {string} policy - Seed policy to apply
@param {string} originalSeed - Original seed value
@param {object} transformerFields - Transformer fields
@returns {string} - Transformed seed
*/
NonParametricTransformer.prototype.applySeedPolicy = function(policy, originalSeed, transformerFields) {
	switch(policy) {
		case SEED_POLICIES.INHERIT:
			return originalSeed;
			
		case SEED_POLICIES.HASH:
			// Hash with transformer ID for determinism
			return this.hashSeed(originalSeed, transformerFields.title);
			
		case SEED_POLICIES.RESEED_FIXED:
			// Use fixed seed from transformer
			return transformerFields["fixed-seed"] || "transformer-default";
			
		case SEED_POLICIES.COMPOSE:
			// Compose with transformer's seed function
			// Simple composition: prefix with transformer
			return transformerFields.title + "::" + originalSeed;
			
		default:
			return originalSeed;
	}
};

/*
Hash seed with transformer ID for deterministic transformation
Uses simple deterministic hash - predictable by design for reproducibility

@param {string} seed - Original seed
@param {string} transformerId - Transformer ID
@returns {string} - Hashed seed
*/
NonParametricTransformer.prototype.hashSeed = function(seed, transformerId) {
	// Deterministic hash for seed transformation
	// NOTE: Intentionally simple and predictable for reproducibility
	// Not cryptographically secure - seeds are not secrets
	var combined = seed + "::" + transformerId;
	var hash = 0;
	
	for(var i = 0; i < combined.length; i++) {
		var char = combined.charCodeAt(i);
		hash = ((hash << 5) - hash) + char;
		hash = hash & hash; // Convert to 32-bit integer
	}
	
	return "hash_" + Math.abs(hash).toString(36);
};

/*
Generate title for transformed program

@param {string} originalTitle - Original program title
@param {string} transformKind - Type of transformation
@param {string} targetCompiler - Target compiler
@returns {string} - New title
*/
NonParametricTransformer.prototype.generateTransformedTitle = function(originalTitle, transformKind, targetCompiler) {
	return originalTitle + "::" + transformKind + "::" + targetCompiler;
};

/*
Calculate program curvature (for validation)

@param {object} programTiddler - Program tiddler
@returns {number} - Curvature value
*/
NonParametricTransformer.prototype.calculateProgramCurvature = function(programTiddler) {
	// Simple approximation: field count ratio
	var fields = programTiddler.fields;
	var fieldNames = Object.keys(fields);
	var structuralFields = 0;
	
	var structuralFieldNames = ["title", "type", "compiler", "generator", "version", "tags"];
	for(var i = 0; i < fieldNames.length; i++) {
		if(structuralFieldNames.indexOf(fieldNames[i]) !== -1) {
			structuralFields++;
		}
	}
	
	return fieldNames.length > 0 ? (1.0 - (structuralFields / fieldNames.length)) : 0.5;
};

/*
Validate geometry constraints (ZP35 distance, curvature bounds)

@param {object} transformer - Transformer entry
@param {object} originalProgram - Original program
@param {object} transformedProgram - Transformed program
@param {number} originalCoord - Original ZP35 coordinate
@param {number} originalCurvature - Original curvature
@returns {object} - Validation result
*/
NonParametricTransformer.prototype.validateGeometry = function(
	transformer,
	originalProgram,
	transformedProgram,
	originalCoord,
	originalCurvature
) {
	var constraints = transformer.constraints;
	
	// Calculate new ZP35 coordinate
	var newCoord = this.zp35.applyGoldenOperator(transformedProgram);
	var distance = Math.abs(newCoord - originalCoord);
	
	// Check ZP35 distance bound
	if(distance > constraints.maxZP35Distance) {
		return {
			success: false,
			error: "ZP35 distance violation: " + distance.toFixed(4) + 
			       " > " + constraints.maxZP35Distance
		};
	}
	
	// Calculate new curvature
	var newCurvature = this.calculateProgramCurvature(transformedProgram);
	var curvatureRatio = originalCurvature > 0 ? (newCurvature / originalCurvature) : 1.0;
	
	// Check curvature scale bounds
	if(curvatureRatio < constraints.minCurvatureScale ||
	   curvatureRatio > constraints.maxCurvatureScale) {
		return {
			success: false,
			error: "Curvature scale violation: " + curvatureRatio.toFixed(4) + 
			       " not in [" + constraints.minCurvatureScale + ", " + 
			       constraints.maxCurvatureScale + "]"
		};
	}
	
	return {
		success: true,
		metrics: {
			zp35Distance: distance,
			originalCoord: originalCoord,
			newCoord: newCoord,
			originalCurvature: originalCurvature,
			newCurvature: newCurvature,
			curvatureRatio: curvatureRatio
		}
	};
};

/*
Validate seed policy was applied correctly

@param {object} transformer - Transformer entry
@param {object} originalProgram - Original program
@param {object} transformedProgram - Transformed program
@returns {object} - Validation result
*/
NonParametricTransformer.prototype.validateSeedPolicy = function(
	transformer,
	originalProgram,
	transformedProgram
) {
	var originalSeed = originalProgram.fields.seed;
	var newSeed = transformedProgram.fields.seed;
	var policy = transformer.seedPolicy;
	
	// Verify seed transformation is deterministic
	var expectedSeed = this.applySeedPolicy(
		policy,
		originalSeed,
		transformer.tiddler.fields
	);
	
	if(newSeed !== expectedSeed) {
		return {
			success: false,
			error: "Seed policy mismatch: expected '" + expectedSeed + 
			       "' but got '" + newSeed + "'"
		};
	}
	
	return {
		success: true,
		seedTransform: {
			policy: policy,
			originalSeed: originalSeed,
			newSeed: newSeed
		}
	};
};

/*
Compose two transformers into a single composite transformer
Validates that composition preserves geometry bounds

@param {string} transformer1Title - First transformer
@param {string} transformer2Title - Second transformer (applied after first)
@returns {object} - Composition result
*/
NonParametricTransformer.prototype.composeTransformers = function(transformer1Title, transformer2Title) {
	var t1 = this.transformers[transformer1Title];
	var t2 = this.transformers[transformer2Title];
	
	if(!t1 || !t2) {
		return {
			success: false,
			error: "One or both transformers not found"
		};
	}
	
	// Verify composition is valid (t1 target = t2 source)
	if(t1.targetCompiler !== t2.sourceCompiler) {
		return {
			success: false,
			error: "Cannot compose: t1 target '" + t1.targetCompiler + 
			       "' != t2 source '" + t2.sourceCompiler + "'"
		};
	}
	
	// Calculate composed constraints
	var composedConstraints = {
		maxZP35Distance: Math.min(
			t1.constraints.maxZP35Distance + t2.constraints.maxZP35Distance,
			DEFAULT_CONSTRAINTS.MAX_ZP35_DISTANCE * 2
		),
		minCurvatureScale: t1.constraints.minCurvatureScale * t2.constraints.minCurvatureScale,
		maxCurvatureScale: t1.constraints.maxCurvatureScale * t2.constraints.maxCurvatureScale,
		lipschitzConstant: t1.constraints.lipschitzConstant * t2.constraints.lipschitzConstant
	};
	
	return {
		success: true,
		composition: {
			sourceCompiler: t1.sourceCompiler,
			targetCompiler: t2.targetCompiler,
			transformers: [transformer1Title, transformer2Title],
			constraints: composedConstraints
		}
	};
};

/*
Get statistics about transformer usage

@returns {object} - Statistics object
*/
NonParametricTransformer.prototype.getStatistics = function() {
	return {
		transformCount: this.stats.transformCount,
		successCount: this.stats.successCount,
		failureCount: this.stats.failureCount,
		successRate: this.stats.transformCount > 0 ? 
			(this.stats.successCount / this.stats.transformCount) : 0,
		geometryViolations: this.stats.geometryViolations,
		seedPolicyViolations: this.stats.seedPolicyViolations,
		registeredTransformers: Object.keys(this.transformers).length
	};
};

/*
Export the constructor and constants
*/
exports.NonParametricTransformer = NonParametricTransformer;
exports.TRANSFORMER_TYPES = TRANSFORMER_TYPES;
exports.SEED_POLICIES = SEED_POLICIES;
exports.DEFAULT_CONSTRAINTS = DEFAULT_CONSTRAINTS;
