/*\
title: $:/core/modules/utils/compiler-program-router.js
type: application/javascript
module-type: utils

Compiler-Program Router for TiddlyWiki

This module implements the "coherent data as compiler / chaotic data as program"
pattern for TiddlyWiki. It classifies tiddlers as either:

- **Compiler tiddlers**: High-coherence, stable assets that define semantic kernels
- **Program tiddlers**: Chaotic, task-specific data that gets routed through compilers

The router uses ZP35 coherence metrics to decide which compiler should own a given
program, enabling a clean separation between:
- What exists (compilers/kernels/asset manifolds)
- How we think about it (semantic transformations)
- What we're trying to do right now (programs/prompts)

This is analogous to ML systems where:
- Training builds the compiler (coherent latent geometry)
- Reasoning runs programs through the compiler (prompts → outputs)

\*/

"use strict";

/*
Classification thresholds for compiler vs program tiddlers
These are based on ZP35 coherence metrics
*/
var THRESHOLDS = {
	// Tiddlers with coherence > HIGH_COHERENCE are compilers
	HIGH_COHERENCE: 0.65,
	
	// Tiddlers with coherence < LOW_COHERENCE are programs
	LOW_COHERENCE: 0.35,
	
	// Guardian threshold for composition safety
	KAPPA: 0.35,
	
	// Out-of-distribution threshold
	OOD_THRESHOLD: 0.70
};

/*
Constants for coherence analysis
*/
var COHERENCE_CONSTANTS = {
	// Scaling factor for semantic distance from plateau center
	SEMANTIC_DISTANCE_SCALE: 10,
	
	// Score bounds
	SCORE_MIN: 0,
	SCORE_MAX: 1,
	
	// Maximum number of candidate compilers to return
	MAX_CANDIDATES: 3
};

/*
Compiler-Program Router Constructor
@param {object} wiki - TiddlyWiki instance
@param {object} zp35Operator - ZP35 operator instance
@param {object} regenZipVM - REGEN-ZIP VM instance
*/
function CompilerProgramRouter(wiki, zp35Operator, regenZipVM) {
	this.wiki = wiki;
	this.zp35 = zp35Operator;
	this.vm = regenZipVM;
	
	// Registry of compiler tiddlers
	this.compilers = {};
	
	// Registry of program tiddlers
	this.programs = {};
	
	// Routing cache
	this.routingCache = {};
}

/*
Classify a tiddler as compiler, program, or intermediate
Uses ZP35 coherence metrics to determine classification

@param {object} tiddler - Tiddler to classify
@returns {object} - Classification result
*/
CompilerProgramRouter.prototype.classify = function(tiddler) {
	if(!tiddler) {
		return {
			type: "unknown",
			confidence: 0.0,
			message: "Invalid tiddler"
		};
	}
	
	// Calculate ZP35 metrics
	var coord = this.zp35.applyGoldenOperator(tiddler);
	var height = this.zp35.calculateOrdinalHeight(tiddler);
	var signature = this.zp35.calculateSignature(tiddler);
	
	// Analyze coherence characteristics
	var coherence = this.analyzeCoherence(tiddler, coord, height);
	
	// Classify based on coherence
	if(coherence.score > THRESHOLDS.HIGH_COHERENCE) {
		// High coherence → Compiler tiddler
		return {
			type: "compiler",
			confidence: coherence.score,
			coord: coord,
			height: height,
			signature: signature,
			coherence: coherence,
			message: "High-coherence asset - acts as semantic kernel/compiler",
			characteristics: {
				stability: "high",
				role: "defines valid transformations",
				usage: "receives programs, materializes assets"
			}
		};
	} else if(coherence.score < THRESHOLDS.LOW_COHERENCE) {
		// Low coherence → Program tiddler
		return {
			type: "program",
			confidence: 1.0 - coherence.score,
			coord: coord,
			height: height,
			signature: signature,
			coherence: coherence,
			message: "Chaotic data - acts as program to be compiled",
			characteristics: {
				stability: "low",
				role: "ephemeral task specification",
				usage: "routed to compilers for execution"
			}
		};
	} else {
		// Intermediate coherence
		return {
			type: "intermediate",
			confidence: 0.5,
			coord: coord,
			height: height,
			signature: signature,
			coherence: coherence,
			message: "Intermediate coherence - can act as bridge or mediator",
			characteristics: {
				stability: "medium",
				role: "bridges semantic boundaries",
				usage: "facilitates composition"
			}
		};
	}
};

/*
Analyze coherence characteristics of a tiddler
Considers multiple factors: stability, structure, complexity

@param {object} tiddler - Tiddler to analyze
@param {number} coord - Fractal coordinate
@param {number} height - Ordinal height
@returns {object} - Coherence analysis
*/
CompilerProgramRouter.prototype.analyzeCoherence = function(tiddler, coord, height) {
	var factors = {
		structural: 0,
		semantic: 0,
		temporal: 0
	};
	
	// Structural coherence: field stability, type consistency
	if(tiddler.fields.type && tiddler.fields.type !== "text/vnd.tiddlywiki") {
		factors.structural += 0.3; // Typed content is more structured
	}
	
	if(tiddler.fields.tags && tiddler.fields.tags.length > 0) {
		factors.structural += 0.2; // Tags indicate categorization
	}
	
	if(tiddler.fields.generator || tiddler.fields["regen-zip"]) {
		factors.structural += 0.4; // Regenerative content is highly structured
	}
	
	// Semantic coherence: coordinate stability in fractal space
	// Coordinates near plateau centers indicate stable semantic positions
	var plateauCenter = Math.round(coord * COHERENCE_CONSTANTS.SEMANTIC_DISTANCE_SCALE) / COHERENCE_CONSTANTS.SEMANTIC_DISTANCE_SCALE;
	var distanceFromCenter = Math.abs(coord - plateauCenter);
	factors.semantic = Math.max(0, 1.0 - (distanceFromCenter * COHERENCE_CONSTANTS.SEMANTIC_DISTANCE_SCALE)); // Scale to [0, 1], clamp at 0
	
	// Temporal coherence: version fields, modification patterns
	if(tiddler.fields.version) {
		factors.temporal += 0.3; // Versioned content is more stable
	}
	
	if(tiddler.fields.seed) {
		factors.temporal += 0.3; // Seeded generation implies stability
	}
	
	// Combine factors
	var structuralWeight = 0.4;
	var semanticWeight = 0.4;
	var temporalWeight = 0.2;
	
	var score = (
		factors.structural * structuralWeight +
		factors.semantic * semanticWeight +
		factors.temporal * temporalWeight
	);
	
	// Normalize to [SCORE_MIN, SCORE_MAX]
	score = Math.max(COHERENCE_CONSTANTS.SCORE_MIN, Math.min(COHERENCE_CONSTANTS.SCORE_MAX, score));
	
	return {
		score: score,
		factors: factors,
		weights: {
			structural: structuralWeight,
			semantic: semanticWeight,
			temporal: temporalWeight
		}
	};
};

/*
Register a tiddler as a compiler
Compilers define semantic kernels and valid transformation spaces

@param {object} tiddler - Tiddler to register as compiler
@returns {boolean} - Success status
*/
CompilerProgramRouter.prototype.registerCompiler = function(tiddler) {
	if(!tiddler) {
		return false;
	}
	
	var classification = this.classify(tiddler);
	
	// Allow compiler or intermediate to be registered as compilers
	// This gives flexibility for edge cases
	if(classification.type === "program") {
		console.warn("CompilerProgramRouter: Tiddler '" + tiddler.fields.title + 
			"' has low coherence (program-type) - not suitable as compiler");
		return false;
	}
	
	var title = tiddler.fields.title;
	this.compilers[title] = {
		tiddler: tiddler,
		classification: classification,
		programs: [], // Programs routed to this compiler
		metrics: {
			executionCount: 0,
			successCount: 0,
			failureCount: 0
		}
	};
	
	return true;
};

/*
Register a tiddler as a program
Programs are ephemeral, task-specific data that get compiled

@param {object} tiddler - Tiddler to register as program
@returns {boolean} - Success status
*/
CompilerProgramRouter.prototype.registerProgram = function(tiddler) {
	if(!tiddler) {
		return false;
	}
	
	var classification = this.classify(tiddler);
	
	if(classification.type !== "program" && classification.type !== "intermediate") {
		console.warn("CompilerProgramRouter: Tiddler '" + tiddler.fields.title + 
			"' is classified as '" + classification.type + "', not program");
		// Allow registration anyway for flexibility
	}
	
	var title = tiddler.fields.title;
	this.programs[title] = {
		tiddler: tiddler,
		classification: classification,
		routedTo: null, // Compiler this program is routed to
		status: "pending"
	};
	
	return true;
};

/*
Route a program tiddler to the most appropriate compiler tiddler
Uses ZP35 distance to find the closest compiler in semantic space

@param {object} programTiddler - Program tiddler to route
@returns {object} - Routing result
*/
CompilerProgramRouter.prototype.route = function(programTiddler) {
	if(!programTiddler) {
		return {
			success: false,
			message: "Invalid program tiddler"
		};
	}
	
	var programTitle = programTiddler.fields.title;
	
	// Check cache
	if(this.routingCache[programTitle]) {
		return this.routingCache[programTitle];
	}
	
	// Get compiler list
	var compilerTitles = Object.keys(this.compilers);
	
	if(compilerTitles.length === 0) {
		return {
			success: false,
			message: "No compilers registered",
			suggestion: "Register at least one compiler tiddler"
		};
	}
	
	// Find best compiler based on ZP35 distance
	var programCoord = this.zp35.applyGoldenOperator(programTiddler);
	var bestCompiler = null;
	var bestDistance = Infinity;
	var candidates = [];
	
	for(var i = 0; i < compilerTitles.length; i++) {
		var compilerTitle = compilerTitles[i];
		var compilerEntry = this.compilers[compilerTitle];
		var compilerTiddler = compilerEntry.tiddler;
		
		var compilerCoord = this.zp35.applyGoldenOperator(compilerTiddler);
		var distance = Math.abs(programCoord - compilerCoord);
		
		candidates.push({
			title: compilerTitle,
			distance: distance,
			coord: compilerCoord
		});
		
		if(distance < bestDistance) {
			bestDistance = distance;
			bestCompiler = compilerEntry;
		}
	}
	
	// Sort candidates by distance
	candidates.sort(function(a, b) {
		return a.distance - b.distance;
	});
	
	// Check if best compiler is within acceptable range
	var mode = "safe";
	var confidence = 1.0;
	var message = "";
	
	if(bestDistance < THRESHOLDS.KAPPA) {
		mode = "safe";
		confidence = 1.0 - (bestDistance / THRESHOLDS.KAPPA);
		message = "Program routed to compiler within safe coherence range";
	} else if(bestDistance < 2 * THRESHOLDS.KAPPA) {
		mode = "caution";
		confidence = 0.5;
		message = "Program crosses semantic boundary - caution advised";
	} else if(bestDistance < THRESHOLDS.OOD_THRESHOLD) {
		mode = "borderline";
		confidence = 0.3;
		message = "Program at edge of compiler's semantic domain";
	} else {
		mode = "ood";
		confidence = 0.1;
		message = "Program is out-of-distribution - may need new compiler or sandbox";
	}
	
	var result = {
		success: true,
		compiler: bestCompiler,
		compilerTitle: bestCompiler.tiddler.fields.title,
		distance: bestDistance,
		mode: mode,
		confidence: confidence,
		message: message,
		programCoord: programCoord,
		compilerCoord: this.zp35.applyGoldenOperator(bestCompiler.tiddler),
		candidates: candidates.slice(0, COHERENCE_CONSTANTS.MAX_CANDIDATES) // Top candidates
	};
	
	// Cache result
	this.routingCache[programTitle] = result;
	
	// Update routing registry
	if(this.programs[programTitle]) {
		this.programs[programTitle].routedTo = bestCompiler.tiddler.fields.title;
		bestCompiler.programs.push(programTitle);
	}
	
	return result;
};

/*
Execute a program tiddler through its routed compiler
Materializes assets using the REGEN-ZIP VM pipeline

@param {object} programTiddler - Program tiddler to execute
@returns {object} - Execution result with generated assets
*/
CompilerProgramRouter.prototype.execute = function(programTiddler) {
	if(!programTiddler) {
		return {
			success: false,
			message: "Invalid program tiddler"
		};
	}
	
	// Route program to compiler
	var routing = this.route(programTiddler);
	
	if(!routing.success) {
		return {
			success: false,
			message: "Routing failed: " + routing.message,
			routing: routing
		};
	}
	
	// Check if execution is safe
	if(routing.mode === "ood") {
		return {
			success: false,
			message: "Program is out-of-distribution - execution blocked for safety",
			routing: routing,
			suggestion: "Create a new compiler for this semantic domain or sandbox the execution"
		};
	}
	
	var compiler = routing.compiler;
	
	// Prepare execution context
	// Merge program tiddler with compiler's semantic kernel
	var executionTiddler = this.mergeForExecution(compiler.tiddler, programTiddler);
	
	// Execute through REGEN-ZIP VM
	try {
		compiler.metrics.executionCount++;
		
		// Load into VM
		var loadSuccess = this.vm.load(executionTiddler);
		if(!loadSuccess) {
			compiler.metrics.failureCount++;
			return {
				success: false,
				message: "Failed to load execution tiddler into VM",
				routing: routing
			};
		}
		
		// Run VM
		var vmResult = this.vm.run();
		
		if(vmResult.success) {
			compiler.metrics.successCount++;
			
			return {
				success: true,
				assets: vmResult.assets,
				metadata: vmResult.metadata,
				routing: routing,
				compiler: compiler.tiddler.fields.title,
				program: programTiddler.fields.title
			};
		} else {
			compiler.metrics.failureCount++;
			
			return {
				success: false,
				message: "VM execution failed: " + vmResult.error,
				routing: routing
			};
		}
	} catch(e) {
		compiler.metrics.failureCount++;
		
		return {
			success: false,
			message: "Execution error: " + e.message,
			error: e,
			routing: routing
		};
	}
};

/*
Merge compiler and program tiddlers for execution
The compiler provides the semantic kernel, the program provides the specifics

@param {object} compilerTiddler - Compiler tiddler (semantic kernel)
@param {object} programTiddler - Program tiddler (task specification)
@returns {object} - Merged execution tiddler
*/
CompilerProgramRouter.prototype.mergeForExecution = function(compilerTiddler, programTiddler) {
	// Start with compiler as base (provides semantic kernel)
	var merged = {
		fields: Object.assign({}, compilerTiddler.fields)
	};
	
	// Override with program specifics (task-specific parameters)
	// Programs can override: seed, text, custom fields
	// Programs cannot override: generator, type (from compiler)
	var programFields = programTiddler.fields;
	
	// Allow program to specify seed
	if(programFields.seed) {
		merged.fields.seed = programFields.seed;
	}
	
	// Allow program to provide input text/data
	if(programFields.text) {
		merged.fields["program-text"] = programFields.text;
	}
	
	// Allow program to pass parameters
	if(programFields.params) {
		merged.fields.params = programFields.params;
	}
	
	// Preserve program identity for tracking
	merged.fields["program-source"] = programFields.title;
	merged.fields["compiler-source"] = compilerTiddler.fields.title;
	
	// Use a merged title for the execution context
	merged.fields.title = compilerTiddler.fields.title + "::" + programFields.title;
	
	// Ensure regen-zip field is set for VM loading
	// If not present, use generator name as inline reference
	if(!merged.fields["regen-zip"] && merged.fields.generator) {
		merged.fields["regen-zip"] = merged.fields.generator;
	}
	
	return merged;
};

/*
Get statistics about compiler usage
Useful for understanding which compilers are most active

@returns {object} - Statistics object
*/
CompilerProgramRouter.prototype.getStatistics = function() {
	var stats = {
		compilers: Object.keys(this.compilers).length,
		programs: Object.keys(this.programs).length,
		routings: Object.keys(this.routingCache).length,
		compilerDetails: []
	};
	
	// Gather per-compiler statistics
	var compilerTitles = Object.keys(this.compilers);
	for(var i = 0; i < compilerTitles.length; i++) {
		var title = compilerTitles[i];
		var compiler = this.compilers[title];
		
		stats.compilerDetails.push({
			title: title,
			programs: compiler.programs.length,
			executions: compiler.metrics.executionCount,
			successes: compiler.metrics.successCount,
			failures: compiler.metrics.failureCount,
			successRate: compiler.metrics.executionCount > 0 ? 
				(compiler.metrics.successCount / compiler.metrics.executionCount) : 0
		});
	}
	
	return stats;
};

/*
Clear routing cache
Useful when tiddlers have been modified

@returns {void}
*/
CompilerProgramRouter.prototype.clearCache = function() {
	this.routingCache = {};
};

/*
Export the constructor
*/
exports.CompilerProgramRouter = CompilerProgramRouter;

