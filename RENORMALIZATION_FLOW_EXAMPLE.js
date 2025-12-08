/*\
title: RENORMALIZATION_FLOW_EXAMPLE.js
type: application/javascript

Example Usage of Symbolic Renormalization Flow

This file demonstrates how to use the Renormalization Flow module to optimize
TiddlyWiki tiddlers and achieve canonical forms with minimal complexity.

\*/

// ============================================================================
// Example 1: Basic Renormalization of a Single Tiddler
// ============================================================================

function example1_BasicRenormalization() {
	console.log("\n=== Example 1: Basic Renormalization ===\n");
	
	// Create a verbose tiddler with redundant information
	var verboseTiddler = {
		fields: {
			title: "MyArticle",
			type: "text/vnd.tiddlywiki",
			text: "!! Introduction\n\nThis is a long article with lots of verbose content that could be compressed significantly without losing the essential meaning. It contains redundant explanations and repetitive patterns.\n\n!! Section 1\n\nMore verbose content here with unnecessary details.\n\n!! Section 2\n\nEven more redundant information that doesn't add much value.",
			tags: "article documentation tutorial guide reference manual",
			creator: "john.doe@example.com",
			modifier: "jane.smith@example.com",
			created: "20231201120000",
			modified: "20231205143000",
			revision: "5",
			customField1: "some value",
			customField2: "another value",
			notes: "Draft version with todo items"
		}
	};
	
	console.log("Original tiddler:");
	console.log("- Title:", verboseTiddler.fields.title);
	console.log("- Field count:", Object.keys(verboseTiddler.fields).length);
	console.log("- Text length:", verboseTiddler.fields.text.length);
	
	// Renormalize the tiddler
	var result = $tw.wiki.renormalizeTiddler(verboseTiddler, {
		maxIterations: 10,
		verbose: true
	});
	
	if(result.success) {
		console.log("\nRenormalization succeeded!");
		console.log("- Converged:", result.converged);
		console.log("- Iterations:", result.iterations);
		console.log("- Initial complexity:", result.initialComplexity.toFixed(2));
		console.log("- Final complexity:", result.finalComplexity.toFixed(2));
		console.log("- Complexity reduction:", result.complexityReduction.toFixed(2));
		console.log("- Coordinate drift:", result.coordinateDrift.toFixed(6));
		console.log("- Coordinate invariance preserved:", result.coordinateInvariance);
		
		console.log("\nCanonical form:");
		console.log("- Field count:", Object.keys(result.canonicalForm.fields).length);
		console.log("- Text length:", result.canonicalForm.fields.text.length);
		console.log("- Has renormalization metadata:", result.canonicalForm.fields.renormalized === "true");
	} else {
		console.log("Renormalization failed:", result.message);
	}
}

// ============================================================================
// Example 2: Batch Optimization of Multiple Tiddlers
// ============================================================================

function example2_BatchOptimization() {
	console.log("\n=== Example 2: Batch Optimization ===\n");
	
	// Create several tiddlers to optimize
	var tiddlers = [
		{
			fields: {
				title: "Article1",
				text: "Content for article 1 with some verbosity",
				tags: "tag1 tag2 tag3"
			}
		},
		{
			fields: {
				title: "Article2",
				text: "More verbose content for article 2 with extra details",
				tags: "tag1 tag2 tag3 tag4 tag5",
				field1: "value1",
				field2: "value2"
			}
		},
		{
			fields: {
				title: "Article3",
				text: "A".repeat(500), // Very long text
				tags: "tag1 tag2 tag3 tag4 tag5 tag6",
				creator: "test",
				modifier: "test"
			}
		}
	];
	
	console.log("Batch optimizing", tiddlers.length, "tiddlers...");
	
	// Get renormalization flow instance
	var flow = $tw.renormalizationFlow;
	
	if(!flow) {
		console.log("Renormalization flow not available!");
		return;
	}
	
	var results = flow.renormalizeBatch(tiddlers, {
		maxIterations: 10
	});
	
	console.log("\nBatch results:");
	console.log("- Total tiddlers:", results.totalTiddlers);
	console.log("- Success count:", results.successCount);
	console.log("- Failure count:", results.failureCount);
	console.log("- Total complexity reduction:", results.totalComplexityReduction.toFixed(2));
	console.log("- Average reduction:", results.averageReduction.toFixed(2));
	
	// Show per-tiddler results
	console.log("\nPer-tiddler results:");
	results.results.forEach(function(result, index) {
		if(result.success) {
			console.log("  " + tiddlers[index].fields.title + ":");
			console.log("    - Iterations:", result.iterations);
			console.log("    - Complexity reduction:", result.complexityReduction.toFixed(2));
		}
	});
}

// ============================================================================
// Example 3: Kernel Optimization (Shadow Tiddlers)
// ============================================================================

function example3_KernelOptimization() {
	console.log("\n=== Example 3: Kernel Optimization ===\n");
	
	console.log("Optimizing kernel (all shadow tiddlers)...");
	
	// This will find and optimize all shadow tiddlers
	var results = $tw.wiki.optimizeKernel({
		maxIterations: 10,
		verbose: false
	});
	
	if(results.success) {
		console.log("\nKernel optimization results:");
		console.log("- Shadow tiddlers found:", results.successCount + results.failureCount);
		console.log("- Successfully optimized:", results.successCount);
		console.log("- Failed:", results.failureCount);
		console.log("- Total complexity reduction:", results.totalComplexityReduction.toFixed(2));
		console.log("- Average reduction per tiddler:", results.averageReduction.toFixed(2));
	} else {
		console.log("Kernel optimization failed:", results.message);
	}
}

// ============================================================================
// Example 4: Canonical Form Detection
// ============================================================================

function example4_CanonicalDetection() {
	console.log("\n=== Example 4: Canonical Form Detection ===\n");
	
	// Create a tiddler and check if it's canonical
	var tiddler1 = {
		fields: {
			title: "Test1",
			text: "Short",
			renormalized: "true"
		}
	};
	
	var tiddler2 = {
		fields: {
			title: "Test2",
			text: "Long verbose content with lots of unnecessary details",
			tags: "tag1 tag2 tag3 tag4 tag5",
			field1: "value1",
			field2: "value2"
		}
	};
	
	console.log("Checking if tiddlers are in canonical form...");
	
	var isCanonical1 = $tw.wiki.isCanonicalForm(tiddler1);
	var isCanonical2 = $tw.wiki.isCanonicalForm(tiddler2);
	
	console.log("- Tiddler1 is canonical:", isCanonical1);
	console.log("- Tiddler2 is canonical:", isCanonical2);
	
	// Renormalize the non-canonical one
	if(!isCanonical2) {
		console.log("\nRenormalizing Tiddler2...");
		var result = $tw.wiki.renormalizeTiddler(tiddler2);
		if(result.success) {
			console.log("- Converged:", result.converged);
			console.log("- Now canonical:", $tw.wiki.isCanonicalForm(result.canonicalForm));
		}
	}
}

// ============================================================================
// Example 5: Coordinate Preservation Verification
// ============================================================================

function example5_CoordinatePreservation() {
	console.log("\n=== Example 5: Coordinate Preservation ===\n");
	
	var tiddler = {
		fields: {
			title: "TestCoordinates",
			text: "Content for coordinate testing",
			tags: "test example"
		}
	};
	
	// Calculate initial coordinate
	var zp35 = $tw.zp35Operator;
	var initialCoord = zp35.applyGoldenOperator(tiddler);
	
	console.log("Initial ZP35 coordinate:", initialCoord.toFixed(6));
	
	// Renormalize
	var result = $tw.wiki.renormalizeTiddler(tiddler);
	
	if(result.success) {
		// Calculate final coordinate
		var finalCoord = zp35.applyGoldenOperator(result.canonicalForm);
		
		console.log("Final ZP35 coordinate:", finalCoord.toFixed(6));
		console.log("Coordinate drift:", Math.abs(finalCoord - initialCoord).toFixed(6));
		console.log("Invariance preserved:", Math.abs(finalCoord - initialCoord) < 0.001);
		
		// Verify this matches the result's report
		console.log("\nResult reports:");
		console.log("- Coordinate drift:", result.coordinateDrift.toFixed(6));
		console.log("- Invariance:", result.coordinateInvariance);
	}
}

// ============================================================================
// Example 6: Semantic Preservation During Edits
// ============================================================================

function example6_SemanticPreservation() {
	console.log("\n=== Example 6: Semantic Preservation ===\n");
	
	// Create original tiddler
	var original = {
		fields: {
			title: "MyDocument",
			text: "Original content"
		}
	};
	
	// Calculate original coordinate
	var zp35 = $tw.zp35Operator;
	var originalCoord = zp35.applyGoldenOperator(original);
	console.log("Original coordinate:", originalCoord.toFixed(6));
	
	// Make some edits
	var edited = {
		fields: {
			title: "MyDocument",
			text: "Original content with minor edits"
		}
	};
	
	var editedCoord = zp35.applyGoldenOperator(edited);
	console.log("Edited coordinate:", editedCoord.toFixed(6));
	console.log("Coordinate change:", Math.abs(editedCoord - originalCoord).toFixed(6));
	
	// Check if edit preserved semantic identity (within κ = 0.35)
	var drift = Math.abs(editedCoord - originalCoord);
	var kappa = 0.35;
	
	if(drift < kappa) {
		console.log("✓ Semantic identity preserved (drift < κ = " + kappa + ")");
	} else {
		console.log("⚠ Semantic identity changed significantly (drift >= κ = " + kappa + ")");
	}
}

// ============================================================================
// Example 7: Progressive Optimization
// ============================================================================

function example7_ProgressiveOptimization() {
	console.log("\n=== Example 7: Progressive Optimization ===\n");
	
	var tiddler = {
		fields: {
			title: "Progressive",
			text: "A".repeat(1000),
			tags: "tag1 tag2 tag3 tag4 tag5 tag6 tag7 tag8",
			field1: "value1",
			field2: "value2",
			field3: "value3"
		}
	};
	
	var flow = $tw.renormalizationFlow;
	
	console.log("Performing progressive optimization with verbose logging...");
	
	var result = flow.renormalize(tiddler, {
		maxIterations: 10,
		verbose: true
	});
	
	if(result.success) {
		console.log("\nIteration history:");
		result.iterationHistory.forEach(function(iteration) {
			console.log("  Iteration", iteration.iteration + ":");
			console.log("    - Complexity:", iteration.complexity.toFixed(2));
			if(iteration.complexityDelta !== undefined) {
				console.log("    - Delta:", iteration.complexityDelta.toFixed(2));
				console.log("    - Improvement:", (iteration.improvement * 100).toFixed(1) + "%");
			}
		});
	}
}

// ============================================================================
// Main Entry Point
// ============================================================================

function runAllExamples() {
	console.log("========================================");
	console.log("Symbolic Renormalization Flow Examples");
	console.log("========================================");
	
	// Check if renormalization flow is available
	if(!$tw.renormalizationFlow) {
		console.log("\nError: Renormalization flow not initialized!");
		console.log("Make sure the regen-zip startup module has loaded.");
		return;
	}
	
	// Run examples
	try {
		example1_BasicRenormalization();
		example2_BatchOptimization();
		example3_KernelOptimization();
		example4_CanonicalDetection();
		example5_CoordinatePreservation();
		example6_SemanticPreservation();
		example7_ProgressiveOptimization();
	} catch(e) {
		console.log("\nError running examples:", e.message);
		console.log(e.stack);
	}
	
	console.log("\n========================================");
	console.log("Examples complete!");
	console.log("========================================");
}

// Export functions for use in TiddlyWiki
if(typeof exports !== "undefined") {
	exports.runAllExamples = runAllExamples;
	exports.example1_BasicRenormalization = example1_BasicRenormalization;
	exports.example2_BatchOptimization = example2_BatchOptimization;
	exports.example3_KernelOptimization = example3_KernelOptimization;
	exports.example4_CanonicalDetection = example4_CanonicalDetection;
	exports.example5_CoordinatePreservation = example5_CoordinatePreservation;
	exports.example6_SemanticPreservation = example6_SemanticPreservation;
	exports.example7_ProgressiveOptimization = example7_ProgressiveOptimization;
}

// To run these examples in a TiddlyWiki instance:
// 1. Load this file as a module
// 2. Call: require("RENORMALIZATION_FLOW_EXAMPLE.js").runAllExamples()
