/*\
title: SHADOW_INDUCTION_EXAMPLE.js
type: application/javascript

Shadow Induction Usage Examples

This file demonstrates how to use the shadow induction feature
to let tiddlers generate their own compilers.

\*/

"use strict";

/*
Example 1: Basic Shadow Induction
===================================

Create a tiddler and induce its shadow compiler.
*/

function example1_basicInduction() {
	// Get required modules
	var ZP35Operator = $tw.utils.ZP35Operator;
	var CompilerProgramRouter = $tw.utils.CompilerProgramRouter;
	var RegenZipVM = $tw.utils.RegenZipVM;
	
	// Initialize system
	var zp35 = new ZP35Operator();
	var vm = new RegenZipVM($tw.wiki);
	var router = new CompilerProgramRouter($tw.wiki, zp35, vm);
	
	// Create a tiddler with no compiler
	var myTiddler = {
		fields: {
			title: "MyPersonalNote",
			text: "# My Note\n\nThis is **important** content with [[links]]",
			tags: ["notes", "personal"],
			created: new Date().toISOString()
		}
	};
	
	// Induce shadow compiler
	var result = router.induceShadow(myTiddler);
	
	console.log("Shadow induction result:", result.success);
	console.log("Shadow compiler title:", result.shadowCompiler.fields.title);
	console.log("Original marked as self-hosted:", result.selfHostedProgram.fields["program-mode"]);
	
	// The shadow compiler can now interpret the original tiddler
	return result;
}

/*
Example 2: Automatic Routing with Shadow Induction
===================================================

When no suitable compiler exists, routing automatically induces a shadow.
*/

function example2_automaticInduction() {
	var ZP35Operator = $tw.utils.ZP35Operator;
	var CompilerProgramRouter = $tw.utils.CompilerProgramRouter;
	var RegenZipVM = $tw.utils.RegenZipVM;
	
	var zp35 = new ZP35Operator();
	var vm = new RegenZipVM($tw.wiki);
	var router = new CompilerProgramRouter($tw.wiki, zp35, vm);
	
	// Create a program tiddler
	var program = {
		fields: {
			title: "MyTask",
			text: "Task description",
			tags: ["todo"]
		}
	};
	
	// Route the program (no compilers exist)
	// Shadow induction happens automatically
	var routing = router.route(program);
	
	console.log("Routing mode:", routing.mode);                    // "induced"
	console.log("Shadow induced:", routing.shadowInduction);        // true
	console.log("Compiler title:", routing.compilerTitle);          // "MyTask-shadow"
	console.log("Distance:", routing.distance);                     // Small (self-similar)
	
	return routing;
}

/*
Example 3: Shadow Induction for Out-of-Distribution Programs
=============================================================

When a program is too far from any existing compiler, shadow induction
creates a personal compiler.
*/

function example3_oodInduction() {
	var ZP35Operator = $tw.utils.ZP35Operator;
	var CompilerProgramRouter = $tw.utils.CompilerProgramRouter;
	var RegenZipVM = $tw.utils.RegenZipVM;
	
	var zp35 = new ZP35Operator();
	var vm = new RegenZipVM($tw.wiki);
	var router = new CompilerProgramRouter($tw.wiki, zp35, vm);
	
	// Register a very specific compiler
	var specificCompiler = {
		fields: {
			title: "TechnicalDocCompiler",
			type: "application/x-technical-doc",
			generator: "techDocGen",
			tags: ["technical", "documentation", "formal"]
		}
	};
	
	router.registerCompiler(specificCompiler);
	
	// Create a very different program (creative writing)
	var creativeProgram = {
		fields: {
			title: "MyPoem",
			text: "Roses are red\nViolets are blue",
			tags: ["poetry", "creative"]
		}
	};
	
	// Route the program
	// Distance to TechnicalDocCompiler will be large (OOD)
	// Shadow induction happens instead
	var routing = router.route(creativeProgram);
	
	if(routing.mode === "induced") {
		console.log("Program was OOD - shadow induced");
		console.log("Personal compiler:", routing.compilerTitle);  // "MyPoem-shadow"
	}
	
	return routing;
}

/*
Example 4: Analyzing Coherence Before Induction
================================================

Examine the internal coherence analysis to understand
what gets extracted as crisp vs. chaotic.
*/

function example4_coherenceAnalysis() {
	var ZP35Operator = $tw.utils.ZP35Operator;
	var ShadowInducer = $tw.utils.ShadowInducer;
	
	var zp35 = new ZP35Operator();
	var inducer = new ShadowInducer($tw.wiki, zp35);
	
	// Create a tiddler with mixed content
	var tiddler = {
		fields: {
			title: "MixedContentTiddler",
			type: "text/vnd.tiddlywiki",
			generator: "customGen",
			version: "1.0.0",
			text: "Random chaotic content: " + Math.random(),
			tags: ["structured", "test"],
			customData: "asdf1234lkjasdf98234"
		}
	};
	
	// Analyze coherence
	var analysis = inducer.analyzeInternalCoherence(tiddler);
	
	console.log("Crisp fields:", analysis.crispFields.map(f => f.name));
	// Expected: ["title", "type", "generator", "version", "tags"]
	
	console.log("Chaotic fields:", analysis.chaoticFields.map(f => f.name));
	// Expected: ["text", "customData"]
	
	console.log("Curvature coefficient:", analysis.curvatureCoefficient);
	// Lower = more rigid, Higher = more flexible
	
	return analysis;
}

/*
Example 5: Pattern Extraction
==============================

See what structural patterns are extracted from tiddler content.
*/

function example5_patternExtraction() {
	var ZP35Operator = $tw.utils.ZP35Operator;
	var ShadowInducer = $tw.utils.ShadowInducer;
	
	var zp35 = new ZP35Operator();
	var inducer = new ShadowInducer($tw.wiki, zp35);
	
	// Create tiddler with rich markdown content
	var text = 
		"# Main Heading\n\n" +
		"This has **bold** and *italic* text.\n\n" +
		"Also some `code` and [[wiki links]].\n\n" +
		"Plus {{transclusions}} for dynamic content.";
	
	var patterns = inducer.extractPatterns(text);
	
	console.log("Detected patterns:");
	patterns.forEach(function(p) {
		console.log("  -", p.type, "count:", p.count);
	});
	
	// Expected patterns:
	// - heading (1)
	// - bold (1)
	// - italic (1)
	// - code (1)
	// - link (1)
	// - transclusion (1)
	
	return patterns;
}

/*
Example 6: Controlling Shadow Induction
========================================

Shadow induction can be disabled for traditional behavior.
*/

function example6_disableInduction() {
	var ZP35Operator = $tw.utils.ZP35Operator;
	var CompilerProgramRouter = $tw.utils.CompilerProgramRouter;
	var RegenZipVM = $tw.utils.RegenZipVM;
	
	var zp35 = new ZP35Operator();
	var vm = new RegenZipVM($tw.wiki);
	var router = new CompilerProgramRouter($tw.wiki, zp35, vm);
	
	var program = {
		fields: {
			title: "OrphanProgram",
			text: "Content"
		}
	};
	
	// Disable shadow induction
	var routing = router.route(program, { allowShadowInduction: false });
	
	console.log("Success:", routing.success);        // false
	console.log("Message:", routing.message);         // "No compilers registered"
	
	// Enable shadow induction (default)
	var routing2 = router.route(program, { allowShadowInduction: true });
	
	console.log("Success:", routing2.success);       // true
	console.log("Mode:", routing2.mode);             // "induced"
	
	return { disabled: routing, enabled: routing2 };
}

/*
Example 7: Shadow Compiler Structure
=====================================

Examine the structure of a generated shadow compiler.
*/

function example7_shadowStructure() {
	var ZP35Operator = $tw.utils.ZP35Operator;
	var ShadowInducer = $tw.utils.ShadowInducer;
	
	var zp35 = new ZP35Operator();
	var inducer = new ShadowInducer($tw.wiki, zp35);
	
	var tiddler = {
		fields: {
			title: "ExampleTiddler",
			type: "text/vnd.tiddlywiki",
			tags: ["example"],
			text: "Content here"
		}
	};
	
	var result = inducer.induceShadow(tiddler);
	var shadow = result.shadowCompiler;
	
	console.log("Shadow compiler fields:");
	console.log("  title:", shadow.fields.title);                    // "ExampleTiddler-shadow"
	console.log("  type:", shadow.fields.type);                      // "application/x-tiddler-shadow-compiler"
	console.log("  shadow-source:", shadow.fields["shadow-source"]); // "ExampleTiddler"
	console.log("  shadow-type:", shadow.fields["shadow-type"]);     // "induced"
	console.log("  tags:", shadow.fields.tags);                      // ["$:/tags/ShadowCompiler"]
	console.log("  zp35:", shadow.fields.zp35);                      // ZP35 signature
	console.log("  seed:", shadow.fields.seed);                      // Generated seed
	
	return shadow;
}

/*
Example 8: Self-Hosted Program Structure
=========================================

Examine how original tiddlers are marked as self-hosted programs.
*/

function example8_selfHostedStructure() {
	var ZP35Operator = $tw.utils.ZP35Operator;
	var ShadowInducer = $tw.utils.ShadowInducer;
	
	var zp35 = new ZP35Operator();
	var inducer = new ShadowInducer($tw.wiki, zp35);
	
	var tiddler = {
		fields: {
			title: "ExampleTiddler",
			text: "Content",
			tags: ["original"]
		}
	};
	
	var result = inducer.induceShadow(tiddler);
	var program = result.selfHostedProgram;
	
	console.log("Self-hosted program fields:");
	console.log("  title:", program.fields.title);                  // "ExampleTiddler" (unchanged)
	console.log("  compiler:", program.fields.compiler);            // "ExampleTiddler-shadow"
	console.log("  program-mode:", program.fields["program-mode"]); // "self-hosted"
	console.log("  shadow-compiler:", program.fields["shadow-compiler"]); // "ExampleTiddler-shadow"
	console.log("  tags:", program.fields.tags);                    // [...original, "$:/tags/SelfHostedProgram"]
	
	return program;
}

/*
Export examples for use in Node.js or browser
*/
if(typeof exports !== "undefined") {
	exports.example1_basicInduction = example1_basicInduction;
	exports.example2_automaticInduction = example2_automaticInduction;
	exports.example3_oodInduction = example3_oodInduction;
	exports.example4_coherenceAnalysis = example4_coherenceAnalysis;
	exports.example5_patternExtraction = example5_patternExtraction;
	exports.example6_disableInduction = example6_disableInduction;
	exports.example7_shadowStructure = example7_shadowStructure;
	exports.example8_selfHostedStructure = example8_selfHostedStructure;
}
