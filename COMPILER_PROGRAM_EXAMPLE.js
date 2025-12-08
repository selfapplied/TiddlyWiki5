/*\
COMPILER-PROGRAM PATTERN EXAMPLES

This file demonstrates the compiler-program pattern in TiddlyWiki.
It shows how to:
1. Create compiler tiddlers (high-coherence semantic kernels)
2. Create program tiddlers (chaotic task specifications)
3. Route programs through compilers
4. Execute and materialize assets

The pattern maps to ML concepts:
- Compilers ≈ trained models (coherent latent geometry)
- Programs ≈ prompts/inference (ephemeral task specs)
- Execution ≈ running programs through the model

\*/

"use strict";

// ============================================================================
// EXAMPLE 1: Fractal Generator Compiler + Programs
// ============================================================================

/*
Example 1A: Define a Fractal Compiler Tiddler

This is a high-coherence tiddler that acts as a "compiler" or semantic kernel.
It defines a stable transformation space (fractal generation) that programs
can be compiled through.
*/

function createFractalCompilerExample() {
	return {
		fields: {
			title: "FractalCompiler",
			type: "application/x-tiddler-regen-zip",
			generator: "fractalGenerator",
			version: "1.0.0",
			seed: "golden-default",
			zp35: "0.618034.20",
			tags: ["compiler", "graphics", "procedural"],
			text: "Generates fractal images using Mandelbrot algorithm",
			description: "Semantic kernel for fractal generation"
		}
	};
}

/*
Example 1B: Define Fractal Program Tiddlers

These are low-coherence tiddlers that act as "programs".
They specify task-specific parameters that get compiled through
the FractalCompiler kernel.
*/

function createFractalProgramExamples() {
	return [
		// Program 1: Wide view
		{
			fields: {
				title: "Fractal_WideView",
				seed: "wide-view-seed",
				params: JSON.stringify({
					zoom: 1.0,
					centerX: -0.5,
					centerY: 0.0,
					maxIterations: 100,
					width: 800,
					height: 600
				}),
				text: "Generate wide view of Mandelbrot set"
			}
		},
		
		// Program 2: Zoomed detail
		{
			fields: {
				title: "Fractal_ZoomedDetail",
				seed: "zoom-detail-seed",
				params: JSON.stringify({
					zoom: 100.0,
					centerX: -0.7436,
					centerY: 0.1319,
					maxIterations: 500,
					width: 800,
					height: 600
				}),
				text: "Generate zoomed detail of interesting region"
			}
		},
		
		// Program 3: High resolution
		{
			fields: {
				title: "Fractal_HighRes",
				seed: "high-res-seed",
				params: JSON.stringify({
					zoom: 2.5,
					centerX: -0.5,
					centerY: 0.0,
					maxIterations: 200,
					width: 3840,
					height: 2160
				}),
				text: "Generate high resolution fractal for print"
			}
		}
	];
}

/*
Example 1C: Fractal Generator Implementation

This is the actual generator function that the compiler will execute.
It receives context (seed, rng, tiddler, wiki) and produces assets.
*/

function fractalGeneratorExample(context) {
	var seed = context.seed;
	var rng = context.rng;
	var params = JSON.parse(context.tiddler.fields.params || "{}");
	
	// Default parameters
	var zoom = params.zoom || 1.0;
	var centerX = params.centerX || -0.5;
	var centerY = params.centerY || 0.0;
	var maxIterations = params.maxIterations || 100;
	var width = params.width || 800;
	var height = params.height || 600;
	
	// Generate Mandelbrot fractal
	var imageData = generateMandelbrot(
		width, height,
		centerX, centerY,
		zoom, maxIterations,
		rng
	);
	
	// Compute checksum
	var checksum = computeSimpleChecksum(imageData);
	
	return {
		assets: [
			{
				name: "fractal.png",
				type: "image/png",
				data: imageData,
				checksum: checksum,
				metadata: {
					width: width,
					height: height,
					zoom: zoom,
					iterations: maxIterations
				}
			}
		]
	};
}

/*
Helper: Generate Mandelbrot fractal
*/
function generateMandelbrot(width, height, centerX, centerY, zoom, maxIterations, rng) {
	// Simplified implementation for demonstration
	// In production, this would generate actual image data
	
	var pixels = [];
	var scale = 4.0 / zoom;
	
	for(var y = 0; y < height; y++) {
		for(var x = 0; x < width; x++) {
			// Map pixel to complex plane
			var cx = centerX + (x - width/2) * scale / width;
			var cy = centerY + (y - height/2) * scale / height;
			
			// Calculate Mandelbrot iteration
			var zx = 0;
			var zy = 0;
			var iteration = 0;
			
			while(zx*zx + zy*zy < 4 && iteration < maxIterations) {
				var tmp = zx*zx - zy*zy + cx;
				zy = 2*zx*zy + cy;
				zx = tmp;
				iteration++;
			}
			
			// Color based on iteration count
			var color = iteration === maxIterations ? 0 : 
				Math.floor((iteration / maxIterations) * 255);
			
			pixels.push(color);
		}
	}
	
	// Return mock PNG data (in production, encode as actual PNG)
	return "PNG:" + pixels.length + ":" + pixels.slice(0, 10).join(",");
}

// ============================================================================
// EXAMPLE 2: Text Processor Compiler + Programs
// ============================================================================

/*
Example 2A: Define a Text Processor Compiler

This compiler handles text transformation tasks.
*/

function createTextProcessorCompilerExample() {
	return {
		fields: {
			title: "TextProcessorCompiler",
			type: "application/x-tiddler-regen-zip",
			generator: "textProcessor",
			version: "1.0.0",
			zp35: "0.450000.15",
			tags: ["compiler", "text", "transformation"],
			text: "Processes and transforms text content"
		}
	};
}

/*
Example 2B: Text Processing Programs
*/

function createTextProcessorProgramExamples() {
	return [
		// Program 1: Markdown to HTML
		{
			fields: {
				title: "TextProc_MarkdownToHTML",
				params: JSON.stringify({
					format: "markdown",
					output: "html"
				}),
				text: "# Hello World\n\nThis is **bold** text."
			}
		},
		
		// Program 2: Word count
		{
			fields: {
				title: "TextProc_WordCount",
				params: JSON.stringify({
					operation: "wordcount"
				}),
				text: "The quick brown fox jumps over the lazy dog."
			}
		},
		
		// Program 3: Syntax highlighting
		{
			fields: {
				title: "TextProc_SyntaxHighlight",
				params: JSON.stringify({
					language: "javascript",
					theme: "monokai"
				}),
				text: "function hello() {\n  console.log('Hello!');\n}"
			}
		}
	];
}

// ============================================================================
// EXAMPLE 3: Complete Pipeline Demonstration
// ============================================================================

/*
Example 3: Complete end-to-end pipeline

This shows the full flow from setup to execution.
*/

function demonstrateCompilerProgramPipeline() {
	// Assuming TiddlyWiki is loaded and available as $tw
	if(typeof $tw === "undefined") {
		console.log("TiddlyWiki not loaded - this is a demonstration");
		return;
	}
	
	console.log("=== Compiler-Program Pattern Demo ===\n");
	
	// Step 1: Setup components
	console.log("Step 1: Setting up components...");
	var wiki = $tw.wiki;
	var ZP35Operator = $tw.utils.ZP35Operator;
	var RegenZipVM = $tw.utils.RegenZipVM;
	var CompilerProgramRouter = $tw.utils.CompilerProgramRouter;
	
	var zp35 = new ZP35Operator();
	var vm = new RegenZipVM(wiki);
	var router = new CompilerProgramRouter(wiki, zp35, vm);
	
	// Step 2: Register generator with VM
	console.log("Step 2: Registering fractal generator...");
	vm.registerGenerator("fractalGenerator", fractalGeneratorExample, {
		version: "1.0.0",
		zp35: "0.618034.20",
		description: "Mandelbrot fractal generator"
	});
	
	// Step 3: Create and register compiler
	console.log("Step 3: Creating compiler tiddler...");
	var compiler = createFractalCompilerExample();
	var compilerSuccess = router.registerCompiler(compiler);
	console.log("  Compiler registered:", compilerSuccess);
	
	// Step 4: Classify compiler (should be "compiler")
	console.log("\nStep 4: Classifying compiler tiddler...");
	var compilerClass = router.classify(compiler);
	console.log("  Type:", compilerClass.type);
	console.log("  Confidence:", compilerClass.confidence.toFixed(3));
	console.log("  Coherence score:", compilerClass.coherence.score.toFixed(3));
	
	// Step 5: Create and register programs
	console.log("\nStep 5: Creating program tiddlers...");
	var programs = createFractalProgramExamples();
	programs.forEach(function(program, idx) {
		var programSuccess = router.registerProgram(program);
		console.log("  Program " + (idx + 1) + " registered:", programSuccess);
		
		// Classify program (should be "program")
		var programClass = router.classify(program);
		console.log("    Type:", programClass.type);
		console.log("    Coherence score:", programClass.coherence.score.toFixed(3));
	});
	
	// Step 6: Route and execute programs
	console.log("\nStep 6: Routing and executing programs...");
	programs.forEach(function(program, idx) {
		console.log("\n  === Program " + (idx + 1) + ": " + program.fields.title + " ===");
		
		// Route program to compiler
		var routing = router.route(program);
		console.log("    Routed to:", routing.compilerTitle);
		console.log("    Distance:", routing.distance.toFixed(4));
		console.log("    Mode:", routing.mode);
		console.log("    Confidence:", routing.confidence.toFixed(3));
		
		// Execute program
		var result = router.execute(program);
		if(result.success) {
			console.log("    Execution: SUCCESS");
			console.log("    Assets generated:", result.assets.length);
			result.assets.forEach(function(asset) {
				console.log("      - " + asset.name + " (" + asset.type + ")");
			});
		} else {
			console.log("    Execution: FAILED");
			console.log("    Error:", result.message);
		}
	});
	
	// Step 7: Get statistics
	console.log("\n\nStep 7: Router statistics...");
	var stats = router.getStatistics();
	console.log("  Total compilers:", stats.compilers);
	console.log("  Total programs:", stats.programs);
	console.log("  Total routings:", stats.routings);
	
	stats.compilerDetails.forEach(function(detail) {
		console.log("\n  Compiler: " + detail.title);
		console.log("    Programs routed:", detail.programs);
		console.log("    Executions:", detail.executions);
		console.log("    Success rate:", (detail.successRate * 100).toFixed(1) + "%");
	});
	
	console.log("\n=== Demo Complete ===");
}

// ============================================================================
// EXAMPLE 4: Out-of-Distribution Detection
// ============================================================================

/*
Example 4: Demonstrate OOD detection

This shows what happens when a program is too far from any compiler.
*/

function demonstrateOODDetection() {
	console.log("=== OOD Detection Demo ===\n");
	
	// Setup (abbreviated)
	var wiki = $tw.wiki;
	var zp35 = new $tw.utils.ZP35Operator();
	var vm = new $tw.utils.RegenZipVM(wiki);
	var router = new $tw.utils.CompilerProgramRouter(wiki, zp35, vm);
	
	// Register fractal compiler
	var fractalCompiler = createFractalCompilerExample();
	router.registerCompiler(fractalCompiler);
	
	// Create a program that's semantically very different (audio processing)
	var audioProgram = {
		fields: {
			title: "Audio_GenerateTone",
			params: JSON.stringify({
				frequency: 440,
				duration: 1.0,
				waveform: "sine"
			}),
			text: "Generate a 440Hz sine wave"
		}
	};
	
	console.log("Program:", audioProgram.fields.title);
	
	// Classify program
	var programClass = router.classify(audioProgram);
	console.log("Classification:", programClass.type);
	
	// Route program
	var routing = router.route(audioProgram);
	console.log("Routing to:", routing.compilerTitle);
	console.log("Distance:", routing.distance.toFixed(4));
	console.log("Mode:", routing.mode);
	
	// Attempt execution
	var result = router.execute(audioProgram);
	console.log("\nExecution result:", result.success ? "SUCCESS" : "BLOCKED");
	console.log("Message:", result.message);
	
	if(routing.mode === "ood") {
		console.log("\nSuggestion:", result.suggestion);
		console.log("\nThis program is out-of-distribution for available compilers.");
		console.log("Action needed: Create an AudioProcessorCompiler or sandbox execution.");
	}
	
	console.log("\n=== OOD Demo Complete ===");
}

// ============================================================================
// EXAMPLE 5: Compiler Evolution
// ============================================================================

/*
Example 5: Show how compilers can evolve over time

This demonstrates versioning and backward compatibility.
*/

function demonstrateCompilerEvolution() {
	console.log("=== Compiler Evolution Demo ===\n");
	
	// Version 1.0 compiler
	var compilerV1 = {
		fields: {
			title: "FractalCompiler:v1",
			generator: "fractalGeneratorV1",
			version: "1.0.0",
			zp35: "0.618034.20",
			tags: ["compiler", "graphics", "v1"],
			text: "Original fractal generator"
		}
	};
	console.log("Created compiler v1.0.0");
	
	// Version 2.0 compiler (enhanced)
	var compilerV2 = {
		fields: {
			title: "FractalCompiler:v2",
			generator: "fractalGeneratorV2",
			version: "2.0.0",
			zp35: "0.618034.22", // Slightly different signature (evolved)
			compatible_with: ["1.0.0"],
			tags: ["compiler", "graphics", "v2"],
			text: "Enhanced fractal generator with anti-aliasing"
		}
	};
	console.log("Created compiler v2.0.0 (backward compatible)");
	
	// Programs can specify version constraints
	var programWithConstraint = {
		fields: {
			title: "Fractal_RequiresV2",
			requires_compiler_version: ">=2.0.0",
			params: JSON.stringify({
				antialias: true, // V2-only feature
				zoom: 5.0
			}),
			text: "Generate anti-aliased fractal"
		}
	};
	console.log("\nProgram requires compiler v2.0.0 or higher");
	console.log("Feature used: anti-aliasing (v2 only)");
	
	// Routing would select appropriate compiler version
	console.log("\nRouting logic:");
	console.log("1. Check program's version constraint");
	console.log("2. Filter compatible compilers");
	console.log("3. Select best match based on ZP35 distance");
	console.log("4. Verify backward compatibility");
	
	console.log("\n=== Evolution Demo Complete ===");
}

// ============================================================================
// EXAMPLE 6: Multi-Kernel Composition
// ============================================================================

/*
Example 6: Compose multiple compilers/kernels

This shows how to chain transformations across different semantic domains.
*/

function demonstrateMultiKernelComposition() {
	console.log("=== Multi-Kernel Composition Demo ===\n");
	
	// Compiler 1: Fractal generator
	var fractalCompiler = {
		fields: {
			title: "FractalCompiler",
			generator: "fractalGenerator",
			zp35: "0.618034.20",
			output_type: "image/png"
		}
	};
	console.log("Kernel 1: FractalCompiler (generates images)");
	
	// Compiler 2: Image to 3D mesh converter
	var meshCompiler = {
		fields: {
			title: "MeshGeneratorCompiler",
			generator: "imageToMeshGenerator",
			zp35: "0.550000.18",
			input_type: "image/png",
			output_type: "model/gltf"
		}
	};
	console.log("Kernel 2: MeshGeneratorCompiler (image → 3D mesh)");
	
	// Bridge tiddler (intermediate coherence)
	var bridge = {
		fields: {
			title: "FractalToMesh_Bridge",
			type: "intermediate",
			source_compiler: "FractalCompiler",
			target_compiler: "MeshGeneratorCompiler",
			transform: "convertToHeightmap",
			zp35: "0.584000.19" // Between the two compilers
		}
	};
	console.log("Bridge: FractalToMesh (facilitates composition)");
	
	// Program that uses the pipeline
	var pipelineProgram = {
		fields: {
			title: "Generate_3D_Fractal_Landscape",
			pipeline: JSON.stringify([
				"FractalCompiler",
				"FractalToMesh_Bridge",
				"MeshGeneratorCompiler"
			]),
			params: JSON.stringify({
				fractal: {zoom: 10.0, iterations: 200},
				mesh: {scale: 1.0, smoothing: true}
			}),
			text: "Generate 3D landscape from fractal"
		}
	};
	console.log("\nProgram: Generate 3D fractal landscape");
	console.log("Pipeline: Fractal → Image → Bridge → Mesh");
	
	// Execution flow
	console.log("\nExecution flow:");
	console.log("1. Route program to FractalCompiler");
	console.log("2. Generate fractal image");
	console.log("3. Pass through FractalToMesh_Bridge (convert to heightmap)");
	console.log("4. Route to MeshGeneratorCompiler");
	console.log("5. Generate 3D mesh");
	console.log("6. Return final asset");
	
	console.log("\n=== Multi-Kernel Demo Complete ===");
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function computeSimpleChecksum(data) {
	var hash = 0;
	var str = String(data);
	for(var i = 0; i < str.length; i++) {
		var char = str.charCodeAt(i);
		hash = ((hash << 5) - hash) + char;
		hash = hash & hash;
	}
	return "checksum:" + Math.abs(hash).toString(16);
}

// ============================================================================
// EXPORTS
// ============================================================================

exports.createFractalCompilerExample = createFractalCompilerExample;
exports.createFractalProgramExamples = createFractalProgramExamples;
exports.fractalGeneratorExample = fractalGeneratorExample;
exports.createTextProcessorCompilerExample = createTextProcessorCompilerExample;
exports.createTextProcessorProgramExamples = createTextProcessorProgramExamples;
exports.demonstrateCompilerProgramPipeline = demonstrateCompilerProgramPipeline;
exports.demonstrateOODDetection = demonstrateOODDetection;
exports.demonstrateCompilerEvolution = demonstrateCompilerEvolution;
exports.demonstrateMultiKernelComposition = demonstrateMultiKernelComposition;

