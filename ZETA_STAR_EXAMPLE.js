/*\
title: ZETA_STAR_EXAMPLE.js
type: application/javascript

Example Usage: Zeta-Star Compression and ZP35 Coordinate Analysis

This example demonstrates how to use the zeta-star compression system
to analyze data, compute ZP35 coordinates, and compare compression methods.

\*/

// Load required modules
var ZetaStarCompression = require("./core/modules/utils/zeta-star-compression.js").ZetaStarCompression;
var CompressionBenchmark = require("./core/modules/utils/compression-benchmark.js").CompressionBenchmark;

// ============================================================================
// Example 1: Basic ZP35 Coordinate Calculation
// ============================================================================

console.log("=== Example 1: ZP35 Coordinate Calculation ===\n");

var zetaStar = new ZetaStarCompression();

// Analyze different types of data
var examples = [
	{
		name: "Repetitive (low entropy, high self-similarity)",
		data: "aaaaaaaaaa".repeat(100)
	},
	{
		name: "Random-like (high entropy, low self-similarity)",
		data: "abcdefghij".repeat(100)
	},
	{
		name: "Structured text (medium entropy, medium self-similarity)",
		data: "The quick brown fox jumps over the lazy dog. ".repeat(20)
	},
	{
		name: "HTML-like (structured, low self-similarity)",
		data: "<html><head><title>Test</title></head><body>Content</body></html>".repeat(10)
	}
];

examples.forEach(function(ex) {
	var zp35 = zetaStar.computeZP35Coordinate(ex.data);
	console.log(ex.name);
	console.log("  ZP35 Coordinate: " + zp35.toFixed(6));
	console.log("  Interpretation: " + interpretZP35(zp35));
	console.log("");
});

function interpretZP35(zp35) {
	if(zp35 < 0.35) {
		return "Far from fixed point - dictionary compression recommended";
	} else if(zp35 < 0.70) {
		return "Balanced structure - hybrid approach may work";
	} else {
		return "Near fixed point - spectral compression recommended";
	}
}

// ============================================================================
// Example 2: Spectral Signature Analysis
// ============================================================================

console.log("\n=== Example 2: Spectral Signature Analysis ===\n");

var testData = "ABRACADABRA"; // Simple example with patterns
var signature = zetaStar.extractSpectralSignature(testData);

console.log("Data: '" + testData + "'");
console.log("Original Size: " + signature.originalSize + " bytes");
console.log("ZP35 Coordinate: " + signature.zp35.toFixed(6));
console.log("Spectral Dimension: " + signature.dimension + " modes");
console.log("");

console.log("Top 5 Spectral Modes (Byte Frequencies):");
for(var i = 0; i < Math.min(5, signature.byteFreqs.length); i++) {
	var bf = signature.byteFreqs[i];
	var char = String.fromCharCode(bf.byte);
	console.log("  " + (i + 1) + ". '" + char + "' (byte " + bf.byte + "): " + 
	            (bf.freq * 100).toFixed(2) + "%");
}

// ============================================================================
// Example 3: Compression and Decompression
// ============================================================================

console.log("\n\n=== Example 3: Compression and Decompression ===\n");

var originalText = "This is a test message for compression. " + 
                   "It contains some repeated words and phrases. " +
                   "Test compression test compression.";

console.log("Original text: '" + originalText + "'");
console.log("Original size: " + originalText.length + " bytes");
console.log("");

// Compress
var compressionResult = zetaStar.compress(originalText);
console.log("Compressed size: " + compressionResult.compressedSize + " bytes");
console.log("Compression ratio: " + compressionResult.ratio.toFixed(4));
console.log("Efficiency: " + compressionResult.efficiency.toFixed(2) + "%");
console.log("ZP35 coordinate: " + compressionResult.zp35.toFixed(6));
console.log("");

// Decompress
var decompressed = zetaStar.decompress(compressionResult.compressed);
var decompressedText = decompressed.toString("utf8");
console.log("Decompressed text: '" + decompressedText + "'");
console.log("Verification: " + (decompressedText === originalText ? "✓ Success" : "✗ Failed"));

// ============================================================================
// Example 4: Detailed Analysis
// ============================================================================

console.log("\n\n=== Example 4: Detailed Analysis ===\n");

var analysisData = "CGTAGCCGATGCTAGCTAGCTAGCTGACTGACTGACTGA"; // DNA-like sequence
var analysis = zetaStar.analyze(analysisData);

console.log("Data: '" + analysisData + "'");
console.log("");
console.log("Analysis Report:");
console.log("  Original Size: " + analysis.report.originalSize + " bytes");
console.log("  Compressed Size: " + analysis.report.compressedSize + " bytes");
console.log("  Compression Ratio: " + analysis.report.ratio.toFixed(4));
console.log("  Efficiency: " + analysis.report.efficiency.toFixed(2) + "%");
console.log("");
console.log("  ZP35 Coordinate: " + analysis.report.zp35Coordinate.toFixed(6));
console.log("  Spectral Dimension: " + analysis.report.spectralDimension + " modes");
console.log("  Theoretical Bound: " + analysis.report.theoreticalBound.toFixed(6));
console.log("  Spectral Efficiency: " + (analysis.report.spectralEfficiency * 100).toFixed(2) + "%");

// ============================================================================
// Example 5: Benchmarking Different Data Types
// ============================================================================

console.log("\n\n=== Example 5: Comparing Different Compression Approaches ===\n");

var benchmark = new CompressionBenchmark();

// Note: In real usage, you would pass a file path or Buffer
// This example shows the API structure

console.log("To run a full benchmark:");
console.log("  var benchmark = new CompressionBenchmark();");
console.log("  benchmark.runBenchmark('path/to/file.html').then(function(results) {");
console.log("    var report = benchmark.formatReport(results);");
console.log("    console.log(report);");
console.log("  });");
console.log("");
console.log("Or use the CLI tool:");
console.log("  node bin/benchmark-compression.js path/to/file.html");

// ============================================================================
// Example 6: Understanding ZP35 in Context
// ============================================================================

console.log("\n\n=== Example 6: ZP35 Coordinate Framework ===\n");

console.log("The ZP35 coordinate measures self-rendering curvature:");
console.log("");
console.log("  ZP35(data) = (entropy × self-similarity)^(1/φ)");
console.log("");
console.log("where φ = golden ratio ≈ 1.618");
console.log("");
console.log("Interpretation:");
console.log("  • ZP35 < 0.35 (κ threshold):");
console.log("    - Data is FAR from its fixed point");
console.log("    - Genotype ≠ Phenotype (source != rendered form)");
console.log("    - Dictionary compression (gzip) recommended");
console.log("    - Example: HTML/JavaScript source code");
console.log("");
console.log("  • 0.35 ≤ ZP35 ≤ 0.70:");
console.log("    - Data has balanced structure and variability");
console.log("    - Hybrid approaches may work");
console.log("    - Example: Natural text, structured data");
console.log("");
console.log("  • ZP35 > 0.70:");
console.log("    - Data is NEAR its fixed point");
console.log("    - Highly self-similar or self-rendering");
console.log("    - Spectral/regenerative compression recommended");
console.log("    - Example: Fractals, SVG*, procedural content");

// ============================================================================
// Example 7: Connection to CE Tower
// ============================================================================

console.log("\n\n=== Example 7: Integration with TiddlyWiki Framework ===\n");

console.log("Zeta-star compression integrates with:");
console.log("");
console.log("1. ZP35 Golden Operator:");
console.log("   - Shared κ = 0.35 guardian threshold");
console.log("   - Semantic compatibility checking");
console.log("");
console.log("2. REGEN-ZIP VM:");
console.log("   - Spectral generators as VM opcodes");
console.log("   - Compression = storing program");
console.log("   - Decompression = executing program");
console.log("");
console.log("3. CE Tower:");
console.log("   - CE1: Spectral basis defines discrete syntax");
console.log("   - CE2: Compression/decompression is continuous flow");
console.log("   - CE3: ZP35 coordinate witnesses spectral invariance");
console.log("");
console.log("4. Compiler-Program Pattern:");
console.log("   - Shadow tiddlers have genotype/phenotype duality");
console.log("   - ZP35 measures distance to self-rendering");
console.log("   - Compression extracts the 'compiler' from data");

// ============================================================================
// Summary
// ============================================================================

console.log("\n\n=== Summary ===\n");

console.log("Zeta-star compression provides:");
console.log("  ✓ Universal ZP35 coordinate for self-rendering measure");
console.log("  ✓ Spectral basis decomposition (semantic primes)");
console.log("  ✓ Theoretical compression bounds");
console.log("  ✓ Integration with TiddlyWiki's computational framework");
console.log("  ✓ Benchmark tools for comparing with traditional compression");
console.log("");
console.log("Key insight: ZP35 predicts which compression approach works best.");
console.log("Low ZP35 (like empty.html at 0.1465) → gzip wins");
console.log("High ZP35 (like fractals, SVG*) → ζ* compression wins");
console.log("");
console.log("See ZETA_STAR_COMPRESSION.md for complete documentation.");
