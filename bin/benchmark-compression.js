#!/usr/bin/env node

/*\
Compression Benchmark CLI Tool

Usage:
  node bin/benchmark-compression.js <file-path>
  node bin/benchmark-compression.js editions/empty/output/empty.html

Benchmarks compression of the specified file using:
- Standard gzip compression
- Zeta-star (ζ*) spectral basis compression

\*/

var path = require('path');
var fs = require('fs');

// Load benchmark utilities
var CompressionBenchmark = require('../core/modules/utils/compression-benchmark.js').CompressionBenchmark;

// Parse command line arguments
var args = process.argv.slice(2);

if(args.length === 0) {
	console.log("Usage: node bin/benchmark-compression.js <file-path>");
	console.log("");
	console.log("Example:");
	console.log("  node bin/benchmark-compression.js editions/empty/output/empty.html");
	process.exit(1);
}

var filePath = args[0];

// Check if file exists
if(!fs.existsSync(filePath)) {
	console.error("Error: File not found: " + filePath);
	process.exit(1);
}

// Run benchmark
console.log("Starting compression benchmark...");
console.log("");

var benchmark = new CompressionBenchmark({ verbose: true });

benchmark.runBenchmark(filePath).then(function(results) {
	// Print formatted report
	var report = benchmark.formatReport(results);
	console.log(report);
	
	// Write results to JSON file
	var outputPath = path.join(
		path.dirname(filePath),
		path.basename(filePath, path.extname(filePath)) + '-benchmark.json'
	);
	
	fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
	console.log("");
	console.log("Detailed results saved to: " + outputPath);
	
}).catch(function(err) {
	console.error("Benchmark failed:", err.message);
	console.error(err.stack);
	process.exit(1);
});
