/*\
title: $:/core/modules/utils/compression-benchmark.js
type: application/javascript
module-type: utils

Compression Benchmark Utility

Compares different compression methods including:
- Standard gzip compression (baseline)
- Zeta-star (ζ*) spectral basis compression
- Analysis of ZP35 coordinates and fixed-point properties

\*/

"use strict";

var zlib = require('zlib');
var fs = require('fs');
var path = require('path');

/*
Compression Benchmark Constructor
*/
function CompressionBenchmark(options) {
	options = options || {};
	this.verbose = options.verbose || false;
}

/*
Benchmark gzip compression
@param {Buffer} data - Input data
@returns {Promise<object>} - Compression results
*/
CompressionBenchmark.prototype.benchmarkGzip = function(data) {
	return new Promise(function(resolve, reject) {
		var startTime = Date.now();
		
		zlib.gzip(data, function(err, compressed) {
			if(err) {
				reject(err);
				return;
			}
			
			var endTime = Date.now();
			var compressionTime = endTime - startTime;
			
			// Decompress to verify
			var decompressStartTime = Date.now();
			zlib.gunzip(compressed, function(err, decompressed) {
				if(err) {
					reject(err);
					return;
				}
				
				var decompressEndTime = Date.now();
				var decompressionTime = decompressEndTime - decompressStartTime;
				
				// Verify correctness
				var correct = Buffer.compare(data, decompressed) === 0;
				
				resolve({
					method: 'gzip',
					originalSize: data.length,
					compressedSize: compressed.length,
					ratio: compressed.length / data.length,
					efficiency: (1 - (compressed.length / data.length)) * 100,
					compressionTime: compressionTime,
					decompressionTime: decompressionTime,
					totalTime: compressionTime + decompressionTime,
					correct: correct
				});
			});
		});
	});
};

/*
Benchmark zeta-star compression
@param {Buffer} data - Input data
@param {object} zetaStar - ZetaStarCompression instance
@returns {object} - Compression results
*/
CompressionBenchmark.prototype.benchmarkZetaStar = function(data, zetaStar) {
	var startTime = Date.now();
	var analysis = zetaStar.analyze(data);
	var compressionTime = Date.now() - startTime;
	
	// Decompress to verify
	var decompressStartTime = Date.now();
	var decompressed = zetaStar.decompress(analysis.compression.compressed);
	var decompressionTime = Date.now() - decompressStartTime;
	
	// Verify correctness
	var correct = Buffer.compare(data, decompressed) === 0;
	
	return {
		method: 'zeta-star',
		originalSize: data.length,
		compressedSize: analysis.compression.compressedSize,
		ratio: analysis.compression.ratio,
		efficiency: analysis.compression.efficiency,
		compressionTime: compressionTime,
		decompressionTime: decompressionTime,
		totalTime: compressionTime + decompressionTime,
		correct: correct,
		zp35Coordinate: analysis.signature.zp35,
		spectralDimension: analysis.signature.dimension,
		theoreticalBound: analysis.theoreticalBound,
		spectralEfficiency: analysis.spectralEfficiency
	};
};

/*
Run comprehensive benchmark comparing both methods
@param {string|Buffer} filePath - Path to file or buffer data
@param {object} options - Benchmark options
@returns {Promise<object>} - Benchmark results
*/
CompressionBenchmark.prototype.runBenchmark = function(filePath, options) {
	var self = this;
	options = options || {};
	
	return new Promise(function(resolve, reject) {
		// Load data
		var data;
		if(typeof filePath === 'string') {
			try {
				data = fs.readFileSync(filePath);
			} catch(e) {
				reject(new Error("Failed to read file: " + e.message));
				return;
			}
		} else if(Buffer.isBuffer(filePath)) {
			data = filePath;
		} else {
			reject(new Error("Invalid input: must be file path or Buffer"));
			return;
		}
		
		// Initialize zeta-star compression
		var ZetaStarCompression = require('./zeta-star-compression.js').ZetaStarCompression;
		var zetaStar = new ZetaStarCompression({
			kappa: options.kappa || 0.35,
			spectralResolution: options.spectralResolution || 128
		});
		
		// Run benchmarks
		var results = {
			filename: typeof filePath === 'string' ? path.basename(filePath) : 'buffer',
			fileSize: data.length,
			methods: {}
		};
		
		// Benchmark gzip
		self.benchmarkGzip(data).then(function(gzipResult) {
			results.methods.gzip = gzipResult;
			
			// Benchmark zeta-star
			var zetaStarResult = self.benchmarkZetaStar(data, zetaStar);
			results.methods.zetaStar = zetaStarResult;
			
			// Compute comparison metrics
			results.comparison = {
				zetaStarVsGzip: {
					ratioImprovement: (gzipResult.ratio - zetaStarResult.ratio) / gzipResult.ratio * 100,
					efficiencyDiff: zetaStarResult.efficiency - gzipResult.efficiency,
					timeRatio: zetaStarResult.totalTime / gzipResult.totalTime,
					sizeRatio: zetaStarResult.compressedSize / gzipResult.compressedSize
				},
				winner: gzipResult.compressedSize < zetaStarResult.compressedSize ? 'gzip' : 'zeta-star'
			};
			
			resolve(results);
		}).catch(function(err) {
			reject(err);
		});
	});
};

/*
Format benchmark results as a readable report
@param {object} results - Benchmark results
@returns {string} - Formatted report
*/
CompressionBenchmark.prototype.formatReport = function(results) {
	var lines = [];
	
	lines.push("╔══════════════════════════════════════════════════════════════════════════╗");
	lines.push("║              Compression Benchmark Report                               ║");
	lines.push("╚══════════════════════════════════════════════════════════════════════════╝");
	lines.push("");
	lines.push("File: " + results.filename);
	lines.push("Original Size: " + this.formatBytes(results.fileSize));
	lines.push("");
	lines.push("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
	lines.push("");
	
	// GZIP Results
	var gzip = results.methods.gzip;
	lines.push("┌─ GZIP Compression ─────────────────────────────────────────────────────┐");
	lines.push("│ Compressed Size:    " + this.pad(this.formatBytes(gzip.compressedSize), 50) + " │");
	lines.push("│ Compression Ratio:  " + this.pad(gzip.ratio.toFixed(4) + " (" + gzip.efficiency.toFixed(2) + "% saved)", 50) + " │");
	lines.push("│ Compression Time:   " + this.pad(gzip.compressionTime + " ms", 50) + " │");
	lines.push("│ Decompression Time: " + this.pad(gzip.decompressionTime + " ms", 50) + " │");
	lines.push("│ Total Time:         " + this.pad(gzip.totalTime + " ms", 50) + " │");
	lines.push("│ Verified:           " + this.pad(gzip.correct ? "✓ Correct" : "✗ Failed", 50) + " │");
	lines.push("└────────────────────────────────────────────────────────────────────────┘");
	lines.push("");
	
	// Zeta-Star Results
	var zeta = results.methods.zetaStar;
	lines.push("┌─ Zeta-Star (ζ*) Compression ───────────────────────────────────────────┐");
	lines.push("│ Compressed Size:    " + this.pad(this.formatBytes(zeta.compressedSize), 50) + " │");
	lines.push("│ Compression Ratio:  " + this.pad(zeta.ratio.toFixed(4) + " (" + zeta.efficiency.toFixed(2) + "% saved)", 50) + " │");
	lines.push("│ Compression Time:   " + this.pad(zeta.compressionTime + " ms", 50) + " │");
	lines.push("│ Decompression Time: " + this.pad(zeta.decompressionTime + " ms", 50) + " │");
	lines.push("│ Total Time:         " + this.pad(zeta.totalTime + " ms", 50) + " │");
	lines.push("│ Verified:           " + this.pad(zeta.correct ? "✓ Correct" : "✗ Failed", 50) + " │");
	lines.push("│                                                                          │");
	lines.push("│ ── Spectral Analysis ──                                                 │");
	lines.push("│ ZP35 Coordinate:    " + this.pad(zeta.zp35Coordinate.toFixed(6) + " (self-rendering curvature)", 50) + " │");
	lines.push("│ Spectral Dimension: " + this.pad(zeta.spectralDimension + " modes", 50) + " │");
	lines.push("│ Theoretical Bound:  " + this.pad(zeta.theoreticalBound.toFixed(6), 50) + " │");
	lines.push("│ Spectral Efficiency:" + this.pad((zeta.spectralEfficiency * 100).toFixed(2) + "%", 50) + " │");
	lines.push("└────────────────────────────────────────────────────────────────────────┘");
	lines.push("");
	
	// Comparison
	var comp = results.comparison;
	lines.push("┌─ Comparison: Zeta-Star vs GZIP ────────────────────────────────────────┐");
	lines.push("│ Winner:              " + this.pad(comp.winner + " (smaller compressed size)", 50) + " │");
	lines.push("│ Size Ratio (ζ*/gz):  " + this.pad(comp.zetaStarVsGzip.sizeRatio.toFixed(4), 50) + " │");
	lines.push("│ Efficiency Diff:     " + this.pad(comp.zetaStarVsGzip.efficiencyDiff.toFixed(2) + "%", 50) + " │");
	lines.push("│ Time Ratio (ζ*/gz):  " + this.pad(comp.zetaStarVsGzip.timeRatio.toFixed(4), 50) + " │");
	lines.push("└────────────────────────────────────────────────────────────────────────┘");
	lines.push("");
	
	// Analysis Notes
	lines.push("┌─ Analysis Notes ───────────────────────────────────────────────────────┐");
	lines.push("│                                                                          │");
	lines.push("│ ZP35 Coordinate Interpretation:                                         │");
	lines.push("│ • " + this.pad("ZP35 = " + zeta.zp35Coordinate.toFixed(4) + " measures self-rendering curvature", 68) + " │");
	
	if(zeta.zp35Coordinate < 0.35) {
		lines.push("│ • " + this.pad("Low ZP35 → Data far from fixed point, less self-similar", 68) + " │");
	} else if(zeta.zp35Coordinate > 0.70) {
		lines.push("│ • " + this.pad("High ZP35 → Data near fixed point, highly self-similar", 68) + " │");
	} else {
		lines.push("│ • " + this.pad("Medium ZP35 → Balanced structure and variability", 68) + " │");
	}
	
	lines.push("│                                                                          │");
	lines.push("│ Current Implementation Note:                                             │");
	lines.push("│ • " + this.pad("ζ* stores spectral signature + base64 literal data", 68) + " │");
	lines.push("│ • " + this.pad("Full implementation would use arithmetic/ANS coding", 68) + " │");
	lines.push("│ • " + this.pad("Theoretical bound shows potential compression ratio", 68) + " │");
	lines.push("│                                                                          │");
	lines.push("└────────────────────────────────────────────────────────────────────────┘");
	
	return lines.join('\n');
};

/*
Pad string to fixed width
@param {string} str - Input string
@param {number} width - Target width
@returns {string} - Padded string
*/
CompressionBenchmark.prototype.pad = function(str, width) {
	str = String(str);
	while(str.length < width) {
		str += ' ';
	}
	return str;
};

/*
Format bytes as human-readable string
@param {number} bytes - Number of bytes
@returns {string} - Formatted string
*/
CompressionBenchmark.prototype.formatBytes = function(bytes) {
	if(bytes < 1024) {
		return bytes + " B";
	} else if(bytes < 1024 * 1024) {
		return (bytes / 1024).toFixed(2) + " KB";
	} else {
		return (bytes / (1024 * 1024)).toFixed(2) + " MB";
	}
};

exports.CompressionBenchmark = CompressionBenchmark;
