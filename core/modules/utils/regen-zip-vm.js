/*\
title: $:/core/modules/utils/regen-zip-vm.js
type: application/javascript
module-type: utils

REGEN-ZIP Virtual Machine for Tiddlers

This module implements a regenerative, declarative, ZIP-backed virtual machine
for tiddlers. It provides an execution substrate where ZIP entries become 
instructions that regenerate rather than inflate data.

The VM provides:
- Segment-based execution semantics
- Jump tables for entry addressing
- Versioned feature support
- Checksum verification
- Generator-based asset creation

\*/

"use strict";

/*
Opcode definitions for the REGEN-ZIP VM
Each opcode represents a regeneration operation
*/
var OPCODES = {
	OP_SEED: 0x01,           // Initialize with seed data
	OP_GENERATOR: 0x02,      // Execute generator function
	OP_VERIFY: 0x03,         // Verify checksum/signature
	OP_ATTACH: 0x04,         // Attach generated asset
	OP_ZP35_CHECK: 0x05,     // Check ZP35 coherence
	OP_TW_INSERT: 0x06       // Insert into TiddlyWiki DOM
};

/*
REGEN-ZIP Virtual Machine Constructor
*/
function RegenZipVM(wiki) {
	this.wiki = wiki;
	this.generators = {};         // Registered generator functions
	this.context = {};            // Execution context
	this.assets = [];             // Generated assets
	this.state = "idle";          // VM state: idle, loading, running, complete, error
	this.kappa = 0.35;            // ZP35 coherence threshold (guardian)
}

/*
Register a generator function
@param {string} name - Generator name
@param {function} fn - Generator function
@param {object} metadata - Generator metadata (version, seed, etc.)
*/
RegenZipVM.prototype.registerGenerator = function(name, fn, metadata) {
	metadata = metadata || {};
	this.generators[name] = {
		fn: fn,
		version: metadata.version || "1.0.0",
		seed: metadata.seed || null,
		zp35: metadata.zp35 || null,
		description: metadata.description || ""
	};
};

/*
Load a tiddler into the VM
@param {object} tiddler - Tiddler object with regen-zip fields
@returns {boolean} - Success status
*/
RegenZipVM.prototype.load = function(tiddler) {
	if(!tiddler) {
		console.error("RegenZipVM: Cannot load null tiddler");
		return false;
	}
	
	this.state = "loading";
	this.context.tiddler = tiddler;
	this.context.title = tiddler.fields.title;
	
	// Check for regen-zip field
	var regenZipField = tiddler.fields["regen-zip"];
	if(regenZipField) {
		return this.loadRegenZip(regenZipField);
	}
	
	this.state = "idle";
	return false;
};

/*
Load regen-zip data (base64 or external reference)
@param {string} regenZipData - Base64 encoded or reference string
@returns {boolean} - Success status
*/
RegenZipVM.prototype.loadRegenZip = function(regenZipData) {
	if(!regenZipData) {
		console.error("RegenZipVM: No regen-zip data provided");
		return false;
	}
	
	try {
		// Check if it's a base64 string or external reference
		if(regenZipData.startsWith("data:") || this.isBase64(regenZipData)) {
			// Base64 encoded data
			this.context.regenZipData = this.decodeBase64(regenZipData);
		} else if(regenZipData.startsWith("http://") || regenZipData.startsWith("https://")) {
			// External reference - would need async loading
			console.warn("RegenZipVM: External references not yet implemented");
			return false;
		} else {
			// Treat as inline generator reference
			this.context.generatorName = regenZipData;
		}
		
		return true;
	} catch(e) {
		console.error("RegenZipVM: Error loading regen-zip data:", e);
		this.state = "error";
		return false;
	}
};

/*
Execute the VM program
@returns {object} - Execution result with generated assets
*/
RegenZipVM.prototype.run = function() {
	if(this.state !== "loading") {
		console.error("RegenZipVM: Cannot run - VM not in loading state");
		return { success: false, error: "Invalid state" };
	}
	
	this.state = "running";
	this.assets = [];
	
	try {
		// Get tiddler metadata
		var tiddler = this.context.tiddler;
		var generatorName = tiddler.fields.generator || this.context.generatorName;
		var seed = tiddler.fields.seed;
		var zp35 = tiddler.fields.zp35;
		var version = tiddler.fields.version;
		
		// Execute instruction sequence
		var result = this.executeInstructions(generatorName, seed, zp35, version);
		
		this.state = "complete";
		return {
			success: true,
			assets: this.assets,
			metadata: result.metadata
		};
	} catch(e) {
		console.error("RegenZipVM: Execution error:", e);
		this.state = "error";
		return { success: false, error: e.message };
	}
};

/*
Execute instruction sequence
@param {string} generatorName - Name of generator to execute
@param {string} seed - Seed hash
@param {string} zp35 - ZP35 coherence signature
@param {string} version - Version string
@returns {object} - Execution result
*/
RegenZipVM.prototype.executeInstructions = function(generatorName, seed, zp35, version) {
	var instructions = [];
	var metadata = {};
	
	// OP_SEED: Initialize with seed data
	if(seed) {
		instructions.push({ opcode: OPCODES.OP_SEED, data: seed });
		this.executeSeed(seed);
		metadata.seed = seed;
	}
	
	// OP_ZP35_CHECK: Verify coherence before execution
	if(zp35) {
		instructions.push({ opcode: OPCODES.OP_ZP35_CHECK, data: zp35 });
		var coherenceResult = this.executeZP35Check(generatorName, zp35);
		if(!coherenceResult.allowed) {
			throw new Error("ZP35 coherence check failed: " + coherenceResult.message);
		}
		metadata.zp35 = zp35;
		metadata.coherence = coherenceResult;
	}
	
	// OP_GENERATOR: Execute generator function
	if(generatorName) {
		instructions.push({ opcode: OPCODES.OP_GENERATOR, data: generatorName });
		this.executeGenerator(generatorName, seed, version);
		metadata.generator = generatorName;
		metadata.version = version;
	}
	
	// OP_VERIFY: Verify generated assets
	instructions.push({ opcode: OPCODES.OP_VERIFY });
	this.executeVerify();
	
	// OP_TW_INSERT: Insert into TiddlyWiki (would be handled by caller)
	instructions.push({ opcode: OPCODES.OP_TW_INSERT });
	
	metadata.instructions = instructions;
	return { metadata: metadata };
};

/*
Execute OP_SEED: Initialize execution context with seed
*/
RegenZipVM.prototype.executeSeed = function(seed) {
	this.context.seed = seed;
	this.context.rng = this.createSeededRNG(seed);
};

/*
Execute OP_GENERATOR: Run generator function
*/
RegenZipVM.prototype.executeGenerator = function(generatorName, seed, version) {
	var generator = this.generators[generatorName];
	
	if(!generator) {
		throw new Error("Generator not found: " + generatorName);
	}
	
	// Check version compatibility if specified
	if(version && generator.version !== version) {
		console.warn("RegenZipVM: Version mismatch - requested:", version, "available:", generator.version);
	}
	
	// Execute generator with context
	var generatorContext = {
		seed: seed,
		rng: this.context.rng,
		tiddler: this.context.tiddler,
		wiki: this.wiki
	};
	
	var result = generator.fn(generatorContext);
	
	// Validate result format
	if(!result) {
		console.warn("RegenZipVM: Generator '" + generatorName + "' returned no result");
		return;
	}
	
	if(!result.assets) {
		console.warn("RegenZipVM: Generator '" + generatorName + "' returned result without 'assets' property");
		return;
	}
	
	if(!Array.isArray(result.assets)) {
		console.warn("RegenZipVM: Generator '" + generatorName + "' returned non-array 'assets' property");
		return;
	}
	
	// Store generated assets
	this.assets = this.assets.concat(result.assets);
};

/*
Execute OP_ZP35_CHECK: Verify coherence using ZP35 golden operator
*/
RegenZipVM.prototype.executeZP35Check = function(generatorName, zp35Signature) {
	var generator = this.generators[generatorName];
	
	if(!generator) {
		return {
			allowed: false,
			distance: 1.0,
			message: "Generator not found"
		};
	}
	
	// Calculate semantic distance
	// If generator has ZP35 signature, compare it
	if(generator.zp35 && zp35Signature) {
		var distance = this.calculateZP35Distance(generator.zp35, zp35Signature);
		
		if(distance < this.kappa) {
			return {
				allowed: true,
				mode: "safe",
				distance: distance,
				confidence: 1.0 - (distance / this.kappa),
				message: "Generator maintains semantic coherence"
			};
		} else if(distance < 2 * this.kappa) {
			return {
				allowed: true,
				mode: "caution",
				distance: distance,
				confidence: 0.5,
				message: "Generator crosses semantic boundary - caution advised"
			};
		} else {
			return {
				allowed: false,
				mode: "blocked",
				distance: distance,
				confidence: 0.0,
				message: "Generator violates coherence threshold"
			};
		}
	}
	
	// If no signature comparison possible, allow with caution
	return {
		allowed: true,
		mode: "unchecked",
		distance: null,
		confidence: 0.5,
		message: "No ZP35 signature available for comparison"
	};
};

/*
Execute OP_VERIFY: Verify checksums and signatures
*/
RegenZipVM.prototype.executeVerify = function() {
	// Verify generated assets
	for(var i = 0; i < this.assets.length; i++) {
		var asset = this.assets[i];
		if(asset.checksum) {
			var computed = this.computeChecksum(asset.data);
			if(computed !== asset.checksum) {
				throw new Error("Checksum verification failed for asset: " + asset.name);
			}
		}
	}
};

/*
Calculate ZP35 semantic distance between two signatures
Using simplified distance metric based on coherence curvature
*/
RegenZipVM.prototype.calculateZP35Distance = function(sig1, sig2) {
	// Simplified implementation - would use full ZP35 golden operator
	if(sig1 === sig2) {
		return 0.0;
	}
	
	// Parse ZP35 signature format: "fractalCoord.ordinalHeight" (e.g., "0.500000.10")
	// The fractal coordinate is the first two parts when split by "."
	var parts1 = sig1.split(".");
	var parts2 = sig2.split(".");
	
	// Reconstruct fractal coordinates
	var coord1 = parseFloat(parts1[0] + "." + (parts1[1] || "0"));
	var coord2 = parseFloat(parts2[0] + "." + (parts2[1] || "0"));
	
	// Calculate distance in fractal space
	var distance = Math.abs(coord1 - coord2);
	
	// Ensure distance is in [0, 1] range
	distance = Math.min(1.0, distance);
	
	return distance;
};

/*
Create seeded random number generator
Uses xorshift128 algorithm for better statistical properties than LCG
*/
RegenZipVM.prototype.createSeededRNG = function(seed) {
	var hash = this.hashString(seed);
	
	// Initialize xorshift128 state from seed
	var x = hash & 0xFFFFFFFF;
	var y = (hash >>> 16) & 0xFFFFFFFF;
	var z = 362436069;
	var w = 88675123;
	
	return function() {
		// xorshift128 algorithm
		var t = x ^ (x << 11);
		x = y;
		y = z;
		z = w;
		w = (w ^ (w >>> 19)) ^ (t ^ (t >>> 8));
		
		// Ensure w stays positive for division
		var result = (w >>> 0) / 4294967296;
		return result;
	};
};

/*
Hash string to numeric seed
*/
RegenZipVM.prototype.hashString = function(str) {
	var hash = 0;
	for(var i = 0; i < str.length; i++) {
		var char = str.charCodeAt(i);
		hash = ((hash << 5) - hash) + char;
		hash = hash & hash; // Convert to 32-bit integer
	}
	return Math.abs(hash);
};

/*
Compute checksum for data
Note: This is a simple checksum for development/testing.
Production use should employ cryptographic hash functions (e.g., SHA-256)
from $tw.utils.crypto if available.
*/
RegenZipVM.prototype.computeChecksum = function(data) {
	var dataStr = String(data);
	
	// Check if crypto module is available for secure hashing
	if($tw && $tw.utils && $tw.utils.sha256) {
		return $tw.utils.sha256(dataStr);
	}
	
	// Fallback to simple hash (non-cryptographic)
	// This provides basic integrity checking but not security
	return "simple:" + this.hashString(dataStr).toString(16);
};

/*
Check if string is base64 encoded
*/
RegenZipVM.prototype.isBase64 = function(str) {
	if(!str || typeof str !== "string") {
		return false;
	}
	
	// Remove whitespace and check against base64 pattern
	var cleaned = str.replace(/\s/g, "");
	
	// Base64 pattern: alphanumeric, +, /, and optional = padding
	var base64Pattern = /^[A-Za-z0-9+/]*={0,2}$/;
	
	if(!base64Pattern.test(cleaned)) {
		return false;
	}
	
	// Length should be multiple of 4
	if(cleaned.length % 4 !== 0) {
		return false;
	}
	
	// Try to decode and re-encode
	try {
		if(typeof atob !== "undefined") {
			atob(cleaned);
			return true;
		}
		return true; // In Node.js, assume valid if pattern matches
	} catch(e) {
		return false;
	}
};

/*
Decode base64 string
*/
RegenZipVM.prototype.decodeBase64 = function(str) {
	if(str.startsWith("data:")) {
		// Extract data from data URL
		var parts = str.split(",");
		if(parts.length > 1) {
			str = parts[1];
		}
	}
	
	if(typeof atob !== "undefined") {
		return atob(str);
	} else {
		// Node.js environment
		return Buffer.from(str, "base64").toString("binary");
	}
};

/*
Reset VM state
*/
RegenZipVM.prototype.reset = function() {
	this.context = {};
	this.assets = [];
	this.state = "idle";
};

/*
Get VM state
*/
RegenZipVM.prototype.getState = function() {
	return {
		state: this.state,
		context: this.context,
		assets: this.assets,
		generators: Object.keys(this.generators)
	};
};

// Export constructor and opcodes
exports.RegenZipVM = RegenZipVM;
exports.OPCODES = OPCODES;
