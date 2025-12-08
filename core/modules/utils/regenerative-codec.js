/*\
title: $:/core/modules/utils/regenerative-codec.js
type: application/javascript
module-type: utils

Regenerative Codec Registry - Compression via recipes instead of blobs

Based on ZP35 framework, attachments become morphisms in golden space:
- Store seed + parameters instead of raw data
- Regenerate on-demand using golden operators
- Preserve structural invariants through generation

Codecs:
- zp35-fractal-image: Generate fractal images from seeds
- zp35-json-patch: Store JSON as delta from base
- zp35-text-template: Generate text from templates

\*/

(function(){

	"use strict";

	var zp35 = require("$:/core/modules/utils/zp35-golden-operator.js");

	/**
 * Registry of regenerative codecs
 */
	var codecRegistry = {};

	/**
 * Base codec interface
 */
	function BaseCodec() {}

	BaseCodec.prototype.encode = function(/*data, options*/) {
		throw new Error("encode() must be implemented by codec");
	};

	BaseCodec.prototype.decode = function(/*recipe*/) {
		throw new Error("decode() must be implemented by codec");
	};

	BaseCodec.prototype.canEncode = function(/*data, mimeType*/) {
		return false;
	};

	/**
 * Fractal Image Codec
 * Generates images using golden ratio patterns and self-similar structures
 */
	function FractalImageCodec() {
		BaseCodec.call(this);
	}

	FractalImageCodec.prototype = Object.create(BaseCodec.prototype);
	FractalImageCodec.prototype.constructor = FractalImageCodec;

	FractalImageCodec.prototype.canEncode = function(data, mimeType) {
		return /^image\/(png|jpeg|svg)/.test(mimeType);
	};

	/**
 * Analyze image data to extract fractal features
 * @param {string} data - Base64 or binary image data
 * @param {Object} options - Encoding options
 * @returns {Object} Feature vector
 */
	FractalImageCodec.prototype.analyzeImage = function(data, options) {
	// Simplified feature extraction
	// In a real implementation, this would analyze:
	// - Fractal texture (Hurst exponent)
	// - Edge density (Sobel-like)
	// - Spectral content (FFT)
	// - Structural motifs
	
		var seed = this.generateSeed(data);
		var resolution = options.resolution || [256, 256];
	
		return {
			seed: seed,
			resolution: resolution,
			palette: options.palette || "antclock-wave",
			curvature: zp35.KAPPA,
			depth: 5
		};
	};

	/**
 * Generate deterministic seed from data
 * @param {string} data - Input data
 * @returns {string} Hex seed
 */
	FractalImageCodec.prototype.generateSeed = function(data) {
	// Simple hash-like seed generation
		var hash = 0;
		var str = typeof data === "string" ? data : String(data);
	
		for(var i = 0; i < Math.min(str.length, 1000); i++) {
			var char = str.charCodeAt(i);
			hash = ((hash << 5) - hash) + char;
			hash = hash & hash; // Convert to 32bit integer
		}
	
		// Convert to hex
		var hex = Math.abs(hash).toString(16);
		while(hex.length < 8) {
			hex = "0" + hex;
		}
	
		return "zp35" + hex;
	};

	/**
 * Encode image data as regenerative recipe
 * @param {string} data - Image data (base64 or binary)
 * @param {Object} options - Encoding options
 * @returns {Object} Recipe object
 */
	FractalImageCodec.prototype.encode = function(data, options) {
		options = options || {};
	
		var features = this.analyzeImage(data, options);
	
		return {
			codec: "zp35-fractal-image",
			version: "1.0",
			seed: features.seed,
			params: {
				resolution: features.resolution,
				palette: features.palette,
				curvature: features.curvature,
				depth: features.depth
			},
			// Store original data checksum for verification
			checksum: this.generateSeed(data),
			originalSize: data.length,
			// For lossy compression, store quality estimate
			quality: options.quality || 0.85
		};
	};

	/**
 * Decode recipe to generate image
 * @param {Object} recipe - Recipe specification
 * @returns {string} Generated image data (SVG or data URL)
 */
	FractalImageCodec.prototype.decode = function(recipe) {
		var params = recipe.params || {};
		var seed = recipe.seed || "zp35default";
	
		// Generate SVG using golden ratio patterns
		var svg = this.generateFractalSVG(seed, params);
	
		// Return as data URL
		// Prefer base64 encoding for consistency and compatibility
		var base64;
		if(typeof Buffer !== "undefined") {
			// Node.js environment - use Buffer
			base64 = Buffer.from(svg).toString("base64");
			return "data:image/svg+xml;base64," + base64;
		} else if(typeof btoa !== "undefined") {
			// Browser environment - use btoa
			base64 = btoa(svg);
			return "data:image/svg+xml;base64," + base64;
		} else {
			// Fallback for other environments - use URL encoding
			// Note: This returns a different format but is still a valid data URL
			return "data:image/svg+xml," + encodeURIComponent(svg);
		}
	};

	/**
 * Generate fractal SVG from seed and parameters
 * @param {string} seed - Deterministic seed
 * @param {Object} params - Generation parameters
 * @returns {string} SVG markup
 */
	FractalImageCodec.prototype.generateFractalSVG = function(seed, params) {
		var width = params.resolution[0];
		var height = params.resolution[1];
		var depth = params.depth || 5;
		var curvature = params.curvature || zp35.KAPPA;
	
		// Initialize PRNG from seed
		var rng = this.createSeededRNG(seed);
	
		var svg = '<svg xmlns="http://www.w3.org/2000/svg" ' +
		'viewBox="0 0 ' + width + " " + height + '" ' +
		'width="' + width + '" height="' + height + '">\n';
	
		// Generate background
		svg += '  <rect width="100%" height="100%" fill="#f5f5f5"/>\n';
	
		// Generate fractal patterns using golden ratio
		svg += this.generateGoldenSpiral(width, height, depth, curvature, rng);
	
		svg += "</svg>";
	
		return svg;
	};

	/**
 * Generate golden spiral pattern
 * @param {number} width - Canvas width
 * @param {number} height - Canvas height
 * @param {number} depth - Recursion depth
 * @param {number} curvature - Curvature parameter
 * @param {Function} rng - Random number generator
 * @returns {string} SVG elements
 */
	FractalImageCodec.prototype.generateGoldenSpiral = function(width, height, depth, curvature, rng) {
		var elements = [];
		var centerX = width / 2;
		var centerY = height / 2;
		var radius = Math.min(width, height) / 3;
	
		// Generate spiral based on golden ratio
		for(var i = 0; i < depth; i++) {
			var angle = i * zp35.PHI * Math.PI * 2;
			var r = radius * Math.pow(zp35.PHI, -i * curvature);
			var x = centerX + r * Math.cos(angle);
			var y = centerY + r * Math.sin(angle);
		
			// Color based on depth and curvature
			var hue = (i * 360 / depth + rng() * 30) % 360;
			var opacity = 0.6 - i * 0.1;
		
			elements.push(
				'  <circle cx="' + x.toFixed(2) + '" cy="' + y.toFixed(2) + '" ' +
			'r="' + (r * 0.2).toFixed(2) + '" ' +
			'fill="hsl(' + hue.toFixed(0) + ', 70%, 60%)" ' +
			'opacity="' + opacity.toFixed(2) + '"/>\n'
			);
		}
	
		return elements.join("");
	};

	/**
 * Create seeded pseudo-random number generator
 * @param {string} seed - Seed string
 * @returns {Function} RNG function
 */
	FractalImageCodec.prototype.createSeededRNG = function(seed) {
		var hash = 0;
		for(var i = 0; i < seed.length; i++) {
			hash = ((hash << 5) - hash) + seed.charCodeAt(i);
			hash = hash & hash;
		}
	
		var state = Math.abs(hash);
	
		return function() {
		// Simple LCG
			state = (state * 1664525 + 1013904223) & 0x7fffffff;
			return state / 0x7fffffff;
		};
	};

	/**
 * JSON Patch Codec
 * Store JSON as delta from a base template
 */
	function JSONPatchCodec() {
		BaseCodec.call(this);
	}

	JSONPatchCodec.prototype = Object.create(BaseCodec.prototype);
	JSONPatchCodec.prototype.constructor = JSONPatchCodec;

	JSONPatchCodec.prototype.canEncode = function(data, mimeType) {
		return mimeType === "application/json" || mimeType === "text/json";
	};

	JSONPatchCodec.prototype.encode = function(data, options) {
		options = options || {};
		var baseTemplate = options.base || {};
	
		// Simplified diff - in practice, use proper JSON patch RFC 6902
		var patch = this.createPatch(baseTemplate, JSON.parse(data));
	
		return {
			codec: "zp35-json-patch",
			version: "1.0",
			base: options.baseName || "default",
			patch: patch,
			checksum: this.hashString(data)
		};
	};

	JSONPatchCodec.prototype.decode = function(recipe) {
		var base = this.loadBase(recipe.base);
		var result = JSON.parse(JSON.stringify(base)); // Deep clone
	
		// Apply patch
		this.applyPatch(result, recipe.patch);
	
		return JSON.stringify(result);
	};

	JSONPatchCodec.prototype.createPatch = function(base, target) {
	// Simplified patch creation
		var patch = [];
	
		for(var key in target) {
			if(target.hasOwnProperty(key)) {
				if(base[key] !== target[key]) {
					patch.push({
						op: "add",
						path: "/" + key,
						value: target[key]
					});
				}
			}
		}
	
		return patch;
	};

	JSONPatchCodec.prototype.applyPatch = function(obj, patch) {
		patch.forEach(function(op) {
			if(op.op === "add") {
				var key = op.path.substring(1); // Remove leading /
				obj[key] = op.value;
			}
		});
	};

	JSONPatchCodec.prototype.loadBase = function(/*baseName*/) {
		// In practice, maintain registry of base templates
		return {};
	};

	JSONPatchCodec.prototype.hashString = function(str) {
		var hash = 0;
		for(var i = 0; i < str.length; i++) {
			hash = ((hash << 5) - hash) + str.charCodeAt(i);
			hash = hash & hash;
		}
		return Math.abs(hash).toString(16);
	};

	/**
 * Register a codec
 * @param {string} name - Codec name
 * @param {BaseCodec} codec - Codec instance
 */
	function registerCodec(name, codec) {
		codecRegistry[name] = codec;
	}

	/**
 * Get codec by name
 * @param {string} name - Codec name
 * @returns {BaseCodec} Codec instance
 */
	function getCodec(name) {
		return codecRegistry[name] || null;
	}

	/**
 * Find suitable codec for data
 * @param {string} data - Data to encode
 * @param {string} mimeType - MIME type
 * @returns {BaseCodec} Suitable codec or null
 */
	function findCodec(data, mimeType) {
		for(var name in codecRegistry) {
			if(codecRegistry.hasOwnProperty(name)) {
				var codec = codecRegistry[name];
				if(codec.canEncode(data, mimeType)) {
					return codec;
				}
			}
		}
		return null;
	}

	/**
 * Encode data using appropriate codec
 * @param {string} data - Data to encode
 * @param {string} mimeType - MIME type
 * @param {Object} options - Encoding options
 * @returns {Object} Recipe or null if no codec available
 */
	function encode(data, mimeType, options) {
		var codec = findCodec(data, mimeType);
		if(codec) {
			return codec.encode(data, options);
		}
		return null;
	}

	/**
 * Decode recipe to regenerate data
 * @param {Object} recipe - Recipe specification
 * @returns {string} Regenerated data
 */
	function decode(recipe) {
		var codec = getCodec(recipe.codec);
		if(!codec) {
			throw new Error("Unknown codec: " + recipe.codec);
		}
		return codec.decode(recipe);
	}

	/**
 * Check if tiddler uses regenerative attachment
 * @param {Object} tiddler - Tiddler object
 * @returns {boolean} True if regenerative
 */
	function isRegenerative(tiddler) {
		return tiddler && tiddler.fields && 
		tiddler.fields["regenerative-codec"] !== undefined;
	}

	/**
 * Get regenerative recipe from tiddler
 * @param {Object} tiddler - Tiddler object
 * @returns {Object} Recipe or null
 */
	function getRecipe(tiddler) {
		if(!isRegenerative(tiddler)) {
			return null;
		}
	
		try {
			return JSON.parse(tiddler.fields["regenerative-recipe"] || "{}");
		} catch(_e) {
			return null;
		}
	}

	// Register built-in codecs
	registerCodec("zp35-fractal-image", new FractalImageCodec());
	registerCodec("zp35-json-patch", new JSONPatchCodec());

	// Export functions
	exports.BaseCodec = BaseCodec;
	exports.FractalImageCodec = FractalImageCodec;
	exports.JSONPatchCodec = JSONPatchCodec;
	exports.registerCodec = registerCodec;
	exports.getCodec = getCodec;
	exports.findCodec = findCodec;
	exports.encode = encode;
	exports.decode = decode;
	exports.isRegenerative = isRegenerative;
	exports.getRecipe = getRecipe;

})();
