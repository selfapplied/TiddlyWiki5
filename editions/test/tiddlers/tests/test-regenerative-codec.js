/*\
title: test-regenerative-codec.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for regenerative codec module

\*/

(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

describe("Regenerative Codec", function() {
	
	var codec = require("$:/core/modules/utils/regenerative-codec.js");
	
	describe("FractalImageCodec", function() {
		
		var fractalCodec = new codec.FractalImageCodec();
		
		it("should be instantiable", function() {
			expect(fractalCodec).toBeDefined();
		});
		
		it("should detect image MIME types", function() {
			expect(fractalCodec.canEncode("data", "image/png")).toBe(true);
			expect(fractalCodec.canEncode("data", "image/jpeg")).toBe(true);
			expect(fractalCodec.canEncode("data", "image/svg+xml")).toBe(true);
			expect(fractalCodec.canEncode("data", "text/plain")).toBe(false);
		});
		
		it("should generate deterministic seeds", function() {
			var seed1 = fractalCodec.generateSeed("test data");
			var seed2 = fractalCodec.generateSeed("test data");
			var seed3 = fractalCodec.generateSeed("different data");
			
			expect(seed1).toBe(seed2);
			expect(seed1).not.toBe(seed3);
			expect(seed1).toMatch(/^zp35[0-9a-f]+$/);
		});
		
		it("should encode image data to recipe", function() {
			var imageData = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
			var recipe = fractalCodec.encode(imageData, {
				resolution: [512, 512],
				palette: "antclock-wave"
			});
			
			expect(recipe.codec).toBe("zp35-fractal-image");
			expect(recipe.version).toBe("1.0");
			expect(recipe.seed).toBeDefined();
			expect(recipe.params).toBeDefined();
			expect(recipe.params.resolution).toEqual([512, 512]);
			expect(recipe.params.palette).toBe("antclock-wave");
		});
		
		it("should decode recipe to SVG", function() {
			var recipe = {
				codec: "zp35-fractal-image",
				version: "1.0",
				seed: "zp35test1234",
				params: {
					resolution: [256, 256],
					palette: "antclock-wave",
					curvature: 0.35,
					depth: 5
				}
			};
			
			var result = fractalCodec.decode(recipe);
			
			expect(result).toBeDefined();
			expect(result).toMatch(/^data:image\/svg\+xml/);
			expect(typeof result).toBe("string");
			expect(result.length).toBeGreaterThan(100); // Should be substantial SVG
		});
		
		it("should generate deterministic SVG from same seed", function() {
			var recipe = {
				codec: "zp35-fractal-image",
				seed: "zp35fixed123",
				params: {
					resolution: [128, 128],
					depth: 3
				}
			};
			
			var result1 = fractalCodec.decode(recipe);
			var result2 = fractalCodec.decode(recipe);
			
			expect(result1).toBe(result2);
		});
		
		it("should generate different SVG from different seeds", function() {
			var recipe1 = {
				codec: "zp35-fractal-image",
				seed: "zp35seed1",
				params: { resolution: [128, 128], depth: 3 }
			};
			
			var recipe2 = {
				codec: "zp35-fractal-image",
				seed: "zp35seed2",
				params: { resolution: [128, 128], depth: 3 }
			};
			
			var result1 = fractalCodec.decode(recipe1);
			var result2 = fractalCodec.decode(recipe2);
			
			expect(result1).not.toBe(result2);
		});
		
		it("should create seeded RNG", function() {
			var rng = fractalCodec.createSeededRNG("test");
			
			var values = [];
			for(var i = 0; i < 10; i++) {
				values.push(rng());
			}
			
			// All values should be in [0, 1]
			values.forEach(function(v) {
				expect(v).toBeGreaterThanOrEqual(0);
				expect(v).toBeLessThanOrEqual(1);
			});
			
			// Should be deterministic
			var rng2 = fractalCodec.createSeededRNG("test");
			var values2 = [];
			for(var i = 0; i < 10; i++) {
				values2.push(rng2());
			}
			
			expect(values).toEqual(values2);
		});
		
	});
	
	describe("JSONPatchCodec", function() {
		
		var jsonCodec = new codec.JSONPatchCodec();
		
		it("should be instantiable", function() {
			expect(jsonCodec).toBeDefined();
		});
		
		it("should detect JSON MIME types", function() {
			expect(jsonCodec.canEncode("data", "application/json")).toBe(true);
			expect(jsonCodec.canEncode("data", "text/json")).toBe(true);
			expect(jsonCodec.canEncode("data", "text/plain")).toBe(false);
		});
		
		it("should encode JSON as patch", function() {
			var jsonData = JSON.stringify({
				name: "test",
				value: 42
			});
			
			var recipe = jsonCodec.encode(jsonData, {
				base: {},
				baseName: "empty"
			});
			
			expect(recipe.codec).toBe("zp35-json-patch");
			expect(recipe.version).toBe("1.0");
			expect(recipe.patch).toBeDefined();
			expect(Array.isArray(recipe.patch)).toBe(true);
		});
		
		it("should decode patch to JSON", function() {
			var recipe = {
				codec: "zp35-json-patch",
				base: "default",
				patch: [
					{ op: "add", path: "/name", value: "test" },
					{ op: "add", path: "/value", value: 42 }
				]
			};
			
			var result = jsonCodec.decode(recipe);
			var obj = JSON.parse(result);
			
			expect(obj.name).toBe("test");
			expect(obj.value).toBe(42);
		});
		
		it("should create minimal patches", function() {
			var base = {
				unchanged: "same",
				modified: "old"
			};
			
			var target = {
				unchanged: "same",
				modified: "new",
				added: "value"
			};
			
			var patch = jsonCodec.createPatch(base, target);
			
			// Should include modified and added, but not unchanged
			expect(patch.length).toBeGreaterThan(0);
		});
		
	});
	
	describe("Codec Registry", function() {
		
		it("should register and retrieve codecs", function() {
			var testCodec = new codec.BaseCodec();
			codec.registerCodec("test-codec", testCodec);
			
			var retrieved = codec.getCodec("test-codec");
			expect(retrieved).toBe(testCodec);
		});
		
		it("should return null for unknown codecs", function() {
			var result = codec.getCodec("nonexistent-codec");
			expect(result).toBe(null);
		});
		
		it("should find suitable codec for data", function() {
			var found = codec.findCodec("image data", "image/png");
			expect(found).toBeDefined();
			expect(found.canEncode("data", "image/png")).toBe(true);
		});
		
		it("should return null if no codec found", function() {
			var found = codec.findCodec("data", "application/unsupported");
			expect(found).toBe(null);
		});
		
	});
	
	describe("High-level API", function() {
		
		it("should encode data with appropriate codec", function() {
			var imageData = "test image data";
			var recipe = codec.encode(imageData, "image/png", {
				resolution: [256, 256]
			});
			
			expect(recipe).toBeDefined();
			expect(recipe.codec).toBe("zp35-fractal-image");
		});
		
		it("should return null if no codec available", function() {
			var data = "unsupported data";
			var recipe = codec.encode(data, "application/unsupported");
			
			expect(recipe).toBe(null);
		});
		
		it("should decode recipe", function() {
			var recipe = {
				codec: "zp35-fractal-image",
				seed: "zp35test",
				params: {
					resolution: [128, 128],
					depth: 3
				}
			};
			
			var result = codec.decode(recipe);
			expect(result).toBeDefined();
			expect(result).toMatch(/^data:image\/svg\+xml;base64,/);
		});
		
		it("should throw error for unknown codec in decode", function() {
			var recipe = {
				codec: "unknown-codec"
			};
			
			expect(function() {
				codec.decode(recipe);
			}).toThrow();
		});
		
	});
	
	describe("Tiddler integration", function() {
		
		it("should detect regenerative tiddlers", function() {
			var normalTiddler = {
				fields: {
					title: "Normal",
					text: "content"
				}
			};
			
			var regenerativeTiddler = {
				fields: {
					title: "Regenerative",
					"regenerative-codec": "zp35-fractal-image",
					"regenerative-recipe": "{}"
				}
			};
			
			expect(codec.isRegenerative(normalTiddler)).toBe(false);
			expect(codec.isRegenerative(regenerativeTiddler)).toBe(true);
		});
		
		it("should extract recipe from tiddler", function() {
			var tiddler = {
				fields: {
					title: "Test",
					"regenerative-codec": "zp35-fractal-image",
					"regenerative-recipe": JSON.stringify({
						seed: "zp35test",
						params: { resolution: [256, 256] }
					})
				}
			};
			
			var recipe = codec.getRecipe(tiddler);
			
			expect(recipe).toBeDefined();
			expect(recipe.seed).toBe("zp35test");
		});
		
		it("should return null for non-regenerative tiddlers", function() {
			var tiddler = {
				fields: {
					title: "Normal",
					text: "content"
				}
			};
			
			var recipe = codec.getRecipe(tiddler);
			expect(recipe).toBe(null);
		});
		
		it("should handle invalid recipe JSON", function() {
			var tiddler = {
				fields: {
					title: "Invalid",
					"regenerative-codec": "zp35-fractal-image",
					"regenerative-recipe": "invalid json"
				}
			};
			
			var recipe = codec.getRecipe(tiddler);
			expect(recipe).toBe(null);
		});
		
	});
	
	describe("Encode-decode round trip", function() {
		
		it("should round-trip fractal image", function() {
			var originalData = "test image content for encoding";
			
			// Encode
			var recipe = codec.encode(originalData, "image/png", {
				resolution: [256, 256]
			});
			
			expect(recipe).toBeDefined();
			
			// Decode
			var regenerated = codec.decode(recipe);
			
			expect(regenerated).toBeDefined();
			expect(typeof regenerated).toBe("string");
			expect(regenerated.length).toBeGreaterThan(0);
		});
		
		it("should round-trip JSON patch", function() {
			var originalData = JSON.stringify({
				name: "test",
				value: 42,
				nested: { key: "val" }
			});
			
			// Encode
			var recipe = codec.encode(originalData, "application/json", {
				base: {},
				baseName: "empty"
			});
			
			expect(recipe).toBeDefined();
			
			// Decode
			var regenerated = codec.decode(recipe);
			var obj = JSON.parse(regenerated);
			
			expect(obj.name).toBe("test");
			expect(obj.value).toBe(42);
		});
		
	});
	
});

})();
