/*\
title: test-zp35-operator.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for the ZP35 Golden Operator

\*/

(function() {

"use strict";

describe("ZP35 Golden Operator", function() {
	var ZP35Operator = $tw.utils.ZP35Operator;
	var ZP35_KAPPA = $tw.utils.ZP35_KAPPA;
	var operator;
	
	beforeEach(function() {
		operator = new ZP35Operator();
	});
	
	describe("Construction", function() {
		it("should create operator instance", function() {
			expect(operator).toBeDefined();
			expect(operator.kappa).toBe(0.35);
		});
		
		it("should have golden ratio constant", function() {
			var expectedPhi = (1 + Math.sqrt(5)) / 2;
			expect(operator.phi).toBeCloseTo(expectedPhi, 10);
		});
		
		it("should export kappa constant", function() {
			expect(ZP35_KAPPA).toBe(0.35);
		});
	});
	
	describe("Ordinal Height Calculation", function() {
		it("should calculate height for simple tiddler", function() {
			var tiddler = {
				fields: {
					title: "Simple",
					text: "Hello"
				}
			};
			
			var height = operator.calculateOrdinalHeight(tiddler);
			expect(height).toBeGreaterThan(0);
			expect(height).toBeLessThanOrEqual(100);
		});
		
		it("should increase with field count", function() {
			var tiddler1 = {
				fields: {
					title: "T1"
				}
			};
			
			var tiddler2 = {
				fields: {
					title: "T2",
					text: "",
					tags: "",
					type: ""
				}
			};
			
			var height1 = operator.calculateOrdinalHeight(tiddler1);
			var height2 = operator.calculateOrdinalHeight(tiddler2);
			
			expect(height2).toBeGreaterThan(height1);
		});
		
		it("should increase with text length", function() {
			var tiddler1 = {
				fields: {
					title: "T1",
					text: "Short"
				}
			};
			
			var tiddler2 = {
				fields: {
					title: "T2",
					text: "A much longer text that contains significantly more content and should result in a higher ordinal height value"
				}
			};
			
			var height1 = operator.calculateOrdinalHeight(tiddler1);
			var height2 = operator.calculateOrdinalHeight(tiddler2);
			
			expect(height2).toBeGreaterThan(height1);
		});
		
		it("should increase with tag count", function() {
			var tiddler1 = {
				fields: {
					title: "T1",
					tags: ["tag1"]
				}
			};
			
			var tiddler2 = {
				fields: {
					title: "T2",
					tags: ["tag1", "tag2", "tag3"]
				}
			};
			
			var height1 = operator.calculateOrdinalHeight(tiddler1);
			var height2 = operator.calculateOrdinalHeight(tiddler2);
			
			expect(height2).toBeGreaterThan(height1);
		});
		
		it("should handle string tags", function() {
			var tiddler = {
				fields: {
					title: "T1",
					tags: "tag1 tag2 tag3"
				}
			};
			
			var height = operator.calculateOrdinalHeight(tiddler);
			expect(height).toBeGreaterThan(0);
		});
		
		it("should increase for special types", function() {
			var tiddler1 = {
				fields: {
					title: "T1",
					type: "text/vnd.tiddlywiki"
				}
			};
			
			var tiddler2 = {
				fields: {
					title: "T2",
					type: "application/javascript"
				}
			};
			
			var height1 = operator.calculateOrdinalHeight(tiddler1);
			var height2 = operator.calculateOrdinalHeight(tiddler2);
			
			expect(height2).toBeGreaterThan(height1);
		});
	});
	
	describe("Cantor Embedding", function() {
		it("should map to [0, 1]", function() {
			for(var i = 0; i <= 100; i += 10) {
				var coord = operator.cantorEmbedding(i);
				expect(coord).toBeGreaterThanOrEqual(0);
				expect(coord).toBeLessThanOrEqual(1);
			}
		});
		
		it("should be monotonic", function() {
			var coords = [];
			for(var i = 0; i <= 50; i += 10) {
				coords.push(operator.cantorEmbedding(i));
			}
			
			for(var j = 1; j < coords.length; j++) {
				expect(coords[j]).toBeGreaterThanOrEqual(coords[j-1]);
			}
		});
		
		it("should create plateaus (quantization)", function() {
			var coord0 = operator.cantorEmbedding(0);
			var coord1 = operator.cantorEmbedding(1);
			
			// Should be different (no single plateau for all values)
			expect(coord0).not.toBe(coord1);
		});
	});
	
	describe("Golden Scaling", function() {
		it("should map to [0, 1]", function() {
			var phi = operator.phi;
			
			for(var i = 0; i <= 1; i += 0.1) {
				var scaled = operator.goldenScale(i, phi);
				expect(scaled).toBeGreaterThanOrEqual(0);
				expect(scaled).toBeLessThan(1);
			}
		});
		
		it("should preserve self-similarity", function() {
			var phi = operator.phi;
			var coord = 0.5;
			
			// Apply scaling multiple times
			var scaled1 = operator.goldenScale(coord, phi);
			var scaled2 = operator.goldenScale(scaled1, phi);
			
			// Results should stay in [0, 1]
			expect(scaled1).toBeGreaterThanOrEqual(0);
			expect(scaled1).toBeLessThan(1);
			expect(scaled2).toBeGreaterThanOrEqual(0);
			expect(scaled2).toBeLessThan(1);
		});
	});
	
	describe("Golden Operator Application", function() {
		it("should map tiddler to fractal coordinate", function() {
			var tiddler = {
				fields: {
					title: "Test",
					text: "Content"
				}
			};
			
			var coord = operator.applyGoldenOperator(tiddler);
			expect(coord).toBeGreaterThanOrEqual(0);
			expect(coord).toBeLessThan(1);
		});
		
		it("should be deterministic", function() {
			var tiddler = {
				fields: {
					title: "Test",
					text: "Content"
				}
			};
			
			var coord1 = operator.applyGoldenOperator(tiddler);
			var coord2 = operator.applyGoldenOperator(tiddler);
			
			expect(coord1).toBe(coord2);
		});
		
		it("should use cache", function() {
			var tiddler = {
				fields: {
					title: "CachedTest",
					text: "Content"
				}
			};
			
			var coord1 = operator.applyGoldenOperator(tiddler);
			
			// Check cache
			expect(operator.coordinateCache["CachedTest"]).toBe(coord1);
			
			var coord2 = operator.applyGoldenOperator(tiddler);
			expect(coord2).toBe(coord1);
		});
		
		it("should handle null tiddler", function() {
			var coord = operator.applyGoldenOperator(null);
			expect(coord).toBe(0);
		});
	});
	
	describe("Coherence Checking", function() {
		it("should allow close tiddlers (safe mode)", function() {
			var tiddler1 = {
				fields: {
					title: "T1",
					text: "Content"
				}
			};
			
			var tiddler2 = {
				fields: {
					title: "T2",
					text: "Content"
				}
			};
			
			var result = operator.checkCoherence(tiddler1, tiddler2);
			
			expect(result.allowed).toBe(true);
			expect(result.distance).toBeLessThan(operator.kappa);
			expect(result.mode).toBe("safe");
			expect(result.confidence).toBeGreaterThan(0);
		});
		
		it("should reject null tiddlers", function() {
			var result = operator.checkCoherence(null, null);
			
			expect(result.allowed).toBe(false);
			expect(result.mode).toBe("error");
		});
		
		it("should provide coordinates in result", function() {
			var tiddler1 = {
				fields: {
					title: "T1",
					text: "Content"
				}
			};
			
			var tiddler2 = {
				fields: {
					title: "T2",
					text: "Different"
				}
			};
			
			var result = operator.checkCoherence(tiddler1, tiddler2);
			
			expect(result.sourceCoord).toBeDefined();
			expect(result.targetCoord).toBeDefined();
			expect(result.distance).toBeDefined();
		});
		
		it("should calculate correct confidence", function() {
			var tiddler1 = {
				fields: {
					title: "T1",
					text: "A"
				}
			};
			
			var tiddler2 = {
				fields: {
					title: "T2",
					text: "A"
				}
			};
			
			var result = operator.checkCoherence(tiddler1, tiddler2);
			
			if(result.mode === "safe") {
				expect(result.confidence).toBe(1.0 - (result.distance / operator.kappa));
			}
		});
	});
	
	describe("Mediation Suggestions", function() {
		it("should generate suggestions", function() {
			var tiddler1 = {
				fields: {
					title: "T1",
					tags: ["common"]
				}
			};
			
			var tiddler2 = {
				fields: {
					title: "T2",
					tags: ["common"]
				}
			};
			
			var suggestions = operator.generateMediationSuggestions(tiddler1, tiddler2);
			
			expect(suggestions).toBeDefined();
			expect(Array.isArray(suggestions)).toBe(true);
			expect(suggestions.length).toBeGreaterThan(0);
		});
		
		it("should suggest common tag when both have tags", function() {
			var tiddler1 = {
				fields: {
					title: "T1",
					tags: ["tag1"]
				}
			};
			
			var tiddler2 = {
				fields: {
					title: "T2",
					tags: ["tag2"]
				}
			};
			
			var suggestions = operator.generateMediationSuggestions(tiddler1, tiddler2);
			
			var hasTagSuggestion = suggestions.some(function(s) {
				return s.type === "common-tag";
			});
			
			expect(hasTagSuggestion).toBe(true);
		});
	});
	
	describe("Alternative Suggestions", function() {
		it("should suggest alternatives for blocked compositions", function() {
			var tiddler1 = {
				fields: {
					title: "T1"
				}
			};
			
			var tiddler2 = {
				fields: {
					title: "T2"
				}
			};
			
			var alternatives = operator.suggestAlternatives(tiddler1, tiddler2);
			
			expect(alternatives).toBeDefined();
			expect(Array.isArray(alternatives)).toBe(true);
			expect(alternatives.length).toBeGreaterThan(0);
		});
		
		it("should include separation suggestion", function() {
			var tiddler1 = {
				fields: {
					title: "T1"
				}
			};
			
			var tiddler2 = {
				fields: {
					title: "T2"
				}
			};
			
			var alternatives = operator.suggestAlternatives(tiddler1, tiddler2);
			
			var hasSeparate = alternatives.some(function(a) {
				return a.type === "separate";
			});
			
			expect(hasSeparate).toBe(true);
		});
	});
	
	describe("ZP35 Signature", function() {
		it("should calculate signature", function() {
			var tiddler = {
				fields: {
					title: "Test",
					text: "Content"
				}
			};
			
			var signature = operator.calculateSignature(tiddler);
			
			expect(signature).toBeDefined();
			expect(typeof signature).toBe("string");
			expect(signature).toMatch(/^\d\.\d+\.\d+$/);
		});
		
		it("should be deterministic", function() {
			var tiddler = {
				fields: {
					title: "Test",
					text: "Content"
				}
			};
			
			var sig1 = operator.calculateSignature(tiddler);
			var sig2 = operator.calculateSignature(tiddler);
			
			expect(sig1).toBe(sig2);
		});
		
		it("should differ for different tiddlers", function() {
			var tiddler1 = {
				fields: {
					title: "T1",
					text: "Content 1"
				}
			};
			
			var tiddler2 = {
				fields: {
					title: "T2",
					text: "Very different content with much more text"
				}
			};
			
			var sig1 = operator.calculateSignature(tiddler1);
			var sig2 = operator.calculateSignature(tiddler2);
			
			expect(sig1).not.toBe(sig2);
		});
	});
	
	describe("Signature Verification", function() {
		it("should verify matching signature", function() {
			var tiddler = {
				fields: {
					title: "Test",
					text: "Content"
				}
			};
			
			var signature = operator.calculateSignature(tiddler);
			var result = operator.verifySignature(tiddler, signature);
			
			expect(result.valid).toBe(true);
			expect(result.distance).toBe(0);
		});
		
		it("should reject mismatched signature", function() {
			var tiddler = {
				fields: {
					title: "Test",
					text: "Content"
				}
			};
			
			var result = operator.verifySignature(tiddler, "0.999999.99");
			
			expect(result.valid).toBe(false);
			expect(result.distance).toBeGreaterThan(0);
		});
		
		it("should provide computed and expected values", function() {
			var tiddler = {
				fields: {
					title: "Test",
					text: "Content"
				}
			};
			
			var result = operator.verifySignature(tiddler, "0.123456.10");
			
			expect(result.computed).toBeDefined();
			expect(result.expected).toBe("0.123456.10");
		});
	});
	
	describe("Cluster Structure Analysis", function() {
		it("should handle empty array", function() {
			var analysis = operator.analyzeClusterStructure([]);
			
			expect(analysis.valid).toBe(true);
			expect(analysis.clusters.length).toBe(0);
		});
		
		it("should handle single tiddler", function() {
			var tiddlers = [
				{
					fields: {
						title: "T1",
						text: "Content"
					}
				}
			];
			
			var analysis = operator.analyzeClusterStructure(tiddlers);
			
			expect(analysis.valid).toBe(true);
		});
		
		it("should identify clusters", function() {
			var tiddlers = [
				{
					fields: {
						title: "T1",
						text: "A"
					}
				},
				{
					fields: {
						title: "T2",
						text: "B"
					}
				},
				{
					fields: {
						title: "T3",
						text: "C very different with much more content to create distance"
					}
				}
			];
			
			var analysis = operator.analyzeClusterStructure(tiddlers);
			
			expect(analysis.valid).toBe(true);
			expect(analysis.clusterCount).toBeGreaterThan(0);
			expect(analysis.clusters).toBeDefined();
		});
		
		it("should include cluster metadata", function() {
			var tiddlers = [
				{
					fields: {
						title: "T1",
						text: "A"
					}
				},
				{
					fields: {
						title: "T2",
						text: "B"
					}
				}
			];
			
			var analysis = operator.analyzeClusterStructure(tiddlers);
			
			if(analysis.clusters.length > 0) {
				var cluster = analysis.clusters[0];
				expect(cluster.size).toBeDefined();
				expect(cluster.minCoord).toBeDefined();
				expect(cluster.maxCoord).toBeDefined();
				expect(cluster.spread).toBeDefined();
				expect(cluster.titles).toBeDefined();
			}
		});
	});
	
	describe("Cache Management", function() {
		it("should clear cache", function() {
			var tiddler = {
				fields: {
					title: "CachedTiddler",
					text: "Content"
				}
			};
			
			operator.applyGoldenOperator(tiddler);
			expect(Object.keys(operator.coordinateCache).length).toBeGreaterThan(0);
			
			operator.clearCache();
			expect(Object.keys(operator.coordinateCache).length).toBe(0);
		});
	});
});

})();
