/*\
title: test-renormalization-flow.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for the Renormalization Flow module

\*/

describe("Renormalization Flow", function() {
	
	var RenormalizationFlow = require("$:/core/modules/utils/renormalization-flow.js").RenormalizationFlow;
	var ZP35Operator = require("$:/core/modules/utils/zp35-operator.js").ZP35Operator;
	var ShadowInducer = require("$:/core/modules/utils/induce-shadow.js").ShadowInducer;
	
	var wiki, zp35, shadowInducer, renormFlow;
	
	beforeEach(function() {
		// Create a fresh wiki for each test
		wiki = new $tw.Wiki();
		zp35 = new ZP35Operator();
		shadowInducer = new ShadowInducer(wiki, zp35);
		renormFlow = new RenormalizationFlow(wiki, zp35, shadowInducer);
	});
	
	describe("Constructor", function() {
		it("should initialize with correct default values", function() {
			expect(renormFlow.wiki).toBe(wiki);
			expect(renormFlow.zp35).toBe(zp35);
			expect(renormFlow.shadowInducer).toBe(shadowInducer);
			expect(renormFlow.CONVERGENCE_THRESHOLD).toBe(0.01);
			expect(renormFlow.MAX_ITERATIONS).toBe(10);
		});
	});
	
	describe("Forward Step (Z)", function() {
		it("should map tiddler to ZP35 coordinate", function() {
			var tiddler = {
				fields: {
					title: "TestTiddler",
					type: "text/vnd.tiddlywiki",
					text: "Test content"
				}
			};
			
			var coord = renormFlow.forwardStep(tiddler);
			
			expect(typeof coord).toBe("number");
			expect(coord).toBeGreaterThanOrEqual(0);
			expect(coord).toBeLessThanOrEqual(1);
		});
		
		it("should return 0 for null tiddler", function() {
			var coord = renormFlow.forwardStep(null);
			expect(coord).toBe(0);
		});
		
		it("should produce consistent coordinates for same tiddler", function() {
			var tiddler = {
				fields: {
					title: "ConsistentTest",
					text: "Same content"
				}
			};
			
			var coord1 = renormFlow.forwardStep(tiddler);
			var coord2 = renormFlow.forwardStep(tiddler);
			
			expect(coord1).toBe(coord2);
		});
	});
	
	describe("Inverse Step (Z^-1)", function() {
		it("should reconstruct tiddler from coordinate", function() {
			var seedTiddler = {
				fields: {
					title: "SeedTiddler",
					type: "text/vnd.tiddlywiki",
					text: "Original content with some extra details",
					tags: "tag1 tag2 tag3",
					customField: "custom value"
				}
			};
			
			var coord = renormFlow.forwardStep(seedTiddler);
			var result = renormFlow.inverseStep(coord, seedTiddler);
			
			expect(result.success).toBe(true);
			expect(result.tiddler).toBeDefined();
			expect(result.tiddler.fields.title).toBe("SeedTiddler");
		});
		
		it("should produce tiddler with minimal complexity", function() {
			var seedTiddler = {
				fields: {
					title: "VerboseTiddler",
					type: "text/vnd.tiddlywiki",
					text: "A very long text with lots of redundant information that could be compressed significantly without losing the essential meaning",
					tags: "tag1 tag2 tag3 redundant1 redundant2",
					creator: "test",
					modifier: "test",
					created: "20231201",
					modified: "20231202",
					customField1: "value1",
					customField2: "value2"
				}
			};
			
			var originalComplexity = renormFlow.calculateBracketComplexity(seedTiddler);
			var coord = renormFlow.forwardStep(seedTiddler);
			var result = renormFlow.inverseStep(coord, seedTiddler);
			
			expect(result.success).toBe(true);
			
			var reconstructedComplexity = renormFlow.calculateBracketComplexity(result.tiddler);
			expect(reconstructedComplexity).toBeLessThan(originalComplexity);
		});
		
		it("should preserve coordinate (invariance)", function() {
			var seedTiddler = {
				fields: {
					title: "InvarianceTest",
					text: "Test content for invariance"
				}
			};
			
			var originalCoord = renormFlow.forwardStep(seedTiddler);
			var result = renormFlow.inverseStep(originalCoord, seedTiddler);
			var reconstructedCoord = renormFlow.forwardStep(result.tiddler);
			
			// Allow small numerical error
			expect(Math.abs(reconstructedCoord - originalCoord)).toBeLessThan(0.02);
		});
		
		it("should fail gracefully with null seed", function() {
			var result = renormFlow.inverseStep(0.5, null);
			
			expect(result.success).toBe(false);
			expect(result.error).toBeDefined();
		});
	});
	
	describe("Bracket Complexity", function() {
		it("should calculate complexity for simple tiddler", function() {
			var tiddler = {
				fields: {
					title: "Simple",
					text: "Short text"
				}
			};
			
			var complexity = renormFlow.calculateBracketComplexity(tiddler);
			expect(complexity).toBeGreaterThan(0);
		});
		
		it("should give higher complexity for verbose tiddlers", function() {
			var simple = {
				fields: {
					title: "Simple",
					text: "Short"
				}
			};
			
			var verbose = {
				fields: {
					title: "Verbose",
					text: "A".repeat(1000),
					tags: "tag1 tag2 tag3 tag4 tag5",
					field1: "value1",
					field2: "value2",
					field3: "value3",
					creator: "test",
					modifier: "test"
				}
			};
			
			var simpleComplexity = renormFlow.calculateBracketComplexity(simple);
			var verboseComplexity = renormFlow.calculateBracketComplexity(verbose);
			
			expect(verboseComplexity).toBeGreaterThan(simpleComplexity);
		});
		
		it("should return 0 for null tiddler", function() {
			var complexity = renormFlow.calculateBracketComplexity(null);
			expect(complexity).toBe(0);
		});
	});
	
	describe("Full Renormalization Cycle", function() {
		it("should converge to canonical form", function() {
			var tiddler = {
				fields: {
					title: "ToRenormalize",
					type: "text/vnd.tiddlywiki",
					text: "Some content with patterns and structure",
					tags: "tag1 tag2",
					customField: "value"
				}
			};
			
			var result = renormFlow.renormalize(tiddler);
			
			expect(result.success).toBe(true);
			expect(result.converged).toBe(true);
			expect(result.canonicalForm).toBeDefined();
		});
		
		it("should reduce complexity through renormalization", function() {
			var tiddler = {
				fields: {
					title: "ComplexTiddler",
					type: "text/vnd.tiddlywiki",
					text: "Long text with lots of information that could be compressed",
					tags: "tag1 tag2 tag3 tag4",
					field1: "value1",
					field2: "value2",
					creator: "test",
					modifier: "test"
				}
			};
			
			var result = renormFlow.renormalize(tiddler);
			
			expect(result.success).toBe(true);
			expect(result.complexityReduction).toBeGreaterThan(0);
			expect(result.finalComplexity).toBeLessThan(result.initialComplexity);
		});
		
		it("should preserve coordinate invariance", function() {
			var tiddler = {
				fields: {
					title: "InvarianceTest",
					text: "Content for invariance testing"
				}
			};
			
			var result = renormFlow.renormalize(tiddler);
			
			expect(result.success).toBe(true);
			expect(result.coordinateInvariance).toBe(true);
			expect(result.coordinateDrift).toBeLessThan(0.001);
		});
		
		it("should handle already minimal tiddlers", function() {
			var tiddler = {
				fields: {
					title: "Minimal",
					text: "Short"
				}
			};
			
			var result = renormFlow.renormalize(tiddler);
			
			expect(result.success).toBe(true);
			// Should converge quickly for already minimal tiddlers
			expect(result.iterations).toBeLessThanOrEqual(2);
		});
		
		it("should respect maximum iterations", function() {
			var tiddler = {
				fields: {
					title: "TestTiddler",
					text: "Content"
				}
			};
			
			var result = renormFlow.renormalize(tiddler, { maxIterations: 3 });
			
			expect(result.success).toBe(true);
			expect(result.iterations).toBeLessThanOrEqual(3);
		});
		
		it("should record iteration history", function() {
			var tiddler = {
				fields: {
					title: "HistoryTest",
					text: "Content for history tracking"
				}
			};
			
			var result = renormFlow.renormalize(tiddler);
			
			expect(result.success).toBe(true);
			expect(result.iterationHistory).toBeDefined();
			expect(result.iterationHistory.length).toBeGreaterThan(0);
			expect(result.iterationHistory[0].iteration).toBe(0);
		});
		
		it("should mark canonical form with metadata", function() {
			var tiddler = {
				fields: {
					title: "MetadataTest",
					text: "Content"
				}
			};
			
			var result = renormFlow.renormalize(tiddler);
			
			expect(result.success).toBe(true);
			expect(result.canonicalForm.fields.renormalized).toBe("true");
			expect(result.canonicalForm.fields["renorm-source"]).toBe("MetadataTest");
			expect(result.canonicalForm.fields["renorm-coord"]).toBeDefined();
		});
	});
	
	describe("Canonical Form Detection", function() {
		it("should detect canonical tiddlers", function() {
			var canonical = {
				fields: {
					title: "Canonical",
					text: "Minimal content",
					renormalized: "true"
				}
			};
			
			var isCanonical = renormFlow.isCanonical(canonical);
			expect(isCanonical).toBe(true);
		});
		
		it("should detect non-canonical tiddlers", function() {
			var nonCanonical = {
				fields: {
					title: "NonCanonical",
					text: "Long verbose content with lots of redundant information",
					tags: "tag1 tag2 tag3 tag4 tag5",
					field1: "value1",
					field2: "value2"
				}
			};
			
			// This might take a moment as it does a test renormalization
			var isCanonical = renormFlow.isCanonical(nonCanonical);
			expect(isCanonical).toBe(false);
		});
	});
	
	describe("Batch Renormalization", function() {
		it("should renormalize multiple tiddlers", function() {
			var tiddlers = [
				{
					fields: {
						title: "Tiddler1",
						text: "Content 1"
					}
				},
				{
					fields: {
						title: "Tiddler2",
						text: "Content 2"
					}
				},
				{
					fields: {
						title: "Tiddler3",
						text: "Content 3"
					}
				}
			];
			
			var result = renormFlow.renormalizeBatch(tiddlers);
			
			expect(result.totalTiddlers).toBe(3);
			expect(result.successCount).toBeGreaterThan(0);
			expect(result.results.length).toBe(3);
		});
		
		it("should calculate batch statistics", function() {
			var tiddlers = [
				{
					fields: {
						title: "Batch1",
						text: "Content with some verbosity",
						tags: "tag1 tag2"
					}
				},
				{
					fields: {
						title: "Batch2",
						text: "More verbose content here",
						field1: "value1"
					}
				}
			];
			
			var result = renormFlow.renormalizeBatch(tiddlers);
			
			expect(result.totalComplexityReduction).toBeGreaterThanOrEqual(0);
			expect(result.averageReduction).toBeGreaterThanOrEqual(0);
		});
		
		it("should handle empty batch", function() {
			var result = renormFlow.renormalizeBatch([]);
			
			expect(result.totalTiddlers).toBe(0);
			expect(result.successCount).toBe(0);
		});
	});
	
	describe("Tag Minimalization", function() {
		it("should remove duplicate tags", function() {
			var tags = ["tag1", "tag2", "tag1", "tag3"];
			var minimalized = renormFlow.minimalizeTagSet(tags);
			
			expect(minimalized.length).toBe(3);
			expect(minimalized.indexOf("tag1")).not.toBe(-1);
			expect(minimalized.indexOf("tag2")).not.toBe(-1);
			expect(minimalized.indexOf("tag3")).not.toBe(-1);
		});
		
		it("should handle string tags", function() {
			var tags = "tag1 tag2 tag3";
			var minimalized = renormFlow.minimalizeTagSet(tags);
			
			expect(Array.isArray(minimalized)).toBe(true);
			expect(minimalized.length).toBe(3);
		});
		
		it("should handle null tags", function() {
			var minimalized = renormFlow.minimalizeTagSet(null);
			expect(minimalized.length).toBe(0);
		});
	});
	
	describe("Minimal Text Generation", function() {
		it("should preserve short text", function() {
			var crispStructure = {
				schema: {},
				patterns: []
			};
			var seedTiddler = {
				fields: {
					title: "Short",
					text: "Brief"
				}
			};
			
			var minimalText = renormFlow.generateMinimalText(crispStructure, seedTiddler);
			expect(minimalText).toBe("Brief");
		});
		
		it("should generate placeholder for empty patterns", function() {
			var crispStructure = {
				schema: {},
				patterns: []
			};
			var seedTiddler = {
				fields: {
					title: "Test",
					text: "A".repeat(500)
				}
			};
			
			var minimalText = renormFlow.generateMinimalText(crispStructure, seedTiddler);
			expect(minimalText).toContain("Test");
			expect(minimalText).toContain("Canonical form");
		});
	});
	
	describe("Integration with ZP35 and Shadow Induction", function() {
		it("should use ZP35 for coordinate mapping", function() {
			var tiddler = {
				fields: {
					title: "ZP35Test",
					text: "Content"
				}
			};
			
			var coord1 = renormFlow.forwardStep(tiddler);
			var coord2 = zp35.applyGoldenOperator(tiddler);
			
			expect(coord1).toBe(coord2);
		});
		
		it("should use shadow inducer for structure extraction", function() {
			var tiddler = {
				fields: {
					title: "ShadowTest",
					type: "text/vnd.tiddlywiki",
					text: "Content with structure"
				}
			};
			
			var result = renormFlow.renormalize(tiddler);
			
			expect(result.success).toBe(true);
			// The renormalization uses shadow inducer internally
			// Just verify it completes successfully
		});
	});
	
	describe("Error Handling", function() {
		it("should handle null tiddler gracefully", function() {
			var result = renormFlow.renormalize(null);
			
			expect(result.success).toBe(false);
			expect(result.message).toBeDefined();
		});
		
		it("should handle tiddler without fields", function() {
			var result = renormFlow.renormalize({});
			
			// Should fail or handle gracefully
			expect(result.success).toBe(false);
		});
	});
});
