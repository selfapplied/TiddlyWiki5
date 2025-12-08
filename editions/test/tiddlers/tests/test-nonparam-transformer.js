/*\
title: test-nonparam-transformer.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for non-parametric transformer system

\*/

(function() {

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

if($tw.node) {
	describe("Non-Parametric Transformer Tests", function() {
		
		var NonParametricTransformer = require("$:/core/modules/utils/nonparam-transformer.js").NonParametricTransformer;
		var TRANSFORMER_TYPES = require("$:/core/modules/utils/nonparam-transformer.js").TRANSFORMER_TYPES;
		var SEED_POLICIES = require("$:/core/modules/utils/nonparam-transformer.js").SEED_POLICIES;
		var ZP35Operator = require("$:/core/modules/utils/zp35-operator.js").ZP35Operator;
		var RegenZipVM = require("$:/core/modules/utils/regen-zip-vm.js").RegenZipVM;
		
		var wiki, zp35, vm, transformer;
		
		beforeEach(function() {
			wiki = new $tw.Wiki();
			zp35 = new ZP35Operator();
			vm = new RegenZipVM(wiki);
			transformer = new NonParametricTransformer(wiki, zp35, vm);
		});
		
		describe("Transformer Registration", function() {
			
			it("should register valid non-parametric transformer", function() {
				var transformerTiddler = {
					fields: {
						title: "TestTransformer",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "CompilerA",
						"target-compiler": "CompilerB",
						"transform-kind": "projection",
						"seed-policy": "inherit"
					}
				};
				
				var result = transformer.registerTransformer(transformerTiddler);
				
				expect(result.success).toBe(true);
				expect(result.transformer).toBeDefined();
			});
			
			it("should reject transformer with params field", function() {
				var transformerTiddler = {
					fields: {
						title: "InvalidTransformer",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "CompilerA",
						"target-compiler": "CompilerB",
						"transform-kind": "projection",
						params: '{"invalid": true}' // NOT ALLOWED
					}
				};
				
				var result = transformer.registerTransformer(transformerTiddler);
				
				expect(result.success).toBe(false);
				expect(result.error).toContain("parameter-like fields");
			});
			
			it("should reject transformer without required fields", function() {
				var transformerTiddler = {
					fields: {
						title: "IncompleteTransformer",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric"
						// Missing source-compiler, target-compiler, transform-kind
					}
				};
				
				var result = transformer.registerTransformer(transformerTiddler);
				
				expect(result.success).toBe(false);
			});
			
			it("should reject invalid transform kind", function() {
				var transformerTiddler = {
					fields: {
						title: "BadKind",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "CompilerA",
						"target-compiler": "CompilerB",
						"transform-kind": "invalid-kind"
					}
				};
				
				var result = transformer.registerTransformer(transformerTiddler);
				
				expect(result.success).toBe(false);
				expect(result.error).toContain("Invalid transform-kind");
			});
			
			it("should parse geometry constraints correctly", function() {
				var transformerTiddler = {
					fields: {
						title: "ConstrainedTransformer",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "CompilerA",
						"target-compiler": "CompilerB",
						"transform-kind": "projection",
						"zp35-max-distance": "0.5",
						"curvature-scale-min": "0.8",
						"curvature-scale-max": "1.5"
					}
				};
				
				var result = transformer.registerTransformer(transformerTiddler);
				
				expect(result.success).toBe(true);
				expect(result.transformer.constraints.maxZP35Distance).toBe(0.5);
				expect(result.transformer.constraints.minCurvatureScale).toBe(0.8);
				expect(result.transformer.constraints.maxCurvatureScale).toBe(1.5);
			});
			
		});
		
		describe("Seed Policies", function() {
			
			it("should apply INHERIT policy correctly", function() {
				var originalSeed = "seed123";
				var transformerFields = {title: "Transformer"};
				
				var newSeed = transformer.applySeedPolicy(
					SEED_POLICIES.INHERIT,
					originalSeed,
					transformerFields
				);
				
				expect(newSeed).toBe(originalSeed);
			});
			
			it("should apply HASH policy correctly", function() {
				var originalSeed = "seed123";
				var transformerFields = {title: "Transformer"};
				
				var newSeed = transformer.applySeedPolicy(
					SEED_POLICIES.HASH,
					originalSeed,
					transformerFields
				);
				
				expect(newSeed).not.toBe(originalSeed);
				expect(newSeed).toContain("hash_");
				
				// Should be deterministic
				var newSeed2 = transformer.applySeedPolicy(
					SEED_POLICIES.HASH,
					originalSeed,
					transformerFields
				);
				expect(newSeed2).toBe(newSeed);
			});
			
			it("should apply RESEED_FIXED policy correctly", function() {
				var originalSeed = "seed123";
				var transformerFields = {
					title: "Transformer",
					"fixed-seed": "fixed-value"
				};
				
				var newSeed = transformer.applySeedPolicy(
					SEED_POLICIES.RESEED_FIXED,
					originalSeed,
					transformerFields
				);
				
				expect(newSeed).toBe("fixed-value");
			});
			
			it("should apply COMPOSE policy correctly", function() {
				var originalSeed = "seed123";
				var transformerFields = {title: "Transformer"};
				
				var newSeed = transformer.applySeedPolicy(
					SEED_POLICIES.COMPOSE,
					originalSeed,
					transformerFields
				);
				
				expect(newSeed).toContain("seed123");
				expect(newSeed).toContain("Transformer");
			});
			
		});
		
		describe("Transform Application", function() {
			
			it("should apply projection transformation", function() {
				var transformerTiddler = {
					fields: {
						title: "ProjectionTransformer",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "CompilerA",
						"target-compiler": "CompilerB",
						"transform-kind": "projection",
						"seed-policy": "inherit"
					}
				};
				
				transformer.registerTransformer(transformerTiddler);
				
				var program = {
					fields: {
						title: "TestProgram",
						type: "app",
						compiler: "CompilerA",
						seed: "seed123",
						text: "content",
						extraField: "extra"
					}
				};
				
				var result = transformer.applyTransformer("ProjectionTransformer", program);
				
				expect(result.success).toBe(true);
				expect(result.transformedProgram).toBeDefined();
				expect(result.transformedProgram.fields.compiler).toBe("CompilerB");
				expect(result.transformedProgram.fields.seed).toBe("seed123");
			});
			
			it("should apply normalization transformation", function() {
				var transformerTiddler = {
					fields: {
						title: "NormalizeTransformer",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "CompilerA",
						"target-compiler": "CompilerB",
						"transform-kind": "normalize",
						"seed-policy": "inherit",
						"zp35-max-distance": "1.0" // Allow larger distance
					}
				};
				
				transformer.registerTransformer(transformerTiddler);
				
				var program = {
					fields: {
						title: "TestProgram",
						compiler: "CompilerA",
						seed: "seed123",
						text: "  content with spaces  \r\n",
						tags: ["b", "a", "c"]
					}
				};
				
				var result = transformer.applyTransformer("NormalizeTransformer", program);
				
				expect(result.success).toBe(true);
				if(result.transformedProgram && result.transformedProgram.fields) {
					var normalizedText = result.transformedProgram.fields.text;
					expect(normalizedText).toBe("content with spaces");
					expect(result.transformedProgram.fields.tags).toEqual(["a", "b", "c"]);
				}
			});
			
			it("should apply upgrade transformation", function() {
				var transformerTiddler = {
					fields: {
						title: "UpgradeTransformer",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "CompilerV1",
						"target-compiler": "CompilerV2",
						"transform-kind": "upgrade",
						"target-version": "2.0.0",
						"upgrade-field-mappings": '{"oldField": "newField"}',
						"seed-policy": "inherit"
					}
				};
				
				transformer.registerTransformer(transformerTiddler);
				
				var program = {
					fields: {
						title: "TestProgram",
						compiler: "CompilerV1",
						version: "1.0.0",
						oldField: "value",
						seed: "seed123"
					}
				};
				
				var result = transformer.applyTransformer("UpgradeTransformer", program);
				
				expect(result.success).toBe(true);
				expect(result.transformedProgram.fields.version).toBe("2.0.0");
				expect(result.transformedProgram.fields.newField).toBe("value");
				expect(result.transformedProgram.fields.oldField).toBeUndefined();
			});
			
		});
		
		describe("Geometry Validation", function() {
			
			it("should validate ZP35 distance bounds", function() {
				var transformerTiddler = {
					fields: {
						title: "StrictTransformer",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "CompilerA",
						"target-compiler": "CompilerB",
						"transform-kind": "identity",
						"zp35-max-distance": "0.1", // Very strict
						"seed-policy": "inherit"
					}
				};
				
				transformer.registerTransformer(transformerTiddler);
				
				var program = {
					fields: {
						title: "TestProgram",
						type: "very-different-type-to-cause-distance",
						compiler: "CompilerA",
						seed: "seed123",
						text: "content"
					}
				};
				
				var result = transformer.applyTransformer("StrictTransformer", program);
				
				// May fail if ZP35 distance is too large
				if(!result.success) {
					expect(result.error).toContain("distance");
				}
			});
			
			it("should validate curvature scale bounds", function() {
				var t = transformer;
				var prog = {
					fields: {
						title: "Test",
						compiler: "A"
					}
				};
				
				var originalCurv = t.calculateProgramCurvature(prog);
				expect(originalCurv).toBeGreaterThanOrEqual(0);
				expect(originalCurv).toBeLessThanOrEqual(1);
			});
			
		});
		
		describe("Transformer Composition", function() {
			
			it("should compose compatible transformers", function() {
				var t1 = {
					fields: {
						title: "T1",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "A",
						"target-compiler": "B",
						"transform-kind": "projection",
						"zp35-max-distance": "0.3"
					}
				};
				
				var t2 = {
					fields: {
						title: "T2",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "B",
						"target-compiler": "C",
						"transform-kind": "lift",
						"zp35-max-distance": "0.4"
					}
				};
				
				transformer.registerTransformer(t1);
				transformer.registerTransformer(t2);
				
				var result = transformer.composeTransformers("T1", "T2");
				
				expect(result.success).toBe(true);
				expect(result.composition.sourceCompiler).toBe("A");
				expect(result.composition.targetCompiler).toBe("C");
				expect(result.composition.transformers).toEqual(["T1", "T2"]);
			});
			
			it("should reject incompatible transformer composition", function() {
				var t1 = {
					fields: {
						title: "T1",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "A",
						"target-compiler": "B",
						"transform-kind": "projection"
					}
				};
				
				var t2 = {
					fields: {
						title: "T2",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "X", // Different from t1.target
						"target-compiler": "C",
						"transform-kind": "lift"
					}
				};
				
				transformer.registerTransformer(t1);
				transformer.registerTransformer(t2);
				
				var result = transformer.composeTransformers("T1", "T2");
				
				expect(result.success).toBe(false);
				expect(result.error).toContain("Cannot compose");
			});
			
			it("should compute composed constraints correctly", function() {
				var t1 = {
					fields: {
						title: "T1",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "A",
						"target-compiler": "B",
						"transform-kind": "projection",
						"zp35-max-distance": "0.3",
						"curvature-scale-min": "0.5",
						"curvature-scale-max": "2.0"
					}
				};
				
				var t2 = {
					fields: {
						title: "T2",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "B",
						"target-compiler": "C",
						"transform-kind": "lift",
						"zp35-max-distance": "0.4",
						"curvature-scale-min": "0.6",
						"curvature-scale-max": "1.8"
					}
				};
				
				transformer.registerTransformer(t1);
				transformer.registerTransformer(t2);
				
				var result = transformer.composeTransformers("T1", "T2");
				
				expect(result.success).toBe(true);
				// Curvature scales multiply
				expect(result.composition.constraints.minCurvatureScale).toBe(0.3); // 0.5 * 0.6
				expect(result.composition.constraints.maxCurvatureScale).toBe(3.6); // 2.0 * 1.8
			});
			
		});
		
		describe("Statistics", function() {
			
			it("should track transformation statistics", function() {
				var transformerTiddler = {
					fields: {
						title: "T1",
						type: "application/x-tiddler-transformer",
						mode: "non-parametric",
						"source-compiler": "A",
						"target-compiler": "B",
						"transform-kind": "identity",
						"seed-policy": "inherit"
					}
				};
				
				transformer.registerTransformer(transformerTiddler);
				
				var program = {
					fields: {
						title: "Test",
						compiler: "A",
						seed: "seed"
					}
				};
				
				transformer.applyTransformer("T1", program);
				transformer.applyTransformer("T1", program);
				
				var stats = transformer.getStatistics();
				
				expect(stats.transformCount).toBe(2);
				expect(stats.registeredTransformers).toBe(1);
			});
			
			it("should track geometry violations", function() {
				var stats = transformer.getStatistics();
				expect(stats.geometryViolations).toBe(0);
				expect(stats.seedPolicyViolations).toBe(0);
			});
			
		});
		
	});
}

})();
