/*\
title: test-induce-shadow.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for shadow induction system

\*/

(function() {

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

if($tw.node) {
	describe("Shadow Induction Tests", function() {
		
		var ShadowInducer = require("$:/core/modules/utils/induce-shadow.js").ShadowInducer;
		var ZP35Operator = require("$:/core/modules/utils/zp35-operator.js").ZP35Operator;
		
		var wiki, zp35, inducer;
		
		beforeEach(function() {
			wiki = new $tw.Wiki();
			zp35 = new ZP35Operator();
			inducer = new ShadowInducer(wiki, zp35);
		});
		
		describe("Field Coherence Analysis", function() {
			
			it("should classify crisp fields correctly", function() {
				var tiddler = {
					fields: {
						title: "TestTiddler",
						type: "application/json",
						generator: "testGen",
						tags: ["test"],
						version: "1.0.0"
					}
				};
				
				var analysis = inducer.analyzeFieldCoherence(tiddler);
				
				expect(analysis.crispFields.length).toBeGreaterThan(0);
				expect(analysis.totalFields).toBe(5);
				expect(analysis.curvature).toBeLessThan(1.0);
			});
			
			it("should classify chaotic fields correctly", function() {
				var tiddler = {
					fields: {
						title: "TestTiddler",
						text: "Long random text with lots of entropy and variety...",
						seed: "random-seed-123",
						params: '{"key": "value"}'
					}
				};
				
				var analysis = inducer.analyzeFieldCoherence(tiddler);
				
				expect(analysis.chaoticFields.length).toBeGreaterThan(0);
				expect(analysis.curvature).toBeGreaterThan(0.5);
			});
			
			it("should calculate curvature correctly", function() {
				// All crisp fields -> low curvature
				var crispTiddler = {
					fields: {
						title: "Crisp",
						type: "app",
						generator: "gen",
						version: "1.0"
					}
				};
				
				var crispAnalysis = inducer.analyzeFieldCoherence(crispTiddler);
				expect(crispAnalysis.curvature).toBeLessThan(0.3);
				
				// All chaotic fields -> high curvature
				var chaoticTiddler = {
					fields: {
						text: "chaos",
						seed: "random",
						params: "more chaos"
					}
				};
				
				var chaoticAnalysis = inducer.analyzeFieldCoherence(chaoticTiddler);
				expect(chaoticAnalysis.curvature).toBeGreaterThan(0.7);
			});
			
			it("should handle empty tiddler gracefully", function() {
				var tiddler = {
					fields: {}
				};
				
				var analysis = inducer.analyzeFieldCoherence(tiddler);
				
				expect(analysis.totalFields).toBe(0);
				expect(analysis.curvature).toBe(0);
			});
			
			it("should separate intermediate fields", function() {
				var tiddler = {
					fields: {
						title: "Test",
						type: "application/vnd.tiddlywiki",  // Crisp (longer type)
						modified: "123",                     // Intermediate
						description: "A test description",  // Intermediate
						text: "Some longer chaotic content with variety" // Chaotic
					}
				};
				
				var analysis = inducer.analyzeFieldCoherence(tiddler);
				
				expect(analysis.crispFields.length).toBeGreaterThan(0);
				// At least one of intermediate or chaotic should have content
				expect(analysis.intermediateFields.length + analysis.chaoticFields.length).toBeGreaterThan(0);
			});
			
		});
		
		describe("Kernel Extraction", function() {
			
			it("should extract kernel from crisp fields", function() {
				var crispFields = [
					{name: "title", value: "Test", coherence: 0.95},
					{name: "type", value: "application/json", coherence: 0.90},
					{name: "generator", value: "testGen", coherence: 0.95},
					{name: "tags", value: ["test"], coherence: 0.85}
				];
				
				var kernel = inducer.extractKernel(crispFields);
				
				expect(kernel.requiredFields).toContain("title");
				expect(kernel.requiredFields).toContain("type");
				expect(kernel.requiredFields).toContain("generator");
				expect(kernel.fieldTypes.title).toBe("string");
				expect(kernel.structuralPattern.type).toBe("application/json");
			});
			
			it("should preserve semantic type fields in kernel", function() {
				var crispFields = [
					{name: "type", value: "special-type", coherence: 0.90},
					{name: "generator", value: "fractalGen", coherence: 0.95}
				];
				
				var kernel = inducer.extractKernel(crispFields);
				
				expect(kernel.structuralPattern.type).toBe("special-type");
				expect(kernel.structuralPattern.generator).toBe("fractalGen");
			});
			
			it("should handle empty crisp fields", function() {
				var kernel = inducer.extractKernel([]);
				
				expect(kernel.requiredFields).toEqual([]);
				expect(Object.keys(kernel.fieldTypes).length).toBe(0);
			});
			
		});
		
		describe("Shadow Compiler Generation", function() {
			
			it("should generate shadow compiler from tiddler", function() {
				var tiddler = {
					fields: {
						title: "SourceTiddler",
						type: "application/x-tiddler-regen-zip",
						generator: "fractalGenerator",
						tags: ["graphics"],
						text: "Generate fractal",
						seed: "abc123"
					}
				};
				
				var result = inducer.induceShadowCompiler(tiddler);
				
				expect(result.success).toBe(true);
				expect(result.compiler).toBeDefined();
				expect(result.compiler.fields.title).toContain("$:/shadow/compiler/");
				expect(result.compiler.fields["shadow-source"]).toBe("SourceTiddler");
				expect(result.analysis).toBeDefined();
				expect(result.kernel).toBeDefined();
			});
			
			it("should inherit generator from source", function() {
				var tiddler = {
					fields: {
						title: "Source",
						type: "app",
						generator: "myGenerator",
						tags: []
					}
				};
				
				var result = inducer.induceShadowCompiler(tiddler);
				
				expect(result.success).toBe(true);
				expect(result.compiler.fields.generator).toBe("myGenerator");
			});
			
			it("should tag shadow compiler correctly", function() {
				var tiddler = {
					fields: {
						title: "Source",
						type: "app",
						generator: "gen"
					}
				};
				
				var result = inducer.induceShadowCompiler(tiddler);
				
				expect(result.success).toBe(true);
				var tags = result.compiler.fields.tags;
				expect(tags).toContain("$:/tags/shadow-compiler");
				expect(tags).toContain("compiler");
			});
			
			it("should fail gracefully with no crisp fields", function() {
				var tiddler = {
					fields: {
						text: "only chaotic content",
						seed: "random"
					}
				};
				
				var result = inducer.induceShadowCompiler(tiddler);
				
				expect(result.success).toBe(false);
				expect(result.error).toContain("no crisp fields");
			});
			
			it("should warn on high curvature", function() {
				var tiddler = {
					fields: {
						title: "HighCurvature",
						text: "lots",
						seed: "of",
						params: "chaos",
						data1: "more",
						data2: "more",
						data3: "more",
						data4: "more",
						data5: "more"
					}
				};
				
				spyOn(console, 'warn');
				var result = inducer.induceShadowCompiler(tiddler);
				
				// Should succeed but warn
				expect(result.success).toBe(true);
				expect(console.warn).toHaveBeenCalled();
			});
			
		});
		
		describe("Shadow Compiler Caching", function() {
			
			it("should cache induced shadow compilers", function() {
				var tiddler = {
					fields: {
						title: "Test",
						type: "app",
						generator: "gen"
					}
				};
				
				var result1 = inducer.induceShadowCompiler(tiddler);
				expect(inducer.hasShadowCompiler(tiddler)).toBe(true);
				
				var cached = inducer.getShadowCompiler(tiddler);
				expect(cached).toBeDefined();
				expect(cached.source).toBe("Test");
			});
			
			it("should return cached shadow on subsequent calls", function() {
				var tiddler = {
					fields: {
						title: "Test",
						type: "app",
						generator: "gen"
					}
				};
				
				inducer.induceShadowCompiler(tiddler);
				var cached = inducer.getShadowCompiler(tiddler);
				
				expect(cached).toBeDefined();
				expect(cached.compiler.fields.title).toContain("Test");
			});
			
			it("should clear cache when requested", function() {
				var tiddler = {
					fields: {
						title: "Test",
						type: "app",
						generator: "gen"
					}
				};
				
				inducer.induceShadowCompiler(tiddler);
				expect(inducer.hasShadowCompiler(tiddler)).toBe(true);
				
				inducer.clearCache();
				expect(inducer.hasShadowCompiler(tiddler)).toBe(false);
			});
			
		});
		
		describe("Statistics", function() {
			
			it("should track induction statistics", function() {
				var tiddler1 = {
					fields: {
						title: "Test1",
						type: "app",
						generator: "gen"
					}
				};
				
				var tiddler2 = {
					fields: {
						text: "no structure"
					}
				};
				
				inducer.induceShadowCompiler(tiddler1);
				inducer.induceShadowCompiler(tiddler2);
				
				var stats = inducer.getStatistics();
				
				expect(stats.inductionCount).toBe(2);
				expect(stats.successCount).toBe(1);
				expect(stats.failureCount).toBe(1);
				expect(stats.successRate).toBe(0.5);
			});
			
			it("should track cached shadows", function() {
				var tiddler = {
					fields: {
						title: "Test",
						type: "app",
						generator: "gen"
					}
				};
				
				inducer.induceShadowCompiler(tiddler);
				
				var stats = inducer.getStatistics();
				expect(stats.cachedShadows).toBe(1);
			});
			
		});
		
		describe("Title Sanitization", function() {
			
			it("should sanitize special characters in titles", function() {
				expect(inducer.sanitizeTitle("Test Tiddler")).toBe("Test_Tiddler");
				expect(inducer.sanitizeTitle("Test/Path")).toBe("Test_Path");
				expect(inducer.sanitizeTitle("Test!@#$%")).toBe("Test_____");
			});
			
			it("should preserve alphanumeric characters", function() {
				expect(inducer.sanitizeTitle("Test123")).toBe("Test123");
				expect(inducer.sanitizeTitle("ABC_XYZ")).toBe("ABC_XYZ");
			});
			
		});
		
	});
}

})();
