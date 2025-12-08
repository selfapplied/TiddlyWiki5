/*\
title: test-induce-shadow.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for the Shadow Induction module.

\*/

(function(){

	/*jslint node: true, browser: true */
	/*global $tw: false */
	"use strict";

	describe("Shadow Induction", function() {
	
		var ShadowInducer;
		var ZP35Operator;
		var wiki;
	
		beforeEach(function() {
		// Get required modules
			ShadowInducer = $tw.utils.ShadowInducer;
			ZP35Operator = $tw.utils.ZP35Operator;
			wiki = $tw.wiki;
		
			// Verify modules are loaded
			expect(ShadowInducer).toBeDefined();
			expect(ZP35Operator).toBeDefined();
		});
	
		describe("ShadowInducer Construction", function() {
		
			it("should create shadow inducer instance", function() {
				var zp35 = new ZP35Operator();
				var inducer = new ShadowInducer(wiki, zp35);
			
				expect(inducer).toBeDefined();
				expect(inducer.wiki).toBe(wiki);
				expect(inducer.zp35).toBe(zp35);
				expect(inducer.kappa).toBe(0.35);
			});
		
			it("should have correct coherence thresholds", function() {
				var zp35 = new ZP35Operator();
				var inducer = new ShadowInducer(wiki, zp35);
			
				expect(inducer.CRISP_THRESHOLD).toBe(0.65);
				expect(inducer.CHAOTIC_THRESHOLD).toBe(0.35);
			});
		
		});
	
		describe("Internal Coherence Analysis", function() {
		
			var inducer;
		
			beforeEach(function() {
				var zp35 = new ZP35Operator();
				inducer = new ShadowInducer(wiki, zp35);
			});
		
			it("should separate crisp and chaotic fields", function() {
				var tiddler = {
					fields: {
						title: "TestTiddler",
						type: "text/vnd.tiddlywiki",
						generator: "testGen",
						version: "1.0.0",
						text: "Some random chaotic content here"
					}
				};
			
				var analysis = inducer.analyzeInternalCoherence(tiddler);
			
				expect(analysis).toBeDefined();
				expect(analysis.crispFields).toBeDefined();
				expect(analysis.chaoticFields).toBeDefined();
				expect(analysis.curvatureCoefficient).toBeDefined();
			
				// Structural fields should be crisp
				var crispFieldNames = analysis.crispFields.map(function(f) { return f.name; });
				expect(crispFieldNames).toContain("title");
				expect(crispFieldNames).toContain("type");
				expect(crispFieldNames).toContain("generator");
				expect(crispFieldNames).toContain("version");
			});
		
			it("should identify structural fields", function() {
				expect(inducer.isStructuralField("title")).toBe(true);
				expect(inducer.isStructuralField("type")).toBe(true);
				expect(inducer.isStructuralField("generator")).toBe(true);
				expect(inducer.isStructuralField("version")).toBe(true);
				expect(inducer.isStructuralField("customField")).toBe(false);
			});
		
			it("should calculate field coherence correctly", function() {
				var crispField = inducer.analyzeFieldCoherence("version", "1.0.0");
				var chaoticField = inducer.analyzeFieldCoherence("customData", "asdjkl234lkjasdf89234");
			
				expect(crispField.score).toBeGreaterThan(inducer.CRISP_THRESHOLD);
				expect(chaoticField.score).toBeLessThan(inducer.CRISP_THRESHOLD);
			});
		
			it("should calculate curvature coefficient", function() {
				var analysis = {
					crispFields: [{}, {}, {}],
					chaoticFields: [{}]
				};
			
				var curvature = inducer.calculateCurvatureCoefficient(analysis);
			
				expect(curvature).toBeGreaterThan(0);
				expect(curvature).toBeLessThan(1);
				// More crisp fields = lower curvature
				expect(curvature).toBeLessThan(0.5);
			});
		
		});
	
		describe("Pattern Extraction", function() {
		
			var inducer;
		
			beforeEach(function() {
				var zp35 = new ZP35Operator();
				inducer = new ShadowInducer(wiki, zp35);
			});
		
			it("should extract markdown patterns", function() {
				var text = "# Heading\n\nSome **bold** text and *italic* text.\n\n[[Link]] and {{transclusion}}";
			
				var patterns = inducer.extractPatterns(text);
			
				expect(patterns.length).toBeGreaterThan(0);
			
				var patternTypes = patterns.map(function(p) { return p.type; });
				expect(patternTypes).toContain("heading");
				expect(patternTypes).toContain("bold");
				expect(patternTypes).toContain("italic");
				expect(patternTypes).toContain("link");
				expect(patternTypes).toContain("transclusion");
			});
		
			it("should return empty array for empty text", function() {
				var patterns = inducer.extractPatterns("");
				expect(patterns.length).toBe(0);
			});
		
			it("should detect high entropy strings", function() {
			// High entropy: many different characters with even distribution
				var highEntropy = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEF";
				var lowEntropy = "aaaaaaaaaaaaaaaaaaaa";
			
				expect(inducer.hasHighEntropy(highEntropy)).toBe(true);
				expect(inducer.hasHighEntropy(lowEntropy)).toBe(false);
			});
		
		});
	
		describe("Shadow Compiler Generation", function() {
		
			var inducer;
		
			beforeEach(function() {
				var zp35 = new ZP35Operator();
				inducer = new ShadowInducer(wiki, zp35);
			});
		
			it("should generate shadow compiler with correct structure", function() {
				var tiddler = {
					fields: {
						title: "MyTiddler",
						type: "text/vnd.tiddlywiki",
						generator: "myGen",
						version: "1.0.0",
						text: "Content"
					}
				};
			
				var crispStructure = {
					schema: {
						type: "text/vnd.tiddlywiki",
						generator: "myGen",
						version: "1.0.0"
					},
					stableTokens: [],
					patterns: []
				};
			
				var signature = "0.500000.10";
				var coord = 0.5;
				var height = 10;
			
				var shadow = inducer.generateShadowCompiler(tiddler, crispStructure, signature, coord, height);
			
				expect(shadow).toBeDefined();
				expect(shadow.fields).toBeDefined();
				expect(shadow.fields.title).toBe("MyTiddler-shadow");
				expect(shadow.fields.version).toBe("1.0.0");
				expect(shadow.fields.zp35).toBe(signature);
				expect(shadow.fields["shadow-source"]).toBe("MyTiddler");
				expect(shadow.fields["shadow-type"]).toBe("induced");
				expect(shadow.fields.tags).toContain("$:/tags/ShadowCompiler");
			});
		
			it("should generate descriptive text for shadow compiler", function() {
				var tiddler = {
					fields: {
						title: "MyTiddler",
						type: "text/vnd.tiddlywiki"
					}
				};
			
				var crispStructure = {
					schema: {
						type: "text/vnd.tiddlywiki"
					},
					stableTokens: [],
					patterns: []
				};
			
				var text = inducer.generateShadowCompilerText(tiddler, crispStructure);
			
				expect(text).toContain("Shadow Compiler");
				expect(text).toContain("MyTiddler");
				expect(text).toContain("Extracted Schema");
			});
		
		});
	
		describe("Self-Hosted Program Marking", function() {
		
			var inducer;
		
			beforeEach(function() {
				var zp35 = new ZP35Operator();
				inducer = new ShadowInducer(wiki, zp35);
			});
		
			it("should mark tiddler as self-hosted", function() {
				var tiddler = {
					fields: {
						title: "MyTiddler",
						text: "Content"
					}
				};
			
				var shadowCompiler = {
					fields: {
						title: "MyTiddler-shadow"
					}
				};
			
				var result = inducer.markAsSelfHosted(tiddler, shadowCompiler);
			
				expect(result).toBeDefined();
				expect(result.fields).toBeDefined();
				expect(result.fields.compiler).toBe("MyTiddler-shadow");
				expect(result.fields["program-mode"]).toBe("self-hosted");
				expect(result.fields["shadow-compiler"]).toBe("MyTiddler-shadow");
				expect(result.fields.tags).toContain("$:/tags/SelfHostedProgram");
			});
		
			it("should preserve existing fields", function() {
				var tiddler = {
					fields: {
						title: "MyTiddler",
						text: "Content",
						customField: "customValue"
					}
				};
			
				var shadowCompiler = {
					fields: {
						title: "MyTiddler-shadow"
					}
				};
			
				var result = inducer.markAsSelfHosted(tiddler, shadowCompiler);
			
				expect(result.fields.customField).toBe("customValue");
				expect(result.fields.text).toBe("Content");
			});
		
		});
	
		describe("Full Shadow Induction", function() {
		
			var inducer;
		
			beforeEach(function() {
				var zp35 = new ZP35Operator();
				inducer = new ShadowInducer(wiki, zp35);
			});
		
			it("should perform complete shadow induction", function() {
				var tiddler = {
					fields: {
						title: "TestTiddler",
						type: "text/vnd.tiddlywiki",
						generator: "testGen",
						version: "1.0.0",
						seed: "test-seed",
						tags: ["test"],
						text: "# Heading\n\nSome **content**"
					}
				};
			
				var result = inducer.induceShadow(tiddler);
			
				expect(result.success).toBe(true);
				expect(result.shadowCompiler).toBeDefined();
				expect(result.selfHostedProgram).toBeDefined();
				expect(result.coherenceAnalysis).toBeDefined();
				expect(result.crispStructure).toBeDefined();
				expect(result.signature).toBeDefined();
			
				// Check shadow compiler
				expect(result.shadowCompiler.fields.title).toBe("TestTiddler-shadow");
				expect(result.shadowCompiler.fields["shadow-source"]).toBe("TestTiddler");
			
				// Check self-hosted program
				expect(result.selfHostedProgram.fields.compiler).toBe("TestTiddler-shadow");
				expect(result.selfHostedProgram.fields["program-mode"]).toBe("self-hosted");
			});
		
			it("should handle null tiddler", function() {
				var result = inducer.induceShadow(null);
			
				expect(result.success).toBe(false);
				expect(result.message).toContain("Invalid");
			});
		
		});
	
		describe("Shadow Induction Requirements", function() {
		
			var inducer;
		
			beforeEach(function() {
				var zp35 = new ZP35Operator();
				inducer = new ShadowInducer(wiki, zp35);
			});
		
			it("should identify tiddler needing shadow induction", function() {
				var tiddler = {
					fields: {
						title: "MyTiddler",
						text: "Content",
						tags: ["test"]
					}
				};
			
				expect(inducer.needsShadowInduction(tiddler)).toBe(true);
			});
		
			it("should reject tiddler with existing compiler", function() {
				var tiddler = {
					fields: {
						title: "MyTiddler",
						text: "Content",
						compiler: "SomeCompiler"
					}
				};
			
				expect(inducer.needsShadowInduction(tiddler)).toBe(false);
			});
		
			it("should reject tiddler that is already a compiler", function() {
				var tiddler = {
					fields: {
						title: "MyCompiler",
						generator: "myGen"
					}
				};
			
				expect(inducer.needsShadowInduction(tiddler)).toBe(false);
			});
		
			it("should reject system tiddlers", function() {
				var tiddler = {
					fields: {
						title: "$:/core/SystemTiddler",
						text: "Content"
					}
				};
			
				expect(inducer.needsShadowInduction(tiddler)).toBe(false);
			});
		
			it("should reject tiddler with insufficient structure", function() {
				var tiddler = {
					fields: {
						title: "MyTiddler"
					}
				};
			
				expect(inducer.needsShadowInduction(tiddler)).toBe(false);
			});
		
		});
	
		describe("Seed Generation", function() {
		
			var inducer;
		
			beforeEach(function() {
				var zp35 = new ZP35Operator();
				inducer = new ShadowInducer(wiki, zp35);
			});
		
			it("should generate deterministic seed from title", function() {
				var seed1 = inducer.generateSeed("TestTiddler");
				var seed2 = inducer.generateSeed("TestTiddler");
			
				expect(seed1).toBe(seed2);
				expect(seed1).toContain("shadow-");
			});
		
			it("should generate different seeds for different titles", function() {
				var seed1 = inducer.generateSeed("Tiddler1");
				var seed2 = inducer.generateSeed("Tiddler2");
			
				expect(seed1).not.toBe(seed2);
			});
		
		});
	
	});

})();
