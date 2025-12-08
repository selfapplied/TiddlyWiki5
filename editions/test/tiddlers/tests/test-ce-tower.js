/*\
title: test-ce-tower.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for CE Tower implementation

\*/

(function() {

	/* global $tw */

	describe("CE Tower", function() {
	
		var CETower;
	
		// Setup
		beforeEach(function() {
			CETower = require("$:/core/modules/utils/ce-tower.js").CETower;
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	Construction and Configuration
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("Construction", function() {
		
			it("should create CE Tower with default parameters", function() {
				var tower = new CETower();
			
				expect(tower).toBeDefined();
				expect(tower.kappa).toBe(0.35);
				expect(tower.spectralTolerance).toBe(0.05);
				expect(tower.flowTolerance).toBe(0.1);
			});
		
			it("should accept custom parameters", function() {
				var tower = new CETower({
					kappa: 0.5,
					spectralTolerance: 0.1,
					flowTolerance: 0.2
				});
			
				expect(tower.kappa).toBe(0.5);
				expect(tower.spectralTolerance).toBe(0.1);
				expect(tower.flowTolerance).toBe(0.2);
			});
		
			it("should initialize statistics", function() {
				var tower = new CETower();
				var stats = tower.getStatistics();
			
				expect(stats.checks.ce1).toBe(0);
				expect(stats.checks.ce2).toBe(0);
				expect(stats.checks.ce3).toBe(0);
				expect(stats.violations.ce1).toBe(0);
				expect(stats.violations.ce2).toBe(0);
				expect(stats.violations.ce3).toBe(0);
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	CE1: Discrete Syntax Layer
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("CE1: Discrete Syntax", function() {
		
			it("should register syntax rules", function() {
				var tower = new CETower();
				var called = false;
			
				tower.registerSyntaxRule("test", function() {
					called = true;
					return { valid: true, depth: 1 };
				});
			
				expect(tower.syntaxRules["test"]).toBeDefined();
			});
		
			it("should check syntax with registered rule", function() {
				var tower = new CETower();
			
				tower.registerSyntaxRule("compose", function(source, target) {
					return {
						valid: true,
						depth: (source.depth || 0) + (target.depth || 0),
						reason: "Valid composition"
					};
				});
			
				var result = tower.checkSyntax("compose", 
					{ depth: 2 },
					{ depth: 3 }
				);
			
				expect(result.valid).toBe(true);
				expect(result.depth).toBe(5);
			});
		
			it("should handle missing syntax rule gracefully", function() {
				var tower = new CETower();
			
				var result = tower.checkSyntax("unknown", 
					{ depth: 1 },
					{ depth: 1 }
				);
			
				expect(result.valid).toBe(true);
				expect(result.reason).toContain("No syntax rule");
			});
		
			it("should catch syntax rule exceptions", function() {
				var tower = new CETower();
			
				tower.registerSyntaxRule("throws", function() {
					throw new Error("Test error");
				});
			
				var result = tower.checkSyntax("throws", {}, {});
			
				expect(result.valid).toBe(false);
				expect(result.reason).toContain("exception");
			});
		
			it("should track CE1 statistics", function() {
				var tower = new CETower();
			
				tower.registerSyntaxRule("valid", function() {
					return { valid: true, depth: 1 };
				});
			
				tower.registerSyntaxRule("invalid", function() {
					return { valid: false, depth: 0 };
				});
			
				tower.checkSyntax("valid", {}, {});
				tower.checkSyntax("invalid", {}, {});
				tower.checkSyntax("valid", {}, {});
			
				var stats = tower.getStatistics();
				expect(stats.checks.ce1).toBe(3);
				expect(stats.violations.ce1).toBe(1);
			});
		
			it("should get depth from various field names", function() {
				var tower = new CETower();
			
				expect(tower.getDepth({ depth: 5 })).toBe(5);
				expect(tower.getDepth({ compositional_depth: 3 })).toBe(3);
				expect(tower.getDepth({ fields: { depth: 7 } })).toBe(7);
				expect(tower.getDepth({})).toBe(0);
				expect(tower.getDepth(null)).toBe(0);
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	CE2: Continuous Flow Layer
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("CE2: Continuous Flow", function() {
		
			it("should check flow compatibility with simple path", function() {
				var tower = new CETower();
			
				var discretePath = [
					{ coordinate: 0.0 },
					{ coordinate: 0.5 },
					{ coordinate: 1.0 }
				];
			
				var geodesic = function(t) {
					return { coordinate: t };
				};
			
				var result = tower.checkFlowCompatibility(discretePath, geodesic, 3);
			
				expect(result.compatible).toBe(true);
				expect(result.curvature).toBeLessThan(0.1);
			});
		
			it("should detect incompatible flow", function() {
				var tower = new CETower({
					kappa: 0.1,
					flowTolerance: 0.05
				});
			
				var discretePath = [
					{ coordinate: 0.0 },
					{ coordinate: 0.9 },  // Big jump!
					{ coordinate: 1.0 }
				];
			
				var geodesic = function(t) {
					return { coordinate: t };
				};
			
				var result = tower.checkFlowCompatibility(discretePath, geodesic, 5);
			
				expect(result.compatible).toBe(false);
				expect(result.maxCurvature).toBeGreaterThan(0.1);
			});
		
			it("should handle short paths", function() {
				var tower = new CETower();
			
				var result = tower.checkFlowCompatibility([{ coordinate: 0 }], function(t) {
					return { coordinate: t };
				});
			
				expect(result.compatible).toBe(true);
				expect(result.reason).toContain("too short");
			});
		
			it("should compute state distance", function() {
				var tower = new CETower();
			
				var d1 = tower.stateDistance(
					{ coordinate: 0.3 },
					{ coordinate: 0.7 }
				);
				expect(d1).toBeCloseTo(0.4, 2);
			
				var d2 = tower.stateDistance(
					{ coherence: 0.8 },
					{ coherence: 0.6 }
				);
				expect(d2).toBeCloseTo(0.2, 2);
			
				var d3 = tower.stateDistance(null, { coordinate: 0.5 });
				expect(d3).toBe(1.0);
			});
		
			it("should check exponential map expressibility", function() {
				var tower = new CETower();
			
				var result = tower.checkExponentialMap("operation", {
					generator: "test"
				});
			
				expect(result.expressible).toBe(true);
				expect(result.approximationError).toBeLessThan(0.1);
			});
		
			it("should track CE2 statistics", function() {
				var tower = new CETower();
			
				tower.checkFlowCompatibility([
					{ coordinate: 0 },
					{ coordinate: 1 }
				], function(t) { return { coordinate: t }; });
			
				tower.checkExponentialMap("op", { generator: "g" });
			
				var stats = tower.getStatistics();
				expect(stats.checks.ce2).toBe(2);
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	CE3: Spectral Witness Layer
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("CE3: Spectral Witness", function() {
		
			it("should compute spectrum from state", function() {
				var tower = new CETower();
			
				var spectrum = tower.computeSpectrum({
					coherence: 0.8,
					coordinate: 0.5,
					fields: { a: 1, b: 2, c: 3 }
				});
			
				expect(spectrum.eigenvalues.length).toBeGreaterThan(0);
				expect(spectrum.dominantMode).toBe(0.8);
				expect(spectrum.coherence).toBe(0.8);
			});
		
			it("should handle state with existing spectrum", function() {
				var tower = new CETower();
			
				var existingSpectrum = {
					eigenvalues: [0.9, 0.7],
					dominantMode: 0.9,
					coherence: 0.9
				};
			
				var spectrum = tower.computeSpectrum({
					spectrum: existingSpectrum
				});
			
				expect(spectrum).toBe(existingSpectrum);
			});
		
			it("should compute spectral distance", function() {
				var tower = new CETower();
			
				var s1 = { eigenvalues: [0.8, 0.5, 0.2] };
				var s2 = { eigenvalues: [0.7, 0.6, 0.3] };
			
				var distance = tower.spectralDistance(s1, s2);
			
				expect(distance).toBeGreaterThan(0);
				expect(distance).toBeLessThan(1);
			});
		
			it("should detect spectral shift", function() {
				var tower = new CETower({
					spectralTolerance: 0.05
				});
			
				var before = { coherence: 0.8 };
				var after = { coherence: 0.85 };
			
				var result = tower.checkSpectralInvariance(before, after, {});
			
				expect(result.preserved).toBe(true);
			});
		
			it("should reject large spectral shift", function() {
				var tower = new CETower({
					spectralTolerance: 0.05
				});
			
				var before = { coherence: 0.8 };
				var after = { coherence: 0.2 };  // Big shift
			
				var result = tower.checkSpectralInvariance(before, after, {});
			
				expect(result.preserved).toBe(false);
				expect(result.shift).toBeGreaterThan(0.05);
			});
		
			it("should check fixed points", function() {
				var tower = new CETower();
			
				var identity = function(x) { return x; };
				var state = { coordinate: 0.5 };
			
				expect(tower.isFixedPoint(state, identity, 0.01)).toBe(true);
			});
		
			it("should detect non-fixed points", function() {
				var tower = new CETower();
			
				var transform = function(x) {
					return { coordinate: (x.coordinate || 0) + 0.1 };
				};
				var state = { coordinate: 0.5 };
			
				expect(tower.isFixedPoint(state, transform, 0.01)).toBe(false);
			});
		
			it("should track CE3 statistics", function() {
				var tower = new CETower();
			
				tower.checkSpectralInvariance(
					{ coherence: 0.8 },
					{ coherence: 0.79 },
					{}
				);
			
				var stats = tower.getStatistics();
				expect(stats.checks.ce3).toBe(1);
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	Unified Validation
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("Unified Validation", function() {
		
			it("should validate full transformation", function() {
				var tower = new CETower();
				tower.initializeStandardRules();
			
				// Create a finely sampled discrete path to match geodesic closely
				var PATH_SAMPLES = 10;
				var discretePath = [];
				for(var i = 0; i <= PATH_SAMPLES; i++) {
					discretePath.push({ coordinate: i / PATH_SAMPLES });
				}
			
				var transformation = {
					operator: "transclude",
					source: { depth: 1 },
					target: { depth: 2 },
					discretePath: discretePath,
					geodesic: function(t) { return { coordinate: t }; },
					beforeState: { coherence: 0.8 },
					afterState: { coherence: 0.82 }
				};
			
				var result = tower.validateTransformation(transformation);
			
				expect(result.valid).toBe(true);
				expect(result.violations.length).toBe(0);
				expect(result.details.ce1).toBeDefined();
				expect(result.details.ce2).toBeDefined();
				expect(result.details.ce3).toBeDefined();
			});
		
			it("should detect CE1 violations", function() {
				var tower = new CETower();
			
				tower.registerSyntaxRule("invalid", function() {
					return { valid: false, reason: "Test violation" };
				});
			
				var transformation = {
					operator: "invalid",
					source: {},
					target: {}
				};
			
				var result = tower.validateTransformation(transformation);
			
				expect(result.valid).toBe(false);
				expect(result.violations.length).toBeGreaterThan(0);
				expect(result.violations[0].layer).toBe("CE1");
			});
		
			it("should detect CE2 violations", function() {
				var tower = new CETower({ kappa: 0.1 });
			
				var transformation = {
					discretePath: [
						{ coordinate: 0 },
						{ coordinate: 0.9 },
						{ coordinate: 1 }
					],
					geodesic: function(t) { return { coordinate: t }; }
				};
			
				var result = tower.validateTransformation(transformation);
			
				expect(result.valid).toBe(false);
				var ce2Violations = result.violations.filter(function(v) {
					return v.layer === "CE2";
				});
				expect(ce2Violations.length).toBeGreaterThan(0);
			});
		
			it("should detect CE3 violations", function() {
				var tower = new CETower({ spectralTolerance: 0.05 });
			
				var transformation = {
					beforeState: { coherence: 0.8 },
					afterState: { coherence: 0.2 }
				};
			
				var result = tower.validateTransformation(transformation);
			
				expect(result.valid).toBe(false);
				var ce3Violations = result.violations.filter(function(v) {
					return v.layer === "CE3";
				});
				expect(ce3Violations.length).toBeGreaterThan(0);
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	Standard Syntax Rules
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("Standard Syntax Rules", function() {
		
			it("should initialize standard rules", function() {
				var tower = new CETower();
				tower.initializeStandardRules();
			
				expect(tower.syntaxRules["transclude"]).toBeDefined();
				expect(tower.syntaxRules["link"]).toBeDefined();
				expect(tower.syntaxRules["macro"]).toBeDefined();
				expect(tower.syntaxRules["widget"]).toBeDefined();
			});
		
			it("should validate transclusion", function() {
				var tower = new CETower();
				tower.initializeStandardRules();
			
				var result = tower.checkSyntax("transclude",
					{ depth: 2 },
					{ depth: 1 }
				);
			
				expect(result.valid).toBe(true);
				expect(result.depth).toBe(4);
			});
		
			it("should reject deep transclusion", function() {
				var tower = new CETower();
				tower.initializeStandardRules();
			
				var result = tower.checkSyntax("transclude",
					{ depth: 8 },
					{ depth: 5 }
				);
			
				expect(result.valid).toBe(false);
				expect(result.reason).toContain("exceeds maximum");
			});
		
			it("should validate links preserve depth", function() {
				var tower = new CETower();
				tower.initializeStandardRules();
			
				var result = tower.checkSyntax("link",
					{ depth: 5 },
					{ depth: 3 }
				);
			
				expect(result.valid).toBe(true);
				expect(result.depth).toBe(5);
			});
		
			it("should validate macro expansion", function() {
				var tower = new CETower();
				tower.initializeStandardRules();
			
				var result = tower.checkSyntax("macro",
					{ depth: 3 },
					{ depth: 2 }
				);
			
				expect(result.valid).toBe(true);
				expect(result.depth).toBe(5);
			});
		
			it("should reject excessive macro depth", function() {
				var tower = new CETower();
				tower.initializeStandardRules();
			
				var result = tower.checkSyntax("macro",
					{ depth: 11 },
					{ depth: 1 }
				);
			
				expect(result.valid).toBe(false);
			});
		
			it("should validate widget rendering", function() {
				var tower = new CETower();
				tower.initializeStandardRules();
			
				var result = tower.checkSyntax("widget",
					{ depth: 2 },
					{ depth: 1 }
				);
			
				expect(result.valid).toBe(true);
				expect(result.depth).toBe(3);
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	Statistics and Utilities
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("Statistics", function() {
		
			it("should track all layer checks", function() {
				var tower = new CETower();
				tower.initializeStandardRules();
			
				tower.checkSyntax("transclude", { depth: 1 }, { depth: 1 });
				tower.checkFlowCompatibility([
					{ coordinate: 0 },
					{ coordinate: 1 }
				], function(t) { return { coordinate: t }; });
				tower.checkSpectralInvariance(
					{ coherence: 0.8 },
					{ coherence: 0.8 },
					{}
				);
			
				var stats = tower.getStatistics();
			
				expect(stats.checks.ce1).toBe(1);
				expect(stats.checks.ce2).toBe(1);
				expect(stats.checks.ce3).toBe(1);
				expect(stats.checks.total).toBe(3);
			});
		
			it("should compute violation rates", function() {
				var tower = new CETower();
			
				tower.registerSyntaxRule("valid", function() {
					return { valid: true, depth: 1 };
				});
				tower.registerSyntaxRule("invalid", function() {
					return { valid: false, depth: 0 };
				});
			
				tower.checkSyntax("valid", {}, {});
				tower.checkSyntax("valid", {}, {});
				tower.checkSyntax("invalid", {}, {});
			
				var stats = tower.getStatistics();
			
				expect(stats.violationRate.ce1).toBeCloseTo(1/3, 2);
			});
		
			it("should reset statistics", function() {
				var tower = new CETower();
				tower.initializeStandardRules();
			
				tower.checkSyntax("transclude", { depth: 1 }, { depth: 1 });
			
				var stats1 = tower.getStatistics();
				expect(stats1.checks.ce1).toBe(1);
			
				tower.resetStatistics();
			
				var stats2 = tower.getStatistics();
				expect(stats2.checks.ce1).toBe(0);
				expect(stats2.violations.ce1).toBe(0);
			});
		
		});
	
	});

})();
