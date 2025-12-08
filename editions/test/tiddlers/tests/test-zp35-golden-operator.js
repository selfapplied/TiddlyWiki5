/*\
title: test-zp35-golden-operator.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for ZP35 golden operator module

\*/

(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

describe("ZP35 Golden Operator", function() {
	
	var zp35 = require("$:/core/modules/utils/zp35-golden-operator.js");
	
	describe("Constants", function() {
		
		it("should define KAPPA as 0.35", function() {
			expect(zp35.KAPPA).toBe(0.35);
		});
		
		it("should define PHI as golden ratio", function() {
			expect(zp35.PHI).toBeCloseTo(1.618, 3);
		});
		
	});
	
	describe("calculateOrdinalHeight", function() {
		
		it("should return 0 for empty entity", function() {
			var entity = {};
			expect(zp35.calculateOrdinalHeight(entity)).toBe(0);
		});
		
		it("should count transclusions", function() {
			var entity = {
				transclusions: ["A", "B", "C"]
			};
			expect(zp35.calculateOrdinalHeight(entity)).toBe(3);
		});
		
		it("should count multiple features", function() {
			var entity = {
				transclusions: ["A", "B"],
				macros: ["M1"],
				widgets: ["W1", "W2"],
				filters: ["F1"]
			};
			expect(zp35.calculateOrdinalHeight(entity)).toBe(6);
		});
		
		it("should count fields with weight 0.5", function() {
			var entity = {
				fields: { a: 1, b: 2, c: 3, d: 4 }
			};
			expect(zp35.calculateOrdinalHeight(entity)).toBe(2);
		});
		
	});
	
	describe("cantorEmbedding", function() {
		
		it("should return 0 for height 0", function() {
			expect(zp35.cantorEmbedding(0)).toBe(0);
		});
		
		it("should return values in [0,1]", function() {
			for(var i = 0; i < 20; i++) {
				var result = zp35.cantorEmbedding(i);
				expect(result).toBeGreaterThanOrEqual(0);
				expect(result).toBeLessThanOrEqual(1);
			}
		});
		
		it("should be monotonic", function() {
			var prev = 0;
			for(var i = 1; i < 20; i++) {
				var curr = zp35.cantorEmbedding(i);
				expect(curr).toBeGreaterThanOrEqual(prev);
				prev = curr;
			}
		});
		
	});
	
	describe("applyGoldenOperator", function() {
		
		it("should return coordinate in [0,1]", function() {
			var entity = {
				transclusions: ["A", "B"],
				macros: ["M1"]
			};
			var coord = zp35.applyGoldenOperator(entity);
			expect(coord).toBeGreaterThanOrEqual(0);
			expect(coord).toBeLessThanOrEqual(1);
		});
		
		it("should preserve ordering", function() {
			var simpleEntity = {
				transclusions: ["A"]
			};
			var complexEntity = {
				transclusions: ["A", "B", "C"],
				macros: ["M1", "M2"],
				widgets: ["W1"]
			};
			
			var simpleCoord = zp35.applyGoldenOperator(simpleEntity);
			var complexCoord = zp35.applyGoldenOperator(complexEntity);
			
			expect(complexCoord).toBeGreaterThan(simpleCoord);
		});
		
	});
	
	describe("extractFeatureVector", function() {
		
		it("should extract basic structure", function() {
			var entity = {
				transclusions: ["A"],
				macros: ["M1"]
			};
			var vector = zp35.extractFeatureVector(entity);
			
			expect(vector.depth).toBeGreaterThan(0);
			expect(vector.sector).toBeDefined();
			expect(vector.statefulness).toBeDefined();
		});
		
		it("should detect impure entities", function() {
			var entity = {
				hooks: ["hook1"]
			};
			var vector = zp35.extractFeatureVector(entity);
			
			expect(vector.statefulness).toBe("impure");
		});
		
		it("should detect non-idempotent entities", function() {
			var entity = {
				fieldModifications: ["mod1"]
			};
			var vector = zp35.extractFeatureVector(entity);
			
			expect(vector.idempotence).toBe(false);
		});
		
	});
	
	describe("guardianPhi", function() {
		
		it("should return 0 for identical entities", function() {
			var entity = {
				type: "editor"
			};
			var phi = zp35.guardianPhi(entity, entity);
			expect(phi).toBe(0);
		});
		
		it("should detect sector differences", function() {
			var entityA = { type: "editor" };
			var entityB = { type: "storage" };
			
			var phi = zp35.guardianPhi(entityA, entityB);
			expect(phi).toBeGreaterThan(0);
		});
		
		it("should return value in [0,1]", function() {
			var entityA = { type: "editor", hooks: ["h1"] };
			var entityB = { type: "storage" };
			
			var phi = zp35.guardianPhi(entityA, entityB);
			expect(phi).toBeGreaterThanOrEqual(0);
			expect(phi).toBeLessThanOrEqual(1);
		});
		
	});
	
	describe("guardianDelta", function() {
		
		it("should return 0 for identical entities", function() {
			var entity = {};
			var delta = zp35.guardianDelta(entity, entity);
			expect(delta).toBe(0);
		});
		
		it("should detect depth differences", function() {
			var simpleEntity = {
				transclusions: ["A"]
			};
			var complexEntity = {
				transclusions: ["A", "B", "C", "D", "E"],
				macros: ["M1", "M2", "M3"]
			};
			
			var delta = zp35.guardianDelta(simpleEntity, complexEntity);
			expect(delta).toBeGreaterThan(0);
		});
		
		it("should detect hook conflicts", function() {
			var entityA = { hooks: ["h1"] };
			var entityB = { hooks: ["h2"] };
			
			var delta = zp35.guardianDelta(entityA, entityB);
			expect(delta).toBeGreaterThan(0);
		});
		
	});
	
	describe("guardianR", function() {
		
		it("should measure coordinate distance", function() {
			var simpleEntity = {
				transclusions: ["A"]
			};
			var complexEntity = {
				transclusions: ["A", "B", "C"],
				macros: ["M1", "M2"]
			};
			
			var r = zp35.guardianR(simpleEntity, complexEntity);
			expect(r).toBeGreaterThan(0);
		});
		
	});
	
	describe("calculateCompatibility", function() {
		
		it("should mark similar entities as safe", function() {
			var entityA = {
				type: "view",
				transclusions: ["A"]
			};
			var entityB = {
				type: "view",
				transclusions: ["B"]
			};
			
			var result = zp35.calculateCompatibility(entityA, entityB);
			expect(result.compatible).toBe(true);
			expect(result.mode).toBe("safe");
		});
		
		it("should detect incompatible entities", function() {
			var entityA = {
				type: "editor",
				hooks: ["h1", "h2", "h3"],
				fieldModifications: ["f1", "f2"],
				transclusions: Array(20).fill("T")
			};
			var entityB = {
				type: "storage",
				hooks: ["h4", "h5", "h6"],
				fieldModifications: ["f3", "f4"],
				transclusions: ["T1"]
			};
			
			var result = zp35.calculateCompatibility(entityA, entityB);
			expect(result.edgeStrength).toBeGreaterThan(zp35.KAPPA);
		});
		
		it("should return warnings for caution mode", function() {
			var entityA = {
				type: "editor",
				transclusions: ["A", "B", "C"]
			};
			var entityB = {
				type: "storage",
				transclusions: ["D"]
			};
			
			var result = zp35.calculateCompatibility(entityA, entityB);
			if(result.mode === "caution") {
				expect(result.warnings).toBeDefined();
				expect(Array.isArray(result.warnings)).toBe(true);
			}
		});
		
	});
	
	describe("findBridgeMorphism", function() {
		
		it("should find bridge for moderately different entities", function() {
			var entityA = {
				type: "view",
				transclusions: ["A", "B"]
			};
			var entityB = {
				type: "view",
				transclusions: ["C"]
			};
			
			var bridge = zp35.findBridgeMorphism(entityA, entityB);
			expect(bridge.exists).toBe(true);
			expect(bridge.coordinate).toBeGreaterThan(0);
		});
		
		it("should suggest adaptations for conflicts", function() {
			var entityA = {
				type: "view",
				fieldModifications: ["f1"],
				startup: true
			};
			var entityB = {
				type: "view",
				fieldModifications: ["f2"],
				render: true
			};
			
			var bridge = zp35.findBridgeMorphism(entityA, entityB);
			if(bridge.exists) {
				expect(bridge.adaptations).toBeDefined();
				expect(Array.isArray(bridge.adaptations)).toBe(true);
			}
		});
		
		it("should calculate distortion for very different entities", function() {
			var entityA = {
				type: "editor",
				transclusions: Array(50).fill("T"),
				macros: Array(30).fill("M")
			};
			var entityB = {
				type: "storage"
			};
			
			var bridge = zp35.findBridgeMorphism(entityA, entityB);
			// Distortion should be defined and non-negative
			expect(bridge.distortion).toBeGreaterThanOrEqual(0);
			expect(bridge.exists).toBeDefined();
		});
		
	});
	
	describe("Integration: Complete workflow", function() {
		
		it("should analyze, check compatibility, and suggest bridges", function() {
			// Create three plugins
			var pluginA = {
				type: "view",
				transclusions: ["T1", "T2"],
				macros: ["M1"]
			};
			
			var pluginB = {
				type: "view",
				transclusions: ["T3"],
				macros: ["M2"]
			};
			
			var pluginC = {
				type: "storage",
				hooks: ["h1", "h2"],
				fieldModifications: ["f1", "f2"]
			};
			
			// Check compatibility between A and B (should be compatible)
			var compatAB = zp35.calculateCompatibility(pluginA, pluginB);
			expect(compatAB.compatible).toBe(true);
			
			// Check compatibility between A and C (might not be compatible)
			var compatAC = zp35.calculateCompatibility(pluginA, pluginC);
			expect(compatAC.edgeStrength).toBeGreaterThan(0);
			
			// Try to find bridge
			var bridgeAC = zp35.findBridgeMorphism(pluginA, pluginC);
			expect(bridgeAC.exists).toBeDefined();
			expect(bridgeAC.coordinate).toBeGreaterThanOrEqual(0);
		});
		
	});
	
});

})();
