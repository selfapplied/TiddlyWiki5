/*\
title: test-ce1-harmonic.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for the CE1 harmonic operator system

\*/
(function(){

	/*jslint node: true, browser: true */
	/*global $tw: false */
	"use strict";

	describe("CE1 Harmonic Operator System", function() {

		var ce1 = require("$:/core/modules/utils/ce1-harmonic.js");

		describe("HarmonicOperators", function() {

			it("should compute logarithm for positive numbers", function() {
				var result = ce1.HarmonicOperators.logarithm(Math.E);
				expect(Math.abs(result - 1)).toBeLessThan(0.0001);
			});

			it("should compute logarithm for complex numbers", function() {
				var result = ce1.HarmonicOperators.logarithm({ re: 1, im: 0 });
				expect(Math.abs(result.re - 0)).toBeLessThan(0.0001);
			});

			it("should compute zeta function for s=2", function() {
			// ζ(2) ≈ π²/6 ≈ 1.6449
				var result = ce1.HarmonicOperators.zeta(2);
				expect(result).toBeGreaterThan(1.6);
				expect(result).toBeLessThan(1.7);
			});

			it("should compute zeta function for s=3", function() {
			// ζ(3) ≈ 1.202 (Apéry's constant)
				var result = ce1.HarmonicOperators.zeta(3);
				expect(result).toBeGreaterThan(1.1);
				expect(result).toBeLessThan(1.3);
			});

			it("should compute tangent", function() {
			// tan(0) = 0
				var result = ce1.HarmonicOperators.tangent(0);
				expect(Math.abs(result)).toBeLessThan(0.0001);
			});

			it("should compute sine", function() {
			// sin(0) = 0
				var result = ce1.HarmonicOperators.sine(0);
				expect(Math.abs(result)).toBeLessThan(0.0001);
			
				// sin(1/2) = sin(π/2) = 1
				result = ce1.HarmonicOperators.sine(0.5);
				expect(Math.abs(result - 1)).toBeLessThan(0.0001);
			});

			it("should compute cosine", function() {
			// cos(0) = 1
				var result = ce1.HarmonicOperators.cosine(0);
				expect(Math.abs(result - 1)).toBeLessThan(0.0001);
			
				// cos(1/2) = cos(π/2) = 0
				result = ce1.HarmonicOperators.cosine(0.5);
				expect(Math.abs(result)).toBeLessThan(0.0001);
			});

		});

		describe("harmonicOperator", function() {

			it("should compute full harmonic operator for x=2", function() {
				var result = ce1.harmonicOperator(2);
				expect(result).toBeDefined();
				expect(result.re).toBeDefined();
				expect(result.im).toBeDefined();
				expect(result.components).toBeDefined();
				expect(result.components.boundary).toBeDefined();
				expect(result.components.memory).toBeDefined();
				expect(result.components.morphism).toBeDefined();
				expect(result.components.witness_sin).toBeDefined();
				expect(result.components.witness_cos).toBeDefined();
			});

			it("should have all harmonic components", function() {
				var result = ce1.harmonicOperator(1.5);
				var components = result.components;
			
				expect(typeof components.boundary).toBe("number");
				expect(typeof components.memory).toBe("number");
				expect(isNaN(components.morphism) || typeof components.morphism === "number").toBe(true);
				expect(typeof components.witness_sin).toBe("number");
				expect(typeof components.witness_cos).toBe("number");
			});

		});

		describe("CE1Expression", function() {

			it("should create constant expression with height 0", function() {
				var expr = new ce1.CE1Expression("constant", 5, []);
				expect(expr.type).toBe("constant");
				expect(expr.value).toBe(5);
				expect(expr.height).toBe(0);
			});

			it("should create morphism expression with height 1", function() {
				var child = new ce1.CE1Expression("constant", 3, []);
				var expr = new ce1.CE1Expression("morphism", null, [child]);
				expect(expr.type).toBe("morphism");
				expect(expr.height).toBe(1);
			});

			it("should compute nested heights correctly", function() {
				var c = new ce1.CE1Expression("constant", 1, []);
				var m = new ce1.CE1Expression("morphism", null, [c]);
				var w = new ce1.CE1Expression("witness", null, [m]);
				expect(c.height).toBe(0);
				expect(m.height).toBe(1);
				expect(w.height).toBe(2);
			});

		});

		describe("parseCE1", function() {

			it("should parse constant", function() {
				var expr = ce1.parseCE1("3.14");
				expect(expr.type).toBe("constant");
				expect(expr.value).toBe(3.14);
			});

			it("should parse morphism bracket", function() {
				var expr = ce1.parseCE1("(5)");
				expect(expr.type).toBe("morphism");
				expect(expr.children.length).toBe(1);
			});

			it("should parse witness bracket", function() {
				var expr = ce1.parseCE1("<10>");
				expect(expr.type).toBe("witness");
				expect(expr.children.length).toBe(1);
			});

			it("should parse boundary bracket", function() {
				var expr = ce1.parseCE1("{2}");
				expect(expr.type).toBe("boundary");
				expect(expr.children.length).toBe(1);
			});

			it("should parse memory bracket", function() {
				var expr = ce1.parseCE1("[7]");
				expect(expr.type).toBe("memory");
				expect(expr.children.length).toBe(1);
			});

			it("should parse harmonic operator notation", function() {
				var expr = ce1.parseCE1("H 2");
				expect(expr.type).toBe("harmonic");
				expect(expr.value).toBe(2);
			});

			it("should parse nested brackets", function() {
				var expr = ce1.parseCE1("<(5)>");
				expect(expr.type).toBe("witness");
				expect(expr.children[0].type).toBe("morphism");
			});

		});

		describe("evaluateCE1", function() {

			it("should evaluate constant", function() {
				var expr = ce1.parseCE1("42");
				var result = ce1.evaluateCE1(expr);
				expect(result).toBe(42);
			});

			it("should evaluate harmonic operator", function() {
				var expr = ce1.parseCE1("H 2");
				var result = ce1.evaluateCE1(expr);
				expect(result).toBeDefined();
				expect(result.re).toBeDefined();
				expect(result.im).toBeDefined();
			});

			it("should evaluate boundary bracket", function() {
				var expr = ce1.parseCE1("{2.718}");
				var result = ce1.evaluateCE1(expr);
				// Should compute ln(2.718) ≈ 1
				expect(Math.abs(result - 1)).toBeLessThan(0.01);
			});

			it("should evaluate memory bracket", function() {
				var expr = ce1.parseCE1("[2]");
				var result = ce1.evaluateCE1(expr);
				// Should compute ζ(2) ≈ 1.645
				expect(result).toBeGreaterThan(1.6);
				expect(result).toBeLessThan(1.7);
			});

		});

		describe("fixedPointResolver", function() {

			it("should attempt to find fixed point", function() {
				var result = ce1.fixedPointResolver(0.5, 10, 1e-5);
				expect(result).toBeDefined();
				expect(result.value).toBeDefined();
				expect(result.iterations).toBeDefined();
				expect(result.residual).toBeDefined();
				expect(result.converged).toBeDefined();
			});

			it("should return iteration count", function() {
				var result = ce1.fixedPointResolver(1.5, 50, 1e-8);
				expect(result.iterations).toBeGreaterThanOrEqual(0);
				expect(result.iterations).toBeLessThanOrEqual(50);
			});

		});

		describe("Integration: Full CE1 expression", function() {

			it("should parse and evaluate complete harmonic expression", function() {
			// < H(c) > - find fixed point of harmonic operator
				var expr = ce1.parseCE1("<H 0.5>");
				expect(expr.type).toBe("witness");
				expect(expr.children[0].type).toBe("harmonic");
			
				var result = ce1.evaluateCE1(expr);
				expect(result).toBeDefined();
			});

			it("should handle nested CE1 operators", function() {
				var expr = ce1.parseCE1("{2.718}");
				var result = ce1.evaluateCE1(expr);
				expect(typeof result === "number" || typeof result === "object").toBe(true);
			});

		});

	});

})();
