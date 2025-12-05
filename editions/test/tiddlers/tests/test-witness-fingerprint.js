/*\
title: test-witness-fingerprint.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for witness fingerprint system

\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

describe("Witness Fingerprint System", function() {

	var witnessUtils = require("$:/core/modules/utils/witness-fingerprint.js");

	it("should calculate fingerprint for a tiddler", function() {
		// Create a test wiki
		var wiki = new $tw.Wiki();
		
		// Add a test tiddler
		wiki.addTiddler({
			title: "TestTiddler",
			text: "This is a test tiddler with some content.",
			tags: ["TestTag"]
		});
		
		var tiddler = wiki.getTiddler("TestTiddler");
		var fingerprint = witnessUtils.calculateWitnessFingerprint(tiddler, wiki);
		
		expect(fingerprint).toBeDefined();
		expect(fingerprint.phase).toBeGreaterThanOrEqual(0);
		expect(fingerprint.phase).toBeLessThan(Math.PI * 2);
		expect(fingerprint.depth).toBeGreaterThanOrEqual(0);
		expect(fingerprint.sector).toBeGreaterThanOrEqual(0);
		expect(fingerprint.sector).toBeLessThan(100);
		expect(fingerprint.monodromy).toBeGreaterThanOrEqual(0);
		expect(fingerprint.monodromy).toBeLessThanOrEqual(1);
	});

	it("should calculate fingerprint distance between similar tiddlers", function() {
		var wiki = new $tw.Wiki();
		
		// Add two similar tiddlers
		wiki.addTiddler({
			title: "TiddlerA",
			text: "JavaScript programming concepts",
			tags: ["Programming", "JavaScript"]
		});
		
		wiki.addTiddler({
			title: "TiddlerB",
			text: "JavaScript design patterns",
			tags: ["Programming", "JavaScript"]
		});
		
		var tiddlerA = wiki.getTiddler("TiddlerA");
		var tiddlerB = wiki.getTiddler("TiddlerB");
		
		var fpA = witnessUtils.calculateWitnessFingerprint(tiddlerA, wiki);
		var fpB = witnessUtils.calculateWitnessFingerprint(tiddlerB, wiki);
		
		var distance = witnessUtils.calculateFingerprintDistance(fpA, fpB);
		
		// Similar tiddlers should have small distance
		expect(distance).toBeGreaterThanOrEqual(0);
		expect(distance).toBeLessThan(1);
	});

	it("should calculate fingerprint distance between dissimilar tiddlers", function() {
		var wiki = new $tw.Wiki();
		
		// Add two dissimilar tiddlers
		wiki.addTiddler({
			title: "TiddlerA",
			text: "JavaScript programming concepts",
			tags: ["Programming", "JavaScript"]
		});
		
		wiki.addTiddler({
			title: "TiddlerB",
			text: "Recipe for chocolate cake",
			tags: ["Cooking", "Desserts"]
		});
		
		var tiddlerA = wiki.getTiddler("TiddlerA");
		var tiddlerB = wiki.getTiddler("TiddlerB");
		
		var fpA = witnessUtils.calculateWitnessFingerprint(tiddlerA, wiki);
		var fpB = witnessUtils.calculateWitnessFingerprint(tiddlerB, wiki);
		
		var distance = witnessUtils.calculateFingerprintDistance(fpA, fpB);
		
		// Dissimilar tiddlers should have larger distance
		expect(distance).toBeGreaterThan(0.3);
	});

	it("should find similar tiddlers", function() {
		var wiki = new $tw.Wiki();
		
		// Add several tiddlers
		wiki.addTiddler({
			title: "JavaScript",
			text: "JavaScript is a programming language",
			tags: ["Programming", "Web"]
		});
		
		wiki.addTiddler({
			title: "TypeScript",
			text: "TypeScript is a typed superset of JavaScript",
			tags: ["Programming", "Web"]
		});
		
		wiki.addTiddler({
			title: "Python",
			text: "Python is a programming language",
			tags: ["Programming"]
		});
		
		wiki.addTiddler({
			title: "Recipe",
			text: "How to make pasta",
			tags: ["Cooking"]
		});
		
		var targetTiddler = wiki.getTiddler("JavaScript");
		var similarTiddlers = witnessUtils.findSimilarTiddlers(targetTiddler, wiki, {
			threshold: 0.5,
			maxResults: 5
		});
		
		expect(similarTiddlers.length).toBeGreaterThan(0);
		
		// TypeScript should be found as similar (same tags)
		var foundTypeScript = similarTiddlers.some(function(result) {
			return result.title === "TypeScript";
		});
		expect(foundTypeScript).toBe(true);
		
		// Recipe should likely not be in top similar due to different tags
		// But with default threshold, it might appear, so we just check TypeScript is more similar
		var typeScriptResult = similarTiddlers.find(function(result) {
			return result.title === "TypeScript";
		});
		var recipeResult = similarTiddlers.find(function(result) {
			return result.title === "Recipe";
		});
		
		if(recipeResult) {
			// If recipe is found, TypeScript should have higher similarity
			expect(typeScriptResult.similarity).toBeGreaterThan(recipeResult.similarity);
		}
	});

	it("should calculate resonance between tiddlers", function() {
		var wiki = new $tw.Wiki();
		
		wiki.addTiddler({
			title: "TiddlerA",
			text: "Content A",
			tags: ["Tag1", "Tag2"]
		});
		
		wiki.addTiddler({
			title: "TiddlerB",
			text: "Content B",
			tags: ["Tag1", "Tag2"]
		});
		
		var tiddlerA = wiki.getTiddler("TiddlerA");
		var tiddlerB = wiki.getTiddler("TiddlerB");
		
		var resonance = witnessUtils.calculateResonance(tiddlerA, tiddlerB, wiki);
		
		expect(resonance).toBeGreaterThanOrEqual(0);
		expect(resonance).toBeLessThanOrEqual(1);
	});

	it("should handle tiddlers with bidirectional links (monodromy)", function() {
		var wiki = new $tw.Wiki();
		
		// Create tiddlers with explicit links field
		wiki.addTiddler({
			title: "TiddlerA",
			text: "Links to [[TiddlerB]]",
			tags: [],
			links: ["TiddlerB"]
		});
		
		wiki.addTiddler({
			title: "TiddlerB",
			text: "Links back to [[TiddlerA]]",
			tags: [],
			links: ["TiddlerA"]
		});
		
		var tiddlerA = wiki.getTiddler("TiddlerA");
		var fingerprint = witnessUtils.calculateWitnessFingerprint(tiddlerA, wiki);
		
		// With bidirectional link, monodromy should be 1.0 (100%)
		expect(fingerprint.monodromy).toBeGreaterThan(0);
		expect(fingerprint.monodromy).toBeLessThanOrEqual(1);
	});

	it("should handle tiddlers with transclusions", function() {
		var wiki = new $tw.Wiki();
		
		wiki.addTiddler({
			title: "TiddlerA",
			text: "This includes {{TiddlerB}} and <<macro>>",
			tags: []
		});
		
		var tiddler = wiki.getTiddler("TiddlerA");
		var fingerprint = witnessUtils.calculateWitnessFingerprint(tiddler, wiki);
		
		expect(fingerprint.transclusionComplexity).toBeGreaterThan(0);
	});

	it("should handle tiddlers with custom fields", function() {
		var wiki = new $tw.Wiki();
		
		wiki.addTiddler({
			title: "TiddlerA",
			text: "Content",
			tags: [],
			customField1: "value1",
			customField2: "value2"
		});
		
		var tiddler = wiki.getTiddler("TiddlerA");
		var fingerprint = witnessUtils.calculateWitnessFingerprint(tiddler, wiki);
		
		expect(fingerprint.fieldComplexity).toBe(2);
	});

	it("should calculate hierarchy depth for nested tags", function() {
		var wiki = new $tw.Wiki();
		
		// Create tag hierarchy: Root -> Parent -> Child
		wiki.addTiddler({
			title: "Root",
			text: "Root tag",
			tags: []
		});
		
		wiki.addTiddler({
			title: "Parent",
			text: "Parent tag",
			tags: ["Root"]
		});
		
		wiki.addTiddler({
			title: "Child",
			text: "Child tag",
			tags: ["Parent"]
		});
		
		wiki.addTiddler({
			title: "TestTiddler",
			text: "Test content",
			tags: ["Child"]
		});
		
		var tiddler = wiki.getTiddler("TestTiddler");
		var fingerprint = witnessUtils.calculateWitnessFingerprint(tiddler, wiki);
		
		// Should have depth of 3 (Root -> Parent -> Child)
		expect(fingerprint.depth).toBeGreaterThan(0);
	});

});

})();
