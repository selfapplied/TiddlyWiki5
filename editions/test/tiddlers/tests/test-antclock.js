/*\
title: test-antclock.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for antclock (experiential time) system

\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

describe("Antclock System", function() {

	var anticlockUtils = require("$:/core/modules/utils/antclock.js");

	it("should calculate clock rate for minor changes", function() {
		var oldContent = "This is a test.";
		var newContent = "This is a test!"; // Just punctuation change
		
		var clockRate = anticlockUtils.calculateClockRate(oldContent, newContent);
		
		// Minor change should have small clock rate
		expect(clockRate).toBeGreaterThanOrEqual(0);
		expect(clockRate).toBeLessThan(0.2);
	});

	it("should calculate clock rate for major changes", function() {
		var oldContent = "This is a short test.";
		var newContent = "This is a completely different document with many new words, sections, and ideas. It represents a major revision.";
		
		var clockRate = anticlockUtils.calculateClockRate(oldContent, newContent);
		
		// Major change should have larger clock rate
		expect(clockRate).toBeGreaterThan(0.3);
	});

	it("should calculate clock rate for structural changes", function() {
		var oldContent = "Simple paragraph text.";
		var newContent = "# Header\n\n* List item 1\n* List item 2\n\n[[Link]]\n\n{{Transclusion}}";
		
		var clockRate = anticlockUtils.calculateClockRate(oldContent, newContent);
		
		// Structural change should be significant
		expect(clockRate).toBeGreaterThan(0.2);
	});

	it("should return zero for identical content", function() {
		var content = "This is identical content.";
		
		var clockRate = anticlockUtils.calculateClockRate(content, content);
		
		expect(clockRate).toBe(0);
	});

	it("should handle empty content", function() {
		var clockRate = anticlockUtils.calculateClockRate("", "New content");
		
		expect(clockRate).toBeGreaterThan(0);
	});

	it("should get experiential age from tiddler", function() {
		var tiddler = {
			fields: {
				title: "Test",
				"experiential-time": "5.5"
			}
		};
		
		var age = anticlockUtils.getExperientialAge(tiddler);
		
		expect(age).toBe(5.5);
	});

	it("should return zero age for new tiddler", function() {
		var tiddler = {
			fields: {
				title: "Test"
			}
		};
		
		var age = anticlockUtils.getExperientialAge(tiddler);
		
		expect(age).toBe(0);
	});

	it("should record antclock tick", function() {
		var wiki = new $tw.Wiki();
		
		var tiddler = {
			fields: {
				title: "Test",
				"experiential-time": "2.5"
			}
		};
		
		var result = anticlockUtils.recordAnticlockTick(wiki, tiddler, 0.8, {
			reason: "major revision"
		});
		
		expect(result).toBeDefined();
		expect(result.newTime).toBe(3.3); // 2.5 + 0.8
		expect(result.event.clockRate).toBe(0.8);
		expect(result.event.details.reason).toBe("major revision");
	});

	it("should not record tick below significance threshold", function() {
		var wiki = new $tw.Wiki();
		
		var tiddler = {
			fields: {
				title: "Test",
				"experiential-time": "2.5"
			}
		};
		
		// Clock rate of 0.05 is below default threshold of 0.1
		var result = anticlockUtils.recordAnticlockTick(wiki, tiddler, 0.05);
		
		expect(result).toBeNull();
	});

	it("should maintain experiential history", function() {
		var wiki = new $tw.Wiki();
		
		var tiddler = {
			fields: {
				title: "Test",
				"experiential-time": "1.0",
				"experiential-history": JSON.stringify([
					{ timestamp: "2024-01-01", experientialTime: 0.5, clockRate: 0.5 }
				])
			}
		};
		
		var result = anticlockUtils.recordAnticlockTick(wiki, tiddler, 0.3);
		
		expect(result.history.length).toBe(2);
		expect(result.history[0].experientialTime).toBe(0.5);
		expect(result.history[1].experientialTime).toBe(1.3);
	});

	it("should parse experiential history", function() {
		var tiddler = {
			fields: {
				title: "Test",
				"experiential-history": JSON.stringify([
					{ timestamp: "2024-01-01", experientialTime: 0.5, clockRate: 0.5 },
					{ timestamp: "2024-01-02", experientialTime: 1.0, clockRate: 0.5 }
				])
			}
		};
		
		var history = anticlockUtils.getExperientialHistory(tiddler);
		
		expect(history.length).toBe(2);
		expect(history[0].clockRate).toBe(0.5);
		expect(history[1].experientialTime).toBe(1.0);
	});

	it("should handle invalid history JSON", function() {
		var tiddler = {
			fields: {
				title: "Test",
				"experiential-history": "invalid json"
			}
		};
		
		var history = anticlockUtils.getExperientialHistory(tiddler);
		
		expect(history.length).toBe(0);
	});

	it("should calculate recent activity rate", function() {
		var tiddler = {
			fields: {
				title: "Test",
				"experiential-history": JSON.stringify([
					{ timestamp: "2024-01-01", experientialTime: 0.3, clockRate: 0.3 },
					{ timestamp: "2024-01-02", experientialTime: 0.8, clockRate: 0.5 },
					{ timestamp: "2024-01-03", experientialTime: 1.2, clockRate: 0.4 }
				])
			}
		};
		
		var rate = anticlockUtils.getRecentActivityRate(tiddler, 3);
		
		// Average of 0.3, 0.5, 0.4 = 1.2 / 3 = 0.4
		expect(rate).toBeCloseTo(0.4, 2);
	});

	it("should compare experiential activity between tiddlers", function() {
		var tiddler1 = {
			fields: {
				title: "Active",
				"experiential-time": "10.5"
			}
		};
		
		var tiddler2 = {
			fields: {
				title: "Stable",
				"experiential-time": "2.0"
			}
		};
		
		var comparison = anticlockUtils.compareExperientialActivity(tiddler1, tiddler2);
		
		// Should be negative (tiddler1 has more activity)
		expect(comparison).toBeLessThan(0);
	});

	it("should handle semantic changes in content", function() {
		var oldContent = "JavaScript is a programming language.";
		var newContent = "JavaScript and TypeScript are programming languages used for web development.";
		
		var clockRate = anticlockUtils.calculateClockRate(oldContent, newContent);
		
		// Semantic expansion should be significant
		expect(clockRate).toBeGreaterThan(0.2);
	});

	it("should detect structural coherence changes", function() {
		var oldContent = "# Title\n\nParagraph 1.\n\nParagraph 2.";
		var newContent = "Title Paragraph 1. Paragraph 2."; // Lost structure
		
		var clockRate = anticlockUtils.calculateClockRate(oldContent, newContent);
		
		// Loss of structure should register
		expect(clockRate).toBeGreaterThan(0);
	});

});

})();
