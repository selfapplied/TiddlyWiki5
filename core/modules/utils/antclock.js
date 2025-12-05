/*\
title: $:/core/modules/utils/antclock.js
type: application/javascript
module-type: utils

Antclock (Experiential Time) System for TiddlyWiki

Inspired by the CE Tower antclock mechanism from:
https://github.com/selfapplied/antclock/blob/main/arXiv/working.md

The antclock measures time in semantic transition units rather than clock ticks.
It advances when semantically significant state changes occur, not merely on every edit.

This enables:
- Distinguish minor edits from major revisions
- Track conceptual evolution vs temporal sequence
- Filter by semantic activity level
- Identify stable vs rapidly evolving knowledge

\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

// Constants from antclock research
// SIGNIFICANCE_THRESHOLD: Minimum clock rate to record an antclock tick
// This threshold prevents tracking trivial changes (typos, formatting)
// while capturing semantically meaningful revisions
var SIGNIFICANCE_THRESHOLD = 0.1;

// CHI_FEG: Transform quality measure from antclock research
// χ_FEG ≈ 0.638 is derived from the CE Tower's transform operator
// This constant scales the clock rate to match experiential significance
// Source: https://github.com/selfapplied/antclock/blob/main/arXiv/working.md
var CHI_FEG = 0.638;

/**
 * Calculate semantic change between two versions of content
 * Returns a clock rate R(x) representing significance of change
 */
exports.calculateClockRate = function(oldContent, newContent, options) {
	options = options || {};
	
	// Handle empty content cases
	oldContent = oldContent || "";
	newContent = newContent || "";
	
	if(oldContent === newContent) {
		return 0;
	}
	
	// If both are empty, no change
	if(oldContent.length === 0 && newContent.length === 0) {
		return 0;
	}
	
	// Calculate different types of changes
	var structuralChange = calculateStructuralDifference(oldContent, newContent);
	var semanticChange = calculateSemanticDifference(oldContent, newContent);
	var coherenceChange = calculateCoherenceShift(oldContent, newContent);
	
	// Combine into clock rate using antclock formula
	// R(x) = χ_FEG · κ_d(x) · (1 + Q_9/11(x))
	// Simplified version using our metrics
	var clockRate = CHI_FEG * (
		0.4 * structuralChange +
		0.4 * semanticChange +
		0.2 * coherenceChange
	);
	
	return Math.max(0, Math.min(1, clockRate));
};

/**
 * Calculate structural difference (syntax, formatting, structure)
 */
function calculateStructuralDifference(oldContent, newContent) {
	// Count structural elements
	var oldHeaders = (oldContent.match(/^#+\s/gm) || []).length;
	var newHeaders = (newContent.match(/^#+\s/gm) || []).length;
	
	var oldLists = (oldContent.match(/^[*#]\s/gm) || []).length;
	var newLists = (newContent.match(/^[*#]\s/gm) || []).length;
	
	var oldTransclusions = (oldContent.match(/\{\{[^}]+\}\}/g) || []).length;
	var newTransclusions = (newContent.match(/\{\{[^}]+\}\}/g) || []).length;
	
	var oldLinks = (oldContent.match(/\[\[[^\]]+\]\]/g) || []).length;
	var newLinks = (newContent.match(/\[\[[^\]]+\]\]/g) || []).length;
	
	// Calculate relative changes
	var headerChange = Math.abs(newHeaders - oldHeaders) / Math.max(oldHeaders, newHeaders, 1);
	var listChange = Math.abs(newLists - oldLists) / Math.max(oldLists, newLists, 1);
	var transclusionChange = Math.abs(newTransclusions - oldTransclusions) / Math.max(oldTransclusions, newTransclusions, 1);
	var linkChange = Math.abs(newLinks - oldLinks) / Math.max(oldLinks, newLinks, 1);
	
	// Average structural change
	return (headerChange + listChange + transclusionChange + linkChange) / 4;
}

/**
 * Calculate semantic difference (meaning, content)
 */
function calculateSemanticDifference(oldContent, newContent) {
	// Extract words for comparison
	var oldWords = extractWords(oldContent);
	var newWords = extractWords(newContent);
	
	// Calculate word set changes
	var oldWordSet = new Set(oldWords);
	var newWordSet = new Set(newWords);
	
	var addedWords = 0;
	var removedWords = 0;
	
	newWordSet.forEach(function(word) {
		if(!oldWordSet.has(word)) {
			addedWords++;
		}
	});
	
	oldWordSet.forEach(function(word) {
		if(!newWordSet.has(word)) {
			removedWords++;
		}
	});
	
	var totalWords = Math.max(oldWords.length, newWords.length, 1);
	var wordChange = (addedWords + removedWords) / totalWords;
	
	// Calculate length change
	var lengthChange = Math.abs(newContent.length - oldContent.length) / 
	                   Math.max(oldContent.length, newContent.length, 1);
	
	return (0.7 * wordChange + 0.3 * lengthChange);
}

/**
 * Calculate coherence shift (internal consistency)
 */
function calculateCoherenceShift(oldContent, newContent) {
	// Simple coherence metric based on ratio of formatting to content
	var oldCoherence = calculateCoherence(oldContent);
	var newCoherence = calculateCoherence(newContent);
	
	return Math.abs(newCoherence - oldCoherence);
}

/**
 * Calculate coherence score for content
 */
function calculateCoherence(content) {
	if(content.length === 0) {
		return 0;
	}
	
	// Count structural elements
	var headers = (content.match(/^#+\s/gm) || []).length;
	var paragraphs = content.split(/\n\n+/).length;
	var sentences = content.split(/[.!?]+/).length;
	
	// Coherence is ratio of structure to raw length
	return (headers + paragraphs * 0.5 + sentences * 0.1) / (content.length / 100);
}

/**
 * Extract words from content
 */
function extractWords(content) {
	// Remove wiki markup
	var cleaned = content
		.replace(/\{\{[^}]+\}\}/g, '') // Remove transclusions
		.replace(/\[\[[^\]]+\]\]/g, '') // Remove links
		.replace(/^#+\s/gm, '') // Remove headers
		.replace(/^[*#]\s/gm, ''); // Remove lists
	
	// Extract words (alphanumeric sequences)
	var words = cleaned.toLowerCase().match(/\b[a-z0-9]+\b/g) || [];
	
	return words;
}

/**
 * Record an antclock tick for a tiddler
 */
exports.recordAnticlockTick = function(wiki, tiddler, clockRate, details) {
	if(!tiddler || clockRate < SIGNIFICANCE_THRESHOLD) {
		return null;
	}
	
	wiki = wiki || $tw.wiki;
	
	// Get current experiential time from tiddler
	var currentTime = parseFloat(tiddler.fields["experiential-time"] || "0");
	var newTime = currentTime + clockRate;
	
	// Create antclock event record
	var event = {
		timestamp: new Date().toISOString(),
		experientialTime: newTime,
		clockRate: clockRate,
		details: details || {}
	};
	
	// Store event in history (if tracking enabled)
	var historyField = tiddler.fields["experiential-history"];
	var history = [];
	
	if(historyField) {
		try {
			history = JSON.parse(historyField);
		} catch(e) {
			history = [];
		}
	}
	
	history.push(event);
	
	// Limit history size to prevent unbounded growth
	// Default limit of 100 events provides sufficient history
	// while keeping storage manageable. Could be made configurable
	// via a system tiddler field in future versions.
	var MAX_HISTORY_SIZE = 100;
	if(history.length > MAX_HISTORY_SIZE) {
		history = history.slice(-MAX_HISTORY_SIZE);
	}
	
	return {
		newTime: newTime,
		event: event,
		history: history
	};
};

/**
 * Get experiential age of a tiddler
 */
exports.getExperientialAge = function(tiddler) {
	if(!tiddler) {
		return 0;
	}
	
	return parseFloat(tiddler.fields["experiential-time"] || "0");
};

/**
 * Compare tiddlers by experiential activity
 */
exports.compareExperientialActivity = function(tiddler1, tiddler2) {
	var age1 = exports.getExperientialAge(tiddler1);
	var age2 = exports.getExperientialAge(tiddler2);
	
	return age2 - age1; // Higher age = more activity
};

/**
 * Get tiddler's experiential history
 */
exports.getExperientialHistory = function(tiddler) {
	if(!tiddler || !tiddler.fields["experiential-history"]) {
		return [];
	}
	
	try {
		return JSON.parse(tiddler.fields["experiential-history"]);
	} catch(e) {
		return [];
	}
};

/**
 * Calculate average clock rate over recent history
 */
exports.getRecentActivityRate = function(tiddler, windowSize) {
	windowSize = windowSize || 10;
	
	var history = exports.getExperientialHistory(tiddler);
	if(history.length === 0) {
		return 0;
	}
	
	var recentEvents = history.slice(-windowSize);
	var totalRate = recentEvents.reduce(function(sum, event) {
		return sum + (event.clockRate || 0);
	}, 0);
	
	return totalRate / recentEvents.length;
};

})();
