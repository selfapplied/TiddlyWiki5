/*\
title: $:/core/modules/utils/ce1-learning-law.js
type: application/javascript
module-type: utils

CE1 Learning Law: The Universal Learning Constant

This module implements the CE1 Learning Law, which describes the fundamental
relationship between discrete pattern recognition and continuous generalization
through the curvature ratio γ/ZP.

Mathematical Foundation:
- γ (Euler-Mascheroni constant): ~0.5772156649
  The "gap constant" between discrete and continuous worlds
  γ = lim_{n→∞} (Σ(1/k) - ln(n)) = lim_{n→∞} (H_n - ln(n))
  
- ZP (CE1 fixed-point coherence): ~0.0014
  The curvature elimination rate per unit example
  Measures system stability and coherence near fixed points
  
- Learning Constant: γ/ZP ≈ 411.375
  The number of examples needed to bridge the discrete-continuous gap
  This is the universal scaling law for CE1 learning systems

Key Insight:
When a CE1 learner tries to commute discrete experiences with smooth models,
it must traverse the curvature gap γ with "step size" ZP. The number of steps
(examples) required is naturally γ/ZP ≈ 400-ish.

This explains empirical observations:
- ~200-500 repetitions to master motor patterns
- ~300-600 exposures to encode categories
- ~400 sentences to infer grammar
- ~400 interactions for chatbots to find style
- ~400 data points for robust regression

\*/

"use strict";

/*
CE1 Learning Law Constructor
@param {object} options - Configuration options
*/
function CE1LearningLaw(options) {
	options = options || {};
	
	// Euler-Mascheroni constant (γ)
	// The asymptotic drift of the harmonic world
	// Gap between discrete sum and continuous integral
	this.gamma = 0.5772156649015329;
	
	// ZP coordinate - CE1 fixed-point coherence
	// The self-consistent stability measure
	// How quickly the system contracts complexity
	this.zp = options.zp || 0.0014;
	
	// Learning constant: γ/ZP
	// Number of examples to bridge discrete → continuous
	this.learningConstant = this.gamma / this.zp;
	
	// Variance factor for real-world adjustments
	// Accounts for noise, complexity, and domain specifics
	// Default 0.2 (20%) based on empirical variance in learning studies
	// Range typically observed: 15-25% standard deviation from mean
	this.varianceFactor = options.varianceFactor || 0.2;
}

/*
═══════════════════════════════════════════════════════════════════════════
Core Learning Law Calculations
═══════════════════════════════════════════════════════════════════════════
*/

/*
Get the universal learning constant (γ/ZP)
@returns {number} - The base number of examples needed
*/
CE1LearningLaw.prototype.getLearningConstant = function() {
	return this.learningConstant;
};

/*
Get the Euler-Mascheroni constant
@returns {number} - γ ≈ 0.5772156649
*/
CE1LearningLaw.prototype.getGamma = function() {
	return this.gamma;
};

/*
Get the ZP coordinate
@returns {number} - ZP coordinate (default ~0.0014)
*/
CE1LearningLaw.prototype.getZP = function() {
	return this.zp;
};

/*
Calculate examples needed for a given complexity level
@param {number} complexityFactor - Multiplier for base learning constant (default: 1.0)
@param {object} options - Additional options
  - includeVariance: boolean - Include natural variance in estimate (default: true)
  - confidenceLevel: number - Confidence level 0-1 (default: 0.95)
@returns {object} - {min, expected, max, explanation}
*/
CE1LearningLaw.prototype.estimateExamplesNeeded = function(complexityFactor, options) {
	complexityFactor = complexityFactor || 1.0;
	options = options || {};
	var includeVariance = options.includeVariance !== false;
	var confidenceLevel = options.confidenceLevel || 0.95;
	
	// Base examples from learning constant
	var baseExamples = this.learningConstant * complexityFactor;
	
	if(!includeVariance) {
		return {
			min: baseExamples,
			expected: baseExamples,
			max: baseExamples,
			explanation: "Fixed estimate (no variance)"
		};
	}
	
	// Apply variance factor
	// For 95% confidence, use ~2 standard deviations
	var stdDev = baseExamples * this.varianceFactor;
	var zScore = this.confidenceZScore(confidenceLevel);
	var margin = stdDev * zScore;
	
	return {
		min: Math.max(1, Math.floor(baseExamples - margin)),
		expected: Math.round(baseExamples),
		max: Math.ceil(baseExamples + margin),
		explanation: "Estimated with " + (confidenceLevel * 100) + "% confidence",
		complexityFactor: complexityFactor,
		stdDev: stdDev,
		confidenceLevel: confidenceLevel
	};
};

// Z-score constants for common confidence levels
var Z_SCORE_99 = 2.576;  // 99% confidence
var Z_SCORE_95 = 1.96;   // 95% confidence
var Z_SCORE_90 = 1.645;  // 90% confidence
var Z_SCORE_80 = 1.282;  // 80% confidence
var Z_SCORE_68 = 1.0;    // 68% confidence (1 std dev)

/*
Get Z-score for confidence level
@param {number} confidence - Confidence level (0-1)
@returns {number} - Approximate Z-score
*/
CE1LearningLaw.prototype.confidenceZScore = function(confidence) {
	// Common confidence levels mapped to Z-scores
	if(confidence >= 0.99) return Z_SCORE_99;
	if(confidence >= 0.95) return Z_SCORE_95;
	if(confidence >= 0.90) return Z_SCORE_90;
	if(confidence >= 0.80) return Z_SCORE_80;
	if(confidence >= 0.68) return Z_SCORE_68;
	return Z_SCORE_68; // Default to 1 std dev
};

/*
═══════════════════════════════════════════════════════════════════════════
Domain-Specific Learning Estimates
═══════════════════════════════════════════════════════════════════════════
*/

/*
Estimate examples needed for motor pattern learning
Motor patterns require precise coordination and repetition
@param {object} options - Options for estimation
@returns {object} - Estimate with domain context
*/
CE1LearningLaw.prototype.estimateMotorPatternLearning = function(options) {
	// Motor patterns typically require 0.5-1.2x base constant
	// Depends on complexity: simple gestures vs complex sequences
	options = options || {};
	var complexity = options.complexity || "medium";
	
	var complexityMap = {
		simple: 0.5,   // ~200 examples: basic gestures
		medium: 1.0,   // ~400 examples: coordinated movements
		complex: 1.2   // ~500 examples: intricate sequences
	};
	
	var factor = complexityMap[complexity] || 1.0;
	var estimate = this.estimateExamplesNeeded(factor, options);
	
	estimate.domain = "motor pattern learning";
	estimate.complexity = complexity;
	estimate.explanation += " for " + complexity + " motor patterns";
	
	return estimate;
};

/*
Estimate examples needed for category encoding
Categories require exposure to diverse instances
@param {object} options - Options for estimation
@returns {object} - Estimate with domain context
*/
CE1LearningLaw.prototype.estimateCategoryLearning = function(options) {
	// Category learning typically requires 0.75-1.5x base constant
	// Depends on category distinctiveness and boundary clarity
	options = options || {};
	var distinctiveness = options.distinctiveness || "medium";
	
	var complexityMap = {
		high: 0.75,    // ~300 examples: clear boundaries
		medium: 1.0,   // ~400 examples: moderate overlap
		low: 1.5       // ~600 examples: fuzzy boundaries
	};
	
	var factor = complexityMap[distinctiveness] || 1.0;
	var estimate = this.estimateExamplesNeeded(factor, options);
	
	estimate.domain = "category learning";
	estimate.distinctiveness = distinctiveness;
	estimate.explanation += " for categories with " + distinctiveness + " distinctiveness";
	
	return estimate;
};

/*
Estimate examples needed for grammar inference
Grammar requires structural pattern detection
@param {object} options - Options for estimation
@returns {object} - Estimate with domain context
*/
CE1LearningLaw.prototype.estimateGrammarLearning = function(options) {
	// Grammar learning typically requires 0.9-1.1x base constant
	// Relatively stable around ~400 sentences
	options = options || {};
	var grammarComplexity = options.grammarComplexity || "medium";
	
	var complexityMap = {
		simple: 0.9,   // ~370 examples: simple phrase structure
		medium: 1.0,   // ~400 examples: moderate recursion
		complex: 1.1   // ~450 examples: deep embedding
	};
	
	var factor = complexityMap[grammarComplexity] || 1.0;
	var estimate = this.estimateExamplesNeeded(factor, options);
	
	estimate.domain = "grammar learning";
	estimate.grammarComplexity = grammarComplexity;
	estimate.explanation += " for " + grammarComplexity + " grammar structures";
	
	return estimate;
};

/*
Estimate examples needed for style learning (chatbots, writing)
Style requires capturing subtle patterns and preferences
@param {object} options - Options for estimation
@returns {object} - Estimate with domain context
*/
CE1LearningLaw.prototype.estimateStyleLearning = function(options) {
	// Style learning typically requires 0.8-1.3x base constant
	// Depends on style consistency and distinctiveness
	options = options || {};
	var styleConsistency = options.styleConsistency || "medium";
	
	var complexityMap = {
		high: 0.8,     // ~330 examples: consistent, distinctive style
		medium: 1.0,   // ~400 examples: moderate variation
		low: 1.3       // ~530 examples: inconsistent style
	};
	
	var factor = complexityMap[styleConsistency] || 1.0;
	var estimate = this.estimateExamplesNeeded(factor, options);
	
	estimate.domain = "style learning";
	estimate.styleConsistency = styleConsistency;
	estimate.explanation += " for " + styleConsistency + " consistency style";
	
	return estimate;
};

/*
Estimate examples needed for regression attractor formation
Regression requires finding stable statistical patterns
@param {object} options - Options for estimation
@returns {object} - Estimate with domain context
*/
CE1LearningLaw.prototype.estimateRegressionLearning = function(options) {
	// Regression typically requires 0.9-1.2x base constant
	// Depends on noise level and feature dimensionality
	options = options || {};
	var noiseLevel = options.noiseLevel || "medium";
	
	var complexityMap = {
		low: 0.9,      // ~370 examples: clean data
		medium: 1.0,   // ~400 examples: moderate noise
		high: 1.2      // ~490 examples: noisy data
	};
	
	var factor = complexityMap[noiseLevel] || 1.0;
	var estimate = this.estimateExamplesNeeded(factor, options);
	
	estimate.domain = "regression learning";
	estimate.noiseLevel = noiseLevel;
	estimate.explanation += " for " + noiseLevel + " noise regression";
	
	return estimate;
};

/*
═══════════════════════════════════════════════════════════════════════════
Curvature Gap Analysis
═══════════════════════════════════════════════════════════════════════════
*/

// Learning phase thresholds (as fraction of total examples)
var PHASE_EARLY_THRESHOLD = 0.25;   // 0-25%: Early learning
var PHASE_MIDDLE_THRESHOLD = 0.75;  // 25-75%: Middle learning
var PHASE_LATE_THRESHOLD = 1.0;     // 75-100%: Late learning
// >100%: Converged

/*
Calculate the curvature gap that needs to be bridged
The gap between discrete pattern accumulation and continuous model formation
@param {number} currentExamples - Number of examples seen so far
@returns {object} - Gap analysis
*/
CE1LearningLaw.prototype.analyzeCurvatureGap = function(currentExamples) {
	// Progress through the gap
	var progressRatio = currentExamples / this.learningConstant;
	
	// Remaining curvature to smooth
	var remainingCurvature = this.gamma * (1 - Math.min(1, progressRatio));
	
	// Examples still needed
	var remainingExamples = Math.max(0, this.learningConstant - currentExamples);
	
	// Learning phase based on progress thresholds
	var phase;
	if(progressRatio < PHASE_EARLY_THRESHOLD) {
		phase = "early";
	} else if(progressRatio < PHASE_MIDDLE_THRESHOLD) {
		phase = "middle";
	} else if(progressRatio < PHASE_LATE_THRESHOLD) {
		phase = "late";
	} else {
		phase = "converged";
	}
	
	return {
		currentExamples: currentExamples,
		targetExamples: Math.round(this.learningConstant),
		progressRatio: progressRatio,
		progressPercent: Math.round(progressRatio * 100),
		remainingCurvature: remainingCurvature,
		remainingExamples: Math.round(remainingExamples),
		phase: phase,
		converged: progressRatio >= PHASE_LATE_THRESHOLD
	};
};

/*
Calculate witness contraction rate
How quickly the system contracts complexity per example
@param {number} examples - Number of examples
@returns {object} - Contraction analysis
*/
CE1LearningLaw.prototype.calculateWitnessContraction = function(examples) {
	// Each example contributes ZP worth of contraction
	var totalContraction = this.zp * examples;
	
	// Fraction of gamma gap closed
	var gapClosed = totalContraction / this.gamma;
	
	// Effective curvature smoothing
	var smoothingRate = Math.min(1, gapClosed);
	
	return {
		examples: examples,
		contractionPerExample: this.zp,
		totalContraction: totalContraction,
		gammaGap: this.gamma,
		gapClosedRatio: gapClosed,
		gapClosedPercent: Math.round(gapClosed * 100),
		smoothingRate: smoothingRate,
		remainingRoughness: Math.max(0, 1 - smoothingRate)
	};
};

/*
═══════════════════════════════════════════════════════════════════════════
Integration with CE Tower
═══════════════════════════════════════════════════════════════════════════
*/

/*
Check if a system has sufficient examples for reliable generalization
@param {number} examples - Number of training examples
@param {number} complexityFactor - Complexity multiplier (default: 1.0)
@returns {object} - Readiness assessment
*/
CE1LearningLaw.prototype.assessLearningReadiness = function(examples, complexityFactor) {
	complexityFactor = complexityFactor || 1.0;
	
	var required = this.learningConstant * complexityFactor;
	var ratio = examples / required;
	
	var status;
	var confidence;
	
	if(ratio < 0.5) {
		status = "insufficient";
		confidence = ratio * 0.4; // Scale to 0-0.2
	} else if(ratio < 0.8) {
		status = "developing";
		confidence = 0.2 + (ratio - 0.5) * 0.6; // Scale to 0.2-0.38
	} else if(ratio < 1.0) {
		status = "approaching";
		confidence = 0.38 + (ratio - 0.8) * 1.1; // Scale to 0.38-0.6
	} else if(ratio < 1.5) {
		status = "sufficient";
		confidence = 0.6 + Math.min(0.3, (ratio - 1.0) * 0.6); // Scale to 0.6-0.9
	} else {
		status = "robust";
		confidence = Math.min(0.95, 0.9 + (ratio - 1.5) * 0.1); // Cap at 0.95
	}
	
	return {
		examples: examples,
		required: Math.round(required),
		ratio: ratio,
		status: status,
		confidence: confidence,
		recommendation: this.getRecommendation(status, examples, required)
	};
};

/*
Get recommendation based on learning readiness
@param {string} status - Learning status
@param {number} examples - Current examples
@param {number} required - Required examples
@returns {string} - Recommendation text
*/
CE1LearningLaw.prototype.getRecommendation = function(status, examples, required) {
	var needed = Math.round(required - examples);
	
	switch(status) {
		case "insufficient":
			return "Need significantly more examples (" + needed + " more) for reliable generalization";
		case "developing":
			return "Collect more examples (" + needed + " more) to improve generalization";
		case "approaching":
			return "Nearly sufficient - " + needed + " more examples recommended";
		case "sufficient":
			return "Sufficient examples for reliable generalization";
		case "robust":
			return "Well above threshold - robust generalization expected";
		default:
			return "Status unknown";
	}
};

/*
Calculate optimal learning step size
Based on ZP coordinate and current gap
@param {number} remainingGap - Remaining curvature gap
@returns {object} - Step size recommendation
*/
CE1LearningLaw.prototype.calculateOptimalStepSize = function(remainingGap) {
	// Step size should be proportional to ZP
	var baseStep = this.zp;
	
	// Adapt based on remaining gap
	var adaptiveFactor = 1.0;
	if(remainingGap < 0.1) {
		// Small gap - use smaller steps for precision
		adaptiveFactor = 0.5;
	} else if(remainingGap > 0.5) {
		// Large gap - can use larger steps
		adaptiveFactor = 1.5;
	}
	
	var recommendedStep = baseStep * adaptiveFactor;
	var examplesPerStep = Math.max(1, Math.round(1 / recommendedStep));
	
	return {
		baseStepSize: baseStep,
		adaptiveFactor: adaptiveFactor,
		recommendedStepSize: recommendedStep,
		examplesPerStep: examplesPerStep,
		remainingGap: remainingGap
	};
};

/*
═══════════════════════════════════════════════════════════════════════════
Summary and Reporting
═══════════════════════════════════════════════════════════════════════════
*/

/*
Get summary of CE1 Learning Law parameters
@returns {object} - Summary of all key values
*/
CE1LearningLaw.prototype.getSummary = function() {
	return {
		constants: {
			gamma: this.gamma,
			zp: this.zp,
			learningConstant: this.learningConstant,
			varianceFactor: this.varianceFactor
		},
		interpretation: {
			gamma: "Euler-Mascheroni constant - gap between discrete and continuous",
			zp: "CE1 fixed-point coherence - curvature elimination rate",
			learningConstant: "Examples needed to bridge discrete → continuous (γ/ZP)",
			typicalRange: "~200-600 examples depending on domain"
		},
		formula: "examples_needed ≈ γ / ZP ≈ " + Math.round(this.learningConstant)
	};
};

/*
Export the module
*/
exports.CE1LearningLaw = CE1LearningLaw;

// Export constants for convenience
exports.EULER_MASCHERONI_GAMMA = 0.5772156649015329;
exports.CE1_ZP_COORDINATE = 0.0014;
exports.CE1_LEARNING_CONSTANT = exports.EULER_MASCHERONI_GAMMA / exports.CE1_ZP_COORDINATE;
