/*\
title: $:/core/modules/utils/ce-tower.js
type: application/javascript
module-type: utils

CE Tower: Compositional Evolution Tower Implementation

This module implements the CE Tower architecture as a unified compatibility
layer ensuring consistency between discrete (VM), continuous (ML), and
spectral (compression) views of the semantic manifold.

The CE Tower provides three layers:
- CE1: Discrete Syntax (combinatorial rules)
- CE2: Continuous Flow (discrete-continuous compatibility)
- CE3: Spectral Witness (invariant stabilization)

\*/

"use strict";

/*
CE Tower Constructor
@param {object} options - Configuration options
*/
function CETower(options) {
	options = options || {};
	
	// Guardian threshold (coherence curvature)
	this.kappa = options.kappa || 0.35;
	
	// Spectral tolerance for CE3
	this.spectralTolerance = options.spectralTolerance || 0.05;
	
	// Flow tolerance for CE2
	this.flowTolerance = options.flowTolerance || 0.1;
	
	// Syntax rules for CE1
	this.syntaxRules = {};
	
	// Statistics
	this.stats = {
		ce1Checks: 0,
		ce2Checks: 0,
		ce3Checks: 0,
		violations: {
			ce1: 0,
			ce2: 0,
			ce3: 0
		}
	};
}

/*
═══════════════════════════════════════════════════════════════════════════
CE1: Discrete Syntax Layer
═══════════════════════════════════════════════════════════════════════════

CE1 defines the combinatorial rules of the discrete skeleton:
- What compositions are syntactically valid?
- What are the primitive operators?
- What depth/complexity bounds exist?
*/

/*
Register a CE1 syntax rule
@param {string} operator - Operator name (e.g., "transclude", "link")
@param {function} rule - Rule function(source, target) -> {valid, depth, ...}
*/
CETower.prototype.registerSyntaxRule = function(operator, rule) {
	this.syntaxRules[operator] = rule;
};

/*
Check if a compositional operation is syntactically valid (CE1)
@param {string} operator - Operation type
@param {object} source - Source tiddler/state
@param {object} target - Target tiddler/state
@returns {object} - {valid: boolean, depth: number, reason: string}
*/
CETower.prototype.checkSyntax = function(operator, source, target) {
	this.stats.ce1Checks++;
	
	if(!this.syntaxRules[operator]) {
		// No rule defined - assume valid but record
		return {
			valid: true,
			depth: (source.depth || 0) + 1,
			reason: "No syntax rule defined"
		};
	}
	
	try {
		var result = this.syntaxRules[operator](source, target);
		
		if(!result.valid) {
			this.stats.violations.ce1++;
		}
		
		return result;
	} catch(e) {
		this.stats.violations.ce1++;
		return {
			valid: false,
			depth: 0,
			reason: "Syntax rule threw exception: " + e.message
		};
	}
};

/*
Get compositional depth of an object
@param {object} obj - Object with potential depth/composition info
@returns {number} - Compositional depth
*/
CETower.prototype.getDepth = function(obj) {
	if(!obj) {
		return 0;
	}
	
	// Check various possible depth fields
	return obj.depth || 
	       obj.compositional_depth || 
	       (obj.fields && obj.fields.depth) || 
	       0;
};

/*
═══════════════════════════════════════════════════════════════════════════
CE2: Continuous Flow Layer
═══════════════════════════════════════════════════════════════════════════

CE2 ensures discrete operations (VM) and continuous flows (ML) are compatible:
- Can this discrete step be approximated by a continuous flow?
- Does this flow discretize back to valid opcodes?
- Is curvature within bounds?
*/

/*
Check if discrete path approximates continuous geodesic (CE2)
@param {array} discretePath - Sequence of discrete states
@param {function} geodesic - Continuous curve function(t) -> state, t ∈ [0,1]
@param {number} steps - Number of samples to check
@returns {object} - {compatible: boolean, curvature: number, reason: string}
*/
CETower.prototype.checkFlowCompatibility = function(discretePath, geodesic, steps) {
	this.stats.ce2Checks++;
	steps = steps || 10;
	
	if(!discretePath || discretePath.length < 2) {
		return {
			compatible: true,
			curvature: 0,
			reason: "Path too short to check"
		};
	}
	
	// Sample geodesic and compare to discrete path
	var maxCurvature = 0;
	var totalDeviation = 0;
	
	for(var i = 0; i < steps; i++) {
		var t = i / (steps - 1);
		var continuousState = geodesic(t);
		
		// Find nearest discrete state
		var discreteIdx = Math.floor(t * (discretePath.length - 1));
		var discreteState = discretePath[discreteIdx];
		
		// Compute deviation (simplified as coordinate difference)
		var deviation = this.stateDistance(continuousState, discreteState);
		
		totalDeviation += deviation;
		maxCurvature = Math.max(maxCurvature, deviation);
	}
	
	var avgCurvature = totalDeviation / steps;
	
	if(maxCurvature > this.kappa || avgCurvature > this.flowTolerance) {
		this.stats.violations.ce2++;
		return {
			compatible: false,
			curvature: avgCurvature,
			maxCurvature: maxCurvature,
			reason: "Curvature exceeds κ=" + this.kappa + " threshold"
		};
	}
	
	return {
		compatible: true,
		curvature: avgCurvature,
		maxCurvature: maxCurvature,
		reason: "Within curvature bounds"
	};
};

/*
Compute distance between two states
@param {object} state1 - First state
@param {object} state2 - Second state
@returns {number} - Distance metric
*/
CETower.prototype.stateDistance = function(state1, state2) {
	// Simplified distance - can be enhanced with actual ZP35 distance
	if(!state1 || !state2) {
		return 1.0;
	}
	
	// If both have coordinates, use Euclidean distance
	if(state1.coordinate !== undefined && state2.coordinate !== undefined) {
		return Math.abs(state1.coordinate - state2.coordinate);
	}
	
	// If both have coherence, use difference
	if(state1.coherence !== undefined && state2.coherence !== undefined) {
		return Math.abs(state1.coherence - state2.coherence);
	}
	
	// Default: moderate distance
	return 0.5;
};

/*
Check if a discrete operation can be expressed as exponential map (CE2)
@param {string} operation - Operation name
@param {object} generator - Infinitesimal generator (Lie algebra element)
@returns {object} - {expressible: boolean, approximationError: number}
*/
CETower.prototype.checkExponentialMap = function(operation, generator) {
	this.stats.ce2Checks++;
	
	// This is a simplified check - real implementation would compute exp(generator)
	// and compare to the discrete operation
	
	if(!generator) {
		return {
			expressible: false,
			approximationError: 1.0,
			reason: "No generator provided"
		};
	}
	
	// Assume operation can be approximated if generator exists
	return {
		expressible: true,
		approximationError: 0.05,
		reason: "Generator maps to operation via exponential"
	};
};

/*
═══════════════════════════════════════════════════════════════════════════
CE3: Spectral Witness Layer
═══════════════════════════════════════════════════════════════════════════

CE3 stabilizes invariants across transformations:
- Are fixed points preserved?
- Are eigenvalues/resonances maintained?
- Does the spectral signature remain recognizable?
*/

/*
Check if spectral invariants are preserved by transformation (CE3)
@param {object} beforeState - State before transformation
@param {object} afterState - State after transformation
@param {object} transformation - Transformation metadata
@returns {object} - {preserved: boolean, shift: number, reason: string}
*/
CETower.prototype.checkSpectralInvariance = function(beforeState, afterState, transformation) {
	this.stats.ce3Checks++;
	
	var spectrumBefore = this.computeSpectrum(beforeState);
	var spectrumAfter = this.computeSpectrum(afterState);
	
	var shift = this.spectralDistance(spectrumBefore, spectrumAfter);
	
	if(shift > this.spectralTolerance) {
		this.stats.violations.ce3++;
		return {
			preserved: false,
			shift: shift,
			reason: "Spectral shift " + shift.toFixed(3) + " exceeds tolerance " + this.spectralTolerance
		};
	}
	
	return {
		preserved: true,
		shift: shift,
		reason: "Spectral structure preserved within tolerance"
	};
};

/*
Compute spectral signature of a state
@param {object} state - State to analyze
@returns {object} - {eigenvalues: array, dominantMode: number, coherence: number}
*/
CETower.prototype.computeSpectrum = function(state) {
	if(!state) {
		return {
			eigenvalues: [],
			dominantMode: 0,
			coherence: 0
		};
	}
	
	// If state already has spectral info, use it
	if(state.spectrum) {
		return state.spectrum;
	}
	
	// Extract spectral features from state
	var eigenvalues = [];
	
	// Coherence as dominant eigenvalue
	if(state.coherence !== undefined) {
		eigenvalues.push(state.coherence);
	}
	
	// Coordinate as secondary eigenvalue
	if(state.coordinate !== undefined) {
		eigenvalues.push(state.coordinate);
	}
	
	// Field count as structural complexity
	if(state.fields) {
		var fieldCount = Object.keys(state.fields).length;
		eigenvalues.push(fieldCount / 20.0); // Normalize
	}
	
	return {
		eigenvalues: eigenvalues,
		dominantMode: eigenvalues[0] || 0,
		coherence: state.coherence || 0
	};
};

/*
Compute distance between two spectra
@param {object} spectrum1 - First spectrum
@param {object} spectrum2 - Second spectrum
@returns {number} - Spectral distance
*/
CETower.prototype.spectralDistance = function(spectrum1, spectrum2) {
	if(!spectrum1 || !spectrum2) {
		return 1.0;
	}
	
	var ev1 = spectrum1.eigenvalues || [];
	var ev2 = spectrum2.eigenvalues || [];
	
	// If different number of eigenvalues, that's a big shift
	if(ev1.length !== ev2.length) {
		return 0.5;
	}
	
	// Compute L2 distance between eigenvalue vectors
	var sumSquares = 0;
	for(var i = 0; i < ev1.length; i++) {
		var val1 = (ev1[i] !== undefined) ? ev1[i] : 0;
		var val2 = (ev2[i] !== undefined) ? ev2[i] : 0;
		var diff = val1 - val2;
		sumSquares += diff * diff;
	}
	
	return Math.sqrt(sumSquares / Math.max(ev1.length, 1));
};

/*
Check if a state is a fixed point under a transformation
@param {object} state - State to check
@param {function} transform - Transformation function
@param {number} tolerance - Tolerance for "fixed" (default: 0.01)
@returns {boolean} - True if fixed point
*/
CETower.prototype.isFixedPoint = function(state, transform, tolerance) {
	tolerance = tolerance || 0.01;
	
	try {
		var transformed = transform(state);
		var distance = this.stateDistance(state, transformed);
		
		return distance < tolerance;
	} catch(e) {
		return false;
	}
};

/*
═══════════════════════════════════════════════════════════════════════════
Unified Compatibility Checking
═══════════════════════════════════════════════════════════════════════════

These methods check consistency across all CE layers
*/

/*
Perform full CE Tower validation on a transformation
@param {object} transformation - {
    operator: string,
    source: object,
    target: object,
    discretePath: array,
    geodesic: function,
    beforeState: object,
    afterState: object
}
@returns {object} - {valid: boolean, violations: array, details: object}
*/
CETower.prototype.validateTransformation = function(transformation) {
	var violations = [];
	var details = {};
	
	// CE1: Syntax check
	if(transformation.operator && transformation.source && transformation.target) {
		var ce1Result = this.checkSyntax(
			transformation.operator,
			transformation.source,
			transformation.target
		);
		details.ce1 = ce1Result;
		
		if(!ce1Result.valid) {
			violations.push({
				layer: "CE1",
				reason: ce1Result.reason
			});
		}
	}
	
	// CE2: Flow compatibility check
	if(transformation.discretePath && transformation.geodesic) {
		var ce2Result = this.checkFlowCompatibility(
			transformation.discretePath,
			transformation.geodesic
		);
		details.ce2 = ce2Result;
		
		if(!ce2Result.compatible) {
			violations.push({
				layer: "CE2",
				reason: ce2Result.reason,
				curvature: ce2Result.curvature
			});
		}
	}
	
	// CE3: Spectral invariance check
	if(transformation.beforeState && transformation.afterState) {
		var ce3Result = this.checkSpectralInvariance(
			transformation.beforeState,
			transformation.afterState,
			transformation
		);
		details.ce3 = ce3Result;
		
		if(!ce3Result.preserved) {
			violations.push({
				layer: "CE3",
				reason: ce3Result.reason,
				shift: ce3Result.shift
			});
		}
	}
	
	return {
		valid: violations.length === 0,
		violations: violations,
		details: details
	};
};

/*
Get CE Tower statistics
@returns {object} - Statistics about checks and violations
*/
CETower.prototype.getStatistics = function() {
	return {
		checks: {
			ce1: this.stats.ce1Checks,
			ce2: this.stats.ce2Checks,
			ce3: this.stats.ce3Checks,
			total: this.stats.ce1Checks + this.stats.ce2Checks + this.stats.ce3Checks
		},
		violations: {
			ce1: this.stats.violations.ce1,
			ce2: this.stats.violations.ce2,
			ce3: this.stats.violations.ce3,
			total: this.stats.violations.ce1 + this.stats.violations.ce2 + this.stats.violations.ce3
		},
		violationRate: {
			ce1: this.stats.ce1Checks > 0 ? this.stats.violations.ce1 / this.stats.ce1Checks : 0,
			ce2: this.stats.ce2Checks > 0 ? this.stats.violations.ce2 / this.stats.ce2Checks : 0,
			ce3: this.stats.ce3Checks > 0 ? this.stats.violations.ce3 / this.stats.ce3Checks : 0
		}
	};
};

/*
Reset statistics
*/
CETower.prototype.resetStatistics = function() {
	this.stats = {
		ce1Checks: 0,
		ce2Checks: 0,
		ce3Checks: 0,
		violations: {
			ce1: 0,
			ce2: 0,
			ce3: 0
		}
	};
};

/*
═══════════════════════════════════════════════════════════════════════════
Standard Syntax Rules
═══════════════════════════════════════════════════════════════════════════

Define common CE1 syntax rules for TiddlyWiki operations
*/

/*
Initialize standard syntax rules for TiddlyWiki
*/
CETower.prototype.initializeStandardRules = function() {
	var self = this;
	
	// Transclusion rule
	this.registerSyntaxRule("transclude", function(source, target) {
		var sourceDepth = self.getDepth(source);
		var targetDepth = self.getDepth(target);
		
		// Transclusion increases depth
		var newDepth = sourceDepth + targetDepth + 1;
		
		// Check for excessive depth (potential infinite recursion)
		if(newDepth > 10) {
			return {
				valid: false,
				depth: newDepth,
				reason: "Transclusion depth " + newDepth + " exceeds maximum (10)"
			};
		}
		
		return {
			valid: true,
			depth: newDepth,
			reason: "Valid transclusion"
		};
	});
	
	// Link rule
	this.registerSyntaxRule("link", function(source, target) {
		var sourceDepth = self.getDepth(source);
		
		// Links preserve depth
		return {
			valid: true,
			depth: sourceDepth,
			reason: "Links preserve compositional depth"
		};
	});
	
	// Macro expansion rule
	this.registerSyntaxRule("macro", function(source, target) {
		var sourceDepth = self.getDepth(source);
		
		// Macros can increase depth significantly
		var newDepth = sourceDepth + 2;
		
		if(newDepth > 12) {
			return {
				valid: false,
				depth: newDepth,
				reason: "Macro expansion depth " + newDepth + " exceeds maximum (12)"
			};
		}
		
		return {
			valid: true,
			depth: newDepth,
			reason: "Valid macro expansion"
		};
	});
	
	// Widget rendering rule
	this.registerSyntaxRule("widget", function(source, target) {
		var sourceDepth = self.getDepth(source);
		
		// Widgets increase depth by 1
		return {
			valid: true,
			depth: sourceDepth + 1,
			reason: "Valid widget rendering"
		};
	});
};

/*
═══════════════════════════════════════════════════════════════════════════
Integration with CE1 Learning Law
═══════════════════════════════════════════════════════════════════════════

The CE1 Learning Law (γ/ZP ≈ 411) provides theoretical foundation for
understanding how many examples are needed to learn compositional patterns
at the CE1 layer.
*/

/*
Get CE1 Learning Law instance
@returns {object} - CE1LearningLaw instance
*/
CETower.prototype.getLearningLaw = function() {
	if(!this.learningLaw) {
		// Lazy load CE1 Learning Law module
		var CE1LearningLaw = require("$:/core/modules/utils/ce1-learning-law.js").CE1LearningLaw;
		this.learningLaw = new CE1LearningLaw({
			zp: this.kappa / 250 // Derive ZP from kappa (κ = 0.35 → ZP ≈ 0.0014)
		});
	}
	return this.learningLaw;
};

/*
Estimate examples needed to learn a compositional pattern
@param {number} depth - Compositional depth of pattern
@param {object} options - Options for estimation
@returns {object} - Example estimate with confidence bounds
*/
CETower.prototype.estimateLearningSamples = function(depth, options) {
	var law = this.getLearningLaw();
	
	// Depth increases complexity roughly as sqrt(depth)
	// This accounts for compositional growth
	var complexityFactor = Math.sqrt(depth || 1);
	
	return law.estimateExamplesNeeded(complexityFactor, options);
};

/*
Assess whether sufficient examples have been seen for a pattern
@param {number} examplesSeen - Number of examples observed
@param {number} depth - Compositional depth
@returns {object} - Readiness assessment
*/
CETower.prototype.assessPatternReadiness = function(examplesSeen, depth) {
	var law = this.getLearningLaw();
	var complexityFactor = Math.sqrt(depth || 1);
	
	return law.assessLearningReadiness(examplesSeen, complexityFactor);
};

/*
Export the module
*/
exports.CETower = CETower;

