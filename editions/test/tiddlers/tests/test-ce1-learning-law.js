/*\
title: test-ce1-learning-law.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for CE1 Learning Law implementation

\*/

(function() {

	/* global $tw */

	describe("CE1 Learning Law", function() {
	
		var CE1LearningLaw;
	
		// Setup
		beforeEach(function() {
			CE1LearningLaw = require("$:/core/modules/utils/ce1-learning-law.js").CE1LearningLaw;
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	Construction and Constants
	═══════════════════════════════════════════════════════════════════════
	*/
	
		// Test precision constants
	var GAMMA_PRECISION = 9;  // 9 decimal places for Euler-Mascheroni constant
	var ZP_PRECISION = 4;     // 4 decimal places for ZP coordinate

	describe("Construction and Constants", function() {
		
			it("should create CE1 Learning Law with default parameters", function() {
				var law = new CE1LearningLaw();
			
				expect(law).toBeDefined();
				expect(law.gamma).toBeCloseTo(0.5772156649, GAMMA_PRECISION);
				expect(law.zp).toBeCloseTo(0.0014, ZP_PRECISION);
			});
		
			it("should calculate learning constant from γ/ZP", function() {
				var law = new CE1LearningLaw();
				var expected = law.gamma / law.zp;
			
				expect(law.learningConstant).toBeCloseTo(expected, 2);
				expect(law.learningConstant).toBeCloseTo(412.3, 1);
			});
		
			it("should accept custom ZP coordinate", function() {
				var law = new CE1LearningLaw({ zp: 0.002 });
			
				expect(law.zp).toBe(0.002);
				expect(law.learningConstant).toBeCloseTo(288.6, 1);
			});
		
			it("should accept custom variance factor", function() {
				var law = new CE1LearningLaw({ varianceFactor: 0.3 });
			
				expect(law.varianceFactor).toBe(0.3);
			});
		
			it("should export Euler-Mascheroni constant", function() {
				var gamma = require("$:/core/modules/utils/ce1-learning-law.js").EULER_MASCHERONI_GAMMA;
			
				expect(gamma).toBeCloseTo(0.5772156649, GAMMA_PRECISION);
			});
		
			it("should export CE1 ZP coordinate", function() {
				var zp = require("$:/core/modules/utils/ce1-learning-law.js").CE1_ZP_COORDINATE;
			
				expect(zp).toBe(0.0014);
			});
		
			it("should export learning constant", function() {
				var constant = require("$:/core/modules/utils/ce1-learning-law.js").CE1_LEARNING_CONSTANT;
			
				expect(constant).toBeGreaterThan(400);
				expect(constant).toBeLessThan(420);
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	Core Learning Constant Methods
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("Core Methods", function() {
		
			it("should return learning constant", function() {
				var law = new CE1LearningLaw();
				var constant = law.getLearningConstant();
			
				expect(constant).toBeCloseTo(412.3, 1);
			});
		
			it("should return gamma", function() {
				var law = new CE1LearningLaw();
				var gamma = law.getGamma();
			
				expect(gamma).toBeCloseTo(0.5772156649, GAMMA_PRECISION);
			});
		
			it("should return ZP coordinate", function() {
				var law = new CE1LearningLaw();
				var zp = law.getZP();
			
				expect(zp).toBe(0.0014);
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	Examples Estimation
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("Examples Estimation", function() {
		
			it("should estimate base examples needed", function() {
				var law = new CE1LearningLaw();
				var estimate = law.estimateExamplesNeeded(1.0, { includeVariance: false });
			
				expect(estimate.expected).toBeCloseTo(412, 0);
				expect(estimate.min).toBeCloseTo(412, 0);
				expect(estimate.max).toBeCloseTo(412, 0);
			});
		
			it("should estimate with variance", function() {
				var law = new CE1LearningLaw();
				var estimate = law.estimateExamplesNeeded(1.0);
			
				expect(estimate.expected).toBeCloseTo(412, 0);
				expect(estimate.min).toBeLessThan(estimate.expected);
				expect(estimate.max).toBeGreaterThan(estimate.expected);
			});
		
			it("should scale with complexity factor", function() {
				var law = new CE1LearningLaw();
				var simple = law.estimateExamplesNeeded(0.5, { includeVariance: false });
				var complex = law.estimateExamplesNeeded(2.0, { includeVariance: false });
			
				// Check proportional scaling
				expect(complex.expected / simple.expected).toBeCloseTo(4.0, 0);
				expect(simple.expected).toBeGreaterThan(200);
				expect(simple.expected).toBeLessThan(210);
				expect(complex.expected).toBeGreaterThan(820);
				expect(complex.expected).toBeLessThan(830);
			});
		
			it("should respect confidence levels", function() {
				var law = new CE1LearningLaw();
				var low = law.estimateExamplesNeeded(1.0, { confidenceLevel: 0.68 });
				var high = law.estimateExamplesNeeded(1.0, { confidenceLevel: 0.99 });
			
				// Higher confidence should have wider range
				var lowRange = low.max - low.min;
				var highRange = high.max - high.min;
			
				expect(highRange).toBeGreaterThan(lowRange);
			});
		
			it("should return proper estimate structure", function() {
				var law = new CE1LearningLaw();
				var estimate = law.estimateExamplesNeeded(1.0);
			
				expect(estimate.min).toBeDefined();
				expect(estimate.expected).toBeDefined();
				expect(estimate.max).toBeDefined();
				expect(estimate.explanation).toBeDefined();
				expect(estimate.complexityFactor).toBe(1.0);
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	Domain-Specific Estimates
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("Domain-Specific Estimates", function() {
		
			it("should estimate motor pattern learning", function() {
				var law = new CE1LearningLaw();
				var simple = law.estimateMotorPatternLearning({ 
					complexity: "simple",
					includeVariance: false 
				});
				var complex = law.estimateMotorPatternLearning({ 
					complexity: "complex",
					includeVariance: false 
				});
			
				expect(simple.expected).toBeGreaterThan(200); // ~200
				expect(simple.expected).toBeLessThan(210);
				expect(complex.expected).toBeGreaterThan(490); // ~500
				expect(complex.expected).toBeLessThan(500);
				expect(simple.domain).toBe("motor pattern learning");
			});
		
			it("should estimate category learning", function() {
				var law = new CE1LearningLaw();
				var high = law.estimateCategoryLearning({ 
					distinctiveness: "high",
					includeVariance: false 
				});
				var low = law.estimateCategoryLearning({ 
					distinctiveness: "low",
					includeVariance: false 
				});
			
				expect(high.expected).toBeCloseTo(309, 0); // ~300
				expect(low.expected).toBeCloseTo(618, 0); // ~600
				expect(high.domain).toBe("category learning");
			});
		
			it("should estimate grammar learning", function() {
				var law = new CE1LearningLaw();
				var simple = law.estimateGrammarLearning({ 
					grammarComplexity: "simple",
					includeVariance: false 
				});
				var complex = law.estimateGrammarLearning({ 
					grammarComplexity: "complex",
					includeVariance: false 
				});
			
				expect(simple.expected).toBeCloseTo(371, 0); // ~370
				expect(complex.expected).toBeCloseTo(453, 0); // ~450
				expect(simple.domain).toBe("grammar learning");
			});
		
			it("should estimate style learning", function() {
				var law = new CE1LearningLaw();
				var high = law.estimateStyleLearning({ 
					styleConsistency: "high",
					includeVariance: false 
				});
				var low = law.estimateStyleLearning({ 
					styleConsistency: "low",
					includeVariance: false 
				});
			
				expect(high.expected).toBeCloseTo(330, 0); // ~330
				expect(low.expected).toBeCloseTo(536, 0); // ~530
				expect(high.domain).toBe("style learning");
			});
		
			it("should estimate regression learning", function() {
				var law = new CE1LearningLaw();
				var lowNoise = law.estimateRegressionLearning({ 
					noiseLevel: "low",
					includeVariance: false 
				});
				var highNoise = law.estimateRegressionLearning({ 
					noiseLevel: "high",
					includeVariance: false 
				});
			
				expect(lowNoise.expected).toBeGreaterThan(365); // ~370
				expect(lowNoise.expected).toBeLessThan(375);
				expect(highNoise.expected).toBeGreaterThan(490); // ~490
				expect(highNoise.expected).toBeLessThan(500);
				expect(lowNoise.domain).toBe("regression learning");
			});
		
			it("should use defaults for unspecified options", function() {
				var law = new CE1LearningLaw();
				var motor = law.estimateMotorPatternLearning();
				var category = law.estimateCategoryLearning();
				var grammar = law.estimateGrammarLearning();
				var style = law.estimateStyleLearning();
				var regression = law.estimateRegressionLearning();
			
				// All should return reasonable values around 400
				expect(motor.expected).toBeGreaterThan(200);
				expect(motor.expected).toBeLessThan(600);
				expect(category.expected).toBeGreaterThan(200);
				expect(category.expected).toBeLessThan(600);
				expect(grammar.expected).toBeGreaterThan(200);
				expect(grammar.expected).toBeLessThan(600);
				expect(style.expected).toBeGreaterThan(200);
				expect(style.expected).toBeLessThan(600);
				expect(regression.expected).toBeGreaterThan(200);
				expect(regression.expected).toBeLessThan(600);
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	Curvature Gap Analysis
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("Curvature Gap Analysis", function() {
		
			it("should analyze gap at start of learning", function() {
				var law = new CE1LearningLaw();
				var gap = law.analyzeCurvatureGap(0);
			
				expect(gap.currentExamples).toBe(0);
				expect(gap.progressRatio).toBe(0);
				expect(gap.remainingCurvature).toBeCloseTo(law.gamma, GAMMA_PRECISION);
				expect(gap.phase).toBe("early");
				expect(gap.converged).toBe(false);
			});
		
			it("should analyze gap at midpoint", function() {
				var law = new CE1LearningLaw();
				var midpoint = law.learningConstant / 2;
				var gap = law.analyzeCurvatureGap(midpoint);
			
				expect(gap.progressRatio).toBeCloseTo(0.5, 1);
				expect(gap.remainingCurvature).toBeCloseTo(law.gamma / 2, 2);
				expect(gap.phase).toBe("middle");
				expect(gap.converged).toBe(false);
			});
		
			it("should analyze gap near completion", function() {
				var law = new CE1LearningLaw();
				var nearEnd = law.learningConstant * 0.9;
				var gap = law.analyzeCurvatureGap(nearEnd);
			
				expect(gap.progressRatio).toBeCloseTo(0.9, 1);
				expect(gap.phase).toBe("late");
				expect(gap.converged).toBe(false);
			});
		
			it("should detect convergence", function() {
				var law = new CE1LearningLaw();
				var gap = law.analyzeCurvatureGap(law.learningConstant * 1.2);
			
				expect(gap.progressRatio).toBeGreaterThan(1.0);
				expect(gap.phase).toBe("converged");
				expect(gap.converged).toBe(true);
				expect(gap.remainingExamples).toBe(0);
			});
		
			it("should track learning phases correctly", function() {
				var law = new CE1LearningLaw();
				var total = law.learningConstant;
			
				var early = law.analyzeCurvatureGap(total * 0.1);
				var middle = law.analyzeCurvatureGap(total * 0.5);
				var late = law.analyzeCurvatureGap(total * 0.85);
				var converged = law.analyzeCurvatureGap(total * 1.1);
			
				expect(early.phase).toBe("early");
				expect(middle.phase).toBe("middle");
				expect(late.phase).toBe("late");
				expect(converged.phase).toBe("converged");
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	Witness Contraction
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("Witness Contraction", function() {
		
			it("should calculate contraction for zero examples", function() {
				var law = new CE1LearningLaw();
				var contraction = law.calculateWitnessContraction(0);
			
				expect(contraction.totalContraction).toBe(0);
				expect(contraction.gapClosedRatio).toBe(0);
				expect(contraction.smoothingRate).toBe(0);
				expect(contraction.remainingRoughness).toBe(1);
			});
		
			it("should calculate contraction proportional to ZP", function() {
				var law = new CE1LearningLaw();
				var contraction = law.calculateWitnessContraction(100);
			
				expect(contraction.contractionPerExample).toBe(law.zp);
				expect(contraction.totalContraction).toBeCloseTo(law.zp * 100, 4);
			});
		
			it("should calculate gap closure", function() {
				var law = new CE1LearningLaw();
				var halfwayExamples = law.learningConstant / 2;
				var contraction = law.calculateWitnessContraction(halfwayExamples);
			
				expect(contraction.gapClosedRatio).toBeCloseTo(0.5, 1);
				expect(contraction.gapClosedPercent).toBeCloseTo(50, 0);
			});
		
			it("should cap smoothing rate at 1.0", function() {
				var law = new CE1LearningLaw();
				var manyExamples = law.learningConstant * 2;
				var contraction = law.calculateWitnessContraction(manyExamples);
			
				expect(contraction.smoothingRate).toBe(1.0);
				expect(contraction.remainingRoughness).toBe(0);
			});
		
			it("should track gamma gap correctly", function() {
				var law = new CE1LearningLaw();
				var contraction = law.calculateWitnessContraction(100);
			
				expect(contraction.gammaGap).toBe(law.gamma);
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	Learning Readiness Assessment
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("Learning Readiness", function() {
		
			it("should assess insufficient examples", function() {
				var law = new CE1LearningLaw();
				var assessment = law.assessLearningReadiness(100);
			
				expect(assessment.status).toBe("insufficient");
				expect(assessment.confidence).toBeLessThan(0.3);
				expect(assessment.recommendation).toContain("more examples");
			});
		
			it("should assess developing learning", function() {
				var law = new CE1LearningLaw();
				var assessment = law.assessLearningReadiness(law.learningConstant * 0.6);
			
				expect(assessment.status).toBe("developing");
				expect(assessment.confidence).toBeGreaterThan(0.2);
				expect(assessment.confidence).toBeLessThan(0.5);
			});
		
			it("should assess approaching readiness", function() {
				var law = new CE1LearningLaw();
				var assessment = law.assessLearningReadiness(law.learningConstant * 0.9);
			
				expect(assessment.status).toBe("approaching");
				expect(assessment.confidence).toBeGreaterThan(0.4);
				expect(assessment.confidence).toBeLessThan(0.7);
			});
		
			it("should assess sufficient examples", function() {
				var law = new CE1LearningLaw();
				var assessment = law.assessLearningReadiness(law.learningConstant * 1.1);
			
				expect(assessment.status).toBe("sufficient");
				expect(assessment.confidence).toBeGreaterThan(0.6);
			});
		
			it("should assess robust learning", function() {
				var law = new CE1LearningLaw();
				var assessment = law.assessLearningReadiness(law.learningConstant * 2.0);
			
				expect(assessment.status).toBe("robust");
				expect(assessment.confidence).toBeGreaterThan(0.85);
			});
		
			it("should scale with complexity factor", function() {
				var law = new CE1LearningLaw();
				var simple = law.assessLearningReadiness(300, 0.5);
				var complex = law.assessLearningReadiness(300, 2.0);
			
				// Same examples, different complexity
				expect(simple.status).not.toBe("insufficient");
				expect(complex.status).toBe("insufficient");
			});
		
			it("should provide appropriate recommendations", function() {
				var law = new CE1LearningLaw();
			
				var insufficient = law.assessLearningReadiness(100);
				var sufficient = law.assessLearningReadiness(law.learningConstant * 1.1);
				var robust = law.assessLearningReadiness(law.learningConstant * 2.0);
			
				expect(insufficient.recommendation).toContain("Need");
				expect(sufficient.recommendation).toContain("Sufficient");
				expect(robust.recommendation).toMatch(/robust|above/i);
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	Optimal Step Size
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("Optimal Step Size", function() {
		
			it("should calculate base step size from ZP", function() {
				var law = new CE1LearningLaw();
				var step = law.calculateOptimalStepSize(0.3);
			
				expect(step.baseStepSize).toBe(law.zp);
			});
		
			it("should adapt to small remaining gaps", function() {
				var law = new CE1LearningLaw();
				var step = law.calculateOptimalStepSize(0.05);
			
				expect(step.adaptiveFactor).toBeLessThan(1.0);
				expect(step.recommendedStepSize).toBeLessThan(step.baseStepSize);
			});
		
			it("should adapt to large remaining gaps", function() {
				var law = new CE1LearningLaw();
				var step = law.calculateOptimalStepSize(0.6);
			
				expect(step.adaptiveFactor).toBeGreaterThan(1.0);
				expect(step.recommendedStepSize).toBeGreaterThan(step.baseStepSize);
			});
		
			it("should recommend reasonable examples per step", function() {
				var law = new CE1LearningLaw();
				var step = law.calculateOptimalStepSize(0.3);
			
				expect(step.examplesPerStep).toBeGreaterThan(0);
				expect(Number.isInteger(step.examplesPerStep)).toBe(true);
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	Summary and Integration
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("Summary", function() {
		
			it("should provide complete summary", function() {
				var law = new CE1LearningLaw();
				var summary = law.getSummary();
			
				expect(summary.constants).toBeDefined();
				expect(summary.constants.gamma).toBeDefined();
				expect(summary.constants.zp).toBeDefined();
				expect(summary.constants.learningConstant).toBeDefined();
			
				expect(summary.interpretation).toBeDefined();
				expect(summary.formula).toBeDefined();
			});
		
			it("should include all key constants in summary", function() {
				var law = new CE1LearningLaw();
				var summary = law.getSummary();
			
				expect(summary.constants.gamma).toBeCloseTo(0.5772, ZP_PRECISION);
				expect(summary.constants.zp).toBe(0.0014);
				expect(summary.constants.learningConstant).toBeGreaterThan(400);
			});
		
			it("should provide interpretations", function() {
				var law = new CE1LearningLaw();
				var summary = law.getSummary();
			
				expect(summary.interpretation.gamma).toContain("gap");
				expect(summary.interpretation.zp).toContain("coherence");
				expect(summary.interpretation.learningConstant).toContain("Examples");
			});
		
			it("should format formula correctly", function() {
				var law = new CE1LearningLaw();
				var summary = law.getSummary();
			
				expect(summary.formula).toContain("γ / ZP");
				expect(summary.formula).toContain("≈");
			});
		
		});
	
		/*
	═══════════════════════════════════════════════════════════════════════
	Integration Tests
	═══════════════════════════════════════════════════════════════════════
	*/
	
		describe("Integration", function() {
		
			it("should work with different ZP values", function() {
				var highZP = new CE1LearningLaw({ zp: 0.005 });
				var lowZP = new CE1LearningLaw({ zp: 0.001 });
			
				// Higher ZP means fewer examples needed
				expect(highZP.learningConstant).toBeLessThan(lowZP.learningConstant);
			});
		
			it("should maintain consistency across methods", function() {
				var law = new CE1LearningLaw();
				var examples = law.learningConstant;
			
				var gap = law.analyzeCurvatureGap(examples);
				var contraction = law.calculateWitnessContraction(examples);
				var readiness = law.assessLearningReadiness(examples);
			
				expect(gap.converged).toBe(true);
				expect(contraction.gapClosedRatio).toBeGreaterThanOrEqual(1.0);
				expect(readiness.status).not.toBe("insufficient");
			});
		
			it("should handle edge cases gracefully", function() {
				var law = new CE1LearningLaw();
			
				// Zero examples
				expect(function() {
					law.analyzeCurvatureGap(0);
				}).not.toThrow();
			
				// Negative complexity (should still work)
				expect(function() {
					law.estimateExamplesNeeded(-1);
				}).not.toThrow();
			
				// Very large examples
				expect(function() {
					law.assessLearningReadiness(1000000);
				}).not.toThrow();
			});
		
			it("should verify the ~400 examples intuition", function() {
				var law = new CE1LearningLaw();
				var constant = law.getLearningConstant();
			
				// The learning constant should be around 400
				expect(constant).toBeGreaterThan(380);
				expect(constant).toBeLessThan(450);
			
				// Should match the problem statement's "about 400"
				expect(Math.round(constant / 10) * 10).toBeCloseTo(410, 0);
			});
		
		});
	
	});

})();
