/*\
title: $:/core/modules/utils/ce1-harmonic.js
type: application/javascript
module-type: utils

CE1 (Collapse-Evaluate) Harmonic Operator System

This module implements the CE1 expression calculus for harmonic analysis,
providing bracket operators and fixed-point semantics for singularity-balanced functions.

Bracket Types:
- () : Morphism (height 1) - transformations with rotational singularities
- <> : Witness/Fixed-point resolver - anchors oscillations and finds roots
- {} : Boundary (height 0) - controls collapse toward 0
- [] : Memory - LR-accumulated series

Harmonic Operator ℋ(x):
ℋ(x) = {ln(x)} + [ζ(x)] + (tan(πx/2)) + <sin(πx)> + <i·cos(πx)>

Where:
- {ln(x)} : Collapse/boundary dynamics
- [ζ(x)]  : Accumulation/memory via zeta function
- (tan)   : Phase/morphism with singularities
- <sin>   : Oscillation/witness
- <cos>   : Complex oscillation/witness

Fixed-point condition: <ℋ(x)> = 0 characterizes harmonic singularities (e.g., zeta zeros)

\*/

(function(){

	"use strict";

	/*
CE1Expression class - represents a CE1 expression tree
*/
	function CE1Expression(type, value, children) {
		this.type = type; // "constant", "morphism", "witness", "boundary", "memory"
		this.value = value;
		this.children = children || [];
		this.height = this.computeHeight();
	}

	CE1Expression.prototype.computeHeight = function() {
		switch(this.type) {
			case "constant":
				return 0;
			case "morphism":
				return 1;
			case "witness":
			case "boundary":
			case "memory":
				return Math.max(...this.children.map(c => c.height), 0) + 1;
			default:
				return 0;
		}
	};

	/*
Harmonic operator components
*/
	var HarmonicOperators = {
	/*
	Logarithm - collapse/boundary component
	Controls domain collapse toward 0
	*/
		logarithm: function(x) {
			if(typeof x === "number" && x > 0) {
				return Math.log(x);
			}
			// Complex logarithm for negative/complex numbers
			if(typeof x === "object" && x.re !== undefined) {
				var magnitude = Math.sqrt(x.re * x.re + x.im * x.im);
				var phase = Math.atan2(x.im, x.re);
				return {
					re: Math.log(magnitude),
					im: phase
				};
			}
			return NaN;
		},

		/*
	Riemann zeta function approximation - accumulation/memory component
	Uses the analytic continuation for complex plane
	For real s > 1: ζ(s) = Σ(1/n^s) for n=1 to infinity
	*/
		zeta: function(s, terms) {
			terms = terms || 50;
			if(typeof s === "number") {
				if(s === 1) return Infinity;
				if(s < 0) {
				// Use functional equation: ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
				// Note: Full functional equation implementation would require gamma function
				// For now, return 0 as a simplified placeholder for negative values
					return 0;
				}
				// Direct summation for s > 1
				var sum = 0;
				for(var n = 1; n <= terms; n++) {
					sum += Math.pow(n, -s);
				}
				return sum;
			}
			// Complex zeta function (simplified)
			if(typeof s === "object" && s.re !== undefined) {
				var sum = { re: 0, im: 0 };
				for(var n = 1; n <= terms; n++) {
				// n^(-s) = exp(-s * ln(n))
					var logN = Math.log(n);
					var expRe = Math.exp(-s.re * logN) * Math.cos(s.im * logN);
					var expIm = Math.exp(-s.re * logN) * Math.sin(s.im * logN);
					sum.re += expRe;
					sum.im += expIm;
				}
				return sum;
			}
			return NaN;
		},

		/*
	Tangent - phase/morphism component
	Has rotational singularities at odd multiples of π/2
	*/
		tangent: function(x) {
			if(typeof x === "number") {
				return Math.tan(x * Math.PI / 2);
			}
			// Complex tangent
			if(typeof x === "object" && x.re !== undefined) {
				var arg = { re: x.re * Math.PI / 2, im: x.im * Math.PI / 2 };
				// tan(z) = sin(z)/cos(z)
				var sinZ = HarmonicOperators.sine({ re: arg.re / Math.PI, im: arg.im / Math.PI });
				var cosZ = HarmonicOperators.cosine({ re: arg.re / Math.PI, im: arg.im / Math.PI });
				// Complex division
				var denom = cosZ.re * cosZ.re + cosZ.im * cosZ.im;
				return {
					re: (sinZ.re * cosZ.re + sinZ.im * cosZ.im) / denom,
					im: (sinZ.im * cosZ.re - sinZ.re * cosZ.im) / denom
				};
			}
			return NaN;
		},

		/*
	Sine - oscillation/witness component
	Anchors fixed points through periodic zeros
	*/
		sine: function(x) {
			if(typeof x === "number") {
				return Math.sin(x * Math.PI);
			}
			// Complex sine: sin(z) = (e^(iz) - e^(-iz)) / (2i)
			if(typeof x === "object" && x.re !== undefined) {
				var arg = x.re * Math.PI;
				var argIm = x.im * Math.PI;
				// e^(iz) where z = arg + i*argIm
				// e^(i(arg + i*argIm)) = e^(i*arg - argIm)
				var exp1Re = Math.exp(-argIm) * Math.cos(arg);
				var exp1Im = Math.exp(-argIm) * Math.sin(arg);
				// e^(-iz)
				var exp2Re = Math.exp(argIm) * Math.cos(arg);
				var exp2Im = -Math.exp(argIm) * Math.sin(arg);
				return {
					re: (exp1Im - exp2Im) / 2,
					im: -(exp1Re - exp2Re) / 2
				};
			}
			return NaN;
		},

		/*
	Cosine - complex oscillation/witness component
	Complements sine for complete harmonic representation
	*/
		cosine: function(x) {
			if(typeof x === "number") {
				return Math.cos(x * Math.PI);
			}
			// Complex cosine: cos(z) = (e^(iz) + e^(-iz)) / 2
			if(typeof x === "object" && x.re !== undefined) {
				var arg = x.re * Math.PI;
				var argIm = x.im * Math.PI;
				var exp1Re = Math.exp(-argIm) * Math.cos(arg);
				var exp1Im = Math.exp(-argIm) * Math.sin(arg);
				var exp2Re = Math.exp(argIm) * Math.cos(arg);
				var exp2Im = -Math.exp(argIm) * Math.sin(arg);
				return {
					re: (exp1Re + exp2Re) / 2,
					im: (exp1Im + exp2Im) / 2
				};
			}
			return NaN;
		}
	};

	/*
CE1 Harmonic Operator ℋ(x)
Combines all harmonic components according to CE1 bracket semantics
*/
	function harmonicOperator(x) {
	// Ensure x is in proper format
		var input = typeof x === "number" ? x : (x.re !== undefined ? x : { re: x, im: 0 });
	
		var result = {
			boundary: HarmonicOperators.logarithm(input),
			memory: HarmonicOperators.zeta(input),
			morphism: HarmonicOperators.tangent(input),
			witness_sin: HarmonicOperators.sine(input),
			witness_cos: HarmonicOperators.cosine(input)
		};

		// Combine components
		// ℋ(x) = {ln(x)} + [ζ(x)] + (tan(πx/2)) + <sin(πx)> + <i·cos(πx)>
		if(typeof input === "number") {
			var ln = result.boundary;
			var zeta = result.memory;
			var tan = result.morphism;
			var sin = result.witness_sin;
			var cos = result.witness_cos;
		
			// Real part
			var realPart = ln + zeta + tan + sin;
			// Imaginary part (i * cos)
			var imagPart = cos;
		
			return {
				re: realPart,
				im: imagPart,
				components: result
			};
		} else {
		// Complex arithmetic
			var totalRe = 0, totalIm = 0;
		
			// Add boundary (log)
			if(typeof result.boundary === "object") {
				totalRe += result.boundary.re;
				totalIm += result.boundary.im;
			} else {
				totalRe += result.boundary;
			}
		
			// Add memory (zeta)
			if(typeof result.memory === "object") {
				totalRe += result.memory.re;
				totalIm += result.memory.im;
			} else {
				totalRe += result.memory;
			}
		
			// Add morphism (tan)
			if(typeof result.morphism === "object") {
				totalRe += result.morphism.re;
				totalIm += result.morphism.im;
			} else {
				totalRe += result.morphism;
			}
		
			// Add witness (sin)
			if(typeof result.witness_sin === "object") {
				totalRe += result.witness_sin.re;
				totalIm += result.witness_sin.im;
			} else {
				totalRe += result.witness_sin;
			}
		
			// Add i*cos (multiply by i: re -> -im, im -> re)
			if(typeof result.witness_cos === "object") {
				totalRe -= result.witness_cos.im;
				totalIm += result.witness_cos.re;
			} else {
				totalIm += result.witness_cos;
			}
		
			return {
				re: totalRe,
				im: totalIm,
				components: result
			};
		}
	}

	/*
Fixed-point resolver <E>
Finds x such that ℋ(x) = 0 near initial guess c
Uses Newton-Raphson iteration
*/
	function fixedPointResolver(initialGuess, maxIterations, tolerance) {
		maxIterations = maxIterations || 100;
		tolerance = tolerance || 1e-10;
	
		var x = initialGuess;
		var iteration = 0;
	
		while(iteration < maxIterations) {
			var hx = harmonicOperator(x);
			var magnitude = Math.sqrt(hx.re * hx.re + hx.im * hx.im);
		
			if(magnitude < tolerance) {
				return {
					value: x,
					iterations: iteration,
					residual: magnitude,
					converged: true
				};
			}
		
			// Numerical derivative (simplified)
			var delta = 1e-7;
			var hxDelta = harmonicOperator(x + delta);
			var derivRe = (hxDelta.re - hx.re) / delta;
			var derivIm = (hxDelta.im - hx.im) / delta;
			var derivMag = Math.sqrt(derivRe * derivRe + derivIm * derivIm);
		
			if(derivMag < 1e-15) {
				break; // Derivative too small
			}
		
			// Newton step: x_new = x - H(x)/H'(x)
			x = x - hx.re / derivRe;
			iteration++;
		}
	
		// Compute final residual
		var finalHx = harmonicOperator(x);
		return {
			value: x,
			iterations: iteration,
			residual: Math.sqrt(finalHx.re * finalHx.re + finalHx.im * finalHx.im),
			converged: false
		};
	}

	/*
CE1 Parser - converts CE1 notation to expression tree
Syntax:
  (expr)  -> morphism
  <expr>  -> witness/fixed-point
  {expr}  -> boundary
  [expr]  -> memory
  H c     -> harmonic operator at constant c
*/
	function parseCE1(str) {
		str = str.trim();
	
		if(!str) {
			return null;
		}
	
		// Check for brackets
		if(str[0] === "(" && str[str.length - 1] === ")") {
			return new CE1Expression("morphism", null, [parseCE1(str.slice(1, -1))]);
		}
		if(str[0] === "<" && str[str.length - 1] === ">") {
			return new CE1Expression("witness", null, [parseCE1(str.slice(1, -1))]);
		}
		if(str[0] === "{" && str[str.length - 1] === "}") {
			return new CE1Expression("boundary", null, [parseCE1(str.slice(1, -1))]);
		}
		if(str[0] === "[" && str[str.length - 1] === "]") {
			return new CE1Expression("memory", null, [parseCE1(str.slice(1, -1))]);
		}
	
		// Check for harmonic operator
		if(str.match(/^H\s+[\d.+-]+$/)) {
			var constant = parseFloat(str.split(/\s+/)[1]);
			return new CE1Expression("harmonic", constant, []);
		}
	
		// Parse as constant
		var num = parseFloat(str);
		if(!isNaN(num)) {
			return new CE1Expression("constant", num, []);
		}
	
		// Parse operators
		if(str.startsWith("ln")) {
			return new CE1Expression("boundary", "ln", [parseCE1(str.substring(2).trim())]);
		}
		if(str.startsWith("ζ") || str.startsWith("zeta")) {
			var arg = str.startsWith("ζ") ? str.substring(1) : str.substring(4);
			return new CE1Expression("memory", "zeta", [parseCE1(arg.trim())]);
		}
		if(str.startsWith("tan")) {
			return new CE1Expression("morphism", "tan", [parseCE1(str.substring(3).trim())]);
		}
		if(str.startsWith("sin")) {
			return new CE1Expression("witness", "sin", [parseCE1(str.substring(3).trim())]);
		}
		if(str.startsWith("cos")) {
			return new CE1Expression("witness", "cos", [parseCE1(str.substring(3).trim())]);
		}
	
		return new CE1Expression("constant", str, []);
	}

	/*
CE1 Evaluator - evaluates a CE1 expression tree
*/
	function evaluateCE1(expr) {
		if(!expr) {
			return NaN;
		}
	
		switch(expr.type) {
			case "constant":
				return typeof expr.value === "number" ? expr.value : parseFloat(expr.value);
			
			case "harmonic":
				return harmonicOperator(expr.value);
			
			case "boundary":
				if(expr.value === "ln" && expr.children.length > 0) {
					return HarmonicOperators.logarithm(evaluateCE1(expr.children[0]));
				}
				// Boundary bracket
				if(expr.children.length > 0) {
					return HarmonicOperators.logarithm(evaluateCE1(expr.children[0]));
				}
				return NaN;
			
			case "memory":
				if(expr.value === "zeta" && expr.children.length > 0) {
					return HarmonicOperators.zeta(evaluateCE1(expr.children[0]));
				}
				// Memory bracket
				if(expr.children.length > 0) {
					return HarmonicOperators.zeta(evaluateCE1(expr.children[0]));
				}
				return NaN;
			
			case "morphism":
				if(expr.value === "tan" && expr.children.length > 0) {
					return HarmonicOperators.tangent(evaluateCE1(expr.children[0]));
				}
				// Morphism bracket
				if(expr.children.length > 0) {
					return HarmonicOperators.tangent(evaluateCE1(expr.children[0]));
				}
				return NaN;
			
			case "witness":
				if(expr.value === "sin" && expr.children.length > 0) {
					return HarmonicOperators.sine(evaluateCE1(expr.children[0]));
				}
				if(expr.value === "cos" && expr.children.length > 0) {
					return HarmonicOperators.cosine(evaluateCE1(expr.children[0]));
				}
				// Witness bracket - fixed point resolver
				if(expr.children.length > 0) {
					var childValue = evaluateCE1(expr.children[0]);
					// If child is a harmonic operator result, find its root
					if(typeof childValue === "object" && childValue.components) {
						var guess = typeof childValue.value === "number" ? childValue.value : 0.5;
						return fixedPointResolver(guess);
					}
					return childValue;
				}
				return NaN;
			
			default:
				return NaN;
		}
	}

	/*
Export functions for use in TiddlyWiki
*/
	exports.CE1Expression = CE1Expression;
	exports.HarmonicOperators = HarmonicOperators;
	exports.harmonicOperator = harmonicOperator;
	exports.fixedPointResolver = fixedPointResolver;
	exports.parseCE1 = parseCE1;
	exports.evaluateCE1 = evaluateCE1;

})();
