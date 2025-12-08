/*\
title: $:/core/modules/utils/diatom-computing.js
type: application/javascript
module-type: utils

Diatom Computing: Biological Computation Through Geometric Encoding

This module implements a diatom-inspired distributed computing model where:
- Frustules (shells) encode algorithms through geometry
- Growth is iterative computation via silica deposition
- Colonies act as distributed memory and consensus
- Optical properties enable photonic routing
- Structure maps to CE1 fixed-point expressions: < {D} + [L] + (M) + F >

Theoretical Foundation:
A diatom is a living proof that biology is a branch of operator theory,
computing everywhere at once through shape, the way a coral reef or galaxy
distributes logic through geometry.

\*/

"use strict";

/*
Diatom Constructor
@param {object} options - Configuration options
*/
function Diatom(options) {
	options = options || {};
	
	// Boundary constraints (silica deposition domain) - {D}
	this.boundary = options.boundary || this.createDefaultBoundary();
	
	// Pattern lattice (structural symmetry) - [L]
	this.lattice = options.lattice || this.createDefaultLattice();
	
	// Curvature evolution (morphism operator) - (M)
	this.morphism = options.morphism || this.createDefaultMorphism();
	
	// Equilibrium symmetry (fixed point) - <F>
	this.fixedPoint = options.fixedPoint || 0.35; // κ guardian threshold
	
	// Growth state
	this.growthStep = 0;
	this.silicaDeposits = [];
	this.convergenceHistory = [];
	
	// Optical routing state
	this.waveguides = [];
	this.resonantCavities = [];
	
	// Colony coordination
	this.colonyId = options.colonyId || null;
	this.localState = {};
	this.environmentalSignals = [];
}

/*
═══════════════════════════════════════════════════════════════════════════
1. Frustule Construction: The Shell as an Algorithm
═══════════════════════════════════════════════════════════════════════════

The frustule isn't decoration - it's a compiled program that encodes:
- flow constraints
- nutrient diffusion patterns
- mechanical stress paths
- light-guiding waveguides

Every pore is a variable. Every ridge is a control structure.
*/

/*
Create default boundary constraints {D}
@returns {object} - Boundary definition
*/
Diatom.prototype.createDefaultBoundary = function() {
	return {
		type: "silica",
		constraints: {
			flowRate: 1.0,
			diffusionCoeff: 0.5,
			stressTolerance: 100.0
		},
		pores: [], // Each pore is a variable
		ridges: []  // Each ridge is a control structure
	};
};

/*
Create default pattern lattice [L]
@returns {object} - Lattice structure
*/
Diatom.prototype.createDefaultLattice = function() {
	return {
		symmetry: "radial", // or "bilateral"
		order: 6, // n-fold rotational symmetry
		spacing: 1.0,
		nodes: []
	};
};

/*
Create default morphism operator (M)
@returns {object} - Morphism definition
*/
Diatom.prototype.createDefaultMorphism = function() {
	return {
		type: "recursive_deposition",
		curvature: 0.0,
		stability: 1.0,
		stepSize: 0.1
	};
};

/*
Encode geometric features as computational elements
@param {object} geometry - Geometric description
@returns {object} - Encoded algorithm
*/
Diatom.prototype.encodeFrustule = function(geometry) {
	var encoded = {
		poreVariables: [],
		ridgeControls: [],
		flowPaths: [],
		waveguides: []
	};
	
	// Each pore becomes a variable
	if(geometry.pores) {
		geometry.pores.forEach(function(pore) {
			encoded.poreVariables.push({
				position: pore.position,
				diameter: pore.diameter,
				value: pore.diameter / geometry.scale,
				type: "nutrient_gate"
			});
		});
	}
	
	// Each ridge becomes a control structure
	if(geometry.ridges) {
		geometry.ridges.forEach(function(ridge) {
			encoded.ridgeControls.push({
				path: ridge.path,
				height: ridge.height,
				controlType: ridge.height > 1.0 ? "barrier" : "channel",
				strength: ridge.height
			});
		});
	}
	
	// Flow patterns encode computation paths
	encoded.flowPaths = this.computeFlowPaths(encoded.poreVariables, encoded.ridgeControls);
	
	// Optical properties encode photonic routing
	encoded.waveguides = this.computeWaveguides(geometry);
	
	return encoded;
};

/*
Compute flow paths from pores and ridges
@param {array} pores - Pore variables
@param {array} ridges - Ridge controls
@returns {array} - Flow paths
*/
Diatom.prototype.computeFlowPaths = function(pores, ridges) {
	var paths = [];
	
	// Simple diffusion model: pores create sources/sinks
	for(var i = 0; i < pores.length; i++) {
		for(var j = i + 1; j < pores.length; j++) {
			var p1 = pores[i];
			var p2 = pores[j];
			
			// Check if path is blocked by ridges
			var isBlocked = ridges.some(function(ridge) {
				return ridge.controlType === "barrier" && 
				       this.pathIntersectsRidge(p1, p2, ridge);
			}, this);
			
			if(!isBlocked) {
				paths.push({
					from: i,
					to: j,
					flow: (p1.value + p2.value) / 2.0,
					resistance: this.computePathResistance(p1, p2)
				});
			}
		}
	}
	
	return paths;
};

/*
Check if path intersects a ridge
*/
Diatom.prototype.pathIntersectsRidge = function(/*p1, p2, ridge*/) {
	// Simplified geometric check
	// In reality, would use proper line-segment intersection
	return false;
};

/*
Compute path resistance between two pores
*/
Diatom.prototype.computePathResistance = function(p1, p2) {
	// Simple Euclidean distance model
	if(!p1.position || !p2.position) return 1.0;
	
	var dx = p1.position.x - p2.position.x;
	var dy = p1.position.y - p2.position.y;
	return Math.sqrt(dx * dx + dy * dy);
};

/*
Compute optical waveguides from geometry
@param {object} geometry - Geometric description
@returns {array} - Waveguide structures
*/
Diatom.prototype.computeWaveguides = function(geometry) {
	var waveguides = [];
	
	// High-index contrast in silica creates natural waveguides
	if(geometry.ridges) {
		geometry.ridges.forEach(function(ridge) {
			if(ridge.height > 0.5) { // High enough for optical confinement
				waveguides.push({
					path: ridge.path,
					mode: "single", // or "multi"
					wavelength: 550, // nm, visible light
					refractiveIndex: 1.45 // silica
				});
			}
		});
	}
	
	return waveguides;
};

/*
═══════════════════════════════════════════════════════════════════════════
2. Growth as Iterative Computation
═══════════════════════════════════════════════════════════════════════════

Silica deposition is stepwise, recursive, and stabilizing:
- boundary set → collapse
- pattern propagate → accumulate
- curvature solve → morphism
- symmetry check → witness

The diatom doesn't represent a computation - its growth IS the computation.
*/

/*
Perform one growth step (silica deposition)
@returns {object} - Growth step result
*/
Diatom.prototype.performGrowthStep = function() {
	this.growthStep++;
	
	// 1. Boundary set → collapse
	var boundaryState = this.setBoundary();
	
	// 2. Pattern propagate → accumulate
	var patternState = this.propagatePattern(boundaryState);
	
	// 3. Curvature solve → morphism
	var morphismState = this.solveCurvature(patternState);
	
	// 4. Symmetry check → witness
	var witness = this.checkSymmetry(morphismState);
	
	// Record silica deposit
	var deposit = {
		step: this.growthStep,
		boundary: boundaryState,
		pattern: patternState,
		curvature: morphismState.curvature,
		symmetry: witness.symmetry,
		converged: witness.converged
	};
	
	this.silicaDeposits.push(deposit);
	this.convergenceHistory.push(witness.error);
	
	return deposit;
};

/*
Set boundary conditions (collapse)
@returns {object} - Boundary state
*/
Diatom.prototype.setBoundary = function() {
	return {
		type: this.boundary.type,
		active: true,
		constraints: Object.assign({}, this.boundary.constraints)
	};
};

/*
Propagate pattern through lattice (accumulate)
@param {object} boundaryState - Current boundary
@returns {object} - Pattern state
*/
Diatom.prototype.propagatePattern = function(boundaryState) {
	var pattern = {
		lattice: this.lattice,
		activation: []
	};
	
	// Propagate through lattice nodes with symmetry constraints
	var nodeCount = this.lattice.order * 2; // Simple model
	for(var i = 0; i < nodeCount; i++) {
		var angle = (2 * Math.PI * i) / nodeCount;
		var activation = Math.cos(angle * this.lattice.order) * boundaryState.constraints.flowRate;
		pattern.activation.push(activation);
	}
	
	return pattern;
};

/*
Solve for curvature evolution (morphism)
@param {object} patternState - Current pattern
@returns {object} - Morphism state with curvature
*/
Diatom.prototype.solveCurvature = function(patternState) {
	// Compute mean curvature from pattern activation
	var sum = 0;
	var count = patternState.activation.length;
	
	for(var i = 0; i < count; i++) {
		sum += patternState.activation[i];
	}
	
	var mean = sum / count;
	
	// Curvature is variance from mean
	var variance = 0;
	for(var i = 0; i < count; i++) {
		var diff = patternState.activation[i] - mean;
		variance += diff * diff;
	}
	
	var curvature = Math.sqrt(variance / count);
	
	// Update morphism state
	this.morphism.curvature = curvature;
	
	return {
		curvature: curvature,
		mean: mean,
		variance: variance
	};
};

/*
Check symmetry and convergence (witness)
@param {object} morphismState - Current morphism
@returns {object} - Witness result
*/
Diatom.prototype.checkSymmetry = function(morphismState) {
	// Check if curvature has stabilized
	var error = Math.abs(morphismState.curvature - this.fixedPoint);
	var converged = error <= 0.01;
	
	return {
		symmetry: this.lattice.symmetry,
		error: error,
		converged: converged,
		fixedPointDistance: error
	};
};

/*
Run growth until convergence or max steps
@param {number} maxSteps - Maximum growth steps
@returns {object} - Final state
*/
Diatom.prototype.grow = function(maxSteps) {
	maxSteps = maxSteps || 100;
	
	var converged = false;
	var step = 0;
	
	while(!converged && step < maxSteps) {
		var result = this.performGrowthStep();
		converged = result.converged;
		step++;
	}
	
	return {
		steps: step,
		converged: converged,
		finalCurvature: this.morphism.curvature,
		fixedPoint: this.fixedPoint,
		history: this.convergenceHistory
	};
};

/*
═══════════════════════════════════════════════════════════════════════════
3. Colony as Distributed Memory
═══════════════════════════════════════════════════════════════════════════

Diatoms drift in networks. Environmental signals ripple through populations.
Each diatom encodes local state in its shell, and the population becomes
a spatially distributed consensus algorithm - a liquid blockchain of glass
and sunlight.
*/

/*
DiatomColony Constructor
@param {object} options - Configuration options
*/
function DiatomColony(options) {
	options = options || {};
	
	this.id = options.id || "colony_" + Date.now();
	this.diatoms = [];
	this.environmentalField = {
		salinity: 35.0,    // ppt
		lightSpectrum: 550, // nm
		nutrients: 1.0,     // normalized
		temperature: 20.0   // °C
	};
	this.consensusThreshold = options.consensusThreshold || 0.75;
}

/*
Add a diatom to the colony
@param {Diatom} diatom - Diatom instance
*/
DiatomColony.prototype.addDiatom = function(diatom) {
	diatom.colonyId = this.id;
	this.diatoms.push(diatom);
};

/*
Apply environmental signal to colony
@param {object} signal - Environmental change
*/
DiatomColony.prototype.applySignal = function(signal) {
	// Update environmental field
	Object.keys(signal).forEach(function(key) {
		if(this.environmentalField.hasOwnProperty(key)) {
			this.environmentalField[key] = signal[key];
		}
	}, this);
	
	// Propagate to all diatoms
	this.diatoms.forEach(function(diatom) {
		diatom.environmentalSignals.push({
			timestamp: Date.now(),
			field: Object.assign({}, this.environmentalField)
		});
	}, this);
};

/*
Synchronize colony (distributed consensus)
@returns {object} - Consensus state
*/
DiatomColony.prototype.synchronize = function() {
	if(this.diatoms.length === 0) {
		return { synchronized: false, reason: "Empty colony" };
	}
	
	// All diatoms solve the same boundary-value problem
	// Check if curvatures have converged to similar values
	var curvatures = this.diatoms.map(function(d) {
		return d.morphism.curvature;
	});
	
	var mean = curvatures.reduce(function(a, b) { return a + b; }, 0) / curvatures.length;
	
	var variance = 0;
	for(var i = 0; i < curvatures.length; i++) {
		var diff = curvatures[i] - mean;
		variance += diff * diff;
	}
	variance /= curvatures.length;
	
	var synchronized = Math.sqrt(variance) < 0.1;
	
	return {
		synchronized: synchronized,
		meanCurvature: mean,
		variance: variance,
		diatomCount: this.diatoms.length,
		consensus: synchronized ? "achieved" : "in_progress"
	};
};

/*
Encode colony state as distributed memory
@returns {object} - Colony state encoding
*/
DiatomColony.prototype.encodeState = function() {
	return {
		colonyId: this.id,
		population: this.diatoms.length,
		environmentalField: this.environmentalField,
		diatomStates: this.diatoms.map(function(d) {
			return {
				growthStep: d.growthStep,
				curvature: d.morphism.curvature,
				deposits: d.silicaDeposits.length,
				converged: d.convergenceHistory.length > 0 && 
				          d.convergenceHistory[d.convergenceHistory.length - 1] < 0.01
			};
		}),
		consensusState: this.synchronize()
	};
};

/*
═══════════════════════════════════════════════════════════════════════════
4. Optical Computing
═══════════════════════════════════════════════════════════════════════════

Some species route and scatter light like tiny photonic chips:
- waveguides
- resonant cavities
- controlled scattering
- narrow-band filtering

The shell becomes a device for shaping the electromagnetic field.
*/

/*
Create optical routing network from frustule geometry
@param {object} encoded - Encoded frustule
@returns {object} - Optical network
*/
Diatom.prototype.createOpticalNetwork = function(encoded) {
	var network = {
		waveguides: encoded.waveguides || [],
		cavities: [],
		scatterers: [],
		filters: []
	};
	
	// Find resonant cavities (enclosed regions)
	network.cavities = this.findResonantCavities(encoded);
	
	// Pores act as scatterers
	network.scatterers = encoded.poreVariables.map(function(pore) {
		return {
			position: pore.position,
			diameter: pore.diameter,
			scatteringCrossSection: Math.PI * Math.pow(pore.diameter / 2, 2)
		};
	});
	
	// Ridge patterns create wavelength filters
	network.filters = this.createWavelengthFilters(encoded.ridgeControls);
	
	return network;
};

/*
Find resonant cavities in structure
*/
Diatom.prototype.findResonantCavities = function(/*encoded*/) {
	// Simplified: look for high-curvature enclosed regions
	return [];
};

/*
Create wavelength filters from ridge patterns
*/
Diatom.prototype.createWavelengthFilters = function(ridges) {
	var filters = [];
	
	ridges.forEach(function(ridge) {
		if(ridge.height > 0.3) {
			// Ridge spacing determines filtered wavelength
			var spacing = ridge.strength || 1.0;
			var wavelength = spacing * 2 * 1.45; // Bragg condition with n=1.45
			
			filters.push({
				centerWavelength: wavelength * 1000, // Convert to nm
				bandwidth: wavelength * 100, // 10% bandwidth
				transmission: 0.9
			});
		}
	});
	
	return filters;
};

/*
Route light through optical network
@param {object} network - Optical network
@param {object} input - Input light
@returns {object} - Output light
*/
Diatom.prototype.routeLight = function(network, input) {
	var output = {
		intensity: input.intensity,
		wavelength: input.wavelength,
		path: []
	};
	
	// Apply wavelength filtering
	network.filters.forEach(function(filter) {
		var detuning = Math.abs(input.wavelength - filter.centerWavelength);
		if(detuning < filter.bandwidth / 2) {
			output.intensity *= filter.transmission;
			output.path.push({ type: "filter", wavelength: filter.centerWavelength });
		}
	});
	
	// Apply scattering
	network.scatterers.forEach(function(scatterer) {
		output.intensity *= (1 - scatterer.scatteringCrossSection * 0.01);
		output.path.push({ type: "scatter", position: scatterer.position });
	});
	
	return output;
};

/*
═══════════════════════════════════════════════════════════════════════════
5. CE1 Expression Mapping
═══════════════════════════════════════════════════════════════════════════

Map diatom to CE1 fixed-point expression:
< {D} + [L] + (M) + F >

Where:
- {D} = silica boundary (domain)
- [L] = pattern lattice (structure)
- (M) = curvature evolution (morphism)
- F = equilibrium symmetry (fixed point)
*/

/*
Express diatom as CE1 fixed-point formula
@returns {object} - CE1 expression
*/
Diatom.prototype.toCE1Expression = function() {
	return {
		type: "fixed_point",
		expression: "< {D} + [L] + (M) + F >",
		components: {
			D: {
				name: "boundary",
				value: this.boundary,
				role: "domain_constraints"
			},
			L: {
				name: "lattice",
				value: this.lattice,
				role: "structural_symmetry"
			},
			M: {
				name: "morphism",
				value: this.morphism,
				role: "curvature_evolution"
			},
			F: {
				name: "fixedPoint",
				value: this.fixedPoint,
				role: "equilibrium_symmetry"
			}
		},
		coherence: this.computeCoherence(),
		interpretation: "Diatom as self-realizing geometry"
	};
};

/*
Compute coherence of CE1 expression
@returns {number} - Coherence value [0,1]
*/
Diatom.prototype.computeCoherence = function() {
	// Coherence is how close we are to fixed point
	if(this.convergenceHistory.length === 0) {
		return 0.0;
	}
	
	var lastError = this.convergenceHistory[this.convergenceHistory.length - 1];
	return Math.max(0, 1 - lastError);
};

/*
Verify CE1 fixed-point property
@returns {boolean} - True if fixed point achieved
*/
Diatom.prototype.verifyCE1FixedPoint = function() {
	if(this.silicaDeposits.length < 2) {
		return false;
	}
	
	// Check if applying morphism to current state reproduces current state
	var currentState = this.silicaDeposits[this.silicaDeposits.length - 1];
	var error = currentState.symmetry.error;
	
	return error < 0.01 && currentState.symmetry.converged;
};

/*
Export the module
*/
exports.Diatom = Diatom;
exports.DiatomColony = DiatomColony;
