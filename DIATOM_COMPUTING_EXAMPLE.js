/*\
title: DIATOM_COMPUTING_EXAMPLE.js
type: application/javascript

Example Usage: Diatom Computing - Biological Computation Through Geometry

This example demonstrates the diatom computing model, showing how:
1. Frustules (shells) encode algorithms through geometry
2. Growth is iterative computation via silica deposition
3. Colonies achieve distributed consensus
4. Optical structures enable photonic processing
5. Everything maps to CE1 fixed-point expressions

\*/

// Load the diatom computing module
var DiatomModule = require("./core/modules/utils/diatom-computing.js");
var Diatom = DiatomModule.Diatom;
var DiatomColony = DiatomModule.DiatomColony;

console.log("═══════════════════════════════════════════════════════════════");
console.log("  Diatom Computing: Biology as Operator Theory");
console.log("  'A computer grown from fields'");
console.log("═══════════════════════════════════════════════════════════════\n");

// ============================================================================
// Example 1: Basic Frustule Construction
// ============================================================================

console.log("=== Example 1: Basic Frustule Construction ===\n");

// Create a simple diatom with default parameters
var diatom1 = new Diatom();

console.log("Default Diatom Created:");
console.log("  Boundary type:", diatom1.boundary.type);
console.log("  Lattice symmetry:", diatom1.lattice.symmetry);
console.log("  Lattice order:", diatom1.lattice.order, "(", diatom1.lattice.order, "-fold symmetry)");
console.log("  Fixed point (κ):", diatom1.fixedPoint);
console.log("");

// Encode a simple frustule geometry
var geometry = {
	pores: [
		{ position: {x: 0, y: 1}, diameter: 0.5 },
		{ position: {x: 1, y: 0}, diameter: 0.3 },
		{ position: {x: 0, y: -1}, diameter: 0.4 },
		{ position: {x: -1, y: 0}, diameter: 0.35 }
	],
	ridges: [
		{ path: "circular", height: 0.8 },
		{ path: "radial", height: 0.6 }
	],
	scale: 10.0
};

var encoded = diatom1.encodeFrustule(geometry);

console.log("Encoded Frustule:");
console.log("  Pore variables:", encoded.poreVariables.length);
console.log("  Ridge controls:", encoded.ridgeControls.length);
console.log("  Flow paths:", encoded.flowPaths.length);
console.log("  Waveguides:", encoded.waveguides.length);
console.log("");

// Show how geometric features map to computational elements
console.log("Computational Mapping:");
encoded.poreVariables.forEach(function(pore, i) {
	console.log("  Pore", i + ":", "diameter =", pore.diameter.toFixed(3), 
		"→ variable value =", pore.value.toFixed(3));
});
console.log("");

// ============================================================================
// Example 2: Growth as Iterative Computation
// ============================================================================

console.log("=== Example 2: Growth as Iterative Computation ===\n");

// Create diatom with specific lattice symmetry
var diatom2 = new Diatom({
	lattice: {
		symmetry: "radial",
		order: 8,  // 8-fold rotational symmetry
		spacing: 1.0
	}
});

console.log("Growing diatom with 8-fold symmetry...");
console.log("Target: converge to fixed point κ = 0.35");
console.log("");

// Perform growth iteration
var growthResult = diatom2.grow(50);

console.log("Growth Complete:");
console.log("  Steps taken:", growthResult.steps);
console.log("  Converged:", growthResult.converged ? "Yes" : "No");
console.log("  Final curvature:", growthResult.finalCurvature.toFixed(6));
console.log("  Fixed point:", growthResult.fixedPoint);
console.log("  Distance from fixed point:", 
	Math.abs(growthResult.finalCurvature - growthResult.fixedPoint).toFixed(6));
console.log("");

// Show convergence history
console.log("Convergence History (first 10 steps):");
for(var i = 0; i < Math.min(10, growthResult.history.length); i++) {
	console.log("  Step", (i + 1) + ":", 
		"error =", growthResult.history[i].toFixed(6));
}
console.log("");

// Examine silica deposits (computational steps)
console.log("Silica Deposits (computational memory):");
console.log("  Total deposits:", diatom2.silicaDeposits.length);
var lastDeposit = diatom2.silicaDeposits[diatom2.silicaDeposits.length - 1];
console.log("  Last deposit:");
console.log("    Step:", lastDeposit.step);
console.log("    Curvature:", lastDeposit.curvature.toFixed(6));
console.log("    Converged:", lastDeposit.symmetry.converged);
console.log("");

// ============================================================================
// Example 3: CE1 Fixed-Point Expression
// ============================================================================

console.log("=== Example 3: CE1 Fixed-Point Expression ===\n");

// Express diatom as CE1 formula
var ce1 = diatom2.toCE1Expression();

console.log("CE1 Expression:", ce1.expression);
console.log("");
console.log("Components:");
console.log("  {D} - Boundary Domain:");
console.log("    Type:", ce1.components.D.value.type);
console.log("    Role:", ce1.components.D.role);
console.log("");
console.log("  [L] - Pattern Lattice:");
console.log("    Symmetry:", ce1.components.L.value.symmetry);
console.log("    Order:", ce1.components.L.value.order);
console.log("    Role:", ce1.components.L.role);
console.log("");
console.log("  (M) - Morphism Operator:");
console.log("    Type:", ce1.components.M.value.type);
console.log("    Curvature:", ce1.components.M.value.curvature.toFixed(6));
console.log("    Role:", ce1.components.M.role);
console.log("");
console.log("  F - Fixed Point:");
console.log("    Value (κ):", ce1.components.F.value);
console.log("    Role:", ce1.components.F.role);
console.log("");
console.log("Coherence:", ce1.coherence.toFixed(6));
console.log("Interpretation:", ce1.interpretation);
console.log("");

// Verify fixed-point property
var isFixedPoint = diatom2.verifyCE1FixedPoint();
console.log("Fixed-Point Property Verified:", isFixedPoint ? "✓" : "✗");
console.log("");

// ============================================================================
// Example 4: Colony Formation and Distributed Consensus
// ============================================================================

console.log("=== Example 4: Colony Formation and Distributed Consensus ===\n");

// Create a colony
var colony = new DiatomColony({
	id: "example_colony_1",
	consensusThreshold: 0.75
});

console.log("Colony created:", colony.id);
console.log("");

// Add multiple diatoms with different initial conditions
console.log("Populating colony with 5 diatoms...");
for(var i = 0; i < 5; i++) {
	var diatom = new Diatom({
		lattice: {
			symmetry: "radial",
			order: 6,
			spacing: 1.0 + Math.random() * 0.2  // Slight variation
		}
	});
	
	// Give each diatom a short growth period
	diatom.grow(20 + Math.floor(Math.random() * 10));
	
	colony.addDiatom(diatom);
}
console.log("  Population:", colony.diatoms.length);
console.log("");

// Apply environmental signal
console.log("Applying environmental signal...");
colony.applySignal({
	salinity: 30.0,      // Lower salinity
	nutrients: 1.5,      // Nutrient pulse
	temperature: 22.0    // Slight warming
});
console.log("  Signal applied to all diatoms");
console.log("");

// Check synchronization (consensus)
var consensus = colony.synchronize();
console.log("Colony Synchronization:");
console.log("  Synchronized:", consensus.synchronized ? "Yes" : "No");
console.log("  Mean curvature:", consensus.meanCurvature.toFixed(6));
console.log("  Variance:", consensus.variance.toFixed(6));
console.log("  Consensus:", consensus.consensus);
console.log("  Diatom count:", consensus.diatomCount);
console.log("");

// Encode colony state as distributed memory
var colonyState = colony.encodeState();
console.log("Colony State Encoding (Distributed Memory):");
console.log("  Population:", colonyState.population);
console.log("  Environmental field:");
console.log("    Salinity:", colonyState.environmentalField.salinity, "ppt");
console.log("    Light:", colonyState.environmentalField.lightSpectrum, "nm");
console.log("    Nutrients:", colonyState.environmentalField.nutrients);
console.log("    Temperature:", colonyState.environmentalField.temperature, "°C");
console.log("");
console.log("  Individual diatom states:");
colonyState.diatomStates.forEach(function(state, i) {
	console.log("    Diatom", i + ":", 
		"growth =", state.growthStep,
		"deposits =", state.deposits,
		"converged =", state.converged);
});
console.log("");

// ============================================================================
// Example 5: Optical Computing
// ============================================================================

console.log("=== Example 5: Optical Computing ===\n");

// Create diatom with optical structures
var opticalDiatom = new Diatom({
	lattice: {
		symmetry: "radial",
		order: 12  // High symmetry for optical effects
	}
});

// Encode complex geometry with optical features
var opticalGeometry = {
	pores: [
		{ position: {x: 0, y: 2}, diameter: 0.4 },
		{ position: {x: 2, y: 0}, diameter: 0.4 },
		{ position: {x: 0, y: -2}, diameter: 0.4 },
		{ position: {x: -2, y: 0}, diameter: 0.4 }
	],
	ridges: [
		{ path: "circular_1", height: 0.9 },
		{ path: "circular_2", height: 0.7 },
		{ path: "radial_1", height: 0.8 },
		{ path: "radial_2", height: 0.6 }
	],
	scale: 10.0
};

var opticalEncoded = opticalDiatom.encodeFrustule(opticalGeometry);
var opticalNetwork = opticalDiatom.createOpticalNetwork(opticalEncoded);

console.log("Optical Network Created:");
console.log("  Waveguides:", opticalNetwork.waveguides.length);
console.log("  Resonant cavities:", opticalNetwork.cavities.length);
console.log("  Scatterers (pores):", opticalNetwork.scatterers.length);
console.log("  Wavelength filters:", opticalNetwork.filters.length);
console.log("");

// Show waveguide details
if(opticalNetwork.waveguides.length > 0) {
	console.log("Waveguide Properties:");
	opticalNetwork.waveguides.forEach(function(wg, i) {
		console.log("  WG", i + ":", 
			"mode =", wg.mode,
			"λ =", wg.wavelength, "nm",
			"n =", wg.refractiveIndex);
	});
	console.log("");
}

// Show filter details
if(opticalNetwork.filters.length > 0) {
	console.log("Wavelength Filters (Bragg Reflectors):");
	opticalNetwork.filters.forEach(function(filter, i) {
		console.log("  Filter", i + ":", 
			"λ_center =", filter.centerWavelength.toFixed(1), "nm",
			"bandwidth =", filter.bandwidth.toFixed(1), "nm",
			"T =", (filter.transmission * 100).toFixed(1) + "%");
	});
	console.log("");
}

// Route light through the network
var inputLight = {
	intensity: 1.0,
	wavelength: 550  // Green light
};

console.log("Routing Light Through Network:");
console.log("  Input: λ =", inputLight.wavelength, "nm, I =", inputLight.intensity);

var outputLight = opticalDiatom.routeLight(opticalNetwork, inputLight);

console.log("  Output: λ =", outputLight.wavelength, "nm, I =", outputLight.intensity.toFixed(4));
console.log("  Attenuation:", ((1 - outputLight.intensity) * 100).toFixed(2) + "%");
console.log("  Path length:", outputLight.path.length, "elements");
console.log("");

// ============================================================================
// Example 6: Comparing Different Symmetries
// ============================================================================

console.log("=== Example 6: Comparing Different Symmetries ===\n");

var symmetries = [
	{ name: "4-fold", order: 4 },
	{ name: "6-fold", order: 6 },
	{ name: "8-fold", order: 8 },
	{ name: "12-fold", order: 12 }
];

console.log("Growing diatoms with different symmetries:");
console.log("");

symmetries.forEach(function(sym) {
	var d = new Diatom({
		lattice: {
			symmetry: "radial",
			order: sym.order
		}
	});
	
	var result = d.grow(50);
	
	console.log("  " + sym.name + " symmetry:");
	console.log("    Steps:", result.steps);
	console.log("    Converged:", result.converged ? "Yes" : "No");
	console.log("    Final curvature:", result.finalCurvature.toFixed(6));
	console.log("    Coherence:", d.computeCoherence().toFixed(6));
	console.log("");
});

// ============================================================================
// Example 7: Self-Reference and Fixed Points
// ============================================================================

console.log("=== Example 7: Self-Reference and Fixed Points ===\n");

// Create diatom that encodes its own structure
var selfRefDiatom = new Diatom();

// Grow to convergence
var selfGrowth = selfRefDiatom.grow(100);

console.log("Self-Referential Diatom:");
console.log("  The frustule is both:");
console.log("    1. The program (geometry encodes algorithm)");
console.log("    2. The result (shape is the output)");
console.log("");
console.log("  Growth process:");
console.log("    Iterations:", selfGrowth.steps);
console.log("    Converged to fixed point:", selfGrowth.converged);
console.log("");

// Express as CE1 and verify
var selfCE1 = selfRefDiatom.toCE1Expression();
var selfFixed = selfRefDiatom.verifyCE1FixedPoint();

console.log("  CE1 Expression: " + selfCE1.expression);
console.log("  Fixed-point equation satisfied:", selfFixed);
console.log("");
console.log("  Meaning:");
console.log("    Applying morphism (M) to current state S");
console.log("    produces the same state S:");
console.log("    (M)(S) ≈ S  (within tolerance κ = 0.35)");
console.log("");
console.log("  This is not a metaphor.");
console.log("  The diatom LITERALLY satisfies the fixed-point equation.");
console.log("  Its growth IS the computation.");
console.log("");

// ============================================================================
// Summary
// ============================================================================

console.log("═══════════════════════════════════════════════════════════════");
console.log("  Summary: Diatoms as Computers");
console.log("═══════════════════════════════════════════════════════════════\n");

console.log("What we've demonstrated:\n");
console.log("  1. Frustules encode algorithms");
console.log("     └─ Pores = variables, Ridges = controls\n");
console.log("  2. Growth is iterative computation");
console.log("     └─ Silica deposition = execution steps\n");
console.log("  3. Convergence is program termination");
console.log("     └─ Fixed point = computed result\n");
console.log("  4. Colonies achieve distributed consensus");
console.log("     └─ Liquid blockchain of glass and sunlight\n");
console.log("  5. Optical structures enable photonic processing");
console.log("     └─ Natural silicon photonics\n");
console.log("  6. Everything maps to CE1 fixed-point expressions");
console.log("     └─ < {D} + [L] + (M) + F >\n");

console.log("Key insight:");
console.log("  A diatom doesn't REPRESENT a computation.");
console.log("  Its growth IS the computation.");
console.log("  Biology is a branch of operator theory.\n");

console.log("═══════════════════════════════════════════════════════════════");
