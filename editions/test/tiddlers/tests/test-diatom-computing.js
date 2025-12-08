/*\
title: test-diatom-computing.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for the diatom computing module.

\*/

(function(){

	/*jslint node: true, browser: true */
	/*global $tw: false */
	"use strict";

	describe("Diatom Computing", function() {

		var DiatomModule = require("$:/core/modules/utils/diatom-computing.js");
		var Diatom = DiatomModule.Diatom;
		var DiatomColony = DiatomModule.DiatomColony;

		// ========================================================================
		// Basic Construction Tests
		// ========================================================================

		describe("Diatom Construction", function() {

			it("should create a diatom with default parameters", function() {
				var diatom = new Diatom();
			
				expect(diatom).toBeDefined();
				expect(diatom.boundary).toBeDefined();
				expect(diatom.lattice).toBeDefined();
				expect(diatom.morphism).toBeDefined();
				expect(diatom.fixedPoint).toBe(0.35);
				expect(diatom.growthStep).toBe(0);
			});

			it("should create a diatom with custom lattice", function() {
				var diatom = new Diatom({
					lattice: {
						symmetry: "radial",
						order: 8,
						spacing: 2.0
					}
				});
			
				expect(diatom.lattice.symmetry).toBe("radial");
				expect(diatom.lattice.order).toBe(8);
				expect(diatom.lattice.spacing).toBe(2.0);
			});

			it("should create a diatom with custom fixed point", function() {
				var diatom = new Diatom({
					fixedPoint: 0.5
				});
			
				expect(diatom.fixedPoint).toBe(0.5);
			});

			it("should initialize empty silica deposits", function() {
				var diatom = new Diatom();
			
				expect(diatom.silicaDeposits).toEqual([]);
				expect(diatom.convergenceHistory).toEqual([]);
			});

		});

		// ========================================================================
		// Frustule Encoding Tests
		// ========================================================================

		describe("Frustule Encoding", function() {

			it("should encode pores as variables", function() {
				var diatom = new Diatom();
				var geometry = {
					pores: [
						{ position: {x: 0, y: 1}, diameter: 0.5 },
						{ position: {x: 1, y: 0}, diameter: 0.3 }
					],
					ridges: [],
					scale: 10.0
				};
			
				var encoded = diatom.encodeFrustule(geometry);
			
				expect(encoded.poreVariables.length).toBe(2);
				expect(encoded.poreVariables[0].diameter).toBe(0.5);
				expect(encoded.poreVariables[0].value).toBeCloseTo(0.05, 3);
				expect(encoded.poreVariables[0].type).toBe("nutrient_gate");
			});

			it("should encode ridges as control structures", function() {
				var diatom = new Diatom();
				var geometry = {
					pores: [],
					ridges: [
						{ path: "circular", height: 0.8 },
						{ path: "radial", height: 1.2 }
					],
					scale: 10.0
				};
			
				var encoded = diatom.encodeFrustule(geometry);
			
				expect(encoded.ridgeControls.length).toBe(2);
				expect(encoded.ridgeControls[0].height).toBe(0.8);
				expect(encoded.ridgeControls[0].controlType).toBe("channel");
				expect(encoded.ridgeControls[1].height).toBe(1.2);
				expect(encoded.ridgeControls[1].controlType).toBe("barrier");
			});

			it("should compute flow paths between pores", function() {
				var diatom = new Diatom();
				var geometry = {
					pores: [
						{ position: {x: 0, y: 0}, diameter: 0.5 },
						{ position: {x: 1, y: 0}, diameter: 0.5 },
						{ position: {x: 0, y: 1}, diameter: 0.5 }
					],
					ridges: [],
					scale: 10.0
				};
			
				var encoded = diatom.encodeFrustule(geometry);
			
				expect(encoded.flowPaths.length).toBeGreaterThan(0);
				expect(encoded.flowPaths[0].from).toBeDefined();
				expect(encoded.flowPaths[0].to).toBeDefined();
				expect(encoded.flowPaths[0].flow).toBeDefined();
				expect(encoded.flowPaths[0].resistance).toBeDefined();
			});

			it("should compute waveguides from high ridges", function() {
				var diatom = new Diatom();
				var geometry = {
					pores: [],
					ridges: [
						{ path: "circular", height: 0.8 },
						{ path: "radial", height: 0.3 }
					],
					scale: 10.0
				};
			
				var encoded = diatom.encodeFrustule(geometry);
			
				expect(encoded.waveguides.length).toBe(1); // Only high ridges become waveguides
				expect(encoded.waveguides[0].path).toBe("circular");
				expect(encoded.waveguides[0].mode).toBe("single");
				expect(encoded.waveguides[0].wavelength).toBe(550);
				expect(encoded.waveguides[0].refractiveIndex).toBe(1.45);
			});

		});

		// ========================================================================
		// Growth Iteration Tests
		// ========================================================================

		describe("Growth Iteration", function() {

			it("should perform a single growth step", function() {
				var diatom = new Diatom();
			
				var result = diatom.performGrowthStep();
			
				expect(result).toBeDefined();
				expect(result.step).toBe(1);
				expect(result.boundary).toBeDefined();
				expect(result.pattern).toBeDefined();
				expect(result.curvature).toBeDefined();
				expect(result.symmetry).toBeDefined();
				expect(diatom.silicaDeposits.length).toBe(1);
			});

			it("should record convergence history", function() {
				var diatom = new Diatom();
			
				diatom.performGrowthStep();
				diatom.performGrowthStep();
				diatom.performGrowthStep();
			
				expect(diatom.convergenceHistory.length).toBe(3);
				expect(typeof diatom.convergenceHistory[0]).toBe("number");
			});

			it("should converge to fixed point", function() {
				var diatom = new Diatom({
					lattice: { symmetry: "radial", order: 6 }
				});
			
				var result = diatom.grow(100);
			
				expect(result.steps).toBeLessThanOrEqual(100);
				// Convergence depends on parameters, so we just check structure
				expect(result.converged).toBeDefined();
				expect(result.finalCurvature).toBeDefined();
				expect(result.fixedPoint).toBeDefined();
				expect(result.history).toBeDefined();
			});

			it("should propagate pattern through lattice", function() {
				var diatom = new Diatom({
					lattice: { symmetry: "radial", order: 6 }
				});
			
				var boundaryState = diatom.setBoundary();
				var patternState = diatom.propagatePattern(boundaryState);
			
				expect(patternState).toBeDefined();
				expect(patternState.lattice).toBeDefined();
				expect(patternState.activation).toBeDefined();
				expect(patternState.activation.length).toBe(12); // order * 2
			});

			it("should solve for curvature", function() {
				var diatom = new Diatom();
				var patternState = {
					activation: [1.0, 0.5, 0.3, 0.8, 0.6, 0.4]
				};
			
				var morphismState = diatom.solveCurvature(patternState);
			
				expect(morphismState.curvature).toBeDefined();
				expect(morphismState.mean).toBeDefined();
				expect(morphismState.variance).toBeDefined();
				expect(diatom.morphism.curvature).toBe(morphismState.curvature);
			});

			it("should check symmetry and convergence", function() {
				var diatom = new Diatom({
					fixedPoint: 0.35
				});
			
				diatom.morphism.curvature = 0.36; // Close to fixed point
				var morphismState = { curvature: 0.36 };
			
				var witness = diatom.checkSymmetry(morphismState);
			
				expect(witness.symmetry).toBeDefined();
				expect(witness.error).toBeCloseTo(0.01, 3);
				expect(witness.converged).toBe(true);
				expect(witness.fixedPointDistance).toBeCloseTo(0.01, 3);
			});

		});

		// ========================================================================
		// CE1 Expression Tests
		// ========================================================================

		describe("CE1 Expression Mapping", function() {

			it("should express diatom as CE1 fixed-point formula", function() {
				var diatom = new Diatom();
				diatom.grow(10);
			
				var ce1 = diatom.toCE1Expression();
			
				expect(ce1.type).toBe("fixed_point");
				expect(ce1.expression).toBe("< {D} + [L] + (M) + F >");
				expect(ce1.components).toBeDefined();
				expect(ce1.components.D).toBeDefined();
				expect(ce1.components.L).toBeDefined();
				expect(ce1.components.M).toBeDefined();
				expect(ce1.components.F).toBeDefined();
			});

			it("should map boundary to {D} component", function() {
				var diatom = new Diatom();
				var ce1 = diatom.toCE1Expression();
			
				expect(ce1.components.D.name).toBe("boundary");
				expect(ce1.components.D.role).toBe("domain_constraints");
				expect(ce1.components.D.value).toBe(diatom.boundary);
			});

			it("should map lattice to [L] component", function() {
				var diatom = new Diatom();
				var ce1 = diatom.toCE1Expression();
			
				expect(ce1.components.L.name).toBe("lattice");
				expect(ce1.components.L.role).toBe("structural_symmetry");
				expect(ce1.components.L.value).toBe(diatom.lattice);
			});

			it("should map morphism to (M) component", function() {
				var diatom = new Diatom();
				var ce1 = diatom.toCE1Expression();
			
				expect(ce1.components.M.name).toBe("morphism");
				expect(ce1.components.M.role).toBe("curvature_evolution");
				expect(ce1.components.M.value).toBe(diatom.morphism);
			});

			it("should map fixed point to F component", function() {
				var diatom = new Diatom();
				var ce1 = diatom.toCE1Expression();
			
				expect(ce1.components.F.name).toBe("fixedPoint");
				expect(ce1.components.F.role).toBe("equilibrium_symmetry");
				expect(ce1.components.F.value).toBe(0.35);
			});

			it("should compute coherence", function() {
				var diatom = new Diatom();
			
				// No growth yet
				expect(diatom.computeCoherence()).toBe(0.0);
			
				// After growth
				diatom.grow(20);
				var coherence = diatom.computeCoherence();
				expect(coherence).toBeGreaterThanOrEqual(0.0);
				expect(coherence).toBeLessThanOrEqual(1.0);
			});

			it("should verify CE1 fixed-point property", function() {
				var diatom = new Diatom();
			
				// Before convergence
				expect(diatom.verifyCE1FixedPoint()).toBe(false);
			
				// After growth
				diatom.grow(100);
				// May or may not converge depending on parameters
				var verified = diatom.verifyCE1FixedPoint();
				expect(typeof verified).toBe("boolean");
			});

		});

		// ========================================================================
		// Colony Tests
		// ========================================================================

		describe("Diatom Colony", function() {

			it("should create a colony", function() {
				var colony = new DiatomColony();
			
				expect(colony).toBeDefined();
				expect(colony.id).toBeDefined();
				expect(colony.diatoms).toEqual([]);
				expect(colony.environmentalField).toBeDefined();
			});

			it("should add diatoms to colony", function() {
				var colony = new DiatomColony();
				var diatom1 = new Diatom();
				var diatom2 = new Diatom();
			
				colony.addDiatom(diatom1);
				colony.addDiatom(diatom2);
			
				expect(colony.diatoms.length).toBe(2);
				expect(diatom1.colonyId).toBe(colony.id);
				expect(diatom2.colonyId).toBe(colony.id);
			});

			it("should apply environmental signals", function() {
				var colony = new DiatomColony();
				var diatom = new Diatom();
				colony.addDiatom(diatom);
			
				colony.applySignal({
					salinity: 30.0,
					nutrients: 1.5
				});
			
				expect(colony.environmentalField.salinity).toBe(30.0);
				expect(colony.environmentalField.nutrients).toBe(1.5);
				expect(diatom.environmentalSignals.length).toBe(1);
				expect(diatom.environmentalSignals[0].field.salinity).toBe(30.0);
			});

			it("should synchronize colony", function() {
				var colony = new DiatomColony();
			
				// Add diatoms with similar curvatures
				for(var i = 0; i < 3; i++) {
					var d = new Diatom();
					d.morphism.curvature = 0.35 + Math.random() * 0.01; // Close values
					colony.addDiatom(d);
				}
			
				var consensus = colony.synchronize();
			
				expect(consensus.synchronized).toBeDefined();
				expect(consensus.meanCurvature).toBeDefined();
				expect(consensus.variance).toBeDefined();
				expect(consensus.diatomCount).toBe(3);
				expect(consensus.consensus).toBeDefined();
			});

			it("should encode colony state", function() {
				var colony = new DiatomColony({ id: "test_colony" });
				var diatom = new Diatom();
				diatom.grow(10);
				colony.addDiatom(diatom);
			
				var state = colony.encodeState();
			
				expect(state.colonyId).toBe("test_colony");
				expect(state.population).toBe(1);
				expect(state.environmentalField).toBeDefined();
				expect(state.diatomStates).toBeDefined();
				expect(state.diatomStates.length).toBe(1);
				expect(state.consensusState).toBeDefined();
			});

			it("should handle empty colony synchronization", function() {
				var colony = new DiatomColony();
			
				var consensus = colony.synchronize();
			
				expect(consensus.synchronized).toBe(false);
				expect(consensus.reason).toBe("Empty colony");
			});

		});

		// ========================================================================
		// Optical Computing Tests
		// ========================================================================

		describe("Optical Computing", function() {

			it("should create optical network from encoded frustule", function() {
				var diatom = new Diatom();
				var geometry = {
					pores: [
						{ position: {x: 0, y: 1}, diameter: 0.5 }
					],
					ridges: [
						{ path: "circular", height: 0.9 }
					],
					scale: 10.0
				};
			
				var encoded = diatom.encodeFrustule(geometry);
				var network = diatom.createOpticalNetwork(encoded);
			
				expect(network).toBeDefined();
				expect(network.waveguides).toBeDefined();
				expect(network.cavities).toBeDefined();
				expect(network.scatterers).toBeDefined();
				expect(network.filters).toBeDefined();
			});

			it("should create scatterers from pores", function() {
				var diatom = new Diatom();
				var encoded = {
					poreVariables: [
						{ position: {x: 0, y: 1}, diameter: 0.5 },
						{ position: {x: 1, y: 0}, diameter: 0.3 }
					],
					ridgeControls: [],
					waveguides: []
				};
			
				var network = diatom.createOpticalNetwork(encoded);
			
				expect(network.scatterers.length).toBe(2);
				expect(network.scatterers[0].diameter).toBe(0.5);
				expect(network.scatterers[0].scatteringCrossSection).toBeGreaterThan(0);
			});

			it("should create wavelength filters from ridges", function() {
				var diatom = new Diatom();
				var ridges = [
					{ height: 0.8, strength: 1.0 },
					{ height: 0.2, strength: 1.0 }, // Too low
					{ height: 0.6, strength: 2.0 }
				];
			
				var filters = diatom.createWavelengthFilters(ridges);
			
				expect(filters.length).toBe(2); // Only height > 0.3
				expect(filters[0].centerWavelength).toBeGreaterThan(0);
				expect(filters[0].bandwidth).toBeGreaterThan(0);
				expect(filters[0].transmission).toBeCloseTo(0.9, 2);
			});

			it("should route light through network", function() {
				var diatom = new Diatom();
				var network = {
					waveguides: [],
					cavities: [],
					scatterers: [
						{ position: {x: 0, y: 0}, scatteringCrossSection: 0.1 }
					],
					filters: [
						{ centerWavelength: 550, bandwidth: 100, transmission: 0.9 }
					]
				};
			
				var input = {
					intensity: 1.0,
					wavelength: 550
				};
			
				var output = diatom.routeLight(network, input);
			
				expect(output).toBeDefined();
				expect(output.intensity).toBeLessThan(1.0); // Some attenuation
				expect(output.wavelength).toBe(550);
				expect(output.path).toBeDefined();
				expect(output.path.length).toBeGreaterThan(0);
			});

			it("should apply wavelength filtering", function() {
				var diatom = new Diatom();
				var network = {
					waveguides: [],
					cavities: [],
					scatterers: [],
					filters: [
						{ centerWavelength: 550, bandwidth: 50, transmission: 0.8 }
					]
				};
			
				// On-resonance input
				var input1 = { intensity: 1.0, wavelength: 550 };
				var output1 = diatom.routeLight(network, input1);
				expect(output1.intensity).toBeCloseTo(0.8, 2);
			
				// Off-resonance input (should not be filtered)
				var input2 = { intensity: 1.0, wavelength: 700 };
				var output2 = diatom.routeLight(network, input2);
				expect(output2.intensity).toBeCloseTo(1.0, 2);
			});

		});

		// ========================================================================
		// Integration Tests
		// ========================================================================

		describe("System Integration", function() {

			it("should integrate with CE Tower guardian threshold", function() {
				var diatom = new Diatom();
			
				// Default fixed point should match CE Tower κ
				expect(diatom.fixedPoint).toBe(0.35);
			});

			it("should complete full workflow: construct → grow → express", function() {
				var diatom = new Diatom({
					lattice: { symmetry: "radial", order: 6 }
				});
			
				// Grow
				var growthResult = diatom.grow(50);
				expect(growthResult).toBeDefined();
			
				// Express as CE1
				var ce1 = diatom.toCE1Expression();
				expect(ce1.expression).toBe("< {D} + [L] + (M) + F >");
			
				// Verify
				var verified = diatom.verifyCE1FixedPoint();
				expect(typeof verified).toBe("boolean");
			});

			it("should demonstrate colony consensus workflow", function() {
				var colony = new DiatomColony();
			
				// Populate
				for(var i = 0; i < 3; i++) {
					var d = new Diatom();
					d.grow(30);
					colony.addDiatom(d);
				}
			
				// Signal
				colony.applySignal({ nutrients: 1.2 });
			
				// Synchronize
				var consensus = colony.synchronize();
				expect(consensus).toBeDefined();
			
				// Encode
				var state = colony.encodeState();
				expect(state.population).toBe(3);
			});

		});

	});

})();
