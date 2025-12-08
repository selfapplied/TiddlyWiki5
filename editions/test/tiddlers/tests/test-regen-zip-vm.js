/*\
title: test-regen-zip-vm.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for the REGEN-ZIP Virtual Machine

\*/

(function() {

	"use strict";

	describe("REGEN-ZIP VM", function() {
		var RegenZipVM = $tw.utils.RegenZipVM;
		var OPCODES = $tw.utils.OPCODES;
		var wiki;
		var vm;
	
		beforeEach(function() {
			wiki = new $tw.Wiki();
			vm = new RegenZipVM(wiki);
		});
	
		describe("Construction", function() {
			it("should create a VM instance", function() {
				expect(vm).toBeDefined();
				expect(vm.wiki).toBe(wiki);
				expect(vm.state).toBe("idle");
			});
		
			it("should have correct kappa value", function() {
				expect(vm.kappa).toBe(0.35);
			});
		
			it("should initialize empty generators", function() {
				expect(Object.keys(vm.generators).length).toBe(0);
			});
		});
	
		describe("Opcode Constants", function() {
			it("should define all opcodes", function() {
				expect(OPCODES.OP_SEED).toBe(0x01);
				expect(OPCODES.OP_GENERATOR).toBe(0x02);
				expect(OPCODES.OP_VERIFY).toBe(0x03);
				expect(OPCODES.OP_ATTACH).toBe(0x04);
				expect(OPCODES.OP_ZP35_CHECK).toBe(0x05);
				expect(OPCODES.OP_TW_INSERT).toBe(0x06);
			});
		});
	
		describe("Generator Registration", function() {
			it("should register a generator", function() {
				var generator = function(context) {
					return { assets: [] };
				};
			
				vm.registerGenerator("testGen", generator, {
					version: "1.0.0",
					description: "Test generator"
				});
			
				expect(vm.generators.testGen).toBeDefined();
				expect(vm.generators.testGen.fn).toBe(generator);
				expect(vm.generators.testGen.version).toBe("1.0.0");
			});
		
			it("should register generator with default metadata", function() {
				var generator = function(context) {
					return { assets: [] };
				};
			
				vm.registerGenerator("testGen", generator);
			
				expect(vm.generators.testGen.version).toBe("1.0.0");
				expect(vm.generators.testGen.seed).toBe(null);
			});
		});
	
		describe("Tiddler Loading", function() {
			it("should load a tiddler with regen-zip field", function() {
				var tiddler = {
					fields: {
						title: "TestTiddler",
						"regen-zip": "testGenerator",
						generator: "testGenerator"
					}
				};
			
				var result = vm.load(tiddler);
				expect(result).toBe(true);
				expect(vm.state).toBe("loading");
				expect(vm.context.title).toBe("TestTiddler");
			});
		
			it("should reject null tiddler", function() {
				var result = vm.load(null);
				expect(result).toBe(false);
			});
		
			it("should handle tiddler without regen-zip field", function() {
				var tiddler = {
					fields: {
						title: "NormalTiddler"
					}
				};
			
				var result = vm.load(tiddler);
				expect(result).toBe(false);
				expect(vm.state).toBe("idle");
			});
		});
	
		describe("Seeded RNG", function() {
			it("should create deterministic RNG", function() {
				var rng1 = vm.createSeededRNG("test-seed");
				var rng2 = vm.createSeededRNG("test-seed");
			
				var values1 = [rng1(), rng1(), rng1()];
				var values2 = [rng2(), rng2(), rng2()];
			
				expect(values1[0]).toBe(values2[0]);
				expect(values1[1]).toBe(values2[1]);
				expect(values1[2]).toBe(values2[2]);
			});
		
			it("should generate different sequences for different seeds", function() {
				var rng1 = vm.createSeededRNG("seed1");
				var rng2 = vm.createSeededRNG("seed2");
			
				var val1 = rng1();
				var val2 = rng2();
			
				expect(val1).not.toBe(val2);
			});
		
			it("should generate values in [0, 1]", function() {
				var rng = vm.createSeededRNG("test");
			
				for(var i = 0; i < 100; i++) {
					var val = rng();
					expect(val).toBeGreaterThanOrEqual(0);
					expect(val).toBeLessThan(1);
				}
			});
		});
	
		describe("OP_SEED Execution", function() {
			it("should initialize context with seed", function() {
				vm.executeSeed("test-seed");
			
				expect(vm.context.seed).toBe("test-seed");
				expect(vm.context.rng).toBeDefined();
				expect(typeof vm.context.rng).toBe("function");
			});
		});
	
		describe("OP_GENERATOR Execution", function() {
			it("should execute registered generator", function() {
				var executed = false;
				var generator = function(context) {
					executed = true;
					return {
						assets: [
							{ name: "test.txt", type: "text/plain", data: "Hello" }
						]
					};
				};
			
				vm.registerGenerator("testGen", generator);
				vm.context.tiddler = { fields: { title: "Test" } };
				vm.executeGenerator("testGen", "seed", "1.0.0");
			
				expect(executed).toBe(true);
				expect(vm.assets.length).toBe(1);
				expect(vm.assets[0].name).toBe("test.txt");
			});
		
			it("should pass context to generator", function() {
				var receivedContext = null;
				var generator = function(context) {
					receivedContext = context;
					return { assets: [] };
				};
			
				vm.registerGenerator("testGen", generator);
				vm.context.tiddler = { fields: { title: "Test" } };
				vm.context.rng = function() { return 0.5; };
				vm.executeGenerator("testGen", "my-seed", "1.0.0");
			
				expect(receivedContext).not.toBe(null);
				expect(receivedContext.seed).toBe("my-seed");
				expect(receivedContext.rng).toBeDefined();
				expect(receivedContext.wiki).toBe(wiki);
			});
		
			it("should throw error for unknown generator", function() {
				expect(function() {
					vm.executeGenerator("unknownGen", "seed", "1.0.0");
				}).toThrow();
			});
		});
	
		describe("OP_ZP35_CHECK Execution", function() {
			it("should allow when signatures match", function() {
				var generator = function(context) {
					return { assets: [] };
				};
			
				vm.registerGenerator("testGen", generator, {
					zp35: "0.500000.10"
				});
			
				var result = vm.executeZP35Check("testGen", "0.500000.10");
			
				expect(result.allowed).toBe(true);
				expect(result.mode).toBe("safe");
				expect(result.distance).toBe(0);
			});
		
			it("should block when generator not found", function() {
				var result = vm.executeZP35Check("unknownGen", "0.500000.10");
			
				expect(result.allowed).toBe(false);
			});
		
			it("should allow with caution for moderate distances", function() {
				var generator = function(context) {
					return { assets: [] };
				};
			
				vm.registerGenerator("testGen", generator, {
					zp35: "0.200000.10"
				});
			
				// Distance = 0.45, which is > κ (0.35) but < 2κ (0.70)
				var result = vm.executeZP35Check("testGen", "0.650000.10");
			
				expect(result.allowed).toBe(true);
				expect(result.mode).toBe("caution");
			});
		});
	
		describe("OP_VERIFY Execution", function() {
			it("should verify checksums", function() {
				vm.assets = [
					{
						name: "test.txt",
						data: "hello",
						checksum: vm.computeChecksum("hello")
					}
				];
			
				expect(function() {
					vm.executeVerify();
				}).not.toThrow();
			});
		
			it("should throw on checksum mismatch", function() {
				vm.assets = [
					{
						name: "test.txt",
						data: "hello",
						checksum: "wrongchecksum"
					}
				];
			
				expect(function() {
					vm.executeVerify();
				}).toThrow();
			});
		});
	
		describe("Full VM Execution", function() {
			it("should execute complete workflow", function() {
			// Register generator
				var generator = function(context) {
					return {
						assets: [
							{
								name: "output.txt",
								type: "text/plain",
								data: "Generated: " + context.seed
							}
						]
					};
				};
			
				vm.registerGenerator("fullTestGen", generator, {
					version: "1.0.0",
					zp35: "0.500000.10"
				});
			
				// Create tiddler
				var tiddler = {
					fields: {
						title: "TestTiddler",
						"regen-zip": "fullTestGen",
						generator: "fullTestGen",
						seed: "test-seed",
						version: "1.0.0",
						zp35: "0.500000.10"
					}
				};
			
				// Load and run
				var loadResult = vm.load(tiddler);
				expect(loadResult).toBe(true);
			
				var runResult = vm.run();
				expect(runResult.success).toBe(true);
				expect(runResult.assets.length).toBe(1);
				expect(runResult.assets[0].name).toBe("output.txt");
				expect(runResult.assets[0].data).toBe("Generated: test-seed");
				expect(vm.state).toBe("complete");
			});
		
			it("should handle execution errors gracefully", function() {
				var generator = function(context) {
					throw new Error("Generation failed");
				};
			
				vm.registerGenerator("errorGen", generator);
			
				var tiddler = {
					fields: {
						title: "ErrorTiddler",
						"regen-zip": "errorGen",
						generator: "errorGen"
					}
				};
			
				vm.load(tiddler);
				var result = vm.run();
			
				expect(result.success).toBe(false);
				expect(result.error).toBeDefined();
				expect(vm.state).toBe("error");
			});
		});
	
		describe("VM State Management", function() {
			it("should return current state", function() {
				var state = vm.getState();
			
				expect(state.state).toBe("idle");
				expect(state.context).toBeDefined();
				expect(state.assets).toBeDefined();
				expect(state.generators).toBeDefined();
			});
		
			it("should reset VM", function() {
				vm.context.test = "data";
				vm.assets = [{ name: "test" }];
				vm.state = "running";
			
				vm.reset();
			
				expect(Object.keys(vm.context).length).toBe(0);
				expect(vm.assets.length).toBe(0);
				expect(vm.state).toBe("idle");
			});
		});
	
		describe("Utility Functions", function() {
			it("should hash strings consistently", function() {
				var hash1 = vm.hashString("test");
				var hash2 = vm.hashString("test");
				var hash3 = vm.hashString("different");
			
				expect(hash1).toBe(hash2);
				expect(hash1).not.toBe(hash3);
			});
		
			it("should compute checksums", function() {
				var checksum = vm.computeChecksum("test data");
			
				expect(checksum).toBeDefined();
				expect(typeof checksum).toBe("string");
				expect(checksum.length).toBeGreaterThan(0);
			});
		});
	
		describe("ZP35 Distance Calculation", function() {
			it("should return 0 for identical signatures", function() {
				var distance = vm.calculateZP35Distance("0.500000", "0.500000");
				expect(distance).toBe(0);
			});
		
			it("should calculate distance for different signatures", function() {
				var distance = vm.calculateZP35Distance("0.300000", "0.500000");
				expect(distance).toBeGreaterThan(0);
				expect(distance).toBeLessThanOrEqual(1);
			});
		});
	
		describe("Deterministic Generation", function() {
			it("should produce identical output for same seed", function() {
				var generator = function(context) {
					var val = context.rng();
					return {
						assets: [
							{ name: "output", type: "text/plain", data: String(val) }
						]
					};
				};
			
				vm.registerGenerator("detGen", generator);
			
				// First execution
				var tiddler1 = {
					fields: {
						title: "T1",
						"regen-zip": "detGen",
						generator: "detGen",
						seed: "fixed-seed"
					}
				};
			
				vm.load(tiddler1);
				var result1 = vm.run();
				var output1 = result1.assets[0].data;
			
				// Second execution (reset VM)
				vm.reset();
				vm.registerGenerator("detGen", generator);
			
				var tiddler2 = {
					fields: {
						title: "T2",
						"regen-zip": "detGen",
						generator: "detGen",
						seed: "fixed-seed"
					}
				};
			
				vm.load(tiddler2);
				var result2 = vm.run();
				var output2 = result2.assets[0].data;
			
				expect(output1).toBe(output2);
			});
		});
	});

})();
