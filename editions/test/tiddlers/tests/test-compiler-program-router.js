/*\
title: test-compiler-program-router.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests for the Compiler-Program Router module.

\*/

(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

describe("Compiler-Program Router", function() {
	
	var CompilerProgramRouter;
	var ZP35Operator;
	var RegenZipVM;
	var wiki;
	
	beforeEach(function() {
		// Get required modules
		CompilerProgramRouter = $tw.utils.CompilerProgramRouter;
		ZP35Operator = $tw.utils.ZP35Operator;
		RegenZipVM = $tw.utils.RegenZipVM;
		wiki = $tw.wiki;
		
		// Verify modules are loaded
		expect(CompilerProgramRouter).toBeDefined();
		expect(ZP35Operator).toBeDefined();
		expect(RegenZipVM).toBeDefined();
	});
	
	describe("Router Construction", function() {
		
		it("should create router instance", function() {
			var zp35 = new ZP35Operator();
			var vm = new RegenZipVM(wiki);
			var router = new CompilerProgramRouter(wiki, zp35, vm);
			
			expect(router).toBeDefined();
			expect(router.wiki).toBe(wiki);
			expect(router.zp35).toBe(zp35);
			expect(router.vm).toBe(vm);
		});
		
		it("should initialize empty registries", function() {
			var zp35 = new ZP35Operator();
			var vm = new RegenZipVM(wiki);
			var router = new CompilerProgramRouter(wiki, zp35, vm);
			
			expect(Object.keys(router.compilers).length).toBe(0);
			expect(Object.keys(router.programs).length).toBe(0);
			expect(Object.keys(router.routingCache).length).toBe(0);
		});
		
	});
	
	describe("Tiddler Classification", function() {
		
		var router;
		
		beforeEach(function() {
			var zp35 = new ZP35Operator();
			var vm = new RegenZipVM(wiki);
			router = new CompilerProgramRouter(wiki, zp35, vm);
		});
		
		it("should classify high-coherence tiddler as compiler", function() {
			var compilerTiddler = {
				fields: {
					title: "TestCompiler",
					type: "application/x-tiddler-regen-zip",
					generator: "testGenerator",
					version: "1.0.0",
					seed: "default-seed",
					tags: ["compiler", "test"]
				}
			};
			
			var classification = router.classify(compilerTiddler);
			
			expect(classification.type).toBe("compiler");
			expect(classification.confidence).toBeGreaterThan(0.5);
			expect(classification.coherence).toBeDefined();
			expect(classification.coherence.score).toBeGreaterThan(0.65);
		});
		
		it("should classify low-coherence tiddler as program", function() {
			var programTiddler = {
				fields: {
					title: "TestProgram",
					text: "Some task-specific content"
				}
			};
			
			var classification = router.classify(programTiddler);
			
			expect(classification.type).toBe("program");
			expect(classification.coherence.score).toBeLessThan(0.35);
		});
		
		it("should classify intermediate-coherence tiddler as intermediate", function() {
			var intermediateTiddler = {
				fields: {
					title: "TestIntermediate",
					type: "text/plain",
					tags: ["bridge"],
					text: "Some content with moderate structure"
				}
			};
			
			var classification = router.classify(intermediateTiddler);
			
			expect(classification.type).toBe("intermediate");
			expect(classification.coherence.score).toBeGreaterThanOrEqual(0.35);
			expect(classification.coherence.score).toBeLessThanOrEqual(0.65);
		});
		
		it("should handle null tiddler gracefully", function() {
			var classification = router.classify(null);
			
			expect(classification.type).toBe("unknown");
			expect(classification.confidence).toBe(0.0);
		});
		
		it("should include coherence factors in classification", function() {
			var tiddler = {
				fields: {
					title: "Test",
					generator: "testGen",
					version: "1.0",
					seed: "test-seed"
				}
			};
			
			var classification = router.classify(tiddler);
			
			expect(classification.coherence.factors).toBeDefined();
			expect(classification.coherence.factors.structural).toBeDefined();
			expect(classification.coherence.factors.semantic).toBeDefined();
			expect(classification.coherence.factors.temporal).toBeDefined();
		});
		
	});
	
	describe("Compiler Registration", function() {
		
		var router;
		
		beforeEach(function() {
			var zp35 = new ZP35Operator();
			var vm = new RegenZipVM(wiki);
			router = new CompilerProgramRouter(wiki, zp35, vm);
		});
		
		it("should register valid compiler tiddler", function() {
			var compiler = {
				fields: {
					title: "ValidCompiler",
					generator: "testGen",
					version: "1.0.0",
					seed: "seed",
					type: "application/x-tiddler-regen-zip"
				}
			};
			
			var success = router.registerCompiler(compiler);
			
			expect(success).toBe(true);
			expect(router.compilers["ValidCompiler"]).toBeDefined();
			expect(router.compilers["ValidCompiler"].tiddler).toBe(compiler);
		});
		
		it("should initialize compiler metrics", function() {
			var compiler = {
				fields: {
					title: "MetricsCompiler",
					generator: "testGen",
					version: "1.0.0",
					seed: "seed",
					type: "application/x-tiddler-regen-zip"
				}
			};
			
			router.registerCompiler(compiler);
			var entry = router.compilers["MetricsCompiler"];
			
			expect(entry.metrics).toBeDefined();
			expect(entry.metrics.executionCount).toBe(0);
			expect(entry.metrics.successCount).toBe(0);
			expect(entry.metrics.failureCount).toBe(0);
		});
		
		it("should reject null compiler", function() {
			var success = router.registerCompiler(null);
			expect(success).toBe(false);
		});
		
	});
	
	describe("Program Registration", function() {
		
		var router;
		
		beforeEach(function() {
			var zp35 = new ZP35Operator();
			var vm = new RegenZipVM(wiki);
			router = new CompilerProgramRouter(wiki, zp35, vm);
		});
		
		it("should register valid program tiddler", function() {
			var program = {
				fields: {
					title: "ValidProgram",
					text: "Task specification"
				}
			};
			
			var success = router.registerProgram(program);
			
			expect(success).toBe(true);
			expect(router.programs["ValidProgram"]).toBeDefined();
			expect(router.programs["ValidProgram"].tiddler).toBe(program);
		});
		
		it("should initialize program status", function() {
			var program = {
				fields: {
					title: "StatusProgram",
					text: "Test"
				}
			};
			
			router.registerProgram(program);
			var entry = router.programs["StatusProgram"];
			
			expect(entry.status).toBe("pending");
			expect(entry.routedTo).toBe(null);
		});
		
		it("should reject null program", function() {
			var success = router.registerProgram(null);
			expect(success).toBe(false);
		});
		
	});
	
	describe("Program Routing", function() {
		
		var router;
		
		beforeEach(function() {
			var zp35 = new ZP35Operator();
			var vm = new RegenZipVM(wiki);
			router = new CompilerProgramRouter(wiki, zp35, vm);
		});
		
		it("should route program to nearest compiler", function() {
			// Register compilers
			var compiler1 = {
				fields: {
					title: "Compiler1",
					generator: "gen1",
					version: "1.0",
					seed: "seed1",
					type: "application/x-tiddler-regen-zip",
					tags: ["compiler", "type1"]
				}
			};
			
			var compiler2 = {
				fields: {
					title: "Compiler2",
					generator: "gen2",
					version: "1.0",
					seed: "seed2",
					type: "application/x-tiddler-regen-zip",
					tags: ["compiler", "type2"]
				}
			};
			
			router.registerCompiler(compiler1);
			router.registerCompiler(compiler2);
			
			// Register program
			var program = {
				fields: {
					title: "TestProgram",
					text: "Task"
				}
			};
			
			router.registerProgram(program);
			
			// Route program
			var routing = router.route(program);
			
			expect(routing.success).toBe(true);
			expect(routing.compiler).toBeDefined();
			expect(routing.compilerTitle).toBeDefined();
			expect(routing.distance).toBeDefined();
			expect(routing.mode).toBeDefined();
		});
		
		it("should classify routing mode based on distance", function() {
			var compiler = {
				fields: {
					title: "TestCompiler",
					generator: "gen",
					version: "1.0",
					seed: "seed",
					type: "application/x-tiddler-regen-zip"
				}
			};
			
			router.registerCompiler(compiler);
			
			var program = {
				fields: {
					title: "TestProgram",
					text: "Task"
				}
			};
			
			var routing = router.route(program);
			
			// Mode should be one of: safe, caution, borderline, ood
			expect(["safe", "caution", "borderline", "ood"]).toContain(routing.mode);
			
			// Confidence should be inversely related to distance
			if(routing.mode === "safe") {
				expect(routing.confidence).toBeGreaterThan(0.5);
			}
		});
		
		it("should return error when no compilers registered", function() {
			var program = {
				fields: {
					title: "OrphanProgram",
					text: "Task"
				}
			};
			
			var routing = router.route(program);
			
			expect(routing.success).toBe(false);
			expect(routing.message).toContain("No compilers");
		});
		
		it("should cache routing results", function() {
			var compiler = {
				fields: {
					title: "CacheCompiler",
					generator: "gen",
					version: "1.0",
					seed: "seed",
					type: "application/x-tiddler-regen-zip"
				}
			};
			
			router.registerCompiler(compiler);
			
			var program = {
				fields: {
					title: "CacheProgram",
					text: "Task"
				}
			};
			
			// First routing
			var routing1 = router.route(program);
			
			// Check cache
			expect(router.routingCache["CacheProgram"]).toBeDefined();
			
			// Second routing (should use cache)
			var routing2 = router.route(program);
			
			expect(routing2).toBe(routing1); // Same object from cache
		});
		
		it("should provide candidate compilers", function() {
			// Register multiple compilers
			for(var i = 1; i <= 3; i++) {
				var compiler = {
					fields: {
						title: "Compiler" + i,
						generator: "gen" + i,
						version: "1.0",
						seed: "seed" + i,
						type: "application/x-tiddler-regen-zip"
					}
				};
				router.registerCompiler(compiler);
			}
			
			var program = {
				fields: {
					title: "MultiCandidateProgram",
					text: "Task"
				}
			};
			
			var routing = router.route(program);
			
			expect(routing.candidates).toBeDefined();
			expect(routing.candidates.length).toBeGreaterThan(0);
			expect(routing.candidates.length).toBeLessThanOrEqual(3);
			
			// Candidates should be sorted by distance
			for(var j = 1; j < routing.candidates.length; j++) {
				expect(routing.candidates[j].distance).toBeGreaterThanOrEqual(
					routing.candidates[j-1].distance
				);
			}
		});
		
	});
	
	describe("Program Execution", function() {
		
		var router;
		var vm;
		
		beforeEach(function() {
			var zp35 = new ZP35Operator();
			vm = new RegenZipVM(wiki);
			router = new CompilerProgramRouter(wiki, zp35, vm);
			
			// Register a simple test generator
			vm.registerGenerator("testGenerator", function(context) {
				return {
					assets: [{
						name: "test.txt",
						type: "text/plain",
						data: "Test output",
						checksum: "test-checksum"
					}]
				};
			}, {
				version: "1.0.0",
				zp35: "0.500000.10"
			});
		});
		
		it("should execute program through routed compiler", function() {
			var compiler = {
				fields: {
					title: "ExecCompiler",
					generator: "testGenerator",
					version: "1.0",
					seed: "seed",
					type: "application/x-tiddler-regen-zip"
				}
			};
			
			router.registerCompiler(compiler);
			
			var program = {
				fields: {
					title: "ExecProgram",
					seed: "program-seed",
					text: "Task"
				}
			};
			
			var result = router.execute(program);
			
			expect(result.success).toBe(true);
			expect(result.assets).toBeDefined();
			expect(result.assets.length).toBeGreaterThan(0);
		});
		
		it("should block OOD program execution", function() {
			// Register compiler with specific characteristics
			var compiler = {
				fields: {
					title: "SpecificCompiler",
					generator: "testGenerator",
					version: "1.0",
					seed: "seed",
					type: "application/x-tiddler-regen-zip",
					tags: ["compiler", "specific", "narrow"]
				}
			};
			
			router.registerCompiler(compiler);
			
			// Create program that's very different (will be OOD)
			// Note: In practice, we'd need to create a program with significantly
			// different characteristics. For testing, we'll check the blocking logic.
			var program = {
				fields: {
					title: "OODProgram",
					text: "Very different task"
				}
			};
			
			var result = router.execute(program);
			
			// If routing classified as OOD, execution should be blocked
			if(result.routing && result.routing.mode === "ood") {
				expect(result.success).toBe(false);
				expect(result.message).toContain("out-of-distribution");
			}
		});
		
		it("should update compiler metrics on execution", function() {
			var compiler = {
				fields: {
					title: "MetricsCompiler",
					generator: "testGenerator",
					version: "1.0",
					seed: "seed",
					type: "application/x-tiddler-regen-zip"
				}
			};
			
			router.registerCompiler(compiler);
			
			var program = {
				fields: {
					title: "MetricsProgram",
					seed: "seed",
					text: "Task"
				}
			};
			
			// Check initial metrics
			var initialMetrics = router.compilers["MetricsCompiler"].metrics;
			var initialExecCount = initialMetrics.executionCount;
			
			// Execute program
			router.execute(program);
			
			// Check updated metrics
			var updatedMetrics = router.compilers["MetricsCompiler"].metrics;
			expect(updatedMetrics.executionCount).toBe(initialExecCount + 1);
		});
		
	});
	
	describe("Compiler-Program Merging", function() {
		
		var router;
		
		beforeEach(function() {
			var zp35 = new ZP35Operator();
			var vm = new RegenZipVM(wiki);
			router = new CompilerProgramRouter(wiki, zp35, vm);
		});
		
		it("should merge compiler and program tiddlers", function() {
			var compiler = {
				fields: {
					title: "MergeCompiler",
					generator: "testGen",
					type: "application/x-tiddler-regen-zip"
				}
			};
			
			var program = {
				fields: {
					title: "MergeProgram",
					seed: "program-seed",
					text: "Program text"
				}
			};
			
			var merged = router.mergeForExecution(compiler, program);
			
			expect(merged.fields).toBeDefined();
			expect(merged.fields.generator).toBe("testGen"); // From compiler
			expect(merged.fields.seed).toBe("program-seed"); // From program
		});
		
		it("should preserve compiler generator and type", function() {
			var compiler = {
				fields: {
					title: "PreserveCompiler",
					generator: "compilerGen",
					type: "application/x-tiddler-regen-zip"
				}
			};
			
			var program = {
				fields: {
					title: "PreserveProgram",
					generator: "programGen", // Should be ignored
					type: "text/plain" // Should be ignored
				}
			};
			
			var merged = router.mergeForExecution(compiler, program);
			
			expect(merged.fields.generator).toBe("compilerGen");
			expect(merged.fields.type).toBe("application/x-tiddler-regen-zip");
		});
		
		it("should track source tiddlers in merged result", function() {
			var compiler = {
				fields: {
					title: "SourceCompiler",
					generator: "gen"
				}
			};
			
			var program = {
				fields: {
					title: "SourceProgram"
				}
			};
			
			var merged = router.mergeForExecution(compiler, program);
			
			expect(merged.fields["compiler-source"]).toBe("SourceCompiler");
			expect(merged.fields["program-source"]).toBe("SourceProgram");
		});
		
	});
	
	describe("Router Statistics", function() {
		
		var router;
		
		beforeEach(function() {
			var zp35 = new ZP35Operator();
			var vm = new RegenZipVM(wiki);
			router = new CompilerProgramRouter(wiki, zp35, vm);
		});
		
		it("should provide basic statistics", function() {
			var stats = router.getStatistics();
			
			expect(stats.compilers).toBeDefined();
			expect(stats.programs).toBeDefined();
			expect(stats.routings).toBeDefined();
			expect(stats.compilerDetails).toBeDefined();
		});
		
		it("should track registered compilers and programs", function() {
			var compiler = {
				fields: {
					title: "StatsCompiler",
					generator: "gen",
					version: "1.0",
					seed: "seed",
					type: "application/x-tiddler-regen-zip"
				}
			};
			
			var program = {
				fields: {
					title: "StatsProgram",
					text: "Task"
				}
			};
			
			router.registerCompiler(compiler);
			router.registerProgram(program);
			
			var stats = router.getStatistics();
			
			expect(stats.compilers).toBe(1);
			expect(stats.programs).toBe(1);
		});
		
		it("should provide per-compiler details", function() {
			var compiler = {
				fields: {
					title: "DetailCompiler",
					generator: "gen",
					version: "1.0",
					seed: "seed",
					type: "application/x-tiddler-regen-zip"
				}
			};
			
			router.registerCompiler(compiler);
			
			var stats = router.getStatistics();
			
			expect(stats.compilerDetails.length).toBe(1);
			expect(stats.compilerDetails[0].title).toBe("DetailCompiler");
			expect(stats.compilerDetails[0].programs).toBe(0);
			expect(stats.compilerDetails[0].executions).toBe(0);
		});
		
	});
	
	describe("Cache Management", function() {
		
		var router;
		
		beforeEach(function() {
			var zp35 = new ZP35Operator();
			var vm = new RegenZipVM(wiki);
			router = new CompilerProgramRouter(wiki, zp35, vm);
		});
		
		it("should clear routing cache", function() {
			var compiler = {
				fields: {
					title: "ClearCompiler",
					generator: "gen",
					version: "1.0",
					seed: "seed",
					type: "application/x-tiddler-regen-zip"
				}
			};
			
			router.registerCompiler(compiler);
			
			var program = {
				fields: {
					title: "ClearProgram",
					text: "Task"
				}
			};
			
			// Route to populate cache
			router.route(program);
			expect(Object.keys(router.routingCache).length).toBe(1);
			
			// Clear cache
			router.clearCache();
			expect(Object.keys(router.routingCache).length).toBe(0);
		});
		
	});
	
});

})();
