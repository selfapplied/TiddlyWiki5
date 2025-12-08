# The Compiler-Program Pattern in TiddlyWiki

**Version:** 1.0  
**Date:** December 8, 2024  
**Status:** Implementation Specification

---

## Executive Summary

This document describes the **Compiler-Program Pattern** - a novel architectural pattern for TiddlyWiki that reframes how we think about data coherence and computation.

### The Core Insight

Instead of treating all tiddlers uniformly, we recognize two distinct roles:

1. **Compiler Tiddlers** (high coherence)
   - Act as semantic kernels or type systems
   - Define valid transformation spaces
   - Provide stable, long-lived asset manifolds
   - Analogous to: trained models, compilers, fixed assets

2. **Program Tiddlers** (low coherence / chaotic)
   - Act as ephemeral programs or prompts
   - Specify task-specific behavior
   - Get routed through compilers for execution
   - Analogous to: inference prompts, programs, datasets

This pattern maps directly to ML concepts:
- **Training** = building compilers (coherent latent geometry)
- **Reasoning** = running programs through compilers (prompts → outputs)

---

## 1. Conceptual Framework

### 1.1 Coherent Data as Compiler

**High-coherence data** (tight clusters, strong ZP35 signature, low curvature):
- Behaves like a **compiler** or **type system**
- Defines what "valid transformations" look like in this region
- Provides semantic constraints and guarantees
- Remains stable across invocations

Example compiler tiddler characteristics:
- Has registered generator function
- Has versioned, typed content
- Has stable ZP35 signature
- Defines a semantic kernel (e.g., "fractal image generator", "text processor")

### 1.2 Chaotic Data as Program

**Chaotic/heterogeneous data** (messy, multi-cluster, boundary-heavy):
- Behaves like a **program** being run through the compiler
- Asks "given this mess, how do I express it in the compiler's dialect?"
- Task-specific, ephemeral
- Changes frequently

Example program tiddler characteristics:
- No generator, just input data
- Task-specific parameters
- Unstable ZP35 signature
- Specifies "what to do now" (e.g., "generate fractal with seed X")

### 1.3 Fixed Assets, Evolving Compilers

The pattern enables:

1. **Stable asset formats** - Generators + seeds remain valid
2. **Evolving compilers** - Can reinterpret assets in richer ways
3. **Clean separation** between:
   - *What exists* (compilers / world model / kernels)
   - *How we think about it* (semantic transformations)
   - *What we're trying to do* (programs / prompts)

This is analogous to:
- Foundation model weights = fixed asset manifold
- Fine-tuning = compiler refinements
- Prompts = ephemeral programs

---

## 2. Architecture

### 2.1 System Components

```
┌─────────────────────────────────────────────────┐
│           TiddlyWiki Kernel                      │
│       (tiddler store, parser, UI)                │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│      Compiler-Program Router                     │
│   • Classify tiddlers (compiler vs program)      │
│   • Route programs to compilers                  │
│   • Use ZP35 distance for selection              │
│   • Handle out-of-distribution cases             │
└────────────────┬────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼────────┐       ┌───────▼──────┐
│ Compiler   │       │  Program      │
│ Tiddlers   │       │  Tiddlers     │
│ (high      │       │  (chaotic     │
│ coherence) │       │  data)        │
└───┬────────┘       └───────┬───────┘
    │                        │
    └────────┬───────────────┘
             │
┌────────────▼────────────────────────────────────┐
│         REGEN-ZIP Virtual Machine                │
│   • Merge compiler + program                     │
│   • Execute generator with context               │
│   • Materialize assets                           │
└────────────┬────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────┐
│        ZP35 Compatibility Layer                  │
│   • Verify coherence (κ = 0.35)                  │
│   • Measure semantic distance                    │
│   • Ensure composition safety                    │
└──────────────────────────────────────────────────┘
```

### 2.2 Classification Algorithm

The router classifies tiddlers using ZP35 coherence metrics:

```javascript
function classify(tiddler):
  // Calculate ZP35 metrics
  coord = zp35.applyGoldenOperator(tiddler)
  height = zp35.calculateOrdinalHeight(tiddler)
  
  // Analyze coherence (structural, semantic, temporal)
  coherence = analyzeCoherence(tiddler)
  
  // Classify based on coherence score
  if coherence.score > 0.65:
    return "compiler"  // High coherence
  else if coherence.score < 0.35:
    return "program"   // Low coherence (chaotic)
  else:
    return "intermediate"  // Bridge/mediator
```

### 2.3 Routing Algorithm

Programs are routed to compilers using ZP35 distance:

```javascript
function route(programTiddler):
  programCoord = zp35.applyGoldenOperator(programTiddler)
  
  // Find nearest compiler in fractal space
  bestCompiler = null
  bestDistance = Infinity
  
  for each compiler in compilers:
    compilerCoord = zp35.applyGoldenOperator(compiler)
    distance = |programCoord - compilerCoord|
    
    if distance < bestDistance:
      bestDistance = distance
      bestCompiler = compiler
  
  // Check safety
  if bestDistance < κ:
    mode = "safe"
  else if bestDistance < 2κ:
    mode = "caution"
  else if bestDistance < 0.70:
    mode = "borderline"
  else:
    mode = "out-of-distribution"  // Block execution
  
  return {compiler: bestCompiler, distance, mode}
```

### 2.4 Execution Pipeline

```javascript
function execute(programTiddler):
  // 1. Route program to compiler
  routing = route(programTiddler)
  
  if routing.mode == "out-of-distribution":
    return error("Program OOD - execution blocked")
  
  // 2. Merge compiler + program
  executionTiddler = merge(routing.compiler, programTiddler)
  
  // 3. Execute through REGEN-ZIP VM
  vm.load(executionTiddler)
  result = vm.run()
  
  // 4. Return materialized assets
  return result.assets
```

---

## 3. API Reference

### 3.1 CompilerProgramRouter Constructor

```javascript
var router = new CompilerProgramRouter(wiki, zp35Operator, regenZipVM);
```

**Parameters:**
- `wiki` - TiddlyWiki instance
- `zp35Operator` - ZP35 operator instance
- `regenZipVM` - REGEN-ZIP VM instance

### 3.2 Classification

```javascript
var classification = router.classify(tiddler);
```

**Returns:**
```javascript
{
  type: "compiler" | "program" | "intermediate",
  confidence: 0.85,
  coord: 0.618034,
  height: 15,
  signature: "0.618034.15",
  coherence: {
    score: 0.75,
    factors: {
      structural: 0.8,
      semantic: 0.7,
      temporal: 0.6
    }
  },
  message: "High-coherence asset - acts as semantic kernel"
}
```

### 3.3 Register Compiler

```javascript
var success = router.registerCompiler(compilerTiddler);
```

Registers a high-coherence tiddler as a compiler/kernel.

### 3.4 Register Program

```javascript
var success = router.registerProgram(programTiddler);
```

Registers a chaotic tiddler as a program to be compiled.

### 3.5 Route Program

```javascript
var routing = router.route(programTiddler);
```

**Returns:**
```javascript
{
  success: true,
  compiler: compilerObject,
  compilerTitle: "FractalCompiler",
  distance: 0.15,
  mode: "safe",
  confidence: 0.95,
  message: "Program routed to compiler within safe coherence range",
  programCoord: 0.500000,
  compilerCoord: 0.618034,
  candidates: [...]  // Top 3 compiler candidates
}
```

### 3.6 Execute Program

```javascript
var result = router.execute(programTiddler);
```

**Returns:**
```javascript
{
  success: true,
  assets: [
    {
      name: "fractal.png",
      type: "image/png",
      data: ...,
      checksum: "..."
    }
  ],
  metadata: {...},
  routing: {...},
  compiler: "FractalCompiler",
  program: "GenerateFractal_Task1"
}
```

### 3.7 Get Statistics

```javascript
var stats = router.getStatistics();
```

**Returns:**
```javascript
{
  compilers: 5,
  programs: 20,
  routings: 20,
  compilerDetails: [
    {
      title: "FractalCompiler",
      programs: 12,
      executions: 45,
      successes: 43,
      failures: 2,
      successRate: 0.955
    }
  ]
}
```

---

## 4. Usage Examples

### 4.1 Creating a Compiler Tiddler

```javascript
// Create a high-coherence tiddler that acts as a compiler
var compilerTiddler = {
  fields: {
    title: "FractalCompiler",
    type: "application/x-tiddler-regen-zip",
    generator: "fractalGenerator",
    version: "1.0.0",
    seed: "golden-default",
    zp35: "0.618034.20",
    tags: ["compiler", "graphics", "procedural"],
    text: "Generates fractal images using Mandelbrot algorithm"
  }
};

// Register as compiler
router.registerCompiler(compilerTiddler);
```

### 4.2 Creating a Program Tiddler

```javascript
// Create a low-coherence tiddler that acts as a program
var programTiddler = {
  fields: {
    title: "GenerateFractal_Task1",
    seed: "task-specific-seed-42",
    params: JSON.stringify({
      zoom: 2.5,
      centerX: -0.5,
      centerY: 0.0,
      maxIterations: 100
    }),
    text: "Generate a Mandelbrot fractal zoomed into the main set"
  }
};

// Register as program
router.registerProgram(programTiddler);
```

### 4.3 Executing a Program

```javascript
// Execute program through routed compiler
var result = router.execute(programTiddler);

if(result.success) {
  console.log("Generated " + result.assets.length + " assets");
  console.log("Compiled by: " + result.compiler);
  console.log("Routing mode: " + result.routing.mode);
  
  // Use generated assets
  result.assets.forEach(function(asset) {
    console.log("Asset: " + asset.name + " (" + asset.type + ")");
  });
} else {
  console.error("Execution failed: " + result.message);
}
```

### 4.4 Complete Pipeline Example

```javascript
// Setup
var wiki = $tw.wiki;
var zp35 = new ZP35Operator();
var vm = new RegenZipVM(wiki);
var router = new CompilerProgramRouter(wiki, zp35, vm);

// Register a generator with VM
vm.registerGenerator("fractalGenerator", function(context) {
  var seed = context.seed;
  var rng = context.rng;
  var params = JSON.parse(context.tiddler.fields.params || "{}");
  
  // Generate fractal image
  var imageData = generateMandelbrot(params, rng);
  
  return {
    assets: [{
      name: "fractal.png",
      type: "image/png",
      data: imageData,
      checksum: computeChecksum(imageData)
    }]
  };
}, {
  version: "1.0.0",
  zp35: "0.618034.20"
});

// Create and register compiler
var compiler = createCompilerTiddler("FractalCompiler");
router.registerCompiler(compiler);

// Create and execute multiple programs
var programs = [
  createProgramTiddler("Task1", {zoom: 2.5}),
  createProgramTiddler("Task2", {zoom: 5.0}),
  createProgramTiddler("Task3", {zoom: 10.0})
];

programs.forEach(function(program) {
  router.registerProgram(program);
  var result = router.execute(program);
  
  console.log("Program: " + program.fields.title);
  console.log("Routed to: " + result.compiler);
  console.log("Assets: " + result.assets.length);
});

// Get statistics
var stats = router.getStatistics();
console.log("Total executions: " + 
  stats.compilerDetails[0].executions);
console.log("Success rate: " + 
  (stats.compilerDetails[0].successRate * 100).toFixed(1) + "%");
```

---

## 5. Mapping to ML Concepts

### 5.1 Training vs Reasoning

| ML Concept | TiddlyWiki Pattern |
|------------|-------------------|
| **Training** | Building compiler tiddlers |
| Learns from chaotic data | Analyzes patterns, creates generators |
| Produces weights/model | Produces generator functions |
| Creates latent geometry | Creates semantic kernel (ZP35 space) |
| Defines valid transitions | Defines valid transformations |

| ML Concept | TiddlyWiki Pattern |
|------------|-------------------|
| **Reasoning/Inference** | Executing program tiddlers |
| Takes prompt (program) | Takes program tiddler |
| Runs through model | Routes through compiler |
| Uses latent geometry | Uses ZP35 distance |
| Produces output | Materializes assets |

### 5.2 Foundation Models Analogy

**Foundation Model Weights** ↔ **Compiler Tiddlers**
- Stable, long-lived
- Define capability manifold
- Expensive to create
- Reusable across tasks

**Prompts** ↔ **Program Tiddlers**
- Ephemeral, task-specific
- Specify current intent
- Cheap to create
- Single-use or few-shot

**Fine-tuning** ↔ **Compiler Evolution**
- Refine semantic kernel
- Improve for specific domains
- Maintain backward compatibility

**Context Window** ↔ **Execution Context**
- Merge compiler + program
- Provide runtime environment
- Isolate execution

### 5.3 Out-of-Distribution Detection

Both ML systems and the compiler-program pattern need to detect when inputs are OOD:

**ML Approach:**
- Measure distance in latent space
- Detect anomalies
- Use uncertainty estimates

**TiddlyWiki Approach:**
- Measure ZP35 distance
- κ threshold for safety
- Mode classification (safe/caution/OOD)

When a program is OOD (distance > 0.70):
- Execution is blocked
- Suggest creating new compiler
- Or sandbox the execution

---

## 6. Benefits and Use Cases

### 6.1 Benefits

1. **Separation of Concerns**
   - Compilers define *what's possible*
   - Programs specify *what to do now*
   - Clean interface between stable and ephemeral

2. **Composition Safety**
   - ZP35 distance ensures semantic coherence
   - Guardian threshold prevents breaking changes
   - OOD detection blocks unsafe execution

3. **Reusability**
   - Compilers are reusable across programs
   - Programs are disposable
   - Assets stay fixed while compilers evolve

4. **Inspectability**
   - Explicit routing decisions
   - Traceable execution paths
   - Measurable coherence metrics

5. **Scalability**
   - Multiple compilers for different domains
   - Automatic routing
   - Parallel execution potential

### 6.2 Use Cases

**1. Procedural Content Generation**
- Compiler: "FractalGenerator" with stable algorithms
- Programs: Various zoom levels, parameters, seeds
- Assets: Generated images

**2. Text Processing**
- Compiler: "MarkdownProcessor" with parsing rules
- Programs: Different documents to process
- Assets: Rendered HTML

**3. Data Transformation**
- Compiler: "CSVTransformer" with schema
- Programs: Different CSV files
- Assets: Normalized data

**4. Plugin System**
- Compilers: Plugin analyzers/validators
- Programs: Plugin configurations
- Assets: Validated plugins

**5. Test Generation**
- Compilers: Test template generators
- Programs: Test specifications
- Assets: Generated test cases

---

## 7. Advanced Topics

### 7.1 Kernel Splitting/Merging

As domains evolve, compilers may need to split or merge:

**Splitting** (one compiler → multiple)
- When semantic distance within compiler programs grows
- Create specialized compilers for sub-domains
- Re-route programs to appropriate compiler

**Merging** (multiple compilers → one)
- When compilers have overlapping domains
- Consolidate to reduce redundancy
- Maintain compatibility with existing programs

### 7.2 Compiler Versioning

Compilers evolve over time:

```javascript
// V1 compiler
{
  title: "FractalCompiler:v1",
  version: "1.0.0",
  generator: "fractalGeneratorV1"
}

// V2 compiler (backward compatible)
{
  title: "FractalCompiler:v2",
  version: "2.0.0",
  generator: "fractalGeneratorV2",
  compatible_with: ["1.0.0"]
}
```

Programs can specify version constraints:

```javascript
{
  title: "Task1",
  requires_compiler_version: ">=1.0.0"
}
```

### 7.3 Bridge Tiddlers

Intermediate-coherence tiddlers act as bridges:

- Facilitate composition across semantic boundaries
- Mediate between incompatible compilers
- Translate between different kernels

```javascript
// Bridge tiddler
{
  title: "FractalToMesh_Bridge",
  type: "intermediate",
  source_compiler: "FractalCompiler",
  target_compiler: "MeshGenerator",
  transform: "convertImageToHeightmap"
}
```

---

## 8. Implementation Notes

### 8.1 Performance Considerations

- **Caching**: Router caches routing decisions
- **Lazy Loading**: Compilers loaded on-demand
- **Parallel Execution**: Programs can execute in parallel

### 8.2 Safety Guarantees

- **κ Threshold**: Enforced at routing time
- **OOD Blocking**: Programs > 0.70 distance blocked
- **Checksum Verification**: All assets verified

### 8.3 Debugging

The router provides introspection:

```javascript
// Trace a program's execution
var trace = router.trace(programTiddler);
// Returns: classification → routing → execution → assets

// Inspect compiler statistics
var stats = router.getStatistics();
// Shows: usage patterns, success rates, failure modes
```

---

## 9. Future Directions

### 9.1 Machine Learning Integration

- Train actual ML models as compilers
- Use embeddings for semantic distance
- Online learning from execution patterns

### 9.2 Distributed Compilers

- Remote compiler execution
- Compiler marketplace
- Federated learning

### 9.3 Compiler Composition

- Chain multiple compilers
- Pipeline transformations
- Compose complex behaviors from simple kernels

---

## 10. Conclusion

The Compiler-Program Pattern provides a principled way to think about coherence and computation in TiddlyWiki:

- **Compilers** = stable semantic kernels (high coherence)
- **Programs** = ephemeral task specifications (chaotic)
- **Routing** = ZP35 distance-based selection
- **Execution** = REGEN-ZIP VM pipeline

This pattern maps directly to ML concepts (training vs reasoning) and provides:
- Clean separation of concerns
- Composition safety via ZP35
- Reusable, evolving compilers
- Inspectable execution

By making the compiler-program distinction explicit, we get a toy model of "training builds the compiler; reasoning runs programs through it" - all within TiddlyWiki as the OS.

---

**See Also:**
- `REGEN_ZIP_VM.md` - REGEN-ZIP VM specification
- `ZP35_GOLDEN_OPERATOR.md` - ZP35 mathematical foundations
- `COMPILER_PROGRAM_EXAMPLE.js` - Complete code examples
- `core/modules/utils/compiler-program-router.js` - Implementation
