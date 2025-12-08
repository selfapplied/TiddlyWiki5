# Compiler-Program Pattern Implementation Summary

**Date:** December 8, 2024  
**Status:** Complete  
**Test Status:** ✅ All 1499 specs passing  
**Security Status:** ✅ No vulnerabilities found

---

## Overview

This implementation adds the **Compiler-Program Pattern** to TiddlyWiki - a novel architectural pattern that reframes how we think about data coherence and computation by recognizing two distinct roles for tiddlers:

- **Compiler Tiddlers**: High-coherence semantic kernels that define valid transformation spaces
- **Program Tiddlers**: Low-coherence, ephemeral task specifications that get routed through compilers

The pattern maps directly to ML concepts where training builds the compiler (coherent latent geometry) and reasoning runs programs through the compiler (prompts → outputs).

---

## What Was Implemented

### 1. Core Module: Compiler-Program Router

**File:** `core/modules/utils/compiler-program-router.js` (600+ lines)

**Key Features:**
- **Classification**: Analyzes tiddler coherence using ZP35 metrics
  - High coherence (>0.65) → Compiler
  - Low coherence (<0.35) → Program
  - Intermediate (0.35-0.65) → Bridge/mediator

- **Routing**: Uses ZP35 distance to find best compiler for each program
  - Safe mode: distance < κ (0.35)
  - Caution mode: distance < 2κ (0.70)
  - Borderline: distance < 0.70
  - Out-of-distribution (blocked): distance ≥ 0.70

- **Execution**: Merges compiler + program and executes through REGEN-ZIP VM
  - Compiler provides semantic kernel (generator, type, etc.)
  - Program provides task specifics (seed, params, etc.)
  - VM materializes assets

- **Safety**: 
  - OOD detection blocks unsafe execution
  - ZP35 coherence checking
  - Checksum verification

- **Observability**:
  - Execution metrics per compiler
  - Routing statistics
  - Success/failure tracking

**Architecture:**
```
Tiddler → Classify → Register as Compiler/Program
                         ↓
Program → Route → Find nearest Compiler by ZP35 distance
                         ↓
         Merge → Compiler kernel + Program specifics
                         ↓
         Execute → REGEN-ZIP VM → Assets
```

### 2. Documentation

**File:** `COMPILER_PROGRAM_PATTERN.md` (900+ lines)

**Contents:**
- Conceptual framework (coherent data as compiler / chaotic data as program)
- Architecture diagrams
- API reference
- Usage examples
- Mapping to ML concepts (training vs reasoning)
- Advanced topics (kernel splitting/merging, versioning, bridges)
- Future directions

### 3. Examples

**File:** `COMPILER_PROGRAM_EXAMPLE.js` (600+ lines)

**Examples provided:**
1. Fractal generator compiler + programs
2. Text processor compiler + programs
3. Complete pipeline demonstration
4. Out-of-distribution detection
5. Compiler evolution (versioning)
6. Multi-kernel composition

### 4. Comprehensive Tests

**File:** `editions/test/tiddlers/tests/test-compiler-program-router.js` (700+ lines)

**Test Coverage:**
- Router construction (2 tests)
- Tiddler classification (5 tests)
- Compiler registration (3 tests)
- Program registration (3 tests)
- Program routing (6 tests)
- Program execution (3 tests)
- Compiler-program merging (3 tests)
- Router statistics (3 tests)
- Cache management (1 test)

**Total:** 29 test cases covering all functionality

---

## How It Maps to the Problem Statement

The implementation directly addresses the problem statement's request:

### "Composable VMs as Semantic Kernels"

✅ **Implemented:**
- Compilers act as semantic kernels with fixed transformation spaces
- Programs are composed through these kernels
- ZP35 provides the type system ensuring safe composition

### "Coherent Data as Compiler, Chaotic Data as Program"

✅ **Implemented:**
- Classification system distinguishes high-coherence (compiler) from low-coherence (program) tiddlers
- Coherence analysis uses structural, semantic, and temporal factors
- Programs route through appropriate compilers based on semantic distance

### "Assets Stay Fixed, Kernels/Compilers Get Better"

✅ **Implemented:**
- REGEN-ZIP generators (assets) remain stable
- Compilers can evolve via versioning
- Programs are ephemeral and disposable
- Clean separation between what exists (compilers), how we think (transformations), what we do (programs)

### "Training vs Reasoning"

✅ **Implemented:**
- Compilers ≈ trained models (coherent latent geometry)
- Programs ≈ prompts/inference (task specifications)
- Execution ≈ running programs through model
- OOD detection ≈ uncertainty estimation

### "ZP35 Distance for Routing"

✅ **Implemented:**
- Programs routed to nearest compiler in fractal space
- Guardian threshold (κ=0.35) enforced
- OOD threshold (0.70) blocks unsafe execution
- Candidate ranking by distance

---

## Key Design Decisions

### 1. Coherence Scoring Algorithm

Uses three factors:
- **Structural** (40%): Type, tags, generator fields
- **Semantic** (40%): Distance from plateau center in fractal space
- **Temporal** (20%): Version, seed fields

This balances immediate observable properties (structural) with deep semantic positioning (ZP35-based) and stability indicators (temporal).

### 2. Classification Thresholds

- High coherence: >0.65 (compiler)
- Low coherence: <0.35 (program)
- Intermediate: 0.35-0.65 (bridge)

These thresholds create clear separation while allowing for intermediate bridging tiddlers.

### 3. Routing Safety

Three-tier safety system:
1. **Safe** (d < κ): Execute immediately, high confidence
2. **Caution** (d < 2κ): Execute with warning, medium confidence
3. **OOD** (d ≥ 0.70): Block execution, suggest alternatives

### 4. Merge Strategy

Compiler provides:
- Generator function
- Type
- Version
- ZP35 signature

Program provides:
- Seed
- Parameters
- Task-specific data

This ensures semantic kernel stability while allowing task flexibility.

### 5. Constants vs Magic Numbers

All thresholds and scaling factors defined as named constants:
```javascript
var THRESHOLDS = {
  HIGH_COHERENCE: 0.65,
  LOW_COHERENCE: 0.35,
  KAPPA: 0.35,
  OOD_THRESHOLD: 0.70
};

var COHERENCE_CONSTANTS = {
  SEMANTIC_DISTANCE_SCALE: 10,
  SCORE_MIN: 0,
  SCORE_MAX: 1,
  MAX_CANDIDATES: 3
};
```

This improves maintainability and makes the system's behavior more explicit.

---

## Testing Results

### Test Execution
- **Total specs:** 1499
- **Failures:** 0
- **Pending:** 2 (unrelated to this feature)
- **Time:** ~8.3 seconds

### New Tests
- **Compiler-Program Router:** 29 new test cases
- **Coverage:** All major functionality tested
- **Edge cases:** OOD handling, null inputs, caching, metrics

### Security
- **CodeQL scan:** ✅ No vulnerabilities
- **Code review:** ✅ Addressed all feedback

---

## Usage Example

```javascript
// Setup
var wiki = $tw.wiki;
var zp35 = new ZP35Operator();
var vm = new RegenZipVM(wiki);
var router = new CompilerProgramRouter(wiki, zp35, vm);

// Register generator with VM
vm.registerGenerator("fractalGenerator", generatorFn, {
  version: "1.0.0",
  zp35: "0.618034.20"
});

// Create compiler (high coherence)
var compiler = {
  fields: {
    title: "FractalCompiler",
    generator: "fractalGenerator",
    version: "1.0.0",
    seed: "default",
    type: "application/x-tiddler-regen-zip"
  }
};
router.registerCompiler(compiler);

// Create program (low coherence)
var program = {
  fields: {
    title: "GenerateFractal_Task1",
    seed: "task-seed-42",
    params: JSON.stringify({zoom: 2.5})
  }
};
router.registerProgram(program);

// Execute program through routed compiler
var result = router.execute(program);

console.log("Success:", result.success);
console.log("Compiler:", result.compiler);
console.log("Assets:", result.assets.length);
console.log("Routing mode:", result.routing.mode);
```

---

## Integration with Existing Systems

### REGEN-ZIP VM
- Router uses VM for execution
- Merges compiler + program into executable tiddler
- VM handles opcode execution, verification, asset generation

### ZP35 Operator
- Router uses ZP35 for classification
- Golden operator maps tiddlers to fractal coordinates
- Distance metrics determine routing decisions

### TiddlyWiki Core
- Works with existing tiddler infrastructure
- Compatible with field-based metadata
- Integrates with tag system

---

## Future Enhancements

### Possible Extensions
1. **Distributed Compilers**: Remote execution, compiler marketplace
2. **ML Integration**: Actual model weights as compilers, embeddings for distance
3. **Compiler Composition**: Chain multiple compilers, pipeline transformations
4. **Auto-tuning**: Learn optimal routing based on execution history
5. **Visualization**: Visual tools for exploring compiler-program graphs

### Backward Compatibility
- All changes are additive
- No breaking changes to existing APIs
- New module doesn't affect existing functionality

---

## Maintenance Notes

### Key Files to Monitor
- `core/modules/utils/compiler-program-router.js` - Core implementation
- `core/modules/utils/zp35-operator.js` - Used for classification/routing
- `core/modules/utils/regen-zip-vm.js` - Used for execution

### Constants to Review
If learnability research updates the guardian threshold:
- Update `THRESHOLDS.KAPPA` and `THRESHOLDS.OOD_THRESHOLD`
- Document reasoning in commit message
- Update documentation

### Test Maintenance
- Run full test suite after any changes to router
- Monitor for regressions in classification logic
- Keep tests synchronized with threshold changes

---

## Conclusion

This implementation successfully brings the "compiler-program pattern" to TiddlyWiki, providing:

✅ **Clear separation** between stable kernels (compilers) and ephemeral tasks (programs)  
✅ **Safe composition** via ZP35 distance and OOD detection  
✅ **Observability** through metrics and statistics  
✅ **Extensibility** via clean APIs and versioning support  
✅ **Robustness** with comprehensive tests and security scanning  

The pattern provides a concrete, inspectable toy model of "training builds the compiler; reasoning runs programs through it" - all within TiddlyWiki as the OS.

---

**Implementation Team:** GitHub Copilot  
**Review Status:** Approved  
**Ready for Merge:** ✅ Yes
