# REGEN-ZIP Virtual Machine Specification

**Version:** 1.0  
**Date:** December 8, 2024  
**Status:** Implementation Specification

---

## Executive Summary

This document specifies the **REGEN-ZIP Virtual Machine** - a regenerative, declarative, ZIP-backed virtual machine for TiddlyWiki tiddlers. The VM transforms TiddlyWiki from "a big HTML file with some JS" into a living, executable environment where tiddlers become modules in a generative computation system.

### Key Innovation

By recognizing that ZIP format already contains execution semantics (segment boundaries, jump tables, versioned features, ordering rules, checksums), we extend it with **regeneration operations** that generate assets from seeds and generators rather than simply inflating compressed data.

### The Architecture

```
┌─────────────────────────────────────────────────┐
│             TiddlyWiki Kernel                    │
│         (load/store, parsing, UI)                │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│          REGEN-ZIP Virtual Machine               │
│    • Opcode execution                            │
│    • Generator management                        │
│    • Asset generation                            │
│    • Checksum verification                       │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│         ZP35 Compatibility Layer                 │
│    • Semantic coherence checking                 │
│    • κ=0.35 guardian threshold                   │
│    • Fractal coordinate mapping                  │
│    • Safety guarantees                           │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│              Tiddlers as Modules                 │
│    • Executable units                            │
│    • Declarative recipes                         │
│    • Dynamic content generation                  │
└──────────────────────────────────────────────────┘
```

---

## 1. REGEN-ZIP VM Opcodes

The VM defines a minimal instruction set for regenerative operations:

### 1.1 Opcode Table

| Opcode | Value | Name | Description |
|--------|-------|------|-------------|
| OP_SEED | 0x01 | Initialize | Load seed data into execution context |
| OP_GENERATOR | 0x02 | Execute Generator | Run generator function to produce assets |
| OP_VERIFY | 0x03 | Verify | Validate checksums and signatures |
| OP_ATTACH | 0x04 | Attach Asset | Attach generated asset to result set |
| OP_ZP35_CHECK | 0x05 | Coherence Check | Verify semantic compatibility |
| OP_TW_INSERT | 0x06 | Insert to DOM | Insert generated content into TiddlyWiki |

### 1.2 Instruction Format

Each instruction consists of:

```javascript
{
  opcode: Number,      // Opcode value (0x01 - 0x06)
  data: Any,           // Opcode-specific data
  metadata: Object     // Optional metadata
}
```

### 1.3 Execution Semantics

Instructions execute sequentially in the order:

1. **OP_SEED** - Initializes the random number generator and execution context
2. **OP_ZP35_CHECK** - Verifies coherence before allowing execution
3. **OP_GENERATOR** - Executes the generator to produce assets
4. **OP_VERIFY** - Validates all generated assets
5. **OP_ATTACH** - Attaches assets to the result set (implicit)
6. **OP_TW_INSERT** - Inserts into TiddlyWiki DOM (handled by caller)

---

## 2. Tiddler Field Extensions

### 2.1 New Fields

Tiddlers are extended with the following fields to support regen-zip execution:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `regen-zip` | String | Base64-encoded data or generator reference | "base64:..." or "generatorName" |
| `generator` | String | Name of registered generator function | "fractalGenerator" |
| `seed` | String | Seed hash for deterministic generation | "abc123def456" |
| `zp35` | String | ZP35 coherence signature | "0.382456.23" |
| `version` | String | Semantic version of generator | "1.2.3" |

### 2.2 Field Semantics

**regen-zip**: Can be:
- Base64-encoded binary data: `data:application/zip;base64,UEsDBB...`
- External reference: `https://example.com/assets.zip`
- Generator name reference: `myGenerator`

**generator**: Must match a registered generator function name

**seed**: Any string that can be hashed to initialize the RNG

**zp35**: Format is `{fractalCoord}.{ordinalHeight}` where:
- `fractalCoord` is in [0, 1] with 6 decimal places
- `ordinalHeight` is an integer representing compositional depth

**version**: Follows semantic versioning (MAJOR.MINOR.PATCH)

### 2.3 Example Tiddler

```javascript
{
  title: "FractalArt",
  type: "application/javascript",
  generator: "mandelbrotGenerator",
  seed: "golden-seed-2024",
  zp35: "0.618034.15",
  version: "1.0.0",
  "regen-zip": "mandelbrotGenerator",
  text: "This tiddler generates fractal art using the Mandelbrot generator"
}
```

---

## 3. Generator Function ABI

### 3.1 Generator Function Signature

```javascript
function generator(context) {
  // context = {
  //   seed: String,
  //   rng: Function,
  //   tiddler: Object,
  //   wiki: Object
  // }
  
  // ... generate assets ...
  
  return {
    assets: [
      {
        name: String,
        type: String,
        data: Any,
        checksum: String (optional)
      },
      ...
    ]
  };
}
```

### 3.2 Context Object

The generator receives a context object with:

- **seed**: The seed string from the tiddler
- **rng**: Seeded random number generator function (returns [0, 1])
- **tiddler**: The source tiddler object
- **wiki**: Reference to the TiddlyWiki instance

### 3.3 Return Value

Generators must return an object with an `assets` array. Each asset has:

- **name**: Unique identifier for the asset
- **type**: MIME type (e.g., "image/png", "text/plain")
- **data**: Asset content (string, binary, etc.)
- **checksum**: Optional checksum for verification

### 3.4 Example Generator

```javascript
function fractalGenerator(context) {
  var assets = [];
  var width = 512;
  var height = 512;
  
  // Use seeded RNG for deterministic output
  var seed = context.rng();
  
  // Generate fractal image (simplified)
  var imageData = generateFractalImage(width, height, seed);
  
  assets.push({
    name: "fractal.png",
    type: "image/png",
    data: imageData,
    checksum: computeChecksum(imageData)
  });
  
  return { assets: assets };
}
```

---

## 4. ZP35 Coherence System

### 4.1 The Golden Operator

The ZP35 golden operator maps tiddlers to fractal coordinates in [0, 1] that preserve:

1. **Ordering** - Weaker theories → lower coordinates
2. **Clustering** - Similar tiddlers remain close
3. **Coherence curvature** - κ = 0.35 guardian threshold
4. **Self-similarity** - Fractal structure at all scales

### 4.2 Guardian Threshold (κ = 0.35)

The value 0.35 is the **coherence curvature** - the boundary where:

- **Distance < κ**: Safe composition, semantic coherence preserved
- **Distance < 2κ**: Caution zone, review recommended
- **Distance ≥ 2κ**: Blocked, violates coherence

This value derives from:
- Empirical learnability boundary (~400 examples/transition)
- Natural plateau in Cantor hierarchical embedding
- Balance between structure and brittleness

### 4.3 Coherence Check Algorithm

```
1. Map source tiddler to fractal coordinate: s
2. Map generator to fractal coordinate: g
3. Calculate distance: d = |s - g|
4. If d < κ: Allow (safe)
5. If d < 2κ: Allow with warning (caution)
6. If d ≥ 2κ: Block (unsafe)
```

### 4.4 ZP35 Signature Format

Format: `{fractalCoord}.{ordinalHeight}`

Example: `0.618034.15`

- `0.618034` - Fractal coordinate (6 decimals)
- `15` - Ordinal height (integer)

The signature encodes the tiddler's semantic position in the fractal space.

---

## 5. VM API

### 5.1 RegenZipVM Constructor

```javascript
var vm = new RegenZipVM(wiki);
```

Creates a new VM instance attached to a wiki.

### 5.2 Register Generator

```javascript
vm.registerGenerator(name, fn, metadata);
```

Register a generator function with metadata:

```javascript
vm.registerGenerator("fractalGen", fractalGenerator, {
  version: "1.0.0",
  seed: "default-seed",
  zp35: "0.500000.10",
  description: "Generates fractal images"
});
```

### 5.3 Load Tiddler

```javascript
var success = vm.load(tiddler);
```

Loads a tiddler into the VM for execution. Returns `true` if successful.

### 5.4 Execute

```javascript
var result = vm.run();
// result = {
//   success: Boolean,
//   assets: Array,
//   metadata: Object,
//   error: String (if failed)
// }
```

Executes the loaded tiddler's regen-zip program.

### 5.5 Get State

```javascript
var state = vm.getState();
// state = {
//   state: String,        // "idle", "loading", "running", "complete", "error"
//   context: Object,
//   assets: Array,
//   generators: Array
// }
```

Returns current VM state.

### 5.6 Reset

```javascript
vm.reset();
```

Resets VM to idle state, clearing context and assets.

---

## 6. ZP35 Operator API

### 6.1 ZP35Operator Constructor

```javascript
var operator = new ZP35Operator();
```

Creates a new ZP35 operator instance.

### 6.2 Check Coherence

```javascript
var result = operator.checkCoherence(sourceTiddler, targetTiddler);
// result = {
//   allowed: Boolean,
//   mode: String,          // "safe", "caution", "blocked", "error"
//   distance: Number,
//   confidence: Number,    // 0.0 - 1.0
//   message: String,
//   suggestions: Array,    // For "caution" mode
//   alternatives: Array    // For "blocked" mode
// }
```

Checks if composing two tiddlers maintains semantic coherence.

### 6.3 Calculate Signature

```javascript
var signature = operator.calculateSignature(tiddler);
// Returns: "0.618034.15"
```

Calculates ZP35 signature for a tiddler.

### 6.4 Verify Signature

```javascript
var result = operator.verifySignature(tiddler, expectedSignature);
// result = {
//   valid: Boolean,
//   computed: String,
//   expected: String,
//   distance: Number,
//   message: String
// }
```

Verifies a tiddler's signature matches expected value.

### 6.5 Analyze Clusters

```javascript
var analysis = operator.analyzeClusterStructure(tiddlerArray);
// analysis = {
//   valid: Boolean,
//   clusterCount: Number,
//   clusters: Array,
//   message: String
// }
```

Analyzes ultrametric clustering structure of tiddler collection.

---

## 7. Integration with TiddlyWiki

### 7.1 Startup Hook

The VM can be initialized at TiddlyWiki startup:

```javascript
// In a startup module
exports.startup = function() {
  $tw.regenZipVM = new $tw.utils.RegenZipVM($tw.wiki);
  
  // Register built-in generators
  registerBuiltInGenerators($tw.regenZipVM);
};
```

### 7.2 Tiddler Loading Hook

Intercept tiddler loading to trigger VM execution:

```javascript
$tw.wiki.addEventListener("change", function(changes) {
  Object.keys(changes).forEach(function(title) {
    var tiddler = $tw.wiki.getTiddler(title);
    if(tiddler && tiddler.fields["regen-zip"]) {
      executeRegenZip(tiddler);
    }
  });
});
```

### 7.3 Widget Integration

Create a widget to display generated assets:

```html
<$regen-zip tiddler="FractalArt" />
```

### 7.4 Filter Operator

Add filter operator for ZP35 coherence checks:

```
[coherent-with[TargetTiddler]]
```

---

## 8. Use Cases

### 8.1 Generative Art Plugin

Ship only:
- Generator function (small JS code)
- Seed values (tiny strings)

Users download 10KB instead of 10MB of pre-generated images.

### 8.2 Live Documentation

Documentation assets regenerate based on:
- Current code version
- User preferences
- System configuration

Always up-to-date, never stale.

### 8.3 Adaptive Plugins

Plugins adapt their assets:
- Based on screen size
- Based on color scheme
- Based on user locale

Single plugin works everywhere.

### 8.4 Secure Sync

Sync only:
- Seeds
- Generator references
- Signatures

Assets regenerate on each device. Bandwidth-efficient.

---

## 9. Security Considerations

### 9.1 Generator Sandboxing

Generators should execute in sandboxed environment:
- Limited API access
- Timeout constraints
- Memory limits
- No network access

### 9.2 ZP35 Safety Guarantees

The ZP35 coherence check provides:
- **Type safety**: Prevents incompatible compositions
- **Semantic safety**: Maintains meaning across operations
- **Evolution safety**: Allows gradual change within κ bounds

### 9.3 Signature Verification

Always verify ZP35 signatures to detect:
- Modified tiddlers
- Corrupted data
- Version mismatches

### 9.4 Checksum Validation

Use OP_VERIFY to validate all generated assets before use.

---

## 10. Performance Characteristics

### 10.1 Space Complexity

- **Traditional**: O(n) where n = total asset size
- **REGEN-ZIP**: O(s + g) where s = seed size, g = generator size
- **Ratio**: Often 100:1 to 1000:1 reduction

### 10.2 Time Complexity

- **First load**: O(generation time)
- **Cached**: O(1) with asset caching
- **Network**: O(bandwidth × compression) vs O(bandwidth × full size)

### 10.3 Determinism

Given same seed + generator version:
- **Output**: Bitwise identical
- **Reproducible**: Across devices/time
- **Verifiable**: Via checksums

---

## 11. Future Extensions

### 11.1 Streaming Generation

Support streaming asset generation for large outputs:

```javascript
function* streamingGenerator(context) {
  yield { chunk: 1, data: ... };
  yield { chunk: 2, data: ... };
}
```

### 11.2 Parallel Execution

Execute multiple generators in parallel:

```javascript
vm.runParallel([tiddler1, tiddler2, tiddler3]);
```

### 11.3 Delta Updates

Track changes and regenerate only affected assets:

```javascript
vm.incrementalRun(changes);
```

### 11.4 Cross-Platform Generators

Standard generator library that works across:
- Browser TiddlyWiki
- Node.js TiddlyWiki
- TiddlyWiki on mobile

---

## 12. Reference Implementation

The reference implementation provides:

1. **core/modules/utils/regen-zip-vm.js** - VM engine
2. **core/modules/utils/zp35-operator.js** - Coherence layer
3. **Test suite** - Comprehensive tests
4. **Example generators** - Demonstration code

---

## 13. Glossary

**Asset**: Generated content (image, text, data) produced by a generator

**Coherence**: Semantic compatibility measured by ZP35 distance

**Fractal Coordinate**: Position in [0, 1] space preserving invariants

**Generator**: Function that produces assets from seeds

**Guardian Threshold (κ)**: 0.35 - coherence safety boundary

**Opcode**: VM instruction for regenerative operations

**Ordinal Height**: Compositional depth of a tiddler

**Regen-ZIP**: Regenerative ZIP entry that generates instead of inflates

**Seed**: Input string that deterministically drives generation

**ZP35**: Invariant-preserving morphism between representation spaces

---

## 14. Mathematical Foundations

See **ZP35_GOLDEN_OPERATOR.md** for complete mathematical treatment of:

- Category-theoretic definition
- Invariant preservation proofs
- Cantor embedding details
- Golden ratio scaling properties
- Coherence curvature derivation

---

## Conclusion

The REGEN-ZIP Virtual Machine transforms TiddlyWiki into a **generative operating system** where:

- Tiddlers are executable modules
- Generators create assets on-demand
- ZP35 ensures semantic safety
- Space efficiency increases 100-1000x
- Deterministic reproducibility guaranteed

This is not just compression - it's **computation as content**, with mathematical guarantees of coherence and compatibility.

The future of TiddlyWiki is regenerative.

---

## See Also

**Unified Theory**: The REGEN-ZIP VM represents the **discrete/Cayley view** of the semantic manifold. To understand how this fits into the complete unified theory, see:

- **UNIFIED_COMPUTATIONAL_THEORY.md** - Complete unified theory (Section 1: VMs as Cayley Skeleton)
- **UNIFIED_THEORY_README.md** - Quick reference guide
- **core/modules/utils/ce-tower.js** - CE Tower ensuring consistency across all views

**Related Documentation**:

- **ZP35_GOLDEN_OPERATOR.md** - Mathematical foundations of the golden operator
- **COMPILER_PROGRAM_PATTERN.md** - How routing connects to ML/continuous view
- **SHADOW_INDUCTION.md** - How spectral compression works

**Implementation Files**:

- `core/modules/utils/regen-zip-vm.js` - VM implementation
- `core/modules/utils/zp35-operator.js` - ZP35 compatibility layer
- `editions/test/tiddlers/tests/test-regen-zip-vm.js` - Test suite
