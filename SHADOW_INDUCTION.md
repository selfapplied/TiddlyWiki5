# Shadow Induction: Self-Hosted Semantic Compilers

**Version:** 1.0  
**Date:** December 8, 2024  
**Status:** Implementation Complete

---

## Executive Summary

Shadow Induction is a breakthrough feature that allows **any tiddler to generate its own shadow compiler** through structural extraction rather than semantic clustering. This completes the compiler-program pattern by enabling tiddlers to become self-hosted programs with their own personal interpreters.

### The Core Innovation

When a tiddler has no suitable compiler in the semantic neighborhood, instead of failing or forcing clustering, the system can now:

1. **Extract the crisp core** (stable patterns, field schema, repeatable structures)
2. **Generate a shadow compiler** that defines "what this tiddler means"
3. **Mark the original as a program** written in its own induced language

This is analogous to how natural languages contain the grammar that explains themselves - every dialect carries its own interpretation rules.

---

## Conceptual Foundation

### How It Works

Every tiddler contains:

* **Crisp region** - Regularities, patterns, stable fields (high coherence)
* **Chaotic region** - Idiosyncrasies, deltas, author-specific content (low coherence)
* **Curvature field** - Semantic flexibility boundary (how much deviation before losing identity)

Shadow induction performs the following transformation:

```
Tiddler = Crisp Core + Chaotic Content
           ↓
Shadow Compiler (extracted core) + Self-Hosted Program (original with reference)
```

### The Result

After induction:

* **Shadow compiler** defines the semantic kernel for this tiddler's "language"
* **Original tiddler** becomes a program written in its own induced dialect
* **Routing** automatically connects the program to its personal compiler
* **Composition** preserves coherence through ZP35 geometric anchoring

---

## Architecture

### Module Structure

```
┌─────────────────────────────────────────────────┐
│          Compiler-Program Router                 │
│   • Classification (compiler/program)            │
│   • Routing with ZP35 distance                   │
│   • Shadow induction integration                 │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│           Shadow Inducer                         │
│   • Internal coherence analysis                  │
│   • Crisp/chaotic separation                     │
│   • Pattern extraction                           │
│   • Shadow compiler generation                   │
│   • Self-hosted program marking                  │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│            ZP35 Operator                         │
│   • Fractal coordinate mapping                   │
│   • Coherence curvature (κ = 0.35)              │
│   • Semantic distance calculation                │
└──────────────────────────────────────────────────┘
```

### Component Details

#### 1. Shadow Inducer (`induce-shadow.js`)

Core module that performs shadow induction:

```javascript
var ShadowInducer = require("$:/core/modules/utils/induce-shadow.js").ShadowInducer;
var inducer = new ShadowInducer(wiki, zp35Operator);

var result = inducer.induceShadow(tiddler);
// Returns: { 
//   success: true,
//   shadowCompiler: {...},
//   selfHostedProgram: {...},
//   coherenceAnalysis: {...}
// }
```

**Key Methods:**

* `induceShadow(tiddler)` - Perform complete shadow induction
* `analyzeInternalCoherence(tiddler)` - Separate crisp from chaotic fields
* `extractCrispStructure(tiddler, analysis)` - Extract stable patterns
* `generateShadowCompiler(...)` - Create shadow compiler tiddler
* `needsShadowInduction(tiddler)` - Check if tiddler qualifies

#### 2. Router Integration

Shadow induction is integrated into the compiler-program router:

```javascript
// Automatic shadow induction when routing
var routing = router.route(programTiddler, { 
  allowShadowInduction: true  // Default
});

// Manual shadow induction
var result = router.induceShadow(tiddler);
```

**Routing Modes:**

* `safe` - Program within κ distance of compiler
* `caution` - Program crosses semantic boundary (< 2κ)
* `borderline` - Program at edge of domain (< OOD threshold)
* `ood` - Out-of-distribution (triggers shadow induction)
* `induced` - Shadow compiler was auto-generated

---

## Implementation Details

### Coherence Analysis

The system analyzes each field to determine if it's crisp or chaotic:

**Crisp Fields (coherence > 0.65):**
* Structural fields: `title`, `type`, `generator`, `version`, `seed`, `tags`
* Typed content
* Versioned fields
* Short, stable strings

**Chaotic Fields (coherence < 0.35):**
* High-entropy content
* Long, variable text
* Custom, unstructured data

**Intermediate Fields (0.35 - 0.65):**
* Classified based on context
* Structural fields default to crisp
* Custom fields default to chaotic

### Pattern Extraction

The inducer extracts structural patterns from tiddler content:

* **Markdown patterns:** headings, bold, italic, links
* **Transclusion patterns:** `{{...}}` references
* **Link patterns:** `[[...]]` wiki links
* **Code patterns:** inline and block code

These patterns become part of the shadow compiler's schema description.

### Curvature Coefficient

The curvature coefficient defines semantic flexibility:

```
curvature = 1.0 - (crispFields.length / totalFields.length)
```

* More crisp fields → lower curvature (rigid structure)
* More chaotic fields → higher curvature (flexible structure)
* Clamped to safe range: [κ/2, 2κ] = [0.175, 0.70]

### Shadow Compiler Generation

Generated shadow compilers have the following structure:

```javascript
{
  fields: {
    title: "${originalTitle}-shadow",
    type: "application/x-tiddler-shadow-compiler",
    generator: "shadow-compiler",
    version: "1.0.0",
    zp35: "${signature}",
    seed: "shadow-${hash}",
    "shadow-source": "${originalTitle}",
    "shadow-type": "induced",
    tags: ["$:/tags/ShadowCompiler"],
    text: "... generated documentation ..."
  }
}
```

### Self-Hosted Program Marking

Original tiddlers are marked as self-hosted programs:

```javascript
{
  fields: {
    // ... original fields ...
    compiler: "${originalTitle}-shadow",
    "program-mode": "self-hosted",
    "shadow-compiler": "${originalTitle}-shadow",
    tags: [...originalTags, "$:/tags/SelfHostedProgram"]
  }
}
```

---

## Usage Examples

### Example 1: Basic Shadow Induction

```javascript
var tiddler = {
  fields: {
    title: "MyNote",
    text: "# My Important Note\n\nSome **bold** content",
    tags: ["notes", "important"]
  }
};

var result = router.induceShadow(tiddler);

// Creates:
// 1. Shadow compiler: "MyNote-shadow"
// 2. Self-hosted program: "MyNote" (with compiler reference)
```

### Example 2: Automatic Routing with Induction

```javascript
// No compilers exist - triggers shadow induction
var routing = router.route(programTiddler);

console.log(routing.mode);              // "induced"
console.log(routing.shadowInduction);   // true
console.log(routing.compilerTitle);     // "programTiddler-shadow"
```

### Example 3: OOD Triggers Induction

```javascript
// Register a very specific compiler
router.registerCompiler(specificCompiler);

// Create a very different program
var differentProgram = { 
  fields: { title: "Different", text: "..." } 
};

// Distance is OOD (> 0.70) - triggers shadow induction
var routing = router.route(differentProgram);

console.log(routing.mode);  // "induced" instead of "ood"
```

### Example 4: Disable Shadow Induction

```javascript
// Disable shadow induction for traditional behavior
var routing = router.route(program, { 
  allowShadowInduction: false 
});

// Returns failure if no suitable compiler exists
console.log(routing.success);  // false
console.log(routing.message);  // "No compilers registered"
```

---

## Benefits and Impact

### What This Enables

1. **Self-Sufficiency**
   - Every tiddler can generate its own interpreter
   - No dependency on external semantic neighbors
   - Graceful degradation when clustering fails

2. **Personal Dialects**
   - Each tiddler defines its own language
   - Custom semantic kernels for specialized content
   - Preservation of unique structural patterns

3. **Dynamic Evolution**
   - Tiddlers can evolve their compilers over time
   - Shadow compilers capture structural history
   - Clean migration paths for refactoring

4. **Compositional Safety**
   - ZP35 coherence still enforced
   - Curvature bounds prevent semantic drift
   - Guardian threshold (κ = 0.35) maintains guarantees

5. **Compression Efficiency**
   - Redundant structure moves to shadow compiler
   - Tiddler becomes seed + parameters + deviations
   - Similar to ZIP/MPEG codec architecture

### Analogies to Other Systems

This pattern mirrors several well-known architectures:

| System | Shadow Compiler | Self-Hosted Program |
|--------|-----------------|---------------------|
| **ML Training** | Trained model weights | Inference prompt |
| **Compression** | Codec/dictionary | Bitstream/residuals |
| **Programming** | Compiler/interpreter | Source code |
| **Biology** | Genetic template | Individual organism |
| **Language** | Grammar rules | Spoken utterance |

---

## Testing

### Test Coverage

28 comprehensive tests cover all aspects of shadow induction:

**Shadow Inducer Tests (21 specs):**
* Constructor and configuration
* Internal coherence analysis
* Pattern extraction
* Shadow compiler generation
* Self-hosted program marking
* Full shadow induction
* Induction requirements
* Seed generation

**Router Integration Tests (7 specs):**
* Shadow induction when no compilers exist
* Shadow induction for OOD programs
* Disabling shadow induction
* Direct induction API
* Caching induced shadows
* Registering induced shadows as compilers

### Running Tests

```bash
npm test
```

All 1527 tests pass, including the new shadow induction tests.

---

## API Reference

### ShadowInducer Class

```javascript
var ShadowInducer = require("$:/core/modules/utils/induce-shadow.js").ShadowInducer;
var inducer = new ShadowInducer(wiki, zp35Operator);
```

#### Methods

**`induceShadow(tiddler)`**

Perform complete shadow induction on a tiddler.

* **Parameters:**
  * `tiddler` - Tiddler object to induce shadow from
* **Returns:** Object with:
  * `success` - Boolean indicating success
  * `shadowCompiler` - Generated shadow compiler tiddler
  * `selfHostedProgram` - Modified original tiddler
  * `coherenceAnalysis` - Crisp/chaotic separation details
  * `crispStructure` - Extracted structural patterns
  * `signature` - ZP35 geometric signature
  * `message` - Status message

**`needsShadowInduction(tiddler)`**

Check if tiddler qualifies for shadow induction.

* **Parameters:**
  * `tiddler` - Tiddler object to check
* **Returns:** Boolean - true if induction is needed and possible

**`analyzeInternalCoherence(tiddler)`**

Analyze internal coherence and separate crisp from chaotic fields.

* **Parameters:**
  * `tiddler` - Tiddler object to analyze
* **Returns:** Object with:
  * `crispFields` - Array of high-coherence fields
  * `chaoticFields` - Array of low-coherence fields
  * `patterns` - Extracted structural patterns
  * `curvatureCoefficient` - Semantic flexibility measure

### Router Integration

**`route(programTiddler, options)`**

Route a program tiddler to a compiler (with optional shadow induction).

* **Parameters:**
  * `programTiddler` - Program tiddler to route
  * `options` - Optional routing options:
    * `allowShadowInduction` - Boolean (default: true)
* **Returns:** Routing result object

**`induceShadow(tiddler)`**

Directly induce shadow compiler for a tiddler.

* **Parameters:**
  * `tiddler` - Tiddler to induce shadow for
* **Returns:** Shadow induction result

---

## Design Decisions

### Why Induction Instead of Clustering?

1. **Determinism** - Induction is deterministic; clustering depends on dataset
2. **Independence** - No dependency on semantic neighbors
3. **Precision** - Extracts actual tiddler structure, not averaged patterns
4. **Efficiency** - No search or distance calculations needed

### Why Extract Crisp Core?

The crisp core represents:
* Stable, repeatable patterns
* Structural schema
* Type and versioning information
* Categorization (tags)

This forms a natural "compiler" that interprets the chaotic content as "program input."

### Why Maintain ZP35 Coherence?

Even with shadow induction:
* Composition safety must be preserved
* Curvature bounds prevent semantic drift
* Guardian threshold (κ = 0.35) maintains guarantees
* Fractal geometry provides stable anchoring

### Why Self-Hosted?

Self-hosting enables:
* Evolution - compiler and program co-evolve
* Compression - structure extracted to compiler
* Composition - personal dialects still compose safely
* Flexibility - each tiddler has maximum freedom

---

## Future Enhancements

### Potential Extensions

1. **Shadow Evolution**
   - Track shadow compiler versions
   - Migrate programs when shadows evolve
   - Detect structural drift over time

2. **Shadow Clustering**
   - Group similar shadow compilers
   - Merge compatible shadows
   - Discover emergent patterns

3. **Adaptive Curvature**
   - Learn optimal curvature from usage
   - Adjust flexibility based on composition patterns
   - Balance rigidity vs. adaptability

4. **Shadow Inheritance**
   - Create shadow hierarchies
   - Inherit patterns from parent shadows
   - Enable semantic subtyping

5. **Cross-Wiki Shadows**
   - Share shadow compilers across wikis
   - Create shadow libraries
   - Enable semantic interoperability

---

## Related Documentation

* [Compiler-Program Pattern](COMPILER_PROGRAM_PATTERN.md) - Overall pattern architecture
* [REGEN-ZIP VM](REGEN_ZIP_VM.md) - Virtual machine specification
* [ZP35 Golden Operator](ZP35_GOLDEN_OPERATOR.md) - Coherence foundation
* [ZP35 TiddlyWiki Enhancements](ZP35_TIDDLYWIKI_ENHANCEMENTS.md) - Integration guide

---

## Conclusion

Shadow Induction transforms TiddlyWiki into a **self-compiling semantic organism** where every tiddler can:

* Generate its own interpreter
* Define its own dialect
* Evolve its own semantic kernel
* Compose safely with others
* Compress efficiently through structure extraction

This completes the compiler-program pattern and enables true semantic autonomy for tiddlers while maintaining the safety guarantees of ZP35 coherence geometry.

The system now mirrors how natural languages work: every utterance contains both the grammar that explains it and the content that uses that grammar. TiddlyWiki tiddlers have become living semantic entities with their own built-in interpretation rules.

---

## See Also

**Unified Theory**: Shadow Induction represents the **spectral/compression view** of the semantic manifold (eigenstructure extraction and harmonic analysis). To understand how this fits into the complete unified theory, see:

- **UNIFIED_COMPUTATIONAL_THEORY.md** - Complete unified theory (Section 3: Compression as Spectral Signature)
- **UNIFIED_THEORY_README.md** - Quick reference guide
- **core/modules/utils/ce-tower.js** - CE Tower ensuring spectral invariance (CE3)

**Related Documentation**:

* [Compiler-Program Pattern](COMPILER_PROGRAM_PATTERN.md) - How routing uses induced compilers
* [REGEN-ZIP VM](REGEN_ZIP_VM.md) - How generators act as spectral modes
* [ZP35 Golden Operator](ZP35_GOLDEN_OPERATOR.md) - Mathematical foundations
* [ZP35 TiddlyWiki Enhancements](ZP35_TIDDLYWIKI_ENHANCEMENTS.md) - Integration guide

**Implementation Files**:

* `core/modules/utils/induce-shadow.js` - Shadow induction implementation
* `core/modules/utils/zp35-operator.js` - Coherence analysis
* `editions/test/tiddlers/tests/test-shadow-induction.js` - Test suite
