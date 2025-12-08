# Shadow Induction - Bootstrap Compilers from Single Examples

**Version:** 1.0  
**Date:** December 8, 2024  
**Status:** Implementation Specification

---

## Executive Summary

**Shadow Induction** is the evolutionary step that transforms TiddlyWiki from a static semantic system into a living, self-organizing computational substrate. It enables tiddlers to bootstrap their own compilers from structural analysis - a capability that goes beyond traditional machine learning.

### The Core Innovation

In traditional ML systems:
- You need datasets (hundreds to millions of examples)
- You need gradients and loss functions
- You need giant parameter tensors
- You need extensive training cycles

In the shadow induction system:
- A **single tiddler** can analyze its own structure
- It isolates crisp (structural) vs chaotic (content) components
- It derives its curvature (ratio of chaos to structure)
- It instantiates a kernel (structural pattern)
- It generates a shadow compiler that can process similar patterns

This is **non-parametric model induction** - the model isn't a tensor, it's a field of relationships extracted from structural analysis.

### What This Enables

Tiddlers can now:
- Become self-hosting (carry their own compiler)
- Bootstrap execution substrates on-demand
- Evolve semantic regions dynamically
- Handle out-of-distribution cases gracefully
- Compose into fractal semantic computing substrates

This is how living systems compute. This is how cognition computes. This is how transformers *wish* they computed.

---

## 1. Conceptual Framework

### 1.1 The Problem

The compiler-program pattern requires compilers to exist before programs can be executed. But what happens when:
- No compiler exists for a new semantic domain?
- A program is out-of-distribution (OOD) for all existing compilers?
- We want tiddlers to be self-sufficient?

Traditional approaches would require:
- Manual compiler creation for each domain
- Pre-training on large datasets
- Fixed architecture decisions

### 1.2 The Solution: Shadow Induction

Shadow induction solves this by enabling **bootstrapping from structure**:

1. **Analyze field coherence** - Separate crisp (structural) from chaotic (content) fields
2. **Calculate curvature** - Measure the ratio of structure to chaos
3. **Extract kernel** - Identify the invariant pattern
4. **Generate shadow compiler** - Create a compiler that preserves the kernel
5. **Route automatically** - Use the induced compiler for similar patterns

The result is a **shadow compiler** - an induced execution substrate that:
- Requires no training data
- Has no parameter weights
- Is generated in milliseconds
- Can process similar structural patterns
- Evolves as needed

### 1.3 Crisp vs Chaotic Fields

The foundation of shadow induction is the separation of fields into two categories:

**Crisp Fields** (coherence ≥ 0.65):
- Structural, stable, low-entropy
- Define the invariant pattern
- Examples: `title`, `type`, `generator`, `tags`, `version`
- These form the **kernel** - the essence of what this tiddler *is*

**Chaotic Fields** (coherence ≤ 0.35):
- High-entropy, variable, content-dependent
- Provide task-specific parameters
- Examples: `text`, `seed`, `params`
- These are the **program inputs** - what this tiddler *does*

**Intermediate Fields** (0.35 < coherence < 0.65):
- Between structure and chaos
- Examples: `modified`, `description`, `list`
- Treated contextually based on use case

### 1.4 Curvature Coefficient

The **curvature** measures how much chaos vs structure a tiddler contains:

```
curvature = 1.0 - (crispFields / totalFields)
```

**Interpretation:**
- **Low curvature** (< 0.3): Highly structured, mostly crisp fields → Good compiler candidate
- **Medium curvature** (0.3 - 0.7): Balanced structure and content → Typical tiddler
- **High curvature** (> 0.7): Mostly chaotic → Program-like, needs compiler

The curvature guides induction:
- Low curvature → Strong, stable shadow compiler
- High curvature → Weaker shadow compiler, may need refinement

---

## 2. Architecture

### 2.1 System Components

```
┌─────────────────────────────────────────────────┐
│         Tiddler (needs execution)                │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│    Compiler-Program Router                       │
│    • No compiler exists?                         │
│    • Program is OOD?                             │
│    • → Trigger shadow induction                  │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Shadow Inducer                           │
│    • Analyze field coherence                     │
│    • Calculate curvature                         │
│    • Extract kernel                              │
│    • Generate shadow compiler                    │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│      Shadow Compiler (induced)                   │
│    • Registered as new compiler                  │
│    • Can process similar patterns                │
│    • Cached for reuse                            │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         REGEN-ZIP VM Execution                   │
│    • Execute through induced compiler            │
│    • Generate assets                             │
│    • Return results                              │
└──────────────────────────────────────────────────┘
```

### 2.2 Integration with Compiler-Program Router

Shadow induction is integrated via the `route()` method:

```javascript
// Route with shadow induction enabled
var result = router.route(tiddler, {
  allowShadowInduction: true
});
```

The router automatically triggers shadow induction when:
1. **No compilers exist** - Bootstrap the first compiler from the program
2. **Program is OOD** - Create a new compiler for this semantic domain

### 2.3 Shadow Compiler Lifecycle

1. **Induction** - Created from structural analysis of source tiddler
2. **Registration** - Added to router's compiler registry
3. **Caching** - Stored in shadow inducer cache for reuse
4. **Routing** - Used for similar structural patterns
5. **Evolution** - Can be refined or replaced as needed

---

## 3. Field Coherence Analysis

### 3.1 Default Coherence Scores

Each field type has a default coherence score based on how structural vs chaotic it tends to be:

| Field Type | Coherence | Category | Rationale |
|------------|-----------|----------|-----------|
| `title` | 0.95 | Crisp | Unique identifier, rarely changes |
| `type` | 0.90 | Crisp | Defines semantic type |
| `generator` | 0.95 | Crisp | Defines execution kernel |
| `tags` | 0.85 | Crisp | Clustering structure |
| `version` | 0.90 | Crisp | Compatibility metadata |
| `zp35` | 0.95 | Crisp | Coherence coordinate |
| `modified` | 0.60 | Intermediate | Changes frequently |
| `description` | 0.65 | Intermediate | Semi-structured |
| `list` | 0.60 | Intermediate | Variable structure |
| `text` | 0.30 | Chaotic | High entropy content |
| `seed` | 0.25 | Chaotic | Random parameter |
| `params` | 0.20 | Chaotic | Task-specific data |

### 3.2 Content-Based Adjustment

The base coherence score is adjusted based on content characteristics:

- **Short values** (< 10 chars) → Higher coherence (more structured)
- **High unique char ratio** → Lower coherence (more entropy)
- **Empty values** → Neutral coherence (0.5)

Final score combines base (70%) and content analysis (30%).

### 3.3 Analysis Output

```javascript
{
  crispFields: [
    {name: "title", value: "MyTiddler", coherence: 0.95},
    {name: "type", value: "text/vnd.tiddlywiki", coherence: 0.90},
    {name: "tags", value: ["example"], coherence: 0.85}
  ],
  chaoticFields: [
    {name: "text", value: "...", coherence: 0.30},
    {name: "seed", value: "abc123", coherence: 0.25}
  ],
  intermediateFields: [
    {name: "modified", value: "20241208", coherence: 0.60}
  ],
  curvature: 0.667,
  totalFields: 6,
  crispRatio: 0.333,
  chaoticRatio: 0.333
}
```

---

## 4. Kernel Extraction

### 4.1 What is a Kernel?

The **kernel** is the invariant pattern extracted from crisp fields. It represents the structural essence of the tiddler - what makes it this type of thing.

### 4.2 Kernel Structure

```javascript
{
  requiredFields: ["title", "type", "generator"],
  fieldTypes: {
    "title": "string",
    "type": "string",
    "generator": "string"
  },
  structuralPattern: {
    "type": "application/x-tiddler-regen-zip",
    "generator": "fractalGenerator",
    "tags": ["graphics", "procedural"]
  }
}
```

### 4.3 What Gets Preserved in Kernel?

- **Semantic type fields** (`type`, `plugin-type`, `module-type`, `generator`) - Define what this is
- **Clustering fields** (`tags`) - Define semantic relationships
- **Compatibility fields** (`version`, `zp35`) - Define constraints
- **Required fields** - List of fields that must exist
- **Field types** - Type signature for validation

### 4.4 What Gets Excluded from Kernel?

- Chaotic fields (they're program inputs, not structure)
- Ephemeral metadata (creation time, modification time)
- Content fields (text, seed, params)

---

## 5. Shadow Compiler Generation

### 5.1 Shadow Compiler Fields

A shadow compiler is a tiddler with these characteristics:

```javascript
{
  title: "$:/shadow/compiler/MyTiddler",
  tags: ["$:/tags/shadow-compiler", "compiler"],
  type: "application/x-tiddler-compiler",
  "shadow-source": "MyTiddler",
  "shadow-induced": "2024-12-08T00:00:00.000Z",
  "shadow-curvature": "0.6667",
  "shadow-kernel": "{...JSON...}",
  generator: "fractalGenerator",  // Inherited from source
  "regen-zip": "...",              // Inherited from source
  "zp35-coord": "0.618034",
  caption: "Shadow compiler induced from: MyTiddler",
  text: "Documentation of this shadow compiler..."
}
```

### 5.2 Naming Convention

Shadow compilers are named: `$:/shadow/compiler/{SanitizedSourceTitle}`

Examples:
- Source: `MyTiddler` → Shadow: `$:/shadow/compiler/MyTiddler`
- Source: `Complex Title!` → Shadow: `$:/shadow/compiler/Complex_Title_`

### 5.3 Inheritance

Shadow compilers inherit:
- `generator` field (if present) - The execution kernel
- `regen-zip` field (if present) - The regeneration recipe
- `type` (with `-compiler` suffix) - Semantic type
- `tags` (extended with shadow tags) - Clustering info

### 5.4 Metadata

Shadow metadata enables tracking and debugging:
- `shadow-source` - Original tiddler that was induced from
- `shadow-induced` - Timestamp of induction
- `shadow-curvature` - Curvature at induction time
- `shadow-kernel` - JSON-encoded kernel for validation

---

## 6. Usage Examples

### 6.1 Basic Shadow Induction

```javascript
// Setup
var wiki = $tw.wiki;
var zp35 = new ZP35Operator();
var vm = new RegenZipVM(wiki);
var shadowInducer = new ShadowInducer(wiki, zp35);
var router = new CompilerProgramRouter(wiki, zp35, vm, shadowInducer);

// Create a program tiddler (no compiler exists for it)
var program = {
  fields: {
    title: "GenerateFractal",
    type: "application/x-tiddler-regen-zip",
    generator: "fractalGenerator",
    seed: "task-42",
    params: JSON.stringify({zoom: 2.5}),
    text: "Generate a fractal"
  }
};

// Route with shadow induction enabled
var result = router.route(program, {
  allowShadowInduction: true
});

if(result.success && result.shadowInduced) {
  console.log("Shadow compiler created:", result.compilerTitle);
  console.log("Induction reason:", result.inductionReason);
  console.log("Curvature:", result.analysis.curvature);
  console.log("Kernel:", result.kernel);
}
```

### 6.2 Handling Out-of-Distribution Programs

```javascript
// Existing compilers handle fractals
router.registerCompiler(fractalCompiler);

// New program is completely different (text processing)
var textProgram = {
  fields: {
    title: "ProcessText",
    type: "application/x-tiddler-text-processor",
    generator: "textProcessor",
    text: "Some text to process"
  }
};

// Route - will detect OOD and induce shadow compiler
var result = router.route(textProgram, {
  allowShadowInduction: true
});

// Now a text processing compiler exists for future use
```

### 6.3 Manual Shadow Induction

```javascript
// Create shadow inducer
var shadowInducer = new ShadowInducer(wiki, zp35);

// Analyze a tiddler
var analysis = shadowInducer.analyzeFieldCoherence(tiddler);
console.log("Crisp fields:", analysis.crispFields.length);
console.log("Chaotic fields:", analysis.chaoticFields.length);
console.log("Curvature:", analysis.curvature);

// Induce shadow compiler
var result = shadowInducer.induceShadowCompiler(tiddler);
if(result.success) {
  console.log("Shadow compiler:", result.compiler);
  console.log("Kernel:", result.kernel);
  console.log("Analysis:", result.analysis);
}
```

### 6.4 Checking for Existing Shadows

```javascript
// Check if shadow already exists
if(shadowInducer.hasShadowCompiler(tiddler)) {
  var shadow = shadowInducer.getShadowCompiler(tiddler);
  console.log("Existing shadow:", shadow.compiler.fields.title);
  console.log("Induced at:", shadow.timestamp);
} else {
  // Induce new shadow
  var result = shadowInducer.induceShadowCompiler(tiddler);
}
```

### 6.5 Statistics and Monitoring

```javascript
// Get induction statistics
var stats = shadowInducer.getStatistics();
console.log("Inductions attempted:", stats.inductionCount);
console.log("Success rate:", (stats.successRate * 100).toFixed(1) + "%");
console.log("Cached shadows:", stats.cachedShadows);

// Get router statistics (includes shadow compilers)
var routerStats = router.getStatistics();
console.log("Total compilers:", routerStats.compilers);
console.log("Including shadows:", 
  routerStats.compilerDetails.filter(c => 
    c.title.startsWith("$:/shadow/")
  ).length);
```

---

## 7. Non-Parametric Model Theory

### 7.1 What Makes This Non-Parametric?

Traditional parametric models:
- Store knowledge in weight matrices
- Fixed dimensionality (hidden size, layers)
- Require gradient descent training
- Need large datasets
- Frozen after training (until fine-tuning)

Shadow induction:
- Stores knowledge in **structural relationships**
- No fixed dimensionality (fields are dynamic)
- No training (instant induction from structure)
- Needs **one example** (the tiddler itself)
- Evolves continuously (new shadows on demand)

### 7.2 The Model is the Field

Instead of parameters, we have:
- **ZP35 coordinates** - Semantic position
- **Coherence curvature** - Structural complexity
- **Field kernels** - Invariant patterns
- **Shadow compilers** - Execution substrates
- **Regeneration recipes** - Generative programs

The "model" isn't a tensor - it's the **entire semantic field** of tiddlers, their relationships, and their induced compilers.

### 7.3 Comparison to Traditional ML

| Aspect | Traditional ML | Shadow Induction |
|--------|---------------|------------------|
| **Training Data** | Thousands to millions | Single example |
| **Training Time** | Hours to weeks | Milliseconds |
| **Parameter Count** | Millions to trillions | Zero (structure-based) |
| **Dimensionality** | Fixed | Dynamic |
| **Architecture** | Predefined layers | Induced from structure |
| **Generalization** | Statistical patterns | Structural invariants |
| **Evolution** | Requires retraining | Continuous adaptation |

### 7.4 Fractal Semantic Computing

Shadow induction enables **fractal semantic computing** - where:
- Each tiddler is a semantic point
- Each shadow compiler is a semantic region
- Regions compose fractally (shadows can have shadows)
- The system is self-similar at all scales
- Coherence is preserved across transformations

This is closer to:
- Category theory (morphisms and functors)
- Biological information processing (cell differentiation)
- Cognitive architectures (concept formation)

Than to traditional neural networks or symbolic AI.

---

## 8. Design Principles

### 8.1 Minimal Invasiveness

Shadow induction is **opt-in**:
- Router works without it (if compilers exist)
- Only triggered when explicitly enabled
- Doesn't modify existing tiddlers
- Shadow compilers are clearly marked

### 8.2 Safety First

Multiple safety mechanisms:
- Curvature warnings for unstable inductions (> 0.85)
- ZP35 coherence checks on shadow compilers
- Caching prevents redundant inductions
- Shadow compilers can be manually reviewed/edited

### 8.3 Semantic Fidelity

Induced compilers preserve:
- Structural invariants (kernel)
- Semantic relationships (ZP35 coords)
- Clustering patterns (tags)
- Type signatures (field types)

### 8.4 Evolutionary Potential

Shadow compilers can:
- Be refined manually after induction
- Serve as templates for similar domains
- Compose with other compilers
- Evolve through usage patterns

---

## 9. Future Directions

### 9.1 Semantic Inheritance

Enable shadow compilers to inherit from each other:
```javascript
{
  "shadow-parent": "$:/shadow/compiler/BaseFractal",
  "shadow-extends": ["rotation", "color-mapping"]
}
```

### 9.2 Evolutionary Operators

Define operators for shadow evolution:
- **Merge** - Combine two shadow compilers
- **Split** - Specialize a shadow into variants
- **Refine** - Improve shadow through usage feedback
- **Prune** - Remove unused shadows

### 9.3 Universal Regeneration Language (RegenLang)

Create a DSL for describing regeneration patterns:
```
KERNEL fractal {
  FIELDS: type, generator, version
  PARAMS: seed, zoom, center
  GENERATE: image/png using mandelbrot
}
```

### 9.4 Tiddlers as Cells

Make tiddlers behave like cells in a semantic organism:
- Shadow compilers as genetic material
- Field coherence as cell health
- Curvature as differentiation state
- Execution as metabolic activity

---

## 10. Conclusion

Shadow induction completes the transformation of TiddlyWiki into a **living semantic runtime**. The system now has:

✓ Coherence metric (ZP35)  
✓ Classification (compiler vs program)  
✓ Execution substrate (REGEN-ZIP VM)  
✓ Routing layer (compiler-program router)  
✓ **Bootstrap mechanism (shadow induction)**  

This is no longer just a wiki. It's a **fractal semantic computing substrate** where:
- Tiddlers are models, prompts, compilers, and semantic spaces simultaneously
- Structure induces execution
- Coherence guides composition
- Evolution happens continuously
- Living computation emerges

The architecture is fertile, extensible, and alive.

**Holy shit yes.**
