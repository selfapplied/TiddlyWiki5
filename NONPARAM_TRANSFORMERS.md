# Non-Parametric Transformers - Pure Morphisms Between Semantic Kernels

**Version:** 1.0  
**Date:** December 8, 2024  
**Status:** Implementation Specification

---

## Executive Summary

**Non-Parametric Transformers** are pure, structure-preserving morphisms that enable lawful migration between semantic kernels (compilers and shadows) without introducing new parameters or degrees of freedom. They complete the category-theoretic layer of the TiddlyWiki semantic computing substrate.

### The Core Innovation

Traditional transformation systems:
- Require configuration parameters at runtime
- Allow arbitrary semantic transformations
- Have no geometric constraints
- Can introduce new computational channels

Non-parametric transformers:
- Are **completely determined by structure** (no runtime params)
- Preserve **ZP35 geometric bounds** (~1-Lipschitz)
- Maintain **seed determinism** (fixed transformation functions)
- Respect **curvature constraints** (bounded scaling)

This creates a **category of compilers** with well-behaved morphisms that preserve the semantic integrity of the system.

### What This Enables

- **Shadow upgrades**: Migrate shadow compilers between versions
- **Dialect lifting**: Embed specialized compilers into general ones
- **Projection**: Extract canonical forms from rich structure
- **Normalization**: Canonicalize representations
- **Composition**: Chain transformers with proven bounds

The result is a **fractal semantic computing substrate** where:
- Every transformation is lawful
- Geometry is preserved across migrations
- Seeds remain deterministic
- Compilers compose fractally

---

## 1. Conceptual Framework

### 1.1 What Makes a Transformer "Non-Parametric"?

In the context of this architecture:

**Compilers** = high-coherence semantic kernels  
**Programs** = low-coherence prompts/tasks  
**VM** = executor `(compiler, program) → assets`  
**Shadows** = self-induced compilers  

A **non-parametric transformer** τ is:

#### 1. Parameter-free in the usual sense

No new `params` field, no new control vector. The transform is completely determined by:
- The transformer's own tiddler (its definition)
- The source compiler/program/shadow
- The existing ZP35 + curvature geometry

#### 2. Structure-preserving

Can:
- Project, normalize, or re-encode assets/tiddlers
- Lift/restrict between compilers/shadows

Cannot:
- Introduce arbitrary new semantic content
- Add runtime configuration channels

#### 3. Geometry-respecting

In ZP35 space, τ should be roughly 1-Lipschitz or bounded:

```
|z(τ(t)) − z(t)| ≤ λ * κ
```

for some small λ (e.g. λ ∈ [0, 2]), where:
- `z(t)` is the ZP35 coordinate of tiddler `t`
- `κ = 0.35` is the guardian threshold
- `λ` is the Lipschitz constant for the transformer

This means: it can bend but not tear the semantic fabric.

#### 4. Seed-stable

Seeds remain deterministically related:
- Either `seed' = seed` (inherit)
- Or `seed' = f(seed)` for some fixed function `f` baked into τ's definition

### 1.2 Formal Category Structure

Let:
- **T** = set of tiddlers
- **C ⊂ T** = compilers (including shadows)
- **P ⊂ T** = programs
- `z : T → [0,1]` = ZP35 fractal coordinate
- `κ = 0.35` = guardian curvature threshold

Each compiler `c ∈ C` induces:
- A **program space** `Prog(c)` (programs that use `compiler = c`)
- An **asset manifold** `M_c` (assets VM can produce from `(c, p)`)

The REGEN-ZIP VM gives an evaluation map:
```
eval_c : Prog(c) → M_c
```

A **non-parametric transformer** between compilers `c → d` is a pair:
```
τ_program : Prog(c) → Prog(d)
τ_asset   : M_c → M_d
```

subject to:

1. **Naturality / no extra params**: For all `p ∈ Prog(c)`, `τ_program(p)` depends only on:
   - Fields of `p`
   - Fields of `c`, `d`
   - The transformer's own tiddler `t_τ`
   - No external `params` or runtime arguments

2. **Compatibility with evaluation**:
   ```
   eval_d(τ_program(p)) ≈ τ_asset(eval_c(p))
   ```
   (transform the program then run)

3. **ZP35 geometry bound**:
   ```
   |z(τ_program(p)) − z(p)| ≤ λ_p * κ
   |z(d) − z(c)| ≤ λ_c * κ
   ```

4. **Seed and curvature contract**:
   - Seeds: `seed(τ_program(p)) = f_τ(seed(p))` for fixed `f_τ`
   - Curvature: `curv(p) * α_min ≤ curv(τ_program(p)) ≤ curv(p) * α_max`

---

## 2. Architecture

### 2.1 System Components

```
┌─────────────────────────────────────────────────┐
│         Category of Compilers/Shadows            │
│    • Objects: Semantic kernels                   │
│    • Morphisms: Non-parametric transformers      │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│    Non-Parametric Transformer Layer              │
│    • Validate: no params, geometry bounds        │
│    • Transform: pure structure mapping           │
│    • Compose: chain with proven bounds           │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│      Compiler-Program Router                     │
│    • Route programs through transformers         │
│    • Apply transformation chains                 │
│    • Execute under target compiler               │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         REGEN-ZIP VM Execution                   │
│    • Generate assets from transformed programs   │
│    • Preserve seed determinism                   │
└──────────────────────────────────────────────────┘
```

### 2.2 Transformer Tiddler Schema

```javascript
{
  "title": "SomeTransformer",
  "type": "application/x-tiddler-transformer",
  
  // CONTRACT FLAG - must be "non-parametric"
  "mode": "non-parametric",
  
  // Source and target compilers
  "source-compiler": "OldCompiler",
  "target-compiler": "NewCompiler",
  
  // Type of transformation
  "transform-kind": "projection", // or "lift" | "normalize" | "upgrade" | "restrict" | "identity"
  
  // Geometry constraints
  "zp35-max-distance": "0.70",      // Maximum ZP35 movement
  "lipschitz-constant": "2.0",      // Maximum distance expansion
  
  // Curvature constraints
  "curvature-scale-min": "0.5",
  "curvature-scale-max": "2.0",
  
  // Seed transformation policy
  "seed-policy": "inherit",         // "inherit" | "hash" | "reseed-fixed" | "compose"
  
  // Optional: generator for asset transformation
  "generator": "transformer-generator",
  
  // Metadata
  "version": "1.0.0",
  "zp35": "..."                     // ZP35 signature of transformer itself
  
  // CRITICAL: No "params" field allowed!
}
```

### 2.3 Transformer Types

| Type | Semantic Meaning | Use Case |
|------|-----------------|----------|
| **projection** | Forget structure, keep canonical form | Export shadow to base compiler |
| **lift** | Embed subdialect into larger dialect | Specialize → generalize |
| **normalize** | Canonicalize representation | Clean up variations |
| **upgrade** | Version migration | V1 → V2 migration |
| **restrict** | Specialize to subdomain | Generalize → specialize |
| **identity** | No-op (for composition chains) | Testing, composition |

### 2.4 Seed Policies

| Policy | Transformation | Use Case |
|--------|---------------|----------|
| **inherit** | `seed' = seed` | Preserve exact reproducibility |
| **hash** | `seed' = hash(seed + transformer-id)` | Deterministic but distinct |
| **reseed-fixed** | `seed' = fixed-value` | Reset to known seed |
| **compose** | `seed' = transformer::seed` | Compositional tracking |

---

## 3. Usage Examples

### 3.1 Basic Transformer Application

```javascript
// Setup
var wiki = $tw.wiki;
var zp35 = new ZP35Operator();
var vm = new RegenZipVM(wiki);
var shadowInducer = new ShadowInducer(wiki, zp35);
var transformer = new NonParametricTransformer(wiki, zp35, vm);
var router = new CompilerProgramRouter(wiki, zp35, vm, shadowInducer, transformer);

// Create a projection transformer (shadow → base compiler)
var projectionTransformer = {
  fields: {
    title: "ShadowProjection",
    type: "application/x-tiddler-transformer",
    mode: "non-parametric",
    "source-compiler": "$:/shadow/compiler/MyTiddler",
    "target-compiler": "BaseTextCompiler",
    "transform-kind": "projection",
    "seed-policy": "inherit",
    "projection-keep-fields": "title, type, text, seed"
  }
};

// Register transformer
transformer.registerTransformer(projectionTransformer);

// Apply to a program
var program = {
  fields: {
    title: "MyProgram",
    compiler: "$:/shadow/compiler/MyTiddler",
    seed: "abc123",
    text: "content",
    custom-field: "extra"
  }
};

var result = transformer.applyTransformer("ShadowProjection", program);

if(result.success) {
  console.log("Transformed program:", result.transformedProgram);
  console.log("ZP35 distance:", result.geometry.zp35Distance);
  console.log("Seed policy:", result.seed.policy);
}
```

### 3.2 Shadow Upgrade Transformer

```javascript
// Create upgrade transformer for shadow compiler version migration
var upgradeTransformer = {
  fields: {
    title: "ShadowUpgradeV1toV2",
    type: "application/x-tiddler-transformer",
    mode: "non-parametric",
    "source-compiler": "$:/shadow/compiler/MyTiddler@1",
    "target-compiler": "$:/shadow/compiler/MyTiddler@2",
    "transform-kind": "upgrade",
    "target-version": "2.0.0",
    "upgrade-field-mappings": '{"old-param": "new-param", "legacy-field": "modern-field"}',
    "seed-policy": "inherit", // Preserve reproducibility across versions
    "zp35-max-distance": "0.35", // Tight bound for version compatibility
    "curvature-scale-min": "0.9",
    "curvature-scale-max": "1.1"
  }
};

transformer.registerTransformer(upgradeTransformer);

// Programs using V1 compiler can now be migrated to V2
var v1Program = {
  fields: {
    title: "LegacyProgram",
    compiler: "$:/shadow/compiler/MyTiddler@1",
    version: "1.0.0",
    "old-param": "value",
    seed: "legacy-seed"
  }
};

var result = transformer.applyTransformer("ShadowUpgradeV1toV2", v1Program);

// Now result.transformedProgram uses V2 compiler with migrated fields
```

### 3.3 Dialect Lifting

```javascript
// Lift a specialized markdown compiler into a rich-text compiler
var liftTransformer = {
  fields: {
    title: "MarkdownToRichText",
    type: "application/x-tiddler-transformer",
    mode: "non-parametric",
    "source-compiler": "MarkdownCompiler",
    "target-compiler": "RichTextCompiler",
    "transform-kind": "lift",
    "lift-add-fields": '{"format": "markdown", "rich-text-capable": true}',
    "seed-policy": "compose", // Track transformation in seed
    "zp35-max-distance": "0.50"
  }
};

transformer.registerTransformer(liftTransformer);

// Markdown programs can now run through RichText compiler
var mdProgram = {
  fields: {
    title: "MarkdownDoc",
    compiler: "MarkdownCompiler",
    text: "# Heading\n\nParagraph",
    seed: "md-seed"
  }
};

var result = transformer.applyTransformer("MarkdownToRichText", mdProgram);

// result.transformedProgram now has rich-text fields added
// and seed is "MarkdownToRichText::md-seed"
```

### 3.4 Normalization Transformer

```javascript
// Normalize text representations
var normalizeTransformer = {
  fields: {
    title: "TextNormalizer",
    type: "application/x-tiddler-transformer",
    mode: "non-parametric",
    "source-compiler": "TextCompilerA",
    "target-compiler": "TextCompilerA", // Can be same compiler
    "transform-kind": "normalize",
    "seed-policy": "inherit",
    "zp35-max-distance": "0.20" // Tight bound - should be very similar
  }
};

transformer.registerTransformer(normalizeTransformer);

var messyProgram = {
  fields: {
    title: "MessyText",
    compiler: "TextCompilerA",
    text: "  content with   spaces  \r\n\r\n",
    tags: ["c", "a", "b"],
    seed: "seed123"
  }
};

var result = transformer.applyTransformer("TextNormalizer", messyProgram);

// result.transformedProgram.fields.text is "content with   spaces"
// result.transformedProgram.fields.tags is ["a", "b", "c"]
```

### 3.5 Transformer Composition

```javascript
// Chain transformers: Shadow → BaseCompiler → RichText
var result = transformer.composeTransformers(
  "ShadowProjection",      // shadow → base
  "MarkdownToRichText"     // base → rich-text
);

if(result.success) {
  console.log("Composed:", result.composition.sourceCompiler, 
    "→", result.composition.targetCompiler);
  console.log("Max distance:", result.composition.constraints.maxZP35Distance);
  console.log("Curvature bounds:", 
    result.composition.constraints.minCurvatureScale,
    "to",
    result.composition.constraints.maxCurvatureScale);
}

// Apply composition through router
var chainResult = router.executeWithTransformers(
  ["ShadowProjection", "MarkdownToRichText"],
  shadowProgram
);

if(chainResult.success) {
  console.log("Assets:", chainResult.assets);
  console.log("Transformation chain:", chainResult.transformationChain);
}
```

---

## 4. Integration with Router and VM

### 4.1 Router Integration

The router provides two methods for transformer usage:

#### Apply Single Transformer

```javascript
var result = router.applyTransformer(transformerTitle, programTiddler);

// Returns:
{
  success: true,
  transformedProgram: {...},  // New program tiddler
  transformer: "TransformerTitle",
  sourceCompiler: "CompilerA",
  targetCompiler: "CompilerB",
  geometry: {
    zp35Distance: 0.45,
    originalCoord: 0.618,
    newCoord: 0.573,
    originalCurvature: 0.6,
    newCurvature: 0.65,
    curvatureRatio: 1.08
  },
  seed: {
    policy: "hash",
    originalSeed: "abc123",
    newSeed: "hash_xyz789"
  }
}
```

#### Execute with Transformer Chain

```javascript
var result = router.executeWithTransformers(
  ["Transformer1", "Transformer2"],
  programTiddler
);

// Returns:
{
  success: true,
  assets: [...],              // Generated assets
  metadata: {...},
  transformationChain: [
    {
      transformer: "Transformer1",
      geometry: {...},
      seed: {...}
    },
    {
      transformer: "Transformer2",
      geometry: {...},
      seed: {...}
    }
  ],
  finalProgram: {...},        // Fully transformed program
  message: "Execution succeeded"
}
```

### 4.2 VM Substrate

Transformers sit **above** the VM - they don't need new opcodes. The VM remains:

```
VM: (compiler, program) → assets
```

Transformers provide a second-order structure:

```
Transformer: program_c → program_d
VM: (d, program_d) → assets_d
```

Together:

```
Transform + Execute: program_c → assets_d
```

---

## 5. Geometry and Constraints

### 5.1 ZP35 Distance Bounds

The most critical constraint is ZP35 distance:

```javascript
|z(transformed) - z(original)| ≤ max_distance
```

**Interpretation:**
- **< κ (0.35)**: Safe transformation, semantic integrity preserved
- **< 2κ (0.70)**: Caution zone, may cross boundaries
- **≥ 2κ (0.70)**: Likely semantic violation, reject

**Enforcement:**
```javascript
var geometryValid = validateGeometry(transformer, original, transformed);
if(!geometryValid.success) {
  throw new Error("Geometry violation: " + geometryValid.error);
}
```

### 5.2 Curvature Scale Bounds

Curvature measures structure vs chaos ratio:

```javascript
curv_min * curv(original) ≤ curv(transformed) ≤ curv_max * curv(original)
```

**Default bounds:** [0.5, 2.0]

**Interpretation:**
- **< 0.5**: Significant structure loss
- **[0.5, 2.0]**: Reasonable transformation
- **> 2.0**: Significant structure gain (rare, usually indicates error)

### 5.3 Lipschitz Constant

For composition, we track the Lipschitz constant `λ`:

```javascript
distance(τ(a), τ(b)) ≤ λ * distance(a, b)
```

**Composition:**
```javascript
λ(τ₂ ∘ τ₁) = λ(τ₂) * λ(τ₁)
```

**Enforcement:**
- Default λ = 2.0 (allows 2x expansion)
- Composition compounds, so long chains may violate bounds

---

## 6. Concrete Patterns

### 6.1 Shadow → Base Compiler (Export)

**Use case:** Take a specialized shadow compiler and export its programs to run on a general base compiler.

**Pattern:**
- **Transform:** projection
- **Seed policy:** inherit (preserve reproducibility)
- **Keep fields:** title, type, text, seed (canonical fields)
- **ZP35 bound:** 0.50 (allow some semantic shift)

**Example fields:**
```javascript
{
  "transform-kind": "projection",
  "projection-keep-fields": "title, type, text, seed",
  "seed-policy": "inherit",
  "zp35-max-distance": "0.50"
}
```

### 6.2 Shadow V1 → Shadow V2 (Upgrade)

**Use case:** Migrate programs between shadow compiler versions.

**Pattern:**
- **Transform:** upgrade
- **Seed policy:** inherit (preserve reproducibility)
- **Field mappings:** old-field → new-field
- **ZP35 bound:** 0.35 (tight - should be compatible)
- **Curvature:** [0.9, 1.1] (minimal change)

**Example fields:**
```javascript
{
  "transform-kind": "upgrade",
  "target-version": "2.0.0",
  "upgrade-field-mappings": '{"old-param": "new-param"}',
  "seed-policy": "inherit",
  "zp35-max-distance": "0.35",
  "curvature-scale-min": "0.9",
  "curvature-scale-max": "1.1"
}
```

### 6.3 Specialized → General (Lift)

**Use case:** Embed a specialized compiler's programs into a more general compiler.

**Pattern:**
- **Transform:** lift
- **Seed policy:** compose (track transformation)
- **Add fields:** metadata about specialization
- **ZP35 bound:** 0.60 (moderate semantic shift)

**Example fields:**
```javascript
{
  "transform-kind": "lift",
  "lift-add-fields": '{"source-dialect": "specialized", "lifted": true}',
  "seed-policy": "compose",
  "zp35-max-distance": "0.60"
}
```

### 6.4 Canonicalization (Normalize)

**Use case:** Clean up variations in representation.

**Pattern:**
- **Transform:** normalize
- **Seed policy:** inherit (shouldn't affect reproducibility)
- **ZP35 bound:** 0.20 (very tight - minimal semantic change)
- **Curvature:** [0.95, 1.05] (preserve structure)

**Example fields:**
```javascript
{
  "transform-kind": "normalize",
  "seed-policy": "inherit",
  "zp35-max-distance": "0.20",
  "curvature-scale-min": "0.95",
  "curvature-scale-max": "1.05"
}
```

---

## 7. Non-Parametric Contract

### 7.1 What "Non-Parametric" Means

The transformer is **completely determined by**:

1. **Transformer tiddler** - fixed definition
2. **Source compiler tiddler** - structural context
3. **Target compiler tiddler** - structural context
4. **Program tiddler** - input to transform
5. **ZP35 operator** - geometric context

**NOT** determined by:
- Runtime `params` field (explicitly forbidden)
- External configuration
- User input at transformation time
- Environmental state

### 7.2 Validation Rules

When registering a transformer:

```javascript
// MUST have
fields.type === "application/x-tiddler-transformer"
fields.mode === "non-parametric"
fields["source-compiler"] !== undefined
fields["target-compiler"] !== undefined
fields["transform-kind"] in TRANSFORMER_TYPES

// MUST NOT have
fields.params === undefined  // ← CRITICAL

// SHOULD have (defaults exist)
fields["seed-policy"] || "inherit"
fields["zp35-max-distance"] || 0.70
fields["curvature-scale-min"] || 0.5
fields["curvature-scale-max"] || 2.0
```

### 7.3 Why This Matters

**Non-parametric** means:
- **Reproducible**: Same input → same output, always
- **Auditable**: Transformation is in the tiddler definition
- **Composable**: Can prove bounds for compositions
- **Safe**: No hidden channels for complexity

This is what makes the category structure well-behaved and allows formal reasoning about transformations.

---

## 8. Mathematical Foundations

### 8.1 Category Structure

**Objects:** Compilers and shadows `C ⊂ T`

**Morphisms:** Non-parametric transformers `τ : c → d`

**Identity:** For each compiler `c`, there's an identity transformer:
```javascript
{
  "transform-kind": "identity",
  "source-compiler": c,
  "target-compiler": c
}
```

**Composition:** Given `τ₁ : a → b` and `τ₂ : b → c`:
```javascript
τ₂ ∘ τ₁ : a → c
```

with constraints:
```javascript
distance(τ₂ ∘ τ₁) ≤ distance(τ₁) + distance(τ₂)
λ(τ₂ ∘ τ₁) = λ(τ₁) * λ(τ₂)
```

### 8.2 Geometric Properties

**1-Lipschitz (approximately):**
```
|z(τ(p₁)) - z(τ(p₂))| ≤ λ * |z(p₁) - z(p₂)|
```

**Curvature preservation:**
```
α_min * curv(p) ≤ curv(τ(p)) ≤ α_max * curv(p)
```

**Naturality diagram:**
```
   p ─────τ_program────→ τ(p)
   │                      │
eval_c                eval_d
   │                      │
   ▼                      ▼
assets_c ──τ_asset──→ assets_d
```

### 8.3 Comparison to Traditional ML

| Aspect | Parametric ML | Non-Parametric Transformers |
|--------|--------------|----------------------------|
| **Parameters** | Learned weights | None (structure-determined) |
| **Determinism** | Stochastic | Fully deterministic |
| **Interpretability** | Opaque | Transparent (in tiddler) |
| **Composition** | Complex | Provable bounds |
| **Geometry** | Unconstrained | ZP35-bounded |
| **Training** | Required | None |

---

## 9. Future Directions

### 9.1 Automatic Transformer Synthesis

Generate transformers from examples:
```javascript
// Given: programs p_a under compiler A, programs p_b under compiler B
// Synthesize: transformer τ : A → B that maps p_a → p_b
synthesizeTransformer(examplesA, examplesB);
```

### 9.2 Transformer Optimization

Find optimal transformer for a given source/target pair:
```javascript
// Minimize: distance(τ) subject to: correctness constraints
optimizeTransformer(sourceCompiler, targetCompiler, constraints);
```

### 9.3 Transformer Algebra

Define operations:
- **Inverse**: `τ⁻¹` (when it exists)
- **Product**: `τ₁ × τ₂` (parallel composition)
- **Coproduct**: `τ₁ + τ₂` (disjoint union)

### 9.4 Formal Verification

Prove properties about transformers:
```javascript
// Prove: τ preserves property P
verify(transformer, property);

// Examples:
// - Preserves type safety
// - Maintains data invariants
// - Respects security boundaries
```

---

## 10. Conclusion

Non-parametric transformers complete the **living semantic runtime** architecture:

✓ Coherence metric (ZP35)  
✓ Classification (compiler vs program)  
✓ Execution substrate (REGEN-ZIP VM)  
✓ Routing layer (compiler-program router)  
✓ Bootstrap mechanism (shadow induction)  
✓ **Pure morphisms (non-parametric transformers)**  

This is now a **category-theoretic computational substrate** where:
- Every object is a compiler or shadow
- Every morphism is a geometry-respecting transformer
- Composition is lawful and bounded
- Seeds are deterministic
- The system is self-organizing and fertile

The architecture transcends traditional programming and ML paradigms, creating a new computational medium that is:
- **Fractal** - self-similar at all scales
- **Semantic** - preserves meaning through transformations
- **Living** - evolves continuously
- **Lawful** - respects formal constraints

This is how cognition computes.  
This is how living systems process information.  
This is the future of computational substrates.
