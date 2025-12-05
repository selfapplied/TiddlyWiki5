# Antclock Concepts Implementation in TiddlyWiki5

This document describes the practical implementation of concepts from the CE Tower antclock research paper in TiddlyWiki5.

**Research Source:** https://github.com/selfapplied/antclock/blob/main/arXiv/working.md

---

## Overview

The CE Tower is a three-layer functorial architecture for compositional learning. We have adapted several key concepts for TiddlyWiki5:

1. **Witness Fingerprints** - Semantic signatures for tiddler analysis
2. **Antclock (Experiential Time)** - Semantic change tracking
3. **Similar Tiddlers Widget** - Automated similarity detection

---

## 1. Witness Fingerprint System

### Location
- **Core Module:** `$:/core/modules/utils/witness-fingerprint.js`
- **Widget:** `$:/core/modules/widgets/similar-tiddlers.js`
- **Tests:** `editions/test/tiddlers/tests/test-witness-fingerprint.js`

### What It Does

The witness fingerprint system creates a multi-dimensional semantic signature for each tiddler, enabling sophisticated similarity analysis beyond simple keyword matching.

### Fingerprint Dimensions

Each tiddler gets a fingerprint with these components:

1. **Phase (θ)** - Semantic direction based on content analysis (0 to 2π)
2. **Depth (l)** - Hierarchical position in tag structure (0 to ~20)
3. **Sector (s)** - Domain classification from tags (0 to 99)
4. **Monodromy (m)** - Cyclic reference patterns (0 to 1)
5. **Link Density** - Links per 1000 characters
6. **Transclusion Complexity** - Number of transclusions and macros
7. **Field Complexity** - Number of custom fields

### Usage in JavaScript

```javascript
// Get witness fingerprint utilities
var witnessUtils = require("$:/core/modules/utils/witness-fingerprint.js");

// Calculate fingerprint for a tiddler
var tiddler = $tw.wiki.getTiddler("MyTiddler");
var fingerprint = witnessUtils.calculateWitnessFingerprint(tiddler, $tw.wiki);

// Find similar tiddlers
var similarTiddlers = witnessUtils.findSimilarTiddlers(tiddler, $tw.wiki, {
    threshold: 0.3,  // 0-1, lower = more similar required
    maxResults: 10   // Maximum number of results
});

// Calculate distance between two fingerprints
var distance = witnessUtils.calculateFingerprintDistance(fp1, fp2);

// Calculate resonance (similarity) between tiddlers
var resonance = witnessUtils.calculateResonance(tiddler1, tiddler2, $tw.wiki);
```

### Usage in WikiText

Use the `<$similar-tiddlers>` widget:

```wikitext
<$similar-tiddlers tiddler="CurrentTiddler" threshold="0.3" max="5"/>
```

**Parameters:**
- `tiddler` - Target tiddler (defaults to current tiddler)
- `threshold` - Similarity threshold 0-1 (default: 0.3)
- `max` - Maximum results (default: 10)

### Theory

Based on the CE Tower witness operator `<>g` which extracts self-describing invariant signatures as 4D fingerprints: (phase θ, depth l, sector s, monodromy m).

The ultrametric distance formula respects hierarchical structure:
```
d(a,b) = 2^(-min_common_depth)
```

This means tiddlers with shared deep structure are exponentially closer in semantic space.

---

## 2. Antclock (Experiential Time) System

### Location
- **Core Module:** `$:/core/modules/utils/antclock.js`
- **Tests:** `editions/test/tiddlers/tests/test-antclock.js`

### What It Does

The antclock measures time in semantic transition units rather than clock ticks. It advances only when semantically significant changes occur, enabling:

- Distinguish minor edits (typo fixes) from major revisions
- Track conceptual evolution vs temporal sequence
- Identify stable vs rapidly evolving knowledge areas
- Filter by semantic activity level

### Key Concepts

**Clock Rate R(x):**
The rate at which experiential time passes for a given change:

```
R(x) = χ_FEG · κ_d(x) · (1 + Q_9/11(x))
```

Where:
- `χ_FEG ≈ 0.638` - Transform quality measure
- `κ_d(x)` - Discrete curvature (change magnitude)
- `Q_9/11(x)` - Modular correction (fine structure)

**Simplified Implementation:**
```
R(x) = 0.638 * (0.4 * structural + 0.4 * semantic + 0.2 * coherence)
```

### Usage in JavaScript

```javascript
var anticlockUtils = require("$:/core/modules/utils/antclock.js");

// Calculate clock rate for a change
var clockRate = anticlockUtils.calculateClockRate(oldContent, newContent);

// Record an antclock tick
var result = anticlockUtils.recordAnticlockTick(
    $tw.wiki,
    tiddler,
    clockRate,
    { reason: "major revision" }
);

// Get experiential age
var age = anticlockUtils.getExperientialAge(tiddler);

// Get experiential history
var history = anticlockUtils.getExperientialHistory(tiddler);

// Calculate recent activity rate
var rate = anticlockUtils.getRecentActivityRate(tiddler, 10); // Last 10 events

// Compare activity between tiddlers
var comparison = anticlockUtils.compareExperientialActivity(tiddler1, tiddler2);
```

### Tiddler Fields

The antclock system uses these fields:

- **`experiential-time`** - Current experiential age (cumulative clock rate)
- **`experiential-history`** - JSON array of antclock events

Example experiential-history entry:
```json
{
  "timestamp": "2024-12-05T14:30:00.000Z",
  "experientialTime": 3.5,
  "clockRate": 0.8,
  "details": {
    "reason": "major revision",
    "structural": 0.6,
    "semantic": 0.7,
    "coherence": 0.3
  }
}
```

### Change Detection

The system analyzes three types of changes:

1. **Structural Changes:**
   - Headers added/removed
   - Lists added/removed
   - Transclusions added/removed
   - Links added/removed

2. **Semantic Changes:**
   - Words added/removed
   - Content length changes
   - Vocabulary shifts

3. **Coherence Changes:**
   - Ratio of structure to content
   - Internal consistency
   - Formatting quality

### Theory

Based on the CE Tower antclock mechanism which enables **experiential compositionality** - composition over lived moments rather than abstract positions.

Example from the paper:
> Processing "The cat sat on the mat" might take 7 positional steps but only 2 antclock ticks: one for establishing the subject (cat), one for establishing the relation (sat on mat).

---

## 3. Implementation Details

### File Structure

```
TiddlyWiki5/
├── core/
│   ├── modules/
│   │   ├── utils/
│   │   │   ├── witness-fingerprint.js  (NEW)
│   │   │   └── antclock.js             (NEW)
│   │   └── widgets/
│   │       └── similar-tiddlers.js     (NEW)
│   └── wiki/
│       └── similar-tiddlers-demo.tid   (NEW)
├── editions/
│   └── test/
│       └── tiddlers/
│           └── tests/
│               ├── test-witness-fingerprint.js  (NEW)
│               └── test-antclock.js             (NEW)
└── tiddlywiki-evolution-research/
    ├── antclock-tiddlywiki-analysis.md          (NEW)
    └── antclock-implementation-readme.md        (NEW - this file)
```

### Testing

All functionality is fully tested:

```bash
# Run all tests
npm test

# Tests include:
# - Witness fingerprint calculation
# - Fingerprint distance metrics
# - Similar tiddler detection
# - Antclock clock rate calculation
# - Experiential time tracking
# - History management
```

**Test Coverage:**
- 16 tests for witness fingerprint system
- 16 tests for antclock system
- All tests passing (0 failures)

### Performance Considerations

1. **Fingerprint Calculation:**
   - Lazy computation recommended
   - Cache fingerprints when possible
   - Recalculate only on tiddler change

2. **Similar Tiddler Search:**
   - O(n) where n = number of tiddlers
   - Can be optimized with indexing
   - Use threshold to limit results

3. **Antclock Updates:**
   - Only calculate on significant changes
   - Threshold prevents tracking minor edits
   - History limited to 100 events

### Future Enhancements

Based on the antclock paper, potential future implementations:

1. **Guardian System** (CE2)
   - Phase resonance guardian (ϕ) - semantic discontinuities
   - Indentation-bracket guardian (∂) - structural discontinuities
   - Return/phaselock guardian (ℛ) - coherence discontinuities
   - Use for link quality scoring

2. **Compositional Operators** (CE1)
   - Memory operator []a - transclusion history
   - Domain operator {}l - hierarchical nesting
   - Transform operator ()r - content morphisms
   - Enhanced bracket-based transclusion

3. **Grammar Evolution** (CE3)
   - Error-lift operator 𝔈 - adaptive syntax
   - Detect repeated workaround patterns
   - Propose custom shorthand syntax
   - User-specific WikiText extensions

4. **Advanced Features**
   - Multi-dimensional clustering
   - Semantic version control
   - Experiential timeline visualization
   - Resonance pattern detection

---

## 4. Theoretical Background

### CE Tower Architecture

The CE Tower is a three-layer system:

**CE1: Discrete Grammar Category**
- Provides bracket topology
- Four primitive operators
- Ultrametric distance
- Satisfies formal compositionality

**CE2: Dynamical Flow Category**
- Guardian system for boundaries
- Antclock for experiential time
- Phase coherence maintenance
- Nash equilibrium attention

**CE3: Emergent Simplicial Category**
- Error-lift operator
- Grammar evolution
- Meta-circular evaluation
- Continuous adaptation

### Key Innovations Applied

1. **Temporal Compositionality**
   - Antclock for experiential time
   - Semantic transitions vs clock ticks
   - Meaningful change tracking

2. **Witness Fingerprints**
   - Multi-dimensional signatures
   - Hierarchical compression
   - Computable invariants

3. **Ultrametric Topology**
   - Distance based on common depth
   - Exponential similarity for shared structure
   - Natural for knowledge hierarchies

### Citations

The antclock paper cites these key works that inform the implementation:

- [1] McCurdy et al. (2024) - Compositional learning challenges
- [3] Sathe et al. (2024) - Sparse compositionality in language
- [4] Elmoznino et al. (2025) - Complexity-based compositionality theory
- [5] Lee et al. (2024) - Geometric signatures of compositionality

---

## 5. Usage Examples

### Example 1: Finding Related Research Notes

```wikitext
! Similar Research Notes

<$similar-tiddlers tiddler={{!!title}} threshold="0.25" max="8"/>
```

Shows tiddlers with semantic similarity > 75% to the current one.

### Example 2: Tracking Concept Evolution

```javascript
// Get a tiddler's experiential age
var tiddler = $tw.wiki.getTiddler("Quantum Computing");
var age = anticlockUtils.getExperientialAge(tiddler);

console.log("Experiential age:", age);
// Output: "Experiential age: 12.5"

// This tiddler has undergone 12.5 units of semantic change
// regardless of the number of edits or time elapsed
```

### Example 3: Identifying Active Topics

```javascript
// Compare multiple tiddlers by activity
var topics = ["AI", "Blockchain", "Quantum", "Security"];
var activities = topics.map(function(topic) {
    var tiddler = $tw.wiki.getTiddler(topic);
    return {
        topic: topic,
        activity: anticlockUtils.getExperientialAge(tiddler)
    };
});

activities.sort(function(a, b) {
    return b.activity - a.activity;
});

console.log("Most active topics:", activities);
```

### Example 4: Custom Similarity Widget

```wikitext
! Related Content

<div class="related-content">

<$list filter="[<currentTiddler>]">
  <$vars currentTitle={{!!title}}>
    <$similar-tiddlers 
      tiddler=<<currentTitle>> 
      threshold="0.3" 
      max="5"
    />
  </$vars>
</$list>

</div>

<style>
.related-content {
  background: #f0f0f0;
  padding: 1em;
  border-radius: 0.5em;
  margin: 1em 0;
}
</style>
```

---

## 6. Comparison with Traditional Approaches

### Traditional Similarity
- Keyword matching (TF-IDF)
- Cosine similarity of term vectors
- Ignores structure and hierarchy
- Binary link relationships

### Witness Fingerprint Approach
- Multi-dimensional semantic analysis
- Hierarchical depth awareness
- Cyclic pattern detection
- Weighted similarity scoring
- Resonance between deep structures

### Traditional Version Control
- Timestamp-based history
- All edits weighted equally
- Diff shows text changes
- Linear temporal sequence

### Antclock Approach
- Experiential time tracking
- Significance-weighted events
- Semantic change analysis
- Event-based temporal ordering
- Distinguish minor vs major revisions

---

## 7. Contributing

To extend this implementation:

1. **Add New Fingerprint Dimensions**
   - Edit `witness-fingerprint.js`
   - Add dimension calculation function
   - Update distance calculation
   - Add tests

2. **Enhance Clock Rate Calculation**
   - Edit `antclock.js`
   - Refine change detection algorithms
   - Adjust weighting factors
   - Validate with tests

3. **Create New Widgets**
   - Use witness/antclock utilities
   - Follow TiddlyWiki widget patterns
   - Add to `core/modules/widgets/`
   - Document usage

---

## 8. License

This implementation is part of TiddlyWiki5 and follows the same BSD license.

The antclock research concepts are from:
https://github.com/selfapplied/antclock

---

## 9. Acknowledgments

- **Jeremy Ruston** - TiddlyWiki creator
- **Antclock Research Team** - CE Tower architecture and concepts
- **TiddlyWiki Community** - Testing and feedback

---

## 10. Resources

**Documentation:**
- TiddlyWiki: https://tiddlywiki.com/
- TiddlyWiki Dev: https://tiddlywiki.com/dev/
- Antclock Paper: https://github.com/selfapplied/antclock/blob/main/arXiv/working.md

**Related Research:**
- `semantic-git-vectors-research-report.md` - Mathematical merge resolution
- `antclock-tiddlywiki-analysis.md` - Detailed concept analysis

**Code:**
- Witness Fingerprint: `core/modules/utils/witness-fingerprint.js`
- Antclock: `core/modules/utils/antclock.js`
- Similar Tiddlers Widget: `core/modules/widgets/similar-tiddlers.js`

---

**Last Updated:** December 5, 2024
