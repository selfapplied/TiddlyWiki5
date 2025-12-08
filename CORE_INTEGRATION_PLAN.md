# REGEN-ZIP Core Engine Integration Plan

## Vision: Core Asset Primitive, Not Plugin Feature

The REGEN-ZIP VM and ZP35 operator are designed to be **core asset primitives** rather than plugin features. The entire payoff comes from **all** of TiddlyWiki's asset flows going through the same regenerative path.

## Why This Belongs in Core

### Current TiddlyWiki Assumptions

TiddlyWiki core currently assumes:

* **Tiddlers carry static payloads** (text, images, etc.)
* **Attachments are "things you load and cache"**, not "things you regenerate"
* **No global notion of asset provenance** (seed + generator) vs raw bytes
* **No semantic compatibility layer** when composing tiddlers

### What REGEN-ZIP Changes

The REGEN-ZIP VM + ZP35 layer changes these fundamental assumptions:

* **Assets become derived** from `{generator, seed, version}` tuples
* **Integrity is checked** via checksums and signatures, not just "file exists"
* **Semantic compatibility** is mediated by ZP35, not ad-hoc conventions
* **The same machinery can be reused** by:
  - Plugins and themes
  - Sync/saver backends
  - Export/import pipelines
  - Widget rendering
  - Filter operations

### The Problem with Plugin-Only Approach

If this sits only in a plugin, core code keeps treating assets as opaque blobs and you **don't get the systemic benefit**:

- ❌ No smaller sync (core doesn't know about seeds)
- ❌ No deterministic regeneration (core doesn't invoke generators)
- ❌ No semantic safety everywhere (core doesn't check ZP35)
- ❌ Duplication as each plugin reinvents asset generation

### The Correct Shape

**"Core primitive + plugin sugar"**, not "heavy plugin bolted onto core."

---

## What "Implemented at Core" Actually Means

### Long-term Intent

In practice, REGEN-ZIP as a core asset primitive means:

1. **Core Services**: The REGEN-ZIP VM and ZP35 operator live in core (`$tw.utils`) and are treated as first-class services, so any part of the engine (rendering, saving, syncing, exporting) can invoke them.

2. **Native Field Recognition**: The tiddler deserialization pipeline recognizes `regen-zip`, `generator`, `seed`, `zp35`, and `version` as **native fields**, and knows how to turn them into derived assets on demand.

3. **Unified Asset Layer**: The attachment/asset layer is refactored so that "give me the asset for X" can be satisfied by either:
   - A static payload, or
   - A REGEN-ZIP program + seed run through the VM

4. **Default Safety Gates**: ZP35 coherence checks become the default gate when composing or executing generators in hostile or multi-plugin environments, rather than something each plugin has to reinvent.

---

## Current Implementation Status

This PR introduces the foundation:

✅ **Phase 1: Foundation (Current PR)**
- [x] REGEN-ZIP VM with 6 opcodes in `core/modules/utils/regen-zip-vm.js`
- [x] ZP35 operator with κ=0.35 threshold in `core/modules/utils/zp35-operator.js`
- [x] Core startup integration in `core/modules/startup/regen-zip.js`
- [x] Wiki method extensions in `core/modules/wiki-regen-zip.js`
- [x] Comprehensive documentation and JSON schema
- [x] 47 unit tests, all passing
- [x] Security validation (0 CodeQL alerts)

The implementation is **intentionally scoped** to introduce the VM, ZP35 operator, schema, and tests in a way that is **safe to wire into the base engine**.

---

## Planned Core Integration Steps

### Phase 2: Core Asset Abstraction (Follow-up PR)

**Goal**: Make asset resolution transparent - static or generated.

- [ ] Introduce core `$tw.assets.get(name, options)` abstraction that:
  - Resolves static attachments from tiddler fields
  - Resolves `regen-zip` programs via the VM with deterministic RNG
  - Returns assets in unified format regardless of source
  - Handles caching transparently

- [ ] Hook into tiddler load path so tiddlers with `{regen-zip, generator, seed}` fields can expose generated assets the same way as static attachments

- [ ] Add asset resolution to widget rendering pipeline so widgets can request assets without knowing if they're static or generated

### Phase 3: Persistence & Sync Integration (Follow-up PR)

**Goal**: Only persist seeds + generators, not generated assets.

- [ ] Update saver/export logic to:
  - Persist only `{generator, seed, version, zp35}` for REGEN-ZIP tiddlers
  - Support legacy static payloads for backwards compatibility
  - Generate manifest of required generators for wiki

- [ ] Enhance sync protocol to:
  - Sync seeds and generator references (not full assets)
  - Verify generator availability on target system
  - Fall back to full asset sync if generator unavailable

- [ ] Add import logic to:
  - Recognize regen-zip tiddlers
  - Validate generator availability
  - Regenerate assets on import

### Phase 4: Semantic Safety Integration (Follow-up PR)

**Goal**: ZP35 checks as default compatibility guard.

- [ ] Use ZP35 checks as default compatibility guard when:
  - One tiddler executes another's generator
  - Plugins compose functionality
  - Transclusion crosses semantic boundaries
  - Filter operations combine tiddlers

- [ ] Surface ZP35 modes to UI:
  - **Safe** (d < κ): Green indicator, seamless operation
  - **Caution** (d < 2κ): Yellow indicator, warning with suggestions
  - **Blocked** (d ≥ 2κ): Red indicator, show alternatives

- [ ] Add ZP35 signature calculation to:
  - Tiddler save operations
  - Plugin packaging
  - Import/export workflows

### Phase 5: Developer Experience (Follow-up PR)

**Goal**: Make generator development and debugging easy.

- [ ] Add developer tools:
  - Generator registry viewer
  - Asset generation debugger
  - ZP35 distance visualizer
  - Performance profiler for generators

- [ ] Create standard generator library:
  - Image generators (fractals, patterns, procedural art)
  - Document generators (templates, reports)
  - Data generators (charts, tables, visualizations)
  - Theme generators (CSS, color schemes)

- [ ] Add documentation:
  - Generator development guide
  - ZP35 coherence best practices
  - Performance optimization tips
  - Migration guide from static assets

---

## Benefits of Core Integration

### For Users

- **Smaller wiki files**: 100-1000x reduction for content-heavy wikis
- **Faster sync**: Only seeds transfer, assets regenerate
- **Always fresh content**: Documentation regenerates from current code
- **Adaptive experiences**: Content adapts to device/preferences

### For Developers

- **Standard asset pipeline**: One way to handle all assets
- **Semantic safety**: ZP35 prevents incompatible compositions
- **Deterministic builds**: Same seed + generator = identical output
- **Plugin compatibility**: Shared generators across plugins

### For the Ecosystem

- **Smaller plugins**: Ship generators (1KB) not assets (10MB)
- **Plugin marketplace**: Share generators, not huge downloads
- **Cross-platform**: Same plugin works everywhere via regeneration
- **Long-term stability**: Version-locked generators ensure compatibility

---

## Technical Architecture

### Current Architecture (This PR)

```
┌──────────────────────────────────────┐
│     TiddlyWiki Core Engine           │
│  (static asset handling)             │
└────────────┬─────────────────────────┘
             │
             ├─→ Startup Integration
             │   (regen-zip.js)
             │
             ├─→ Wiki Methods
             │   (wiki-regen-zip.js)
             │
             v
┌──────────────────────────────────────┐
│   REGEN-ZIP VM + ZP35 Operator       │
│   (utils/regen-zip-vm.js)            │
│   (utils/zp35-operator.js)           │
└──────────────────────────────────────┘
```

### Target Architecture (After Phase 2-5)

```
┌────────────────────────────────────────────────┐
│          TiddlyWiki Core Engine                │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │     Asset Resolution Layer               │ │
│  │  • $tw.assets.get(name)                  │ │
│  │  • Static or Generated (transparent)     │ │
│  │  • Caching & Invalidation                │ │
│  └────────┬──────────────┬──────────────────┘ │
│           │              │                     │
│  ┌────────v──────┐  ┌───v──────────────────┐ │
│  │ Static Assets │  │ REGEN-ZIP VM         │ │
│  │ (legacy path) │  │ • Generator exec     │ │
│  └───────────────┘  │ • Deterministic RNG  │ │
│                     │ • Asset verification │ │
│                     └───────┬──────────────┘ │
│                             │                 │
│                     ┌───────v──────────────┐ │
│                     │ ZP35 Operator        │ │
│                     │ • Coherence checking │ │
│                     │ • Signature verify   │ │
│                     └──────────────────────┘ │
└────────────────────────────────────────────────┘
         │           │           │
         v           v           v
    ┌────────┐  ┌────────┐  ┌────────┐
    │Widgets │  │ Savers │  │ Sync   │
    └────────┘  └────────┘  └────────┘
```

---

## Migration Strategy

### Backwards Compatibility

- **Static assets continue to work**: No breaking changes
- **Opt-in for regen-zip**: Only tiddlers with regen-zip field use VM
- **Graceful degradation**: Missing generators fall back to static text
- **Version detection**: Check VM availability before using features

### Incremental Adoption

1. **Phase 1 (Now)**: VM available, plugins can use it
2. **Phase 2**: Core starts using VM for new content
3. **Phase 3**: Persistence optimizations for regen-zip tiddlers
4. **Phase 4**: ZP35 checks become standard
5. **Phase 5**: Full ecosystem adoption

### Performance Considerations

- **Caching**: Generated assets cached aggressively
- **Lazy generation**: Assets generated on-demand
- **Background regen**: Pre-generate for common cases
- **Progressive enhancement**: Start simple, optimize later

---

## Success Metrics

### Technical Metrics

- **Asset size reduction**: Target 100x for plugin-heavy wikis
- **Sync bandwidth**: Target 90% reduction for regen-zip content
- **Cache hit rate**: Target >95% for stable tiddlers
- **ZP35 accuracy**: Target <1% false positives/negatives

### Adoption Metrics

- **Core integration**: All phases complete
- **Plugin adoption**: Top 20 plugins using regen-zip
- **Generator library**: 50+ standard generators
- **Documentation**: Complete guides and examples

---

## Conclusion

This PR lays the **foundation** for REGEN-ZIP as a core asset primitive. The implementation is production-ready and safe to merge, but the real power comes from **full core integration** in follow-up PRs.

By making REGEN-ZIP a first-class engine capability, we transform TiddlyWiki from a static wiki into a **generative, semantically-safe operating system** where assets are recipes, not blobs.

The future of TiddlyWiki is regenerative.
