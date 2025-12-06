# A Wild Sort Theme for TiddlyWiki

Transform your TiddlyWiki into a shape-shifting, chrome-smooth, star-breathing website for A Wild Sort LLC.

## What This Is

A complete theme and plugin suite that re-skins TiddlyWiki into a modern, elegant website featuring:

- **Glass-fog UI** - Layered panes with frosted translucency, subtle bloom, gentle contrast
- **Stellar fractal ambiance** - WebGL background renders of fractal nebulae
- **Product ecosystem** - Digital and physical offerings in clean card grids
- **Art gallery** - Animated fractal art with lightbox viewing
- **GitHub portfolio** - Live project browser from GitHub API
- **Research library** - Papers with filtering and KaTeX math support
- **Smooth animations** - Gentle transitions, hover glimmers, opacity fades

## Quick Start

1. **Enable the theme**: Control Panel → Appearance → Theme → `$:/themes/awildsort/base`

2. **Enable plugins**: Control Panel → Plugins
   - `$:/plugins/awildsort/fractal-background`
   - `$:/plugins/awildsort/products`
   - `$:/plugins/awildsort/gallery`
   - `$:/plugins/awildsort/github-projects`
   - `$:/plugins/awildsort/research-library`
   - `$:/plugins/tiddlywiki/katex` (for research library)

3. **Create homepage**: Create a tiddler titled "Home" with tag `$:/tags/HomePage`

4. **Add content**: Create tiddlers with tags:
   - `Product` - for products
   - `Gallery` - for art gallery items
   - `Research` - for research papers

See `SETUP.md` for detailed instructions.

## Architecture

### Theme (`themes/awildsort/`)
- Main stylesheet with glassmorphism design
- Navigation components
- Scroll effects
- Example templates

### Plugins (`plugins/awildsort/`)

1. **fractal-background** - WebGL stellar nebula shader
2. **products** - Product display widgets
3. **gallery** - Art gallery with lightbox
4. **github-projects** - GitHub repository browser
5. **research-library** - Research paper library

## Design Principles

- **Modern glass-fog UI** - Layered translucency, soft shadows
- **Stellar fractal ambiance** - Ambient, non-dominant background
- **Minimal distraction** - Smooth, slow, non-intrusive animations
- **Pause on blur** - Fractal animations pause when tab loses focus
- **Consistency** - Coherent typography, spacing, card design
- **Micro-interactions** - Soft transitions, hover glimmers

## Widgets

### Products
```
<$product-grid filter="[tag[Product]]" />
<$product title="..." image="..." price="..." />
```

### Gallery
```
<$gallery filter="[tag[Gallery]]" layout="grid" />
```

### GitHub Projects
```
<$github-projects org="selfapplied" />
```

### Research Library
```
<$research-library filter="[tag[Research]]" />
```

## Customization

### Colors
Edit CSS variables in `$:/themes/awildsort/styles`:
- `--aws-starlight` - Primary text color
- `--aws-deepspace` - Background color
- `--aws-aurora-blue` - Accent colors
- And more...

### Fractal Background
Set `$:/plugins/awildsort/fractal-background/intensity` (0.0 - 1.0)

## Browser Support

- Modern browsers with WebGL support
- Responsive design (mobile/tablet/desktop)
- Graceful degradation if WebGL unavailable

## Performance

- Fractal background pauses on blur
- Optimized WebGL shaders
- Efficient DOM manipulation
- Lazy loading where appropriate

## Documentation

- `SETUP.md` - Detailed setup instructions
- `IMPLEMENTATION_SUMMARY.md` - Technical overview
- Plugin `readme.tid` files - Widget usage guides

## License

Part of TiddlyWiki5 ecosystem. See main TiddlyWiki license.

## Credits

Designed for A Wild Sort LLC
Built on TiddlyWiki5

