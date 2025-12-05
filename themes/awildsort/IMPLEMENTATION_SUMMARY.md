# A Wild Sort Theme - Implementation Summary

## Overview

This implementation transforms TiddlyWiki into a fully-designed website for A Wild Sort LLC, featuring:

- Glass-fog UI with stellar fractal ambiance
- Product ecosystem display
- Art gallery with lightbox
- GitHub projects browser
- Research library with filtering
- Smooth animations and responsive design

## Structure

### Theme (`themes/awildsort/`)

- `plugin.info` - Theme plugin metadata
- `styles.tid` - Main stylesheet with glassmorphism, typography, and responsive design
- `navigation.tid` - Navigation-specific styles
- `scroll-handler.js` - Scroll effects for navigation blur
- `homepage.tid` - Example homepage template
- `readme.tid` - Theme documentation
- `SETUP.md` - Setup guide
- Example tiddlers for products, gallery, and research

### Plugins

#### 1. Fractal Background (`plugins/awildsort/fractal-background/`)

- `plugin.info` - Plugin metadata
- `fractal-background.js` - WebGL startup module with stellar nebula shader
- `readme.tid` - Documentation

Features:
- WebGL-based stellar nebula animation
- Automatic pause on tab/window blur
- Intensity control via tiddler state
- Performance-optimized rendering

#### 2. Products (`plugins/awildsort/products/`)

- `plugin.info` - Plugin metadata
- `product-widget.js` - Individual product card widget
- `product-grid-widget.js` - Product grid widget
- `styles.tid` - Product-specific styles
- `readme.tid` - Documentation

Widgets:
- `<$product>` - Individual product card
- `<$product-grid>` - Responsive product grid

#### 3. Gallery (`plugins/awildsort/gallery/`)

- `plugin.info` - Plugin metadata
- `gallery-widget.js` - Gallery widget with lightbox
- `styles.tid` - Gallery-specific styles
- `readme.tid` - Documentation

Widget:
- `<$gallery>` - Gallery with masonry/grid/carousel layouts

Features:
- Fullscreen lightbox with fade transitions
- Support for images and videos
- Keyboard navigation (Escape to close)
- Hover effects

#### 4. GitHub Projects (`plugins/awildsort/github-projects/`)

- `plugin.info` - Plugin metadata
- `github-widget.js` - GitHub API integration widget
- `readme.tid` - Documentation

Widget:
- `<$github-projects>` - Fetches and displays GitHub repositories

Features:
- Fetches from GitHub API
- Shows stars, update dates, languages
- Direct links to repositories

#### 5. Research Library (`plugins/awildsort/research-library/`)

- `plugin.info` - Plugin metadata
- `research-widget.js` - Research paper library widget
- `styles.tid` - Research-specific styles
- `readme.tid` - Documentation

Widget:
- `<$research-library>` - Research paper library with filtering

Features:
- Topic-based filtering
- Abstract display
- PDF, arXiv, and general links
- KaTeX math support (requires katex plugin)

## Design Philosophy

### Glass-fog UI
- Layered panes with frosted translucency
- Backdrop blur effects
- Soft shadows and gentle contrast
- Subtle borders

### Stellar Fractal Ambiance
- WebGL shader-based animations
- Ambient, non-dominant background
- Automatic pause on blur
- Performance-optimized

### Smooth Animations
- Cubic-bezier easing curves
- 400-900ms durations
- Opacity fades and parallax shifts
- Micro-interactions on hover

### Typography
- Inter (sans-serif) for UI
- Crimson Pro (serif) for headings
- JetBrains Mono (monospace) for code/math

### Color Palette
- Starlight whites
- Deep-space blues
- Aurora gradients
- Feigenbaum greens
- Zeta purples

## Usage

### Basic Setup

1. Enable theme: `$:/themes/awildsort/base`
2. Enable all required plugins
3. Create homepage tiddler with tag `$:/tags/HomePage`
4. Add content tiddlers with appropriate tags

### Content Tags

- `Product` - Products
- `Gallery` - Art gallery items
- `Research` - Research papers

### Widget Usage

```
<$product-grid filter="[tag[Product]]" />
<$gallery filter="[tag[Gallery]]" layout="grid" />
<$github-projects org="selfapplied" />
<$research-library filter="[tag[Research]]" />
```

## Customization

### CSS Variables

Edit `$:/themes/awildsort/styles` to customize:
- Colors (CSS variables in `:root`)
- Spacing and layout
- Animation durations
- Typography

### Fractal Background

Control intensity via:
`$:/plugins/awildsort/fractal-background/intensity` (0.0 - 1.0)

## Browser Support

- Modern browsers with WebGL support
- Responsive design for mobile/tablet/desktop
- Graceful degradation if WebGL unavailable

## Performance

- Fractal background pauses on blur
- Lazy loading for gallery images
- Optimized WebGL shaders
- Efficient DOM manipulation

## Future Enhancements

Potential additions:
- Additional fractal modes (CE1 harmonics, Mandelbulb)
- More gallery layout options
- Enhanced research paper viewer
- Custom navigation components
- Additional product display formats


