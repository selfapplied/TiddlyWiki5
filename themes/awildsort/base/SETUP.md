# A Wild Sort Theme Setup Guide

This guide will help you transform your TiddlyWiki into the A Wild Sort website.

## Installation

### 1. Enable the Theme

In your TiddlyWiki, go to:
- Control Panel → Appearance → Theme
- Select: `$:/themes/awildsort/base`

### 2. Enable Required Plugins

Go to Control Panel → Plugins and enable:

- `$:/plugins/awildsort/fractal-background` - WebGL fractal background
- `$:/plugins/awildsort/products` - Product display widgets
- `$:/plugins/awildsort/gallery` - Art gallery with lightbox
- `$:/plugins/awildsort/github-projects` - GitHub repository browser
- `$:/plugins/awildsort/research-library` - Research paper library
- `$:/plugins/tiddlywiki/katex` - Math rendering (required for research library)

### 3. Set Homepage

Create or edit a tiddler titled "Home" with tag `$:/tags/HomePage`:

```
title: Home
tags: [[$:/tags/HomePage]]

! A Wild Sort

Welcome content here...

<$product-grid filter="[tag[Product]]" />
<$github-projects org="selfapplied" />
<$research-library filter="[tag[Research]]" />
<$gallery filter="[tag[Gallery]]" />
```

## Creating Content

### Products

Create tiddlers with tag `Product` and fields:

- `image` - Product image URL
- `description` - Short description
- `price` - Price string
- `category` - Category (Education, Consulting, Tools, Research Artifacts)
- `link` - Product link
- `cta` - Call-to-action text (default: "Learn More")

Example:
```
title: My Product
tags: Product
image: /images/product.jpg
description: A powerful tool for researchers
price: $99
category: Tools
link: https://example.com/product
cta: Buy Now
```

### Gallery Items

Create tiddlers with tag `Gallery` and fields:

- `image` - Image URL (or `video` for videos)
- `title` - Artwork title
- `notes` - Artist notes

Example:
```
title: Fractal Nebula
tags: Gallery
image: /images/fractal.jpg
title: Stellar Formation
notes: Generated using CE1 harmonics
```

### Research Papers

Create tiddlers with tag `Research` and fields:

- `authors` - Author names
- `abstract` - Paper abstract
- `pdf` - PDF link
- `arxiv` - arXiv link
- `link` - General link
- `tags` - Topic tags (chaos, renormalization, symbolic-dynamics, CE1, field-equations, fractals)

Example:
```
title: Chaos Theory in Symbolic Dynamics
tags: Research chaos symbolic-dynamics
authors: Jane Doe, John Smith
abstract: This paper explores...
pdf: /papers/chaos.pdf
arxiv: https://arxiv.org/abs/...
```

### GitHub Projects

The GitHub widget automatically fetches repositories. Just use:

```
<$github-projects org="selfapplied" />
```

Replace "selfapplied" with your GitHub organization or username.

## Customization

### Colors

Edit `$:/themes/awildsort/styles` to modify CSS variables:

```css
:root {
	--aws-starlight: #f8f9fa;
	--aws-deepspace: #0a0e27;
	--aws-aurora-blue: #1e3a5f;
	/* ... */
}
```

### Fractal Background Intensity

Set the tiddler `$:/plugins/awildsort/fractal-background/intensity` to a value between 0.0 and 1.0 (default: 0.4).

## Building for Production

To build a static HTML file:

```bash
tiddlywiki yourwiki --build index
```

This creates `output/index.html` with all content bundled.

## Troubleshooting

### Fractal background not showing
- Check browser WebGL support
- Verify plugin is enabled
- Check browser console for errors

### Widgets not rendering
- Ensure all required plugins are enabled
- Check tiddler tags match widget filters
- Verify widget syntax is correct

### Styles not applying
- Clear browser cache
- Verify theme is selected
- Check for CSS conflicts

