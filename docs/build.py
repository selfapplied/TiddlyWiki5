#!/usr/bin/env python3
"""
Simple tool to convert markdown files to TiddlyWiki tiddlers and build the wiki.
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def convert_markdown_to_tiddler(file_path, docs_root):
    """Convert a markdown file to tiddler format."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Get relative path for labels
    rel_path = file_path.relative_to(docs_root)
    
    # Extract title from first heading or filename
    title = file_path.stem.replace('-', ' ').replace('_', ' ').title()
    for line in content.split('\n'):
        if line.startswith('#'):
            title = line.lstrip('#').strip()
            break
    
    # Generate labels from subdirectory structure
    labels = []
    for part in rel_path.parent.parts:
        if part != '.':
            labels.append(part)
    
    # Create tiddler content
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    tags_str = ' '.join(labels) if labels else ''
    
    tiddler = f"""title: {title}
type: text/markdown
created: {timestamp}
modified: {timestamp}"""
    
    if tags_str:
        tiddler += f"\ntags: {tags_str}"
    
    tiddler += f"\n\n{content}"
    
    return title, tiddler

def build_wiki():
    """Main function to build the wiki."""
    docs_dir = Path('.')
    output_dir = docs_dir / '.out'
    tiddlers_dir = output_dir / 'tiddlers'
    
    # Create output directory
    tiddlers_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all markdown files
    markdown_files = list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.markdown"))
    
    print(f"Found {len(markdown_files)} markdown files")
    
    # Convert each markdown file to tiddler
    for md_file in markdown_files:
        print(f"Converting: {md_file.name}")
        title, tiddler_content = convert_markdown_to_tiddler(md_file, docs_dir)
        
        # Write tiddler file
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
        tiddler_file = tiddlers_dir / f"{safe_title}.tid"
        
        with open(tiddler_file, 'w', encoding='utf-8') as f:
            f.write(tiddler_content)
    
    print(f"Converted {len(markdown_files)} files to tiddlers")
    
    # Build the wiki using TiddlyWiki
    print("Building TiddlyWiki...")
    try:
        # Use the TiddlyWiki from refs/tiddlywiki/tiddlywiki.js
        tw_path = Path('../refs/tiddlywiki/tiddlywiki.js')
        if tw_path.exists():
            subprocess.run(['node', str(tw_path), '.', '--output', str(output_dir / 'wiki'), '--build', 'index'], 
                         cwd=docs_dir, check=True)
            print("Wiki built successfully!")
            print(f"Open {output_dir/'wiki'/'index.html'} in your browser")
        else:
            print("TiddlyWiki not found at ../refs/tiddlywiki/tiddlywiki.js")
            print("Falling back to npx tiddlywiki...")
            subprocess.run(['npx', 'tiddlywiki', '.', '--output', str(output_dir / 'wiki'), '--build', 'index'], 
                         cwd=docs_dir, check=True)
            print("Wiki built successfully!")
            print(f"Open {output_dir/'wiki'/'index.html'} in your browser")
    except subprocess.CalledProcessError as e:
        print(f"Error building wiki: {e}")
        return False
    
    return True

if __name__ == '__main__':
    build_wiki()
