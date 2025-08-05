#!/usr/bin/env node

// Simple test for markdown deserializer
var fs = require("fs");
var path = require("path");

// Test the deserializer function directly
function testDeserializer(text, fields) {
	// Parse metadata from the top of the markdown file
	var lines = text.split(/\r?\n/mg);
	var metadataEnd = -1;
	var metadata = {};
	
	// Look for metadata at the top of the file
	for(var i = 0; i < lines.length; i++) {
		var line = lines[i].trim();
		
		// Skip empty lines at the beginning
		if(line === "" && i === 0) {
			continue;
		}
		
		// Check if this line looks like metadata (key: value format)
		var colonIndex = line.indexOf(":");
		if(colonIndex > 0) {
			var key = line.substring(0, colonIndex).trim();
			var value = line.substring(colonIndex + 1).trim();
			
			// Only accept simple field names (no spaces, special chars)
			if(/^[a-zA-Z0-9_-]+$/.test(key)) {
				metadata[key] = value;
				metadataEnd = i;
				continue;
			}
		}
		
		// If we hit a blank line after metadata, that's the end
		if(line === "" && metadataEnd >= 0) {
			metadataEnd = i;
			break;
		}
		
		// If we hit content (not metadata), stop looking
		if(line !== "" && metadataEnd < 0) {
			break;
		}
	}
	
	// Extract the content (everything after metadata)
	var content = text;
	if(metadataEnd >= 0) {
		content = lines.slice(metadataEnd + 1).join("\n");
	}
	
	// Merge metadata with provided fields
	var result = {};
	for(var f in fields) {
		result[f] = fields[f];
	}
	for(var m in metadata) {
		result[m] = metadata[m];
	}
	result.text = content;
	
	return [result];
}

// Test with sample markdown
var testMarkdown = `title: Test Tiddler
type: text/markdown
tags: test tag1 tag2
created: 20241220
modified: 20241220

# Test Content

This is a test markdown file with metadata.
`;

console.log("Testing markdown deserializer...");
console.log("Input markdown:");
console.log(testMarkdown);
console.log("\n---\n");

var results = testDeserializer(testMarkdown, {});
console.log("Deserializer results:");
console.log(JSON.stringify(results, null, 2));

// Test with actual file
var testFile = path.join(__dirname, "editions/golden-bridge/tiddlers/Golden Bridge Hypothesis.md");
if (fs.existsSync(testFile)) {
    console.log("\n---\n");
    console.log("Testing with actual file:");
    var fileContent = fs.readFileSync(testFile, "utf8");
    console.log("File content (first 200 chars):");
    console.log(fileContent.substring(0, 200));
    console.log("\n---\n");
    
    var fileResults = testDeserializer(fileContent, {});
    console.log("File deserializer results:");
    console.log(JSON.stringify(fileResults, null, 2));
} 