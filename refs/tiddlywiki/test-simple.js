#!/usr/bin/env node

// Simple test to verify .meta file loading
var fs = require("fs");
var path = require("path");

console.log("Testing .meta file loading...");

// Test the metadata loading function from boot.js
function loadMetadataForFile(filepath) {
	var metafilename = filepath + ".meta";
	if(fs.existsSync(metafilename)) {
		var metadata = fs.readFileSync(metafilename, "utf8");
		console.log("Found metadata file:", metafilename);
		console.log("Metadata content:");
		console.log(metadata);
		
		// Parse the metadata (simple key: value format)
		var lines = metadata.split(/\r?\n/mg);
		var parsed = {};
		for(var i = 0; i < lines.length; i++) {
			var line = lines[i].trim();
			if(line === "") continue;
			
			var colonIndex = line.indexOf(":");
			if(colonIndex > 0) {
				var key = line.substring(0, colonIndex).trim();
				var value = line.substring(colonIndex + 1).trim();
				parsed[key] = value;
			}
		}
		
		console.log("Parsed metadata:");
		console.log(JSON.stringify(parsed, null, 2));
		return parsed;
	} else {
		console.log("No metadata file found for:", filepath);
		return null;
	}
}

// Test with our files
var testFile = path.join(__dirname, "editions/golden-bridge/tiddlers/Golden Bridge Hypothesis.md");
console.log("\nTesting file:", testFile);
console.log("File exists:", fs.existsSync(testFile));

var metadata = loadMetadataForFile(testFile);
if (metadata) {
	console.log("\n✅ Metadata loaded successfully!");
	console.log("Title:", metadata.title);
	console.log("Tags:", metadata.tags);
} else {
	console.log("\n❌ No metadata found.");
} 