#!/usr/bin/env node

// Test script to understand TiddlyWiki file loading
var fs = require("fs");
var path = require("path");

console.log("Testing TiddlyWiki file loading...");

// Test the metadata loading function
function loadMetadataForFile(filepath) {
	var metafilename = filepath + ".meta";
	if(fs.existsSync(metafilename)) {
		var metadata = fs.readFileSync(metafilename, "utf8");
		console.log("Found metadata file:", metafilename);
		console.log("Metadata content:");
		console.log(metadata);
		return metadata;
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
	console.log("\nMetadata loaded successfully!");
} else {
	console.log("\nNo metadata found.");
}

// Test .tid file
var tidFile = path.join(__dirname, "editions/golden-bridge/tiddlers/Golden Bridge Hypothesis.tid");
console.log("\nTesting .tid file:", tidFile);
console.log("File exists:", fs.existsSync(tidFile));

if (fs.existsSync(tidFile)) {
	var tidContent = fs.readFileSync(tidFile, "utf8");
	console.log("First 200 chars of .tid file:");
	console.log(tidContent.substring(0, 200));
} 