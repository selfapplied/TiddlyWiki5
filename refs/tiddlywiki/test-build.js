#!/usr/bin/env node

// Test script to verify markdown deserializer is working
var fs = require("fs");
var path = require("path");

// First, let's test the deserializer directly with one of the actual files
var testFile = path.join(__dirname, "editions/golden-bridge/tiddlers/Golden Bridge Hypothesis.md");
var fileContent = fs.readFileSync(testFile, "utf8");

console.log("Testing deserializer with actual file:");
console.log("File content (first 300 chars):");
console.log(fileContent.substring(0, 300));
console.log("\n---\n");

// Test the deserializer function
function testDeserializer(text, fields) {
	var lines = text.split(/\r?\n/mg);
	var metadataEnd = -1;
	var metadata = {};
	
	for(var i = 0; i < lines.length; i++) {
		var line = lines[i].trim();
		
		if(line === "" && i === 0) {
			continue;
		}
		
		var colonIndex = line.indexOf(":");
		if(colonIndex > 0) {
			var key = line.substring(0, colonIndex).trim();
			var value = line.substring(colonIndex + 1).trim();
			
			if(/^[a-zA-Z0-9_-]+$/.test(key)) {
				metadata[key] = value;
				metadataEnd = i;
				continue;
			}
		}
		
		if(line === "" && metadataEnd >= 0) {
			metadataEnd = i;
			break;
		}
		
		if(line !== "" && metadataEnd < 0) {
			break;
		}
	}
	
	var content = text;
	if(metadataEnd >= 0) {
		content = lines.slice(metadataEnd + 1).join("\n");
	}
	
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

var results = testDeserializer(fileContent, {});
console.log("Deserializer results:");
console.log(JSON.stringify(results, null, 2));

// Now let's try building the wiki
console.log("\n---\n");
console.log("Building wiki...");

var { execSync } = require("child_process");
try {
    execSync("node boot/boot.js editions/golden-bridge --output output --build index", { 
        stdio: 'inherit',
        cwd: __dirname 
    });
    
    // Check if the output file was created
    var outputFile = path.join(__dirname, "editions/golden-bridge/output/index.html");
    if (fs.existsSync(outputFile)) {
        console.log("\nOutput file created successfully!");
        
        // Check if our content is in the output
        var outputContent = fs.readFileSync(outputFile, "utf8");
        if (outputContent.includes("Golden Bridge Hypothesis")) {
            console.log("✅ Content found in output!");
        } else {
            console.log("❌ Content not found in output");
        }
        
        if (outputContent.includes("hypothesis research mathematics computation")) {
            console.log("✅ Tags found in output!");
        } else {
            console.log("❌ Tags not found in output");
        }
    } else {
        console.log("❌ Output file not created");
    }
} catch (error) {
    console.error("Build failed:", error.message);
} 