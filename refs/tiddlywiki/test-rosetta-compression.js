const RosettaEncoderV2 = require('./boot/rosetta-encoder.js');
const fs = require('fs');

// Load the empty TiddlyWiki HTML
const htmlContent = fs.readFileSync('./editions/empty/output/index.html', 'utf8');

// Extract JavaScript content
const scriptMatch = htmlContent.match(/<script[^>]*>([\s\S]*?)<\/script>/);
const jsContent = scriptMatch ? scriptMatch[1] : '';

console.log('Empty TiddlyWiki Analysis:');
console.log('Total HTML size:', htmlContent.length, 'bytes');
console.log('JavaScript content size:', jsContent.length, 'bytes');
console.log('JavaScript percentage:', ((jsContent.length / htmlContent.length) * 100).toFixed(1) + '%');

// Test rosetta encoder on JavaScript content
const encoder = new RosettaEncoderV2();

// Test on first 10KB of JavaScript
const testJs = jsContent.substring(0, 10000);
console.log('\nTesting on first 10KB of JavaScript:');
console.log('Original size:', testJs.length, 'bytes');

// Fixed-width encoding
const fixedEncoded = encoder.encode(testJs);
console.log('Fixed-width encoded size:', fixedEncoded.length, 'bytes');
console.log('Fixed-width ratio:', (fixedEncoded.length / testJs.length).toFixed(2) + 'x');

// Variable-length encoding
const variableEncoded = encoder.encodeWithSeparator(testJs);
console.log('Variable-length encoded size:', variableEncoded.length, 'bytes');
console.log('Variable-length ratio:', (variableEncoded.length / testJs.length).toFixed(2) + 'x');

// Test round-trip
const fixedDecoded = encoder.decode(fixedEncoded);
const variableDecoded = encoder.decodeWithSeparator(variableEncoded);

console.log('\nRound-trip verification:');
console.log('Fixed-width round-trip:', fixedDecoded === testJs ? '✅' : '❌');
console.log('Variable-length round-trip:', variableDecoded === testJs ? '✅' : '❌');

// Analyze character distribution
const charFreq = {};
for (let char of testJs) {
    charFreq[char] = (charFreq[char] || 0) + 1;
}

const sortedChars = Object.entries(charFreq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);

console.log('\nTop 10 characters in JavaScript:');
sortedChars.forEach(([char, count]) => {
    const encoded = encoder.encodeChar(char);
    const ratio = encoded.length / char.length;
    console.log(`'${char}' (${count} times) → "${encoded}" (${ratio.toFixed(2)}x)`);
});

// Calculate theoretical compression for full file
const fullFixedSize = Math.ceil(jsContent.length * 2.0); // Fixed-width is ~2.0x
const fullVariableSize = Math.ceil(jsContent.length * 2.74); // Variable-length is ~2.74x

console.log('\nTheoretical full file compression:');
console.log('Original JavaScript:', jsContent.length, 'bytes');
console.log('Fixed-width encoded:', fullFixedSize, 'bytes');
console.log('Variable-length encoded:', fullVariableSize, 'bytes');
console.log('Fixed-width would make file:', ((fullFixedSize + (htmlContent.length - jsContent.length)) / 1024 / 1024).toFixed(1), 'MB');
console.log('Variable-length would make file:', ((fullVariableSize + (htmlContent.length - jsContent.length)) / 1024 / 1024).toFixed(1), 'MB'); 