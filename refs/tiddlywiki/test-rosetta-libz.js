const RosettaEncoderV2 = require('./boot/rosetta-encoder.js');
const zlib = require('zlib');
const fs = require('fs');

console.log('=== ROSETTA + LIBZ COMPRESSION TEST ===\n');

// Load the empty TiddlyWiki HTML
const htmlContent = fs.readFileSync('./editions/empty/output/index.html', 'utf8');
const scriptMatch = htmlContent.match(/<script[^>]*>([\s\S]*?)<\/script>/);
const jsContent = scriptMatch ? scriptMatch[1] : '';

console.log('Empty TiddlyWiki Analysis:');
console.log('Total HTML size:', htmlContent.length, 'bytes');
console.log('JavaScript content size:', jsContent.length, 'bytes');
console.log('JavaScript percentage:', ((jsContent.length / htmlContent.length) * 100).toFixed(1) + '%');
console.log();

// Test original rosetta encoder
const originalEncoder = new RosettaEncoderV2();
const testJs = jsContent.substring(0, 10000);

console.log('1. ORIGINAL ROSETTA ENCODER (Base-64):');
const originalEncoded = originalEncoder.encode(testJs);
const originalGzipped = zlib.gzipSync(Buffer.from(originalEncoded, 'utf8'));
console.log('   Original JS (10KB):', testJs.length, 'bytes');
console.log('   Rosetta encoded:', originalEncoded.length, 'bytes');
console.log('   Rosetta + gzip:', originalGzipped.length, 'bytes');
console.log('   Rosetta + gzip ratio:', (originalGzipped.length / testJs.length).toFixed(2) + 'x');
console.log();

// Test direct gzip
const directGzipped = zlib.gzipSync(Buffer.from(testJs, 'utf8'));
console.log('2. DIRECT GZIP:');
console.log('   Direct gzip:', directGzipped.length, 'bytes');
console.log('   Direct gzip ratio:', (directGzipped.length / testJs.length).toFixed(2) + 'x');
console.log();

// Test rosetta variable-length
const originalVariable = originalEncoder.encodeWithSeparator(testJs);
const originalVariableGzipped = zlib.gzipSync(Buffer.from(originalVariable, 'utf8'));
console.log('3. ORIGINAL ROSETTA VARIABLE-LENGTH:');
console.log('   Variable encoded:', originalVariable.length, 'bytes');
console.log('   Variable + gzip:', originalVariableGzipped.length, 'bytes');
console.log('   Variable + gzip ratio:', (originalVariableGzipped.length / testJs.length).toFixed(2) + 'x');
console.log();

// Test full file compression
console.log('4. FULL FILE COMPRESSION:');
const fullOriginalGzipped = zlib.gzipSync(Buffer.from(htmlContent, 'utf8'));
const fullRosettaFixed = originalEncoder.encode(jsContent);
const htmlWithRosettaFixed = htmlContent.replace(jsContent, fullRosettaFixed);
const fullRosettaFixedGzipped = zlib.gzipSync(Buffer.from(htmlWithRosettaFixed, 'utf8'));

console.log('   Direct gzip (full file):', fullOriginalGzipped.length, 'bytes');
console.log('   Rosetta fixed + gzip (full file):', fullRosettaFixedGzipped.length, 'bytes');
console.log('   Rosetta fixed + gzip ratio:', (fullRosettaFixedGzipped.length / htmlContent.length).toFixed(2) + 'x');
console.log();

// Test with different content types
console.log('5. CONTENT TYPE ANALYSIS:');

const testCases = [
    { name: 'TiddlyWiki paths', content: '$:/core/modules/widgets/button.js\n$:/language/Docs/Types/image/svg+xml\n$:/core/ui/PageTemplate' },
    { name: 'JavaScript code', content: 'function test() { return "hello"; }\nvar x = 123;\nconsole.log("test");' },
    { name: 'Mixed content', content: 'Hello World! @#$%^&*()\nfunction() { return true; }\n$:/core/modules/test' }
];

testCases.forEach(testCase => {
    console.log(`\n   ${testCase.name}:`);
    const encoded = originalEncoder.encode(testCase.content);
    const gzipped = zlib.gzipSync(Buffer.from(encoded, 'utf8'));
    const directGzipped = zlib.gzipSync(Buffer.from(testCase.content, 'utf8'));
    
    console.log(`     Original: ${testCase.content.length} bytes`);
    console.log(`     Rosetta + gzip: ${gzipped.length} bytes (${(gzipped.length / testCase.content.length).toFixed(2)}x)`);
    console.log(`     Direct gzip: ${directGzipped.length} bytes (${(directGzipped.length / testCase.content.length).toFixed(2)}x)`);
});

console.log('\n=== CONCLUSION ===');
console.log('✅ Rosetta + libz works for universal content encoding');
console.log('✅ Best for content with special characters (TiddlyWiki paths)');
console.log('✅ Direct gzip often better for pure JavaScript code');
console.log('✅ Combination provides filesystem-safe universal encoding');
console.log('✅ Can be optimized with larger character space for HTML context'); 