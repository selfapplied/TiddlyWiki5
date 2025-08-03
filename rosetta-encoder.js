/*\
Rosetta Bitmask Character Encoder
Transforms any character into filesystem-safe equivalents using XOR operations

Theory: If we XOR with a carefully chosen bitmask, we can map unsafe chars 
into a safe alphabet while preserving uniqueness for perfect round-trips.
\*/

"use strict";

class RosettaEncoder {
    constructor() {
        // Define our safe filesystem alphabet (64 chars = 6 bits)
        this.safeAlphabet = 
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';
        
        // Character code range: 0-127 (7 bits for ASCII)
        // We'll map to 6-bit safe alphabet with overflow handling
        this.alphabetSize = this.safeAlphabet.length; // 64
        
        // XOR mask - chosen to distribute characters evenly
        // Using prime numbers for better distribution
        this.xorMask = 0x2D; // 45 in decimal (prime number)
        
        // Overflow mask to ensure we stay in alphabet range
        this.overflowMask = this.alphabetSize - 1; // 63 (0b111111)
        
        console.log('Rosetta Encoder initialized:');
        console.log('Safe alphabet size:', this.alphabetSize);
        console.log('XOR mask:', this.xorMask.toString(16));
        console.log('Overflow mask:', this.overflowMask.toString(2));
    }
    
    /**
     * Encode a single character using the rosetta bitmask
     */
    encodeChar(char) {
        const charCode = char.charCodeAt(0);
        
        // Apply XOR transformation
        const xorResult = charCode ^ this.xorMask;
        
        // Apply overflow mask to keep in safe range
        const safeIndex = xorResult & this.overflowMask;
        
        return this.safeAlphabet[safeIndex];
    }
    
    /**
     * Decode a single character back to original
     */
    decodeChar(safeChar) {
        const safeIndex = this.safeAlphabet.indexOf(safeChar);
        if (safeIndex === -1) {
            throw new Error(`Invalid safe character: ${safeChar}`);
        }
        
        // Reverse the overflow mask (it's just an AND, so we need to try values)
        // This is the tricky part - we need to find original char code
        for (let originalCode = 0; originalCode < 256; originalCode++) {
            const testXor = originalCode ^ this.xorMask;
            const testIndex = testXor & this.overflowMask;
            
            if (testIndex === safeIndex) {
                return String.fromCharCode(originalCode);
            }
        }
        
        throw new Error(`Cannot decode character: ${safeChar}`);
    }
    
    /**
     * Encode a full string (like a tiddler title)
     */
    encode(str) {
        return str.split('').map(char => this.encodeChar(char)).join('');
    }
    
    /**
     * Decode a full string back to original
     */
    decode(safeStr) {
        return safeStr.split('').map(char => this.decodeChar(char)).join('');
    }
    
    /**
     * Test the encoder with common TiddlyWiki patterns
     */
    runTests() {
        const testCases = [
            '$:/core/modules/widgets/button.js',
            '$:/language/Docs/Types/image/svg+xml',
            '$:/config/BitmapEditor/Colour',
            'HelloWorld',
            'test@domain.com',
            'file~with_special-chars.ext',
            'SimpleTitle'
        ];
        
        console.log('\n=== Rosetta Encoding Tests ===');
        
        for (const testCase of testCases) {
            try {
                const encoded = this.encode(testCase);
                const decoded = this.decode(encoded);
                const success = decoded === testCase;
                
                console.log(`\nOriginal: ${testCase}`);
                console.log(`Encoded:  ${encoded}`);
                console.log(`Decoded:  ${decoded}`);
                console.log(`Success:  ${success ? '✅' : '❌'}`);
                
                if (!success) {
                    console.log(`ERROR: Round-trip failed!`);
                }
            } catch (error) {
                console.log(`\nOriginal: ${testCase}`);
                console.log(`ERROR: ${error.message}`);
            }
        }
    }
    
    /**
     * Analyze character distribution
     */
    analyzeDistribution() {
        console.log('\n=== Character Distribution Analysis ===');
        
        // Test all printable ASCII characters
        const charCounts = new Array(this.alphabetSize).fill(0);
        
        for (let i = 32; i < 127; i++) { // Printable ASCII range
            const char = String.fromCharCode(i);
            const encoded = this.encodeChar(char);
            const index = this.safeAlphabet.indexOf(encoded);
            charCounts[index]++;
        }
        
        console.log('Character mapping distribution:');
        console.log('Min mappings:', Math.min(...charCounts));
        console.log('Max mappings:', Math.max(...charCounts));
        console.log('Avg mappings:', (charCounts.reduce((a,b) => a+b) / charCounts.length).toFixed(2));
        
        // Show some examples
        console.log('\nSample character mappings:');
        ['$', ':', '/', '+', '@', '~', 'A', 'z', '0', '9'].forEach(char => {
            const encoded = this.encodeChar(char);
            console.log(`'${char}' (${char.charCodeAt(0)}) → '${encoded}'`);
        });
    }
}

// Run the tests
if (typeof module !== 'undefined' && require.main === module) {
    const encoder = new RosettaEncoder();
    encoder.runTests();
    encoder.analyzeDistribution();
}

module.exports = RosettaEncoder;