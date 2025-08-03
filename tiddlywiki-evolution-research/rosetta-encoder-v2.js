/*\
Rosetta Bitmask Character Encoder v2
Bijective encoding ensuring perfect round-trips

The insight: XOR + overflow mask loses information. We need bijective mapping.
Solution: Use base-conversion with safe alphabet for guaranteed reversibility.
\*/

"use strict";

class RosettaEncoderV2 {
    constructor() {
        // Safe filesystem alphabet (ordered for base conversion)
        this.safeAlphabet = 
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';
        
        this.base = this.safeAlphabet.length; // 64
        
        console.log('Rosetta Encoder V2 initialized:');
        console.log('Base:', this.base);
        console.log('Safe alphabet:', this.safeAlphabet);
    }
    
    /**
     * Encode a single character using base conversion
     * Maps char code to safe alphabet using bijective base-64
     */
    encodeChar(char) {
        let charCode = char.charCodeAt(0);
        
        // Handle the bijective base conversion (no zero mapping issues)
        if (charCode === 0) {
            return this.safeAlphabet[0];
        }
        
        let result = '';
        while (charCode > 0) {
            result = this.safeAlphabet[charCode % this.base] + result;
            charCode = Math.floor(charCode / this.base);
        }
        
        return result || this.safeAlphabet[0];
    }
    
    /**
     * Decode a single encoded character back to original
     */
    decodeChar(encodedChar) {
        let charCode = 0;
        
        for (let i = 0; i < encodedChar.length; i++) {
            const digit = this.safeAlphabet.indexOf(encodedChar[i]);
            if (digit === -1) {
                throw new Error(`Invalid character in encoded string: ${encodedChar[i]}`);
            }
            charCode = charCode * this.base + digit;
        }
        
        return String.fromCharCode(charCode);
    }
    
    /**
     * Encode a full string with character boundary markers
     * Problem: Variable-length encoding needs delimiters
     * Solution: Use fixed-width encoding for simplicity
     */
    encode(str) {
        return str.split('').map(char => {
            const encoded = this.encodeChar(char);
            // Pad to fixed width (2 chars handles 0-4095 range)
            return encoded.padStart(2, this.safeAlphabet[0]);
        }).join('');
    }
    
    /**
     * Decode a fixed-width encoded string
     */
    decode(encodedStr) {
        if (encodedStr.length % 2 !== 0) {
            throw new Error('Encoded string length must be even (fixed-width encoding)');
        }
        
        let result = '';
        for (let i = 0; i < encodedStr.length; i += 2) {
            const encodedChar = encodedStr.substr(i, 2);
            result += this.decodeChar(encodedChar);
        }
        
        return result;
    }
    
    /**
     * Alternative: Variable-length with explicit separator
     */
    encodeWithSeparator(str) {
        const separator = this.safeAlphabet[this.safeAlphabet.length - 1]; // Use last char as sep
        return str.split('').map(char => this.encodeChar(char)).join(separator);
    }
    
    decodeWithSeparator(encodedStr) {
        const separator = this.safeAlphabet[this.safeAlphabet.length - 1];
        return encodedStr.split(separator).map(part => this.decodeChar(part)).join('');
    }
    
    /**
     * Test both encoding methods
     */
    runTests() {
        const testCases = [
            '$:/core/modules/widgets/button.js',
            '$:/language/Docs/Types/image/svg+xml',
            'HelloWorld',
            'test@domain.com',
            'A', // Single char test
            '/', // Common separator
            '~', // High ASCII
        ];
        
        console.log('\n=== Fixed-Width Encoding Tests ===');
        
        for (const testCase of testCases) {
            try {
                const encoded = this.encode(testCase);
                const decoded = this.decode(encoded);
                const success = decoded === testCase;
                
                console.log(`\nOriginal: "${testCase}"`);
                console.log(`Encoded:  "${encoded}"`);
                console.log(`Decoded:  "${decoded}"`);
                console.log(`Success:  ${success ? '✅' : '❌'}`);
                console.log(`Ratio:    ${(encoded.length / testCase.length).toFixed(2)}x`);
                
            } catch (error) {
                console.log(`ERROR: ${error.message}`);
            }
        }
        
        console.log('\n=== Variable-Length with Separator Tests ===');
        
        for (const testCase of testCases) {
            try {
                const encoded = this.encodeWithSeparator(testCase);
                const decoded = this.decodeWithSeparator(encoded);
                const success = decoded === testCase;
                
                console.log(`\nOriginal: "${testCase}"`);
                console.log(`Encoded:  "${encoded}"`);
                console.log(`Decoded:  "${decoded}"`);
                console.log(`Success:  ${success ? '✅' : '❌'}`);
                console.log(`Ratio:    ${(encoded.length / testCase.length).toFixed(2)}x`);
                
            } catch (error) {
                console.log(`ERROR: ${error.message}`);
            }
        }
    }
    
    /**
     * Analyze the encoding efficiency
     */
    analyzeEfficiency() {
        console.log('\n=== Encoding Efficiency Analysis ===');
        
        // Test common characters
        console.log('Single character encodings:');
        ['$', ':', '/', 'A', 'a', '0', '~', '@'].forEach(char => {
            const encoded = this.encodeChar(char);
            const charCode = char.charCodeAt(0);
            console.log(`'${char}' (${charCode}) → "${encoded}" (${encoded.length} chars)`);
        });
        
        // Test typical TiddlyWiki title
        const typical = '$:/core/ui/PageTemplate';
        const fixedEncoded = this.encode(typical);
        const variableEncoded = this.encodeWithSeparator(typical);
        
        console.log('\nTypical TiddlyWiki title efficiency:');
        console.log(`Original:  "${typical}" (${typical.length} chars)`);
        console.log(`Fixed:     "${fixedEncoded}" (${fixedEncoded.length} chars, ${(fixedEncoded.length/typical.length).toFixed(2)}x)`);
        console.log(`Variable:  "${variableEncoded}" (${variableEncoded.length} chars, ${(variableEncoded.length/typical.length).toFixed(2)}x)`);
    }
}

// Run the tests
if (typeof module !== 'undefined' && require.main === module) {
    const encoder = new RosettaEncoderV2();
    encoder.runTests();
    encoder.analyzeEfficiency();
}

module.exports = RosettaEncoderV2;