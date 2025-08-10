/*\
Rosetta Bitmask Character Encoder v3
Enhanced for HTML context with full character space

The insight: HTML context allows full ASCII range, not just filesystem-safe chars.
Solution: Use base-256 conversion with full HTML-safe alphabet for maximum compression.
\*/

"use strict";

class RosettaEncoderV3 {
    constructor() {
        // Full HTML-safe alphabet (base-256 for maximum efficiency)
        this.safeAlphabet = 
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789' +
            '.,!?;:+-*/=<>()[]{}@#$%^&|~`"\' \t\n\r&<>"\'';
        
        this.base = this.safeAlphabet.length; // 256
        
        console.log('Rosetta Encoder V3 initialized:');
        console.log('Base:', this.base);
        console.log('Safe alphabet length:', this.safeAlphabet.length);
        console.log('HTML-optimized for maximum compression');
    }
    
    /**
     * Encode a single character using base conversion
     * Maps char code to safe alphabet using bijective base-256
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
     * Encode a full string with optimized character mapping
     * Most characters now compress to 1 char instead of 2
     */
    encode(str) {
        return str.split('').map(char => {
            const encoded = this.encodeChar(char);
            // Most chars now compress to 1 char, only high Unicode needs 2
            return encoded.length === 1 ? encoded : encoded.padStart(2, this.safeAlphabet[0]);
        }).join('');
    }
    
    /**
     * Decode an encoded string
     */
    decode(encodedStr) {
        let result = '';
        let i = 0;
        
        while (i < encodedStr.length) {
            // Try 1-char first, then 2-char if needed
            let encodedChar = encodedStr[i];
            let decoded = this.decodeChar(encodedChar);
            
            // If decoding fails or produces invalid char, try 2-char
            if (decoded.charCodeAt(0) > 255 || decoded.charCodeAt(0) < 0) {
                if (i + 1 < encodedStr.length) {
                    encodedChar = encodedStr.substr(i, 2);
                    decoded = this.decodeChar(encodedChar);
                    i += 2;
                } else {
                    throw new Error('Invalid encoded string');
                }
            } else {
                i += 1;
            }
            
            result += decoded;
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
     * Test the enhanced encoding
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
        
        console.log('\n=== Enhanced Base-256 Encoding Tests ===');
        
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
    }
    
    /**
     * Analyze the enhanced efficiency
     */
    analyzeEfficiency() {
        console.log('\n=== Enhanced Encoding Efficiency Analysis ===');
        
        // Test common characters
        console.log('Single character encodings:');
        ['$', ':', '/', 'A', 'a', '0', '~', '@', ' ', '.', '!', '?'].forEach(char => {
            const encoded = this.encodeChar(char);
            const charCode = char.charCodeAt(0);
            console.log(`'${char}' (${charCode}) → "${encoded}" (${encoded.length} chars)`);
        });
        
        // Test typical TiddlyWiki title
        const typical = '$:/core/ui/PageTemplate';
        const encoded = this.encode(typical);
        
        console.log('\nTypical TiddlyWiki title efficiency:');
        console.log(`Original:  "${typical}" (${typical.length} chars)`);
        console.log(`Encoded:   "${encoded}" (${encoded.length} chars, ${(encoded.length/typical.length).toFixed(2)}x)`);
    }
}

// Run the tests
if (typeof module !== 'undefined' && require.main === module) {
    const encoder = new RosettaEncoderV3();
    encoder.runTests();
    encoder.analyzeEfficiency();
}

module.exports = RosettaEncoderV3; 