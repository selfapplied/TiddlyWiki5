/*\
Recursive Palette Encoder
Like a fractal compression system for TiddlyWiki titles

The insight: Common patterns at different scales can be mapped to increasingly 
compact representations, with the decoding stack providing perfect reconstruction.
\*/

"use strict";

class RecursivePaletteEncoder {
    constructor() {
        this.safeAlphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';
        this.alphabetIndex = 0;
        
        // Build hierarchical palettes
        this.palettes = this.buildPalettes();
        
        console.log('Recursive Palette Encoder initialized');
        console.log('Palettes built:', Object.keys(this.palettes).length);
    }
    
    buildPalettes() {
        const palettes = {};
        
        // Level 1: Single character mappings
        palettes[1] = {
            '$': this.nextChar(),  // A
            ':': this.nextChar(),  // B  
            '/': this.nextChar(),  // C
            '-': this.nextChar(),  // D
            '_': this.nextChar(),  // E
            '.': this.nextChar(),  // F
            '~': this.nextChar(),  // G
            '@': this.nextChar(),  // H
            '+': this.nextChar()   // I
        };
        
        // Level 2: Common sequences (2-4 chars → 1 char)
        palettes[2] = {
            '$:/': this.nextChar(),           // J
            'core/': this.nextChar(),         // K
            'modules/': this.nextChar(),      // L
            'widgets/': this.nextChar(),      // M
            'images/': this.nextChar(),       // N
            'language/': this.nextChar(),     // O
            'ui/': this.nextChar(),           // P
            'config/': this.nextChar(),       // Q
            'plugins/': this.nextChar(),      // R
            'themes/': this.nextChar(),       // S
            '.js': this.nextChar(),           // T
            '.tid': this.nextChar(),          // U
            '.css': this.nextChar(),          // V
            '.html': this.nextChar()          // W
        };
        
        // Level 3: Full namespace prefixes (long patterns → 1 char)
        palettes[3] = {
            '$:/core/': this.nextChar(),              // X
            '$:/core/modules/': this.nextChar(),      // Y
            '$:/core/modules/widgets/': this.nextChar(), // Z
            '$:/core/ui/': this.nextChar(),           // a
            '$:/core/images/': this.nextChar(),       // b
            '$:/language/': this.nextChar(),          // c
            '$:/config/': this.nextChar(),            // d
            '$:/plugins/tiddlywiki/': this.nextChar() // e
        };
        
        return palettes;
    }
    
    nextChar() {
        return this.safeAlphabet[this.alphabetIndex++];
    }
    
    /**
     * Encode with recursive palette matching (greedy longest match)
     */
    encode(input) {
        const encodingStack = [];
        let encoded = '';
        let pos = 0;
        
        while (pos < input.length) {
            const match = this.findLongestMatch(input, pos);
            
            if (match) {
                encoded += match.encoded;
                encodingStack.push({
                    level: match.level,
                    original: match.original,
                    encoded: match.encoded,
                    length: match.encoded.length,
                    originalLength: match.original.length
                });
                pos += match.original.length;
            } else {
                // No palette match, encode single character
                const char = input[pos];
                const encodedChar = this.encodeBaseChar(char);
                encoded += encodedChar;
                encodingStack.push({
                    level: 0,
                    original: char,
                    encoded: encodedChar,
                    length: encodedChar.length,
                    originalLength: 1
                });
                pos += 1;
            }
        }
        
        return {
            encoded,
            stack: encodingStack,
            compression: input.length / encoded.length
        };
    }
    
    /**
     * Find the longest matching pattern across all palette levels
     */
    findLongestMatch(input, pos) {
        let bestMatch = null;
        
        // Check palettes from highest level (most compression) to lowest
        for (let level = 3; level >= 1; level--) {
            const palette = this.palettes[level];
            
            for (const [pattern, encoded] of Object.entries(palette)) {
                if (input.substr(pos, pattern.length) === pattern) {
                    if (!bestMatch || pattern.length > bestMatch.original.length) {
                        bestMatch = {
                            level,
                            original: pattern,
                            encoded,
                            originalLength: pattern.length
                        };
                    }
                }
            }
        }
        
        return bestMatch;
    }
    
    /**
     * Base character encoding (for non-palette characters)
     */
    encodeBaseChar(char) {
        const code = char.charCodeAt(0);
        if (code < 64) {
            return this.safeAlphabet[code];
        } else {
            // Use base-64 style encoding for higher codes
            let result = '';
            let remaining = code;
            while (remaining > 0) {
                result = this.safeAlphabet[remaining % 64] + result;
                remaining = Math.floor(remaining / 64);
            }
            return result;
        }
    }
    
    /**
     * Decode using the encoding stack
     */
    decode(encoded, stack) {
        let pos = 0;
        let result = '';
        
        for (const step of stack) {
            const chunk = encoded.substr(pos, step.length);
            
            if (step.level === 0) {
                // Base character decoding
                result += this.decodeBaseChar(chunk);
            } else {
                // Palette lookup (reverse mapping)
                const palette = this.palettes[step.level];
                const reverseMap = this.getReverseMap(palette);
                result += reverseMap[chunk];
            }
            
            pos += step.length;
        }
        
        return result;
    }
    
    getReverseMap(palette) {
        const reverse = {};
        for (const [key, value] of Object.entries(palette)) {
            reverse[value] = key;
        }
        return reverse;
    }
    
    decodeBaseChar(encoded) {
        if (encoded.length === 1) {
            const index = this.safeAlphabet.indexOf(encoded);
            return String.fromCharCode(index);
        } else {
            // Multi-character base decoding
            let code = 0;
            for (const char of encoded) {
                const digit = this.safeAlphabet.indexOf(char);
                code = code * 64 + digit;
            }
            return String.fromCharCode(code);
        }
    }
    
    /**
     * Test the recursive palette system
     */
    runTests() {
        const testCases = [
            '$:/core/modules/widgets/button.js',
            '$:/core/ui/PageTemplate',
            '$:/core/images/close-button',
            '$:/language/Help/help.tid',
            '$:/config/AutoSave',
            '$:/plugins/tiddlywiki/katex/plugin.info',
            'HelloWorld',
            'regular-filename.txt'
        ];
        
        console.log('\n=== Recursive Palette Encoding Tests ===');
        
        for (const testCase of testCases) {
            const result = this.encode(testCase);
            const decoded = this.decode(result.encoded, result.stack);
            const success = decoded === testCase;
            
            console.log(`\nOriginal:    "${testCase}" (${testCase.length} chars)`);
            console.log(`Encoded:     "${result.encoded}" (${result.encoded.length} chars)`);
            console.log(`Compression: ${result.compression.toFixed(2)}x`);
            console.log(`Stack steps: ${result.stack.length}`);
            console.log(`Decoded:     "${decoded}"`);
            console.log(`Success:     ${success ? '✅' : '❌'}`);
            
            if (!success) {
                console.log('ERROR: Round-trip failed!');
                console.log('Stack:', JSON.stringify(result.stack, null, 2));
            }
        }
    }
    
    /**
     * Analyze palette efficiency
     */
    analyzePalettes() {
        console.log('\n=== Palette Analysis ===');
        
        for (let level = 1; level <= 3; level++) {
            const palette = this.palettes[level];
            console.log(`\nLevel ${level} Palette (${Object.keys(palette).length} entries):`);
            
            for (const [pattern, encoded] of Object.entries(palette)) {
                const savings = pattern.length - encoded.length;
                console.log(`  "${pattern}" → "${encoded}" (saves ${savings} chars)`);
            }
        }
    }
}

// Run the tests
if (typeof module !== 'undefined' && require.main === module) {
    const encoder = new RecursivePaletteEncoder();
    encoder.runTests();
    encoder.analyzePalettes();
}

module.exports = RecursivePaletteEncoder;