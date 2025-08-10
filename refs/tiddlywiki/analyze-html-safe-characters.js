console.log('=== HTML-SAFE CHARACTER ANALYSIS ===\n');

console.log('1. HTML CONTEXT SAFETY:');
console.log('   Since this is encoded within HTML files, not filenames:');
console.log('   - We can use ANY character that\'s safe in HTML content');
console.log('   - No filesystem restrictions');
console.log('   - No URL encoding concerns');
console.log('   - Much larger character space available');
console.log();

console.log('2. HTML-SAFE CHARACTER CATEGORIES:');

const htmlSafeCategories = {
    'Basic ASCII': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
    'Common punctuation': '.,!?;:',
    'Math symbols': '+-*/=<>',
    'Brackets': '()[]{}',
    'Other symbols': '@#$%^&|~`',
    'Quotes': '"\'',
    'Spaces and tabs': ' \t',
    'Special HTML chars': '&<>"\'',
    'Control chars': '\n\r'
};

Object.entries(htmlSafeCategories).forEach(([name, chars]) => {
    console.log(`   ${name}: "${chars}" (${chars.length} chars)`);
});

console.log();
console.log('3. CHARACTER RESTRICTIONS IN HTML:');

console.log('   Characters that need escaping in HTML:');
console.log('   - < → &lt;');
console.log('   - > → &gt;');
console.log('   - & → &amp;');
console.log('   - " → &quot;');
console.log('   - \' → &#39;');
console.log();

console.log('4. EXPANDED ALPHABET OPPORTUNITIES:');

// Calculate total safe characters
const allHtmlSafe = Object.values(htmlSafeCategories).join('');
const uniqueHtmlSafe = [...new Set(allHtmlSafe)];

console.log('   Total HTML-safe characters:', uniqueHtmlSafe.length);
console.log('   Unique characters:', uniqueHtmlSafe.join(''));
console.log();

console.log('5. BASE EXPANSION POSSIBILITIES:');

const bases = [64, 128, 256, 512, 1024, 2048];
bases.forEach(base => {
    const maxValue = Math.pow(base, 2) - 1;
    const bitRange = Math.ceil(Math.log2(maxValue));
    console.log(`   Base ${base}: 0-${maxValue} (${bitRange}-bit range)`);
});

console.log();
console.log('6. CURRENT VS POTENTIAL:');

const currentBase = 64;
const potentialBase = 256; // Much more realistic for HTML context

console.log(`   Current base-${currentBase}:`);
console.log(`     - Characters: 64`);
console.log(`     - Range: 0-4095`);
console.log(`     - Efficiency: Many chars expand to 2 chars`);

console.log(`   Potential base-${potentialBase}:`);
console.log(`     - Characters: 256`);
console.log(`     - Range: 0-65535`);
console.log(`     - Efficiency: Most chars compress to 1 char`);

console.log();
console.log('7. IMPLEMENTATION RECOMMENDATIONS:');

console.log('   For HTML context, we can use:');
console.log('   ✅ All printable ASCII (32-126)');
console.log('   ✅ Extended ASCII (128-255)');
console.log('   ✅ Unicode characters (if needed)');
console.log('   ✅ Special symbols and punctuation');
console.log('   ✅ Math symbols and operators');
console.log();

console.log('   Recommended expansion:');
console.log('   - Base-256 alphabet (full ASCII range)');
console.log('   - Much better compression ratios');
console.log('   - Larger encoding range');
console.log('   - More efficient character mapping');

console.log();
console.log('=== CONCLUSION ===');
console.log('✅ HTML context allows MUCH larger character space');
console.log('✅ No filesystem restrictions apply');
console.log('✅ Can use full ASCII range (256 characters)');
console.log('✅ Dramatically better compression possible');
console.log('✅ Current 64-char limit is unnecessarily restrictive'); 