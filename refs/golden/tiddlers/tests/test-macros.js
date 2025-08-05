/*\
title: test-macros.js
type: application/javascript
tags: [[$:/tags/test-spec]]

Tests our custom macros

\*/

"use strict";

describe("Macro Tests", function() {
    
    it("should define and use test-macro", function() {
        var wiki = new $tw.Wiki();
        
        // Add our test macro tiddler
        wiki.addTiddler({
            title: "Test Macro",
            text: "\\define test-macro(name:\"Default\") Hello, I'm $name$",
            type: "text/vnd.tiddlywiki",
            tags: "macro test"
        });
        
        // Test the macro with default parameter
        var result = wiki.renderText("text/plain", "{{Test Macro}}", {parseAsInline: true});
        expect(result).toBe("Hello, I'm Default");
        
        // Test with custom parameter
        var result2 = wiki.renderText("text/plain", "<<test-macro \"Alice\">>", {parseAsInline: true});
        expect(result2).toBe("Hello, I'm Alice");
    });
    
    it("should define and use inline-note macro", function() {
        var wiki = new $tw.Wiki();
        
        // Add our inline note macro
        wiki.addTiddler({
            title: "Inline Note Macro",
            text: "\\define inline-note(tiddler) <$button popup=\"$:/state/popup/inline-note/$tiddler$\" class=\"tc-inline-note-trigger\">$tiddler$</$button><$reveal type=\"popup\" state=\"$:/state/popup/inline-note/$tiddler$\" position=\"below\" animate=\"yes\"><div class=\"tc-inline-note-popup\"><$transclude tiddler=\"$tiddler$\" mode=\"block\"/></div></$reveal>",
            type: "text/vnd.tiddlywiki",
            tags: "macro inline-notes"
        });
        
        // Add a test tiddler to reference
        wiki.addTiddler({
            title: "Test Tiddler",
            text: "This is a test tiddler for inline notes",
            type: "text/plain"
        });
        
        // Test the macro renders HTML with expected elements
        var result = wiki.renderText("text/html", "<<inline-note \"Test Tiddler\">>", {parseAsInline: true});
        expect(result).toContain("Test Tiddler");
        expect(result).toContain("tc-inline-note-trigger");
        expect(result).toContain("tc-inline-note-popup");
    });
    
    it("should handle macro with missing tiddler gracefully", function() {
        var wiki = new $tw.Wiki();
        
        // Add our inline note macro
        wiki.addTiddler({
            title: "Inline Note Macro",
            text: "\\define inline-note(tiddler) <$button popup=\"$:/state/popup/inline-note/$tiddler$\" class=\"tc-inline-note-trigger\">$tiddler$</$button><$reveal type=\"popup\" state=\"$:/state/popup/inline-note/$tiddler$\" position=\"below\" animate=\"yes\"><div class=\"tc-inline-note-popup\"><$transclude tiddler=\"$tiddler$\" mode=\"block\"/></div></$reveal>",
            type: "text/vnd.tiddlywiki",
            tags: "macro inline-notes"
        });
        
        // Test with non-existent tiddler
        var result = wiki.renderText("text/html", "<<inline-note \"NonExistentTiddler\">>", {parseAsInline: true});
        expect(result).toContain("NonExistentTiddler");
        expect(result).toContain("tc-inline-note-trigger");
    });
}); 