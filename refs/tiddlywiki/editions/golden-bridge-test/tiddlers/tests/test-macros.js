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
        
        // Add a tiddler that defines and uses the macro
        wiki.addTiddler({
            title: "Test Macro Usage",
            text: "\\define test-macro(name:\"Default\") Hello, I'm $name$\n\\end\n\n<$link to=<<test-macro \"Alice\">>>Click here</$link>",
            type: "text/vnd.tiddlywiki"
        });
        
        // Test the macro usage in HTML context
        var result = wiki.renderTiddler("text/html", "Test Macro Usage");
        console.log("Result:", result);
        // The \end gets rendered as a paragraph, but the macro works in the link
        expect(result).toBe("<p>\\end</p><p><a class=\"tc-tiddlylink tc-tiddlylink-missing\" href=\"#Hello%2C%20I%27m%20Alice\">Click here</a></p>");
    });
    
    it("should test basic transclusion", function() {
        var wiki = new $tw.Wiki();
        
        // Add a simple tiddler
        wiki.addTiddler({
            title: "Simple Test",
            text: "Hello World",
            type: "text/plain"
        });
        
        // Test basic transclusion
        var result = wiki.renderTiddler("text/plain", "Simple Test");
        console.log("Transclusion result:", result);
        expect(result).toBe("Hello World");
    });
}); 