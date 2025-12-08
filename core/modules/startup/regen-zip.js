/*\
title: $:/core/modules/startup/regen-zip.js
type: application/javascript
module-type: startup

Initialize REGEN-ZIP VM and integrate with core TiddlyWiki engine

\*/

"use strict";

exports.name = "regen-zip";
exports.platforms = ["browser", "node"];
exports.after = ["load-modules"];
exports.synchronous = true;

exports.startup = function() {
	// Initialize the REGEN-ZIP VM if available
	if($tw.utils.RegenZipVM) {
		$tw.regenZipVM = new $tw.utils.RegenZipVM($tw.wiki);
		
		// Initialize ZP35 operator for semantic compatibility checking
		if($tw.utils.ZP35Operator) {
			$tw.zp35Operator = new $tw.utils.ZP35Operator();
		}
		
		// Register core integration hooks
		setupCoreIntegration();
		
		console.log("REGEN-ZIP VM initialized and integrated with core engine");
	}
};

/*
Setup core integration so the entire TiddlyWiki system can benefit from asset generation
*/
function setupCoreIntegration() {
	// Cache for generated assets to avoid regeneration on every access
	$tw.regenZipCache = {};
	
	// Hook into wiki getTiddlerText to automatically generate assets
	var originalGetTiddlerText = $tw.wiki.getTiddlerText;
	$tw.wiki.getTiddlerText = function(title, defaultText) {
		var tiddler = this.getTiddler(title);
		
		// Check if this tiddler has a regen-zip field
		if(tiddler && tiddler.fields["regen-zip"]) {
			return getGeneratedText(title, tiddler, defaultText);
		}
		
		// Otherwise use original implementation
		return originalGetTiddlerText.call(this, title, defaultText);
	};
	
	// Listen for tiddler changes to invalidate cache
	$tw.wiki.addEventListener("change", function(changes) {
		Object.keys(changes).forEach(function(title) {
			// Clear cache for changed tiddlers
			if($tw.regenZipCache[title]) {
				delete $tw.regenZipCache[title];
			}
			
			// Also clear cache if generator or seed changed
			var tiddler = $tw.wiki.getTiddler(title);
			if(tiddler) {
				var generator = tiddler.fields.generator;
				if(generator) {
					// Clear all cached items that used this generator
					clearCacheByGenerator(generator);
				}
			}
		});
	});
	
	// Add helper method to wiki for manual asset generation
	$tw.wiki.generateAssets = function(title) {
		var tiddler = this.getTiddler(title);
		if(!tiddler) {
			return null;
		}
		
		if(!tiddler.fields["regen-zip"]) {
			return null;
		}
		
		// Clear cache and regenerate
		delete $tw.regenZipCache[title];
		
		var vm = $tw.regenZipVM;
		vm.reset();
		
		if(vm.load(tiddler)) {
			var result = vm.run();
			if(result.success) {
				// Cache the result
				$tw.regenZipCache[title] = {
					assets: result.assets,
					metadata: result.metadata,
					timestamp: Date.now()
				};
				return result;
			}
		}
		
		return null;
	};
	
	// Add helper method to check ZP35 coherence between tiddlers
	$tw.wiki.checkCoherence = function(sourceTiddler, targetTiddler) {
		if(!$tw.zp35Operator) {
			return {
				allowed: true,
				mode: "unchecked",
				message: "ZP35 operator not available"
			};
		}
		
		var source = typeof sourceTiddler === "string" ? 
			this.getTiddler(sourceTiddler) : sourceTiddler;
		var target = typeof targetTiddler === "string" ?
			this.getTiddler(targetTiddler) : targetTiddler;
		
		return $tw.zp35Operator.checkCoherence(source, target);
	};
	
	// Add method to calculate ZP35 signature for a tiddler
	$tw.wiki.calculateZP35Signature = function(title) {
		if(!$tw.zp35Operator) {
			return null;
		}
		
		var tiddler = typeof title === "string" ? 
			this.getTiddler(title) : title;
		
		if(!tiddler) {
			return null;
		}
		
		return $tw.zp35Operator.calculateSignature(tiddler);
	};
}

/*
Get generated text for a tiddler with regen-zip field
*/
function getGeneratedText(title, tiddler, defaultText) {
	// Check cache first
	if($tw.regenZipCache[title]) {
		var cached = $tw.regenZipCache[title];
		
		// Return the first text asset if available
		if(cached.assets && cached.assets.length > 0) {
			var textAsset = cached.assets.find(function(asset) {
				return asset.type.startsWith("text/");
			});
			
			if(textAsset) {
				return textAsset.data;
			}
		}
	}
	
	// Generate assets
	var result = $tw.wiki.generateAssets(title);
	
	if(result && result.success && result.assets && result.assets.length > 0) {
		// Find first text asset
		var textAsset = result.assets.find(function(asset) {
			return asset.type.startsWith("text/");
		});
		
		if(textAsset) {
			return textAsset.data;
		}
		
		// If no text asset, return metadata as text
		return "Generated " + result.assets.length + " asset(s)";
	}
	
	// Fallback to original text field or default
	return tiddler.fields.text || defaultText || "";
}

/*
Clear cache entries that used a specific generator
*/
function clearCacheByGenerator(generatorName) {
	Object.keys($tw.regenZipCache).forEach(function(title) {
		var tiddler = $tw.wiki.getTiddler(title);
		if(tiddler && tiddler.fields.generator === generatorName) {
			delete $tw.regenZipCache[title];
		}
	});
}
