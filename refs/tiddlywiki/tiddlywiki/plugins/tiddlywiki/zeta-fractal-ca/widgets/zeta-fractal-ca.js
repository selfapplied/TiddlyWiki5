/*\
title: $:/plugins/tiddlywiki/zeta-fractal-ca/widgets/zeta-fractal-ca.js
type: application/javascript
module-type: widget

Zeta Fractal Cellular Automaton Widget
Recursive attractor descent toward the critical line using WebGL shaders and WebAssembly

\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

var Widget = require("$:/core/modules/widgets/widget.js").widget;

var ZetaFractalCAWidget = function(parseTreeNode,options) {
	this.initialise(parseTreeNode,options);
};

/*
Inherit from the base widget class
*/
ZetaFractalCAWidget.prototype = new Widget();

/*
Render this widget into the DOM
*/
ZetaFractalCAWidget.prototype.render = function(parent,nextSibling) {
	var self = this;
	this.parentDomNode = parent;
	this.computeAttributes();
	this.execute();
	
	// Create canvas element
	this.canvasElement = this.document.createElement("canvas");
	this.canvasElement.width = parseInt(this.getAttribute("width", "800"));
	this.canvasElement.height = parseInt(this.getAttribute("height", "600"));
	this.canvasElement.style.border = "1px solid #ccc";
	this.canvasElement.style.display = "block";
	this.canvasElement.style.margin = "10px auto";
	
	parent.insertBefore(this.canvasElement, nextSibling);
	this.domNodes.push(this.canvasElement);
	
	// Initialize WebGL context
	this.initWebGL();
	
	// Initialize CA system
	this.initZetaCA();
	
	// Start animation loop
	this.startAnimation();
};

/*
Compute the internal state of the widget
*/
ZetaFractalCAWidget.prototype.execute = function() {
	// Get widget attributes
	this.zetaReal = parseFloat(this.getAttribute("real", "0.5"));
	this.zetaImag = parseFloat(this.getAttribute("imag", "14.134725"));
	this.maxIterations = parseInt(this.getAttribute("iterations", "30"));
	this.autoStart = this.getAttribute("autostart", "true") === "true";
	this.showControls = this.getAttribute("controls", "true") === "true";
};

/*
Initialize WebGL context with Safari compatibility
*/
ZetaFractalCAWidget.prototype.initWebGL = function() {
	var self = this;
	
	// Get WebGL context (with Safari fallbacks)
	this.gl = this.canvasElement.getContext("webgl") || 
	          this.canvasElement.getContext("experimental-webgl");
	
	if (!this.gl) {
		console.error("WebGL not supported");
		this.renderFallback();
		return;
	}
	
	// Set viewport
	this.gl.viewport(0, 0, this.canvasElement.width, this.canvasElement.height);
	
	// Create shaders
	this.createShaders();
	
	// Create buffers
	this.createBuffers();
	
	// Set up texture for CA state
	this.createTextures();
};

/*
Create Safari-compatible WebGL shaders
*/
ZetaFractalCAWidget.prototype.createShaders = function() {
	var gl = this.gl;
	
	// Vertex shader (Safari GLSL ES 2.0 compatible)
	var vertexShaderSource = `
		attribute vec2 a_position;
		attribute vec2 a_texCoord;
		varying vec2 v_texCoord;
		
		void main() {
			gl_Position = vec4(a_position, 0.0, 1.0);
			v_texCoord = a_texCoord;
		}
	`;
	
	// Fragment shader for Zeta CA visualization
	var fragmentShaderSource = `
		precision mediump float;
		
		uniform sampler2D u_caState;
		uniform sampler2D u_primeField;
		uniform float u_time;
		uniform float u_iteration;
		uniform vec2 u_resolution;
		uniform vec2 u_zetaCenter;
		
		varying vec2 v_texCoord;
		
		// Complex number operations
		vec2 complexMul(vec2 a, vec2 b) {
			return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
		}
		
		vec2 complexDiv(vec2 a, vec2 b) {
			float denom = b.x * b.x + b.y * b.y;
			return vec2((a.x * b.x + a.y * b.y) / denom, (a.y * b.x - a.x * b.y) / denom);
		}
		
		// Möbius function approximation
		float moebiusMu(float n) {
			// Simplified Möbius function for shader
			float result = 1.0;
			float temp = n;
			
			// Check for small prime factors
			for (float p = 2.0; p <= 7.0; p += 1.0) {
				float count = 0.0;
				while (mod(temp, p) < 0.1) {
					temp /= p;
					count += 1.0;
				}
				if (count > 1.5) return 0.0; // Squared factor
				if (count > 0.5) result *= -1.0;
			}
			
			return result;
		}
		
		// Euler factor computation
		vec2 eulerFactor(float p, vec2 s) {
			// Compute (1 - p^(-s))^(-1)
			float logP = log(p);
			vec2 minusS = vec2(-s.x, -s.y);
			vec2 pToMinusS = vec2(exp(minusS.x * logP) * cos(minusS.y * logP),
			                      exp(minusS.x * logP) * sin(minusS.y * logP));
			
			vec2 oneMinusPToMinusS = vec2(1.0 - pToMinusS.x, -pToMinusS.y);
			
			// Return 1 / (1 - p^(-s))
			return complexDiv(vec2(1.0, 0.0), oneMinusPToMinusS);
		}
		
		// Fractal subdivision threshold
		float subdivisionThreshold(vec2 pos, float depth) {
			// Higher subdivision near critical line (Re(s) = 0.5)
			float distFromCritical = abs(pos.x - 0.5);
			return 0.1 / (1.0 + distFromCritical * 10.0) / (depth + 1.0);
		}
		
		void main() {
			vec2 pos = v_texCoord;
			
			// Map to complex plane
			vec2 s = u_zetaCenter + (pos - 0.5) * 4.0;
			
			// Get current CA state
			vec4 caState = texture2D(u_caState, pos);
			vec4 primeData = texture2D(u_primeField, pos);
			
			// Extract cell properties
			vec2 zetaValue = caState.xy;
			float moebiusSpin = caState.z;
			float error = caState.w;
			
			float primeContent = primeData.x;
			float depth = primeData.y;
			
			// Apply time-based Möbius evolution
			float evolutionRate = sin(u_time * 0.1 + pos.x * 6.28) * 0.1;
			moebiusSpin += evolutionRate * moebiusMu(primeContent + 1.0);
			
			// Möbius transform
			vec2 moebiusA = vec2(1.0, 0.1 * moebiusSpin);
			vec2 moebiusB = vec2(0.1 * error, 0.0);
			float moebiusC = 0.01 * primeContent;
			float moebiusD = 1.0;
			
			vec2 transformed = complexDiv(
				complexMul(moebiusA, s) + moebiusB,
				vec2(moebiusC, 0.0) * s + vec2(moebiusD, 0.0)
			);
			
			// Critical line attraction
			float criticalPull = 0.01 * (0.5 - s.x);
			transformed.x += criticalPull;
			
			// Color mapping
			float phase = atan(zetaValue.y, zetaValue.x);
			float magnitude = length(zetaValue);
			
			// Hue: Möbius inversion (sign of μ(n))
			float hue = (moebiusSpin + 1.0) * 0.5;
			
			// Brightness: log-scale magnitude
			float brightness = log(magnitude + 1.0) * 0.3;
			brightness = clamp(brightness, 0.0, 1.0);
			
			// Depth visualization through saturation
			float saturation = 1.0 - depth * 0.1;
			saturation = clamp(saturation, 0.3, 1.0);
			
			// HSV to RGB conversion
			vec3 hsv = vec3(hue, saturation, brightness);
			vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
			vec3 p = abs(fract(hsv.xxx + K.xyz) * 6.0 - K.www);
			vec3 rgb = hsv.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), hsv.y);
			
			// Add fractal shimmer
			float shimmer = sin(phase * 10.0 + u_time) * 0.1 + 0.9;
			rgb *= shimmer;
			
			// Error visualization (red overlay)
			if (error > subdivisionThreshold(s, depth)) {
				rgb.r += 0.3;
			}
			
			gl_FragColor = vec4(rgb, 1.0);
		}
	`;
	
	// Compile shaders
	this.vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexShaderSource);
	this.fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fragmentShaderSource);
	
	// Create program
	this.program = gl.createProgram();
	gl.attachShader(this.program, this.vertexShader);
	gl.attachShader(this.program, this.fragmentShader);
	gl.linkProgram(this.program);
	
	if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
		console.error("Shader program linking failed:", gl.getProgramInfoLog(this.program));
		return;
	}
	
	// Get attribute and uniform locations
	this.locations = {
		attributes: {
			position: gl.getAttribLocation(this.program, "a_position"),
			texCoord: gl.getAttribLocation(this.program, "a_texCoord")
		},
		uniforms: {
			caState: gl.getUniformLocation(this.program, "u_caState"),
			primeField: gl.getUniformLocation(this.program, "u_primeField"),
			time: gl.getUniformLocation(this.program, "u_time"),
			iteration: gl.getUniformLocation(this.program, "u_iteration"),
			resolution: gl.getUniformLocation(this.program, "u_resolution"),
			zetaCenter: gl.getUniformLocation(this.program, "u_zetaCenter")
		}
	};
};

/*
Compile a shader
*/
ZetaFractalCAWidget.prototype.compileShader = function(type, source) {
	var gl = this.gl;
	var shader = gl.createShader(type);
	gl.shaderSource(shader, source);
	gl.compileShader(shader);
	
	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		console.error("Shader compilation failed:", gl.getShaderInfoLog(shader));
		gl.deleteShader(shader);
		return null;
	}
	
	return shader;
};

/*
Create vertex buffers
*/
ZetaFractalCAWidget.prototype.createBuffers = function() {
	var gl = this.gl;
	
	// Quad vertices
	var vertices = new Float32Array([
		-1.0, -1.0,  0.0, 0.0,
		 1.0, -1.0,  1.0, 0.0,
		-1.0,  1.0,  0.0, 1.0,
		 1.0,  1.0,  1.0, 1.0
	]);
	
	this.vertexBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
	gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
};

/*
Create textures for CA state
*/
ZetaFractalCAWidget.prototype.createTextures = function() {
	var gl = this.gl;
	var width = 256;
	var height = 256;
	
	// CA state texture (RGBA = zetaValue.xy, moebiusSpin, error)
	this.caStateTexture = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, this.caStateTexture);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
	
	// Initialize with starting state
	var caData = new Float32Array(width * height * 4);
	for (var i = 0; i < width * height; i++) {
		caData[i * 4 + 0] = 1.0;      // zetaValue.real
		caData[i * 4 + 1] = 0.0;      // zetaValue.imag
		caData[i * 4 + 2] = 0.0;      // moebiusSpin
		caData[i * 4 + 3] = 1.0;      // error
	}
	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.FLOAT, caData);
	
	// Prime field texture (RG = primeContent, depth)
	this.primeFieldTexture = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, this.primeFieldTexture);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
	
	var primeData = new Float32Array(width * height * 4);
	for (var i = 0; i < width * height; i++) {
		primeData[i * 4 + 0] = 0.0;   // primeContent
		primeData[i * 4 + 1] = 0.0;   // depth
		primeData[i * 4 + 2] = 0.0;   // unused
		primeData[i * 4 + 3] = 1.0;   // unused
	}
	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.FLOAT, primeData);
};

/*
Initialize CA system
*/
ZetaFractalCAWidget.prototype.initZetaCA = function() {
	this.iteration = 0;
	this.primeIndex = 0;
	this.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
	this.startTime = Date.now();
	this.isRunning = this.autoStart;
};

/*
Animation loop
*/
ZetaFractalCAWidget.prototype.startAnimation = function() {
	var self = this;
	
	function animate() {
		if (self.isRunning && self.iteration < self.maxIterations) {
			self.updateCA();
		}
		self.render();
		requestAnimationFrame(animate);
	}
	
	animate();
};

/*
Update CA state (this would use WebAssembly in production)
*/
ZetaFractalCAWidget.prototype.updateCA = function() {
	// Placeholder for WebAssembly CA update
	// In production, this would call into WASM module
	this.iteration++;
	if (this.primeIndex < this.primes.length) {
		this.primeIndex++;
	}
};

/*
Render frame
*/
ZetaFractalCAWidget.prototype.render = function() {
	var gl = this.gl;
	if (!gl || !this.program) return;
	
	gl.useProgram(this.program);
	
	// Bind vertex buffer
	gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
	gl.enableVertexAttribArray(this.locations.attributes.position);
	gl.enableVertexAttribArray(this.locations.attributes.texCoord);
	gl.vertexAttribPointer(this.locations.attributes.position, 2, gl.FLOAT, false, 16, 0);
	gl.vertexAttribPointer(this.locations.attributes.texCoord, 2, gl.FLOAT, false, 16, 8);
	
	// Bind textures
	gl.activeTexture(gl.TEXTURE0);
	gl.bindTexture(gl.TEXTURE_2D, this.caStateTexture);
	gl.uniform1i(this.locations.uniforms.caState, 0);
	
	gl.activeTexture(gl.TEXTURE1);
	gl.bindTexture(gl.TEXTURE_2D, this.primeFieldTexture);
	gl.uniform1i(this.locations.uniforms.primeField, 1);
	
	// Set uniforms
	gl.uniform1f(this.locations.uniforms.time, (Date.now() - this.startTime) * 0.001);
	gl.uniform1f(this.locations.uniforms.iteration, this.iteration);
	gl.uniform2f(this.locations.uniforms.resolution, this.canvasElement.width, this.canvasElement.height);
	gl.uniform2f(this.locations.uniforms.zetaCenter, this.zetaReal, this.zetaImag);
	
	// Draw
	gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
};

/*
Fallback for non-WebGL browsers
*/
ZetaFractalCAWidget.prototype.renderFallback = function() {
	var ctx = this.canvasElement.getContext("2d");
	ctx.fillStyle = "#333";
	ctx.fillRect(0, 0, this.canvasElement.width, this.canvasElement.height);
	ctx.fillStyle = "#fff";
	ctx.font = "20px Arial";
	ctx.textAlign = "center";
	ctx.fillText("WebGL required for Zeta CA", this.canvasElement.width/2, this.canvasElement.height/2);
};

/*
Selectively refreshes the widget if needed
*/
ZetaFractalCAWidget.prototype.refresh = function(changedTiddlers) {
	return false;
};

exports["zeta-fractal-ca"] = ZetaFractalCAWidget;

})();