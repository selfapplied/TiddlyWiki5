/*\
title: $:/plugins/awildsort/fractal-background/fractal-background.js
type: application/javascript
module-type: startup

Fractal background WebGL renderer with pause-on-blur
\*/
(function(){

/*jslint node: true, browser: true */
/*global $tw: false */
"use strict";

exports.name = "fractal-background";
exports.platforms = ["browser"];
exports.after = ["startup"];
exports.synchronous = true;

exports.startup = function() {
	if($tw.browser) {
		var canvas, gl, program, animationId;
		var isPaused = false;
		var time = 0;
		var intensity = 0.4;
		var mode = "nebula"; // nebula, ce1, mandelbulb, particles
		
		// Create canvas
		canvas = document.createElement("canvas");
		canvas.id = "aws-fractal-background";
		document.body.appendChild(canvas);
		
		// Set canvas size
		function resizeCanvas() {
			canvas.width = window.innerWidth;
			canvas.height = window.innerHeight;
		}
		resizeCanvas();
		window.addEventListener("resize", resizeCanvas);
		
		// Get WebGL context
		gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
		if(!gl) {
			console.warn("WebGL not supported, fractal background disabled");
			return;
		}
		
		// Vertex shader
		var vertexShaderSource = `
			attribute vec2 a_position;
			void main() {
				gl_Position = vec4(a_position, 0.0, 1.0);
			}
		`;
		
		// Fragment shader - Stellar Nebula
		var nebulaFragmentShader = `
			precision mediump float;
			uniform float u_time;
			uniform vec2 u_resolution;
			uniform float u_intensity;
			
			vec3 hash3(vec2 p) {
				vec3 q = vec3(dot(p, vec2(127.1, 311.7)),
							  dot(p, vec2(269.5, 183.3)),
							  dot(p, vec2(419.2, 371.9)));
				return fract(sin(q) * 43758.5453);
			}
			
			float noise(vec2 p) {
				vec2 i = floor(p);
				vec2 f = fract(p);
				f = f * f * (3.0 - 2.0 * f);
				vec2 u = vec2(1.0 - f.x, f.x);
				return mix(mix(dot(hash3(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)),
							   dot(hash3(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
						   mix(dot(hash3(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
							   dot(hash3(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x), u.y);
			}
			
			float fbm(vec2 p) {
				float value = 0.0;
				float amplitude = 0.5;
				for(int i = 0; i < 4; i++) {
					value += amplitude * noise(p);
					p *= 2.0;
					amplitude *= 0.5;
				}
				return value;
			}
			
			void main() {
				vec2 uv = gl_FragCoord.xy / u_resolution.xy;
				uv.x *= u_resolution.x / u_resolution.y;
				
				vec2 p = uv * 3.0;
				p += u_time * 0.1;
				
				float n = fbm(p);
				float n2 = fbm(p * 1.5 + vec2(u_time * 0.15, 0.0));
				float n3 = fbm(p * 2.0 - vec2(0.0, u_time * 0.12));
				
				vec3 color = vec3(0.0);
				color += vec3(0.2, 0.4, 0.8) * n * u_intensity;
				color += vec3(0.4, 0.2, 0.6) * n2 * u_intensity * 0.7;
				color += vec3(0.1, 0.3, 0.5) * n3 * u_intensity * 0.5;
				
				// Add some star-like points
				float stars = pow(noise(uv * 200.0), 20.0) * u_intensity * 0.3;
				color += vec3(stars);
				
				gl_FragColor = vec4(color, 1.0);
			}
		`;
		
		// Compile shader
		function createShader(gl, type, source) {
			var shader = gl.createShader(type);
			gl.shaderSource(shader, source);
			gl.compileShader(shader);
			if(!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
				console.error("Shader compile error:", gl.getShaderInfoLog(shader));
				gl.deleteShader(shader);
				return null;
			}
			return shader;
		}
		
		// Create program
		function createProgram(gl, vertexSource, fragmentSource) {
			var vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
			var fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
			if(!vertexShader || !fragmentShader) return null;
			
			var program = gl.createProgram();
			gl.attachShader(program, vertexShader);
			gl.attachShader(program, fragmentShader);
			gl.linkProgram(program);
			if(!gl.getProgramParameter(program, gl.LINK_STATUS)) {
				console.error("Program link error:", gl.getProgramInfoLog(program));
				gl.deleteProgram(program);
				return null;
			}
			return program;
		}
		
		program = createProgram(gl, vertexShaderSource, nebulaFragmentShader);
		if(!program) return;
		
		// Setup geometry (full screen quad)
		var positionBuffer = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
			-1, -1,
			1, -1,
			-1, 1,
			1, 1
		]), gl.STATIC_DRAW);
		
		// Animation loop
		function animate() {
			if(isPaused) return;
			
			time += 0.016; // ~60fps
			
			gl.useProgram(program);
			
			// Set uniforms
			var timeLocation = gl.getUniformLocation(program, "u_time");
			var resolutionLocation = gl.getUniformLocation(program, "u_resolution");
			var intensityLocation = gl.getUniformLocation(program, "u_intensity");
			
			gl.uniform1f(timeLocation, time);
			gl.uniform2f(resolutionLocation, canvas.width, canvas.height);
			gl.uniform1f(intensityLocation, intensity);
			
			// Set attributes
			var positionLocation = gl.getAttribLocation(program, "a_position");
			gl.enableVertexAttribArray(positionLocation);
			gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
			gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
			
			// Draw
			gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
			
			animationId = requestAnimationFrame(animate);
		}
		
		// Pause on blur
		function handleVisibilityChange() {
			if(document.hidden) {
				isPaused = true;
				if(animationId) {
					cancelAnimationFrame(animationId);
					animationId = null;
				}
			} else {
				isPaused = false;
				animate();
			}
		}
		
		document.addEventListener("visibilitychange", handleVisibilityChange);
		window.addEventListener("blur", function() {
			isPaused = true;
			if(animationId) {
				cancelAnimationFrame(animationId);
				animationId = null;
			}
		});
		window.addEventListener("focus", function() {
			isPaused = false;
			animate();
		});
		
		// Start animation
		animate();
		
		// Expose controls via tiddler state
		$tw.wiki.addEventListener("change", function(changes) {
			var intensityTiddler = "$:/plugins/awildsort/fractal-background/intensity";
			if(changes[intensityTiddler]) {
				var newIntensity = parseFloat($tw.wiki.getTiddlerText(intensityTiddler) || "0.4");
				if(!isNaN(newIntensity)) {
					intensity = Math.max(0, Math.min(1, newIntensity));
				}
			}
		});
	}
};

})();


