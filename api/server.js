/*\
TiddlyWiki5 Vercel Serverless Function
Provides a stateless server for TiddlyWiki on Vercel

This handler serves TiddlyWiki in read-only mode with CE1 harmonic operator support
*/

const path = require("path");
const fs = require("fs");

// Initialize TiddlyWiki
const $tw = require("../boot/boot.js").TiddlyWiki();

// Boot the wiki
$tw.boot.argv = [
	path.resolve(__dirname, "../editions/tw5.com"),
	"--listen",
	"port=8080",
	"host=0.0.0.0"
];

module.exports = async (req, res) => {
	try {
		// Handle CE1 harmonic operator API endpoint
		if (req.url && req.url.startsWith("/api/ce1")) {
			return handleCE1Request(req, res);
		}

		// For root path, serve a simple HTML page
		if (req.url === "/" || req.url === "") {
			res.setHeader("Content-Type", "text/html");
			res.status(200).send(`
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>TiddlyWiki5 - Vercel Deployment</title>
	<style>
		body {
			font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
			max-width: 800px;
			margin: 50px auto;
			padding: 20px;
			line-height: 1.6;
		}
		h1 { color: #333; }
		.info-box {
			background: #f5f5f5;
			border-left: 4px solid #5778d8;
			padding: 15px;
			margin: 20px 0;
		}
		code {
			background: #f0f0f0;
			padding: 2px 6px;
			border-radius: 3px;
		}
		pre {
			background: #f5f5f5;
			padding: 15px;
			border-radius: 5px;
			overflow-x: auto;
		}
	</style>
</head>
<body>
	<h1>🚀 TiddlyWiki5 on Vercel</h1>
	
	<div class="info-box">
		<strong>Status:</strong> Successfully deployed!
	</div>

	<h2>Features</h2>
	<ul>
		<li>✅ TiddlyWiki5 v5.4.0-prerelease</li>
		<li>✅ CE1 Harmonic Operator System</li>
		<li>✅ Serverless deployment on Vercel</li>
	</ul>

	<h2>CE1 Harmonic Operator API</h2>
	<p>The CE1 (Collapse-Evaluate) harmonic operator system is available via API:</p>
	
	<h3>Evaluate Harmonic Operator</h3>
	<pre>GET /api/ce1/harmonic?x=2</pre>
	
	<h3>Parse CE1 Expression</h3>
	<pre>POST /api/ce1/parse
Content-Type: application/json

{"expression": "&lt;H 2&gt;"}</pre>
	
	<h3>Find Fixed Point</h3>
	<pre>POST /api/ce1/fixedpoint
Content-Type: application/json

{"initialGuess": 0.5, "maxIterations": 100}</pre>

	<h2>Documentation</h2>
	<p>See <code>CE1_OPERATOR_SYSTEM.md</code> for complete documentation.</p>

	<h2>About</h2>
	<p>
		TiddlyWiki is a non-linear personal web notebook. This deployment includes
		the CE1 operator system for harmonic analysis and singularity-balanced functions.
	</p>

	<footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
		<p>TiddlyWiki © Jeremy Ruston and contributors</p>
	</footer>
</body>
</html>
			`);
			return;
		}

		// Default response
		res.status(200).json({
			name: "TiddlyWiki5",
			version: "5.4.0-prerelease",
			status: "running",
			features: [
				"CE1 Harmonic Operator System",
				"Vercel Serverless Deployment"
			],
			endpoints: {
				ce1: {
					harmonic: "/api/ce1/harmonic?x=<number>",
					parse: "/api/ce1/parse",
					fixedpoint: "/api/ce1/fixedpoint"
				}
			}
		});

	} catch (error) {
		console.error("Error:", error);
		res.status(500).json({
			error: "Internal server error",
			message: error.message
		});
	}
};

/*
Handle CE1 harmonic operator API requests
*/
function handleCE1Request(req, res) {
	const ce1 = require("../core/modules/utils/ce1-harmonic.js");
	
	const url = new URL(req.url, `http://${req.headers.host}`);
	const pathname = url.pathname;

	// GET /api/ce1/harmonic?x=<number>
	if (pathname === "/api/ce1/harmonic" && req.method === "GET") {
		const x = parseFloat(url.searchParams.get("x") || "2");
		
		if (isNaN(x)) {
			res.status(400).json({ error: "Invalid parameter 'x'" });
			return;
		}

		const result = ce1.harmonicOperator(x);
		res.status(200).json({
			input: x,
			result: result,
			description: "Harmonic operator ℋ(x) = {ln(x)} + [ζ(x)] + (tan(πx/2)) + <sin(πx)> + <i·cos(πx)>"
		});
		return;
	}

	// POST /api/ce1/parse
	if (pathname === "/api/ce1/parse" && req.method === "POST") {
		let body = "";
		req.on("data", chunk => {
			body += chunk.toString();
		});
		req.on("end", () => {
			try {
				const data = JSON.parse(body);
				const expression = data.expression;
				
				if (!expression) {
					res.status(400).json({ error: "Missing 'expression' field" });
					return;
				}

				const parsed = ce1.parseCE1(expression);
				const evaluated = ce1.evaluateCE1(parsed);
				
				res.status(200).json({
					input: expression,
					parsed: {
						type: parsed.type,
						value: parsed.value,
						height: parsed.height
					},
					evaluated: evaluated
				});
			} catch (error) {
				res.status(400).json({ error: "Invalid JSON or CE1 expression", message: error.message });
			}
		});
		return;
	}

	// POST /api/ce1/fixedpoint
	if (pathname === "/api/ce1/fixedpoint" && req.method === "POST") {
		let body = "";
		req.on("data", chunk => {
			body += chunk.toString();
		});
		req.on("end", () => {
			try {
				const data = JSON.parse(body);
				const initialGuess = parseFloat(data.initialGuess || 0.5);
				const maxIterations = parseInt(data.maxIterations || 100);
				const tolerance = parseFloat(data.tolerance || 1e-10);
				
				if (isNaN(initialGuess)) {
					res.status(400).json({ error: "Invalid 'initialGuess'" });
					return;
				}

				const result = ce1.fixedPointResolver(initialGuess, maxIterations, tolerance);
				
				res.status(200).json({
					parameters: {
						initialGuess: initialGuess,
						maxIterations: maxIterations,
						tolerance: tolerance
					},
					result: result,
					description: "Finds x such that ℋ(x) ≈ 0 using Newton-Raphson iteration"
				});
			} catch (error) {
				res.status(400).json({ error: "Invalid JSON", message: error.message });
			}
		});
		return;
	}

	// Invalid CE1 endpoint
	res.status(404).json({
		error: "Not found",
		availableEndpoints: [
			"GET /api/ce1/harmonic?x=<number>",
			"POST /api/ce1/parse",
			"POST /api/ce1/fixedpoint"
		]
	});
}
