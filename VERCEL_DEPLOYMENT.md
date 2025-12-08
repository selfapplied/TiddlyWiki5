# TiddlyWiki5 Vercel Deployment Guide

This document explains how to deploy TiddlyWiki5 with the CE1 Harmonic Operator System on Vercel.

## Overview

This repository is configured for serverless deployment on Vercel with:
- TiddlyWiki5 v5.4.0-prerelease
- CE1 (Collapse-Evaluate) Harmonic Operator System
- RESTful API endpoints for CE1 operations
- Optimized serverless function configuration

## Quick Deploy

### Option 1: Deploy to Vercel (Recommended)

1. **Fork this repository** to your GitHub account

2. **Visit [Vercel](https://vercel.com)** and sign in with GitHub

3. **Import your fork**:
   - Click "New Project"
   - Import your forked repository
   - Vercel will automatically detect the `vercel.json` configuration

4. **Deploy**:
   - Click "Deploy"
   - Wait for the build to complete
   - Visit your deployment URL

### Option 2: Deploy with Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Clone the repository
git clone https://github.com/YOUR_USERNAME/TiddlyWiki5.git
cd TiddlyWiki5

# Install dependencies
npm install

# Deploy to Vercel
vercel
```

## Configuration

The deployment is configured via `vercel.json`:

```json
{
  "version": 2,
  "name": "tiddlywiki5",
  "builds": [
    {
      "src": "api/server.js",
      "use": "@vercel/node"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/server.js"
    },
    {
      "src": "/(.*)",
      "dest": "/api/server.js"
    }
  ]
}
```

### Function Configuration

- **Memory**: 1024 MB
- **Max Duration**: 10 seconds
- **Region**: Auto (can be configured in Vercel dashboard)

## API Endpoints

Once deployed, the following endpoints are available:

### 1. Home Page
```
GET /
```
Returns an HTML page with information about the deployment and available APIs.

### 2. Harmonic Operator Evaluation
```
GET /api/ce1/harmonic?x=<number>
```

**Example:**
```bash
curl https://your-deployment.vercel.app/api/ce1/harmonic?x=2
```

**Response:**
```json
{
  "input": 2,
  "result": {
    "re": 1.234,
    "im": 0.567,
    "components": {
      "boundary": 0.693,
      "memory": 1.645,
      "morphism": -1.0,
      "witness_sin": 0.0,
      "witness_cos": 1.0
    }
  },
  "description": "Harmonic operator ℋ(x) = {ln(x)} + [ζ(x)] + (tan(πx/2)) + <sin(πx)> + <i·cos(πx)>"
}
```

### 3. Parse CE1 Expression
```
POST /api/ce1/parse
Content-Type: application/json

{
  "expression": "<H 2>"
}
```

**Example:**
```bash
curl -X POST https://your-deployment.vercel.app/api/ce1/parse \
  -H "Content-Type: application/json" \
  -d '{"expression": "{2.718}"}'
```

**Response:**
```json
{
  "input": "{2.718}",
  "parsed": {
    "type": "boundary",
    "value": null,
    "height": 1
  },
  "evaluated": 0.9999
}
```

### 4. Find Fixed Point
```
POST /api/ce1/fixedpoint
Content-Type: application/json

{
  "initialGuess": 0.5,
  "maxIterations": 100,
  "tolerance": 1e-10
}
```

**Example:**
```bash
curl -X POST https://your-deployment.vercel.app/api/ce1/fixedpoint \
  -H "Content-Type: application/json" \
  -d '{"initialGuess": 0.5, "maxIterations": 100}'
```

**Response:**
```json
{
  "parameters": {
    "initialGuess": 0.5,
    "maxIterations": 100,
    "tolerance": 1e-10
  },
  "result": {
    "value": 0.5001,
    "iterations": 15,
    "residual": 9.8e-11,
    "converged": true
  },
  "description": "Finds x such that ℋ(x) ≈ 0 using Newton-Raphson iteration"
}
```

## Environment Variables

You can configure the following environment variables in the Vercel dashboard:

- `NODE_ENV`: Set to `production` (default)
- `TW_PORT`: Port for internal TiddlyWiki server (default: 8080)
- `TW_HOST`: Host for internal server (default: 0.0.0.0)

## Local Development

To run the server locally:

```bash
# Install dependencies
npm install

# Run TiddlyWiki server
npm run dev

# Or run tests
npm test

# Or run linting
npm run lint
```

The local server will be available at `http://localhost:8080`.

## CE1 Harmonic Operator System

The deployment includes the CE1 (Collapse-Evaluate) operator system for harmonic analysis. See `CE1_OPERATOR_SYSTEM.md` for complete documentation.

### Quick CE1 Examples

**1. Evaluate harmonic operator:**
```javascript
const ce1 = require("./core/modules/utils/ce1-harmonic.js");
const result = ce1.harmonicOperator(2);
console.log(result);
```

**2. Parse CE1 expression:**
```javascript
const expr = ce1.parseCE1("<H 2>");
const result = ce1.evaluateCE1(expr);
console.log(result);
```

**3. Find fixed point:**
```javascript
const solution = ce1.fixedPointResolver(0.5, 100, 1e-10);
console.log(solution);
```

## Limitations

### Vercel Serverless Limitations

1. **Stateless**: Each request is handled independently. No persistent state between requests.
2. **Max Duration**: Functions timeout after 10 seconds (configurable up to 60s on paid plans).
3. **Memory**: Limited to 1024 MB (configurable on paid plans).
4. **No Persistent Storage**: TiddlyWiki operates in read-only mode. Changes are not saved between requests.

### TiddlyWiki Considerations

This deployment is optimized for:
- ✅ Serving static TiddlyWiki content
- ✅ CE1 harmonic operator computations
- ✅ API-based mathematical operations
- ❌ Real-time editing (use Node.js server instead)
- ❌ Persistent wiki storage (use traditional hosting)

## Troubleshooting

### Build Fails

1. Check that `package.json` and `vercel.json` are present
2. Verify all dependencies are listed in `package.json`
3. Check build logs in Vercel dashboard

### Function Timeout

If CE1 operations timeout:
1. Reduce `maxIterations` parameter
2. Use simpler expressions
3. Upgrade to Vercel Pro for longer timeout limits

### API Returns 404

1. Verify the endpoint URL matches the API routes
2. Check that `api/server.js` exists
3. Review deployment logs for errors

## Performance Optimization

### Cold Starts

Serverless functions have "cold start" latency on the first request. Subsequent requests are faster.

**Tips:**
- Use Vercel's edge caching
- Implement request batching for multiple operations
- Consider warming functions with periodic requests

### Caching

Static assets are cached by default. API responses include cache headers:
```
Cache-Control: public, max-age=3600, must-revalidate
```

## Security

Security headers are configured in `vercel.json`:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: SAMEORIGIN`
- `X-XSS-Protection: 1; mode=block`

### Best Practices

1. **HTTPS Only**: Vercel provides automatic HTTPS
2. **Rate Limiting**: Consider implementing rate limits for API endpoints
3. **Input Validation**: All API inputs are validated before processing
4. **Error Handling**: Errors return appropriate HTTP status codes

## Monitoring

### Vercel Analytics

Enable Vercel Analytics in the dashboard to track:
- Request count
- Response times
- Error rates
- Geographic distribution

### Custom Logging

The server logs to Vercel's logging system:
```javascript
console.log("Info message");
console.error("Error message");
```

View logs in the Vercel dashboard under Functions → Logs.

## Advanced Configuration

### Custom Domain

1. Go to Vercel dashboard → Settings → Domains
2. Add your custom domain
3. Configure DNS records as instructed
4. SSL certificate is automatically provisioned

### Multiple Environments

Vercel supports multiple environments:
- **Production**: Main branch deployments
- **Preview**: Pull request deployments
- **Development**: Local development

Configure branch-specific settings in the Vercel dashboard.

## Contributing

To contribute to the CE1 system or Vercel deployment:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `npm test`
5. Run linting: `npm run lint`
6. Submit a pull request

## Resources

- [TiddlyWiki Documentation](https://tiddlywiki.com)
- [CE1 Operator System Documentation](./CE1_OPERATOR_SYSTEM.md)
- [Vercel Documentation](https://vercel.com/docs)
- [Vercel Node.js Runtime](https://vercel.com/docs/runtimes#official-runtimes/node-js)

## License

TiddlyWiki5 is released under the BSD 3-Clause License. See `license` file for details.

## Support

For issues related to:
- **TiddlyWiki**: [GitHub Issues](https://github.com/TiddlyWiki/TiddlyWiki5/issues)
- **CE1 System**: [GitHub Discussions](https://github.com/TiddlyWiki/TiddlyWiki5/discussions)
- **Vercel Deployment**: Check Vercel support documentation

---

**Note**: This is a demonstration deployment showcasing the CE1 Harmonic Operator System. For production TiddlyWiki instances with editing capabilities, consider traditional Node.js hosting or the official TiddlyWiki hosting options.
