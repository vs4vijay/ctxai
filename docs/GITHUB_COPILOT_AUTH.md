# GitHub Copilot Authentication Guide

This guide explains how to authenticate with GitHub Copilot using OAuth device code flow in ctxai.

## Overview

ctxai now supports GitHub Copilot authentication using OAuth 2.0 device authorization grant. This allows you to use your GitHub Copilot subscription with ctxai without manually managing API keys.

## Prerequisites

- Active GitHub account
- GitHub Copilot subscription (Individual, Business, or Enterprise)

## Quick Start

### Authenticate with GitHub Copilot

```bash
ctxai login github-copilot
```

This will:
1. Request a device code from GitHub
2. Display a user code and verification URL
3. Wait for you to authorize the application
4. Retrieve and store your Copilot API token

### Start Using Copilot

```bash
ctxai chat --provider github-copilot
```

### Logout

```bash
ctxai logout github-copilot
```

## How It Works

### OAuth Device Code Flow

GitHub Copilot uses the OAuth 2.0 device authorization grant, designed for CLI applications:

**Step 1: Device Code Request**
- ctxai requests a device code from GitHub
- Receives a user code and verification URL

**Step 2: User Authorization**
- User visits `https://github.com/login/device`
- Enters the provided user code
- Authorizes the application

**Step 3: Token Polling**
- ctxai polls GitHub's token endpoint
- Waits for user to complete authorization
- Receives access token upon success

**Step 4: Copilot Token Exchange**
- Exchanges GitHub access token for Copilot API token
- Stores token securely in `~/.ctxai/keys.json`

### Authentication Flow Diagram

```
┌─────────┐                ┌────────┐                ┌─────────────┐
│  ctxai  │                │ GitHub │                │    User     │
└────┬────┘                └───┬────┘                └──────┬──────┘
     │                         │                            │
     │ 1. Request device code  │                            │
     ├────────────────────────>│                            │
     │                         │                            │
     │ 2. Device + user code   │                            │
     │<────────────────────────┤                            │
     │                         │                            │
     │ 3. Display user code    │                            │
     ├─────────────────────────────────────────────────────>│
     │                         │                            │
     │                         │ 4. Visit URL + enter code  │
     │                         │<───────────────────────────┤
     │                         │                            │
     │ 5. Poll for token       │                            │
     ├────────────────────────>│                            │
     │ (repeats every 5s)      │                            │
     │                         │                            │
     │ 6. Access token         │                            │
     │<────────────────────────┤                            │
     │                         │                            │
     │ 7. Request Copilot token│                            │
     ├────────────────────────>│                            │
     │                         │                            │
     │ 8. Copilot API token    │                            │
     │<────────────────────────┤                            │
     │                         │                            │
     │ 9. Store token securely │                            │
     │                         │                            │
```

## Authentication Details

### OAuth Client

- **Client ID**: `Iv1.b507a08c87ecfe98` (GitHub Copilot public client)
- **Scope**: `read:user`
- **Grant Type**: Device authorization

### API Endpoints

1. **Device Code**: `https://github.com/login/device/code`
2. **Access Token**: `https://github.com/login/oauth/access_token`
3. **Copilot Token**: `https://api.github.com/copilot_internal/v2/token`

### Headers

The implementation identifies as VS Code to ensure compatibility:

```
User-Agent: GitHubCopilotChat/0.35.0
Editor-Version: vscode/1.99.3
Editor-Plugin-Version: copilot-chat/0.35.0
```

## Stored Token Data

Unlike simple API keys, GitHub Copilot stores structured token data:

```json
{
  "github-copilot": {
    "token": "gho_xxxxx...",
    "expires_at": 1234567890,
    "refresh_token": "gho_refresh_xxxxx...",
    "access_token": "gho_access_xxxxx..."
  }
}
```

This allows for:
- Token expiration tracking
- Automatic token refresh (future feature)
- Access token storage for API requests

## Security

### Token Storage

- Stored in `~/.ctxai/keys.json`
- File permissions: `0600` (owner read/write only) on Unix
- Not committed to version control

### OAuth Flow Security

- No client secrets required (public client)
- Device code is single-use
- User explicitly authorizes via GitHub
- Tokens have expiration times
- No passwords stored locally

## Troubleshooting

### Device Code Expired

If you take too long to authorize:
```
Error: Device code expired. Please try again.
```

**Solution**: Run `ctxai login github-copilot` again and authorize within the time limit (usually 15 minutes).

### Authorization Denied

If you cancel or deny the authorization:
```
Error: Authorization denied by user.
```

**Solution**: Run `ctxai login github-copilot` again and click "Authorize" on GitHub.

### No Copilot Subscription

If you don't have an active Copilot subscription:
```
Error: HTTP 403: Copilot subscription required
```

**Solution**: Subscribe to GitHub Copilot at https://github.com/features/copilot

### Token Expired

If your token expires during use:
```
Error: Copilot token expired
```

**Solution**: Re-authenticate with `ctxai login github-copilot`

### Connection Issues

If you can't reach GitHub:
```
Error: Request failed: Connection timeout
```

**Solution**:
1. Check your internet connection
2. Verify GitHub is accessible: `curl https://github.com`
3. Check for proxy/firewall issues

## Advanced Usage

### Manual Token Management

View stored tokens:
```bash
cat ~/.ctxai/keys.json
```

Clear all tokens:
```bash
rm ~/.ctxai/keys.json
```

### Environment Variable Override

You can still use environment variables if preferred:
```bash
export GITHUB_COPILOT_TOKEN=your-token-here
ctxai chat --provider github-copilot
```

## Comparison with OpenRouter OAuth

| Feature | GitHub Copilot | OpenRouter |
|---------|---------------|------------|
| OAuth Flow | Device Code | PKCE |
| Browser Needed | Yes (separate) | Yes (embedded) |
| Client Secret | No | No |
| Token Type | Structured data | Simple string |
| Expiration | Yes, tracked | No |
| Refresh Support | Yes (future) | No |
| CLI Friendly | Very | Moderate |

## Implementation Reference

Based on OpenCode's implementation:
- [anomalyco/opencode-copilot-auth](https://github.com/anomalyco/opencode-copilot-auth)
- [sst/opencode-github-copilot](https://github.com/sst/opencode-github-copilot)

## FAQ

**Q: Do I need VS Code installed?**
A: No, ctxai doesn't require VS Code. We just identify as VS Code for API compatibility.

**Q: Can I use this with GitHub Enterprise?**
A: The current implementation targets github.com, but could be extended for Enterprise support.

**Q: How long does the token last?**
A: Tokens typically last several hours. Check the `expires_at` field in the stored data.

**Q: Can I use multiple accounts?**
A: Currently, only one Copilot account can be stored at a time. Logout and login with a different account.

**Q: Is this officially supported by GitHub?**
A: This uses GitHub's public OAuth endpoints. While not explicitly documented for third-party CLI tools, it follows standard OAuth practices.

## Resources

- [GitHub Copilot Official Site](https://github.com/features/copilot)
- [OAuth 2.0 Device Authorization Grant](https://datatracker.ietf.org/doc/html/rfc8628)
- [GitHub OAuth Documentation](https://docs.github.com/en/developers/apps/building-oauth-apps)
- [OpenCode Copilot Integration](https://github.blog/changelog/2026-01-16-github-copilot-now-supports-opencode/)
