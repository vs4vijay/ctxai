# OAuth Authentication Guide

This guide explains how to use the OAuth PKCE authentication flow with ctxai, specifically for OpenRouter.

## Overview

ctxai now supports OAuth 2.0 PKCE (Proof Key for Code Exchange) authentication for OpenRouter. This provides a more convenient and secure way to authenticate compared to manually copying API keys.

## Benefits of OAuth Login

1. **One-Click Authentication**: Authenticate directly through your browser
2. **Secure**: Uses PKCE flow (no client secrets needed)
3. **Convenient**: API key is automatically stored and used
4. **Familiar**: Same flow as GitHub, Google, etc.

## Quick Start

### Login to OpenRouter

```bash
ctxai login openrouter
```

This will:
1. Open your browser to OpenRouter's authorization page
2. Prompt you to log in to OpenRouter (if not already logged in)
3. Ask you to authorize ctxai
4. Receive the API key and store it securely in `~/.ctxai/keys.json`

### Start Chatting

After logging in, you can immediately start using the chat:

```bash
ctxai chat --provider openrouter
```

The chat will automatically use your stored API key from the OAuth login.

### Check Login Status

You can check which providers you're logged into by running:

```bash
ctxai chat --provider openrouter
```

The provider status screen will show "OpenRouter configured" if you're logged in.

### Logout

To remove your stored credentials:

```bash
ctxai logout openrouter
```

## How It Works

### OAuth PKCE Flow

1. **Code Verifier Generation**: A cryptographically random string (128 chars) is generated
2. **Code Challenge**: The SHA256 hash of the verifier is computed and base64url-encoded
3. **Authorization Request**: User is redirected to OpenRouter with the challenge
4. **User Authorization**: User logs in and authorizes the application
5. **Callback**: OpenRouter redirects back with an authorization code
6. **Token Exchange**: The code is exchanged for an API key using the verifier
7. **Secure Storage**: The API key is stored in `~/.ctxai/keys.json` with restricted permissions

### Security Features

- **PKCE**: Prevents authorization code interception attacks
- **Local Server**: Callback server runs only on localhost
- **Restricted Permissions**: Keystore file is only readable/writable by owner (Unix)
- **No Client Secrets**: Doesn't require storing application secrets

## Advanced Usage

### Custom Callback Port

If port 8080 is in use, specify a different port:

```bash
ctxai login openrouter --port 3000
```

### Manual API Key (Alternative)

You can still use environment variables instead of OAuth:

```bash
export OPENROUTER_API_KEY=your-key-here
ctxai chat --provider openrouter
```

Environment variables take precedence over stored OAuth keys.

## Troubleshooting

### Browser Doesn't Open

If your browser doesn't open automatically, copy the URL from the terminal and paste it into your browser.

### Port Already in Use

If you get a "port in use" error, try a different port:

```bash
ctxai login openrouter --port 8081
```

### Authentication Failed

If authentication fails:
1. Check that you have a valid OpenRouter account
2. Try logging out and logging in again
3. Check your internet connection
4. Ensure no firewall is blocking localhost connections

### Stored Key Not Working

If the stored key stops working:
1. Logout: `ctxai logout openrouter`
2. Login again: `ctxai login openrouter`

## File Locations

- **Keystore**: `~/.ctxai/keys.json`
- **Permissions**: `0600` (owner read/write only) on Unix

## API Reference

### Commands

```bash
# Login with OAuth
ctxai login [provider] [--port PORT]

# Logout (remove stored credentials)
ctxai logout [provider]

# Use stored credentials
ctxai chat --provider [provider]
```

### Supported Providers

Currently, OAuth is supported for:
- **openrouter**: Full OAuth PKCE support

Other providers (Anthropic, OpenAI) still use environment variables:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`

## Implementation Details

For developers interested in the implementation:

- **PKCE Module**: `src/ctxai/auth/oauth_pkce.py`
- **Keystore**: `src/ctxai/auth/keystore.py`
- **CLI Commands**: `src/ctxai/app.py` (login, logout commands)
- **Provider Integration**: `src/ctxai/agent/llm/factory.py`

The implementation follows RFC 7636 (OAuth PKCE) and OpenRouter's authentication specification.

## Resources

- [OpenRouter OAuth PKCE Documentation](https://openrouter.ai/docs/api/reference/authentication)
- [RFC 7636 - PKCE Specification](https://datatracker.ietf.org/doc/html/rfc7636)
- [OAuth 2.0 Authorization Framework](https://datatracker.ietf.org/doc/html/rfc6749)
