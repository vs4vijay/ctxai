# Systemd Deployment

Run ctxai as a managed service on bare metal or VM.

## Install

```bash
sudo useradd --system --create-home --home /var/lib/ctxai ctxai
sudo mkdir -p /opt/ctxai /etc/ctxai /var/lib/ctxai
sudo pip install ctxai[all] fastapi 'uvicorn[standard]'
sudo cp systemd/ctxai.service /etc/systemd/system/ctxai.service

# Configure secrets
sudo tee /etc/ctxai/env <<'EOF'
OPENROUTER_API_KEY=...
ANTHROPIC_API_KEY=...
EOF
sudo chmod 600 /etc/ctxai/env

sudo systemctl daemon-reload
sudo systemctl enable --now ctxai.service
```

## Inspect

```bash
sudo systemctl status ctxai
sudo journalctl -u ctxai -f
```

## Reload

```bash
sudo systemctl restart ctxai      # full restart
sudo systemctl reload ctxai       # SIGHUP (config reload)
```

## Hardening

The unit file enables:
- `NoNewPrivileges` — block setuid escalation.
- `ProtectSystem=full`, `ProtectHome=yes` — read-only system, no home access.
- `PrivateTmp` — process-private `/tmp`.
- Automatic restart on failure with a 5-second backoff.
