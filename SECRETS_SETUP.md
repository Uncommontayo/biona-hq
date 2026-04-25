# Biona GitHub Secrets Setup Guide
# Add all of these at: github.com/<your-org>/biona/settings/secrets/actions

## Required secrets

### Infrastructure
KUBECONFIG                  # base64-encoded kubeconfig for your k8s cluster
                            # Generate: cat ~/.kube/config | base64 -w 0

### Container Registry
# GitHub Container Registry (ghcr.io) uses GITHUB_TOKEN automatically —
# no secret needed if your repo is under the same GitHub org.

### GPU Training (Lambda Labs)
LAMBDA_API_KEY              # From: cloud.lambdalabs.com/api-keys
LAMBDA_SSH_KEY              # Private key matching the "biona-ci" SSH key
                            # registered in Lambda Labs

### Biona Model Encryption
BIONA_AES_KEY               # 32-byte hex AES-256 key for .biona bundle encryption
                            # Generate: openssl rand -hex 32

### ML Experiment Tracking
MLFLOW_TRACKING_URI         # e.g. https://mlflow.yourdomain.com
WANDB_API_KEY               # From: wandb.ai/settings

### Notifications
SLACK_WEBHOOK_URL           # From: api.slack.com/apps → Incoming Webhooks

## Environment-specific secrets
# For production environment (Settings → Environments → production):
KUBECONFIG                  # Production cluster kubeconfig (different from staging)

## How to add a secret via GitHub CLI
# gh secret set BIONA_AES_KEY --body "$(openssl rand -hex 32)"
# gh secret set LAMBDA_API_KEY --body "your-lambda-key"
# gh secret set KUBECONFIG --body "$(cat ~/.kube/config | base64 -w 0)"

## Branch protection rules (recommended)
# main branch:
#   - Require status checks: ci-gate
#   - Require pull request reviews: 1 approver
#   - Restrict pushes: only GitHub Actions and repo admins
#   - Require signed commits: yes (once GPG keys are set up)

## Recommended repository settings
# Settings → Actions → General:
#   - Allow GitHub Actions to create and approve pull requests: OFF
#   - Default workflow permissions: Read repository contents and packages
