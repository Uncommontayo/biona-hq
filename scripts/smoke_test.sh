#!/usr/bin/env bash
# Biona post-deployment smoke tests
# Usage: ./scripts/smoke_test.sh <environment>

set -euo pipefail

ENVIRONMENT="${1:-staging}"
NAMESPACE="biona-${ENVIRONMENT}"

echo "Running smoke tests against biona-${ENVIRONMENT}..."

# ── Resolve service URLs ──────────────────────────────────────────────────────
LAB_URL=$(kubectl get svc biona-lab -n "${NAMESPACE}" \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || \
  kubectl get svc biona-lab -n "${NAMESPACE}" \
  -o jsonpath='{.spec.clusterIP}')

NORA_URL=$(kubectl get svc biona-nora-health -n "${NAMESPACE}" \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || \
  kubectl get svc biona-nora-health -n "${NAMESPACE}" \
  -o jsonpath='{.spec.clusterIP}')

# ── Lab health check ──────────────────────────────────────────────────────────
echo "Checking Biona Lab health..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  --max-time 10 "http://${LAB_URL}/health")

if [[ "$HTTP_STATUS" != "200" ]]; then
  echo "FAIL: Biona Lab health check returned HTTP $HTTP_STATUS"
  exit 1
fi
echo "PASS: Biona Lab healthy (HTTP 200)"

# ── Lab contract version check ────────────────────────────────────────────────
echo "Checking contract version endpoint..."
CONTRACT_VERSION=$(curl -s --max-time 10 \
  "http://${LAB_URL}/contract/version" | python3 -c \
  "import sys,json; print(json.load(sys.stdin).get('version','unknown'))")

if [[ "$CONTRACT_VERSION" == "unknown" ]]; then
  echo "FAIL: Could not retrieve contract version"
  exit 1
fi
echo "PASS: Contract version $CONTRACT_VERSION"

# ── Nora Health health check ──────────────────────────────────────────────────
echo "Checking Nora Health health..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  --max-time 15 "http://${NORA_URL}/")

if [[ "$HTTP_STATUS" != "200" ]]; then
  echo "FAIL: Nora Health returned HTTP $HTTP_STATUS"
  exit 1
fi
echo "PASS: Nora Health healthy (HTTP 200)"

# ── Pod status check ──────────────────────────────────────────────────────────
echo "Checking pod status..."
UNHEALTHY_PODS=$(kubectl get pods -n "${NAMESPACE}" \
  --field-selector=status.phase!=Running \
  --no-headers 2>/dev/null | grep -v "Completed" | wc -l)

if [[ "$UNHEALTHY_PODS" -gt "0" ]]; then
  echo "FAIL: $UNHEALTHY_PODS unhealthy pods in namespace ${NAMESPACE}"
  kubectl get pods -n "${NAMESPACE}"
  exit 1
fi
echo "PASS: All pods running in ${NAMESPACE}"

echo ""
echo "All smoke tests passed for biona-${ENVIRONMENT}"
