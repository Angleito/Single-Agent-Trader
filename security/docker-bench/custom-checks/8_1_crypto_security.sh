#!/bin/bash
# Custom Check 8.1: Cryptocurrency Security Checks
# Validates security measures specific to cryptocurrency trading applications

check_8_1() {
    local id="8.1"
    local desc="Ensure cryptocurrency keys are not exposed in environment variables"
    local remediation="Remove sensitive keys from environment variables and use Docker secrets or external key management"
    local remediationImpact="High - Prevents unauthorized access to trading accounts"

    local totalChecks=0
    local totalFailed=0

    # Check for exposed private keys in environment variables
    for container in $(docker ps --format "{{.Names}}" | grep -E "(trading|bluefin|coinbase)"); do
        totalChecks=$((totalChecks + 1))

        # Check environment variables for sensitive patterns
        if docker exec "$container" env 2>/dev/null | grep -iE "(private_key|secret_key|api_key|mnemonic)" | grep -v "\\*\\*\\*"; then
            warn "$id     * Exposed cryptocurrency keys found in container: $container"
            logjson "WARN" "$id" "$desc" "$container" "Keys may be exposed in environment variables"
            totalFailed=$((totalFailed + 1))
        fi
    done

    if [ $totalFailed -eq 0 ]; then
        pass "$id     * No exposed cryptocurrency keys found in containers"
        logjson "PASS" "$id" "$desc"
    fi
}
