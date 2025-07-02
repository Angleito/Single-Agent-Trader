#!/bin/bash
# Custom Check 8.2: Trading Network Security
# Validates network security for trading bot communications

check_8_2() {
    local id="8.2"
    local desc="Ensure trading bot network communications are secure"
    local remediation="Configure proper network isolation and encryption for trading communications"
    local remediationImpact="High - Prevents network-based attacks on trading systems"
    
    local totalChecks=0
    local totalFailed=0
    
    # Check for trading network isolation
    if docker network ls | grep -q "trading-network"; then
        totalChecks=$((totalChecks + 1))
        
        # Check network configuration
        network_info=$(docker network inspect trading-network)
        
        # Verify network is not using default bridge
        if echo "$network_info" | jq -r '.[0].Driver' | grep -q "bridge"; then
            # Check for custom bridge configuration
            if echo "$network_info" | jq -r '.[0].Name' | grep -q "bridge"; then
                warn "$id     * Trading network using default bridge - security risk"
                logjson "WARN" "$id" "$desc" "trading-network" "Using potentially insecure default bridge"
                totalFailed=$((totalFailed + 1))
            fi
        fi
        
        # Check for external connectivity restrictions
        if echo "$network_info" | jq -r '.[0].Internal' | grep -q "false"; then
            info "$id     * Trading network allows external connectivity"
        fi
    else
        warn "$id     * Trading network not found - containers may be using default network"
        logjson "WARN" "$id" "$desc" "system" "No custom trading network configured"
        totalFailed=$((totalFailed + 1))
    fi
    
    if [ $totalFailed -eq 0 ]; then
        pass "$id     * Trading network security configuration verified"
        logjson "PASS" "$id" "$desc"
    fi
}