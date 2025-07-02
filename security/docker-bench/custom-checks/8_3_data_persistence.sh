#!/bin/bash
# Custom Check 8.3: Trading Data Security
# Validates security of persistent trading data

check_8_3() {
    local id="8.3"
    local desc="Ensure trading data volumes have appropriate security permissions"
    local remediation="Set proper file permissions and ownership on trading data directories"
    local remediationImpact="Medium - Prevents unauthorized access to trading data and logs"

    local totalChecks=0
    local totalFailed=0

    # Check trading data directories
    local data_dirs=("./data" "./logs" "./config")

    for dir in "${data_dirs[@]}"; do
        if [ -d "$dir" ]; then
            totalChecks=$((totalChecks + 1))

            # Check permissions (should not be world-writable)
            if [ "$(stat -c %a "$dir" | cut -c3)" = "7" ]; then
                warn "$id     * Directory $dir is world-writable - security risk"
                logjson "WARN" "$id" "$desc" "$dir" "World-writable permissions"
                totalFailed=$((totalFailed + 1))
            fi

            # Check for sensitive files with wrong permissions
            find "$dir" -name "*.json" -o -name "*.log" -o -name "*.key" | while read -r file; do
                if [ -f "$file" ] && [ "$(stat -c %a "$file" | cut -c2-3)" = "77" ]; then
                    warn "$id     * Sensitive file $file has permissive permissions"
                    logjson "WARN" "$id" "$desc" "$file" "Permissive file permissions"
                    totalFailed=$((totalFailed + 1))
                fi
            done
        fi
    done

    if [ $totalFailed -eq 0 ]; then
        pass "$id     * Trading data directories have appropriate security permissions"
        logjson "PASS" "$id" "$desc"
    fi
}
