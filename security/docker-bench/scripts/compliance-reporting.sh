#!/bin/bash

# Docker Security Compliance Reporting System
# Generates comprehensive security compliance reports and metrics for AI Trading Bot

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="${BASE_DIR}/reports"
TEMPLATES_DIR="${BASE_DIR}/templates"
CONFIG_DIR="${BASE_DIR}/config"
LOGS_DIR="${BASE_DIR}/logs"

# Load configuration
COMPLIANCE_CONFIG="${CONFIG_DIR}/compliance.conf"
if [ -f "$COMPLIANCE_CONFIG" ]; then
    source "$COMPLIANCE_CONFIG"
fi

# Default configuration
REPORT_FORMAT=${REPORT_FORMAT:-"html,json,pdf"}
COMPLIANCE_STANDARDS=${COMPLIANCE_STANDARDS:-"cis-docker,nist-csf,trading-security"}
HISTORICAL_DAYS=${HISTORICAL_DAYS:-30}
TREND_ANALYSIS_ENABLED=${TREND_ANALYSIS_ENABLED:-true}
EXECUTIVE_SUMMARY=${EXECUTIVE_SUMMARY:-true}

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOGS_DIR}/compliance-reporting.log"
}

info() { log "INFO" "${BLUE}$1${NC}"; }
success() { log "SUCCESS" "${GREEN}$1${NC}"; }
warning() { log "WARNING" "${YELLOW}$1${NC}"; }
error() { log "ERROR" "${RED}$1${NC}"; }

# Ensure required directories exist
mkdir -p "$REPORTS_DIR" "$TEMPLATES_DIR" "$LOGS_DIR"

# Data collection functions

# Collect current security status
collect_current_status() {
    local output_file="$1"

    info "Collecting current security status..."

    # Get latest security scan results
    local latest_scan
    latest_scan=$(ls -t "$REPORTS_DIR"/security-analysis-*.json 2>/dev/null | head -1)

    if [ -z "$latest_scan" ] || [ ! -f "$latest_scan" ]; then
        warning "No recent security scan found - running new scan"
        "${SCRIPT_DIR}/run-security-scan.sh" > /dev/null
        latest_scan=$(ls -t "$REPORTS_DIR"/security-analysis-*.json 2>/dev/null | head -1)
    fi

    local current_status="{}"
    if [ -f "$latest_scan" ] && command -v jq &> /dev/null; then
        current_status=$(jq '.' "$latest_scan")
    fi

    echo "$current_status" > "$output_file"
    success "Current security status collected"
}

# Collect historical data for trend analysis
collect_historical_data() {
    local output_file="$1"
    local days="$2"

    info "Collecting historical data for last $days days..."

    local historical_data="[]"

    # Find all analysis files within the specified time range
    local cutoff_date
    cutoff_date=$(date -d "$days days ago" '+%Y%m%d')

    if command -v jq &> /dev/null; then
        for analysis_file in "$REPORTS_DIR"/security-analysis-*.json; do
            if [ -f "$analysis_file" ]; then
                # Extract date from filename
                local file_date
                file_date=$(basename "$analysis_file" | grep -o '[0-9]\{8\}' | head -1)

                if [ -n "$file_date" ] && [ "$file_date" -ge "$cutoff_date" ]; then
                    local scan_data
                    scan_data=$(jq ". + {\"scan_date\": \"$file_date\"}" "$analysis_file" 2>/dev/null || echo "{}")
                    historical_data=$(echo "$historical_data" | jq ". + [$scan_data]")
                fi
            fi
        done
    fi

    echo "$historical_data" > "$output_file"
    success "Historical data collected ($(echo "$historical_data" | jq length) scans)"
}

# Calculate compliance scores
calculate_compliance_scores() {
    local current_status_file="$1"
    local output_file="$2"

    info "Calculating compliance scores..."

    if ! command -v jq &> /dev/null; then
        echo '{"error": "jq not available for compliance calculation"}' > "$output_file"
        return 1
    fi

    local current_status
    current_status=$(cat "$current_status_file")

    # CIS Docker Benchmark compliance
    local cis_total_checks
    local cis_passed_checks
    local cis_critical_failures
    local cis_high_failures

    cis_total_checks=$(echo "$current_status" | jq -r '.summary.total_checks // 0')
    cis_passed_checks=$(echo "$current_status" | jq -r '.summary.passed_checks // 0')
    cis_critical_failures=$(echo "$current_status" | jq -r '.summary.critical_issues // 0')
    cis_high_failures=$(echo "$current_status" | jq -r '.summary.high_issues // 0')

    local cis_score=0
    if [ "$cis_total_checks" -gt 0 ]; then
        cis_score=$(( cis_passed_checks * 100 / cis_total_checks ))
    fi

    # NIST Cybersecurity Framework alignment
    local nist_score=100

    # Deduct points for critical issues (more severe penalty)
    nist_score=$(( nist_score - (cis_critical_failures * 25) ))

    # Deduct points for high issues
    nist_score=$(( nist_score - (cis_high_failures * 10) ))

    # Ensure score doesn't go below 0
    if [ "$nist_score" -lt 0 ]; then
        nist_score=0
    fi

    # Trading-specific security score
    local trading_score=100

    # Check for trading-specific security issues
    local containers_checked=0
    local containers_compliant=0

    for container in ai-trading-bot bluefin-service dashboard-backend; do
        if docker ps -q -f name="$container" &> /dev/null; then
            containers_checked=$((containers_checked + 1))

            # Check key security configurations
            local user_config
            local privileged_mode
            local readonly_fs

            user_config=$(docker inspect "$container" --format '{{.Config.User}}' 2>/dev/null || echo "")
            privileged_mode=$(docker inspect "$container" --format '{{.HostConfig.Privileged}}' 2>/dev/null || echo "false")
            readonly_fs=$(docker inspect "$container" --format '{{.HostConfig.ReadonlyRootfs}}' 2>/dev/null || echo "false")

            # Container is compliant if it has proper user, is not privileged, and has readonly fs
            if [ -n "$user_config" ] && [ "$user_config" != "root" ] && [ "$privileged_mode" = "false" ] && [ "$readonly_fs" = "true" ]; then
                containers_compliant=$((containers_compliant + 1))
            fi
        fi
    done

    if [ "$containers_checked" -gt 0 ]; then
        trading_score=$(( containers_compliant * 100 / containers_checked ))
    fi

    # Overall security posture score (weighted average)
    local overall_score=$(( (cis_score * 40 + nist_score * 30 + trading_score * 30) / 100 ))

    # Generate compliance report
    cat > "$output_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "compliance_scores": {
        "cis_docker_benchmark": {
            "score": $cis_score,
            "total_checks": $cis_total_checks,
            "passed_checks": $cis_passed_checks,
            "critical_failures": $cis_critical_failures,
            "high_failures": $cis_high_failures,
            "status": "$([ $cis_score -ge 80 ] && echo "COMPLIANT" || echo "NON_COMPLIANT")"
        },
        "nist_cybersecurity_framework": {
            "score": $nist_score,
            "status": "$([ $nist_score -ge 75 ] && echo "COMPLIANT" || echo "NON_COMPLIANT")",
            "critical_gaps": $cis_critical_failures,
            "high_gaps": $cis_high_failures
        },
        "trading_security": {
            "score": $trading_score,
            "containers_checked": $containers_checked,
            "containers_compliant": $containers_compliant,
            "status": "$([ $trading_score -ge 85 ] && echo "COMPLIANT" || echo "NON_COMPLIANT")"
        },
        "overall_security_posture": {
            "score": $overall_score,
            "rating": "$(
                if [ $overall_score -ge 90 ]; then echo "EXCELLENT"
                elif [ $overall_score -ge 80 ]; then echo "GOOD"
                elif [ $overall_score -ge 70 ]; then echo "FAIR"
                elif [ $overall_score -ge 60 ]; then echo "POOR"
                else echo "CRITICAL"
                fi
            )"
        }
    },
    "recommendations": [
        $([ $cis_critical_failures -gt 0 ] && echo '"Address all critical CIS Docker Benchmark failures immediately",' || echo '')
        $([ $cis_high_failures -gt 0 ] && echo '"Remediate high-severity security issues within 24 hours",' || echo '')
        $([ $trading_score -lt 85 ] && echo '"Enhance trading-specific container security configurations",' || echo '')
        $([ $overall_score -lt 80 ] && echo '"Implement comprehensive security hardening program",' || echo '')
        "Maintain regular security scanning and monitoring"
    ]
}
EOF

    success "Compliance scores calculated - Overall score: $overall_score%"
}

# Generate trend analysis
generate_trend_analysis() {
    local historical_data_file="$1"
    local output_file="$2"

    info "Generating trend analysis..."

    if ! command -v jq &> /dev/null; then
        echo '{"error": "jq not available for trend analysis"}' > "$output_file"
        return 1
    fi

    local historical_data
    historical_data=$(cat "$historical_data_file")

    # Calculate trends
    local trend_data
    trend_data=$(echo "$historical_data" | jq '
    [
        .[] | {
            date: .scan_date,
            critical: (.summary.critical_issues // 0),
            high: (.summary.high_issues // 0),
            medium: (.summary.medium_issues // 0),
            low: (.summary.low_issues // 0),
            total_checks: (.summary.total_checks // 0),
            passed_checks: (.summary.passed_checks // 0)
        }
    ] | sort_by(.date)
    ')

    # Calculate trends and insights
    local trend_summary
    trend_summary=$(echo "$trend_data" | jq '
    {
        "trend_period_days": length,
        "data_points": .,
        "trends": {
            "critical_issues": {
                "current": (last.critical // 0),
                "previous": (first.critical // 0),
                "trend": (
                    if (first.critical // 0) == 0 and (last.critical // 0) == 0 then "STABLE"
                    elif (last.critical // 0) > (first.critical // 0) then "INCREASING"
                    elif (last.critical // 0) < (first.critical // 0) then "DECREASING"
                    else "STABLE"
                    end
                )
            },
            "high_issues": {
                "current": (last.high // 0),
                "previous": (first.high // 0),
                "trend": (
                    if (first.high // 0) == 0 and (last.high // 0) == 0 then "STABLE"
                    elif (last.high // 0) > (first.high // 0) then "INCREASING"
                    elif (last.high // 0) < (first.high // 0) then "DECREASING"
                    else "STABLE"
                    end
                )
            },
            "compliance_score": {
                "current": (if (last.total_checks // 0) > 0 then ((last.passed_checks // 0) * 100 / (last.total_checks // 0)) else 0 end),
                "previous": (if (first.total_checks // 0) > 0 then ((first.passed_checks // 0) * 100 / (first.total_checks // 0)) else 0 end),
                "trend": (
                    if (first.total_checks // 0) == 0 then "NO_DATA"
                    else
                        (
                            (((last.passed_checks // 0) * 100 / (last.total_checks // 0)) -
                             ((first.passed_checks // 0) * 100 / (first.total_checks // 0))) |
                            if . > 5 then "IMPROVING"
                            elif . < -5 then "DEGRADING"
                            else "STABLE"
                            end
                        )
                    end
                )
            }
        }
    }
    ')

    echo "$trend_summary" > "$output_file"
    success "Trend analysis generated"
}

# Generate executive summary
generate_executive_summary() {
    local compliance_scores_file="$1"
    local trend_analysis_file="$2"
    local output_file="$3"

    info "Generating executive summary..."

    if ! command -v jq &> /dev/null; then
        echo "Executive summary requires jq" > "$output_file"
        return 1
    fi

    local compliance_data
    local trend_data
    compliance_data=$(cat "$compliance_scores_file")
    trend_data=$(cat "$trend_analysis_file")

    # Generate executive summary
    cat > "$output_file" << 'EOF'
# AI Trading Bot - Security Compliance Executive Summary

## Overview
This report provides a comprehensive assessment of the Docker security posture for the AI Trading Bot infrastructure.

## Key Findings

### Security Posture
EOF

    # Add dynamic content based on scores
    local overall_score
    overall_score=$(echo "$compliance_data" | jq -r '.compliance_scores.overall_security_posture.score')
    local overall_rating
    overall_rating=$(echo "$compliance_data" | jq -r '.compliance_scores.overall_security_posture.rating')

    cat >> "$output_file" << EOF

**Overall Security Score: ${overall_score}% (${overall_rating})**

EOF

    # Add compliance status for each standard
    echo "$compliance_data" | jq -r '
    .compliance_scores | to_entries[] |
    select(.key != "overall_security_posture") |
    "- **\(.key | gsub("_"; " ") | ascii_upcase)**: \(.value.score)% (\(.value.status))"
    ' >> "$output_file"

    # Add trend information if available
    if [ -f "$trend_analysis_file" ]; then
        cat >> "$output_file" << 'EOF'

### Security Trends
EOF

        echo "$trend_data" | jq -r '
        .trends | to_entries[] |
        "- **\(.key | gsub("_"; " ") | ascii_upcase)**: \(.value.trend)"
        ' >> "$output_file"
    fi

    # Add recommendations
    cat >> "$output_file" << 'EOF'

### Immediate Actions Required
EOF

    echo "$compliance_data" | jq -r '
    .recommendations[] | select(length > 0) | "- \(.)"
    ' >> "$output_file"

    cat >> "$output_file" << 'EOF'

### Risk Assessment

#### Critical Risks
EOF

    local critical_issues
    critical_issues=$(echo "$compliance_data" | jq -r '.compliance_scores.cis_docker_benchmark.critical_failures')

    if [ "$critical_issues" -gt 0 ]; then
        echo "- **${critical_issues} critical security issues** require immediate attention" >> "$output_file"
    else
        echo "- No critical security issues identified" >> "$output_file"
    fi

    cat >> "$output_file" << 'EOF'

#### High Priority Risks
EOF

    local high_issues
    high_issues=$(echo "$compliance_data" | jq -r '.compliance_scores.cis_docker_benchmark.high_failures')

    if [ "$high_issues" -gt 0 ]; then
        echo "- **${high_issues} high-severity issues** should be addressed within 24 hours" >> "$output_file"
    else
        echo "- No high-priority security issues identified" >> "$output_file"
    fi

    cat >> "$output_file" << EOF

### Trading Bot Specific Security

The trading bot infrastructure has achieved a **$(echo "$compliance_data" | jq -r '.compliance_scores.trading_security.score')%** security score for cryptocurrency-specific configurations.

**Container Security Status:**
EOF

    local containers_compliant
    local containers_checked
    containers_compliant=$(echo "$compliance_data" | jq -r '.compliance_scores.trading_security.containers_compliant')
    containers_checked=$(echo "$compliance_data" | jq -r '.compliance_scores.trading_security.containers_checked')

    echo "- ${containers_compliant}/${containers_checked} trading containers meet security requirements" >> "$output_file"

    cat >> "$output_file" << 'EOF'

### Next Steps

1. **Immediate**: Address any critical and high-severity security issues
2. **Short-term**: Implement automated remediation for common misconfigurations
3. **Long-term**: Establish continuous security monitoring and compliance checking
4. **Ongoing**: Regular security training and awareness programs

---
*Report generated on $(date)*
*System: $(hostname)*
EOF

    success "Executive summary generated"
}

# Generate HTML report
generate_html_report() {
    local compliance_scores_file="$1"
    local trend_analysis_file="$2"
    local executive_summary_file="$3"
    local output_file="$4"

    info "Generating HTML report..."

    cat > "$output_file" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trading Bot - Security Compliance Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .score-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .score-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .score-card.excellent { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); }
        .score-card.good { background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); }
        .score-card.fair { background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); }
        .score-card.poor { background: linear-gradient(135deg, #FF5722 0%, #D32F2F 100%); }
        .score-card.critical { background: linear-gradient(135deg, #F44336 0%, #B71C1C 100%); }
        .score-number {
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }
        .trend-chart {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .recommendations {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .critical-alert {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .status-badge {
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .status-compliant {
            background: #d4edda;
            color: #155724;
        }
        .status-non-compliant {
            background: #f8d7da;
            color: #721c24;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 AI Trading Bot Security Compliance Report</h1>
            <p>Comprehensive Docker Security Assessment</p>
            <p><strong>Generated:</strong> <span id="report-date"></span></p>
        </div>

        <div id="executive-summary">
            <h2>📊 Executive Summary</h2>
            <div id="summary-content"></div>
        </div>

        <div id="compliance-scores">
            <h2>🎯 Compliance Scores</h2>
            <div class="score-grid" id="score-grid"></div>
        </div>

        <div id="detailed-analysis">
            <h2>📈 Detailed Analysis</h2>
            <div id="analysis-content"></div>
        </div>

        <div id="recommendations">
            <h2>⚡ Recommendations</h2>
            <div class="recommendations" id="recommendations-content"></div>
        </div>

        <div class="footer">
            <p>AI Trading Bot Security Compliance System | Generated by Docker Bench Security Automation</p>
        </div>
    </div>

    <script>
        document.getElementById('report-date').textContent = new Date().toLocaleString();

        // Load and display compliance data
        function loadComplianceData() {
            // This would normally load from the JSON files
            // For now, we'll add placeholder functionality
            console.log('Loading compliance data...');
        }

        // Initialize report
        loadComplianceData();
    </script>
</body>
</html>
EOF

    success "HTML report template generated"
}

# Main report generation function
generate_compliance_report() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local report_prefix="${REPORTS_DIR}/compliance-${timestamp}"

    info "Starting compliance report generation..."

    # Temporary files for data collection
    local current_status_file="${report_prefix}-current.json"
    local historical_data_file="${report_prefix}-historical.json"
    local compliance_scores_file="${report_prefix}-scores.json"
    local trend_analysis_file="${report_prefix}-trends.json"
    local executive_summary_file="${report_prefix}-summary.md"

    # Collect data
    collect_current_status "$current_status_file"

    if [ "$TREND_ANALYSIS_ENABLED" = "true" ]; then
        collect_historical_data "$historical_data_file" "$HISTORICAL_DAYS"
        generate_trend_analysis "$historical_data_file" "$trend_analysis_file"
    fi

    # Calculate compliance scores
    calculate_compliance_scores "$current_status_file" "$compliance_scores_file"

    # Generate executive summary
    if [ "$EXECUTIVE_SUMMARY" = "true" ]; then
        generate_executive_summary "$compliance_scores_file" "$trend_analysis_file" "$executive_summary_file"
    fi

    # Generate reports in requested formats
    if echo "$REPORT_FORMAT" | grep -q "html"; then
        generate_html_report "$compliance_scores_file" "$trend_analysis_file" "$executive_summary_file" "${report_prefix}.html"
    fi

    if echo "$REPORT_FORMAT" | grep -q "json"; then
        # Combine all data into comprehensive JSON report
        if command -v jq &> /dev/null; then
            jq -s '
            {
                "timestamp": now | strftime("%Y-%m-%dT%H:%M:%S%Z"),
                "report_type": "compliance",
                "current_status": .[0],
                "compliance_scores": .[1],
                "trend_analysis": (if .[2] then .[2] else null end)
            }
            ' "$current_status_file" "$compliance_scores_file" "$trend_analysis_file" > "${report_prefix}.json"
        fi
    fi

    # Clean up temporary files
    rm -f "$current_status_file" "$historical_data_file"

    success "Compliance report generated: ${report_prefix}"

    # Display summary
    if command -v jq &> /dev/null && [ -f "$compliance_scores_file" ]; then
        echo
        info "Compliance Summary:"
        jq -r '.compliance_scores | to_entries[] | "  \(.key | gsub("_"; " ") | ascii_upcase): \(.value.score)% (\(.value.status // "N/A"))"' "$compliance_scores_file"
    fi

    echo "$report_prefix"
}

# Command line interface
case "${1:-generate}" in
    "generate")
        generate_compliance_report
        ;;
    "current-status")
        collect_current_status "${2:-/tmp/current-status.json}"
        ;;
    "historical")
        collect_historical_data "${2:-/tmp/historical.json}" "${3:-30}"
        ;;
    "scores")
        if [ $# -lt 2 ]; then
            error "Usage: $0 scores <current_status_file> [output_file]"
            exit 1
        fi
        calculate_compliance_scores "$2" "${3:-/tmp/compliance-scores.json}"
        ;;
    *)
        echo "Usage: $0 {generate|current-status|historical|scores}"
        echo "  generate          - Generate complete compliance report"
        echo "  current-status    - Collect current security status"
        echo "  historical        - Collect historical data"
        echo "  scores           - Calculate compliance scores"
        exit 1
        ;;
esac
