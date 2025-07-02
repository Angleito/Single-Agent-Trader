#!/bin/bash
# AI Trading Bot - Encrypted Volume Performance Optimization
# This script optimizes LUKS encrypted volumes for trading workload performance

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root for system optimization"
        error "Please run: sudo $0"
        exit 1
    fi
}

# Detect CPU capabilities
detect_cpu_features() {
    log "Detecting CPU features for optimization..."
    
    local cpu_features=""
    
    # Check for AES-NI support
    if grep -q "aes" /proc/cpuinfo; then
        cpu_features="$cpu_features AES-NI"
        log "✓ AES-NI hardware acceleration available"
    else
        warning "AES-NI not available - encryption will use software implementation"
    fi
    
    # Check for AVX support
    if grep -q "avx" /proc/cpuinfo; then
        cpu_features="$cpu_features AVX"
        log "✓ AVX vector instructions available"
    fi
    
    # Check for AVX2 support
    if grep -q "avx2" /proc/cpuinfo; then
        cpu_features="$cpu_features AVX2"
        log "✓ AVX2 vector instructions available"
    fi
    
    # Get CPU core count
    local cpu_cores=$(nproc)
    log "CPU cores available: $cpu_cores"
    
    # Get memory info
    local total_mem=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
    log "Total memory: ${total_mem}GB"
    
    success "CPU feature detection completed"
}

# Optimize LUKS parameters for performance
optimize_luks_performance() {
    log "Optimizing LUKS encrypted volumes for performance..."
    
    local volumes=("data" "logs" "config" "backup")
    
    for volume in "${volumes[@]}"; do
        local device_path="/dev/mapper/trading-${volume}"
        
        if [ -e "$device_path" ]; then
            log "Optimizing volume: $volume"
            
            # Enable discard/TRIM support for SSDs
            if lsblk -D | grep -q "trading-${volume}"; then
                log "Enabling TRIM support for $volume"
                cryptsetup refresh "trading-${volume}" --allow-discards 2>/dev/null || {
                    warning "Could not enable TRIM for $volume (may not be supported)"
                }
            fi
            
            # Tune filesystem parameters
            if [ "$(blkid -o value -s TYPE "$device_path")" = "ext4" ]; then
                log "Tuning ext4 filesystem for $volume"
                tune2fs -o journal_data_writeback "$device_path" 2>/dev/null || true
                tune2fs -O ^has_journal "$device_path" 2>/dev/null || {
                    warning "Could not disable journaling for $volume"
                }
            fi
        else
            warning "Volume $volume not found, skipping optimization"
        fi
    done
    
    success "LUKS volume optimization completed"
}

# Configure kernel parameters for encryption performance
optimize_kernel_parameters() {
    log "Optimizing kernel parameters for encryption performance..."
    
    # Create sysctl configuration for trading bot
    cat > /etc/sysctl.d/99-trading-bot-crypto.conf << 'EOF'
# AI Trading Bot - Crypto Performance Optimization

# VM parameters for better I/O performance
vm.dirty_ratio = 5
vm.dirty_background_ratio = 2
vm.dirty_expire_centisecs = 1000
vm.dirty_writeback_centisecs = 500

# Network buffer optimization for real-time data
net.core.rmem_default = 262144
net.core.rmem_max = 16777216
net.core.wmem_default = 262144
net.core.wmem_max = 16777216

# TCP optimization for trading connections
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.ipv4.tcp_congestion_control = bbr

# File descriptor limits
fs.file-max = 2097152

# Crypto-specific optimizations
kernel.random.read_wakeup_threshold = 64
kernel.random.write_wakeup_threshold = 128
EOF

    # Apply sysctl settings
    sysctl -p /etc/sysctl.d/99-trading-bot-crypto.conf
    
    success "Kernel parameters optimized"
}

# Configure I/O scheduler for encrypted volumes
optimize_io_scheduler() {
    log "Optimizing I/O scheduler for encrypted volumes..."
    
    # Find the underlying block devices
    local devices=$(lsblk -no NAME,TYPE | grep "loop" | awk '{print $1}')
    
    for device in $devices; do
        local device_path="/dev/$device"
        local scheduler_path="/sys/block/$device/queue/scheduler"
        
        if [ -f "$scheduler_path" ]; then
            # Set to deadline scheduler for better latency
            echo "deadline" > "$scheduler_path" 2>/dev/null || {
                # Fallback to mq-deadline for newer kernels
                echo "mq-deadline" > "$scheduler_path" 2>/dev/null || {
                    warning "Could not set I/O scheduler for $device"
                }
            }
            
            # Optimize queue depth
            echo 32 > "/sys/block/$device/queue/nr_requests" 2>/dev/null || true
            
            # Enable read-ahead optimization
            echo 128 > "/sys/block/$device/queue/read_ahead_kb" 2>/dev/null || true
            
            log "Optimized I/O scheduler for $device"
        fi
    done
    
    success "I/O scheduler optimization completed"
}

# Create performance monitoring script
create_performance_monitor() {
    log "Creating performance monitoring script..."
    
    cat > /opt/trading-bot/scripts/monitor-crypto-performance.sh << 'EOF'
#!/bin/bash
# Monitor encryption performance for AI Trading Bot

# Function to get I/O stats
get_io_stats() {
    local device=$1
    if [ -f "/sys/block/$device/stat" ]; then
        cat "/sys/block/$device/stat"
    fi
}

# Function to get LUKS performance stats
get_luks_stats() {
    local mapper_device=$1
    if [ -e "/dev/mapper/$mapper_device" ]; then
        cryptsetup status "$mapper_device" 2>/dev/null | grep -E "(cipher|sector size|offset)"
    fi
}

# Function to monitor CPU crypto usage
get_crypto_cpu_usage() {
    # Monitor crypto-related processes
    top -bn1 | grep -E "(kworker|crypto|luks)" | head -5
}

# Main monitoring loop
monitor_performance() {
    local duration=${1:-60}  # Default 60 seconds
    local interval=5
    local iterations=$((duration / interval))
    
    echo "=== AI Trading Bot Crypto Performance Monitor ==="
    echo "Monitoring for ${duration} seconds (${iterations} samples)..."
    echo ""
    
    for i in $(seq 1 $iterations); do
        echo "Sample $i/$(($iterations)) - $(date)"
        
        # LUKS device stats
        echo "LUKS Devices:"
        for volume in data logs config backup; do
            if [ -e "/dev/mapper/trading-$volume" ]; then
                echo "  trading-$volume:"
                get_luks_stats "trading-$volume" | sed 's/^/    /'
            fi
        done
        echo ""
        
        # I/O stats
        echo "I/O Statistics:"
        iostat -x 1 1 | grep -E "(Device|loop|dm-)" | head -10
        echo ""
        
        # Memory usage
        echo "Memory Usage:"
        free -h | head -2
        echo ""
        
        # Crypto CPU usage
        echo "Crypto-related processes:"
        get_crypto_cpu_usage
        echo ""
        
        if [ $i -lt $iterations ]; then
            echo "----------------------------------------"
            sleep $interval
        fi
    done
    
    echo "=== Monitoring Complete ==="
}

# Run monitoring
case "${1:-}" in
    "continuous")
        while true; do
            monitor_performance 300  # 5 minutes
            sleep 60
        done
        ;;
    *)
        monitor_performance "${1:-60}"
        ;;
esac
EOF

    chmod +x /opt/trading-bot/scripts/monitor-crypto-performance.sh
    
    success "Performance monitoring script created"
}

# Create crypto performance benchmarking script
create_crypto_benchmark() {
    log "Creating crypto performance benchmark..."
    
    cat > /opt/trading-bot/scripts/benchmark-crypto.sh << 'EOF'
#!/bin/bash
# Benchmark crypto performance for AI Trading Bot

# Test encryption/decryption speed
test_crypto_speed() {
    echo "=== Crypto Speed Benchmark ==="
    
    # Test different block sizes
    local sizes=("1K" "4K" "16K" "64K" "1M")
    local test_file="/tmp/crypto_test"
    
    for size in "${sizes[@]}"; do
        echo "Testing block size: $size"
        
        # Create test data
        dd if=/dev/urandom of="$test_file" bs="$size" count=1000 2>/dev/null
        
        # Time encryption
        local encrypt_time=$(time (gpg --batch --quiet --symmetric --cipher-algo AES256 --passphrase "test" --output "${test_file}.gpg" "$test_file") 2>&1 | grep "real" | awk '{print $2}')
        
        # Time decryption
        local decrypt_time=$(time (gpg --batch --quiet --decrypt --passphrase "test" "${test_file}.gpg" > "${test_file}.dec") 2>&1 | grep "real" | awk '{print $2}')
        
        # Calculate throughput
        local file_size=$(stat -c%s "$test_file")
        echo "  File size: $file_size bytes"
        echo "  Encrypt time: $encrypt_time"
        echo "  Decrypt time: $decrypt_time"
        echo ""
        
        # Cleanup
        rm -f "$test_file" "${test_file}.gpg" "${test_file}.dec"
    done
}

# Test filesystem performance on encrypted volumes
test_filesystem_performance() {
    echo "=== Filesystem Performance on Encrypted Volumes ==="
    
    local volumes=("data" "logs" "config")
    
    for volume in "${volumes[@]}"; do
        local mount_point="/mnt/trading-$volume"
        
        if mountpoint -q "$mount_point" 2>/dev/null; then
            echo "Testing volume: $volume ($mount_point)"
            
            # Sequential write test
            echo "  Sequential write test:"
            dd if=/dev/zero of="$mount_point/test_write" bs=1M count=100 2>&1 | grep -E "(copied|s,)"
            
            # Sequential read test
            echo "  Sequential read test:"
            dd if="$mount_point/test_write" of=/dev/null bs=1M 2>&1 | grep -E "(copied|s,)"
            
            # Random I/O test (if fio is available)
            if command -v fio >/dev/null 2>&1; then
                echo "  Random I/O test:"
                fio --name=random-test --ioengine=libaio --rw=randrw --bs=4k --size=100M --numjobs=1 --runtime=10 --directory="$mount_point" --output-format=terse 2>/dev/null | cut -d';' -f7,8,48,49 | tr ';' '\n' | sed 's/^/    /'
            fi
            
            # Cleanup
            rm -f "$mount_point/test_write" "$mount_point/fio-randrw-test"*
            echo ""
        else
            echo "Volume $volume not mounted, skipping"
        fi
    done
}

# Test network crypto performance
test_network_crypto() {
    echo "=== Network Crypto Performance ==="
    
    # Test SSL/TLS performance (simulating API connections)
    if command -v openssl >/dev/null 2>&1; then
        echo "OpenSSL speed test (simulating API encryption):"
        openssl speed -evp aes-256-gcm 2>/dev/null | tail -n 5 | head -n 1
        echo ""
    fi
}

# Run all benchmarks
echo "AI Trading Bot - Crypto Performance Benchmark"
echo "============================================="
echo ""

test_crypto_speed
test_filesystem_performance
test_network_crypto

echo "Benchmark completed - $(date)"
EOF

    chmod +x /opt/trading-bot/scripts/benchmark-crypto.sh
    
    success "Crypto benchmark script created"
}

# Configure tmpfs for high-performance temporary storage
optimize_tmpfs() {
    log "Optimizing tmpfs for high-performance temporary storage..."
    
    # Create tmpfs mount points for trading bot
    cat >> /etc/fstab << 'EOF'

# AI Trading Bot - High-performance tmpfs mounts
tmpfs /opt/trading-bot/tmp tmpfs defaults,noatime,nosuid,nodev,noexec,mode=1777,size=512M 0 0
tmpfs /opt/trading-bot/cache tmpfs defaults,noatime,nosuid,nodev,noexec,mode=755,size=256M 0 0
EOF

    # Create directories
    mkdir -p /opt/trading-bot/tmp
    mkdir -p /opt/trading-bot/cache
    
    # Mount tmpfs
    mount -a 2>/dev/null || {
        warning "Some tmpfs mounts may have failed"
    }
    
    success "Tmpfs optimization completed"
}

# Main execution
main() {
    log "Starting encrypted volume performance optimization..."
    
    check_root
    detect_cpu_features
    optimize_luks_performance
    optimize_kernel_parameters
    optimize_io_scheduler
    optimize_tmpfs
    create_performance_monitor
    create_crypto_benchmark
    
    success "Performance optimization completed!"
    
    cat << 'EOF'

=== Performance Optimization Summary ===

Optimizations Applied:
✓ LUKS volumes optimized for performance
✓ Kernel parameters tuned for crypto workloads
✓ I/O scheduler configured for low latency
✓ Tmpfs configured for high-performance temporary storage
✓ Performance monitoring tools created

Available Commands:
- Monitor performance: /opt/trading-bot/scripts/monitor-crypto-performance.sh
- Run benchmark: /opt/trading-bot/scripts/benchmark-crypto.sh
- Continuous monitoring: /opt/trading-bot/scripts/monitor-crypto-performance.sh continuous

Next Steps:
1. Run benchmark to establish baseline performance
2. Monitor performance during trading operations
3. Adjust Docker container resource limits if needed
4. Consider SSD/NVMe storage for better I/O performance

EOF
}

# Run main function
main "$@"