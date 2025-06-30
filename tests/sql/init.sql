-- Test Database Initialization Script
--
-- This script sets up the test database schema for orderbook testing
-- including tables for storing test results, performance metrics, and test data

-- Create test database user and schema
CREATE ROLE test_user WITH LOGIN PASSWORD 'test_password';
CREATE DATABASE orderbook_test OWNER test_user;

-- Connect to test database
\c orderbook_test;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE orderbook_test TO test_user;
GRANT ALL ON SCHEMA public TO test_user;

-- =============================================================================
-- TEST RESULTS TABLES
-- =============================================================================

-- Test runs table
CREATE TABLE IF NOT EXISTS test_runs (
    id SERIAL PRIMARY KEY,
    run_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    test_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    duration_seconds INTEGER,
    total_tests INTEGER,
    passed_tests INTEGER,
    failed_tests INTEGER,
    skipped_tests INTEGER,
    environment JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Individual test results
CREATE TABLE IF NOT EXISTS test_results (
    id SERIAL PRIMARY KEY,
    test_run_id INTEGER REFERENCES test_runs(id),
    test_name VARCHAR(255) NOT NULL,
    test_module VARCHAR(255) NOT NULL,
    test_category VARCHAR(50),
    status VARCHAR(20) NOT NULL,
    duration_seconds DECIMAL(10, 6),
    error_message TEXT,
    error_traceback TEXT,
    parameters JSONB,
    assertions JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance benchmarks
CREATE TABLE IF NOT EXISTS performance_benchmarks (
    id SERIAL PRIMARY KEY,
    test_run_id INTEGER REFERENCES test_runs(id),
    benchmark_name VARCHAR(255) NOT NULL,
    operation_name VARCHAR(100) NOT NULL,
    iterations INTEGER NOT NULL,
    total_time_seconds DECIMAL(15, 9) NOT NULL,
    mean_time_seconds DECIMAL(15, 9) NOT NULL,
    median_time_seconds DECIMAL(15, 9) NOT NULL,
    min_time_seconds DECIMAL(15, 9) NOT NULL,
    max_time_seconds DECIMAL(15, 9) NOT NULL,
    std_dev_seconds DECIMAL(15, 9) NOT NULL,
    ops_per_second DECIMAL(15, 3) NOT NULL,
    memory_usage_mb DECIMAL(10, 3),
    parameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- ORDERBOOK TEST DATA TABLES
-- =============================================================================

-- Mock orderbook snapshots for testing
CREATE TABLE IF NOT EXISTS mock_orderbook_snapshots (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    sequence_number BIGINT NOT NULL,
    bids JSONB NOT NULL,
    asks JSONB NOT NULL,
    mid_price DECIMAL(20, 8),
    spread DECIMAL(20, 8),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Mock trade data
CREATE TABLE IF NOT EXISTS mock_trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    side VARCHAR(10) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    size DECIMAL(20, 8) NOT NULL,
    trade_id VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Mock WebSocket messages
CREATE TABLE IF NOT EXISTS mock_websocket_messages (
    id SERIAL PRIMARY KEY,
    channel VARCHAR(50) NOT NULL,
    message_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    sequence_number BIGINT,
    message_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- STRESS TEST TABLES
-- =============================================================================

-- Connection stress test results
CREATE TABLE IF NOT EXISTS stress_test_connections (
    id SERIAL PRIMARY KEY,
    test_run_id INTEGER REFERENCES test_runs(id),
    connection_id VARCHAR(50) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL,
    messages_sent INTEGER DEFAULT 0,
    messages_received INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    latency_stats JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Message throughput test results
CREATE TABLE IF NOT EXISTS stress_test_throughput (
    id SERIAL PRIMARY KEY,
    test_run_id INTEGER REFERENCES test_runs(id),
    test_duration_seconds INTEGER NOT NULL,
    target_messages_per_second INTEGER NOT NULL,
    actual_messages_per_second DECIMAL(10, 3) NOT NULL,
    total_messages INTEGER NOT NULL,
    successful_messages INTEGER NOT NULL,
    failed_messages INTEGER NOT NULL,
    average_latency_ms DECIMAL(10, 3),
    p95_latency_ms DECIMAL(10, 3),
    p99_latency_ms DECIMAL(10, 3),
    memory_usage_stats JSONB,
    cpu_usage_stats JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- PROPERTY TESTING TABLES
-- =============================================================================

-- Hypothesis test results
CREATE TABLE IF NOT EXISTS property_test_results (
    id SERIAL PRIMARY KEY,
    test_run_id INTEGER REFERENCES test_runs(id),
    property_name VARCHAR(255) NOT NULL,
    examples_generated INTEGER NOT NULL,
    examples_valid INTEGER NOT NULL,
    examples_invalid INTEGER NOT NULL,
    falsifying_examples JSONB,
    seed INTEGER,
    status VARCHAR(20) NOT NULL,
    execution_time_seconds DECIMAL(10, 6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Test runs indexes
CREATE INDEX idx_test_runs_timestamp ON test_runs(run_timestamp);
CREATE INDEX idx_test_runs_type ON test_runs(test_type);
CREATE INDEX idx_test_runs_status ON test_runs(status);

-- Test results indexes
CREATE INDEX idx_test_results_run_id ON test_results(test_run_id);
CREATE INDEX idx_test_results_name ON test_results(test_name);
CREATE INDEX idx_test_results_status ON test_results(status);
CREATE INDEX idx_test_results_category ON test_results(test_category);

-- Performance benchmarks indexes
CREATE INDEX idx_performance_benchmarks_run_id ON performance_benchmarks(test_run_id);
CREATE INDEX idx_performance_benchmarks_name ON performance_benchmarks(benchmark_name);
CREATE INDEX idx_performance_benchmarks_ops_per_second ON performance_benchmarks(ops_per_second);

-- Orderbook data indexes
CREATE INDEX idx_mock_orderbook_symbol ON mock_orderbook_snapshots(symbol);
CREATE INDEX idx_mock_orderbook_timestamp ON mock_orderbook_snapshots(timestamp);
CREATE INDEX idx_mock_orderbook_sequence ON mock_orderbook_snapshots(sequence_number);

CREATE INDEX idx_mock_trades_symbol ON mock_trades(symbol);
CREATE INDEX idx_mock_trades_timestamp ON mock_trades(timestamp);

CREATE INDEX idx_mock_websocket_channel ON mock_websocket_messages(channel);
CREATE INDEX idx_mock_websocket_timestamp ON mock_websocket_messages(timestamp);

-- Stress test indexes
CREATE INDEX idx_stress_connections_run_id ON stress_test_connections(test_run_id);
CREATE INDEX idx_stress_throughput_run_id ON stress_test_throughput(test_run_id);

-- =============================================================================
-- VIEWS FOR REPORTING
-- =============================================================================

-- Test summary view
CREATE OR REPLACE VIEW test_summary AS
SELECT
    tr.id,
    tr.run_timestamp,
    tr.test_type,
    tr.status,
    tr.duration_seconds,
    tr.total_tests,
    tr.passed_tests,
    tr.failed_tests,
    tr.skipped_tests,
    ROUND(
        (tr.passed_tests::DECIMAL / NULLIF(tr.total_tests, 0)) * 100, 2
    ) AS success_rate_percent,
    COUNT(tres.id) AS detailed_results_count
FROM test_runs tr
LEFT JOIN test_results tres ON tr.id = tres.test_run_id
GROUP BY tr.id, tr.run_timestamp, tr.test_type, tr.status, tr.duration_seconds,
         tr.total_tests, tr.passed_tests, tr.failed_tests, tr.skipped_tests;

-- Performance benchmark summary
CREATE OR REPLACE VIEW benchmark_summary AS
SELECT
    pb.benchmark_name,
    pb.operation_name,
    COUNT(*) AS run_count,
    AVG(pb.ops_per_second) AS avg_ops_per_second,
    MIN(pb.ops_per_second) AS min_ops_per_second,
    MAX(pb.ops_per_second) AS max_ops_per_second,
    STDDEV(pb.ops_per_second) AS stddev_ops_per_second,
    AVG(pb.mean_time_seconds) AS avg_mean_time_seconds,
    MAX(pb.created_at) AS last_run_time
FROM performance_benchmarks pb
GROUP BY pb.benchmark_name, pb.operation_name;

-- Recent test failures
CREATE OR REPLACE VIEW recent_test_failures AS
SELECT
    tr.test_name,
    tr.test_module,
    tr.test_category,
    tr.error_message,
    tr.duration_seconds,
    tr.created_at,
    trun.test_type,
    trun.run_timestamp
FROM test_results tr
JOIN test_runs trun ON tr.test_run_id = trun.id
WHERE tr.status = 'failed'
  AND tr.created_at > NOW() - INTERVAL '7 days'
ORDER BY tr.created_at DESC;

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to cleanup old test data
CREATE OR REPLACE FUNCTION cleanup_old_test_data(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
    cutoff_date TIMESTAMP WITH TIME ZONE;
BEGIN
    cutoff_date := NOW() - (days_to_keep || ' days')::INTERVAL;

    -- Delete old test results
    DELETE FROM test_results
    WHERE test_run_id IN (
        SELECT id FROM test_runs WHERE created_at < cutoff_date
    );

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Delete old test runs
    DELETE FROM test_runs WHERE created_at < cutoff_date;

    -- Delete old mock data
    DELETE FROM mock_orderbook_snapshots WHERE created_at < cutoff_date;
    DELETE FROM mock_trades WHERE created_at < cutoff_date;
    DELETE FROM mock_websocket_messages WHERE created_at < cutoff_date;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SAMPLE DATA FOR TESTING
-- =============================================================================

-- Insert sample test run
INSERT INTO test_runs (
    test_type, status, duration_seconds, total_tests,
    passed_tests, failed_tests, skipped_tests,
    environment, metadata
) VALUES (
    'unit', 'completed', 45, 25, 23, 1, 1,
    '{"python_version": "3.12", "docker": true}',
    '{"git_commit": "abc123", "test_suite": "orderbook"}'
);

-- Grant all permissions to test user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO test_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO test_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO test_user;

-- Enable row level security (optional)
-- ALTER TABLE test_runs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE test_results ENABLE ROW LEVEL SECURITY;

COMMIT;
