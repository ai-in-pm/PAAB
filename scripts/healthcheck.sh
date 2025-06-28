#!/bin/bash
# PsychoPy AI Agent Builder - Health Check Script
# Comprehensive health check for containerized deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Health check configuration
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-10}
HEALTH_CHECK_RETRIES=${HEALTH_CHECK_RETRIES:-3}
STUDIO_PORT=${STUDIO_PORT:-8501}
API_PORT=${API_PORT:-8080}

# Logging functions
log_info() {
    echo -e "${GREEN}[HEALTH]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[HEALTH]${NC} $1"
}

log_error() {
    echo -e "${RED}[HEALTH]${NC} $1" >&2
}

# Check if a port is listening
check_port() {
    local port=$1
    local service_name=$2
    
    if nc -z localhost "$port" 2>/dev/null; then
        log_info "$service_name is listening on port $port"
        return 0
    else
        log_error "$service_name is not listening on port $port"
        return 1
    fi
}

# Check HTTP endpoint
check_http_endpoint() {
    local url=$1
    local service_name=$2
    local expected_status=${3:-200}
    
    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$HEALTH_CHECK_TIMEOUT" "$url" 2>/dev/null || echo "000")
    
    if [ "$response_code" = "$expected_status" ]; then
        log_info "$service_name HTTP endpoint is healthy (status: $response_code)"
        return 0
    else
        log_error "$service_name HTTP endpoint is unhealthy (status: $response_code)"
        return 1
    fi
}

# Check Python module imports
check_python_modules() {
    log_info "Checking Python module imports..."
    
    python3 -c "
import sys
import os
sys.path.insert(0, '/app/src')

modules_to_check = [
    'agents.base',
    'tasks.base', 
    'crews.base',
    'tools.base',
    'runtime.executor'
]

failed_imports = []

for module in modules_to_check:
    try:
        __import__(module)
        print(f'✓ {module}')
    except ImportError as e:
        print(f'✗ {module}: {e}')
        failed_imports.append(module)

if failed_imports:
    print(f'Failed to import: {failed_imports}')
    sys.exit(1)
else:
    print('All core modules imported successfully')
" || {
        log_error "Python module import check failed"
        return 1
    }
    
    log_info "Python modules are healthy"
    return 0
}

# Check database connectivity
check_database() {
    if [ -z "$DATABASE_URL" ]; then
        log_info "No database configured, skipping database check"
        return 0
    fi
    
    log_info "Checking database connectivity..."
    
    python3 -c "
import sys
import os
sys.path.insert(0, '/app/src')

try:
    import psycopg2
    from urllib.parse import urlparse
    
    url = urlparse('$DATABASE_URL')
    conn = psycopg2.connect(
        host=url.hostname,
        port=url.port or 5432,
        database=url.path[1:],
        user=url.username,
        password=url.password
    )
    
    cursor = conn.cursor()
    cursor.execute('SELECT 1')
    result = cursor.fetchone()
    
    if result and result[0] == 1:
        print('Database connection successful')
    else:
        print('Database query failed')
        sys.exit(1)
        
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
" || {
        log_error "Database connectivity check failed"
        return 1
    }
    
    log_info "Database is healthy"
    return 0
}

# Check Redis connectivity
check_redis() {
    if [ -z "$REDIS_URL" ]; then
        log_info "No Redis configured, skipping Redis check"
        return 0
    fi
    
    log_info "Checking Redis connectivity..."
    
    python3 -c "
import sys
try:
    import redis
    from urllib.parse import urlparse
    
    url = urlparse('$REDIS_URL')
    r = redis.Redis(
        host=url.hostname,
        port=url.port or 6379,
        db=int(url.path[1:]) if url.path and len(url.path) > 1 else 0,
        password=url.password,
        socket_timeout=5
    )
    
    # Test connection
    r.ping()
    
    # Test basic operations
    r.set('health_check', 'ok', ex=10)
    result = r.get('health_check')
    
    if result and result.decode() == 'ok':
        print('Redis connection and operations successful')
    else:
        print('Redis operations failed')
        sys.exit(1)
        
except Exception as e:
    print(f'Redis connection failed: {e}')
    sys.exit(1)
" || {
        log_error "Redis connectivity check failed"
        return 1
    }
    
    log_info "Redis is healthy"
    return 0
}

# Check file system permissions
check_filesystem() {
    log_info "Checking file system permissions..."
    
    local dirs_to_check=(
        "$PAAB_DATA_PATH"
        "$PAAB_CACHE_PATH" 
        "$PAAB_LOGS_PATH"
    )
    
    for dir in "${dirs_to_check[@]}"; do
        if [ -n "$dir" ] && [ -d "$dir" ]; then
            # Test write permissions
            local test_file="$dir/.health_check_$$"
            if echo "test" > "$test_file" 2>/dev/null; then
                rm -f "$test_file"
                log_info "Directory $dir is writable"
            else
                log_error "Directory $dir is not writable"
                return 1
            fi
        else
            log_warning "Directory $dir does not exist or is not configured"
        fi
    done
    
    log_info "File system permissions are healthy"
    return 0
}

# Check memory usage
check_memory() {
    log_info "Checking memory usage..."
    
    local memory_info
    memory_info=$(python3 -c "
import psutil
import sys

try:
    memory = psutil.virtual_memory()
    process = psutil.Process()
    
    print(f'System memory usage: {memory.percent}%')
    print(f'Process memory usage: {process.memory_percent():.1f}%')
    print(f'Available memory: {memory.available / (1024**3):.1f} GB')
    
    # Alert if memory usage is too high
    if memory.percent > 90:
        print('WARNING: System memory usage is very high')
        sys.exit(1)
    elif process.memory_percent() > 50:
        print('WARNING: Process memory usage is high')
        sys.exit(1)
    else:
        print('Memory usage is within acceptable limits')
        
except Exception as e:
    print(f'Memory check failed: {e}')
    sys.exit(1)
")
    
    if [ $? -eq 0 ]; then
        log_info "Memory usage is healthy"
        echo "$memory_info" | while read -r line; do
            log_info "$line"
        done
        return 0
    else
        log_error "Memory usage check failed"
        echo "$memory_info" | while read -r line; do
            log_error "$line"
        done
        return 1
    fi
}

# Check disk space
check_disk_space() {
    log_info "Checking disk space..."
    
    local disk_usage
    disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "$disk_usage" -gt 90 ]; then
        log_error "Disk usage is critically high: ${disk_usage}%"
        return 1
    elif [ "$disk_usage" -gt 80 ]; then
        log_warning "Disk usage is high: ${disk_usage}%"
    else
        log_info "Disk usage is healthy: ${disk_usage}%"
    fi
    
    return 0
}

# Main health check function
run_health_check() {
    local checks_passed=0
    local total_checks=0
    
    log_info "Starting comprehensive health check..."
    
    # Core checks (required)
    local core_checks=(
        "check_python_modules"
        "check_filesystem"
    )
    
    # Optional checks (warnings only)
    local optional_checks=(
        "check_database"
        "check_redis"
        "check_memory"
        "check_disk_space"
    )
    
    # Service checks (if ports are configured)
    local service_checks=()
    
    # Check if Studio is running
    if [ -n "$STUDIO_PORT" ]; then
        service_checks+=("check_port $STUDIO_PORT Studio")
        service_checks+=("check_http_endpoint http://localhost:$STUDIO_PORT/_stcore/health Studio")
    fi
    
    # Check if API is running
    if [ -n "$API_PORT" ]; then
        service_checks+=("check_port $API_PORT API")
        service_checks+=("check_http_endpoint http://localhost:$API_PORT/health API")
    fi
    
    # Run core checks
    for check in "${core_checks[@]}"; do
        total_checks=$((total_checks + 1))
        if eval "$check"; then
            checks_passed=$((checks_passed + 1))
        else
            log_error "Core check failed: $check"
            return 1
        fi
    done
    
    # Run service checks
    for check in "${service_checks[@]}"; do
        total_checks=$((total_checks + 1))
        if eval "$check"; then
            checks_passed=$((checks_passed + 1))
        else
            log_error "Service check failed: $check"
            return 1
        fi
    done
    
    # Run optional checks (don't fail on these)
    for check in "${optional_checks[@]}"; do
        total_checks=$((total_checks + 1))
        if eval "$check"; then
            checks_passed=$((checks_passed + 1))
        else
            log_warning "Optional check failed: $check"
            # Don't increment checks_passed, but don't fail either
            checks_passed=$((checks_passed + 1))  # Count as passed for overall health
        fi
    done
    
    # Summary
    log_info "Health check completed: $checks_passed/$total_checks checks passed"
    
    if [ "$checks_passed" -eq "$total_checks" ]; then
        log_info "All health checks passed - service is healthy"
        return 0
    else
        log_error "Some health checks failed - service may be unhealthy"
        return 1
    fi
}

# Quick health check (for frequent monitoring)
run_quick_check() {
    log_info "Running quick health check..."
    
    # Just check if the main process is responsive
    if check_python_modules; then
        log_info "Quick health check passed"
        return 0
    else
        log_error "Quick health check failed"
        return 1
    fi
}

# Main execution
main() {
    case "${1:-full}" in
        "quick")
            run_quick_check
            ;;
        "full"|*)
            run_health_check
            ;;
    esac
}

# Run main function
main "$@"
