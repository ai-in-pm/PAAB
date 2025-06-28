#!/bin/bash
# PsychoPy AI Agent Builder - Docker Entrypoint Script
# Handles initialization and startup for containerized deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] PAAB:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

# Environment setup
setup_environment() {
    log "Setting up environment..."
    
    # Set default values
    export PAAB_ENV=${PAAB_ENV:-production}
    export PAAB_LOG_LEVEL=${PAAB_LOG_LEVEL:-INFO}
    export PAAB_DATA_PATH=${PAAB_DATA_PATH:-/app/data}
    export PAAB_CACHE_PATH=${PAAB_CACHE_PATH:-/app/cache}
    export PAAB_LOGS_PATH=${PAAB_LOGS_PATH:-/app/logs}
    export PAAB_MAX_CONCURRENT_CREWS=${PAAB_MAX_CONCURRENT_CREWS:-10}
    export PAAB_MAX_WORKERS=${PAAB_MAX_WORKERS:-50}
    
    # Create required directories
    mkdir -p "$PAAB_DATA_PATH" "$PAAB_CACHE_PATH" "$PAAB_LOGS_PATH"
    
    # Set permissions
    chmod 755 "$PAAB_DATA_PATH" "$PAAB_CACHE_PATH" "$PAAB_LOGS_PATH"
    
    log_success "Environment setup complete"
}

# Database initialization
init_database() {
    log "Initializing database connections..."
    
    # Wait for PostgreSQL if configured
    if [ -n "$DATABASE_URL" ]; then
        log "Waiting for PostgreSQL..."
        
        # Extract host and port from DATABASE_URL
        DB_HOST=$(echo "$DATABASE_URL" | sed -n 's/.*@\([^:]*\):.*/\1/p')
        DB_PORT=$(echo "$DATABASE_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
        
        if [ -n "$DB_HOST" ] && [ -n "$DB_PORT" ]; then
            timeout=30
            while ! nc -z "$DB_HOST" "$DB_PORT" 2>/dev/null; do
                timeout=$((timeout - 1))
                if [ $timeout -le 0 ]; then
                    log_error "PostgreSQL connection timeout"
                    exit 1
                fi
                log "Waiting for PostgreSQL at $DB_HOST:$DB_PORT... ($timeout seconds remaining)"
                sleep 1
            done
            log_success "PostgreSQL is ready"
        fi
    fi
    
    # Wait for Redis if configured
    if [ -n "$REDIS_URL" ]; then
        log "Waiting for Redis..."
        
        REDIS_HOST=$(echo "$REDIS_URL" | sed -n 's/redis:\/\/\([^:]*\):.*/\1/p')
        REDIS_PORT=$(echo "$REDIS_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
        
        if [ -n "$REDIS_HOST" ] && [ -n "$REDIS_PORT" ]; then
            timeout=30
            while ! nc -z "$REDIS_HOST" "$REDIS_PORT" 2>/dev/null; do
                timeout=$((timeout - 1))
                if [ $timeout -le 0 ]; then
                    log_error "Redis connection timeout"
                    exit 1
                fi
                log "Waiting for Redis at $REDIS_HOST:$REDIS_PORT... ($timeout seconds remaining)"
                sleep 1
            done
            log_success "Redis is ready"
        fi
    fi
    
    # Wait for MongoDB if configured
    if [ -n "$MONGODB_URL" ]; then
        log "Waiting for MongoDB..."
        
        MONGO_HOST=$(echo "$MONGODB_URL" | sed -n 's/mongodb:\/\/\([^:]*\):.*/\1/p')
        MONGO_PORT=$(echo "$MONGODB_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
        
        if [ -n "$MONGO_HOST" ] && [ -n "$MONGO_PORT" ]; then
            timeout=30
            while ! nc -z "$MONGO_HOST" "$MONGO_PORT" 2>/dev/null; do
                timeout=$((timeout - 1))
                if [ $timeout -le 0 ]; then
                    log_error "MongoDB connection timeout"
                    exit 1
                fi
                log "Waiting for MongoDB at $MONGO_HOST:$MONGO_PORT... ($timeout seconds remaining)"
                sleep 1
            done
            log_success "MongoDB is ready"
        fi
    fi
}

# Configuration validation
validate_config() {
    log "Validating configuration..."
    
    # Check required API keys for production
    if [ "$PAAB_ENV" = "production" ]; then
        if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
            log_warning "No LLM API keys configured. Some features may not work."
        fi
    fi
    
    # Validate resource limits
    if [ -n "$PAAB_MAX_CONCURRENT_CREWS" ]; then
        if ! [[ "$PAAB_MAX_CONCURRENT_CREWS" =~ ^[0-9]+$ ]] || [ "$PAAB_MAX_CONCURRENT_CREWS" -lt 1 ]; then
            log_error "Invalid PAAB_MAX_CONCURRENT_CREWS value: $PAAB_MAX_CONCURRENT_CREWS"
            exit 1
        fi
    fi
    
    if [ -n "$PAAB_MAX_WORKERS" ]; then
        if ! [[ "$PAAB_MAX_WORKERS" =~ ^[0-9]+$ ]] || [ "$PAAB_MAX_WORKERS" -lt 1 ]; then
            log_error "Invalid PAAB_MAX_WORKERS value: $PAAB_MAX_WORKERS"
            exit 1
        fi
    fi
    
    log_success "Configuration validation complete"
}

# Initialize application
init_application() {
    log "Initializing PAAB application..."
    
    # Run database migrations if needed
    if [ "$PAAB_ENV" = "production" ] && [ -n "$DATABASE_URL" ]; then
        log "Running database migrations..."
        python -c "
import sys
sys.path.insert(0, '/app/src')
try:
    from utils.database import run_migrations
    run_migrations()
    print('Database migrations completed')
except Exception as e:
    print(f'Migration failed: {e}')
    sys.exit(1)
" || {
            log_error "Database migration failed"
            exit 1
        }
    fi
    
    # Initialize cache if Redis is available
    if [ -n "$REDIS_URL" ]; then
        log "Initializing cache..."
        python -c "
import sys
sys.path.insert(0, '/app/src')
try:
    import redis
    from urllib.parse import urlparse
    url = urlparse('$REDIS_URL')
    r = redis.Redis(host=url.hostname, port=url.port, db=url.path[1:] if url.path else 0)
    r.ping()
    print('Cache connection verified')
except Exception as e:
    print(f'Cache initialization failed: {e}')
" || log_warning "Cache initialization failed, continuing without cache"
    fi
    
    # Verify PAAB installation
    python -c "
import sys
sys.path.insert(0, '/app/src')
try:
    import agents.base
    import tasks.base
    import crews.base
    import tools.base
    import runtime.executor
    print('PAAB modules loaded successfully')
except ImportError as e:
    print(f'PAAB module import failed: {e}')
    sys.exit(1)
" || {
        log_error "PAAB module verification failed"
        exit 1
    }
    
    log_success "Application initialization complete"
}

# Signal handlers for graceful shutdown
cleanup() {
    log "Received shutdown signal, cleaning up..."
    
    # Kill background processes
    if [ -n "$BACKGROUND_PID" ]; then
        kill "$BACKGROUND_PID" 2>/dev/null || true
    fi
    
    # Cleanup temporary files
    rm -rf /tmp/paab-* 2>/dev/null || true
    
    log_success "Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGQUIT

# Main execution
main() {
    log "Starting PsychoPy AI Agent Builder..."
    log "Version: ${VERSION:-unknown}"
    log "Environment: $PAAB_ENV"
    log "Log Level: $PAAB_LOG_LEVEL"
    
    # Setup
    setup_environment
    validate_config
    init_database
    init_application
    
    # Handle different startup modes
    if [ $# -eq 0 ]; then
        # Default: start studio
        log "Starting PAAB Studio..."
        exec python -m src.cli.main studio --host 0.0.0.0 --port 8501
    elif [ "$1" = "studio" ]; then
        # Studio mode
        log "Starting PAAB Studio..."
        shift
        exec python -m src.cli.main studio --host 0.0.0.0 --port 8501 "$@"
    elif [ "$1" = "api" ]; then
        # API server mode
        log "Starting PAAB API Server..."
        shift
        exec python -m src.api.server --host 0.0.0.0 --port 8080 "$@"
    elif [ "$1" = "worker" ]; then
        # Worker mode
        log "Starting PAAB Worker..."
        shift
        exec python -m src.runtime.worker "$@"
    elif [ "$1" = "cli" ]; then
        # CLI mode
        log "Starting PAAB CLI..."
        shift
        exec python -m src.cli.main "$@"
    elif [ "$1" = "test" ]; then
        # Test mode
        log "Running PAAB tests..."
        shift
        exec python -m pytest tests/ -v "$@"
    elif [ "$1" = "shell" ]; then
        # Interactive shell
        log "Starting interactive shell..."
        exec /bin/bash
    elif [ "$1" = "python" ]; then
        # Python interpreter
        log "Starting Python interpreter..."
        shift
        exec python "$@"
    else
        # Custom command
        log "Executing custom command: $*"
        exec "$@"
    fi
}

# Run main function
main "$@"
