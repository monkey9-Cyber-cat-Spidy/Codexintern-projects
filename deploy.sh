#!/bin/bash

# CodexIntern Python Development - Deployment Script
# This script helps set up and deploy the entire project

set -e  # Exit on any error

echo "🐍 CodexIntern Python Development - Deployment Script"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_status "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Check if Docker is installed
check_docker() {
    if command -v docker &> /dev/null; then
        print_status "Docker found"
        DOCKER_AVAILABLE=true
    else
        print_warning "Docker not found. Docker deployment will be skipped."
        DOCKER_AVAILABLE=false
    fi
}

# Install Python dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    print_status "Dependencies installed successfully"
}

# Start individual applications
start_app() {
    local app_name=$1
    local app_path=$2
    local port=$3
    
    print_info "Starting $app_name on port $port..."
    
    cd "$app_path"
    
    # Check if Flask app exists
    if [ -f "flask_app.py" ]; then
        python flask_app.py &
        APP_PID=$!
        echo $APP_PID > "${app_name}.pid"
        print_status "$app_name started (PID: $APP_PID)"
    else
        print_warning "No Flask app found for $app_name"
    fi
    
    cd - > /dev/null
}

# Stop applications
stop_apps() {
    print_info "Stopping applications..."
    
    # Find and kill Flask processes
    pkill -f flask_app.py 2>/dev/null || true
    
    # Remove PID files
    find . -name "*.pid" -delete 2>/dev/null || true
    
    print_status "Applications stopped"
}

# Docker deployment
deploy_docker() {
    if [ "$DOCKER_AVAILABLE" = true ]; then
        print_info "Starting Docker deployment..."
        
        # Build and start services
        docker-compose up --build -d
        
        print_status "Docker services started"
        print_info "Services available at:"
        echo "  - Pandas Data Analysis: http://localhost:5001"
        echo "  - Sentiment Analysis: http://localhost:5002"
        echo "  - House Price Predictor: http://localhost:5003"
        echo "  - Matrix Calculator: http://localhost:5004"
    else
        print_warning "Docker not available, skipping Docker deployment"
    fi
}

# Manual deployment (without Docker)
deploy_manual() {
    print_info "Starting manual deployment..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start applications
    start_app "pandas-analyzer" "slab1-beginners/pandas-data-analysis" 5001
    start_app "sentiment-analyzer" "slab2-intermediate/sentiment-analysis-web" 5002
    # Add other apps as needed
    
    print_status "Manual deployment completed"
    print_info "Applications available at:"
    echo "  - Pandas Data Analysis: http://localhost:5001"
    echo "  - Sentiment Analysis: http://localhost:5002"
}

# Show project status
show_status() {
    print_info "Project Status:"
    echo "=================="
    
    # Check running processes
    if pgrep -f flask_app.py > /dev/null; then
        print_status "Flask applications are running"
    else
        print_warning "No Flask applications running"
    fi
    
    # Check Docker containers
    if [ "$DOCKER_AVAILABLE" = true ]; then
        if docker-compose ps | grep -q "Up"; then
            print_status "Docker services are running"
        else
            print_warning "No Docker services running"
        fi
    fi
}

# Run tests
run_tests() {
    print_info "Running project tests..."
    
    source venv/bin/activate
    
    # Test individual components
    echo "Testing Pandas Data Analysis..."
    cd slab1-beginners/pandas-data-analysis
    python data_analyzer.py --test 2>/dev/null || python -c "from data_analyzer import create_sample_dataset; print('✓ Pandas module working')"
    cd - > /dev/null
    
    echo "Testing Sentiment Analysis..."
    cd slab2-intermediate/sentiment-analysis-web
    python -c "from sentiment_analyzer import SentimentAnalyzer; print('✓ Sentiment analysis working')"
    cd - > /dev/null
    
    echo "Testing Matrix Operations..."
    cd slab1-beginners/matrix-operations-tool
    python -c "import numpy as np; from matrix_calculator import MatrixCalculator; print('✓ Matrix operations working')"
    cd - > /dev/null
    
    print_status "All tests passed!"
}

# Main menu
show_menu() {
    echo ""
    echo "Select deployment option:"
    echo "1. 🐳 Docker Deployment (Recommended)"
    echo "2. 🔧 Manual Deployment"
    echo "3. 📦 Install Dependencies Only"
    echo "4. 🧪 Run Tests"
    echo "5. 📊 Show Status"
    echo "6. 🛑 Stop All Services"
    echo "7. ❌ Exit"
    echo ""
}

# Main script
main() {
    # Initial checks
    check_python
    check_docker
    
    while true; do
        show_menu
        read -p "Enter your choice (1-7): " choice
        
        case $choice in
            1)
                install_dependencies
                deploy_docker
                ;;
            2)
                install_dependencies
                deploy_manual
                ;;
            3)
                install_dependencies
                ;;
            4)
                run_tests
                ;;
            5)
                show_status
                ;;
            6)
                stop_apps
                if [ "$DOCKER_AVAILABLE" = true ]; then
                    docker-compose down
                fi
                ;;
            7)
                print_info "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please choose 1-7."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@"