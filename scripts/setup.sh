#!/bin/bash

# NeuroLab Setup Script
# This script sets up the NeuroLab EEG Analysis platform

set -e  # Exit on error

echo "=========================================="
echo "NeuroLab EEG Analysis Platform Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if Python is installed
echo "Checking prerequisites..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
print_success "Python 3 is installed"

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python version must be 3.8 or higher. Current version: $PYTHON_VERSION"
    exit 1
fi
print_success "Python version $PYTHON_VERSION meets requirements"

# Check if Docker is installed (optional)
if command -v docker &> /dev/null; then
    print_success "Docker is installed"
    DOCKER_AVAILABLE=true
else
    print_info "Docker is not installed (optional for deployment)"
    DOCKER_AVAILABLE=false
fi

# Create virtual environment
echo ""
echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate || source venv/Scripts/activate
print_success "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
print_success "Pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
print_success "Dependencies installed"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data processed logs test_data models utils api config preprocessing tests
print_success "Directories created"

# Copy environment template
echo ""
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_success "Environment file created from template"
        print_info "Please edit .env file with your configuration"
    else
        print_info "No .env.example found, skipping environment file creation"
    fi
else
    print_info ".env file already exists"
fi

# Check if model exists
echo ""
if [ -f "processed/trained_model.h5" ]; then
    print_success "Trained model found"
else
    print_info "No trained model found. You can train one using: python train_model.py"
fi

# Run tests
echo ""
read -p "Do you want to run tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running tests..."
    python -m pytest tests/ -v || print_info "Some tests may fail without a trained model"
fi

# Docker setup
if [ "$DOCKER_AVAILABLE" = true ]; then
    echo ""
    read -p "Do you want to build Docker images? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Building Docker images..."
        docker-compose build
        print_success "Docker images built"
    fi
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Train a model: python train_model.py"
echo "3. Start the API server: python main.py"
echo "   or with Docker: docker-compose up"
echo ""
echo "Access the API at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
print_success "Setup completed successfully!"
