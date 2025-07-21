#!/bin/bash

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "\n${BLUE}=====================================================${NC}"
echo -e "${BLUE}      CONSENSUS MECHANISM AI AGENTS - DASHBOARD${NC}"
echo -e "${BLUE}=====================================================${NC}\n"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 not found. Please install Python and try again.${NC}"
    exit 1
fi

# Check if dashboard script exists
if [ ! -f "dashboard.py" ]; then
    echo -e "${RED}Error: File dashboard.py not found${NC}"
    exit 1
fi

# Check if run.py script exists
if [ ! -f "run.py" ]; then
    echo -e "${RED}Error: File run.py not found${NC}"
    exit 1
fi

# Run the dashboard
echo "Launching Dashboard..."
echo ""
python3 run.py

# Check if there was an error
if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}An error occurred. Please check the error messages above.${NC}"
    exit 1
fi 