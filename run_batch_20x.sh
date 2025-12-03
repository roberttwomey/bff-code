#!/bin/bash

# Run batch-video and batch-image commands 20 times

for i in {1..20}; do
    echo "==================================="
    echo "Starting iteration $i of 20"
    echo "==================================="
    
    echo "Running batch-video..."
    python dream-manager.py batch-video --prompt-file prompts.txt
    
    # echo "Running batch-image..."
    # python dream-manager.py batch-image --prompt-file prompts.txt
    
    echo "Completed iteration $i of 20"
    echo ""
done

echo "==================================="
echo "All 20 iterations completed!"
echo "==================================="

