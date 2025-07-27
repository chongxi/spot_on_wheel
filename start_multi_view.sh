#!/bin/bash

# Start spot_wheel_isaacsim_socket.py in the background
echo "Starting spot_wheel_isaacsim_socket.py..."
python spot_wheel_isaacsim_socket.py &
WHEEL_PID=$!

# Wait a moment for the first script to initialize
sleep 2

# Check if the first script is still running
if ps -p $WHEEL_PID > /dev/null; then
    echo "spot_wheel_isaacsim_socket.py started successfully (PID: $WHEEL_PID)"
    
    # Start spot_minimal_multiview.py with center-skeleton option
    echo "Starting spot_minimal_multiview.py with centered skeleton..."
    python spot_minimal_multiview.py --center-skeleton &
    MULTI_VIEW_PID=$!
    
    echo "spot_minimal_multiview.py started (PID: $MULTI_VIEW_PID)"
    echo "Both scripts are now running."
    
    # Wait for both processes
    wait $WHEEL_PID $MULTI_VIEW_PID
else
    echo "Error: spot_wheel_isaacsim_socket.py failed to start"
    exit 1
fi