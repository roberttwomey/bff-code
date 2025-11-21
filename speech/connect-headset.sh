#!/usr/bin/env bash

# --- CONFIG ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.headset.env"

# --- HELPER FUNCTIONS ---
load_last_headset() {
    if [ -f "$ENV_FILE" ]; then
        source "$ENV_FILE"
    fi
}

save_headset() {
    local mac="$1"
    local name="$2"
    # Write values in a format that can be safely sourced
    printf "LAST_HEADSET_MAC=%q\n" "$mac" > "$ENV_FILE"
    printf "LAST_HEADSET_NAME=%q\n" "$name" >> "$ENV_FILE"
}

try_auto_connect() {
    local mac="$1"
    local name="$2"
    
    if [ -z "$mac" ] || [ -z "$name" ]; then
        return 1
    fi
    
    echo "Attempting to auto-connect to last headset: $name ($mac)"
    
    # Initialize Bluetooth
    bluetoothctl power on > /dev/null 2>&1
    bluetoothctl agent on > /dev/null 2>&1
    bluetoothctl default-agent > /dev/null 2>&1
    
    # Try to connect
    bluetoothctl << EOF > /dev/null 2>&1
trust $mac
connect $mac
EOF
    
    # Wait for connection to establish
    sleep 3
    
    # Check if connection was successful by looking for the card in PulseAudio
    CARD=$(pactl list cards short | grep "$mac" | awk '{print $1}')
    
    if [ -n "$CARD" ]; then
        echo "Auto-connect successful!"
        pactl set-card-profile "$CARD" headset-head-unit
        echo "Connected $name ($mac) in headset (HFP/HSP) mode."
        return 0
    else
        echo "Auto-connect failed. Device not found or not available."
        return 1
    fi
}

# --- LOAD LAST HEADSET AND TRY AUTO-CONNECT ---
load_last_headset

if [ -n "$LAST_HEADSET_MAC" ] && [ -n "$LAST_HEADSET_NAME" ]; then
    if try_auto_connect "$LAST_HEADSET_MAC" "$LAST_HEADSET_NAME"; then
        exit 0
    fi
    echo ""
fi

# --- SCAN FOR BLUETOOTH DEVICES ---
echo "Scanning for Bluetooth devices..."
bluetoothctl power on > /dev/null 2>&1
bluetoothctl agent on > /dev/null 2>&1
bluetoothctl default-agent > /dev/null 2>&1

# Start scanning in background
bluetoothctl scan on > /dev/null 2>&1 &
SCAN_PID=$!

# Wait a bit for devices to be discovered
sleep 5

# Stop scanning
kill $SCAN_PID 2>/dev/null
bluetoothctl scan off > /dev/null 2>&1

# --- LIST AVAILABLE DEVICES ---
echo ""
echo "Available Bluetooth devices:"
echo "============================"

# Get list of devices and filter for headsets/audio devices
DEVICES=$(bluetoothctl devices | grep -E "(Headset|headset|Audio|audio|Speaker|speaker|Earbuds|earbuds|AirPods|airpods)" || bluetoothctl devices)

if [ -z "$DEVICES" ]; then
    echo "No devices found. Trying to show all devices..."
    DEVICES=$(bluetoothctl devices)
fi

# Create arrays to store MAC addresses and names
declare -a MACS
declare -a NAMES
declare -a DISPLAY_LIST

INDEX=1
while IFS= read -r line; do
    if [ -n "$line" ]; then
        MAC=$(echo "$line" | awk '{print $2}')
        NAME=$(echo "$line" | cut -d' ' -f3-)
        MACS+=("$MAC")
        NAMES+=("$NAME")
        DISPLAY_LIST+=("$INDEX) $NAME ($MAC)")
        ((INDEX++))
    fi
done <<< "$DEVICES"

# If no devices found, show all devices
if [ ${#MACS[@]} -eq 0 ]; then
    echo "No headset-specific devices found. Showing all devices:"
    while IFS= read -r line; do
        if [ -n "$line" ]; then
            MAC=$(echo "$line" | awk '{print $2}')
            NAME=$(echo "$line" | cut -d' ' -f3-)
            MACS+=("$MAC")
            NAMES+=("$NAME")
            DISPLAY_LIST+=("$INDEX) $NAME ($MAC)")
            ((INDEX++))
        fi
    done <<< "$(bluetoothctl devices)"
fi

# Display the list
for item in "${DISPLAY_LIST[@]}"; do
    echo "$item"
done

# --- USER SELECTION ---
echo ""
read -p "Select a device (1-${#MACS[@]}) or 'q' to quit: " SELECTION

if [ "$SELECTION" = "q" ] || [ "$SELECTION" = "Q" ]; then
    echo "Cancelled."
    exit 0
fi

# Validate selection
if ! [[ "$SELECTION" =~ ^[0-9]+$ ]] || [ "$SELECTION" -lt 1 ] || [ "$SELECTION" -gt ${#MACS[@]} ]; then
    echo "Invalid selection."
    exit 1
fi

# Get selected MAC address (convert to 0-based index)
SELECTED_INDEX=$((SELECTION - 1))
MAC="${MACS[$SELECTED_INDEX]}"
NAME="${NAMES[$SELECTED_INDEX]}"

echo ""
echo "Connecting to: $NAME ($MAC)"

# --- BLUETOOTH CONNECT ---
bluetoothctl << EOF
trust $MAC
connect $MAC
EOF

# Wait a moment for connection to establish
sleep 2

# --- SELECT HEADSET PROFILE ---
# (forces microphone-capable headset mode)
CARD=$(pactl list cards short | grep "$MAC" | awk '{print $1}')

if [ -n "$CARD" ]; then
    echo "Setting headset profile..."
    pactl set-card-profile "$CARD" headset-head-unit
    echo "Connected $NAME ($MAC) in headset (HFP/HSP) mode."
    # Save to .env file for next time
    save_headset "$MAC" "$NAME"
    echo "Saved headset info for auto-connect next time."
else
    echo "Connected to $NAME ($MAC), but card not found in PulseAudio yet."
    echo "The device may need a moment to fully initialize."
    # Still save it, in case it connects later
    save_headset "$MAC" "$NAME"
fi
