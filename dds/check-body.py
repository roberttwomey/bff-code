import time
import os
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


def is_interface_active(interface_name):
    """Check if the network interface exists and is active (UP)."""
    operstate_path = f"/sys/class/net/{interface_name}/operstate"
    if not os.path.exists(operstate_path):
        return False
    
    try:
        with open(operstate_path, 'r') as f:
            operstate = f.read().strip()
        return operstate == "up"
    except (IOError, OSError):
        return False


def LowStateHandler(msg: LowState_):
    global message_received
    message_received = True


if __name__ == "__main__":
    # Check if ethernet interface is active first
    interface_name = "enP8p1s0"
    if not is_interface_active(interface_name):
        print(f"not connected: interface {interface_name} is not active")
        exit(0)
    
    # Initialize channel factory
    ChannelFactoryInitialize(0, interface_name)
    
    # Set up subscriber
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    message_received = False
    sub.Init(LowStateHandler, 10)
    
    # Wait for a message with a timeout (2 seconds)
    timeout = 2.0
    start_time = time.time()
    
    while not message_received and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    # Check connection status
    if message_received:
        print(f"connected: message received on interface {interface_name}")
    else:
        print(f"not connected: message not received on interface {interface_name}")
