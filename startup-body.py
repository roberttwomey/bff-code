import time
import os
import json
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.rpc.client import Client


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
        
        # Initialize robot components for startup
        print("Initializing robot components...")
        
        # Lidar publisher
        lidar_publisher = ChannelPublisher("rt/utlidar/switch", String_)
        lidar_publisher.Init()
        lidar_cmd = std_msgs_msg_dds__String_()
        
        # Sport client for robot commands
        sport_client = SportClient()
        sport_client.SetTimeout(10.0)
        sport_client.Init()
        
        # Motion switcher client
        motion_switcher = MotionSwitcherClient()
        motion_switcher.SetTimeout(10.0)
        motion_switcher.Init()
        
        # Robot state client
        robot_state_client = RobotStateClient()
        robot_state_client.SetTimeout(10.0)
        robot_state_client.Init()
        
        # VUI client for LED colors
        vui_client = Client('vui')
        vui_client.SetTimeout(3.0)
        vui_client._RegistApi(1007, 0)
        
        # Turn on lidar
        print("Turning on lidar...")
        lidar_cmd.data = "ON"
        try:
            lidar_publisher.Write(lidar_cmd)
            print("Lidar set to ON")
            time.sleep(2)
        except Exception as e:
            print(f"Error setting lidar: {e}")
        
        # Set walking mode (green LED)
        print("Setting walking mode (green LED)...")
        p = {"color": "green", "time": 9999}
        parameter = json.dumps(p)
        code, result = vui_client._Call(1007, parameter)
        if code != 0:
            print(f"Set color error. code: {code}, {result}")
        else:
            print("Set color green success")
        
        # Ensure MCF mode is active
        print("Ensuring MCF mode...")
        try:
            robot_state_client.ServiceSwitch("mcf", True)
            time.sleep(2)
            
            status, result = motion_switcher.CheckMode()
            if result['name'] != 'mcf':
                print(f"Current mode: {result['name']}, switching to MCF...")
                motion_switcher.SelectMode("mcf")
                time.sleep(2)
            else:
                print("Already in MCF mode")
        except Exception as e:
            print(f"Error ensuring MCF mode: {e}")
        
        # Stand up
        print("Standing up...")
        try:
            sport_client.StandUp()
            print("StandUp command sent")
            time.sleep(2)
        except Exception as e:
            print(f"Error in StandUp: {e}")
        
        # Balance stand
        print("Entering balance stand...")
        try:
            sport_client.BalanceStand()
            print("BalanceStand command sent")
            time.sleep(2)
        except Exception as e:
            print(f"Error in BalanceStand: {e}")
        
        print("Robot startup sequence complete!")
    else:
        print(f"not connected: message not received on interface {interface_name}")
