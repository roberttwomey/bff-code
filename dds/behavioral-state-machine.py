import time
import sys
import json
import asyncio
import enum

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.rpc.client import Client
from unitree_sdk2py.utils.crc import CRC
import unitree_legged_const as go2

ethernet_interface = "enP8p1s0"

class RobotState(enum.Enum):
    WALKING_MODE = "walking"
    SPEAKING_MODE = "speaking"
    THINKING_MODE = "thinking"
    DREAMING_MODE = "dreaming"
    POWER_OFF = "power_off"

class BehavioralStateMachine:
    def __init__(self):
        # Initialize components
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()
        self.InitLowCmd()
        
        # Sport client for robot commands
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        
        # Motion switcher client
        self.motion_switcher = MotionSwitcherClient()
        self.motion_switcher.SetTimeout(10.0)
        self.motion_switcher.Init()
        
        # VUI client for LED colors
        self.vui_client = Client('vui')
        self.vui_client.SetTimeout(3.0)
        self.vui_client._RegistApi(1007, 0)
        
        # State management
        self.current_state = RobotState.WALKING_MODE
        self.last_activity_time = time.time()
        self.idle_timeout = 30  # seconds
        self.monitoring = True
        
        # Wireless controller state
        self.controller_active = False
        self.last_controller_data = None
        
        # Subscribe to low state for controller input
        self.lowstate_subscriber = ChannelSubscriber("rt/lf/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)
        
    def InitLowCmd(self):
        """Initialize LowCmd with required header and motor values"""
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        
        # Initialize all motor commands to safe values
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # PMSM mode
            self.low_cmd.motor_cmd[i].q = go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0
    
    def LowStateMessageHandler(self, msg: LowState_):
        """Handle incoming low state messages, including wireless controller data"""
        self.last_controller_data = msg
        self.check_controller_activity(msg)
    
    def check_controller_activity(self, msg: LowState_):
        """Check if wireless controller is being used"""
        try:
            wireless_data = msg.wireless_remote
            if wireless_data:
                # Check if any joystick values are non-zero or buttons are pressed
                # Extract float values for joysticks
                import struct
                lx = struct.unpack('<f', wireless_data[4:8])[0] if len(wireless_data) > 7 else 0
                rx = struct.unpack('<f', wireless_data[8:12])[0] if len(wireless_data) > 11 else 0
                ry = struct.unpack('<f', wireless_data[12:16])[0] if len(wireless_data) > 15 else 0
                ly = struct.unpack('<f', wireless_data[20:24])[0] if len(wireless_data) > 23 else 0
                
                # Check button presses
                data1 = wireless_data[2] if len(wireless_data) > 2 else 0
                data2 = wireless_data[3] if len(wireless_data) > 3 else 0
                
                # Determine if controller is active
                self.controller_active = (
                    abs(lx) > 0.01 or abs(rx) > 0.01 or abs(ry) > 0.01 or abs(ly) > 0.01 or
                    data1 != 0 or data2 != 0
                )
                
                if self.controller_active:
                    self.last_activity_time = time.time()
                    print(f"Controller activity detected - Joysticks: Lx={lx:.2f}, Rx={rx:.2f}, Ry={ry:.2f}, Ly={ly:.2f}")
        except Exception as e:
            print(f"Error checking controller activity: {e}")
    
    def set_vui_color(self, color, duration=5):
        """Set the VUI LED color"""
        p = {}
        p["color"] = color
        p["time"] = duration
        parameter = json.dumps(p)
        
        code, result = self.vui_client._Call(1007, parameter)
        
        if code != 0:
            print(f"Set color error. code: {code}, {result}")
            return False
        else:
            print(f"Set color {color} success")
            return True
    
    def transition_to_state(self, new_state: RobotState):
        """Transition to a new state and update VUI color"""
        if self.current_state == new_state:
            return
        
        print(f"Transitioning from {self.current_state.value} to {new_state.value}")
        self.current_state = new_state
        
        # Update VUI color based on state
        color_map = {
            RobotState.WALKING_MODE: "yellow",
            RobotState.SPEAKING_MODE: "green",
            RobotState.THINKING_MODE: "purple",
            RobotState.DREAMING_MODE: "cyan",
            RobotState.POWER_OFF: "white"  # Will turn off
        }
        
        color = color_map.get(new_state, "white")
        self.set_vui_color(color, duration=0)  # duration=0 for persistent
    
    def stand_down(self):
        """Make robot stand down"""
        print("Standing down...")
        try:
            self.sport_client.StandDown()
            print("StandDown command sent")
        except Exception as e:
            print(f"Error in StandDown: {e}")
    
    def balance_stand(self):
        """Make robot balance stand"""
        print("Entering balance stand...")
        try:
            self.sport_client.BalanceStand()
            print("BalanceStand command sent")
        except Exception as e:
            print(f"Error in BalanceStand: {e}")
    
    def release_mfc_mode(self):
        """Release MFC mode service"""
        print("Checking and releasing MFC mode...")
        try:
            status, result = self.motion_switcher.CheckMode()
            
            if result['name']:
                print(f"Current mode: {result['name']}, releasing...")
                self.motion_switcher.ReleaseMode()
                time.sleep(1)
                
                # Verify mode was released
                status, result = self.motion_switcher.CheckMode()
                if result['name']:
                    print(f"Warning: Mode still active: {result['name']}")
                else:
                    print("MFC mode released successfully")
            else:
                print("No active mode found")
        except Exception as e:
            print(f"Error releasing MFC mode: {e}")
    
    def send_bms_off_command(self):
        """Send BMS command to turn off (off=0xA5)"""
        print("Sending BMS off command...")
        try:
            self.low_cmd.bms_cmd.off = 0xA5
            self.low_cmd.bms_cmd.reserve = [0, 0, 0]
            
            # Calculate and set CRC
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            
            # Publish the command
            ret = self.lowcmd_publisher.Write(self.low_cmd)
            print(f"BMS off command published. Return value: {ret}")
        except Exception as e:
            print(f"Error sending BMS off command: {e}")
    
    def check_idle_state(self):
        """Check if robot has been idle and transition to DREAMING if needed"""
        if not self.monitoring:
            return
        
        if self.current_state == RobotState.POWER_OFF:
            return
        
        current_time = time.time()
        idle_duration = current_time - self.last_activity_time
        
        # If in WALKING mode and idle for too long, transition to DREAMING
        if self.current_state == RobotState.WALKING_MODE and idle_duration >= self.idle_timeout:
            print(f"Robot idle for {idle_duration:.1f}s. Transitioning to DREAMING...")
            self.stand_down()
            time.sleep(2)
            self.transition_to_state(RobotState.DREAMING_MODE)
    
    def check_wake_from_dreaming(self):
        """Check if controller activity should wake from DREAMING to WALKING"""
        if self.current_state == RobotState.DREAMING_MODE and self.controller_active:
            print("Controller activity detected while dreaming. Waking up to WALKING mode...")
            self.balance_stand()
            time.sleep(2)
            self.transition_to_state(RobotState.WALKING_MODE)
            self.last_activity_time = time.time()
    
    def start(self):
        """Initialize the robot to WALKING mode"""
        print("Starting Behavioral State Machine...")
        print(f"Idle timeout: {self.idle_timeout}s")
        
        # Initial state: WALKING_MODE
        self.transition_to_state(RobotState.WALKING_MODE)
        self.last_activity_time = time.time()
        
        # Make robot stand up
        self.balance_stand()
        time.sleep(3)
    
    def run_state_machine(self):
        """Main state machine loop"""
        print("State machine running. Press Ctrl+C to initiate POWER_OFF...")
        
        try:
            while self.monitoring:
                # Check idle state (WALKING -> DREAMING)
                self.check_idle_state()
                
                # Check wake from dreaming (DREAMING -> WALKING)
                self.check_wake_from_dreaming()
                
                # Display current state
                idle_duration = time.time() - self.last_activity_time
                print(f"\rState: {self.current_state.value.upper():15} | Idle: {idle_duration:.1f}s", end="")
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\nCtrl+C detected. Initiating POWER_OFF sequence...")
            self.transition_to_state(RobotState.POWER_OFF)
            
            # Release MFC mode
            self.release_mfc_mode()
            time.sleep(2)
            
            # Stand down
            self.stand_down()
            time.sleep(10)
            
            # Power off
            self.send_bms_off_command()
            time.sleep(2)
            
            print("Power off sequence complete.")
            self.monitoring = False
    
    def set_thinking_mode(self):
        """Manually set to thinking mode"""
        self.transition_to_state(RobotState.THINKING_MODE)
        self.last_activity_time = time.time()
    
    def set_speaking_mode(self):
        """Manually set to speaking mode"""
        self.transition_to_state(RobotState.SPEAKING_MODE)
        self.last_activity_time = time.time()


if __name__ == '__main__':
    print("WARNING: This script controls robot behavior and will power off the robot when Ctrl+C is pressed!")
    print("Ensure there are no obstacles around the robot.")
    input("Press Enter to continue...")
    
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0, ethernet_interface)
    
    state_machine = BehavioralStateMachine()
    state_machine.start()
    
    # Run the state machine
    state_machine.run_state_machine()
    
    print("Program ended.")

