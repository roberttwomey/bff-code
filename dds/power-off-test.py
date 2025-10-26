import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, BmsCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__BmsCmd_
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.utils.crc import CRC
import unitree_legged_const as go2

ethernet_interface = "enP8p1s0"

class Custom:
    def __init__(self):
        # create publisher for low-level commands
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        
        # Initialize CRC calculator
        self.crc = CRC()
        
        # Initialize the low command with proper headers
        self.InitLowCmd()
        
        # create sport client for stand down
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        
        # create motion switcher client to release MFC mode
        self.motion_switcher = MotionSwitcherClient()
        self.motion_switcher.SetTimeout(10.0)
        self.motion_switcher.Init()
    
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
        
    def stand_down(self):
        """Asking robot to stand down"""
        print("Asking robot to stand down...")
        self.sport_client.StandDown()
    
    def stand_up(self):
        """Asking robot to stand up"""
        print("Asking robot to stand up...")
        self.sport_client.StandUp()

    def damp_mode(self):
        """Asking robot to damp mode"""
        print("Asking robot to damp mode...")
        self.sport_client.Damp()
    
    def balance_stand(self):
        """Asking robot to balance stand"""
        print("Asking robot to balance stand...")
        self.sport_client.BalanceStand()

    def release_mfc_mode(self):
        """Release MFC mode service"""
        print("Checking and releasing MFC mode...")
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
                print(f"MFC mode released successfully {result}")
        else:
            print("No active mode found, continuing...")
        
    def start_mfc_mode(self):
        """Start MFC mode service"""

        print("Checking MFC mode...")
        status, result = self.motion_switcher.CheckMode()

        if result['name']:
            print(f"Current mode: {result['name']}...")

        else:
            print("No active mode found, starting MFC mode...")

        print("Starting MFC mode...")
        self.motion_switcher.SelectMode("mcf")
        
        # Verify mode was started
        status, result = self.motion_switcher.CheckMode()
        if result['name']:
            print(f"MFC mode started successfully: {result['name']}")
        else:
            print("Warning: MFC mode not started")
        

    def send_bms_off_command(self):
        """Send BMS command to turn off (off=0xA5)"""
        print("Sending BMS off command...")
        # Set the bms_cmd field to turn off

        self.low_cmd.bms_cmd.off = 0xA5
        # Set reserve field (3 bytes of zeros)
        self.low_cmd.bms_cmd.reserve = [0, 0, 0]
        
        # Calculate and set CRC (REQUIRED before publishing!)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        
        # Publish the command
        ret = self.lowcmd_publisher.Write(self.low_cmd)
        print("BMS off command published. Return value: ", ret)
        

if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    print("This will release MFC mode, make the robot stand down, and send a power off command!")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0, ethernet_interface)

    custom = Custom()
    
    # Step 1: Make the robot stand down
    custom.stand_down()
    time.sleep(3)

    # custom.damp_mode()
    # time.sleep(3)

    # Step 2: Release MFC mode service
    custom.release_mfc_mode()
    time.sleep(2)
        
    # Step 3: Start MFC mode service
    custom.start_mfc_mode()
    time.sleep(2)

    # # Step 3: Stand up
    # custom.stand_up()
    # time.sleep(2)
   
    # # Step 4: Balance stand
    # custom.balance_stand()
    # time.sleep(2)

    # # Step 3: Send the BMS off command
    custom.send_bms_off_command()
    time.sleep(2)

    print("Commands sent successfully.")


