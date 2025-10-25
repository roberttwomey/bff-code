import time
import sys
import json

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.vui.vui_client import VuiClient
from unitree_sdk2py.rpc.client import Client

ethernet_interface = "enP8p1s0"

class Custom:
    def __init__(self):
        # create VUI client
        self.client = Client('vui')
        self.client.SetTimeout(3.0)
        self.client._RegistApi(1007, 0)
    
    def set_color(self, color, duration=5):
        """
        Set the VUI LED color
        
        Available colors:
        - "cyan" # thinking
        - "green" # talking
        - "blue" # daydreaming
        - "red" # listening
        - "purple"
        
        Args:
            color: Color name as string
            duration: Duration in seconds (default: 5)
        """
        p = {}
        p["color"] = color
        p["time"] = duration
        parameter = json.dumps(p)
        
        code, result = self.client._Call(1007, parameter)
        
        if code != 0:
            print(f"Set color error. code: {code}, {result}")
            return False
        else:
            print(f"Set color {color} success. {result}")
            return True

if __name__ == "__main__":
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0, ethernet_interface)

    custom = Custom()
    
    custom.set_color("purple", 5)
    time.sleep(5)

    custom.set_color("yellow", 5)
    time.sleep(5)

    custom.set_color("green", 5)
    time.sleep(5)

    custom.set_color("blue", 5)
    time.sleep(5)

    custom.set_color("red", 5)
    time.sleep(5)

    custom.set_color("cyan", 5)
    time.sleep(5)

    custom.set_color("white", 5)
    time.sleep(5)
