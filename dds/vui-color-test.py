import time
import sys
import json

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.vui.vui_client import VuiClient
from unitree_sdk2py.rpc.client import Client

ethernet_interface = "enP8p1s0"

if __name__ == "__main__":
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0, ethernet_interface)

    client = Client('vui')
    client.SetTimeout(3.0)
    client._RegistApi(1007, 0)

    # "cyan" # thinking
    # "green" # talking
    # "blue" # daydreaming
    # "red" # listening
    
    p = {}
    p["color"] = "purple"
    p["time"] = 5
    parameter = json.dumps(p)

    code, result  = client._Call(1007, parameter)

    if code != 0:
        print("set color error. code:", code, result)
    else:
        print("set color red success.", result)

    # p = {}
    # p["color"] = "red"
    # p["time"] = 5
    # parameter = json.dumps(p)

    # code, result  = client._Call(1007, parameter)

    # if code != 0:
    #     print("set color error. code:", code, result)
    # else:
    #     print("set color red success.", result)

