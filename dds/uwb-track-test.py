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

    client = Client('uwbswitch')
    client.SetTimeout(3.0)
    client._RegistApi(1001, 0) # set - 1001
    client._RegistApi(1002, 0) # get - 1002
    client._RegistApi(1003, 0) # is_tracking - 1003

    p = {}
    p["enable"] = 1
    parameter = json.dumps(p)

    code, result  = client._Call(1001, parameter)

    if code != 0:
        print("set tracking enable error. code:", code, result)
    else:
        print("set tracking enable success.", result)
    
    time.sleep(3)

    # get
    p = {}
    p["enable"] = 0
    parameter = json.dumps(p)
    code, result  = client._Call(1002, parameter)

    if code != 0:
        print("get tracking enable error. code:", code, result)
    else:
        print("get tracking enable success.", result)

