import time
import sys
import json

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.rpc.client import Client

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_


ethernet_interface = "enP8p1s0"

class Custom:
    def __init__(self):
        # create publisher #
        self.publisher = ChannelPublisher("rt/uwbswitch", String_)
        self.publisher.Init()
        self.low_cmd = std_msgs_msg_dds__String_()   

    def go2_uwb_switch(self,status):
        if status == "OFF":
            self.low_cmd.data = "OFF"
        elif status == "ON":
            self.low_cmd.data = "ON"

        ret = self.publisher.Write(self.low_cmd)
        return ret

if __name__ == "__main__":
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0, ethernet_interface)

    custom = Custom()
    print(custom.go2_uwb_switch("OFF"))
    time.sleep(5)
    
    print(custom.go2_uwb_switch("ON"))
    time.sleep(5)


    # client = Client('uwbswitch')
    # client.SetTimeout(3.0)
    # client._RegistApi(1001, 0) # set - 1001
    # client._RegistApi(1002, 0) # get - 1002
    # client._RegistApi(1003, 0) # is_tracking - 1003


    # p = {}
    # p["color"] = "red"
    # p["time"] = 5
    # parameter = json.dumps(p)

    # code, result  = client._Call(1007, parameter)

    # if code != 0:
    #     print("set color error. code:", code, result)
    # else:
    #     print("set color red success.", result)

