import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__UwbState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import UwbState_

import unitree_go2_const as go2


def UwbStateHandler(msg: UwbState_):
    
    print(msg)


if __name__ == "__main__":
    # Modify "enp2s0" to the actual network interface
    # ChannelFactoryInitialize(0, "enp2s0")
    ChannelFactoryInitialize(0, "enP8p1s0")
    sub = ChannelSubscriber("rt/uwbstate", UwbState_)
    sub.Init(UwbStateHandler, 10)

    while True:
        time.sleep(10.0)
