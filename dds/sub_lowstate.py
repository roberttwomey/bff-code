import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


    # head: types.array[types.uint8, 2]
    # level_flag: types.uint8
    # frame_reserve: types.uint8
    # sn: types.array[types.uint32, 2]
    # version: types.array[types.uint32, 2]
    # bandwidth: types.uint16
    # imu_state: 'unitree_sdk2py.idl.unitree_go.msg.dds_.IMUState_'
    # motor_state: types.array['unitree_sdk2py.idl.unitree_go.msg.dds_.MotorState_', 20]
    # bms_state: 'unitree_sdk2py.idl.unitree_go.msg.dds_.BmsState_'
    # foot_force: types.array[types.int16, 4]
    # foot_force_est: types.array[types.int16, 4]
    # tick: types.uint32
    # wireless_remote: types.array[types.uint8, 40]
    # bit_flag: types.uint8
    # adc_reel: types.float32
    # temperature_ntc1: types.uint8
    # temperature_ntc2: types.uint8
    # power_v: types.float32
    # power_a: types.float32
    # fan_frequency: types.array[types.uint16, 4]
    # reserve: types.uint32
    # crc: types.uint32

def LowStateHandler(msg: LowState_):
    # print(msg.power_v, msg.power_a, "\n")
    # print(msg.head, msg.level_flag, msg.frame_reserve, msg.sn, msg.version, msg.bandwidth, msg.imu_state, msg.motor_state, msg.bms_state, msg.foot_force, msg.foot_force_est, msg.tick, msg.wireless_remote, msg.bit_flag, msg.adc_reel, msg.temperature_ntc1, msg.temperature_ntc2, msg.power_v, msg.power_a, msg.fan_frequency, msg.reserve, msg.crc, "\n")
    print(msg.wireless_remote, "\n")


    
ChannelFactoryInitialize(0, "enP8p1s0")
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(LowStateHandler, 10)

while True:
    time.sleep(10.0)