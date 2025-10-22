import asyncio
import logging
import json
import sys
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)
    
def display_data(message):

    imu_state = message['imu_state']
    quaternion = imu_state['quaternion']
    gyroscope = imu_state['gyroscope']
    accelerometer = imu_state['accelerometer']
    rpy = imu_state['rpy']
    temperature = imu_state['temperature']

    mode = message['mode']
    progress = message['progress']
    gait_type = message['gait_type']
    foot_raise_height = message['foot_raise_height']
    position = message['position']
    body_height = message['body_height']
    velocity = message['velocity']
    yaw_speed = message['yaw_speed']
    range_obstacle = message['range_obstacle']
    foot_force = message['foot_force']
    foot_position_body = message['foot_position_body']
    foot_speed_body = message['foot_speed_body']

    # Clear the entire screen and reset cursor position to top
    sys.stdout.write("\033[H\033[J")

    # Print each piece of data on a separate line
    print("Go2 Robot Status")
    print("===================")
    print(f"Mode: {mode}")
    print(f"Progress: {progress}")
    print(f"Gait Type: {gait_type}")
    print(f"Foot Raise Height: {foot_raise_height} m")
    print(f"Position: {position}")
    print(f"Body Height: {body_height} m")
    print(f"Velocity: {velocity}")
    print(f"Yaw Speed: {yaw_speed}")
    print(f"Range Obstacle: {range_obstacle}")
    print(f"Foot Force: {foot_force}")
    print(f"Foot Position (Body): {foot_position_body}")
    print(f"Foot Speed (Body): {foot_speed_body}")
    print("-------------------")
    print(f"IMU - Quaternion: {quaternion}")
    print(f"IMU - Gyroscope: {gyroscope}")
    print(f"IMU - Accelerometer: {accelerometer}")
    print(f"IMU - RPY: {rpy}")
    print(f"IMU - Temperature: {temperature}°C")
    
    # Optionally, flush to ensure immediate output
    sys.stdout.flush()

async def main():
    try:
        # Choose a connection method (uncomment the correct one)
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.4.30")
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="unitree.local")
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)

        # Connect to the WebRTC service.
        await conn.connect()

        await asyncio.sleep(3)

        ####### NORMAL MODE ########
        print("Checking current motion mode...")

        # Get the current motion_switcher status
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"], 
            {"api_id": 1001}
        )

        if response['data']['header']['status']['code'] == 0:
            data = json.loads(response['data']['data'])
            current_motion_switcher_mode = data['name']
            print(f"Current motion mode: {current_motion_switcher_mode}")


        # # Switch to "normal" mode if not already
        # if current_motion_switcher_mode != "normal":
        #     print(f"Switching motion mode from {current_motion_switcher_mode} to 'normal'...")
        #     response = await conn.datachannel.pub_sub.publish_request_new(
        #         RTC_TOPIC["MOTION_SWITCHER"], 
        #         {
        #             "api_id": 1002,
        #             "parameter": {"name": "normal"}
        #         }
        #     )
        #     await asyncio.sleep(1)  # Wait while it stands up

        #     if response['data']['header']['status']['code'] == 0:
        #         data = json.loads(response['data']['data'])
        #         current_motion_switcher_mode = data['name']
        #         print(f"New motion mode: {current_motion_switcher_mode}")


        # === Hello - WORKING ===
 
        # print("Performing 'Hello' movement...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {"api_id": SPORT_CMD["Hello"]}
        # )

        # await asyncio.sleep(1)

        # print("Checking current motion mode...")

        # Get the current motion_switcher status
        # response = await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["MOTION_SWITCHER"], 
        #     {"api_id": 1001}
        # )

        # if response['data']['header']['status']['code'] == 0:
        #     data = json.loads(response['data']['data'])
        #     current_motion_switcher_mode = data['name']
        #     print(f"Current motion mode: {current_motion_switcher_mode}")


        # # Perform a "Move Forward" movement
        # print("Moving forward...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["Move"],
        #         "parameter": {"x": 0.5, "y": 0, "z": 0}
        #     }
        # )

        # await asyncio.sleep(3)

        # Perform a "Move Forward" movement
        # print("Moving forward...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["FreeWalk"],
        #         "parameter": {"x": 0.5, "y": 0, "z": 0}
        #     }
        # )

        # await asyncio.sleep(3)

        # Perform a "Move Backward" movement
        # print("Moving backward...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["Move"],
        #         "parameter": {"x": -0.5, "y": 0, "z": 0}
        #     }
        # )

        # await asyncio.sleep(3)

        # print("Starting Leashmode (LeadFollow)...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["LeadFollow"],
        #     }
        # )

        # await asyncio.sleep(3)

        # === StandDown - WORKING ===
        # print("Sending StandDown...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["StandDown"],
        #     }
        # )

        # await asyncio.sleep(3)

        # === StandUp - WORKING ===
        # print("Sending StandUp...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["StandUp"],
        #     }
        # )

        # await asyncio.sleep(3)

        # === BalanceStand - WORKING ===
        # print("Sending BalanceStand...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["BalanceStand"],
        #     }
        # )

        # await asyncio.sleep(3)

        ####### AI MODE ########

        # # Switch to AI mode
        # print("Switching motion mode to 'AI'...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["MOTION_SWITCHER"], 
        #     {
        #         "api_id": 1002,
        #         "parameter": {"name": "ai"}
        #     }
        # )
        # await asyncio.sleep(10)

        # # Switch to Handstand Mode
        # print("Switching to Handstand Mode...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["StandOut"],
        #         "parameter": {"data": True}
        #     }
        # )

        # await asyncio.sleep(5)

        # # Switch back to StandUp Mode
        # print("Switching back to StandUp Mode...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["StandOut"],
        #         "parameter": {"data": False}
        #     }
        # )

        # await asyncio.sleep(5)
        # Perform a backflip
        # print(f"Performing BackFlip")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["BackFlip"],
        #         "parameter": {"data": True}
        #     }
        # )

        # walk stair 2049
        # backstand 2050
        # free avoid 2048
        # switch joystick 1017


        # await asyncio.sleep(5)
        # print("...done.")


        # === LeadFollow - WORKING ===
        
        # print("Lead follow")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"],
        #     {
        #         "api_id": 2056,
        #         "parameter": {
        #             "data": 0.18
        #         }
        #     }
        # )

        # await asyncio.sleep(5)
        # print("...done.")


        # # # Switch to AI mode
        # print("Switching motion mode to 'MCF'...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["MOTION_SWITCHER"], 
        #     {
        #         "api_id": 1002,
        #         "parameter": {"name": "mcf"}
        #     }
        # )
        # await asyncio.sleep(10)

        # # Switch to AI mode
        # print("Switching motion mode to 'ai'...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["MOTION_SWITCHER"], 
        #     {
        #         "api_id": 1002,
        #         "parameter": {"name": "ai"}
        #     }
        # )
        # await asyncio.sleep(10)


        # print("Backstand")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"],
        #     {   
        #         "api_id": 2050,
        #         "parameter": {"data": True}
        #     }
        # )

        # await asyncio.sleep(5)
        # print("...done.")


        # === StandUp - WORKING ===

        # print("Sending StandUp...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["StandUp"],
        #     }
        # )

        # await asyncio.sleep(3)


        # # === Stretch - WORKING ===

        # print("Sending Stretch...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["Stretch"],
        #     }
        # )

        # await asyncio.sleep(3)

        # === SwitchJoystick - NOT WORKING ===

        # print("Sending SwitchJoystick off...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": 1027,
        #         "parameter": {"enable": False}
        #     }
        # )

        # await asyncio.sleep(10)

        # print("Sending SwitchJoystick on...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": 1027,
        #         "parameter": {"enable": True}
        #     }
        # )

        # await asyncio.sleep(3)


        # === ObstacleAvoidance off - WORKING ===

        # print("Sending ObstacleAvoidance off...")
        # api_id = 1001
        # response = await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC['OBSTACLES_AVOID'], 
        #     {
        #         "api_id": api_id,
        #         "parameter": {"enable": False}
        #     }
        # )

        # await asyncio.sleep(3)

        # # === ObstacleAvoidance on - WORKING ===

        # print("Sending ObstacleAvoidance on...")
        # api_id = 1001
        # response = await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC['OBSTACLES_AVOID'], 
        #     {
        #         "api_id": api_id,
        #         "parameter": {"enable": True}
        #     }
        # )

        # await asyncio.sleep(3)


        # === Backstand - WORKING ===

        # print("Backstand")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"],
        #     {   
        #         "api_id": 2050,
        #         "parameter": {"data": True}
        #     }
        # )

        # await asyncio.sleep(5)
        # print("...done.")

        # # === Stop Backstand - WORKING ===
        
        # print("Stop Backstand")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"],
        #     {   
        #         "api_id": 2050,
        #         "parameter": {"data": False}
        #     }
        # )

        # await asyncio.sleep(5)
        # print("...done.")

        # === Handstand - WORKING ===

        # print("Handstand")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"],
        #     {   
        #         "api_id": 2044,
        #         "parameter": {"data": True}
        #     }
        # )

        # await asyncio.sleep(5)
        # print("...done.")

        # # === Stop Handstand - WORKING ===

        # print("Stop Handstand")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"],
        #     {   
        #         "api_id": 2044,
        #         "parameter": {"data": False}
        #     }
        # )

        # await asyncio.sleep(5)
        # print("...done.")

        # # Define a callback function to handle sportmode status when received.
        # def sportmodestatus_callback(message):
        #     current_message = message['data']
            
        #     display_data(current_message)


        # # Subscribe to the sportmode status data and use the callback function to process incoming messages.
        # conn.datachannel.pub_sub.subscribe(RTC_TOPIC['LF_SPORT_MOD_STATE'], sportmodestatus_callback)
        
        # level = 1
        # print(f"Sending SpeedLevel command: {level}")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": 1015,
        #         "parameter": {"level": level}
        #     }
        # )

        # === Pose then Euler ===
        # print(f"Sending Pose command")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["Pose"],
        #         "parameter": {"enable": True}
        #     }
        # )

        # === Euler works while in pose mode ===
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["Euler"],
        #         "parameter": {"x": 0.0, "y": 0.0, "z": 0.9}
        #     }
        # )
        # # x is roll (left/right) tilt
        # # y is pitch (up/down) tilt
        # # z is yaw (left/right) tilt
        

        # === BodyHeight - NOT WORKING ===
        # Leg lift, body height, stair climbing disabled in 1.7
        # https://discord.com/channels/1205243330137690195/1205243331055980596/1376067634600083598

        # print("set body height to 18cm")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"],
        #     {
        #         # "api_id": SPORT_CMD["BodyHeight"],
        #         "api_id": 1013,
        #         "parameter": {
        #             "data": -0.18
        #         }
        #     }
        # )

        # === UWB state - works when remote is on ===

        # Define a callback function to handle uwb status when received.
        # def uwbstate_callback(message):
        #     current_message = message['data']
        #     print(current_message)
        #     #display_data(current_message)

        # Subscribe to the sportmode status data and use the callback function to process incoming messages.
        # conn.datachannel.pub_sub.subscribe(RTC_TOPIC['UWB_STATE'], uwbstate_callback)


        # https://github.com/unitreerobotics/unitree_sdk2/blob/008157a44c9bb6a7ec8b0433d6d4c3e0cce27aa6/include/unitree/robot/go2/utrack/utrack_api.hpp#L10

        # set - 1001
        # get - 1002
        # is_tracking - 1003

        # doesn't work
        # print("enable UWB")
        # result = await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["UWB_REQ"],
        #     {   
        #         "api_id": 1001,
        #         # "parameter": {"data": True}
        #     }
        # )

        # await asyncio.sleep(5)
        # print(result)


        # # === Gas Sensor state - TODO ===

        # # Define a callback function to handle uwb status when received.
        # def uwbstate_callback(message):
        #     current_message = message['data']
        #     print(current_message)
        #     #display_data(current_message)

        # # Subscribe to the sportmode status data and use the callback function to process incoming messages.
        # conn.datachannel.pub_sub.subscribe(RTC_TOPIC['GAS_SENSOR'], uwbstate_callback)

        # === Lidar switch - ? ===
        
        print("lidar switch off")
        
        # Publish a message to turn the LIDAR sensor off.
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "off")

        await asyncio.sleep(3)
        print(result)


        # Keep the program running for a while
        await asyncio.sleep(3600)
    
    except ValueError as e:
        # Log any value errors that occur during the process.
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle Ctrl+C to exit gracefully.
        print("\nProgram interrupted by user")
        sys.exit(0)