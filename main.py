from os import close
from aux_functions import *
from configs.read_cfg import read_cfg
import importlib, json
from unreal_envs.initial_positions import *
from airsim.types import YawMode, Quaternionr
import math
import numpy as np
import pickle
import csv
import time
# from aux_functions import *
# TF Debug message suppressed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def generate_json(cfg):
    flag  = True
    path = os.path.expanduser('~\Documents\Airsim')
    if not os.path.exists(path):
        os.makedirs(path)

    filename = path + '\settings.json'

    data = {}


    data['SettingsVersion'] = 1.2
    data['LocalHostIp'] = cfg.ip_address
    data['ApiServerPort'] = cfg.api_port

    data['SimMode'] = cfg.SimMode
    data['ClockSpeed'] = cfg.ClockSpeed
    data["ViewMode"]= "NoDisplay"
    PawnPaths = {}
    PawnPaths["DefaultQuadrotor"] = {}
    PawnPaths["DefaultQuadrotor"]['PawnBP'] = ''' Class'/AirSim/Blueprints/BP_''' + cfg.drone + '''.BP_''' + cfg.drone + '''_C' '''
    data['PawnPaths']=PawnPaths

    # Define agents:
    _, reset_array_raw, _, _ = initial_positions(cfg.env_name, num_agents=cfg.num_agents)
    Vehicles = {}
    if len(reset_array_raw) < cfg.num_agents:
        print("Error: Either reduce the number of agents or add more initial positions")
        flag = False
    else:
        for agents in range(cfg.num_agents):
            name_agent = "drone" + str(agents)
            agent_position = reset_array_raw[name_agent].pop(0)
            Vehicles[name_agent] = {}
            Vehicles[name_agent]["VehicleType"] = "SimpleFlight"
            Vehicles[name_agent]["X"] = agent_position[0]
            Vehicles[name_agent]["Y"] = agent_position[1]
            # Vehicles[name_agent]["Z"] = agent_position[2]
            Vehicles[name_agent]["Z"] = 0
            Vehicles[name_agent]["Yaw"] = agent_position[3]
        data["Vehicles"] = Vehicles

    CameraDefaults = {}
    CameraDefaults['CaptureSettings']=[]
    # CaptureSettings=[]

    camera = {}
    camera['ImageType'] = 0
    camera['Width'] = cfg.width
    camera['Height'] = cfg.height
    camera['FOV_Degrees'] = cfg.fov_degrees

    CameraDefaults['CaptureSettings'].append(camera)

    camera = {}
    camera['ImageType'] = 3
    camera['Width'] = cfg.width
    camera['Height'] = cfg.height
    camera['FOV_Degrees'] = cfg.fov_degrees

    CameraDefaults['CaptureSettings'].append(camera)

    data['CameraDefaults'] = CameraDefaults
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    return flag

# def getObservation(client : airsim.MultirotorClient):
#         responses = client.simGetImages([
#             airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
#             airsim.ImageRequest("1", airsim.ImageType.Segmentation, False, False),
#             airsim.ImageRequest("2", airsim.ImageType.DepthPlanner, True, False)
#         ])
        
#         rgbImg = responses[0]
#         segImg = responses[1]
#         depthImg = responses[2]
#         depth = np.array(depthImg.image_data_float).reshape(depthImg.height, depthImg.width)

#         depth = np.clip(depth, 0, 50)/50
        
#         depth = 1 - depth

#         seg1d = np.fromstring(segImg.image_data_uint8, dtype=np.uint8)
#         seg = seg1d.reshape(segImg.height, segImg.width, 3)

#         rgb1d = np.fromstring(rgbImg.image_data_uint8, dtype=np.uint8)
#         rgb = rgb1d.reshape(rgbImg.height, rgbImg.width, 3)

#         rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

#         observation = {}
#         observation['droneState'] = client.getMultirotorState()
#         observation["collision"] = client.simGetCollisionInfo().has_collided

        

#         observation['depth'] = depth
#         observation['fpv'] = cv2.resize(rgb, (256, 256))

#         position = observation['droneState'].kinematics_estimated.position


#         observation['quadLocation'] =  [
#             position.x_val,
#             position.y_val,
#             position.z_val
#         ]

        
        
#         return observation


def do_action(action):
    
    
    # action[0] = min(max(action[0], 1.5), -1.5)
    # action[1] = min(max(action[1], 0.5), -0.5)
    # action[2] = min(max(action[2], 0.5), -0.5)
    # action[3] = min(max(action[3], 0.5), -0.5)
    # action *= 0.5
    VX_vehicle =  action[0]
    VY_vehicle = action[1] 
    VZ_world =  action[2]
    yaw = action[3]
    ya = action[4]

    VX_world = VX_vehicle * np.cos(yaw) - VY_vehicle * np.sin(yaw)
    VY_world = VX_vehicle * np.sin(yaw) + VY_vehicle * np.cos(yaw)


    # print(VX_vehicle, VY_vehicle, VZ_world, action)


    client.moveByVelocityAsync(VX_world, VY_world, VZ_world, 1,drivetrain=0,yaw_mode=ya)
    

def getEuler(quat):
    # print(quat)
    quat = quat.orientation
    w,x,y,z = quat.w_val,quat.x_val,quat.y_val,quat.z_val
    yaw = math.atan2(2*(w*z+x*y),1-2*(y**2+z**2))
    return yaw*180/math.pi

if __name__ == '__main__':
    # Read the config file
    cfg = read_cfg(config_filename='configs/config.cfg', verbose=True)
    cfg.num_agents=1
    can_proceed = generate_json(cfg)
    name = 'drone0'
    # Check if NVIDIA GPU is available
    try:
        nvidia_smi.nvmlInit()
        cfg.NVIDIA_GPU = True
    except:
        cfg.NVIDIA_GPU = False

    if can_proceed:
        # Start the environment
        to_record = False
        dataset = []
        try:
            env_process, env_folder = start_environment(env_name=cfg.env_name)
            client, old_posit, initZ = connect_drone()
            vx,vy,vz,angle = 0,0,0,0
            drone_state = client.getMultirotorState()
            val = 1
            
            count = 0
            ya = YawMode()
            ya.is_rate = True
            
            yaw_rate = 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            nexus = 0
            # quat = drone_state.kinematics_estimated.orientation
            
            while True:
                # client.moveByVelocityZAsync(0.01, 0, 0, 0.01)
                # obs = getObservation(client)

                image = get_MonocularImageRGB(client, name)
                image2 = image
                quat = client.getImuData()
                # print(quat.angular_velocity.z_val)
                angle = getEuler(quat)
                # text = str(round(angle,2))
                # image = cv2.putText(image, text, org, font, 
                #                 fontScale, color, thickness, cv2.LINE_AA)
                count+=1
                cv2.imshow('a', image)
                k  = cv2.waitKey(1)
                if k == ord('r'):
                    to_record = not to_record
                if to_record:
                    if count%2==0:
                        # to save
                        imname = 'img' + str(count//2) + '.jpg'
                        cv2.imwrite('dataset/'+imname,image2)
                        dic = [vx,vy,vz,ya.yaw_or_rate]
                        dataset.append(dic)

                # time.sleep(0.5)
                if k == ord('o'):
                    # with open('dataset_dict', 'wb') as handle:
                    #     pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    if to_record:
                        with open('vel_data.csv', 'w',newline='') as f:
                            fields = ['vx', 'vy', 'vz', 'yaw'] 
                            write = csv.writer(f)            
                            write.writerow(fields)
                            write.writerows(dataset)
                    close_env(env_process)
                    exit()
                elif k == ord('i'):
                    # client.moveByVelocityAsync(vx, vy, vz+val, 1)
                    vx,vy,vz= vx,vy,val+vz
                elif k == ord('j'):
                    # client.moveByVelocityAsync(0, 0, vz-val, 1)
                    vx,vy,vz= vx,vy,-val+vz

                elif k == ord('w'):
                    vx,vy,vz= val+vx,vy,vz
                    # action = [vx,vy,vz,angle*math.pi/180]
                    # do_action(action)
                    
                elif k == ord('s'):
                    vx,vy,vz= -val+vx,vy,vz
                    # action = [vx,vy,vz,angle*math.pi/180]
                    # do_action(action)
                    # client.moveByVelocityAsync(-val, 0, 0.0,  1)
                elif k == ord('a'):
                    vx,vy,vz= vx,-val+vy,0
                    # action = [vx,vy,vz,angle*math.pi/180]
                    # do_action(action)
                    # client.moveByVelocityAsync(0.0, val, 0.0, 1)
                elif k == ord('d'):
                    vx,vy,vz= vx,val+vy,0
                    # action = [vx,vy,vz,angle*math.pi/180]
                    # do_action(action)
                    # client.moveByVelocityAsync(0.0, val, 0.0,  1)
                elif k == ord('e'):
                    # degrees = 20
                    # nexus = time.time()
                    
                    ya.yaw_or_rate+=yaw_rate
                    # client.moveByVelocityAsync(vx, vy,vz,  1, drivetrain=0,yaw_mode=ya)
                    # angle+=ya.yaw_or_rate
                elif k == ord('q'):
                    # angle-=20
                    
                    ya.yaw_or_rate-=yaw_rate
                    # client.moveByVelocityAsync(vx, vy, vz,  1, drivetrain=0,yaw_mode=ya)
                    # angle-=ya.yaw_or_rate
                action = [vx,vy,vz,angle*math.pi/180,ya]
                do_action(action)
                # angle-=ya.yaw_or_rate
                # client.moveByVelocityAsync(vx, vy, vz, 1)
                # angle = angle%360
        except Exception as e:

            print(e)
            if to_record:
                with open('vel_data.csv', 'w',newline='') as f:
                    fields = ['vx', 'vy', 'vz', 'yaw'] 
                    write = csv.writer(f)            
                    write.writerow(fields)
                    write.writerows(dataset)
            close_env(env_process)
            print("Closed environment")
