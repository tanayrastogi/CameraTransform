# -*- coding: utf-8 -*-
# Python Libraries
import numpy as np 

def intrensic_matrix_IC(camera_params: dict):
    """
    Return the intrensic matrix for converting from image 
    to camera coordinates. 
    """
    fx = camera_params["img_width"]/camera_params["sen_width"]    # px/mm ## Relationship between px and mm
    fy =  camera_params["img_height"]/camera_params["sen_height"]  # px/mm ## Relationship between px and mm
    cx = camera_params["img_width"]/2                             # px    ## Center of the image in x
    cy = camera_params["img_height"]/2                            # px    ## Center of the image in y

    # Matrix
    K = np.array([[1/fx, 0,     -cx/fx],
                  [0,    1/fy,  -cy/fy],
                  [0,    0,          1]])
    transform = camera_params["focal_length"]*np.array([[0, -1, 0],
                                                        [-1, 0, 0],
                                                        [0,  0, 1]])
    return np.dot(transform, K)



def extrensic_matrix_IC(params: dict):
    """
    Return the extrensic matrix for converting from camera  
    to world coordinates. 
    """
    from math import sin as s
    from math import cos as c
    from math import radians
    y = radians(params["yaw"])   # yaw           [radians]
    p = radians(params["pitch"]) # pitch         [radians]
    r = radians(params["roll"])  # roll          [radians]
    x_ext = params["x_t"]        # Translation x [meters]
    y_ext = params["y_t"]        # Translation y [meters]
    z_ext = params["z_t"]        # Translation z [meters]
    
    # Rotation from camera to vehicle
    R_CV = np.array([[c(r)*c(p), c(r)*s(p)*s(y) - s(r)*c(y), c(r)*s(p)*c(y) + s(r)*s(y)],
                     [s(r)*c(p), s(r)*s(p)*s(y) + c(r)*c(y), s(r)*s(p)*c(y) - c(y)*s(r)],
                     [-s(p)    , c(p)*s(y)                 , c(p)*c(y)                 ]])

    # Translation from camera to vehicle
    t_CV = np.array([[x_ext], [y_ext], [z_ext]])

    return np.hstack((R_CV, t_CV))

if __name__=="__main__":
    ####################
    # Intrensic Matrix #
    ####################
    camera_params= dict(img_width    = 1920,            # px ## Width of the image  
                        img_height   = 1080,            # px ## Height of the image
                        sen_width    = 5.18,            # mm ## Width of the sensor array  
                        sen_height   = 3.89,            # mm ## Height of the sensor array  
                        focal_length = 3.93)            # mm ## Focal length of the camera 
    K = intrensic_matrix_IC(camera_params)
    print("\n##################################")
    print("Interensic Matrix of shape ", K.shape)
    print(K)
    print("##################################\n")
    
    # Image coordinates
    u, v = 720, 619
    image_cord = np.array([u, v, 1]).reshape(-1, 1)
    print("Image Coordinates:")
    print(image_cord.flatten())

    # Derived Camera Coordinates
    camera_cord = np.dot(K, image_cord)
    print("Derived Camera Coordinates:")
    print(camera_cord.flatten())


    ####################
    # Extrensic Matrix #
    ####################
    # Rotation Matrix from Camera to World coordinate
    camera_params = dict(yaw  = 326.5061,       # yaw           [radians]
                        pitch = 8.7828,         # pitch         [radians]
                        roll  = 0.0404,         # roll          [radians]
                        x_t   = 0,              # Translation of camera in world cord in x-axis [meters]
                        y_t   = 0,              # Translation of camera in world cord in y-axis [meters]
                        z_t   = 0)              # Translation of camera in world cord in z-axis [meters]
    RotT = extrensic_matrix_IC(camera_params)

    print("\n##################################")
    print("Extrensic Matrix of shape ", RotT.shape)
    print(RotT)
    print("##################################\n")

    # Derived Vehicle Coordinates
    camera_cord = np.append(camera_cord, 1)
    world_cord = np.dot(RotT, camera_cord)
    print("Derived World Coordinates:")
    print(world_cord.flatten())