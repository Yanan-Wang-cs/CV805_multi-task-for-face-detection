
import cv2
import math
import numpy as np
import glob
import os

def face_orientation(frame, landmarks, center):
    # size = frame.shape #(height, width, color_channel)
    image_points = np.array([
                            (landmarks[0], landmarks[1]),     # Left eye left corner
                            (landmarks[2], landmarks[3]),     # Right eye right corne
                            (landmarks[4], landmarks[5]),     # Nose tip
                            (landmarks[6], landmarks[7]),     # Left Mouth corner
                            (landmarks[8], landmarks[9])      # Right mouth corner
                        ], dtype="double")

    model_points = np.array(([-165.0, 170.0, -115.0],  # Left eye
                                [165.0, 170.0, -115.0],  # Right eye
                                [0.0, 0.0, 0.0],  # Nose tip
                                [-150.0, -150.0, -125.0],  # Left Mouth corner
                                [150.0, -150.0, -125.0]), dtype=np.double)  # Right Mouth corner)

    # Camera internals
 
    # center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)

    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    print(pitch, roll, yaw)
    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (landmarks[4], landmarks[5])

# images_path = glob.glob(os.path.join('./dataset/widerface/train/' + '*.jpg'))
images_path = ['/home/yanan/cv/yolov5-face/dataset/widerface_multitask/train/7_Cheering_Cheering_7_532.jpg']
for imagePath in images_path:
    print(imagePath)
    img = cv2.imread(imagePath)
    base_txt = imagePath[:-4] + ".txt"
    update_txt = imagePath[:-4] + "_pose_delete.txt"
    f = open(base_txt, 'r')
    lines = f.readlines()
    height, width, _ = img.shape


    with open(update_txt,'w') as w:
        for line in lines:
            line = line.rstrip()
            point = line.split(' ')[2:]
            x = [int(float(point[x]) * width) for x in range(len(point)) if x % 2 == 0]
            y = [int(float(point[y]) * height) for y in range(len(point)) if y % 2 == 1]
            landmarks = [[x[i], y[i]] for i in range(len(x))]
            landmarks = np.array(landmarks).flatten()[4:]
            if not sum(i>0 for i in landmarks):
                print('skip')
                w.writelines(line+ ' -1'+' '+'-1 '+'-1\n')
                continue
            # img_info = line.split(' ')
            img_path = imagePath
            frame = img
            # landmarks =  map(int, img_info[1:])
            for i in range(0, len(landmarks), 2):
                cv2.circle(frame, (landmarks[i], landmarks[i+1]), 1, (0,255,255),4)
            try:
                imgpts, modelpts, rotate_degree, nose = face_orientation(frame, landmarks, (x[0], y[0]))
                (roll,pitch, yaw) = rotate_degree
                w.writelines(line+' '+ roll+' '+ pitch+' '+ yaw+'\n')
            except:
                w.writelines(line+' '+ '-1'+' '+ '-1'+' '+ '-1'+'\n')
            # cv2.line(frame, nose, tuple(map(int,imgpts[1].ravel())), (0,255,0), 3) #GREEN
            # cv2.line(frame, nose, tuple(map(int, imgpts[0].ravel())), (221,234,17), 3) #BLUE
            # cv2.line(frame, nose, tuple(map(int, imgpts[2].ravel())), (0,0,255), 3) #RED
            # print(x,y)
            # cv2.putText(frame, ('pitch: {}').format(int(pitch)), (x[0]-50,y[0]+80), cv2.FONT_HERSHEY_SIMPLEX, 1, (221,234,17), thickness=2, lineType=2)
            # cv2.putText(frame, ('roll: {}').format(int(roll)), (x[0]-50,y[0]+120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2, lineType=2)
            # cv2.putText(frame, ('yaw: {}').format(int(yaw)), (x[0]-50,y[0]+160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2, lineType=2)

            # cv2.imshow('x', frame)
            # cv2.waitKey()
            # remapping = [2,3,0,4,5,1]
            # for index in range(len(landmarks)/2):
            #     random_color = tuple(np.random.random_integers(0,255,size=3))
            
            #     cv2.circle(frame, (landmarks[index*2], landmarks[index*2+1]), 5, random_color, -1)
            #     cv2.circle(frame,  tuple(modelpts[remapping[index]].ravel().astype(int)), 2, random_color, -1)
                
                    
            # cv2.putText(frame, rotate_degree[0]+' '+rotate_degree[1]+' '+rotate_degree[2], (10, 30),
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
            #            thickness=2, lineType=2)
                        
            # for j in xrange(len(rotate_degree)):
            #             cv2.putText(frame, ('{:05.2f}').format(float(rotate_degree[j])), (10, 30 + (50 * j)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

            cv2.imwrite('test.jpg', frame)

        f.close()

