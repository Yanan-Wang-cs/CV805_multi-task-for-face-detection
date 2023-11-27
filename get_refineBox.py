# -*- coding: utf-8 -*-


import cv2
import math
import numpy as np
import glob
import os

# images_path = glob.glob(os.path.join('./dataset/widerface/val/' + '*.jpg'))
images_path = ['/home/yanan/cv/yolov5-face/dataset/widerface/train/7_Cheering_Cheering_7_532.jpg']
for imagePath in images_path:
    print(imagePath)
    img = cv2.imread(imagePath)
    frame = img
    base_txt = imagePath[:-4] + "_pose.txt"
    update_txt = imagePath[:-4] + "_refine_delete.txt"
    f = open(base_txt, 'r')
    lines = f.readlines()
    height, width, _ = img.shape
    with open(update_txt,'w') as myw:
        for line in lines:
            line = line.rstrip()
            point = line.split(' ')[2:16]
            x = [int(float(point[x]) * width) for x in range(len(point)) if x % 2 == 0]
            y = [int(float(point[y]) * height) for y in range(len(point)) if y % 2 == 1]
            landmarks = [[x[i], y[i]] for i in range(len(x))]
            boxes = np.array(landmarks).flatten()[:4]
            
            landmarks = np.array(landmarks).flatten()[4:]
            if not sum(i>0 for i in landmarks):
                print('skip')
                myw.writelines(line+ ' -1 -1 -1 -1 -1\n')
                continue
            landmarks = landmarks.reshape((-1,2))
            
            # img_info = line.split(' ')
            try:
                img_path = imagePath
                startX = int(boxes[0]-boxes[2]/2)
                startY = int(boxes[1]-boxes[3]/2)
                endX = startX+boxes[2]
                endY = startY+boxes[3]
                
                origin_map = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                                   [201.26117, 371.41043], [313.08905, 371.15118]])
                # origin_map = np.array([
                #     [30.29459953,  51.69630051],
                #     [65.53179932,  51.50139999],
                #     [48.02519989,  71.73660278],
                #     [33.54930115,  92.3655014],
                #     [62.72990036,  92.20410156]
                # ])
                # origin_map = np.array([
                #     [30.29459953,  51.69630051],
                #     [65.53179932,  51.50139999],
                #     [48.02519989,  71.73660278],
                #     [33.54930115,  92.3655014],
                #     [62.72990036,  92.20410156]
                # ])
                H = cv2.estimateAffinePartial2D(origin_map, landmarks, method=cv2.LMEDS)[0]
                pts = np.float32([[0,0],[0,512],[512,512],[512,0]]).reshape(-1,2)
                # pts = np.float32([[0,0],[0,112],[112,112],[112,0]]).reshape(-1,2)
                # pts = np.float32([[0,0],[0,112],[96,112],[96,0],[48,56]]).reshape(-1,2)
                pts = np.hstack([pts, np.ones([len(pts), 1])]).T

                dst = np.dot(H, pts)
                dst = dst.T.reshape(-1,2)
                [centerX, centerY] = np.int32(dst[-1])
                # dst = dst[:-1]
                print(dst)
                def getDist_P2P(Point0,PointA):
                    distance=math.pow((Point0[0]-PointA[0]),2) + math.pow((Point0[1]-PointA[1]),2)
                    distance=math.sqrt(distance)
                    return int(distance)
                w = getDist_P2P(dst[3], dst[0])
                h = getDist_P2P(dst[1], dst[0])
                x = -math.atan2(dst[3][1]-dst[0][1],dst[3][0]-dst[0][0])
                x=x*180/math.pi
                myw.writelines(line+ ' {0} {1} {2} {3} {4}\n'.format(centerX/width, centerY/height, w/width, h/height, x))
            except:
                myw.writelines(line+ ' -1 -1 -1 -1 -1\n')
            # frame = cv2.polylines(frame,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
            # img = cv2.rectangle(img, (boxes[0]-(boxes[2]//2), boxes[1]-boxes[3]//2), (boxes[0]+(boxes[2]//2), boxes[1]+boxes[3]//2), (0,0,255), 2)
            # frame2 = img[dst[1]:dst[3], dst[0]:dst[2]]
            cv2.imwrite('test.jpg', img)
            # cv2.imwrite('test2.jpg', frame)

    f.close()

