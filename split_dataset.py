import os
import random
import shutil
filelist = os.listdir('./dataset/widerface/train/')
piclist = []
for filename in filelist:
    if filename.endswith('.jpg'):
        piclist.append(filename)
valdata = random.sample(piclist, int(0.1*len(piclist)))

trainpath = './dataset/widerface_multitask/train/'
valpath = './dataset/widerface_multitask/val/'
for pic in piclist:
    if pic in valdata:
        print('move to val:', pic)
        shutil.move('./dataset/widerface/train/'+pic, valpath+pic)
        shutil.move('./dataset/widerface/train/'+pic.replace('.jpg', '.txt'), valpath+pic.replace('.jpg', '.txt'))
        shutil.move('./dataset/widerface/train/'+pic.replace('.jpg', '_pose.txt'), valpath+pic.replace('.jpg', '_pose.txt'))
        shutil.move('./dataset/widerface/train/'+pic.replace('.jpg', '_refine.txt'), valpath+pic.replace('.jpg', '_refine.txt'))
    else:
        print('move to train:', pic)
        shutil.move('./dataset/widerface/train/'+pic, trainpath+pic)
        shutil.move('./dataset/widerface/train/'+pic.replace('.jpg', '.txt'), trainpath+pic.replace('.jpg', '.txt'))
        shutil.move('./dataset/widerface/train/'+pic.replace('.jpg', '_pose.txt'), trainpath+pic.replace('.jpg', '_pose.txt'))
        shutil.move('./dataset/widerface/train/'+pic.replace('.jpg', '_refine.txt'), trainpath+pic.replace('.jpg', '_refine.txt'))
