# train
CUDA_VISIBLE_DEVICES="0" python3 train.py --data data/widerface.yaml --cfg models/yolov5n-0.5.yaml --weights 'pretrained models' --hyp data/hyp.scratch.yaml --project runs_0.05/train

