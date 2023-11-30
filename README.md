Install the environment:

conda env create -f CV805.yaml

conda activate CV805

pip install tensorboard

Download the data and model from following link:

https://mbzuaiac-my.sharepoint.com/:f:/g/personal/yanan_wang_mbzuai_ac_ae/EnQj-UX8n9xKrVdgH_kjwtkBcEBqWJZIjJu00prKhZ_qaA?e=eQRghq

Extract the data and model, placing the extracted dataset(widerface_multitask.tar.gz) under the "dataset" folder (./dataset/widerface_multitask) and the models(trained_models.tar.gz) in the root folder(./runs_*). Beside, you can extract the output of the models(trained_models_output.tar.gz) and put them under the "widerface_evaluate" folder (./widerface_evaluate/widerface_txt_*)

Code changes:

Dataset preparation - Pose estimation for get angles for roll, yaw and pitch: ./get_rotation.py
Dataset preparation - Affine transformation for rotate bounding-boxes: ./get_refineBox.py
Split dataset into training and testing: ./split_dataset.py
Get and save the predicted result: ./test_widerface.py, utils/general.py I put notes on the changes.
Architecture Modification - adding output: ./models/yolo.py, ./utils/face_datasets.py
Architecture Modification - adding loss for face alignment: ./utils/loss.py
Evaluation metric for face detection, landmark detect and face alignment: ./widerface_evaluate/evaluation.py

You could check the changes through github commit: https://github.com/Yanan-Wang-cs/CV805_multi-task-for-face-detection/commit/193f5aa928823b85a37b4a4c7f8241be1728d36d


Training the model:
sh train.sh

Evaluating the model:
sh evaluation.sh

Dependencies:
pytorch, tensorboard, opencv-python, numpy and so on. These could check and installed by CV805.yaml

Generate demo images:
Set saveResultImg = True and update the path of output_folder, the output image would saved in this path. ./0_Parade_Parade_0_233.jpg and ./0_Parade_Parade_0_534.jpg shows two examples.



