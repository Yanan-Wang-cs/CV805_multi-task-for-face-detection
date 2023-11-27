# python3 test_widerface.py --weights 'runs_0.005/train/exp/weights/best.pt' --save_folder 'widerface_evaluate/widerface_txt_0.005/'
# python3 widerface_evaluate/evaluation.py --pred 'widerface_evaluate/widerface_txt_0.005/'


python3 test_widerface.py --weights 'runs_0.05/train/exp/weights/best.pt' --save_folder 'widerface_evaluate/widerface_txt_0.05/'
python3 widerface_evaluate/evaluation.py --pred 'widerface_evaluate/widerface_txt_0.05/'


# python3 test_widerface.py --weights 'runs_0.5/train/exp2/weights/best.pt' --save_folder 'widerface_evaluate/widerface_txt_0.5/'
# python3 widerface_evaluate/evaluation.py --pred 'widerface_evaluate/widerface_txt_0.5/'


# python3 test_widerface.py --weights 'runs_1.0/train/exp/weights/best.pt' --save_folder 'widerface_evaluate/widerface_txt_1.0/'
# python3 widerface_evaluate/evaluation.py --pred 'widerface_evaluate/widerface_txt_1.0/'


# python3 test_widerface.py --weights 'runs_2/train/exp/weights/best.pt' --save_folder 'widerface_evaluate/widerface_txt_2/'
# python3 widerface_evaluate/evaluation.py --pred 'widerface_evaluate/widerface_txt_2/'


# python3 test_widerface.py --weights 'runs_face_landmark/train/exp/weights/best.pt' --save_folder 'widerface_evaluate/widerface_txt_face_landmark/'
# python3 widerface_evaluate/evaluation.py --pred 'widerface_evaluate/widerface_txt_face_landmark/'

# python3 test_widerface.py --weights 'runs_face_alignment/train/exp/weights/best.pt' --save_folder 'widerface_evaluate/widerface_txt_face_alignment/'
# python3 widerface_evaluate/evaluation.py --pred 'widerface_evaluate/widerface_txt_face_alignment/'

# python3 test_widerface.py --weights 'runs_onlyface/train/exp/weights/best.pt' --save_folder 'widerface_evaluate/widerface_txt_onlyface/'
# python3 widerface_evaluate/evaluation.py --pred 'widerface_evaluate/widerface_txt_onlyface/'

