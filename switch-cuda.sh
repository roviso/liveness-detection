Script started on 2023-12-11 17:01:51+05:45 [TERM="xterm-256color" TTY="/dev/pts/11" COLUMNS="197" LINES="23"]
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ source switch-cuda.sh 
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ source switch-cuda.sh [K
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python train.py --config co
compute_mean_std.py    configs/               conversion_checker.py  convert_model.py       
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python train.py --config co
compute_mean_std.py    configs/               conversion_checker.py  convert_model.py       
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python train.py --config con
configs/               conversion_checker.py  convert_model.py       
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python train.py --config configs/config.py 
AttrDict({'exp_num': 0, 'dataset': 'celeba_spoof', 'multi_task_learning': True, 'evaluation': True, 'test_steps': None, 'datasets': {'LCCFASD_root': './LCC_FASDcropped', 'Celeba_root': '/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/CelebA_Spoof_small', 'Casia_root': './CASIA'}, 'external': {'train': {}, 'val': {}, 'test': {}}, 'img_norm_cfg': {'mean': [0.5931, 0.469, 0.4229], 'std': [0.2471, 0.2214, 0.2157]}, 'optimizer': {'lr': 0.005, 'momentum': 0.9, 'weight_decay': 0.0005}, 'scheduler': {'milestones': [20, 40], 'gamma': 0.2}, 'data': {'batch_size': 256, 'data_loader_workers': 4, 'sampler': None, 'pin_memory': True}, 'resize': {'height': 128, 'width': 128}, 'checkpoint': {'snapshot_name': 'MobileNet3.pth.tar', 'experiment_path': './logs'}, 'loss': {'loss_type': 'amsoftmax', 'amsoftmax': {'m': 0.5, 's': 1, 'margin_type': 'cross_entropy', 'label_smooth': False, 'smoothing': 0.1, 'ratio': [1, 1], 'gamma': 0}, 'soft_triple': {'cN': 2, 'K': 10, 's': 1, 'tau': 0.2, 'm': 0.35}}, 'epochs': {'start_epoch': 0, 'max_epoch': 71}, 'model': {'model_type': 'Mobilenet3', 'model_size': 'large', 'width_mult': 1.0, 'pretrained': True, 'embeding_dim': 1280, 'imagenet_weights': '/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar'}, 'aug': {'type_aug': None, 'alpha': 0.5, 'beta': 0.5, 'aug_prob': 0.7}, 'curves': {'det_curve': 'det_curve_0.png', 'roc_curve': 'roc_curve_0.png'}, 'dropout': {'prob_dropout': 0.1, 'classifier': 0.35, 'type': 'bernoulli', 'mu': 0.5, 'sigma': 0.3}, 'data_parallel': {'use_parallel': False, 'parallel_params': {'device_ids': [0, 1], 'output_device': 0}}, 'RSC': {'use_rsc': False, 'p': 0.333, 'b': 0.333}, 'test_dataset': {'type': 'celeba_spoof'}, 'conv_cd': {'theta': 0}}) 11111111111111111111111
dict_keys(['train', 'val', 'test']) 11111111111111111111111

==> Loading checkpoint
_______INIT EXPERIMENT 0______
training on celeba_spoof, testing on celeba_spoof


SNAPSHOT
snapshot_name --> MobileNet3.pth.tar
experiment_path --> ./logs


MODEL
model_type --> Mobilenet3
model_size --> large
width_mult --> 1.0
pretrained --> True
embeding_dim --> 1280
imagenet_weights --> /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar


LOSS TYPE : AMSOFTMAX
m --> 0.5
s --> 1
margin_type --> cross_entropy
label_smooth --> False
smoothing --> 0.1
ratio --> [1, 1]
gamma --> 0


DROPOUT PARAMS
prob_dropout --> 0.1
classifier --> 0.35
type --> bernoulli
mu --> 0.5
sigma --> 0.3


OPTIMAIZER
lr --> 0.005
momentum --> 0.9
weight_decay --> 0.0005


ADDITIONAL USING PARAMETRS
MULTI_TASK_LEARNING USING


__VAL__:

AUC = 0.973
EER = 7.03
apcer = 2.57
bpcer = 15.97
acer = 9.27

==> saving checkpoint

accuracy on test data = 87.271
AUC = 0.973
EER = 7.03
apcer = 2.57
bpcer = 15.97
acer = 9.27













































































































































_____________EVAULATION_____________

==> Loading checkpoint


==> Loading checkpoint
AttrDict({'exp_num': 0, 'dataset': 'celeba_spoof', 'multi_task_learning': True, 'evaluation': True, 'test_steps': None, 'datasets': {'LCCFASD_root': './LCC_FASDcropped', 'Celeba_root': '/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/CelebA_Spoof_small', 'Casia_root': './CASIA'}, 'external': {'train': {}, 'val': {}, 'test': {}}, 'img_norm_cfg': {'mean': [0.5931, 0.469, 0.4229], 'std': [0.2471, 0.2214, 0.2157]}, 'optimizer': {'lr': 0.005, 'momentum': 0.9, 'weight_decay': 0.0005}, 'scheduler': {'milestones': [20, 40], 'gamma': 0.2}, 'data': {'batch_size': 256, 'data_loader_workers': 4, 'sampler': None, 'pin_memory': True}, 'resize': {'height': 128, 'width': 128}, 'checkpoint': {'snapshot_name': 'MobileNet3.pth.tar', 'experiment_path': './logs'}, 'loss': {'loss_type': 'amsoftmax', 'amsoftmax': {'m': 0.5, 's': 1, 'margin_type': 'cross_entropy', 'label_smooth': False, 'smoothing': 0.1, 'ratio': [1, 1], 'gamma': 0}, 'soft_triple': {'cN': 2, 'K': 10, 's': 1, 'tau': 0.2, 'm': 0.35}}, 'epochs': {'start_epoch': 0, 'max_epoch': 71}, 'model': {'model_type': 'Mobilenet3', 'model_size': 'large', 'width_mult': 1.0, 'pretrained': True, 'embeding_dim': 1280, 'imagenet_weights': '/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar'}, 'aug': {'type_aug': None, 'alpha': 0.5, 'beta': 0.5, 'aug_prob': 0.7}, 'curves': {'det_curve': 'det_curve_0.png', 'roc_curve': 'roc_curve_0.png'}, 'dropout': {'prob_dropout': 0.1, 'classifier': 0.35, 'type': 'bernoulli', 'mu': 0.5, 'sigma': 0.3}, 'data_parallel': {'use_parallel': False, 'parallel_params': {'device_ids': [0, 1], 'output_device': 0}}, 'RSC': {'use_rsc': False, 'p': 0.333, 'b': 0.333}, 'test_dataset': {'type': 'celeba_spoof'}, 'conv_cd': {'theta': 0}}) 11111111111111111111111
dict_keys(['train', 'val', 'test']) 11111111111111111111111

accuracy on test data = 0.874
auc = 0.973
apcer = 2.57
bpcer = 15.97
acer = 9.27
checkpoint made on 0 epoch
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python convert_model.py --config configs/conm[Kfig.py --model 

Traceback (most recent call last):
  File "convert_model.py", line 76, in <module>
    main()
  File "convert_model.py", line 46, in main
    num_layers = args.num_layers
AttributeError: 'Namespace' object has no attribute 'num_layers'
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model face-detection-0100.pth --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
Traceback (most recent call last):
  File "demo/demo.py", line 32, in <module>
    from demo_tools import TorchCNN, VectorCNN, FaceDetector
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/__init__.py", line 1, in <module>
    from .ie_tools import load_ie_model
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py", line 20, in <module>
    from openvino.inference_engine import IECore
ModuleNotFoundError: No module named 'openvino'
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ source b[Kvenv/bin/activate
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model face-detection-0100.pth --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
Traceback (most recent call last):
  File "demo/demo.py", line 32, in <module>
    from demo_tools import TorchCNN, VectorCNN, FaceDetector
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/__init__.py", line 1, in <module>
    from .ie_tools import load_ie_model
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py", line 20, in <module>
    from openvino.inference_engine import IECore
ModuleNotFoundError: No module named 'openvino'
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ conda deactivate
((venv)) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ conda deactivatepython demo/demo.py --fd_model face-detection-0100.pth --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
Traceback (most recent call last):
  File "demo/demo.py", line 32, in <module>
    from demo_tools import TorchCNN, VectorCNN, FaceDetector
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/__init__.py", line 1, in <module>
    from .ie_tools import load_ie_model
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py", line 20, in <module>
    from openvino.inference_engine import IECore
ModuleNotFoundError: No module named 'openvino'
((venv)) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ deactivate
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ conda t[Kdeat[Ka[Kctivate
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ conda deactivate
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ conda deactivate
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ conda deactivate
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ deactivate
DeprecationWarning: 'source deactivate' is deprecated. Use 'conda deactivate'.
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ deactivate
DeprecationWarning: 'source deactivate' is deprecated. Use 'conda deactivate'.
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ deactivateconda deactivate
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ soue[Kr[K[K[K[Kconda deactivate[6Pdeactivateconda deactivate[6Pdeactivateconda deactivate[6Pdeactivatepython demo/demo.py --fd_model face-detection-0100.pth --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
Traceback (most recent call last):
  File "demo/demo.py", line 32, in <module>
    from demo_tools import TorchCNN, VectorCNN, FaceDetector
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/__init__.py", line 1, in <module>
    from .ie_tools import load_ie_model
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py", line 20, in <module>
    from openvino.inference_engine import IECore
ModuleNotFoundError: No module named 'openvino'
(base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ source venv/bin/activate
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ pip install openvino
Collecting openvino
  Downloading openvino-2023.2.0-13089-cp38-cp38-manylinux2014_x86_64.whl.metadata (8.8 kB)
Requirement already satisfied: numpy>=1.16.6 in ./venv/lib/python3.8/site-packages (from openvino) (1.20.0)
Collecting openvino-telemetry>=2023.2.1 (from openvino)
  Downloading openvino_telemetry-2023.2.1-py3-none-any.whl.metadata (2.3 kB)
Downloading openvino-2023.2.0-13089-cp38-cp38-manylinux2014_x86_64.whl (37.5 MB)
[?25l   [38;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/37.5 MB[0m [31m?[0m eta [36m-:--:--[0m
[?25hDownloading openvino_telemetry-2023.2.1-py3-none-any.whl (23 kB)
Installing collected packages: openvino-telemetry, openvino
Successfully installed openvino-2023.2.0 openvino-telemetry-2023.2.1
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ pip install openvinosource venv/bin/activatepython demo/demo.py --fd_model face-detection-0100.pth --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:30:17.246437 39798 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:30:17.248350 39798 demo.py:112] Reading from cam 0
I1211 17:30:17.396571 39798 ie_tools.py:68] Initializing Inference Engine plugin for CPU
I1211 17:30:17.396710 39798 ie_tools.py:73] Loading network
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 123, in main
    face_detector = FaceDetector(args.fd_model, args.fd_thresh, args.device, args.cpu_extension)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 37, in __init__
    self.net = load_ie_model(model_path, device, None, ext_path)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py", line 74, in load_ie_model
    net = IECore().read_network(model_xml, os.path.splitext(model_xml)[0] + ".bin")
  File "ie_api.pyx", line 381, in openvino.inference_engine.ie_api.IECore.read_network
  File "ie_api.pyx", line 421, in openvino.inference_engine.ie_api.IECore.read_network
Exception: Path to the weights face-detection-0100.bin doesn't exist or it's a directory
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:32:23.845147 40240 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:32:23.846127 40240 demo.py:112] Reading from cam 0
I1211 17:32:23.987848 40240 ie_tools.py:68] Initializing Inference Engine plugin for CPU
I1211 17:32:23.987967 40240 ie_tools.py:73] Loading network
I1211 17:32:24.013897 40240 ie_tools.py:81] Preparing input blobs
I1211 17:32:24.014003 40240 ie_tools.py:87] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/demo.py", line 76, in run
    detections = face_det.get_detections(frame)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 43, in get_detections
    _, _, h, w = self.net.get_input_shape().shape
AttributeError: 'list' object has no attribute 'shape'
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:32:42.682396 40392 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:32:42.683428 40392 demo.py:112] Reading from cam 0
I1211 17:32:42.828655 40392 ie_tools.py:68] Initializing Inference Engine plugin for CPU
I1211 17:32:42.828847 40392 ie_tools.py:73] Loading network
I1211 17:32:42.850388 40392 ie_tools.py:81] Preparing input blobs
I1211 17:32:42.850474 40392 ie_tools.py:87] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/demo.py", line 76, in run
    detections = face_det.get_detections(frame)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 43, in get_detections
    _, _, h, w = self.net.get_input_shape().shape
AttributeError: 'list' object has no attribute 'shape'
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:33:21.665295 40594 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:33:21.666444 40594 demo.py:112] Reading from cam 0
I1211 17:33:21.807914 40594 ie_tools.py:69] Initializing Inference Engine plugin for CPU
I1211 17:33:21.808023 40594 ie_tools.py:74] Loading network
I1211 17:33:21.831095 40594 ie_tools.py:82] Preparing input blobs
I1211 17:33:21.831171 40594 ie_tools.py:88] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
<openvino.inference_engine.ie_api.InputInfoPtr object at 0x7fb153caeed0> 2222222222222
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/demo.py", line 76, in run
    detections = face_det.get_detections(frame)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 43, in get_detections
    _, _, h, w = self.net.get_input_shape().shape
AttributeError: 'list' object has no attribute 'shape'
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:33:32.514186 40701 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:33:32.515233 40701 demo.py:112] Reading from cam 0
I1211 17:33:32.655750 40701 ie_tools.py:69] Initializing Inference Engine plugin for CPU
I1211 17:33:32.655868 40701 ie_tools.py:74] Loading network
I1211 17:33:32.678047 40701 ie_tools.py:82] Preparing input blobs
I1211 17:33:32.678173 40701 ie_tools.py:88] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
<openvino.inference_engine.ie_api.DataPtr object at 0x7f0157caa170> 2222222222222
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/demo.py", line 76, in run
    detections = face_det.get_detections(frame)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 43, in get_detections
    _, _, h, w = self.net.get_input_shape().shape
AttributeError: 'list' object has no attribute 'shape'
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:33:48.809944 40808 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:33:48.810955 40808 demo.py:112] Reading from cam 0
I1211 17:33:48.955640 40808 ie_tools.py:69] Initializing Inference Engine plugin for CPU
I1211 17:33:48.955721 40808 ie_tools.py:74] Loading network
I1211 17:33:48.976544 40808 ie_tools.py:82] Preparing input blobs
I1211 17:33:48.976615 40808 ie_tools.py:88] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
[1, 3, 128, 128] 2222222222222
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/demo.py", line 76, in run
    detections = face_det.get_detections(frame)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 43, in get_detections
    _, _, h, w = self.net.get_input_shape().shape
AttributeError: 'list' object has no attribute 'shape'
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:34:11.681479 41017 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:34:11.682499 41017 demo.py:112] Reading from cam 0
I1211 17:34:11.823813 41017 ie_tools.py:69] Initializing Inference Engine plugin for CPU
I1211 17:34:11.823905 41017 ie_tools.py:74] Loading network
I1211 17:34:11.847940 41017 ie_tools.py:82] Preparing input blobs
I1211 17:34:11.848041 41017 ie_tools.py:88] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
[1, 3, 128, 128] 2222222222222
[1, 3, 128, 128] 2222222222222
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/demo.py", line 76, in run
    detections = face_det.get_detections(frame)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 44, in get_detections
    out = self.net.forward(cv.resize(frame, (w, h)))
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py", line 42, in forward
    res = self.net.infer(inputs={self.input_key: self._preprocess(img)})
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py", line 34, in _preprocess
    _, _, h, w = self.get_input_shape()
TypeError: cannot unpack non-iterable openvino.inference_engine.ie_api.DataPtr object
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:34:36.461795 41153 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:34:36.463591 41153 demo.py:112] Reading from cam 0
I1211 17:34:36.604611 41153 ie_tools.py:69] Initializing Inference Engine plugin for CPU
I1211 17:34:36.604834 41153 ie_tools.py:74] Loading network
I1211 17:34:36.625749 41153 ie_tools.py:82] Preparing input blobs
I1211 17:34:36.625821 41153 ie_tools.py:88] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
[1, 3, 128, 128] 2222222222222
[1, 3, 128, 128] 2222222222222
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/demo.py", line 76, in run
    detections = face_det.get_detections(frame)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 44, in get_detections
    out = self.net.forward(cv.resize(frame, (w, h)))
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py", line 42, in forward
    res = self.net.infer(inputs={self.input_key: self._preprocess(img)})
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py", line 37, in _preprocess
    img = np.expand_dims(cv.resize(img, (w, h)).transpose(2, 0, 1), axis=0)
NameError: name 'cv' is not defined
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:34:55.395357 41286 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:34:55.396440 41286 demo.py:112] Reading from cam 0
I1211 17:34:55.534856 41286 ie_tools.py:69] Initializing Inference Engine plugin for CPU
I1211 17:34:55.534971 41286 ie_tools.py:74] Loading network
I1211 17:34:55.558636 41286 ie_tools.py:82] Preparing input blobs
I1211 17:34:55.558785 41286 ie_tools.py:88] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
[1, 3, 128, 128] 2222222222222
[1, 3, 128, 128] 2222222222222
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/demo.py", line 76, in run
    detections = face_det.get_detections(frame)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 45, in get_detections
    detections = self.__decode_detections(out, frame.shape)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 52, in __decode_detections
    for detection in out[0, 0]:
TypeError: 'numpy.float32' object is not iterable
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:35:29.406582 41559 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:35:29.408602 41559 demo.py:112] Reading from cam 0
I1211 17:35:29.547688 41559 ie_tools.py:69] Initializing Inference Engine plugin for CPU
I1211 17:35:29.547806 41559 ie_tools.py:74] Loading network
I1211 17:35:29.571924 41559 ie_tools.py:82] Preparing input blobs
I1211 17:35:29.572052 41559 ie_tools.py:88] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
[[0.00624364 0.9937563 ]]
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/demo.py", line 76, in run
    detections = face_det.get_detections(frame)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 45, in get_detections
    detections = self.__decode_detections(out, frame.shape)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 53, in __decode_detections
    for detection in out[0, 0]:
TypeError: 'numpy.float32' object is not iterable
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:35:41.390217 41668 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:35:41.391255 41668 demo.py:112] Reading from cam 0
I1211 17:35:41.535692 41668 ie_tools.py:69] Initializing Inference Engine plugin for CPU
I1211 17:35:41.535809 41668 ie_tools.py:74] Loading network
I1211 17:35:41.561275 41668 ie_tools.py:82] Preparing input blobs
I1211 17:35:41.561433 41668 ie_tools.py:88] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
[[0.12945801 0.870542  ]]
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/demo.py", line 76, in run
    detections = face_det.get_detections(frame)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 45, in get_detections
    detections = self.__decode_detections(out, frame.shape)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 53, in __decode_detections
    for detection in out[0, 0]:
TypeError: 'numpy.float32' object is not iterable
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:35:56.560750 41853 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:35:56.562602 41853 demo.py:112] Reading from cam 0
I1211 17:35:56.703687 41853 ie_tools.py:69] Initializing Inference Engine plugin for CPU
I1211 17:35:56.703775 41853 ie_tools.py:74] Loading network
I1211 17:35:56.724779 41853 ie_tools.py:82] Preparing input blobs
I1211 17:35:56.724848 41853 ie_tools.py:88] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
0.08944959
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/demo.py", line 76, in run
    detections = face_det.get_detections(frame)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 45, in get_detections
    detections = self.__decode_detections(out, frame.shape)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 53, in __decode_detections
    for detection in out[0, 0]:
TypeError: 'numpy.float32' object is not iterable
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:36:26.724519 41966 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:36:26.726359 41966 demo.py:112] Reading from cam 0
I1211 17:36:26.871722 41966 ie_tools.py:69] Initializing Inference Engine plugin for CPU
I1211 17:36:26.871804 41966 ie_tools.py:74] Loading network
I1211 17:36:26.893686 41966 ie_tools.py:82] Preparing input blobs
I1211 17:36:26.893754 41966 ie_tools.py:88] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
[[0.00719842 0.99280155]]
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/demo.py", line 76, in run
    detections = face_det.get_detections(frame)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 45, in get_detections
    detections = self.__decode_detections(out, frame.shape)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 53, in __decode_detections
    for detection in out[0, 0]:
TypeError: 'numpy.float32' object is not iterable
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:36:36.433552 42071 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:36:36.435431 42071 demo.py:112] Reading from cam 0
I1211 17:36:36.575945 42071 ie_tools.py:69] Initializing Inference Engine plugin for CPU
I1211 17:36:36.576063 42071 ie_tools.py:74] Loading network
I1211 17:36:36.597440 42071 ie_tools.py:82] Preparing input blobs
I1211 17:36:36.597573 42071 ie_tools.py:88] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
[[0.00641715 0.9935828 ]]
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/demo.py", line 76, in run
    detections = face_det.get_detections(frame)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 45, in get_detections
    detections = self.__decode_detections(out, frame.shape)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 54, in __decode_detections
    confidence = detection[2]
IndexError: index 2 is out of bounds for axis 0 with size 2
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:36:47.118648 42175 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:36:47.120446 42175 demo.py:112] Reading from cam 0
I1211 17:36:47.259897 42175 ie_tools.py:69] Initializing Inference Engine plugin for CPU
I1211 17:36:47.260015 42175 ie_tools.py:74] Loading network
I1211 17:36:47.282675 42175 ie_tools.py:82] Preparing input blobs
I1211 17:36:47.282798 42175 ie_tools.py:88] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/de_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:36:26.724519 41966 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:36:26.726359 41966 demo.py:112] Reading from cam 0
I1211 17:36:26.871722 41966 ie_tools.py:69] Initializing Inference Engine plugin for CPU
I1211 17:36:26.871804 41966 ie_tools.py:74] Loading network
I1211 17:36:26.893686 41966 ie_tools.py:82] Preparing input blobs
I1211 17:36:26.893754 41966 ie_tools.py:88] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
[[0.00719842 0.99280155]]
Traceback (most recent call last):
  File "demo/demo.py", line 137, in <module>
    main()
  File "demo/demo.py", line 134, in main
    run(args, cap, face_detector, spoof_model, write_video)
  File "demo/demo.py", line 76, in run
    detections = face_det.get_detections(frame)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 45, in get_detections
    detections = self.__decode_detections(out, frame.shape)
  File "/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/wrapers.py", line 53, in __decode_detections
    for detection in out[0, 0]:
TypeError: 'numpy.float32' object is not iterable
((venv)) (base) ]0;prixa-ml@prixa-machine-learner: ~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[01;32mprixa-ml@prixa-machine-learner[00m:[01;34m~/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing[00m$ python demo/demo.py --fd_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.xml --spf_model /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar --cam_id 0 --config configs/config.py;
W1211 17:36:36.433552 42071 warnings.py:109] /home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/demo_tools/ie_tools.py:20: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  from openvino.inference_engine import IECore

I1211 17:36:36.435431 42071 demo.py:112] Reading from cam 0
I1211 17:36:36.575945 42071 ie_tools.py:69] Initializing Inference Engine plugin for CPU
I1211 17:36:36.576063 42071 ie_tools.py:74] Loading network
I1211 17:36:36.597440 42071 ie_tools.py:82] Preparing input blobs
I1211 17:36:36.597573 42071 ie_tools.py:88] Loading model to the plugin

==> Loading checkpoint
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
Op