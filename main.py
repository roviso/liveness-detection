from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
import cv2 as cv
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import utils
from demo_tools import TorchCNN, VectorCNN, FaceDetector
import cv2
from deepface import DeepFace
# from tensorflow.python.framework import ops


app = FastAPI()
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.on_event("startup")
async def startup_event():
    print("app started")
    # global graph
    # graph = ops.get_default_graph()
    # do start connection mongodb


class Args:
    def __init__(self, GPU=0, cam_id=0, config='configs/config.py', cpu_extension=None,
                 device='CPU', fd_model='path_to_fd_model.xml', fd_thresh=0.6,
                 spf_model='path_to_spf_model.pth.tar', spoof_thresh=0.4, video=None, write_video=False):
        self.GPU = GPU
        self.cam_id = cam_id
        self.config = config
        self.cpu_extension = cpu_extension
        self.device = device
        self.fd_model = fd_model
        self.fd_thresh = fd_thresh
        self.spf_model = spf_model
        self.spoof_thresh = spoof_thresh
        self.video = video
        self.write_video = write_video

# Initialize args with default values or specify your own
args = Args()


from collections import namedtuple

Args = namedtuple('Args', ['GPU', 'cam_id', 'config', 'cpu_extension', 'device', 
                           'fd_model', 'fd_thresh', 'spf_model', 'spoof_thresh', 'video', 'write_video'])

args = Args(GPU=0, cam_id=0, config='configs/config.py', cpu_extension=None,
            device='CPU', fd_model='/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml', fd_thresh=0.6,
            spf_model='/home/prixa-ml/Desktop/projects/liveness-detection/Face-Recognition/spoof_or_not/light-weight-face-anti-spoofing/AntiSpoofing/spf_models/MN3_antispoof.pth.tar', spoof_thresh=0.4, video=None, write_video=False)

device = args.device + f':{args.GPU}' if args.device == 'cuda' else 'cpu'

# Assuming args is already defined as per your input
fd_model_path = args.fd_model
fd_thresh = args.fd_thresh

config_path = args.config
spf_model_path = args.spf_model

# Initialize FaceDetector
face_detector = FaceDetector(fd_model_path, fd_thresh, args.device, None)

# Load config and initialize SpoofModel
config = utils.read_py_config(config_path)
spoof_model = utils.build_model(config, args, strict=True, mode='eval')
spoof_model = TorchCNN(spoof_model, spf_model_path, config, device)



def pred_spoof(frame, detections, spoof_model):
    """Get prediction for all detected faces on the frame"""
    faces = []
    for rect, _ in detections:
        left, top, right, bottom = rect
        # cut face according coordinates of detections
        faces.append(frame[top:bottom, left:right])
    if faces:
        output = spoof_model.forward(faces)
        output = list(map(lambda x: x.reshape(-1), output))
        return output
    return None, None

def draw_detections(frame, detections, confidence, thresh):
    """Draws detections and labels"""
    for i, rect in enumerate(detections):
        left, top, right, bottom = rect[0]
        if confidence[i][1] > thresh:
            label = f'spoof: {round(confidence[i][1]*100, 3)}%'
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        else:
            label = f'real: {round(confidence[i][0]*100, 3)}%'
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return frame




@app.post("/liveness")
async def detect_spoof(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    frame = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

    # Your face detection logic here
    detections = face_detector.get_detections(frame)


    confidence = pred_spoof(frame, detections, spoof_model)
    is_fake = "FAKE" if confidence and confidence[0][1] > 0.5 else "REAL"  # Adjust according to your model's output
    if is_fake == "REAL":
        status = "success"
    elif is_fake == "FAKE":
        status = "failed"
    else:
        status = "failed"
    return {"status": status,
            "detections": str(detections)}

    # return {"result": is_fake}




models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]


metrics = ["cosine", "euclidean", "euclidean_l2"]


def read_imagefile(file):
    nparr = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


@app.post("/analyze/")
async def analyzer(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())


    # with graph.as_default():

    demography = DeepFace.analyze(img_path = image, 
                                #   actions=["age", "gender", "race"],
                                  detector_backend = backends[3])
    # demography = DeepFace.analyze(img, actions=["emotion"])
    print(demography)
    return {"prediction": demography}


@app.post("/verification/")
async def verification_route(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    print("reading image 1")
    image1 = read_imagefile(await file1.read())

    print("reading image 2")
    image2 = read_imagefile(await file2.read())


    # with graph.as_default():
    print("Varifying Match")
    result = DeepFace.verify(img1_path = image1, 
                             img2_path = image2,
                             distance_metric = metrics[2], 
                             model_name = models[2],
                             detector_backend = backends[4]
                            
                             )
    print(f"result: {result}")

    return True if result['verified'] == True else False 