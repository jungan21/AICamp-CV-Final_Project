#coding:utf-8
# 该文件相当于web server application的server 端
import cv2
import numpy as np
import os
import tensorflow as tf

from flask import (
    Flask,
    request,
    render_template,
    jsonify
)
from flask_bootstrap import Bootstrap

from detect.MtcnnDetector import MtcnnDetector
from detect.detector import Detector
from detect.fcn_detector import FcnDetector
from detect.mtcnn_model import P_Net, R_Net, O_Net
from detect_acc import detect_face
from recognize.facenet import FaceNet


# Init MtcnnDetector 这里是我们九章课上自己train 的MTCNN model
# pnet, rnet, onet's threshold. Only if the number > threshold, we consider this as people's face
thresh = [0.9, 0.6, 0.7]
min_face_size = 24
stride = 2
slide_window = False
shuffle = False
detectors = []
prefix = ['detect/MTCNN_model/PNet_landmark/PNet', 'detect/MTCNN_model/RNet_landmark/RNet', 'detect/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
batch_size = [2048, 256, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

detectors.append(FcnDetector(P_Net, model_path[0]))
detectors.append(Detector(R_Net, 24, batch_size[1], model_path[1]))
detectors.append(Detector(O_Net, 48, batch_size[2], model_path[2]))

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)


# Init another version of MtcnnDetector. 别人train Facenet的时候用的MTCNN model
print('Creating networks and loading parameters')
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
margin = 44
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def detect_face_by_caffemodel(img):
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        
    faces = []
    for i in range(bounding_boxes.shape[0]):
        det = np.squeeze(bounding_boxes[i,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        align = prewhiten(cropped)
        faces.append(align)
    return faces


# Init FaceNet
model_path = 'recognize/facenet_model'
face_net = FaceNet(model_path)


app = Flask(__name__)
bootstrap = Bootstrap(app)

# server 端 默认render 的主页
@app.route('/')
def index():
    return render_template('index.html')
    

#
def calc_score(img1, img2):
    thresh = 0.35

    # MTCNN 抠出人脸来. Note: 这里返回的 faces1, faces2 都输数组类型
    faces1 = mtcnn_detector.get_face_from_single_image(img1)
    faces2 = mtcnn_detector.get_face_from_single_image(img2)
    
    # 保证每张上传的图片里仅仅包含一张人脸
    if len(faces1) != 1 or len(faces2) != 1:
        return 'Please upload image with exact one person.', 0

    # NOTE: 这里face_net model  用的是别人train 好的model. 别人train facenet的时候也会用到MTCNN (别人的)model
    # 而这个 calc_score这个方法里 的MTCNN是我们课上自己train的model
    emb1 = face_net.predict(faces1[0])
    emb2 = face_net.predict(faces2[0])

    score = np.sqrt(np.sum(np.square(np.subtract(emb1, emb2))))
    if score < thresh:
        return 'Same person', str(score)
    else:
        return 'Not same person', str(score)

# MTCNN use to train facenet 的时候用的MTCNN
def calc_score_with_version2_detector(img1, img2):
    thresh = 0.8
    
    faces1 = detect_face_by_caffemodel(img1)
    faces2 = detect_face_by_caffemodel(img2)
    
    
    if len(faces1) != 1 or len(faces2) != 1:
        return 'Please upload image with exact one person.', 0

    # embeding
    emb1 = face_net.predict(faces1[0])
    emb2 = face_net.predict(faces2[0])

    score = np.sqrt(np.sum(np.square(np.subtract(emb1, emb2))))
    if score < thresh:
        return 'Same person', str(score)
    else:
        return 'Not same person', str(score)
    
# server 端 接收到upload event后
@app.route('/get_score', methods=['POST'])
def get_score():
    if request.method == 'POST':
        files1 = request.files['file1']
        files2 = request.files['file2']
        img1 = cv2.imdecode(np.fromstring(files1.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.fromstring(files2.read(), np.uint8), cv2.IMREAD_COLOR)
        
        result, score = calc_score_with_version2_detector(img1, img2)
        return jsonify(result=result, score=score)

        

# server 端的入口
if __name__ == "__main__":
    app.run(host='0.0.0.0')
