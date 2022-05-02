import cv2
import streamlit as st
import numpy as np
from PIL import Image
import os


## CONSTANTS
C_PATH= os.getcwd()
DIR_MODELS = os.path.join(C_PATH, 'models/')
DIR_PROTOS = os.path.join(C_PATH, 'protos/')
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGELIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
O_INDEX = np.array([i for i in range(0, 101)])
FACEPROTO = DIR_PROTOS + "opencv_face_detector.pbtxt"
FACEMODEL = DIR_MODELS + "opencv_face_detector_uint8.pb"
AGEPROTO_A = DIR_PROTOS + "age_deploy.prototxt"
AGEMODEL_A = DIR_MODELS + "age_net.caffemodel"
AGEPROTO_I = DIR_PROTOS + "age.prototxt"
AGEMODEL_I = DIR_MODELS + "dex_chalearn_iccv2015.caffemodel"
# ----------------------------------------------------------------------------------------------------------------------
## Functions
# highlightFace function
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


def check_age(image):
    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    # output_indexes = np.array([i for i in range(0, 101)])

    faceNet=cv2.dnn.readNet(FACEMODEL, FACEPROTO)
    ageNet=cv2.dnn.readNet(AGEMODEL_A, AGEPROTO_A)
    padding=20
    resultImg,faceBoxes=highlightFace(faceNet,image)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face=image[max(0,faceBox[1]-padding):
                min(faceBox[3]+padding,image.shape[0]-1),max(0,faceBox[0]-padding)
                :min(faceBox[2]+padding, image.shape[1]-1)]

        face = cv2.resize(face, (224, 224))
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        # apparent_predictions = round(np.sum(age_dist * output_indexes), 2)

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]

        cv2.putText(resultImg, f'{age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        return resultImg
#----------------------------------------------------------
# --------------------------------------

def main_loop():
    # Using object notation
    add_selectbox = st.sidebar.selectbox(
        "Seleccionar modo de imagen",
        ("Webcam", "Archivo")
    )
    st.title("Detector de edad")
    st.subheader("Aplicación web que permite detectar la edad de una persona")
    if( add_selectbox == "Webcam"):       
        image_file = st.camera_input("Webcam")
        if not image_file:
            return None

        original_image = image_file.getvalue()
        original_image = cv2.imdecode(np.frombuffer(original_image, np.uint8), cv2.IMREAD_ANYCOLOR)
        original_image = cv2.cvtColor(original_image , cv2.COLOR_BGR2RGB)
    else:
        image_file = st.file_uploader("Sube tu imagen", type=['jpg'])
        if not image_file:
            return None

        original_image = Image.open(image_file)
        original_image = np.array(original_image)
    processed_image = check_age(original_image)

    st.text("Resultados:")
    st.image([processed_image])


if __name__ == '__main__':
    main_loop()