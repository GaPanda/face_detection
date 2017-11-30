# -*- coding: utf-8 -*-

from imutils import resize
from imutils import face_utils
from threading import Thread
from argparse import ArgumentParser
from time import strftime
from time import gmtime
from time import time
from dlib import get_frontal_face_detector
from dlib import shape_predictor
from cv2 import imwrite
from cv2 import imdecode
from cv2 import VideoCapture
from cv2 import destroyAllWindows
from cv2 import imshow
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
from cv2 import circle
from cv2 import waitKey
from urllib.request import urlopen
import numpy as np

class SaveThread(Thread):
    #Thread class for saving frame and face landmarks in file
    def __init__(self, num_ses, num_frame, cur_time, frame, dots):
        Thread.__init__(self)
        self.num_ses = num_ses
        self.num_frame = num_frame
        self.cur_time = cur_time
        self.frame = frame
        self.dots = dots
        
    def run(self):
        #Starting thread
        print("[WRITE] Session:", self.num_ses, "Frame:", self.num_frame)
        image_filename = "images\ses_" + str(self.num_ses) + "_" + "im_" + str(self.num_frame) \
            + "_" + strftime("%Y_%m_%d_%H-%M-%S", gmtime(self.cur_time)) + ".png"
        dots_filename = "images\ses_" + str(self.num_ses) + "_" + "dots_" + str(self.num_frame) \
            + "_" + strftime("%Y_%m_%d_%H-%M-%S", gmtime(self.cur_time)) + ".txt"
        imwrite(str(image_filename), self.frame)
        with open(dots_filename, "w") as f:
            point = 1
            print("image: ", image_filename, file = f)
            print("face number: 0", file = f)
            print("points:", file = f)
            print("{", file = f)
            for dot in self.dots:
                print("\tpoint:", str(point),  "{ x:", str(dot[0]) + "; y:", str(dot[1]), "}", file = f)
                point += 1
            print("}", file = f)
                

class IpCamera:
    #Class ipcam stream with urllib
    def __init__(self, url, user=None, password=None):
        self.req = url
        try:
            response = urlopen(self.req)
        except WindowsError:
            raise Exception("[ERROR] Unable to connect with ip camera!")
        except:
            raise

    def get_frame(self):
        #Getting frame from URL request
        response = urlopen(self.req)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = imdecode(img_array, 1)
        return frame

class Camera:
    #Class webcam steam with opencv
    def __init__(self, camera = 0):
        self.cam = VideoCapture(camera)
        try:
            self.shape = self.get_frame().shape
        except AttributeError:
            raise Exception("[ERROR] Webcamera not accessible!")
        except:
            raise

    def get_frame(self):
        #Get frame from webcam
        _, frame = self.cam.read()
        return frame

def stream_face_detection(cam, args):
    status_saving = False
    itr_ses = 0
    itr_frame = 0
    print("[PROCESS] Loading facial landmark predictor...")
    try:
        detector = get_frontal_face_detector()
        predictor = shape_predictor(args["shape_predictor"])
    except:
        raise Exception("[ERROR] Can't load predictor...")
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        frame = cam.get_frame()
        frame_for_save = frame
        frame = resize(frame, width=700)
        gray = cvtColor(frame, COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            if status_saving:
                cur_time = time()
                save_thread = SaveThread(itr_ses, itr_frame, cur_time, frame_for_save, shape)
                save_thread.start()
                itr_frame += 1
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                circle(frame, (x, y), 1, (0, 0, 255), -1)
        # show the frame
        imshow("Stream", frame)
        key = waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            status_saving = True
        elif key == ord("s"):
            status_saving = False
            itr_ses += 1
            itr_frame = 0
    destroyAllWindows()
    print("[INFO] Exit...")

def main():
    #main func
    # construct the argument parse and parse the arguments
    ap = ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=False,
                    default="shape_predictor_68_face_landmarks.dat",
                    help="path to facial landmark predictor")
    args = vars(ap.parse_args())
    # menu of programm
    print("[INFO] Press Q to exit from programm.")
    print("[INFO] Press R to record frames with landmarks.")
    print("[INFO] Press S to stop recording.")
    print("[MENU] 1. Webcamera Stream")
    print("[MENU] 2. Ipcamera Stream")
    print("[MENU] 3. Exit")
    cmd = input("[INPUT] Введите команду: ")
    if cmd == "1":
        print("[PROCESS] Trying to connect with camera...")
        try:
            webcam_obj = Camera()
            stream_face_detection(webcam_obj, args)
        except Exception as err:
            print(err.args[0])
            print("[INFO] Exit...")
    elif cmd == "2":
        print("[PROCESS] Trying to connect with camera...")
        try:
            ipcam_obj = IpCamera("http://192.168.0.102/axis-cgi/jpg/image.cgi")
            stream_face_detection(ipcam_obj, args)
        except Exception as err:
            print(err.args[0])
            print("[INFO] Exit...")
    elif cmd == "3":
        print("[INFO] Exit...")
    else:
        print("[ERROR] Wrong input!")    

if __name__ == '__main__':
    main()