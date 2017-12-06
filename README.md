#Face detection using dlib
Tracking face landmarks with IpCam and WebCam.

##Dependencies:

* [dlib](http://dlib.net/)
* [opencv](https://opencv.org/)
* [python 3.6](https://www.python.org/downloads/release/python-363/)
* imutils
* [shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
* numpy
* postgresql

To install all libraries use:
```
pip install numpy opencv-contrib-python imutils py-postgresql
```

Download library [dlib](https://pypi.python.org/pypi/dlib/19.7.0) and use:
```
pip install path_to\dlib-19.7.0-cp36-cp36m-win_amd64.whl
```

If shape_predictor located in other path:
```
python video_face_landmark.py -p path\shape_predictior_68_face_landmarks.dat
```
