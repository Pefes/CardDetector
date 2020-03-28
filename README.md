# CardDetector

Program that detects white cards with symbols on the image and names them according to sign on them. Made for communication computer-human on studies in Poznan University of Technology.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

What things you need to run the software

Python 3.0+

PyCharm IDE or different

OpenCV

```
pip install opencv-python
```

NumPy

```
pip install numpy
```

### Installing

A step by step series of examples that tell you how to get a development env running

Clone repository or download .zip

```
git clone https://github.com/Pefes/CardDetector
```

Open project in IDE (PyCharm)

Run the program

If you want to run program with your own images just simply put them in one of the directories

```
./images/easy
./images/medium
./images/hard
```

Or change the path directly in main.py code

```
paths = ["images/[myOwnDirectory]"]
```

Program automatically will search for images in given path

## Built With

* [Python](https://docs.python.org/3/) - Programming language
* [OpenCV](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html) - Library for image processing
* [NumPy](https://numpy.org/doc/) - Library for image storage and tranformations (like matrix)

## Authors

* [Pefes](https://github.com/Pefes) - *detecting cards and cut them* 
* [TheTerabit](https://github.com/TheTerabit) - *detecting symbols on cut out cards and return their name* 

## Some outputs

![img1](https://github.com/Pefes/CardDetector/blob/master/report/medium/1.jpg)

![img1](https://github.com/Pefes/CardDetector/blob/master/report/medium/5.jpg)

![img1](https://github.com/Pefes/CardDetector/blob/master/report/medium/7.jpg)

![img1](https://github.com/Pefes/CardDetector/blob/master/report/hard/6.jpg)

![img1](https://github.com/Pefes/CardDetector/blob/master/report/easy/8.jpg)

![img1](https://github.com/Pefes/CardDetector/blob/master/report/medium/3.jpg)
