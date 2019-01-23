# AICamp CV Final Project - Face Recognition System

## Prework

Download the [model](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) of facenet, and put all the files into `recognize/facenet_model`.

```
cp /path/to/unzip/dir/* recognize/facenet_model
```

## Run

### For Docker users

First, build the docker image.
https://docs.docker.com/engine/reference/commandline/build/#build-with-path

```
docker build -t face .

(base) s4467575@Jun-Gan-Macbook:~/Documents/jun/AI/deep_learning/Object_Facce_Detection_Recognition/face_recognition_FaceNet/projects/Face_Recognition_System_jiuzhang$docker build -t face .

(base) s4467575@Jun-Gan-Macbook:~/Documents/jun/AI/deep_learning/Object_Facce_Detection_Recognition/face_recognition_FaceNet/projects/Face_Recognition_System_jiuzhang$docker image ls
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
face                latest              2be5ba22d3f3        4 minutes ago       1.89GB
python              2.7                 6f80c4b7a3c3        3 hours ago         912MB

```

Then, spin up docker container and run the docker image (i.e. face).

```
docker run -it -p 5000:5000 face python app.py

(base) s4467575@Jun-Gan-Macbook:~/Documents/jun/AI/deep_learning/Object_Facce_Detection_Recognition/face_recognition_FaceNet/projects/Face_Recognition_System_jiuzhang$docker container ls
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES

```

### For other users

First, install the python dependencies.

```
pip install -r requirements.txt
```

Then, run the webserver.

```
python app.py
```
