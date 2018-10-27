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

(base) s4467575@wkstncontrols-MacBook-Pro:~/Documents/Jun/jiuzhang_AI/projects/AICamp-CV-Final_Project$docker image ls
REPOSITORY                                              TAG                 IMAGE ID            CREATED             SIZE
face                                                    latest              3004e6b02801        6 minutes ago       1.68GB
python                                                  2.7                 3c43a5d4034a        10 days ago         908MB
jungan21/colt_demo                                      latest              9049e86c2dbf        2 months ago        119MB
jungan21/springiojun/colt_demo                          0.1-SNAPSHOT        9049e86c2dbf        2 months ago        119MB

```

Then, spin up docker container and run the docker image (i.e. face).

```
docker run -it -p 5000:5000 face python app.py

(base) s4467575@wkstncontrols-MacBook-Pro:~/Documents/Jun/jiuzhang_AI/projects/AICamp-CV-Final_Project$docker container ls
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS                    NAMES
6e20d688b8e7        face                "python app.py"     2 minutes ago       Up About a minute   0.0.0.0:5000->5000/tcp   priceless_meninsky
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
