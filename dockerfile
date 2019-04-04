FROM ufoym/deepo:all-py36-cu90       
#FROM achiatti/semantic-map-docker-baseline-cpu
# Set the working directory to /app
WORKDIR /app

#EXPOSE 443


RUN apt-get update

RUN apt-get -y install tmux
RUN apt-get -y install libopenblas-dev

RUN pip3 install torch torchvision visdom split-folders lightnet

#YOLO for lightnet
RUN python -m lightnet download yolo


# Copy the current directory contents into the container at /app
COPY semantic-map-docker /app/semantic-map-docker

WORKDIR /app/semantic-map-docker/ncc-extension

RUN python setup.py install

WORKDIR /app/semantic-map-docker/

#CMD