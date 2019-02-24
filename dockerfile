FROM ufoym/deepo:all-py36-cu90       
#FROM achiatti/semantic-map-docker-baseline-cpu
# Set the working directory to /app
WORKDIR /app

#EXPOSE 443

RUN apt-get update

RUN pip3 install torch torchvision visdom 

RUN apt-get -y install tmux
# Copy the current directory contents into the container at /app
COPY semantic-map-docker /app/semantic-map-docker

WORKDIR /app/semantic-map-docker


#CMD CUDA_VISIBLE_DEVICES=1 python -u train.py -il data/train/imgset_left.npy -ir data/train/imgset_right.npy -l /data/train/gt_labels.py -m results/model_16_e7_20.json -w results/model_16_e7_20.h5
#CMD python -u siamese_normxcorr_fixed.py 0.0001 1e-6 32 50 cpuonly
#CMD CUDA_VISIBLE_DEVICES=0,1 python -u eval.py --flag scores -il data/train/imgset_left.npy -ir data/train/imgset_right.npy -l data/train/gt_labels.npy -m new_results/model_16times6_e7_50.json -w new_results/model_16times6_e7_50.h5 --name shapenetset_test
