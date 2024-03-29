# Getting the ubuntu docker image from the docker hub
FROM ubuntu
MAINTAINER sachin.bulchandani@iosb-ina.fraunhofer.de

RUN apt-get -y update

# Installing CONDA

RUN apt-get install curl wget xz-utils -y
RUN cd /home
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
RUN bash ./Miniconda3-py39_4.9.2-Linux-x86_64.sh

# Installing git, pull repo and creating the VM.
RUN apt-get install git -y
RUN git config credential.helper store
RUN cd /home
RUN git clone https://github.com/sachin301195/Research-Project.git
RUN cd Research-Project
RUN source ~/.bashrc # to be able to start conda
RUN conda env create -f environment.yml
#
RUN echo "Setup installation successfully completed."
RUN "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
#
## Activating conda on the path
RUN conda activate Complex_fms
#
## test run before training
## conda activate Complex_fms
## cd /home/sachin/Research-Project/trial network && python trial_net.py
## cd ..
## python main.py # input requried arguments
#
## create the Docker image for the developed container
#docker commit -a cmfs_comm -m "docker container for the Research-Project" [container name, CMFS] CMFS:1.0
#
## Make the docker start always in the Research-Project directory and the Research-Project-environment activated
## The corresponding Dockerfile looks like this:
#FROM CMFS:1.0
#WORKDIR /Research-Project
## RUN echo "conda activate Research-Project" >> ~/.bashrc
#
## Build this docker, cd to the directory of the Dockerfile
#docker build -t CMFS:1.x .
#
## Run the docker and run tuning
#docker run -it --name cmfstrainer --shm-size=10gb -v ~/docker_exch:/home/docker_exch CMFS:1.0
## --rm is missing -> the docker is not deleted after exiting it
## the docker_exch folder stores the data persistently that the docker container stores into it
## and the docker container has access to all fiels inside it.
## git pull
## python main.py # input required arguments
#Ctrl-p + Ctrl-q
## to detach from docker container correctly, so that it does not get stopped,
## use the escape sequence Ctrl-p + Ctrl-q
## note: This will continue to exist in a stopped state once exited (see "docker ps -a")
#docker attach rlagent # after training completed, attach to container




