FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

ENV USERNAME=francolu

# Set the working directory and user
WORKDIR /home/$USERNAME

# Set timezone
ENV TZ=Europe/Rome
RUN ln -sf /usr/share/zoneinfo/Europe/Rome /etc/localtime

# Clone the repo
RUN apt-get update && apt-get install -y git nano tree screen wget
RUN apt-get install -y ffmpeg libsm6 libxext6 build-essential cmake curl g++ zip unzip ca-certificates

# Clone and install evaluation repo
RUN git clone https://github.com/davisvideochallenge/davis2017-evaluation.git
RUN pip install imageio
RUN python ./davis2017-evaluation/setup.py install

# Set permission
# ARG USER_ID=1038
# ARG GROUP_ID=1040
# RUN addgroup --gid $GROUP_ID $USERNAME
# RUN adduser --home /home/$USERNAME --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USERNAME

# Install requirements
# RUN pip install --ignore-installed -r https://raw.githubusercontent.com/paolomandica/sapienza-video-contrastive/main/requirements.txt

USER $USERNAME