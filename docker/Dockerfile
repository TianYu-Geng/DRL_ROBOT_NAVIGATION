# ------------------------------------------------------------
# Image: Ubuntu 20.04 + ROS Noetic + Python 3.8 + PyTorch 1.10 + TensorBoard
# ------------------------------------------------------------
# ---------- Base image ----------
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# ---------- Basic OS utilities ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata curl gnupg2 lsb-release ca-certificates \
    build-essential git sudo python3.8 python3.8-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ---------- ROS Noetic ----------
# Add ROS repository & key
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros1.list \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# Initialise rosdep
RUN rosdep init && rosdep update

# ---------- Python packages ----------
RUN python3 -m pip install --upgrade pip \
&& pip install --no-cache-dir \
        torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 \
        -f https://download.pytorch.org/whl/cu113/torch_stable.html \
        tensorboard

# -------- ROS 自动 source ----------
RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc

# ---------- Locale & Environment ----------
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    ROS_DISTRO=noetic

# -------- Workspace ----------
ARG USERNAME=dev
ARG UID=1000
ENV WORKSPACE=/home/${USERNAME}/drl_ws
RUN groupadd -g ${UID} ${USERNAME} \
    && useradd -m -u ${UID} -g ${UID} -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p $WORKSPACE && chown ${USERNAME}:${USERNAME} $WORKSPACE
WORKDIR $WORKSPACE

USER ${USERNAME}
SHELL ["/bin/bash", "-c"]
CMD ["bash"]