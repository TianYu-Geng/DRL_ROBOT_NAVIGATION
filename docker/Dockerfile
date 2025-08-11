# ------------------------------------------------------------
# Ubuntu 20.04 + ROS Noetic + Gazebo11 + PyTorch 1.10(CUDA 11.3)
# 支持 gzserver/gzclient GUI，默认离线模型，不卡 Fuel
# ------------------------------------------------------------
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=user
ARG UID=1000
# 时区
ENV TZ=Etc/UTC \
    WORKSPACE=/home/${USERNAME}/drl_navigation

# 基础 & 常用 & GUI 依赖（一次装干净）
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates curl gnupg2 lsb-release \
    build-essential git sudo nano vim tmux htop tree unzip zip \
    python3.8 python3.8-dev python3-pip python-is-python3 \
    # X11/OpenGL for gzclient/RViz
    libgl1-mesa-glx libgl1-mesa-dri libglu1-mesa mesa-utils \
    libx11-6 libxext6 libxrender1 libxrandr2 libxi6 libxxf86vm1 \
    # 网络/诊断
    iproute2 net-tools psmisc \
    && rm -rf /var/lib/apt/lists/*

# ROS Noetic 源 & 安装
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
    > /etc/apt/sources.list.d/ros1.list \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/* \
    && rosdep init && rosdep update

# Gazebo11 + ROS 插件（Noetic 对应 classic 11）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gazebo11 libgazebo11-dev \
    ros-noetic-gazebo-ros ros-noetic-gazebo-ros-control \
    ros-noetic-gazebo-plugins ros-noetic-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*

# Python 包（CUDA 11.3 对应的 torch 1.10.2）
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
    torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html \
    tensorboard

# 默认环境（避免 Fuel 联网；补资源路径；自动 source）
ENV ROS_DISTRO=noetic \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    GAZEBO_MODEL_DATABASE_URI="" \
    GAZEBO_RESOURCE_PATH=/usr/share/gazebo-11:$GAZEBO_RESOURCE_PATH \
    GAZEBO_MODEL_PATH=/usr/share/gazebo-11/models:$GAZEBO_MODEL_PATH \
    GAZEBO_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:$GAZEBO_PLUGIN_PATH

RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc

# 非 root 用户
ENV WORKSPACE=/home/${USERNAME}/drl_navigation
RUN groupadd -g ${UID} ${USERNAME} \
    && useradd -m -u ${UID} -g ${UID} -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p ${WORKSPACE}
USER ${USERNAME}
WORKDIR ${WORKSPACE}

# 便捷初始化：进入容器自动 source ROS
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/${USERNAME}/.bashrc \
    && echo 'export GAZEBO_RESOURCE_PATH=/usr/share/gazebo-11:$GAZEBO_RESOURCE_PATH' >> /home/${USERNAME}/.bashrc \
    && echo 'export GAZEBO_MODEL_PATH=$HOME/.gazebo/models:/usr/share/gazebo-11/models:$GAZEBO_MODEL_PATH' >> /home/${USERNAME}/.bashrc \
    && echo 'export GAZEBO_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:$GAZEBO_PLUGIN_PATH' >> /home/${USERNAME}/.bashrc \
    && mkdir -p /home/${USERNAME}/.gazebo

SHELL ["/bin/bash", "-c"]
CMD ["bash"]
