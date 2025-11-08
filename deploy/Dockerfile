FROM ubuntu:20.04

#####################################################################
# FOR UBUNTU MIRRORS
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list
RUN sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN apt-get update
#####################################################################
# FOR PYPI MIRRORS
RUN apt-get install -y python3-pip
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/simple pip -U
RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple
#####################################################################
# # FOR UV PIP
# RUN touch ~/.config/uv/uv.toml
# RUN echo "[[index]] \
# url = "https://mirrors.ustc.edu.cn/pypi/simple" \
# default = true" >> ~/.config/uv/uv.toml
# #####################################################################
# # FOR CONDA
# RUN touch $HOME/.condarc
# RUN echo "channels: \
#   - nodefaults \
# custom_channels: \
#   conda-forge: https://mirrors.ustc.edu.cn/anaconda/cloud \
#   bioconda: https://mirrors.ustc.edu.cn/anaconda/cloud \
# show_channel_urls: true" >> $HOME/.condarc
# #####################################################################
# # FOR ROS 1 MIRRORS
# RUN apt-get install -y gpg wget curl
# RUN gpg --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# RUN gpg --export C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 | tee /usr/share/keyrings/ros.gpg > /dev/null
# RUN sh -c 'echo "deb [signed-by=/usr/share/keyrings/ros.gpg] https://mirrors.ustc.edu.cn/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# # FOR ROS 2 MIRRORS
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://mirrors.ustc.edu.cn/ros2/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
# RUN apt-get update
#####################################################################
# INSTALL ZSH
RUN apt-get install zsh git -y && chsh -s /bin/zsh
RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh \
    && cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc 
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
RUN sed -i 's/plugins=(git)/plugins=(git zsh-syntax-highlighting)/g' ~/.zshrc
####################################################################

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    wget \
    git \
    build-essential \
    cmake \
    python3-pip \
    python3-dev \
    net-tools \
    iputils-ping \
    vim \
    sudo \
    locales \
    tzdata \
    xvfb \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglfw3-dev \
    libglew-dev \
    libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y \
    ros-foxy-desktop \
    ros-foxy-ros-base \
    ros-foxy-rmw-cyclonedds-cpp \
    ros-foxy-rosidl-generator-dds-idl \
    ros-foxy-rclpy \
    ros-foxy-rclcpp \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update

WORKDIR /root

RUN git clone https://github.com/unitreerobotics/unitree_ros2.git /root/unitree_ros2
RUN rm -rf /root/unitree_ros2/cyclonedds || true
RUN mkdir -p /root/unitree_ros2/cyclonedds_ws/src
WORKDIR /root/unitree_ros2/cyclonedds_ws/src
RUN git clone https://github.com/ros2/rmw_cyclonedds -b foxy && \
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
WORKDIR /root/unitree_ros2/cyclonedds_ws
RUN colcon build --packages-select cyclonedds

RUN /bin/bash -c "source /opt/ros/foxy/setup.bash && \
    source /root/unitree_ros2/cyclonedds_ws/install/setup.bash && \
    colcon build --packages-select rmw_cyclonedds_cpp"

WORKDIR /root/unitree_ros2
RUN /bin/bash -c "source /opt/ros/foxy/setup.bash && \
    source /root/unitree_ros2/cyclonedds_ws/install/setup.bash && \
    colcon build --packages-ignore cyclonedds rmw_cyclonedds_cpp"

RUN git clone https://github.com/yongce-liu/CLONE.git /root/ws/CLONE

WORKDIR /root/ws/CLONE
RUN pip3 install --upgrade pip
RUN pip3 install --ignore-installed -r requirements.txt
RUN pip3 install pyzmq

WORKDIR /root/ws
RUN git clone https://github.com/Yutang-Lin/VisionWrapper.git
WORKDIR /root/ws/VisionWrapper
RUN pip3 install -e .
ENV PYTHONPATH="/root/ws:/root/ws/VisionWrapper:${PYTHONPATH}"

RUN echo "# ROS2 Environment Setup" >> /root/.bashrc && \
    echo "source /opt/ros/foxy/setup.bash" >> /root/.bashrc && \
    echo "source /root/unitree_ros2/cyclonedds_ws/install/setup.bash" >> /root/.bashrc && \
    echo "source /root/unitree_ros2/install/setup.bash" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc && \
    echo "export PYTHONPATH=/root/ws:/root/ws/VisionWrapper:\$PYTHONPATH" >> /root/.bashrc && \
    echo "# Aliases for convenience" >> /root/.bashrc && \
    echo "alias ll='ls -alF'" >> /root/.bashrc && \
    echo "alias la='ls -A'" >> /root/.bashrc && \
    echo "alias l='ls -CF'" >> /root/.bashrc && \
    echo "echo 'ROS2 Foxy environment loaded'" >> /root/.bashrc && \
    echo "echo 'Unitree ROS2 workspace loaded'" >> /root/.bashrc && \
    echo "echo 'VisionWrapper available'" >> /root/.bashrc

RUN echo '#!/bin/bash\n\
source /opt/ros/foxy/setup.bash\n\
source /root/unitree_ros2/cyclonedds_ws/install/setup.bash\n\
source /root/unitree_ros2/install/setup.bash\n\
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp\n\
export PYTHONPATH=/root/ws:/root/ws/VisionWrapper:$PYTHONPATH\n\
exec "$@"' > /ros_entrypoint.sh && \
    chmod +x /ros_entrypoint.sh

WORKDIR /root/ws/CLONE

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["/bin/bash"]