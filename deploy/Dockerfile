FROM ubuntu:20.04

#####################################################################
# FOR UBUNTU MIRRORS
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list
RUN sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN apt-get update
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
######################################################################
# FOR SUDO INSTALLATION
ARG DOCKER_UID=1000
ARG DOCKER_GID=1000
ARG DOCKER_USER=user
ARG HOME=/home/${DOCKER_USER}
## Set environment variables
RUN apt install -y -o Dpkg::Options::="--force-confold" sudo
RUN groupadd -g $DOCKER_GID $DOCKER_USER \
    && useradd --uid $DOCKER_UID --gid $DOCKER_GID -m $DOCKER_USER \
    && echo "$DOCKER_USER ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER ${DOCKER_USER}
#####################################################################
# INSTALL ZSH
RUN sudo apt-get install zsh git -y && sudo chsh -s /bin/zsh
RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ${HOME}/.oh-my-zsh \
    && cp ${HOME}/.oh-my-zsh/templates/zshrc.zsh-template ${HOME}/.zshrc
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
RUN sed -i 's/plugins=(git)/plugins=(git zsh-syntax-highlighting)/g' ${HOME}/.zshrc
#####################################################################
# FOR PYPI MIRRORS
RUN sudo apt-get install -y python3-pip
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/simple pip -U
RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple
#####################################################################
# FOR UV PIP
RUN pip install uv -U
RUN mkdir -p ${HOME}/.config/uv && touch ${HOME}/.config/uv/uv.toml
RUN echo -e "[index]\nurl = \"https://mirrors.ustc.edu.cn/pypi/simple\"\ndefault = true" > ${HOME}/.config/uv/uv.toml
# #####################################################################
# FOR CONDA
RUN sudo apt-get install -y wget bzip2 wget
RUN mkdir $HOME/down && wget "https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh" -O $HOME/down/Miniforge3-Linux-x86_64.sh
RUN bash ${HOME}/down/Miniforge3-Linux-x86_64.sh -b -p ${HOME}/miniforge3 
ENV PATH=${HOME}/miniforge3/bin:$PATH
RUN conda init zsh bash \
    && mamba shell init --shell zsh
RUN touch $HOME/.condarc && echo -e "channels:\n  - nodefaults\ncustom_channels:\n  conda-forge: https://mirrors.ustc.edu.cn/anaconda/cloud\n  bioconda: https://mirrors.ustc.edu.cn/anaconda/cloud\nshow_channel_urls: true" > $HOME/.condarc
RUN conda config --set auto_activate_base false
#####################################################################

ENV DEBIAN_FRONTEND=noninteractive 
ENV TZ=Etc/UTC
RUN sudo apt-get update && sudo apt-get install -y \
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
    tmux \
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

USER ${DOCKER_USER}
# WORKDIR ${HOME}
RUN git clone https://github.com/unitreerobotics/unitree_ros2.git ${HOME}/unitree_ros2
RUN rm -rf ${HOME}/unitree_ros2/cyclonedds || true
RUN mkdir -p ${HOME}/unitree_ros2/cyclonedds_ws/src
WORKDIR ${HOME}/unitree_ros2/cyclonedds_ws/src
RUN git clone https://github.com/ros2/rmw_cyclonedds -b foxy && \
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
WORKDIR ${HOME}/unitree_ros2/cyclonedds_ws
RUN colcon build --packages-select cyclonedds

RUN /bin/bash -c "source /opt/ros/foxy/setup.bash && \
    source ${HOME}/unitree_ros2/cyclonedds_ws/install/setup.bash && \
    colcon build --packages-select rmw_cyclonedds_cpp"

WORKDIR ${HOME}/unitree_ros2
RUN /bin/bash -c "source /opt/ros/foxy/setup.bash && \
    source ${HOME}/unitree_ros2/cyclonedds_ws/install/setup.bash && \
    colcon build --packages-ignore cyclonedds rmw_cyclonedds_cpp"

# RUN git clone https://github.com/yongce-liu/CLONE.git ${HOME}/CLONE

# WORKDIR ${HOME}/CLONE
# RUN pip3 install --upgrade pip
# RUN pip3 install --ignore-installed -r requirements.txt
# RUN pip3 install pyzmq
# RUN rm -rf ${HOME}/CLONE

# WORKDIR ${HOME}
# RUN git clone https://github.com/Yutang-Lin/VisionWrapper.git
# WORKDIR ${HOME}/VisionWrapper
# RUN pip3 install -e .
# ENV PYTHONPATH="${HOME}:${HOME}/VisionWrapper:${PYTHONPATH}"

# RUN echo "# ROS2 Environment Setup" >> ${HOME}/.总收入从 && \
#     echo "source /opt/ros/foxy/setup.zsh" >> ${HOME}/.zshrc && \
#     echo "source ${HOME}/unitree_ros2/cyclonedds_ws/install/setup.zsh" >> ${HOME}/.zshrc && \
#     echo "source ${HOME}/unitree_ros2/install/setup.zsh" >> ${HOME}/.zshrc && \
#     echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ${HOME}/.zshrc && \
#     echo "export PYTHONPATH=${HOME}:${HOME}/VisionWrapper:\$PYTHONPATH" >> ${HOME}/.zshrc && \
#     echo "# Aliases for convenience" >> ${HOME}/.zshrc && \
#     echo "alias ll='ls -alF'" >> ${HOME}/.zshrc && \
#     echo "alias la='ls -A'" >> ${HOME}/.zshrc && \
#     echo "alias l='ls -CF'" >> ${HOME}/.zshrc && \
#     echo "echo 'ROS2 Foxy environment loaded'" >> ${HOME}/.zshrc && \
#     echo "echo 'Unitree ROS2 workspace loaded'" >> ${HOME}/.zshrc && \
#     echo "echo 'VisionWrapper available'" >> ${HOME}/.zshrc

# RUN echo '#!/bin/bash\n\
# source /opt/ros/foxy/setup.zsh\n\
# source ${HOME}/unitree_ros2/cyclonedds_ws/install/setup.zsh\n\
# source ${HOME}/unitree_ros2/install/setup.zsh\n\
# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp\n\
# export PYTHONPATH=${HOME}:${HOME}/VisionWrapper:$PYTHONPATH\n\
# exec "$@"' > /ros_entrypoint.sh && \
#     chmod +x /ros_entrypoint.sh

# WORKDIR ${HOME}/CLONE

# ENTRYPOINT ["/ros_entrypoint.sh"]
# CMD ["/bin/bash"]