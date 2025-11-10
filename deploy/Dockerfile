FROM ubuntu:20.04

#####################################################################
# FOR UBUNTU MIRRORS AMD
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && apt-get upgrade -y
# #  FOR UBUNTU MIRRORS ARM
# RUN sed -i -e 's@//ports.ubuntu.com/\? @//ports.ubuntu.com/ubuntu-ports @g' \
#             -e 's@//ports.ubuntu.com@//mirrors.ustc.edu.cn@g' \
#             /etc/apt/sources.list && \
#     apt-get update && apt-get upgrade -y
# #####################################################################
# FOR ROS 1 MIRRORS
RUN apt-get update && apt-get install -y gpg lsb-release curl && \
    gpg --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    gpg --export C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 | tee /usr/share/keyrings/ros.gpg > /dev/null && \
    sh -c 'echo "deb [signed-by=/usr/share/keyrings/ros.gpg] https://mirrors.ustc.edu.cn/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# FOR ROS 2 MIRRORS
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://mirrors.ustc.edu.cn/ros2/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update
######################################################################
# # FOR INSTALLATION
# ARG DOCKER_UID=1000
# ARG DOCKER_GID=1000
# ARG DOCKER_USER=user
# ARG HOME=/home/${DOCKER_USER}
# ## Set environment variables
# RUN apt install -y -o Dpkg::Options::="--force-confold" sudo
# RUN groupadd -g $DOCKER_GID $DOCKER_USER \
#     && useradd --uid $DOCKER_UID --gid $DOCKER_GID -m $DOCKER_USER \
#     && echo "$DOCKER_USER ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# USER ${DOCKER_USER}
#####################################################################
# INSTALL ZSH
RUN apt-get update && apt-get install zsh git tmux -y && chsh -s /bin/zsh && \
    git clone https://github.com/robbyrussell/oh-my-zsh.git ${HOME}/.oh-my-zsh \
    && cp ${HOME}/.oh-my-zsh/templates/zshrc.zsh-template ${HOME}/.zshrc && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    git clone https://github.com/zsh-users/zsh-autosuggestions.git ${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    sed -i 's/plugins=(git)/plugins=(git zsh-syntax-highlighting zsh-autosuggestions)/g' ${HOME}/.zshrc
SHELL [ "/bin/bash", "-c" ]
#####################################################################
# FOR PYPI MIRRORS
RUN apt-get install -y python3-pip && \
    pip install -i https://mirrors.ustc.edu.cn/pypi/simple pip -U && \
    pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple
RUN ln -s /usr/bin/python3 /usr/bin/python
# ENV PATH=/usr/bin:/usr/local/bin:$PATH
#####################################################################
# FOR UV PIP
RUN apt-get install -y wget && \
    wget https://astral.sh/uv/install.sh && \
    bash install.sh && \
    rm install.sh && \
    mkdir -p ${HOME}/.config/uv && \
    touch ${HOME}/.config/uv/uv.toml && \
    echo -e "[[index]]\nurl = \"https://mirrors.ustc.edu.cn/pypi/simple\"\ndefault = true" > ${HOME}/.config/uv/uv.toml
# #####################################################################
# FOR CONDA
RUN apt-get install -y wget bzip2 && \
    mkdir $HOME/down && wget "https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh" -O $HOME/down/Miniforge3-Linux-x86_64.sh && \
    bash ${HOME}/down/Miniforge3-Linux-x86_64.sh -b -p ${HOME}/miniforge3 
RUN PATH=${HOME}/miniforge3/bin:$PATH && \
    conda init zsh bash && \
    mamba shell init --shell zsh bash && \
    touch $HOME/.condarc && echo -e "channels:\n  - nodefaults\ncustom_channels:\n  conda-forge: https://mirrors.ustc.edu.cn/anaconda/cloud\n  bioconda: https://mirrors.ustc.edu.cn/anaconda/cloud\nshow_channel_urls: true\nauto_activate: false" > $HOME/.condarc
# ENV PATH=/usr/bin:/usr/local/bin:$PATH
# RUN ln -s /usr/bin/python3 /usr/bin/python
#####################################################################

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

# RUN git clone https://github.com/humanoid-clone/CLONE.git /root/ws/CLONE

RUN mkdir -p /root/ws
COPY ./requirements.txt /root/ws
WORKDIR /root/ws
RUN pip3 install --upgrade pip
RUN pip3 install --ignore-installed -r requirements.txt
RUN pip3 install pyzmq
RUN rm -rf /root/ws/requirements.txt

WORKDIR /root/ws
RUN git clone https://github.com/Yutang-Lin/VisionWrapper.git
WORKDIR /root/ws/VisionWrapper
RUN pip3 install -e .
ENV PYTHONPATH="/root/ws:/root/ws/VisionWrapper:${PYTHONPATH}"

RUN touch ${HOME}/.selfshrc
RUN echo "# ROS2 Environment Setup" >> ${HOME}/.selfshrc && \
    echo "source /opt/ros/foxy/setup.sh" >> ${HOME}/.selfshrc && \
    echo "source /root/unitree_ros2/cyclonedds_ws/install/setup.sh" >> ${HOME}/.selfshrc && \
    echo "source /root/unitree_ros2/install/setup.sh" >> ${HOME}/.selfshrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ${HOME}/.selfshrc && \
    echo "export PYTHONPATH=/root/ws:/root/ws/VisionWrapper:\$PYTHONPATH" >> ${HOME}/.selfshrc && \
    echo "# Aliases for convenience" >> ${HOME}/.selfshrc && \
    echo "alias ll='ls -alF'" >> ${HOME}/.selfshrc && \
    echo "alias la='ls -A'" >> ${HOME}/.selfshrc && \
    echo "alias l='ls -CF'" >> ${HOME}/.selfshrc && \
    echo "echo 'ROS2 Foxy environment loaded'" >> ${HOME}/.selfshrc && \
    echo "echo 'Unitree ROS2 workspace loaded'" >> ${HOME}/.selfshrc && \
    echo "echo 'VisionWrapper available'" >> ${HOME}/.selfshrc
RUN echo "source ${HOME}/.selfshrc" >> ${HOME}/.zshrc && \
    echo "source ${HOME}/.selfshrc" >> ${HOME}/.bashrc

# RUN cat >/ros_entrypoint.sh <<'EOF'
# #!/usr/bin/env bash
# set -e
# source /opt/ros/foxy/setup.sh
# source /root/unitree_ros2/cyclonedds_ws/install/setup.sh
# source /root/unitree_ros2/install/setup.sh
# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# export PYTHONPATH=/root/ws:/root/ws/VisionWrapper:$PYTHONPATH
# exec "$@"
# EOF
# RUN chmod +x /ros_entrypoint.sh

WORKDIR /root/ws

# ENTRYPOINT /ros_entrypoint.sh
CMD ["/bin/bash"]