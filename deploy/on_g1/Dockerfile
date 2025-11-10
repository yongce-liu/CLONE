FROM ros:noetic-ros-base-focal

#####################################################################
# # FOR UBUNTU MIRRORS AMD
# RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list && \
#     sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
#     apt-get update && apt-get upgrade -y
# #  FOR UBUNTU MIRRORS ARM
# RUN sed -i -e 's@//ports.ubuntu.com/\? @//ports.ubuntu.com/ubuntu-ports @g' \
#             -e 's@//ports.ubuntu.com@//mirrors.ustc.edu.cn@g' \
#             /etc/apt/sources.list && \
#     apt-get update && apt-get upgrade -y
# #####################################################################
# # FOR ROS 1 MIRRORS
# RUN apt-get update && apt-get install -y gpg lsb-release curl && \
#     gpg --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
#     gpg --export C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 | tee /usr/share/keyrings/ros.gpg > /dev/null && \
#     sh -c 'echo "deb [signed-by=/usr/share/keyrings/ros.gpg] https://mirrors.ustc.edu.cn/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# # FOR ROS 2 MIRRORS
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
#     echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://mirrors.ustc.edu.cn/ros2/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
#     apt-get update
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
# # FOR CONDA
# RUN apt-get install -y wget bzip2 && \
#     mkdir $HOME/down && wget "https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh" -O $HOME/down/Miniforge3-Linux-x86_64.sh && \
#     bash ${HOME}/down/Miniforge3-Linux-x86_64.sh -b -p ${HOME}/miniforge3 
# RUN PATH=${HOME}/miniforge3/bin:$PATH && \
#     conda init zsh bash && \
#     mamba shell init --shell zsh bash && \
#     touch $HOME/.condarc && echo -e "channels:\n  - nodefaults\ncustom_channels:\n  conda-forge: https://mirrors.ustc.edu.cn/anaconda/cloud\n  bioconda: https://mirrors.ustc.edu.cn/anaconda/cloud\nshow_channel_urls: true\nauto_activate: false" > $HOME/.condarc
# # ENV PATH=/usr/bin:/usr/local/bin:$PATH
# # RUN ln -s /usr/bin/python3 /usr/bin/python
#####################################################################

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3-catkin-tools \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    wget \
    curl \
    vim \
    ros-noetic-pcl-ros \
    ros-noetic-pcl-conversions \
    ros-noetic-tf \
    ros-noetic-tf2 \
    ros-noetic-tf2-ros \
    ros-noetic-rviz \
    ros-noetic-ros-numpy \
    ros-noetic-eigen-conversions \
    libpcl-dev \
    libeigen3-dev \
    python3-pip \
    python3-dev \
    python3-zmq \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

RUN cd /catkin_ws/src && \
    git clone https://github.com/Livox-SDK/Livox-SDK2.git && \
    cd Livox-SDK2 && \
    mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install

RUN cd /catkin_ws/src && \
    git clone https://github.com/Livox-SDK/livox_ros_driver2.git

RUN cd /catkin_ws/src && \
    git clone https://github.com/Livox-SDK/livox_ros_driver.git

RUN rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd /catkin_ws/src/livox_ros_driver2 && \
    ./build.sh ROS1"

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd /catkin_ws && \
    catkin_make -DCATKIN_WHITELIST_PACKAGES=livox_ros_driver"


RUN pip3 install --upgrade pip && \
    pip3 install numpy==1.23.5 && \
    pip3 install open3d==0.14.1 pyrsistent==0.18.0 opencv-python

RUN cd /catkin_ws/src && \
    git clone https://github.com/HViktorTsoi/FAST_LIO_LOCALIZATION.git && \
    cd FAST_LIO_LOCALIZATION && \
    git submodule update --init --recursive && \
    sed -i 's|<fast_lio/Pose6D.h>|<fast_lio_localization/Pose6D.h>|g' include/common_lib.h && \
    sed -i 's|typedef fast_lio::Pose6D Pose6D;|typedef fast_lio_localization::Pose6D Pose6D;|g' include/common_lib.h && \
    find scripts -name "*.py" -exec sed -i '1s|#!/usr/bin/env python2|#!/usr/bin/env python3|' {} \; && \
    find scripts -name "*.py" -exec sed -i '1s|#!/usr/bin/env python$|#!/usr/bin/env python3|' {} \; && \
    find scripts -name "*.py" -exec sed -i 's|import thread|import _thread as thread|g' {} \; && \
    find scripts -name "*.py" -exec sed -i 's|from Queue import|from queue import|g' {} \; && \
    find scripts -name "*.py" -exec sed -i 's|import Queue|import queue as Queue|g' {} \;

# COPY onboard /catkin_ws/deploy_onboard

# RUN sed -i '3s|cd nav/rosws/fastlio_localization|cd /catkin_ws|' /catkin_ws/deploy_onboard/localization_server.sh && \
#     sed -i 's|cd \.\.|cd /catkin_ws|g' /catkin_ws/deploy_onboard/localization_server.sh && \
#     sed -i 's|cd livox_ros_driver2|# Already in catkin_ws|' /catkin_ws/deploy_onboard/localization_server.sh && \
#     sed -i 's|cd ~/teleoperation|cd /catkin_ws/deploy_onboard|' /catkin_ws/deploy_onboard/localization_server.sh && \
#     chmod +x /catkin_ws/deploy_onboard/localization_server.sh

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    source /catkin_ws/devel/setup.bash && \
    cd /catkin_ws && \
    catkin_make -DCATKIN_WHITELIST_PACKAGES=''"

# RUN mkdir -p /catkin_ws/src/FAST_LIO_LOCALIZATION/launch && \
    # mkdir -p /catkin_ws/src/FAST_LIO_LOCALIZATION/config && \
# RUN mkdir -p /catkin_ws/src/FAST_LIO_LOCALIZATION/scripts && \
    # cp /catkin_ws/deploy_onboard/launch/*.launch /catkin_ws/src/FAST_LIO_LOCALIZATION/launch/ && \
    # sed -i 's|<arg name="rviz" default="true"|<arg name="rviz" default="false"|g' /catkin_ws/src/FAST_LIO_LOCALIZATION/launch/localization_mid360.launch && \
    # cp /catkin_ws/deploy_onboard/misc/mid360.yaml /catkin_ws/src/FAST_LIO_LOCALIZATION/config/ && \
    # sed -i 's/min_x = points\[:, 0\]\.min()/min_x = int(points[:, 0].min()) if len(points) > 0 else 0/' /catkin_ws/deploy_onboard/read_pcd_publish.py && \
    # sed -i 's/max_x = points\[:, 0\]\.max()/max_x = int(points[:, 0].max()) if len(points) > 0 else 1/' /catkin_ws/deploy_onboard/read_pcd_publish.py && \
    # sed -i 's/min_y = points\[:, 1\]\.min()/min_y = int(points[:, 1].min()) if len(points) > 0 else 0/' /catkin_ws/deploy_onboard/read_pcd_publish.py && \
    # sed -i 's/max_y = points\[:, 1\]\.max()/max_y = int(points[:, 1].max()) if len(points) > 0 else 1/' /catkin_ws/deploy_onboard/read_pcd_publish.py && \
    # sed -i 's/width = max_x - min_x + 5/width = int(max_x - min_x + 5)/' /catkin_ws/deploy_onboard/read_pcd_publish.py && \
    # sed -i 's/height = max_y - min_y + 5/height = int(max_y - min_y + 5)/' /catkin_ws/deploy_onboard/read_pcd_publish.py && \
    # sed -i 's/img\[p\[1\] - min_y, p\[0\] - min_x\]/img[int(p[1]) - min_y, int(p[0]) - min_x]/' /catkin_ws/deploy_onboard/read_pcd_publish.py && \
    # sed -i '139a\    if len(points) == 0:\n        points = np.array([[0, 0, 0]], dtype=np.int32)' /catkin_ws/deploy_onboard/read_pcd_publish.py && \
    # cp /catkin_ws/deploy_onboard/read_pcd_publish.py /catkin_ws/src/FAST_LIO_LOCALIZATION/scripts/ && \
    # mkdir -p /catkin_ws/src/livox_ros_driver2/config && \
    # echo '{"lidar_summary_info": {"lidar_type": 8}, "MID360": {"lidar_net_info": {"cmd_data_port": 56100, "push_msg_port": 56200, "point_data_port": 56300, "imu_data_port": 56400, "log_data_port": 56500}, "host_net_info": {"cmd_data_ip": "192.168.123.164", "cmd_data_port": 56101, "push_msg_ip": "192.168.123.164", "push_msg_port": 56201, "point_data_ip": "192.168.123.164", "point_data_port": 56301, "imu_data_ip": "192.168.123.164", "imu_data_port": 56401, "log_data_ip": "192.168.123.164", "log_data_port": 56501}}}' > /catkin_ws/src/livox_ros_driver2/config/MID360_config.json && \
    # sed -i 's|python |python3 |g' /catkin_ws/deploy_onboard/localization_server.sh && \
    # sed -i 's|python |python3 |g' /catkin_ws/deploy_onboard/pos_server.py && \
    # find /catkin_ws/src/FAST_LIO_LOCALIZATION/scripts -name "*.py" -exec sed -i '1s|#!/usr/bin/env python2|#!/usr/bin/env python3|' {} \; && \
    # find /catkin_ws/src/FAST_LIO_LOCALIZATION/scripts -name "*.py" -exec sed -i '1s|#!/usr/bin/env python$|#!/usr/bin/env python3|' {} \; && \
    # chmod +x /catkin_ws/deploy_onboard/*.py && \
    # chmod +x /catkin_ws/src/FAST_LIO_LOCALIZATION/scripts/*.py

RUN cd /catkin_ws/src/FAST_LIO_LOCALIZATION && \
    sed -i 's|<name>.*</name>|<name>fast_lio_localization</name>|g' package.xml && \
    cd /catkin_ws && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make clean && catkin_make"

# RUN mkdir -p /home/unitree/teleoperation/nav/rosws/FAST_LIO_WS/src/FAST_LIO/PCD && \
#     printf "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 3\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 3\nDATA ascii\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n" > /home/unitree/teleoperation/nav/rosws/FAST_LIO_WS/src/FAST_LIO/PCD/scans.pcd

# RUN echo "#!/bin/bash" > /ros_entrypoint.sh && \
#     echo "source /opt/ros/noetic/setup.bash" >> /ros_entrypoint.sh && \
#     echo "source /catkin_ws/devel/setup.bash" >> /ros_entrypoint.sh && \
#     echo 'exec "$@"' >> /ros_entrypoint.sh && \
#     chmod +x /ros_entrypoint.sh

# ENTRYPOINT ["/ros_entrypoint.sh"]

RUN touch ${HOME}/.selfshrc
RUN echo "# ROS2 Environment Setup" >> ${HOME}/.selfshrc && \
    echo "source /opt/ros/noetic/setup.sh" >> ${HOME}/.selfshrc && \
    echo "source /catkin_ws/devel/setup.sh" >> ${HOME}/.selfshrc
RUN echo "source ${HOME}/.selfshrc" >> ${HOME}/.zshrc && \
    echo "source ${HOME}/.selfshrc" >> ${HOME}/.bashrc
    
WORKDIR /catkin_ws
CMD ["/bin/bash"]