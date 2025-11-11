# Real-World Deployment
**If you encountered errors caused by our code when following steps below, please inform us in issues and we'll try fix it asap.**

checkout [Docker_README.md](Docker_README.md) for quick & simpler installation with Docker.

---

## 0. Hardware Setup

| **Component**            | **Specification**         |
|--------------------------|---------------------------|
| Unitree G1 EDU           | 29-DoF Version            |
| Apple Vision Pro         |                           |
| Router                   |                           |
| Personal Computer        | Linux OS (Monitor Recommended) |

**Connection Setup:**  
1. Connect the Unitree G1 and the server PC to the router via Ethernet.  
2. Connect the Apple Vision Pro to the same router via Wi-Fi.

---

## 1. Prepare Server Environment
### Install ROS 2
Follow the official installation guide at [ros.org](https://www.ros.org/) for your Linux distribution.

### Install Unitree ROS 2 SDK
Follow the instructions at:  
[github.com/unitreerobotics/unitree_ros2](https://github.com/unitreerobotics/unitree_ros2)

### Clone Repository & Install Dependencies
```bash
# Clone repository
git clone https://github.com/humanoid-clone/CLONE.git

# Install dependencies
cd CLONE
pip install -r requirements.txt
```

---

## 2. Configure Unitree G1
### Install LiDAR Odometry on G1's PC2
**Connect G1 to Internet:**  
```bash
ssh unitree@192.168.123.164
bash nomachine.sh
```
Connect via NoMachine on your PC and configure Wi-Fi. Select **ROS1 (Noetic)** if prompted.

**Install LiDAR Odometry:**  
1. Install LIVOX drivers (both `driver` and `driver2`) for ROS1 following:  
   [FAST_LIO](https://github.com/hku-mars/FAST_LIO)  
   *Note: Skip FAST LIO installation if using odometry without point cloud maps.*  
2. Install localization package:  
   [FAST_LIO_LOCALIZATION](https://github.com/HViktorTsoi/FAST_LIO_LOCALIZATION)

### Deploy Onboard Files
Copy contents of `onboard` to G1's PC2. Edit `localization_server.sh`'s first line to:  
```bash
cd <PATH_TO_YOUR_FAST_LIO_LOCALIZATION_FOLDER>
```
### FAQ
If you encountered **missing files** such as `localization_mid360.launch` or `mid360.yaml`, check `onboard/launch` for onboard **ROS launch files** and `onboard/misc` for `mid360.yaml`.

---

## 3. Install VisionWrapper
This package integrates [avp_stream](https://github.com/Improbable-AI/VisionProTeleop) (recommended) and [VUER](https://github.com/unitreerobotics/avp_teleoperate):
```bash
# In Server PC's any directory
git clone https://github.com/Yutang-Lin/VisionWrapper
cd VisionWrapper
pip install -e .
```

---

## Teleoperate Unitree G1
### Prerequisites
- Verify Apple Vision Pro's IP address in deployment scripts
- Start Unitree G1 in Debug Mode ([Unitree Documentation](https://support.unitree.com/home/zh/G1_developer/remote_control))

### Launch Services
1. **On G1 PC2:** Start LiDAR odometry  
   ```bash
   cd <PATH_TO_ONBOARD_FILES>
   bash localization_server.sh
   ```
2. **On Apple Vision Pro:** (for `avp_stream` users) Launch Tracking Streamer 
3. **On Server PC:** Run command publisher for 1kHz relay.
   ```bash
   python lowcmd_publisher.py
   ```
4. **On Server PC:** Run deployment script  
   ```bash
   python g1_server.py
   ```
5. **On Apple Vision Pro:** (for For `VUER` Users) Access control interface at  
   `https://<SERVER_IP>:8012?ws=wss://<SERVER_IP>:8012` 

### Calibration & Operation
1. Align human and humanoid positions by pressing **R1/R2** (press multiple times for best results)  
2. Initiate policy execution by pressing **L1**
