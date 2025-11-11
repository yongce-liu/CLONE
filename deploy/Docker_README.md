### Quick installation with Docker

---

## 0. Hardware Setup

| **Component**            | **Specification**         |
|--------------------------|---------------------------|
| Unitree G1 EDU           | 29-DoF Version            |
| Apple Vision Pro         |                           |
| Router                   |                              |
| Personal Computer        | Linux OS (Monitor Recommended) |

**Connection Setup:**  
1. Connect the Unitree G1 and the server PC to the router via Ethernet.  
2. Connect the Apple Vision Pro to the same router via Wi-Fi.
3. Have Docker V2 and Docker compose V2 installed on both the server PC and G1 PC.

if you don't have Docker V2 and Docker compose V2 installed (check on both the server PC and G1 PC): 

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

---

## 1. Set up G1 PC
Copy the `on_g1` folder to the G1 PC and run the following commands (Note that building the docker image may take up to 10 minutes):
```bash
cd on_g1
docker compose build
docker compose up -d
```

Now enter the container to start the localization server:
```bash
docker exec -it on-g1 bash
cd deploy_onboard
bash localization_server.sh
```
Keep the terminal running and now set up the server PC.

---

## 2. Set up the Server PC

On the server PC, run (Note that building the docker image may take up to 10 minutes):
```bash
xhost +local:docker
cd deploy
docker compose build
docker compose up -d
```

Now enter the container and start the publisher (G1 and the localization server on G1 should be running):
```bash
docker exec -it clone_unitree_server bash
python3 lowcmd_publisher.py
```

Download the app streamer from the [Apple Developer Website](https://developer.apple.com/streaming/app-streaming/) and run it on the Apple Vision Pro.


Keep the previous terminal running and now start the policy server from another terminal on the server PC. Make sure to set the correct AVP IP address (IPv4 address) in the `config.py` file before running the policy server. You can find the IP address of the Apple Vision Pro before clicking the 'Start' button on the App. The IP address can also be found in the AVP WiFi settings.

```bash
docker exec -it clone_unitree_server bash
python3 g1_server.py
```

You should be good to go now. :-) enjoy!


---