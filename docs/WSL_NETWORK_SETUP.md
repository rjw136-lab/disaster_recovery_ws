# [공학 가이드 1] WSL2 미러드 네트워킹 및 CUDA 가속 관제 환경 구축 가이드

---

## 1. 개요 및 학술적/엔지니어링 배경

재난 탐사 로봇 시스템에서 원격 관제 PC와 온보드 임베디드 컴퓨터 간의 저지연 통신은 매우 중요하다. 본 시스템에서는 학교 전산망 보안 정책(고정 IP 및 MAC 주소 인증)을 보존하기 위해 Host PC의 마더보드 유선 LAN은 학교 벽면 포트에 직결하고, 별도의 USB 무선 LAN(Wi-Fi) 동글을 연구실 전용 사설 공유기(192.168.0.x 대역)에 연결하는 Dual NIC 구성을 채택한다.

기존 WSL2의 기본 NAT(Network Address Translation) 가상 스위치는 자체 내부 서브넷(예: 172.x.x.x)을 생성하므로 외부 기기(Raspberry Pi 4)와의 직접적인 UDP 멀티캐스트 통신이 차단되는 구조적 한계가 존재한다. ROS 2의 DDS(Data Distribution Service)는 로컬 서브넷 내 UDP 멀티캐스트를 기반으로 노드를 검색(Discovery)하므로, NAT 모드에서는 포트 포워딩이나 브릿지(Bridged) 네트워크의 복잡한 설정이 강제된다.

이를 해결하기 위해 Windows 11 22H2부터 도입된 **미러드 네트워킹(Mirrored Mode Networking)**을 적용한다 (Microsoft Documentation, 2023). 미러드 모드는 Windows 호스트의 모든 네트워크 인터페이스와 IP 주소를 WSL2 Linux 커널에 1:1로 가상 미러링하여, 호스트의 Wi-Fi IP(192.168.0.x)를 리눅스 터미널에서 그대로 공유하며 완벽한 로컬 멀티캐스트 패킷 송수신을 보장한다.

---

## 2. Windows 11 `.wslconfig` 미러드 네트워킹 설정

### 2.1 파일 생성 및 위치
Windows 호스트의 사용자 홈 디렉터리에 `.wslconfig` 파일을 생성한다.
- 경로: `C:\Users\<사용자계정>\.wslconfig` (예: `C:\Users\rjw13\.wslconfig`)

### 2.2 설정 파일 내용
아래 내용을 메모장 또는 에디터로 입력하여 저장한다:

```ini
[wsl2]
# 미러드 네트워킹 활성화: 윈도우 물리 NIC(연구실 사설 Wi-Fi)의 IP와 인터페이스를 리눅스에 1:1 복제
networkingMode=mirrored

# DNS 터널링 활성화: 윈도우 DNS 설정을 WSL2가 그대로 활용하여 패킷 유실 방지
dnsTunneling=true

# Windows 호스트 방화벽 규칙을 WSL2에 통합 적용
firewall=true

# 로컬호스트 루프백 자동 프록시
autoProxy=true

# CPU 코어 및 메모리 할당 (Intel i7-8700 6코어/12스레드, 16GB RAM 기준 권장값)
processors=8
memory=10GB
swap=4GB
```

### 2.3 WSL2 재시작
Windows PowerShell(관리자 권한)에서 WSL 인스턴스를 완전히 종료하고 재시작한다:
```powershell
wsl --shutdown
wsl
```

WSL2 터미널에서 `ip addr`을 실행하여 연구실 사설 공유기(192.168.0.x) 무선 랜카드의 IP가 리눅스 인터페이스에 그대로 나타나는지 검증한다:
```bash
ip addr show
```

---

## 3. WSL2 Ubuntu 22.04 LTS 및 ROS 2 Humble 설치

### 3.1 시스템 업데이트 및 필수 유틸리티
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl gnupg2 lsb-release build-essential git wget python3-pip
```

### 3.2 ROS 2 Humble 설치
```bash
# ROS 2 apt 리포지토리 GPG 키 등록
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 패키지 인덱스 갱신 및 Humble Desktop 설치
sudo apt update
sudo apt install -y ros-humble-desktop ros-dev-tools python3-colcon-common-extensions

# 환경 변수 등록
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 4. NVIDIA CUDA 가속 및 WSLg GUI 가속 설정 (GTX 1050 Ti)

### 4.1 하드웨어 및 드라이버 원칙
- **주의**: WSL2 내부에는 Linux용 NVIDIA 디스플레이 드라이버(`nvidia-driver-xxx`)를 설치해서는 절대 안 된다. Windows 호스트에 설치된 공식 Game Ready / Studio 드라이버가 Direct3D 12 및 WSL-CUDA 커널 모듈을 통해 WSL2 내부로 자동 패스스루된다.
- Windows 11의 WSLg(Windows Subsystem for Linux GUI)를 통해 X11 및 Wayland 클라이언트(RViz2, Gazebo)가 하드웨어 3D 가속을 받아 60 FPS로 구동된다.

### 4.2 CUDA Toolkit 및 PyTorch 설치 (WSL2 내부)
```bash
# NVIDIA CUDA Repository 핀 설정 (Ubuntu 22.04 전용)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-2

# 환경변수 등록
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# GPU 가속 검증
nvidia-smi
```

### 4.3 AI 프레임워크 (PyTorch, Ultralytics YOLO, Faster-Whisper) 설치
```bash
# PyTorch with CUDA 12.1+
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Ultralytics YOLO 및 Faster-Whisper, OpenCV
pip3 install ultralytics faster-whisper

# ROS 2 Humble cv_bridge 는 NumPy 1.x 로 빌드되어 있으므로 반드시 버전을 고정한다
# (NumPy 2.x 설치 시 cv_bridge import 에러 발생)
pip3 install "numpy==1.26.4" "opencv-python==4.10.0.84"
```

### 4.4 CUDA 가속 무결성 테스트
```bash
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0))"
```
출력 결과:
```
CUDA Available: True
Device Name: NVIDIA GeForce GTX 1050 Ti
```

---

## 5. ROS 2 DDS 초저지연 통신 환경 설정

Host PC와 Raspberry Pi 4 간 1~2ms 수준의 초저지연 토픽 통신을 보장하기 위해 환경 변수를 통일한다.

### 5.1 Host PC 및 Raspberry Pi 4 공통 `.bashrc` 설정
양측 기기의 `~/.bashrc` 최하단에 아래 내용을 동일하게 입력한다:

```bash
# DDS 도메인 통일 (TurtleBot3 기본값 0번)
export ROS_DOMAIN_ID=0

# 로컬호스트 제한 해제 (외부 RPi4 통신 허용)
export ROS_LOCALHOST_ONLY=0

# TurtleBot3 버거 모델 지정
export TURTLEBOT3_MODEL=burger

# RMW 구현체 설정 (FastDDS 또는 CycloneDDS)
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### 5.2 DDS 멀티캐스트 연결 검증
1. Host PC WSL2 터미널:
   ```bash
   ros2 run demo_nodes_cpp talker
   ```
2. Raspberry Pi 4 터미널:
   ```bash
   ros2 run demo_nodes_cpp listener
   ```
3. 왕복 지연 시간(Ping) 검증:
   ```bash
   ping 192.168.0.xxx  # RPi 4의 무선 IP
   ```
   평균 지연 시간이 1.0~2.5ms 내외로 유지됨을 확인한다.

---

## 6. (선택) 원격 접속: Tailscale + CycloneDDS 유니캐스트

연구실 공유기에 직접 붙을 수 없는 경우 Tailscale VPN 위에서 ROS 2를 운용할 수 있다. Tailscale 인터페이스(`tailscale0`)는 멀티캐스트를 지원하지 않으므로 CycloneDDS의 **유니캐스트 피어 디스커버리**를 사용한다.

`~/cyclonedds.xml` (Host PC 예시, Peer 주소는 RPi 4의 Tailscale IP):

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
    <Domain id="any">
        <General>
            <Interfaces>
                <NetworkInterface name="tailscale0" priority="default" multicast="false" />
            </Interfaces>
        </General>
        <Discovery>
            <Peers>
                <Peer address="100.x.y.z"/>   <!-- RPi 4 Tailscale IP -->
                <Peer address="127.0.0.1"/>
            </Peers>
            <ParticipantIndex>auto</ParticipantIndex>
        </Discovery>
    </Domain>
</CycloneDDS>
```

```bash
sudo apt install -y ros-humble-rmw-cyclonedds-cpp
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$HOME/cyclonedds.xml
```

RPi 4에도 동일한 파일을 두되 Peer 주소만 Host PC의 Tailscale IP로 바꾼다. 양쪽 RMW 구현체가 반드시 같아야 한다.

| 비교 항목 | Tailscale VPN | 사설 Wi-Fi 직결 (기본 권장) |
| :--- | :--- | :--- |
| 경로 | 공용망 경유 (직결 실패 시 DERP 릴레이) | 로컬 공유기 1:1 (LAN) |
| 지연 | 수십~150 ms, 지터 큼 | 1~3 ms |
| 암호화 | WireGuard 강제 (RPi CPU 부하) | 없음 (순수 UDP) |
| 워치독 | 패킷 지연 시 3초 타임아웃 오작동 가능 | 10 Hz Heartbeat 안정 |

> 현장 시험에서는 **사설 Wi-Fi 직결 + Fast DDS(기본값)** 를 권장한다. Tailscale 사용 후에는 `.bashrc`의 `RMW_IMPLEMENTATION`/`CYCLONEDDS_URI` 를 주석 처리해 원래 상태로 되돌린다.
