# -------- sender.py --------

import socket
import json

UDP_IP = "127.0.0.1"
UDP_PORT = 5052

class LandmarkSender:
    def __init__(self, ip=UDP_IP, port=UDP_PORT):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, landmarks_dict):
        """Send landmark data to Unity."""
        try:
            data = json.dumps(landmarks_dict)
            self.sock.sendto(data.encode(), (self.ip, self.port))
        except Exception as e:
            print(f"Sender error: {e}")
