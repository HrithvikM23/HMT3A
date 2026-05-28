from __future__ import annotations

import json
import socket

import config as app_config

class OSCSender:
    def __init__(
        self,
        host: str = app_config.LiveUdpDefaults.HOST,
        port: int = app_config.LiveUdpDefaults.PORT,
        enabled: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.enabled = enabled
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if enabled else None

    def send_pose(self, body_points, hands_by_side, joints=None, metadata=None) -> None:
        if not self.enabled or self._socket is None:
            return

        payload = {
            "format": "kinara-live-v2",
            "metadata": metadata or {},
            "people": [
                self._build_person_payload(
                    person_id=1,
                    label="person1",
                    body_points=body_points,
                    hands_by_side=hands_by_side,
                    joints=joints,
                )
            ],
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._socket.sendto(encoded, (self.host, self.port))

    def send_people(self, people, metadata=None) -> None:
        if not self.enabled or self._socket is None:
            return

        payload = {
            "format": "kinara-live-v2",
            "metadata": metadata or {},
            "people": [
                self._build_person_payload(
                    person_id=person["id"],
                    label=person.get("label"),
                    body_points=person["body_points"],
                    hands_by_side=person["hands_by_side"],
                    box=person.get("box"),
                    joints=person.get("joints"),
                    score=person.get("score"),
                    camera_views=person.get("camera_views"),
                )
                for person in people
            ],
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._socket.sendto(encoded, (self.host, self.port))

    def _build_person_payload(self, person_id, label, body_points, hands_by_side, box=None, joints=None, score=None, camera_views=None):
        payload = {
            "id": int(person_id),
            "label": label,
            "box": None if box is None else [int(value) for value in box],
            "score": None if score is None else float(score),
            "body": [
                {
                    "index": index,
                    "x": int(x),
                    "y": int(y),
                    "confidence": float(conf),
                }
                for index, (x, y, conf) in enumerate(body_points)
            ],
            "hands": {
                side: {
                    "box": [int(value) for value in hand_payload["box"]],
                    "points": [
                        {
                            "index": index,
                            "x": int(x),
                            "y": int(y),
                            "confidence": float(conf),
                        }
                        for index, (x, y, conf) in enumerate(hand_payload["points"])
                    ],
                }
                for side, hand_payload in hands_by_side.items()
            },
        }
        if joints is not None:
            payload["joints"] = joints
        if camera_views is not None:
            payload["camera_views"] = list(camera_views)
        return payload
    
    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
