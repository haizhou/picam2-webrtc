import asyncio
import json
import time
import uuid
from aiohttp import web
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from picamera2 import Picamera2
import numpy as np
import av
import cv2

RES = (640, 480)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": RES, "format": "BGR888"}))
picam2.start()

class VideoStream(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.color_mode = 1

    def next_timestamp(self):
        now = time.time()
        return int(now * 90000), "1/90000"

    async def recv(self):
        pts, time_base = self.next_timestamp()

        frame = picam2.capture_array()

        if self.color_mode == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif self.color_mode == 3:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            lower_blue = np.array([170 * 180 // 360, 30, 30])
            upper_blue = np.array([260 * 180 // 360, 255, 255])

            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            mask = cv2.medianBlur(mask, 15)
            mask = np.float32(mask) / 255
            mask3 = cv2.merge([mask, mask, mask])

            hsv[:, :, 0] = (hsv[:, :, 0] + 158 * 180 // 360) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 - 50 / 100), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + 50 / 100), 0, 255)
            pink = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            frame = np.uint8(
                cv2.multiply(frame.astype(np.float32), 1 - mask3)
                + cv2.multiply(pink.astype(np.float32), mask3)
            )

        frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base

        return frame

    def switch_color_mode(self, color_mode):
        self.color_mode = color_mode
        return self.color_mode

peer_connections = {}

async def offer(request):
    params = await request.json()

    stun_servers = [
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
        RTCIceServer(urls="stun:stun1.l.google.com:19302")
    ]

    rtc_config = RTCConfiguration(iceServers=stun_servers)
    pc = RTCPeerConnection(configuration=rtc_config)

    stream = VideoStream()
    pc.addTrack(stream)

    session_id = str(uuid.uuid4())
    peer_connections[session_id] = (pc, stream)

    @pc.on("iceconnectionstatechange")
    async def on_ice_state_change():
        if pc.iceConnectionState in ("failed", "closed"):
            await pc.close()
            peer_connections.pop(session_id, None)

    await pc.setRemoteDescription(RTCSessionDescription(params["sdp"], params["type"]))

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_id": session_id
        }
    )

async def switch(request):
    params = await request.json()
    session_id = params.get("session_id")
    color_mode = int(params.get("color_mode"))

    if session_id in peer_connections:
        _, stream = peer_connections[session_id]
        new_mode = stream.switch_color_mode(color_mode)
        return web.json_response({"color_mode": new_mode})

    return web.json_response({"message": "Session not found"}, status=400)

app = web.Application()
app.router.add_post("/offer", offer)
app.router.add_post("/switch", switch)
app.router.add_get("/", lambda request: web.FileResponse("index.html"))

web.run_app(app, port=8000)
