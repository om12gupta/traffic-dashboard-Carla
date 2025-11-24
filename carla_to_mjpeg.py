
# CARLA -> MJPEG bridge. Run this separately, then read http://127.0.0.1:5000/video from OpenCV.
import time
import numpy as np
import carla
import cv2
from flask import Flask, Response

app = Flask(__name__)
latest_jpeg = b''

def setup_carla(host="127.0.0.1", port=2000):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    bp = world.get_blueprint_library()
    vbp = bp.find("vehicle.tesla.model3")
    spawn = world.get_map().get_spawn_points()[0]
    vehicle = world.try_spawn_actor(vbp, spawn) or world.spawn_actor(vbp, spawn)
    vehicle.set_autopilot(True)

    cbp = bp.find("sensor.camera.rgb")
    cbp.set_attribute("image_size_x", "1280")
    cbp.set_attribute("image_size_y", "720")
    cbp.set_attribute("fov", "90")
    cam_tf = carla.Transform(carla.Location(x=1.6, z=2.2))
    camera = world.spawn_actor(cbp, cam_tf, attach_to=vehicle)

    def on_img(image):
        global latest_jpeg
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
        ok, jpg = cv2.imencode(".jpg", arr)
        if ok:
            latest_jpeg = jpg.tobytes()

    camera.listen(on_img)
    return vehicle, camera

@app.route("/video")
def video():
    def gen():
        boundary = b'--frame\r\n'
        while True:
            if latest_jpeg:
                yield boundary
                yield b'Content-Type: image/jpeg\r\n\r\n' + latest_jpeg + b'\r\n'
            time.sleep(0.01)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    vehicle, camera = setup_carla()
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        camera.stop(); camera.destroy(); vehicle.destroy()
