
# Live CARLA Video -> Streamlit Dashboard

Two ways to get **live CARLA** into the dashboard:

## A) Direct (inside Streamlit)
Run:
```
pip install -r requirements_carla.txt
streamlit run app_carla.py
```
In the sidebar, choose **Source mode → CARLA**. Make sure CARLA server is already running.

## B) Via MJPEG bridge (HTTP URL)
Run CARLA and then:
```
python carla_to_mjpeg.py
```
You will get a stream at `http://127.0.0.1:5000/video`.
In your main dashboard, set the source to that URL (OpenCV can read it).

## Notes
- Match your `carla` Python wheel/egg to the simulator version.
- For stable ID-based counts, switch to YOLO `model.track(..., tracker="bytetrack.yaml")`.
- If you see reloads spawning extra actors in CARLA, stop the app cleanly or wrap actor creation in guards.
