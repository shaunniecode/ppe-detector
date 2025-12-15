# subscriber.py
# MQTT subscriber that receives images, calls run_inference, and prints detections

import paho.mqtt.client as mqtt
import base64
import cv2
import numpy as np
from run_inference_stub import run_inference

BROKER = "localhost"          # change if using remote broker
TOPIC = "ppe/camera/esp32cam/frame"

def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    print("Message received on", msg.topic)
    print("Raw payload type:", type(msg.payload))
    print("Payload length:", len(msg.payload))

    try:
        img_data = base64.b64decode(msg.payload)
        print("Decoded base64 length:", len(img_data))
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            print("cv2.imdecode returned None")
        else:
            detections = run_inference(img)
            for label, conf in detections:
                print(f"Detections: {label} ({conf*100:.1f}%)")
    except Exception as e:
        print("Error in on_message:", e)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("Starting subscriber...")
client.connect(BROKER, 1883, 60)
client.loop_forever()