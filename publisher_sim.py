# publisher_sim.py
# Publishes a test image to MQTT for simulation
# simulates an ESP32-CAM publishing an image

import paho.mqtt.client as mqtt
import base64

BROKER = "localhost"
TOPIC = "ppe/camera/esp32cam/frame"
IMAGE_PATH = r"C:\Users\shaun\OneDrive\Desktop\ppe-detector\images\sim1.png"  # absolute path for clarity

client = mqtt.Client()
client.connect(BROKER, 1883, 60)

# Read and encode image
with open(IMAGE_PATH, "rb") as img_file:
    img_bytes = img_file.read()
    img_b64 = base64.b64encode(img_bytes)

# Publish to topic
result = client.publish(TOPIC, img_b64)
print(f"Published {IMAGE_PATH} to {TOPIC}, result={result.rc}")

# Let the MQTT loop run briefly to flush the message
client.loop(1)   # process network events for 1 second

client.disconnect()