import requests

ROBOT_URL = "http://192.168.0.20:8080/run_task"

task = "Move to the drawer, open the yellow drawer, take out the medicine bottle, place it into the basket, and return to the starting point."

res = requests.post(
    ROBOT_URL,
    json={"task": task}
)

print("Response:", res.json())