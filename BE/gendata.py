import random
from datetime import datetime, timedelta

# Các nhãn có thể có
labels = ["Healthy", "High Stress", "Moderate Stress"]

# Lấy timestamp cuối cùng từ file cũ, hoặc bắt đầu từ một ngày nào đó
start_time = datetime(2024, 11, 2, 5, 0, 0)

rows = []
for i in range(100):
    timestamp = start_time + timedelta(hours=6*i)
    plant_id = 21
    soil_moisture = round(random.uniform(10, 40), 6)
    ambient_temp = round(random.uniform(20, 35), 6)
    soil_temp = round(random.uniform(15, 30), 6)
    humidity = round(random.uniform(40, 70), 6)
    light_intensity = round(random.uniform(200, 900), 6)
    soil_ph = round(random.uniform(5.5, 7.5), 6)
    nitrogen = round(random.uniform(10, 50), 6)
    phosphorus = round(random.uniform(10, 50), 6)
    potassium = round(random.uniform(10, 50), 6)
    chlorophyll = round(random.uniform(10, 50), 6)
    electro_signal = round(random.uniform(0.05, 2.0), 6)
    label = random.choice(labels)
    row = f"{timestamp},{plant_id},{soil_moisture},{ambient_temp},{soil_temp},{humidity},{light_intensity},{soil_ph},{nitrogen},{phosphorus},{potassium},{chlorophyll},{electro_signal},{label}"
    rows.append(row)

# In ra màn hình để copy vào file CSV
for row in rows:
    print(row)