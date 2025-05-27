import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ======== Thông tin file CSV ========
csv_path = "your_data.csv"  # ⚠️ Đổi thành đường dẫn tới file CSV của bạn
target_column = "Plant_Health_Status"  # ⚠️ Đổi thành tên cột mục tiêu thực tế

# ======== Đọc dữ liệu ========
df = pd.read_csv(csv_path)

# ======== Kiểm tra dữ liệu đầu vào ========
print("📊 Các cột:", df.columns.tolist())
print("🔍 Số dòng:", len(df))

# ======== Tách đặc trưng & nhãn ========
drop_cols = [target_column, "Timestamp", "Plant_ID"]  # Loại bỏ các cột không số
X = df.drop(columns=drop_cols)
y = df[target_column]

# Nếu nhãn là chuỗi (tên cây trồng), cần encode
if y.dtype == 'object':
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    joblib.dump(encoder, 'model/label_encoder.pkl')  # Lưu encoder nếu muốn giải mã lại kết quả

# ======== Chia tập train/test và huấn luyện ========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ======== Lưu model ========
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/random_forest_model.pkl")

print("✅ Mô hình đã được huấn luyện và lưu tại model/random_forest_model.pkl")
