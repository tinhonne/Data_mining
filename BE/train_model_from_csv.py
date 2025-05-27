import pandas as pd
import joblib
import os
from river import forest
from river import preprocessing
import pickle

# Đường dẫn file model
model_path = "model/river_random_forest.pkl"

# Đường dẫn file lưu số dòng đã train trước đó
last_row_path = "model/last_trained_row.txt"

# Đường dẫn file lưu bản đồ nhãn
label_map_path = "model/label_map.pkl"

# Đọc dữ liệu mới
df = pd.read_csv("your_data.csv")

# Xác định cột nhãn và các cột loại bỏ
target_column = "Plant_Health_Status"
drop_cols = [target_column, "Timestamp", "Plant_ID"]

# Chuẩn bị dữ liệu
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df[target_column] if target_column in df.columns else None

if y is None:
    print(f"File CSV không có cột nhãn '{target_column}'. Không thể train.")
    exit()

# Nếu nhãn là chuỗi, encode sang số
if y.dtype == 'object':
    y = y.astype("category").cat.codes

# Đọc số dòng đã train trước đó
if os.path.exists(last_row_path):
    with open(last_row_path, "r") as f:
        last_trained_row = int(f.read().strip())
else:
    last_trained_row = 0

# Kiểm tra nếu số dòng đã train > số dòng hiện tại, reset về 0
if last_trained_row > len(X):
    print("Số dòng đã train lớn hơn số dòng hiện tại. Đặt lại về 0 để train lại toàn bộ.")
    last_trained_row = 0

# Chỉ lấy phần data mới
X_new = X.iloc[last_trained_row:]
y_new = y.iloc[last_trained_row:]

if len(X_new) == 0:
    print("Không có dữ liệu mới để train.")
    exit()

# Nếu file model đã tồn tại, in thông tin trước khi train
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("--- Thông tin mô hình trước khi train ---")
    if hasattr(model, 'steps') and isinstance(model.steps, list) and len(model.steps) > 0:
        last_step = model.steps[len(model.steps)-1][1]
        if hasattr(last_step, 'n_models'):
            print(f"Số cây trong rừng: {last_step.n_models}")
        else:
            print("Không xác định được số cây (không phải ARFClassifier).")
    elif hasattr(model, 'n_models'):
        print(f"Số cây trong rừng: {model.n_models}")
    else:
        print("Không xác định được số cây.")
else:
    model = preprocessing.StandardScaler() | forest.ARFClassifier(
        n_models=10,
        max_features="sqrt",
        seed=42
    )
    print("--- Khởi tạo mô hình mới ---")

# Thông tin dữ liệu train
print(f"📊 Số mẫu train: {len(X_new)}")
print(f"🔑 Số class: {len(set(y_new))}")
print(f"Các nhãn: {set(y_new)}")

# Train liên tục từng dòng dữ liệu
for idx, (xi, yi) in enumerate(zip(X_new.to_dict(orient="records"), y_new), last_trained_row + 1):
    model.learn_one(xi, yi)
    if idx % 20 == 0 or idx == len(X):
        print(f"Đã train {idx}/{len(X)} mẫu...")

# Lưu lại model
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("✅ Mô hình đã được train và lưu tại", model_path)

# Thông tin mô hình sau khi train
print("--- Thông tin mô hình sau khi train ---")
if hasattr(model, 'steps') and isinstance(model.steps, list) and len(model.steps) > 0:
    last_step = model.steps[len(model.steps)-1][1]
    if hasattr(last_step, 'n_models'):
        print(f"Số cây trong rừng: {last_step.n_models}")
    else:
        print("Không xác định được số cây (không phải ARFClassifier).")
elif hasattr(model, 'n_models'):
    print(f"Số cây trong rừng: {model.n_models}")
else:
    print("Không xác định được số cây.")

# Dự đoán thử với mẫu cuối cùng
print("Dự đoán thử với mẫu cuối cùng:")
print("  Đầu vào:", xi)
y_pred = model.predict_one(xi)
print("  Nhãn thực tế:", yi)
print("  Dự đoán:", y_pred)

# Sau khi train xong, cập nhật lại số dòng đã train
with open(last_row_path, "w") as f:
    f.write(str(len(X)))
