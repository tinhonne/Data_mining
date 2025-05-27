import pandas as pd
import joblib
import os
from river import forest
from river import preprocessing
import pickle

# Đường dẫn file model
model_path = "model/river_random_forest.pkl"
trained_rows_path = "model/trained_rows.txt"

# Đọc dữ liệu mới
df = pd.read_csv("your_data.csv")

target_column = "Plant_Health_Status"
drop_cols = [target_column, "Timestamp", "Plant_ID"]

X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df[target_column] if target_column in df.columns else None

if y is None:
    print(f"File CSV không có cột nhãn '{target_column}'. Không thể train.")
    exit()

# Làm sạch dữ liệu: ép kiểu float, loại dòng lỗi
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
mask = ~X.isnull().any(axis=1)
X = X[mask]
y = y[mask]

# Loại bỏ các dòng header lặp lại (nếu có)
if X.columns[0] != 'Timestamp':  # Giả sử cột đầu là Timestamp
    X = X[X[X.columns[0]] != 'Timestamp']
    y = y[X.index]

# Nếu nhãn là chuỗi, encode sang số
if y.dtype == 'object':
    y = y.astype("category").cat.codes

# Đọc danh sách các dòng đã train
if os.path.exists(trained_rows_path):
    with open(trained_rows_path, "r", encoding="utf-8") as f:
        trained_rows = set(line.strip() for line in f)
else:
    trained_rows = set()

# Xác định các dòng mới thực sự (chưa train)
rows_str = X.astype(str).agg(','.join, axis=1) + ',' + y.astype(str)
mask_new = ~rows_str.isin(trained_rows)
mask_new = mask_new.astype(bool).values.ravel()  # Đảm bảo là 1D bool array
X_new = X[mask_new]
y_new = y[mask_new]
rows_new_str = rows_str[mask_new]

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
for idx, (xi, yi) in enumerate(zip(X_new.to_dict(orient="records"), y_new), 1):
    model.learn_one(xi, yi)
    if idx % 20 == 0 or idx == len(X_new):
        print(f"Đã train {idx}/{len(X_new)} mẫu...")

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
if len(X_new) > 0:
    xi = X_new.to_dict(orient="records")[-1]
    yi = y_new.iloc[-1]
    y_pred = model.predict_one(xi)
    print("Dự đoán thử với mẫu cuối cùng:")
    print("  Đầu vào:", xi)
    print("  Nhãn thực tế:", yi)
    print("  Dự đoán:", y_pred)

# Sau khi train xong, cập nhật lại các dòng đã train
with open(trained_rows_path, "a", encoding="utf-8") as f:
    for row in rows_new_str:
        f.write(row + "\n")
