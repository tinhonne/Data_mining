document.getElementById('sensorForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const formData = new FormData(e.target);
  const data = {};

  formData.forEach((value, key) => {
    data[key] = parseFloat(value);
  });

  try {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    const result = await response.json();

    if (result.prediction) {
      document.getElementById('result').innerText = '🧠 Dự đoán: ' + result.prediction;
    } else if (result.error) {
      document.getElementById('result').innerText = '❌ Lỗi: ' + result.error;
    }
  } catch (err) {
    document.getElementById('result').innerText = '❌ Không thể kết nối đến server.';
  }
});

// Upload file CSV và train model
const uploadForm = document.getElementById('uploadForm');
const csvFile = document.getElementById('csvFile');
const uploadStatus = document.getElementById('uploadStatus');

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!csvFile.files.length) return;
  const formData = new FormData();
  formData.append('file', csvFile.files[0]);
  uploadStatus.innerText = '⏳ Đang upload và train...';
  try {
    const res = await fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData
    });
    const result = await res.json();
    if (result.status) {
      uploadStatus.innerText = '✅ ' + result.status;
      await loadModelStatus();
    } else {
      uploadStatus.innerText = '❌ Lỗi: ' + (result.error || 'Không xác định');
    }
  } catch (err) {
    uploadStatus.innerText = '❌ Không thể kết nối đến server.';
  }
});

// Lấy trạng thái model
async function loadModelStatus() {
  try {
    const res = await fetch('http://localhost:5000/model_status');
    const data = await res.json();
    document.getElementById('soMauTrain').innerText = data.so_mau_train;
    document.getElementById('cacNhan').innerText = data.cac_nhan.join(', ');
  } catch (err) {
    document.getElementById('soMauTrain').innerText = '?';
    document.getElementById('cacNhan').innerText = '?';
  }
}

// Tải trạng thái model khi trang vừa load
window.addEventListener('DOMContentLoaded', loadModelStatus);
