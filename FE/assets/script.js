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
