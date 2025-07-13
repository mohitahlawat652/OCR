
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mousemove', draw);

function draw(e) {
  if (!drawing) return;
  ctx.fillStyle = 'black';
  ctx.beginPath();
  ctx.arc(e.offsetX, e.offsetY, 5, 0, Math.PI * 2);
  ctx.fill();
}

function getImageData() {
  const pixels = ctx.getImageData(0, 0, 200, 200).data;
  const resized = [];
  for (let y = 0; y < 20; y++) {
    for (let x = 0; x < 20; x++) {
      const px = ctx.getImageData(x * 10, y * 10, 10, 10).data;
      let sum = 0;
      for (let i = 0; i < px.length; i += 4) {
        sum += (255 - px[i]);
      }
      resized.push(sum / 2550);
    }
  }
  return resized;
}

function train() {
  const label = parseInt(document.getElementById('digit').value);
  const data = {
    pixels: getImageData(),
    label: Array(10).fill(0).map((_, i) => i === label ? 1 : 0)
  };
  fetch('http://localhost:8000/train', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(res => res.json()).then(res => alert("Trained!"));
}

function predict() {
  const data = { pixels: getImageData() };
  fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(res => res.json())
    .then(res => {
      document.getElementById('result').innerText = `Prediction: ${res.prediction}, Confidence: ${res.confidence.toFixed(2)}`;
    });
}

