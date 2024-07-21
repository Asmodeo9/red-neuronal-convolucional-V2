let model;

async function loadModel() {
    model = await tf.loadLayersModel('modelo_tfjs/model.json');
    console.log("Modelo cargado");
}

loadModel();

function preprocessImage(image) {
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([28, 28]) // Cambiar el tamaño de la imagen
        .mean(2) // Convertir a escala de grises
        .expandDims(2) // Añadir dimensión extra
        .expandDims() // Añadir otra dimensión para el batch
        .toFloat();
    return tensor.div(255.0); // Normalizar
}

async function predict() {
    let canvas = document.getElementById('canvas');
    let tensor = preprocessImage(canvas);

    let predictions = await model.predict(tensor).data();
    let resultado = Array.from(predictions)
        .map((p, i) => ({ probabilidad: p, clase: i }))
        .sort((a, b) => b.probabilidad - a.probabilidad)
        .slice(0, 1); // Obtener la clase con mayor probabilidad

    document.getElementById('resultado').innerText = `Predicción: ${resultado[0].clase} con probabilidad de ${resultado[0].probabilidad.toFixed(2)}`;
}

// Dibujo en el lienzo
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let drawing = false;

canvas.addEventListener('mousedown', startPosition);
canvas.addEventListener('mouseup', endPosition);
canvas.addEventListener('mousemove', draw);

function startPosition(e) {
    drawing = true;
    draw(e); // Empezar a dibujar inmediatamente al hacer clic
}

function endPosition() {
    drawing = false;
    ctx.beginPath(); // Para que el trazo no continúe
}

function draw(e) {
    if (!drawing) return;

    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
}