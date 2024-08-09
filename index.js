// Include TensorFlow.js library
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>

async function loadModel() {
    // Load the pre-trained model
    model = await tf.loadLayersModel('path/to/model.json'); // Path to your TensorFlow.js model
}

async function preprocessImage(file) {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    await new Promise(resolve => img.onload = resolve);

    const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([256, 256]).toFloat();
    const normalized = tensor.sub(127.5).div(127.5);
    return normalized.expandDims(0);
}

async function generateImage(inputTensor) {
    const generatedTensor = model.predict(inputTensor);
    const denormalized = generatedTensor.squeeze().mul(127.5).add(127.5).clipByValue(0, 255);
    const generatedImage = await tf.browser.toPixels(denormalized);

    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(256, 256);
    imageData.data.set(generatedImage);
    ctx.putImageData(imageData, 0, 0);

    return canvas.toDataURL('image/png');
}

document.getElementById('image-selector').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        const inputTensor = await preprocessImage(file);
        const generatedImageURL = await generateImage(inputTensor);

        document.getElementById('selected-image').src = URL.createObjectURL(file);
        document.getElementById('prediction-result').innerHTML = `<img src="${generatedImageURL}" width="224" height="224">`;
    }
});

// Load the model when the page loads
window.onload = loadModel;
