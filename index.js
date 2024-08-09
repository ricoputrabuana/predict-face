// Global model variable
let model;

async function loadModel() {
    // Load the pre-trained model from a URL
    model = await tf.loadLayersModel('model/model.json'); // Path to your TensorFlow.js model
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

    const canvas = document.getElementById('outputCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 256; // Set canvas width
    canvas.height = 256; // Set canvas height
    const imageData = ctx.createImageData(256, 256);
    imageData.data.set(generatedImage);
    ctx.putImageData(imageData, 0, 0);

    return canvas.toDataURL('image/png');
}

async function handleUpload() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (file) {
        const inputTensor = await preprocessImage(file);
        const generatedImageURL = await generateImage(inputTensor);

        document.getElementById('uploadedImageContainer').innerHTML = `<img src="${URL.createObjectURL(file)}" width="256" height="256">`;
        document.getElementById('generatedImageContainer').innerHTML = `<img src="${generatedImageURL}" width="256" height="256">`;
    }
}

// Load the model when the page loads
window.onload = loadModel;
