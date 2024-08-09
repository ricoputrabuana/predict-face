async function loadModel() {
    // Load the TensorFlow.js model
    const model = await tf.loadLayersModel('model/model.json');
    return model;
}

function preprocessImage(image) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 256;
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const tensor = tf.browser.fromPixels(imgData).toFloat();
            const normalized = tensor.sub(tf.scalar(127.5)).div(tf.scalar(127.5));
            resolve(normalized.expandDims(0));
        };
        img.src = URL.createObjectURL(image);
    });
}

async function generateImage() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please upload an image file.');
        return;
    }

    const model = await loadModel();
    const inputImage = await preprocessImage(file);
    const generatedImageTensor = model.predict(inputImage);
    const generatedImage = generatedImageTensor.squeeze().mul(tf.scalar(127.5)).add(tf.scalar(127.5));
    const generatedImageData = await tf.browser.toPixels(generatedImage);
    
    // Display uploaded image
    const uploadedImageContainer = document.getElementById('uploadedImageContainer');
    uploadedImageContainer.innerHTML = `<h2>Uploaded Image:</h2><img src="${URL.createObjectURL(file)}" alt="Uploaded Image" />`;
    
    // Display generated image
    const generatedImageContainer = document.getElementById('generatedImageContainer');
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    const imgData = new ImageData(new Uint8ClampedArray(generatedImageData), 256, 256);
    ctx.putImageData(imgData, 0, 0);
    generatedImageContainer.innerHTML = `<h2>Generated Image:</h2>`;
    generatedImageContainer.appendChild(canvas);
}

function handleUpload() {
    generateImage();
}
