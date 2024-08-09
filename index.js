// Import TensorFlow.js
import * as tf from '@tensorflow/tfjs';

// Function to load the model
async function loadModel() {
    const model = await tf.loadLayersModel('model/model.json');
    return model;
}

// Function to process and predict using the model
async function predict(imageElement) {
    const model = await loadModel();
    
    // Preprocess the image to fit the model input shape
    const image = tf.browser.fromPixels(imageElement).toFloat();
    const resizedImage = tf.image.resizeBilinear(image, [256, 256]); // Assuming the model expects 256x256 input
    const normalizedImage = resizedImage.div(tf.scalar(127.5)).sub(tf.scalar(1.0)); // Normalize
    
    // Add batch dimension
    const batchedImage = normalizedImage.expandDims(0);
    
    // Predict
    const prediction = model.predict(batchedImage);
    
    // Post-process and use the prediction
    const output = prediction.squeeze().add(tf.scalar(1.0)).mul(tf.scalar(127.5)); // Denormalize
    const outputImage = await tf.browser.toPixels(output);
    
    // Display or use the output image
    const canvas = document.getElementById('outputCanvas');
    const context = canvas.getContext('2d');
    context.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
    context.putImageData(new ImageData(outputImage, canvas.width, canvas.height), 0, 0);
}

// Example usage
const imageElement = document.getElementById('inputImage');
predict(imageElement);
