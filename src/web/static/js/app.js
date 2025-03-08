/**
 * ASL Translator Web App
 * JavaScript for handling webcam, predictions and UI updates
 */

// DOM elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const webcamToggle = document.getElementById('webcamToggle');
const modelSelect = document.getElementById('modelSelect');
const predictionDisplay = document.getElementById('predictionDisplay');
const confidenceDisplay = document.getElementById('confidenceDisplay');
const historyText = document.getElementById('historyText');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const appendBtn = document.getElementById('appendBtn');
const statusMessage = document.getElementById('status-message');

// Global state
let isTranslating = false;
let videoStream = null;
let animationFrameId = null;
let currentPrediction = null;
let translationBuffer = [];
let bufferThreshold = 5; // Consecutive same predictions needed

// Initialize the webcam
async function initWebcam() {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });
        
        video.srcObject = videoStream;
        
        // Wait for video to be ready
        await new Promise(resolve => {
            video.onloadedmetadata = () => {
                resolve();
            };
        });
        
        showStatusMessage('Webcam initialized successfully.', 'success');
        startBtn.disabled = false;
        
    } catch (error) {
        console.error('Error accessing webcam:', error);
        showStatusMessage('Failed to access webcam: ' + error.message, 'danger');
    }
}

// Stop webcam
function stopWebcam() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        videoStream = null;
    }
}

// Capture frame and send for prediction
async function captureAndPredict() {
    if (!isTranslating) return;
    
    const ctx = canvas.getContext('2d');
    
    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Get image data as base64
    const imageData = canvas.toDataURL('image/jpeg');
    
    try {
        // Send to server for prediction
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData,
                model: modelSelect.value
            })
        });
        
        const result = await response.json();
        
        if (result.error) {
            console.error('Prediction error:', result.error);
            showStatusMessage('Prediction error: ' + result.error, 'danger');
        } else {
            // Update prediction display
            updatePrediction(result.class, result.confidence);
            
            // Update canvas with annotated image if available
            if (result.annotated_image) {
                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = result.annotated_image;
            }
        }
    } catch (error) {
        console.error('Error sending prediction request:', error);
        showStatusMessage('Network error: ' + error.message, 'danger');
    }
    
    // Continue loop
    animationFrameId = requestAnimationFrame(captureAndPredict);
}

// Update prediction display
function updatePrediction(predClass, confidence) {
    // Update displays
    predictionDisplay.textContent = predClass;
    confidenceDisplay.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    
    // Apply color based on confidence
    if (confidence > 0.8) {
        predictionDisplay.style.color = '#198754'; // Bootstrap success color
    } else if (confidence > 0.5) {
        predictionDisplay.style.color = '#fd7e14'; // Bootstrap warning color
    } else {
        predictionDisplay.style.color = '#dc3545'; // Bootstrap danger color
    }
    
    // Handle prediction buffer for stability
    if (currentPrediction === predClass) {
        translationBuffer.push(predClass);
    } else {
        translationBuffer = [predClass];
        currentPrediction = predClass;
    }
    
    // Enable append button when confident
    appendBtn.disabled = translationBuffer.length < bufferThreshold;
}

// Add current prediction to history
function appendToHistory() {
    if (translationBuffer.length >= bufferThreshold) {
        // Convert 'space' and 'nothing' to actual space or nothing
        let characterToAdd = currentPrediction;
        if (characterToAdd === 'space') {
            characterToAdd = ' ';
        } else if (characterToAdd === 'nothing') {
            characterToAdd = '';
        }
        
        historyText.textContent += characterToAdd;
        translationBuffer = [];
    }
}

// Clear translation history
function clearHistory() {
    historyText.textContent = '';
}

// Show status message
function showStatusMessage(message, type = 'info') {
    statusMessage.textContent = message;
    statusMessage.className = `alert alert-${type} mt-2`;
    
    // Remove d-none class
    if (statusMessage.classList.contains('d-none')) {
        statusMessage.classList.remove('d-none');
    }
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        statusMessage.classList.add('d-none');
    }, 3000);
}

// Toggle translation
function toggleTranslation(start) {
    if (start) {
        // Start translation
        isTranslating = true;
        startBtn.classList.add('d-none');
        stopBtn.classList.remove('d-none');
        
        // Show canvas
        canvas.style.display = 'block';
        
        // Start prediction loop
        captureAndPredict();
        
    } else {
        // Stop translation
        isTranslating = false;
        stopBtn.classList.add('d-none');
        startBtn.classList.remove('d-none');
        
        // Cancel animation frame
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        
        // Hide canvas
        canvas.style.display = 'none';
    }
}

// Training functions
let trainingStatusInterval = null;

// Start model training
async function startTraining(formData) {
    try {
        // Reset training UI
        document.getElementById('trainingStatus').textContent = 'Starting...';
        document.getElementById('trainingProgress').style.width = '0%';
        document.getElementById('trainingLog').textContent = '';
        document.getElementById('trainingPlotContainer').classList.add('d-none');
        
        // Submit training request
        const response = await fetch('/train', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            // Start polling for status updates
            startStatusPolling();
            
            // Disable form while training
            document.getElementById('trainBtn').disabled = true;
            document.getElementById('trainForm').classList.add('opacity-50');
            
            // Show message
            alert('Training started successfully! Monitor progress in the Training Status panel.');
        } else {
            alert('Error starting training: ' + result.message);
        }
    } catch (error) {
        console.error('Error submitting training request:', error);
        alert('Network error: ' + error.message);
    }
}

// Poll training status
function startStatusPolling() {
    // Clear any existing interval
    if (trainingStatusInterval) {
        clearInterval(trainingStatusInterval);
    }
    
    // Start new polling interval
    trainingStatusInterval = setInterval(updateTrainingStatus, 2000);
}

// Update training status
async function updateTrainingStatus() {
    try {
        const response = await fetch('/training_status');
        const data = await response.json();
        
        // Update status and progress
        document.getElementById('trainingStatus').textContent = capitalizeFirst(data.status);
        document.getElementById('trainingProgress').style.width = `${data.progress}%`;
        document.getElementById('trainingProgress').setAttribute('aria-valuenow', data.progress);
        
        // Update log
        const logElement = document.getElementById('trainingLog');
        logElement.textContent = data.log.join('\n');
        
        // Auto-scroll to bottom
        logElement.scrollTop = logElement.scrollHeight;
        
        // Show plot if training is completed
        if (data.status === 'completed') {
            // Stop polling
            clearInterval(trainingStatusInterval);
            trainingStatusInterval = null;
            
            // Show plot
            document.getElementById('trainingPlotContainer').classList.remove('d-none');
            document.getElementById('trainingPlot').src = `/training_plot?t=${Date.now()}`; // Add timestamp to prevent caching
            
            // Re-enable form
            document.getElementById('trainBtn').disabled = false;
            document.getElementById('trainForm').classList.remove('opacity-50');
            
            // Make sure progress bar is at 100%
            document.getElementById('trainingProgress').style.width = '100%';
            document.getElementById('trainingProgress').setAttribute('aria-valuenow', 100);
        }
        
        // If training failed, stop polling
        if (data.status === 'failed') {
            clearInterval(trainingStatusInterval);
            trainingStatusInterval = null;
            
            // Re-enable form
            document.getElementById('trainBtn').disabled = false;
            document.getElementById('trainForm').classList.remove('opacity-50');
        }
        
    } catch (error) {
        console.error('Error updating training status:', error);
    }
}

// Helper function to capitalize first letter
function capitalizeFirst(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

// Toggle precomputed keypoints field based on model type
function togglePrecomputedField() {
    const modelType = document.getElementById('modelTypeSelect').value;
    const precomputedGroup = document.getElementById('precomputedKeypointsGroup');
    
    if (modelType === 'keypoint') {
        precomputedGroup.style.display = 'block';
    } else {
        precomputedGroup.style.display = 'none';
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initialize webcam if toggle is on
    if (webcamToggle && webcamToggle.checked) {
        initWebcam();
    }
    
    // --- Translation Tab Event Listeners ---
    
    // Webcam toggle
    if (webcamToggle) {
        webcamToggle.addEventListener('change', () => {
            if (webcamToggle.checked) {
                initWebcam();
            } else {
                stopWebcam();
                toggleTranslation(false);
                startBtn.disabled = true;
            }
        });
    }
    
    // Start button
    if (startBtn) {
        startBtn.addEventListener('click', () => {
            toggleTranslation(true);
        });
    }
    
    // Stop button
    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            toggleTranslation(false);
        });
    }
    
    // Append button
    if (appendBtn) {
        appendBtn.addEventListener('click', appendToHistory);
    }
    
    // Clear history button
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', clearHistory);
    }
    
    // Model select change
    if (modelSelect) {
        modelSelect.addEventListener('change', () => {
            showStatusMessage(`Switched to ${modelSelect.value} model`);
            
            // Reset buffer when changing models
            translationBuffer = [];
            currentPrediction = null;
        });
    }
    
    // --- Training Tab Event Listeners ---
    
    // Model type change
    const modelTypeSelect = document.getElementById('modelTypeSelect');
    if (modelTypeSelect) {
        togglePrecomputedField(); // Initial toggle
        modelTypeSelect.addEventListener('change', togglePrecomputedField);
    }
    
    // Training form submission
    const trainForm = document.getElementById('trainForm');
    if (trainForm) {
        trainForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const formData = new FormData(trainForm);
            startTraining(formData);
        });
    }
    
    // Refresh status button
    const refreshStatusBtn = document.getElementById('refreshStatusBtn');
    if (refreshStatusBtn) {
        refreshStatusBtn.addEventListener('click', updateTrainingStatus);
    }
    
    // Check if we should show training status on page load
    updateTrainingStatus();
});