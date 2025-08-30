// static/js/main.js

document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    // --- SHARED UI Elements ---
    const form = document.getElementById('config-form');
    
    // --- TRAINING UI Elements ---
    const trainBtn = document.getElementById('train-btn');
    const logs = document.getElementById('logs');
    const resultsCard = document.getElementById('results-card');
    const progressCard = document.getElementById('progress-card');
    const setupSection = document.getElementById('setup-section');
    const setupStatusTitle = document.getElementById('setup-status-title');
    const setupProgressBar = document.getElementById('setup-progress-bar');
    const trainingSection = document.getElementById('training-section');
    const epochStatus = document.getElementById('epoch-status');
    const epochProgressBar = document.getElementById('epoch-progress-bar');
    const batchProgressBar = document.getElementById('batch-progress-bar');
    const batchLog = document.getElementById('batch-log');
    const finalPlotContainer = document.getElementById('final-plot-container');
    let accuracyChart, lossChart;
    
    // --- BENCHMARKING UI Elements ---
    const benchmarkBtn = document.getElementById('benchmark-btn');
    const benchmarkProgressCard = document.getElementById('benchmark-progress-card');
    const benchmarkLogs = document.getElementById('benchmark-logs');
    const benchmarkResultsCard = document.getElementById('benchmark-results-card');
    const benchmarkResultsContainer = document.getElementById('benchmark-results-container');

    // --- INVARIANTS UI Elements ---
    const invariantImageUpload = document.getElementById('invariant_image_upload');
    const invariantImagePreviewContainer = document.getElementById('invariant-image-preview-container');
    const invariantImagePreview = document.getElementById('invariant-image-preview');
    const invariantImageFilename = document.getElementById('invariant-image-filename');
    const invariantCalculateBtn = document.getElementById('invariant-calculate-btn');
    const invariantProgressCard = document.getElementById('invariant-progress-card');
    const invariantLogs = document.getElementById('invariant-logs');
    const invariantResultsCard = document.getElementById('invariant-results-card');
    const invariantResultsContainer = document.getElementById('invariant-results-container');
    let uploadedImagePath = null;

    // --- Helper Functions ---
    function logMessage(logElement, message) {
        logElement.innerHTML += message + '\n';
        logElement.scrollTop = logElement.scrollHeight;
    }

    function resetTrainingUI() {
        progressCard.style.display = 'none';
        resultsCard.style.display = 'none';
        setupSection.style.display = 'none';
        trainingSection.style.display = 'none';
        logs.innerHTML = '';
        finalPlotContainer.innerHTML = '';
        if (accuracyChart) accuracyChart.destroy();
        if (lossChart) lossChart.destroy();
    }
    
    function resetBenchmarkUI() {
        benchmarkProgressCard.style.display = 'none';
        benchmarkResultsCard.style.display = 'none';
        benchmarkLogs.innerHTML = '';
        benchmarkResultsContainer.innerHTML = '';
    }

    function resetInvariantUI() {
        invariantProgressCard.style.display = 'none';
        invariantResultsCard.style.display = 'none';
        invariantLogs.innerHTML = '';
        invariantResultsContainer.innerHTML = '';
    }

    function initializeCharts() {
        const accCtx = document.getElementById('accuracyChart').getContext('2d');
        accuracyChart = new Chart(accCtx, {
            type: 'line', data: { labels: [], datasets: [] },
            options: { responsive: true, plugins: { title: { display: true, text: 'Validation Accuracy' }, legend: { position: 'top' } } }
        });
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        lossChart = new Chart(lossCtx, {
            type: 'line', data: { labels: [], datasets: [] },
            options: { responsive: true, plugins: { title: { display: true, text: 'Training Loss' }, legend: { position: 'top' } } }
        });
    }

    const modelColors = ['#0d6efd', '#dc3545', '#198754', '#ffc107', '#6f42c1'];
    let modelColorIndex = 0;

    function getModelDataset(chart, modelName) {
        let dataset = chart.data.datasets.find(ds => ds.label === modelName);
        if (!dataset) {
            const color = modelColors[modelColorIndex % modelColors.length];
            modelColorIndex++;
            dataset = {
                label: modelName, data: [], borderColor: color,
                backgroundColor: color + '33', fill: false, tension: 0.1
            };
            chart.data.datasets.push(dataset);
        }
        return dataset;
    }

    // --- General Socket.IO Listeners ---
    socket.on('connect', () => {
        logMessage(logs, '✅ Connected to server. Ready to train.');
        logMessage(benchmarkLogs, '✅ Connected to server. Ready to benchmark.');
        logMessage(invariantLogs, '✅ Connected to server. Ready to calculate.');
    });
    socket.on('disconnect', () => {
        logMessage(logs, '⚠️ Disconnected from server. Please refresh the page.');
        logMessage(benchmarkLogs, '⚠️ Disconnected from server. Please refresh the page.');
        logMessage(invariantLogs, '⚠️ Disconnected from server. Please refresh the page.');
    });

    // --- TRAINING Socket.IO Listeners ---
    socket.on('status_update', (data) => {
        logMessage(logs, data.log_message);
        if (data.stage) {
            progressCard.style.display = 'block';
            if (data.stage.startsWith('setup')) {
                setupSection.style.display = 'block';
                trainingSection.style.display = 'none';
                if (data.stage === 'setup_images') setupStatusTitle.innerText = 'Preprocessing Images...';
                if (data.stage === 'setup_masks') setupStatusTitle.innerText = 'Preprocessing Masks...';
                if (data.stage === 'setup_caching') setupStatusTitle.innerText = 'Caching Features...';
                if(data.progress !== null) {
                    setupProgressBar.style.width = data.progress + '%';
                    setupProgressBar.innerText = data.progress + '%';
                }
            } else if (data.stage === 'training') {
                setupSection.style.display = 'none';
                trainingSection.style.display = 'block';
            }
        }
    });

    socket.on('epoch_update', (data) => {
        const overallProgress = (data.epoch / data.total_epochs) * 100;
        epochStatus.innerText = `Training Model: ${data.model} - Epoch ${data.epoch}/${data.total_epochs}`;
        epochProgressBar.style.width = overallProgress + '%';
    });
    
    socket.on('batch_update', (data) => {
        batchProgressBar.style.width = data.progress + '%';
        batchLog.innerText = data.log;
    });

    socket.on('chart_update', (data) => {
        const { model, epoch, loss, accuracy } = data;
        if (!accuracyChart || !lossChart) return;
        if (accuracyChart.data.labels.length < epoch) {
            accuracyChart.data.labels.push(`Epoch ${epoch}`);
            lossChart.data.labels.push(`Epoch ${epoch}`);
        }
        const accDataset = getModelDataset(accuracyChart, model);
        const lossDataset = getModelDataset(lossChart, model);
        accDataset.data.push(accuracy);
        lossDataset.data.push(loss);
        accuracyChart.update();
        lossChart.update();
    });

    socket.on('training_complete', (data) => {
        logMessage(logs, '🎉 Training complete!');
        trainBtn.disabled = false;
        trainBtn.innerText = 'Start Training';
        finalPlotContainer.innerHTML = `<h5>Final Comparison Plot</h5><img src="/static/${data.plot_url}?t=${new Date().getTime()}" class="img-fluid" alt="Final Plot">`;
    });
    
    socket.on('training_error', (data) => {
        logMessage(logs, `🔴 ERROR: ${data.error}`);
        trainBtn.disabled = false;
        trainBtn.innerText = 'Start Training';
        alert(`An error occurred during training: ${data.error}`);
    });

    // --- BENCHMARKING Socket.IO Listeners ---
    socket.on('benchmark_log', (data) => {
        logMessage(benchmarkLogs, data.data);
    });

    socket.on('benchmark_result', (data) => {
        if (data.type === 'plot') {
            const plotDiv = document.createElement('div');
            plotDiv.className = 'text-center mb-4';
            plotDiv.innerHTML = `<h5 class="mt-3">${data.title}</h5><img src="/static/${data.url}?t=${new Date().getTime()}" class="img-fluid border rounded" alt="${data.title}">`;
            benchmarkResultsContainer.appendChild(plotDiv);
        } else if (data.type === 'table') {
            const tableDiv = document.createElement('div');
            tableDiv.className = 'table-responsive mb-4';
            let tableHTML = `<h5 class="mt-3">Quantitative Results</h5><table class="table table-bordered table-striped"><thead><tr><th>Model</th><th>Transformation</th><th>Accuracy (%)</th><th>IoU (%)</th><th>Dice (%)</th></tr></thead><tbody>`;
            const results = data.data;
            for (const modelName in results) {
                for (const transName in results[modelName]) {
                    const metrics = results[modelName][transName];
                    tableHTML += `<tr><td>${modelName}</td><td>${transName}</td><td>${metrics.accuracy.toFixed(2)}</td><td>${metrics.iou.toFixed(2)}</td><td>${metrics.dice.toFixed(2)}</td></tr>`;
                }
            }
            tableHTML += '</tbody></table>';
            tableDiv.innerHTML = tableHTML;
            benchmarkResultsContainer.insertBefore(tableDiv, benchmarkResultsContainer.firstChild);
        }
    });

    socket.on('benchmark_complete', () => {
        logMessage(benchmarkLogs, '🎉 Benchmarking complete!');
        benchmarkBtn.disabled = false;
        benchmarkBtn.innerText = 'Start Benchmarking';
    });

    socket.on('benchmark_error', (data) => {
        logMessage(benchmarkLogs, `🔴 ERROR: ${data.error}`);
        benchmarkBtn.disabled = false;
        benchmarkBtn.innerText = 'Start Benchmarking';
        alert(`An error occurred during benchmarking: ${data.error}`);
    });

    // --- INVARIANTS Socket.IO Listeners ---
    socket.on('invariant_log', (data) => {
        logMessage(invariantLogs, data.data);
    });

    socket.on('invariant_result', (data) => {
        invariantResultsContainer.innerHTML = ''; // Clear previous results
        data.results.forEach(result => {
            const resultCol = document.createElement('div');
            resultCol.className = 'col-md-6 col-lg-3 mb-4';
            let similarityText = `Cosine Similarity: <strong>${result.similarity.toFixed(4)}</strong>`;
            if (result.name === "Original") {
                similarityText = "<em>(Reference)</em>";
            }
            resultCol.innerHTML = `
                <div class="card h-100">
                    <div class="card-header text-center"><strong>${result.name}</strong></div>
                    <img src="${result.image_path}?t=${new Date().getTime()}" class="card-img-top p-2" alt="${result.name} Image">
                    <div class="card-body p-2">
                        <img src="${result.plot_b64}" class="img-fluid" alt="${result.name} Plot">
                    </div>
                    <div class="card-footer text-center">
                        <small class="text-muted">${similarityText}</small>
                    </div>
                </div>`;
            invariantResultsContainer.appendChild(resultCol);
        });
        invariantCalculateBtn.disabled = false;
        invariantCalculateBtn.innerHTML = 'Calculate Invariants';
    });

    socket.on('invariant_error', (data) => {
        logMessage(invariantLogs, `🔴 ERROR: ${data.error}`);
        invariantCalculateBtn.disabled = false;
        invariantCalculateBtn.innerHTML = 'Calculate Invariants';
        alert(`An error occurred: ${data.error}`);
    });

    // --- Form and Button Event Listeners ---
    document.querySelectorAll('.browse-btn').forEach(button => {
        button.addEventListener('click', async (event) => {
            const targetInput = document.querySelector(event.currentTarget.dataset.target);
            const browseType = event.currentTarget.dataset.type;
            const endpoint = browseType === 'folder' ? '/browse-folder' : '/browse-file';
            button.disabled = true;
            button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span>';
            try {
                const response = await fetch(endpoint);
                const data = await response.json();
                if (data.path) {
                    targetInput.value = data.path.replace(/\\/g, '/');
                }
            } catch (error) {
                console.error(`Error fetching path from ${endpoint}:`, error);
                alert('Could not open the dialog. Is the server running correctly?');
            } finally {
                button.disabled = false;
                button.innerHTML = 'Browse...';
            }
        });
    });

    document.getElementById('preview-btn').addEventListener('click', async () => {
        const dataPreview = document.getElementById('data-preview');
        const previewCard = document.getElementById('preview-card');
        dataPreview.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>';
        previewCard.style.display = 'block';
        const payload = {
            image_dir: document.getElementById('image_dir').value,
            mask_dir: document.getElementById('mask_dir').value,
        };
        try {
            const response = await fetch('/preview_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            dataPreview.innerHTML = '';
            if (data.images && data.images.length > 0) {
                 data.images.forEach((imgPath, i) => {
                    const maskPath = data.masks[i];
                    const col = document.createElement('div');
                    col.className = 'col-md-4 col-lg-3 mb-3 text-center';
                    col.innerHTML = `<h6>Image</h6><img src="/get_image?path=${encodeURIComponent(imgPath)}" class="img-thumbnail mb-2" alt="Image Preview"><h6>Mask</h6><img src="/get_image?path=${encodeURIComponent(maskPath)}" class="img-thumbnail" alt="Mask Preview">`;
                    dataPreview.appendChild(col);
                });
            } else {
                 dataPreview.innerHTML = '<p class="text-danger text-center">No images found. Check paths and extensions.</p>';
            }
        } catch (error) {
            dataPreview.innerHTML = `<p class="text-danger text-center">Error loading preview: ${error.message}</p>`;
        }
    });

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        resetTrainingUI();
        progressCard.style.display = 'block';
        resultsCard.style.display = 'block';
        initializeCharts();
        modelColorIndex = 0;
        const config = {
            image_dir: document.getElementById('image_dir').value,
            mask_dir: document.getElementById('mask_dir').value,
            class_csv: document.getElementById('class_csv').value,
            output_dir: document.getElementById('output_dir').value,
            epochs: document.getElementById('epochs').value,
            batch_size: document.getElementById('batch_size').value,
            learning_rate: document.getElementById('learning_rate').value,
            img_size: document.getElementById('img_size').value,
            data_subset: document.getElementById('data_subset').value,
        };
        trainBtn.disabled = true;
        trainBtn.innerText = 'Training in Progress...';
        socket.emit('start_training', config);
    });

    benchmarkBtn.addEventListener('click', () => {
        resetBenchmarkUI();
        benchmarkProgressCard.style.display = 'block';
        benchmarkResultsCard.style.display = 'block';
        const config = {
            output_dir: document.getElementById('benchmark_output_dir').value,
            class_csv: document.getElementById('benchmark_class_csv').value,
            img_size: document.getElementById('img_size').value,
            data_subset: document.getElementById('data_subset').value,
            batch_size: document.getElementById('batch_size').value,
        };
        if (!config.output_dir || !config.class_csv) {
            alert("Please select both an Output Directory and a Class CSV file for benchmarking.");
            resetBenchmarkUI();
            return;
        }
        benchmarkBtn.disabled = true;
        benchmarkBtn.innerText = 'Benchmarking...';
        socket.emit('start_benchmarking', config);
    });
    
    // --- INVARIANTS Event Listeners ---
    invariantImageUpload.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        invariantImagePreview.src = URL.createObjectURL(file);
        invariantImagePreviewContainer.style.display = 'block';
        invariantImageFilename.innerText = 'Uploading...';
        invariantCalculateBtn.disabled = true;
        try {
            const response = await fetch('/upload-image', { method: 'POST', body: formData });
            const data = await response.json();
            if (response.ok) {
                uploadedImagePath = data.path;
                invariantImageFilename.innerText = file.name;
                invariantCalculateBtn.disabled = false;
            } else { throw new Error(data.error); }
        } catch (error) {
            invariantImageFilename.innerText = `Upload failed: ${error.message}`;
            alert(`Upload failed: ${error.message}`);
        }
    });

    invariantCalculateBtn.addEventListener('click', () => {
        if (!uploadedImagePath) {
            alert('Please upload an image first.');
            return;
        }
        resetInvariantUI();
        invariantProgressCard.style.display = 'block';
        invariantResultsCard.style.display = 'block';
        invariantResultsContainer.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>';
        const selectedMethod = document.querySelector('input[name="invariantMethod"]:checked').value;
        const config = { image_path: uploadedImagePath, method: selectedMethod };
        invariantCalculateBtn.disabled = true;
        invariantCalculateBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Calculating...';
        socket.emit('calculate_invariants', config);
    });
    
    // Slider value displays
    ['epochs', 'batch_size', 'img_size'].forEach(id => {
        const el = document.getElementById(id);
        const valEl = document.getElementById(id.replace('_','-') + '-val');
        if (el && valEl) {
             el.addEventListener('input', () => valEl.innerText = el.value);
        }
    });
});