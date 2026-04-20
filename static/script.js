document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const spinner = document.getElementById('loading-spinner');
    const dashboard = document.getElementById('results-dashboard');
    const uploadSection = document.querySelector('.upload-section');
    const resetBtn = document.getElementById('reset-btn');

    // Drag and Drop Events
    dropZone.addEventListener('click', () => fileInput.click());
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) handleFile(files[0]);
    });

    fileInput.addEventListener('change', function() {
        if (this.files.length) handleFile(this.files[0]);
    });

    resetBtn.addEventListener('click', () => {
        dashboard.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        fileInput.value = '';
    });

    function handleFile(file) {
        // Validation
        if (!file.type.match('image.*')) {
            alert('Please select a valid image file (JPG/PNG).');
            return;
        }

        // UI State
        uploadSection.classList.add('hidden');
        spinner.classList.remove('hidden');

        // Form Data
        const formData = new FormData();
        formData.append('file', file);

        // Fetch API
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Server error'); });
            }
            return response.json();
        })
        .then(data => {
            spinner.classList.add('hidden');
            
            // Map Base64 Images
            document.getElementById('img-original').src = `data:image/jpeg;base64,${data.original}`;
            document.getElementById('img-mask').src = `data:image/jpeg;base64,${data.mask}`;
            document.getElementById('img-heatmap').src = `data:image/jpeg;base64,${data.heatmap}`;
            
            // Separate Disclaimer from Narrative if possible
            const parts = data.explanation.split("*** STRICT MEDICAL DISCLAIMER ***");
            document.getElementById('narrative-text').textContent = parts[0].trim();
            
            dashboard.classList.remove('hidden');
        })
        .catch(error => {
            spinner.classList.add('hidden');
            uploadSection.classList.remove('hidden');
            alert(`Error: ${error.message}`);
        });
    }
});
