document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const btn = document.getElementById('submitBtn');
    const clearBtn = document.getElementById('clearBtn');
    const reviewBox = document.getElementById('reviewBox');
    const resultBox = document.getElementById('resultBox');

    form.onsubmit = function() {
        btn.value = "Analyzing... 🔍";
        btn.style.opacity = "0.7";
    };

    clearBtn.addEventListener('click', function() {
        reviewBox.value = "";
        if (resultBox) resultBox.style.display = 'none';
        btn.value = "Analyze Now ✨";
        btn.style.opacity = "1";
    });
});
