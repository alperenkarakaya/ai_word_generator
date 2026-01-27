// ===== DOM Elements =====
const inputField = document.getElementById('user-input');
const ghostField = document.getElementById('ghost-input');
const unigramList = document.getElementById('unigram-list');
const bigramList = document.getElementById('bigram-list');
const trigramList = document.getElementById('trigram-list');
const bigramWordSpan = document.getElementById('bigram-word');
const trigramContextSpan = document.getElementById('trigram-context');

// ===== State =====
let currentSuggestion = "";
let debounceTimer = null;

// ===== Event Listeners =====
inputField.addEventListener('input', handleInput);
inputField.addEventListener('keydown', handleKeydown);
inputField.addEventListener('scroll', handleScroll);

// İlk yüklemede 1-gram'ı göster
window.addEventListener('DOMContentLoaded', () => {
    updateProbabilities({unigram: [], bigram: [], trigram: [], current_word: "", context: ""}, "");
    // Boş metin ile 1-gram verilerini al
    fetch('/probabilities?text=')
        .then(response => response.json())
        .then(data => updateProbabilities(data, ""))
        .catch(error => console.error('Initial load error:', error));
});

// ===== Input Handler =====
async function handleInput() {
    const text = this.value;
    
    // Debounce API calls
    clearTimeout(debounceTimer);
    
    debounceTimer = setTimeout(async () => {
        // Handle prediction (ghost text)
        if (text.length > 0 && !text.endsWith(" ")) {
            try {
                const response = await fetch(`/predict?text=${encodeURIComponent(text)}`);
                const data = await response.json();
                currentSuggestion = data.prediction || "";

                if (currentSuggestion) {
                    ghostField.textContent = text + currentSuggestion;
                } else {
                    ghostField.textContent = "";
                }
            } catch (error) {
                console.error('Prediction error:', error);
                ghostField.textContent = "";
            }
        } else {
            ghostField.textContent = "";
            currentSuggestion = "";
        }
        
        // Handle probabilities
        try {
            const response = await fetch(`/probabilities?text=${encodeURIComponent(text)}`);
            const data = await response.json();
            updateProbabilities(data, text);
        } catch (error) {
            console.error('Probabilities error:', error);
        }
    }, 150); // 150ms debounce
}

// ===== Keydown Handler =====
function handleKeydown(e) {
    if (e.key === 'Tab') {
        e.preventDefault();
        if (currentSuggestion) {
            this.value += currentSuggestion + " ";
            this.dispatchEvent(new Event('input'));
        }
    }
}

// ===== Scroll Sync =====
function handleScroll() {
    ghostField.scrollTop = this.scrollTop;
}

// ===== Update Probabilities =====
function updateProbabilities(data, text) {
    const hasUnigram = data.unigram && data.unigram.length > 0;
    const hasBigram = data.bigram && data.bigram.length > 0;
    const hasTrigram = data.trigram && data.trigram.length > 0;
    
    // Update 1-gram
    if (hasUnigram) {
        renderPredictions(unigramList, data.unigram);
    } else {
        unigramList.innerHTML = '<div class="no-data">Veri yükleniyor...</div>';
    }
    
    // Update 2-gram
    if (hasBigram) {
        bigramWordSpan.textContent = data.current_word;
        renderPredictions(bigramList, data.bigram);
    } else {
        bigramWordSpan.textContent = data.current_word || '-';
        bigramList.innerHTML = '<div class="no-data">En az 1 kelime gerekli</div>';
    }
    
    // Update 3-gram
    if (hasTrigram) {
        trigramContextSpan.textContent = data.context;
        renderPredictions(trigramList, data.trigram);
    } else {
        trigramContextSpan.textContent = data.context || '-';
        trigramList.innerHTML = '<div class="no-data">En az 2 kelime gerekli</div>';
    }
}

// ===== Render Predictions =====
function renderPredictions(container, predictions) {
    container.innerHTML = '';
    
    predictions.forEach((item, index) => {
        const itemDiv = document.createElement('div');
        itemDiv.className = `prediction-item${index === 0 ? ' top-choice' : ''}`;

        itemDiv.innerHTML = `
            <div class="word-info">
                <span class="word-rank">#${index + 1}</span>
                <span class="word-text">${escapeHtml(item.word)}</span>
            </div>
            <div class="probability-info">
                <div class="probability-bar-container">
                    <div class="probability-bar" style="width: ${item.probability}%"></div>
                </div>
                <span class="probability-value">${item.probability}%</span>
            </div>
        `;

        container.appendChild(itemDiv);
    });
}

// ===== Utility Functions =====
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== Initialize =====
console.log('✓ AI Word Generator (1-gram, 2-gram, 3-gram) initialized');