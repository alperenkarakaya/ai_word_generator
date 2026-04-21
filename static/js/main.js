// ── DOM refs ────────────────────────────────────────────────────────────────
const inputField        = document.getElementById('user-input');
const ghostField        = document.getElementById('ghost-input');
const unigramList       = document.getElementById('unigram-list');
const bigramList        = document.getElementById('bigram-list');
const trigramList       = document.getElementById('trigram-list');
const bigramWordSpan    = document.getElementById('bigram-word');
const trigramCtxSpan    = document.getElementById('trigram-context');

// ── State ───────────────────────────────────────────────────────────────────
let debounceTimer = null;

// ── Boot ─────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    fetch('/probabilities?text=')
        .then(r => r.json())
        .then(d => renderProbabilities(d))
        .catch(() => {});
});

// ── Listeners ────────────────────────────────────────────────────────────────
inputField.addEventListener('input',    handleInput);
inputField.addEventListener('keydown',  handleKeydown);
inputField.addEventListener('scroll',   () => { ghostField.scrollTop = inputField.scrollTop; });

// ── Input handler (debounced) ────────────────────────────────────────────────
function handleInput() {
    const text = this.value;
    clearTimeout(debounceTimer);

    debounceTimer = setTimeout(async () => {
        // Ghost-text: N-gram word completion is fast and always available
        ghostField.textContent = '';
        if (text.length > 0 && !text.endsWith(' ')) {
            // (optional ghost text from N-gram can be added here later)
        }

        // Probability panels
        try {
            const r = await fetch('/probabilities?text=' + encodeURIComponent(text));
            renderProbabilities(await r.json());
        } catch (_) {}
    }, 150);
}

// ── Keydown handler ──────────────────────────────────────────────────────────
function handleKeydown(e) {
    const text = this.value;

    // Shift + Tab → sentence completion (Transformer)
    if (e.key === 'Tab' && e.shiftKey && !e.ctrlKey) {
        e.preventDefault();
        if (!text.length) return;

        fetch('/predict_sentence?text=' + encodeURIComponent(text))
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    console.warn('Sentence error:', data.error);
                    return;
                }
                if (data.completion) {
                    inputField.value += data.completion;
                    inputField.dispatchEvent(new Event('input'));
                }
            })
            .catch(err => console.error('predict_sentence failed:', err));
    }

    // Ctrl + Shift + Tab → paragraph generation (Phase 3)
    if (e.key === 'Tab' && e.shiftKey && e.ctrlKey) {
        e.preventDefault();
        if (!text.length) return;

        fetch('/predict_paragraph?text=' + encodeURIComponent(text))
            .then(r => r.json())
            .then(data => {
                if (data.error) { console.warn('Paragraph error:', data.error); return; }
                if (data.completion) {
                    inputField.value += data.completion;
                    inputField.dispatchEvent(new Event('input'));
                }
            })
            .catch(err => console.error('predict_paragraph failed:', err));
    }
}

// ── Render probability panels ────────────────────────────────────────────────
function renderProbabilities(data) {
    renderList(unigramList, data.unigram  || [], 'Start typing…');

    bigramWordSpan.textContent = data.current_word || '—';
    renderList(bigramList,  data.bigram   || [], 'Need ≥ 1 word');

    trigramCtxSpan.textContent = data.context || '—';
    renderList(trigramList, data.trigram  || [], 'Need ≥ 2 words');
}

function renderList(container, items, emptyMsg) {
    if (!items.length) {
        container.innerHTML = `<div class="no-data">${emptyMsg}</div>`;
        return;
    }
    container.innerHTML = items.map((item, i) => `
        <div class="prediction-item${i === 0 ? ' top-choice' : ''}">
            <div class="word-info">
                <span class="word-rank">#${i + 1}</span>
                <span class="word-text">${escapeHtml(item.word)}</span>
            </div>
            <div class="probability-info">
                <div class="probability-bar-container">
                    <div class="probability-bar" style="width:${item.probability}%"></div>
                </div>
                <span class="probability-value">${item.probability}%</span>
            </div>
        </div>`
    ).join('');
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}
