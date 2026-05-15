// ── DOM refs ─────────────────────────────────────────────────────────────────
const inputField     = document.getElementById('user-input');
const ghostField     = document.getElementById('ghost-input');
const unigramList    = document.getElementById('unigram-list');
const bigramList     = document.getElementById('bigram-list');
const trigramList    = document.getElementById('trigram-list');
const bigramWordSpan = document.getElementById('bigram-word');
const trigramCtxSpan = document.getElementById('trigram-context');
const engineSelect   = document.getElementById('engine-select');

// ── State ────────────────────────────────────────────────────────────────────
let debounceTimer  = null;
let currentGhost   = '';   // full ghost-layer text (user text + gray suggestion)

// ── Helpers ──────────────────────────────────────────────────────────────────
function getEngine() {
    return engineSelect ? engineSelect.value : 'ngram';
}

function escapeHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

/**
 * Renders the ghost overlay.
 * userText  — what the user actually typed
 * ghostFull — userText + gray suggestion (returned by /predict_next)
 *
 * The ghost div sits behind the textarea.  We make the "typed" portion
 * transparent and only the suggestion portion gray — so the user sees their
 * own black text cleanly, with the gray suggestion appearing after it.
 */
function renderGhost(userText, ghostFull) {
    if (!ghostFull || ghostFull.length <= userText.length) {
        ghostField.innerHTML = '';
        currentGhost = '';
        return;
    }
    currentGhost = ghostFull;
    const typed      = ghostFull.slice(0, userText.length);
    const suggestion = ghostFull.slice(userText.length);
    ghostField.innerHTML =
        `<span style="color:transparent">${escapeHtml(typed)}</span>` +
        `<span class="ghost-suggestion">${escapeHtml(suggestion)}</span>`;
}

// ── Boot ─────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    fetch('/probabilities?text=')
        .then(r => r.json())
        .then(d => renderProbabilities(d))
        .catch(() => {});
});

// ── Listeners ────────────────────────────────────────────────────────────────
inputField.addEventListener('input',   handleInput);
inputField.addEventListener('keydown', handleKeydown);
inputField.addEventListener('scroll',  syncScroll);

function syncScroll() {
    ghostField.scrollTop  = inputField.scrollTop;
    ghostField.scrollLeft = inputField.scrollLeft;
}

// ── Input handler (debounced) ─────────────────────────────────────────────────
function handleInput() {
    const text = this.value;

    // Clear ghost immediately when user types so stale suggestion disappears
    ghostField.innerHTML = '';
    currentGhost = '';

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
        // Ghost text — only for N-gram engine
        if (getEngine() === 'ngram') {
            try {
                const r = await fetch('/predict_next?text=' + encodeURIComponent(text));
                const d = await r.json();
                renderGhost(text, d.ghost || '');
            } catch (_) {}
        }

        // Probability panels
        try {
            const r = await fetch('/probabilities?text=' + encodeURIComponent(text));
            renderProbabilities(await r.json());
        } catch (_) {}
    }, 100);
}

// ── Keydown handler ──────────────────────────────────────────────────────────
function handleKeydown(e) {
    const text   = inputField.value;
    const engine = getEngine();

    // ── Ctrl + Shift + Tab → paragraph (must be checked before Shift+Tab) ──
    if (e.key === 'Tab' && e.shiftKey && e.ctrlKey) {
        e.preventDefault();
        if (!text.length) return;
        fetch(`/predict_paragraph?engine=${engine}&text=` + encodeURIComponent(text))
            .then(r => r.json())
            .then(data => {
                if (data.error) { console.warn('Paragraph error:', data.error); return; }
                if (data.completion) {
                    inputField.value += data.completion;
                    inputField.dispatchEvent(new Event('input'));
                }
            })
            .catch(err => console.error('predict_paragraph failed:', err));
        return;
    }

    // ── Shift + Tab → full sentence completion ──────────────────────────────
    if (e.key === 'Tab' && e.shiftKey && !e.ctrlKey) {
        e.preventDefault();
        if (!text.length) return;
        fetch(`/predict_sentence?engine=${engine}&text=` + encodeURIComponent(text))
            .then(r => r.json())
            .then(data => {
                if (data.error) { console.warn('Sentence error:', data.error); return; }
                if (data.completion) {
                    inputField.value += data.completion;
                    inputField.dispatchEvent(new Event('input'));
                }
            })
            .catch(err => console.error('predict_sentence failed:', err));
        return;
    }

    // ── Tab alone → accept ghost word (one word at a time) ─────────────────
    if (e.key === 'Tab' && !e.shiftKey && !e.ctrlKey) {
        e.preventDefault();
        if (!currentGhost || currentGhost === text) return;
        // Set the textarea to the full ghost content (user text + accepted word)
        inputField.value = currentGhost;
        inputField.setSelectionRange(currentGhost.length, currentGhost.length);
        // Clear ghost immediately then trigger re-fetch for the next word
        ghostField.innerHTML = '';
        currentGhost = '';
        inputField.dispatchEvent(new Event('input'));
        return;
    }

    // ── Escape → clear ghost ────────────────────────────────────────────────
    if (e.key === 'Escape') {
        ghostField.innerHTML = '';
        currentGhost = '';
    }
}

// ── Render probability panels ─────────────────────────────────────────────────
function renderProbabilities(data) {
    renderList(unigramList, data.unigram || [], 'Start typing…');

    bigramWordSpan.textContent = data.current_word || '—';
    renderList(bigramList, data.bigram || [], 'Need ≥ 1 word');

    trigramCtxSpan.textContent = data.context || '—';
    renderList(trigramList, data.trigram || [], 'Need ≥ 2 words');
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
