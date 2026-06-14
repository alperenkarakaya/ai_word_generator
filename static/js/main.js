// ── DOM refs ─────────────────────────────────────────────────────────────────
const inputField       = document.getElementById('user-input');
const ghostField       = document.getElementById('ghost-input');
const unigramList      = document.getElementById('unigram-list');
const bigramList       = document.getElementById('bigram-list');
const trigramList      = document.getElementById('trigram-list');
const fourgramList     = document.getElementById('fourgram-list');
const fivegramList     = document.getElementById('fivegram-list');
const sixgramList      = document.getElementById('sixgram-list');
const sevengramList    = document.getElementById('sevengram-list');
const bigramWordSpan   = document.getElementById('bigram-word');
const trigramCtxSpan   = document.getElementById('trigram-context');
const fourgramCtxLabel = document.getElementById('fourgram-ctx-label');
const fivegramCtxLabel = document.getElementById('fivegram-ctx-label');
const sixgramCtxLabel  = document.getElementById('sixgram-ctx-label');
const sevengramCtxLabel= document.getElementById('sevengram-ctx-label');
const engineSelect     = document.getElementById('engine-select');
const temperatureSlider= document.getElementById('temperature-slider');
const temperatureValue = document.getElementById('temperature-value');

// ── State ────────────────────────────────────────────────────────────────────
let debounceTimer  = null;
let currentGhost   = '';   // full ghost-layer text (user text + gray suggestion)

// ── Helpers ──────────────────────────────────────────────────────────────────
function getEngine() {
    return engineSelect ? engineSelect.value : 'ngram';
}

function getTemperature() {
    return temperatureSlider ? temperatureSlider.value : '1.0';
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

    if (temperatureSlider && temperatureValue) {
        temperatureSlider.addEventListener('input', () => {
            temperatureValue.textContent = parseFloat(temperatureSlider.value).toFixed(1);
        });
    }
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
                const r = await fetch(`/predict_next?temperature=${getTemperature()}&text=` + encodeURIComponent(text));
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
        fetch(`/predict_paragraph?engine=${engine}&temperature=${getTemperature()}&text=` + encodeURIComponent(text))
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
        fetch(`/predict_sentence?engine=${engine}&temperature=${getTemperature()}&text=` + encodeURIComponent(text))
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

    // ── Tab alone → accept one word at a time from the 7-word ghost ──────────
    if (e.key === 'Tab' && !e.shiftKey && !e.ctrlKey) {
        e.preventDefault();
        if (!currentGhost || currentGhost === text) return;

        // The suggestion is everything in currentGhost beyond what the user typed.
        // It may start with a space  (" sat on…")  — next-word case
        //                   a letter ("t sat on…") — partial-word-completion case
        //                   a punct  (".")          — only punctuation left
        const suggestionPart = currentGhost.slice(text.length);
        const trimmed        = suggestionPart.trimStart();
        const leadingChars   = suggestionPart.length - trimmed.length;
        const spaceInTrimmed = trimmed.indexOf(' ');

        let newValue;
        if (spaceInTrimmed === -1) {
            // Only one token left (word or punctuation) — accept everything.
            newValue = currentGhost;
        } else {
            // Accept up to (not including) the first inter-word space.
            newValue = text + suggestionPart.slice(0, leadingChars + spaceInTrimmed);
        }

        inputField.value = newValue;
        inputField.setSelectionRange(newValue.length, newValue.length);
        ghostField.innerHTML = '';
        currentGhost = '';
        // Re-fetch: gets a fresh 7-word ghost starting from the new position.
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
function truncateCtx(ctx, maxLen = 24) {
    if (!ctx) return '—';
    return ctx.length > maxLen ? '…' + ctx.slice(-(maxLen - 1)) : ctx;
}

function renderProbabilities(data) {
    renderList(unigramList, data.unigram || [], 'Start typing…');

    bigramWordSpan.textContent = data.current_word || '—';
    renderList(bigramList, data.bigram || [], 'Need ≥ 1 word');

    trigramCtxSpan.textContent = data.context || '—';
    renderList(trigramList, data.trigram || [], 'Need ≥ 2 words');

    if (fourgramCtxLabel) {
        const c4 = data.context4 || '';
        fourgramCtxLabel.textContent = c4 ? `"${truncateCtx(c4)}"` : 'After last 3 words';
        fourgramCtxLabel.title = c4;
    }
    renderList(fourgramList, data.fourgram || [], 'Need ≥ 3 words');

    if (fivegramCtxLabel) {
        const c5 = data.context5 || '';
        fivegramCtxLabel.textContent = c5 ? `"${truncateCtx(c5)}"` : 'After last 4 words';
        fivegramCtxLabel.title = c5;
    }
    renderList(fivegramList, data.fivegram || [], 'Need ≥ 4 words');

    if (sixgramCtxLabel) {
        const c6 = data.context6 || '';
        sixgramCtxLabel.textContent = c6 ? `"${truncateCtx(c6)}"` : 'After last 5 words';
        sixgramCtxLabel.title = c6;
    }
    renderList(sixgramList, data.sixgram || [], 'Need ≥ 5 words');

    if (sevengramCtxLabel) {
        const c7 = data.context7 || '';
        sevengramCtxLabel.textContent = c7 ? `"${truncateCtx(c7)}"` : 'After last 6 words';
        sevengramCtxLabel.title = c7;
    }
    renderList(sevengramList, data.sevengram || [], 'Need ≥ 6 words');
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
