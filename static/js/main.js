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
const engineNotice     = document.getElementById('engine-notice');
const editorStatus     = document.getElementById('editor-status');

// ── State ────────────────────────────────────────────────────────────────────
let debounceTimer  = null;
let currentGhost   = '';
let ghostSeq       = 0;     // monotonic counter — ignore stale ghost responses
let generating     = false; // true while Shift+Tab / Ctrl+Shift+Tab is in-flight

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

function showStatus(msg) {
    if (editorStatus) {
        editorStatus.textContent = msg;
        editorStatus.style.display = msg ? 'block' : 'none';
    }
}

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

function updateEngineNotice() {
    if (!engineNotice) return;
    if (getEngine() === 'transformer') {
        engineNotice.style.display = 'block';
        ghostField.innerHTML = '';
        currentGhost = '';
    } else {
        engineNotice.style.display = 'none';
    }
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

    if (engineSelect) {
        engineSelect.addEventListener('change', () => {
            updateEngineNotice();
            inputField.dispatchEvent(new Event('input'));
        });
    }
    updateEngineNotice();
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

    ghostField.innerHTML = '';
    currentGhost = '';

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
        if (getEngine() === 'ngram') {
            const seq = ++ghostSeq;
            try {
                const r = await fetch(`/predict_next?temperature=${getTemperature()}&text=` + encodeURIComponent(text));
                const d = await r.json();
                if (seq === ghostSeq) {
                    renderGhost(text, d.ghost || '');
                }
            } catch (_) {}
        }

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

    // ── Ctrl + Shift + Tab → paragraph ──
    if (e.key === 'Tab' && e.shiftKey && e.ctrlKey) {
        e.preventDefault();
        if (!text.length || generating) return;
        generating = true;
        showStatus('Generating paragraph...');
        fetch(`/predict_paragraph?engine=${engine}&temperature=${getTemperature()}&text=` + encodeURIComponent(text))
            .then(r => r.json())
            .then(data => {
                if (data.error) { console.warn('Paragraph error:', data.error); return; }
                if (data.completion) {
                    inputField.value += data.completion;
                    inputField.dispatchEvent(new Event('input'));
                }
            })
            .catch(err => console.error('predict_paragraph failed:', err))
            .finally(() => { generating = false; showStatus(''); });
        return;
    }

    // ── Shift + Tab → sentence ──
    if (e.key === 'Tab' && e.shiftKey && !e.ctrlKey) {
        e.preventDefault();
        if (!text.length || generating) return;
        generating = true;
        showStatus('Generating sentence...');
        fetch(`/predict_sentence?engine=${engine}&temperature=${getTemperature()}&text=` + encodeURIComponent(text))
            .then(r => r.json())
            .then(data => {
                if (data.error) { console.warn('Sentence error:', data.error); return; }
                if (data.completion) {
                    inputField.value += data.completion;
                    inputField.dispatchEvent(new Event('input'));
                }
            })
            .catch(err => console.error('predict_sentence failed:', err))
            .finally(() => { generating = false; showStatus(''); });
        return;
    }

    // ── Tab → accept one ghost word ──
    if (e.key === 'Tab' && !e.shiftKey && !e.ctrlKey) {
        e.preventDefault();
        if (!currentGhost || currentGhost === text) return;

        const suggestionPart = currentGhost.slice(text.length);
        const trimmed        = suggestionPart.trimStart();
        const leadingChars   = suggestionPart.length - trimmed.length;
        const spaceInTrimmed = trimmed.indexOf(' ');

        let newValue;
        if (spaceInTrimmed === -1) {
            newValue = currentGhost;
        } else {
            newValue = text + suggestionPart.slice(0, leadingChars + spaceInTrimmed);
        }

        inputField.value = newValue;
        inputField.setSelectionRange(newValue.length, newValue.length);
        ghostField.innerHTML = '';
        currentGhost = '';
        inputField.dispatchEvent(new Event('input'));
        return;
    }

    // ── Escape → clear ghost ──
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
