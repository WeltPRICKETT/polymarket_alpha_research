/* ══════════════════════════════════════════════════════════════
   POLY_ALPHA v3.0 — Dashboard Application Logic
   ══════════════════════════════════════════════════════════════ */

const API = '';  // Same origin — served by FastAPI

// ── State ────────────────────────────────────────────────────────
let timelineChart = null;
let modelChart = null;
let currentInterval = 'day';
let pipelinePollingId = null;
let mlPollingId = null;
let backtestPollingId = null;

// ── Navigation ───────────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        const tab = item.dataset.tab;
        switchTab(tab);
    });
});

function switchTab(tabId) {
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const navEl = document.querySelector(`[data-tab="${tabId}"]`);
    if (navEl) navEl.classList.add('active');

    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    const tabEl = document.getElementById(`tab-${tabId}`);
    if (tabEl) tabEl.classList.add('active');

    const titles = {
        overview: 'SYS_DASHBOARD',
        analytics: 'DATA_ANALYTICS',
        control: 'CONTROL_PANEL',
        models: 'ML_MODELS',
        visuals: 'VISUAL_OUTPUT',
    };
    document.getElementById('header-title').innerHTML =
        `<span class="header-prefix">//</span> ${titles[tabId] || tabId}`;

    if (tabId === 'analytics') {
        loadTimeline();
        loadWallets();
        loadMarkets();
    } else if (tabId === 'models') {
        loadComparison();
    } else if (tabId === 'visuals') {
        loadVisuals();
    }
}

// ── Interval selector ────────────────────────────────────────────
document.querySelectorAll('.interval-selector .btn-sm').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.interval-selector .btn-sm').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentInterval = btn.dataset.interval;
        loadTimeline();
    });
});

// ── API Helpers ──────────────────────────────────────────────────

async function fetchJSON(endpoint) {
    try {
        const res = await fetch(`${API}${endpoint}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (e) {
        console.error(`API error [${endpoint}]:`, e);
        return null;
    }
}

function formatNumber(n) {
    if (n == null) return '—';
    return Number(n).toLocaleString();
}

function truncAddr(addr) {
    if (!addr || addr.length < 12) return addr || '—';
    return addr.slice(0, 8) + '…' + addr.slice(-6);
}

function formatVolume(vol) {
    if (vol == null) return '—';
    if (vol >= 1e6) return (vol / 1e6).toFixed(2) + 'M';
    if (vol >= 1e3) return (vol / 1e3).toFixed(1) + 'K';
    return vol.toFixed(2);
}

// ── Stats (Overview) ─────────────────────────────────────────────

async function loadStats() {
    const data = await fetchJSON('/api/stats');
    if (!data) return;

    const trades = data.total_transactions || 0;
    const markets = data.total_markets || 0;
    const wallets = data.total_wallets || 0;

    document.getElementById('stat-trades').textContent = formatNumber(trades);
    document.getElementById('stat-markets').textContent = formatNumber(markets);
    document.getElementById('stat-wallets').textContent = formatNumber(wallets);
    document.getElementById('stat-last-update').textContent = data.last_update || 'AWAITING_SIGNAL';

    document.getElementById('bar-trades').style.width = Math.min(100, trades / 100) + '%';
    document.getElementById('bar-markets').style.width = Math.min(100, markets / 5) + '%';
    document.getElementById('bar-wallets').style.width = Math.min(100, wallets / 50) + '%';
}

// ── Logs ─────────────────────────────────────────────────────────

async function loadLogs() {
    const data = await fetchJSON('/api/logs?lines=80');
    if (!data || !data.logs) return;

    const container = document.getElementById('log-container');
    container.innerHTML = data.logs.map(line =>
        `<div class="log-line">> ${line.trim()}</div>`
    ).join('');
    container.scrollTop = container.scrollHeight;
}

// ── Timeline Chart ───────────────────────────────────────────────

async function loadTimeline() {
    const data = await fetchJSON(`/api/trades/timeline?interval=${currentInterval}`);
    if (!data || !data.data || data.data.length === 0) return;

    const labels = data.data.map(d => d.period);
    const counts = data.data.map(d => d.trade_count);
    const volumes = data.data.map(d => d.volume || 0);

    const ctx = document.getElementById('timeline-chart').getContext('2d');
    if (timelineChart) timelineChart.destroy();

    timelineChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'TRADE_COUNT',
                    data: counts,
                    backgroundColor: 'rgba(0, 243, 255, 0.4)',
                    borderColor: 'rgba(0, 243, 255, 0.8)',
                    borderWidth: 1,
                    yAxisID: 'y',
                },
                {
                    label: 'VOLUME_USD',
                    data: volumes,
                    type: 'line',
                    borderColor: '#ff00ff',
                    backgroundColor: 'rgba(255, 0, 255, 0.1)',
                    borderWidth: 2,
                    pointRadius: 2,
                    pointBackgroundColor: '#ff00ff',
                    fill: true,
                    yAxisID: 'y1',
                }
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { labels: { color: '#00f3ff', font: { family: "'Share Tech Mono', monospace", size: 10 } } },
            },
            scales: {
                x: { ticks: { color: '#39ff14', font: { family: "'Share Tech Mono'", size: 9 }, maxRotation: 45 }, grid: { color: 'rgba(0, 243, 255, 0.05)' } },
                y: { position: 'left', ticks: { color: '#00f3ff', font: { family: "'Share Tech Mono'", size: 10 } }, grid: { color: 'rgba(0, 243, 255, 0.08)' } },
                y1: { position: 'right', ticks: { color: '#ff00ff', font: { family: "'Share Tech Mono'", size: 10 } }, grid: { drawOnChartArea: false } },
            },
        },
    });
}

// ── Wallet & Market Tables ───────────────────────────────────────

async function loadWallets() {
    const data = await fetchJSON('/api/wallets/top?limit=15');
    if (!data || !data.data) return;
    const maxVol = Math.max(...data.data.map(w => w.total_volume || 1));
    document.getElementById('wallets-table').innerHTML = `
        <table class="data-table"><thead><tr>
            <th>#</th><th>ADDRESS</th><th>TRADES</th><th>VOLUME</th><th>MARKETS</th><th>VOL_BAR</th>
        </tr></thead><tbody>
            ${data.data.map((w, i) => `<tr>
                <td class="text-dim">${i + 1}</td>
                <td class="addr">${truncAddr(w.address)}</td>
                <td>${formatNumber(w.trade_count)}</td>
                <td class="text-green">${formatVolume(w.total_volume)}</td>
                <td>${w.markets_traded}</td>
                <td><span class="vol-bar" style="width: ${(w.total_volume / maxVol * 80)}px"></span></td>
            </tr>`).join('')}
        </tbody></table>`;
}

async function loadMarkets() {
    const data = await fetchJSON('/api/markets/overview?limit=15');
    if (!data || !data.data) return;
    const maxVol = Math.max(...data.data.map(m => m.total_volume || 1));
    document.getElementById('markets-table').innerHTML = `
        <table class="data-table"><thead><tr>
            <th>#</th><th>MARKET_ID</th><th>TRADES</th><th>VOLUME</th><th>TRADERS</th><th>VOL_BAR</th>
        </tr></thead><tbody>
            ${data.data.map((m, i) => `<tr>
                <td class="text-dim">${i + 1}</td>
                <td class="addr">${truncAddr(m.market_id)}</td>
                <td>${formatNumber(m.trade_count)}</td>
                <td class="text-magenta">${formatVolume(m.total_volume)}</td>
                <td>${m.unique_traders}</td>
                <td><span class="vol-bar" style="width: ${(m.total_volume / maxVol * 80)}px; background: var(--magenta); box-shadow: 0 0 4px var(--magenta);"></span></td>
            </tr>`).join('')}
        </tbody></table>`;
}

// ── Model Comparison (Academic Metrics) ──────────────────────────

async function loadComparison() {
    const data = await fetchJSON('/api/comparison-data');
    if (!data || !data.models || data.models.length === 0) {
        document.getElementById('comparison-table').innerHTML =
            '<div class="log-placeholder">No model results yet. Click TRAIN_ALL_MODELS to begin.</div>';
        return;
    }

    // Full academic metrics table
    const html = `
        <table class="data-table">
            <thead><tr>
                <th>MODEL</th><th>ACC</th><th>PREC</th><th>REC</th>
                <th>F1</th><th>AUC-ROC</th><th>AVG_PR</th>
                <th>LOG_LOSS</th><th>BRIER</th><th>CV_AUC</th><th>TIME</th>
            </tr></thead>
            <tbody>
                ${data.models.map(m => `<tr>
                    <td class="text-cyan">${m.name}</td>
                    <td>${(m.accuracy || 0).toFixed(4)}</td>
                    <td>${(m.precision || 0).toFixed(4)}</td>
                    <td>${(m.recall || 0).toFixed(4)}</td>
                    <td class="text-green">${(m.f1 || 0).toFixed(4)}</td>
                    <td class="text-magenta">${(m.auc_roc || 0).toFixed(4)}</td>
                    <td>${(m.avg_precision || 0).toFixed(4)}</td>
                    <td>${(m.log_loss || 0).toFixed(4)}</td>
                    <td>${(m.brier_score || 0).toFixed(4)}</td>
                    <td>${(m.cv_auc || 0).toFixed(4)}</td>
                    <td class="text-dim">${(m.train_time || 0).toFixed(1)}s</td>
                </tr>`).join('')}
            </tbody>
        </table>`;
    document.getElementById('comparison-table').innerHTML = html;

    // Chart: AUC-ROC comparison
    const ctx = document.getElementById('model-chart').getContext('2d');
    if (modelChart) modelChart.destroy();

    const colors = ['#00f3ff', '#ff00ff', '#39ff14', '#ff6600', '#ffff00'];
    const names = data.models.map(m => m.name);
    const aucs = data.models.map(m => m.auc_roc || 0);

    modelChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: names,
            datasets: [{
                label: 'AUC-ROC',
                data: aucs,
                backgroundColor: names.map((_, i) => colors[i % colors.length] + '66'),
                borderColor: names.map((_, i) => colors[i % colors.length]),
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#00f3ff', font: { family: "'Share Tech Mono'", size: 10 } } },
            },
            scales: {
                x: { ticks: { color: '#39ff14', font: { family: "'Share Tech Mono'", size: 9 } }, grid: { color: 'rgba(0, 243, 255, 0.05)' } },
                y: {
                    min: 0, max: 1,
                    ticks: { color: '#00f3ff', font: { family: "'Share Tech Mono'", size: 10 } },
                    grid: { color: 'rgba(0, 243, 255, 0.08)' },
                },
            },
        },
    });
}

// ── Visuals ──────────────────────────────────────────────────────

async function loadVisuals() {
    const data = await fetchJSON('/api/results');
    if (!data || !data.files || data.files.length === 0) {
        document.getElementById('visuals-grid').innerHTML =
            '<div class="log-placeholder" style="grid-column:1/-1;">No visual outputs generated yet.</div>';
        return;
    }
    document.getElementById('visuals-grid').innerHTML = data.files.map(file => `
        <div class="visual-card">
            <div class="visual-card-header">
                <span class="visual-card-title">${file.split('.')[0]}</span>
                <span class="visual-card-badge">RENDER_OK</span>
            </div>
            <img src="/plots/${file}" alt="${file}" loading="lazy">
        </div>
    `).join('');
}

// ── Control Panel Actions ────────────────────────────────────────

async function triggerScrape(mode) {
    const btns = document.querySelectorAll('#tab-control .btn');
    btns.forEach(b => b.disabled = true);
    try {
        const res = await fetch(`${API}/api/scrape?mode=${mode}`, { method: 'POST' });
        const data = await res.json();
        appendPipelineLog(`[CMD] ${data.message || 'Scrape triggered'}`);
        startPipelinePolling();
    } catch (e) {
        appendPipelineLog(`[ERR] Failed to trigger scrape: ${e.message}`);
    }
    setTimeout(() => btns.forEach(b => b.disabled = false), 3000);
}

async function triggerPipeline() {
    const btns = document.querySelectorAll('#tab-control .btn');
    btns.forEach(b => b.disabled = true);
    try {
        const res = await fetch(`${API}/api/pipeline`, { method: 'POST' });
        const data = await res.json();
        appendPipelineLog(`[CMD] ${data.message || 'Pipeline triggered'}`);
        startPipelinePolling();
    } catch (e) {
        appendPipelineLog(`[ERR] Failed to trigger pipeline: ${e.message}`);
    }
    setTimeout(() => btns.forEach(b => b.disabled = false), 3000);
}

async function triggerBacktest() {
    const btns = document.querySelectorAll('#tab-control .btn');
    btns.forEach(b => b.disabled = true);
    try {
        const res = await fetch(`${API}/api/run-backtest?model=xgboost`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        appendPipelineLog(`[CMD] ${data.message || 'Backtest triggered'}`);
        startBacktestPolling();
    } catch (e) {
        appendPipelineLog(`[ERR] Failed to trigger backtest: ${e.message}`);
    }
    setTimeout(() => btns.forEach(b => b.disabled = false), 3000);
}

// ── Pipeline Status Polling ──────────────────────────────────────

function startPipelinePolling() {
    if (pipelinePollingId) return;
    pipelinePollingId = setInterval(pollPipelineStatus, 2000);
}

async function pollPipelineStatus() {
    const data = await fetchJSON('/api/pipeline/status');
    if (!data) return;

    const badge = document.getElementById('pipeline-badge');
    const progress = document.getElementById('pipeline-progress');
    const message = document.getElementById('pipeline-message');

    badge.textContent = data.status.toUpperCase();
    progress.style.width = (data.progress || 0) + '%';
    message.textContent = data.message || 'No active pipeline.';

    if (data.running) {
        badge.className = 'badge badge-running';
    } else {
        badge.className = data.status === 'error' ? 'badge badge-magenta' : 'badge badge-green';
        if (pipelinePollingId) {
            clearInterval(pipelinePollingId);
            pipelinePollingId = null;
        }
        loadStats();
        loadLogs();
    }
}

function appendPipelineLog(msg) {
    const container = document.getElementById('pipeline-log-container');
    const line = document.createElement('div');
    line.className = 'log-line';
    line.textContent = `> [${new Date().toLocaleTimeString()}] ${msg}`;
    container.appendChild(line);
    container.scrollTop = container.scrollHeight;
}

// ── Backtest Status Polling ─────────────────────────────────────

function startBacktestPolling() {
    if (backtestPollingId) return;
    backtestPollingId = setInterval(pollBacktestStatus, 2000);
}

async function pollBacktestStatus() {
    const data = await fetchJSON('/api/backtest/status');
    if (!data) return;

    const badge = document.getElementById('pipeline-badge');
    const progress = document.getElementById('pipeline-progress');
    const message = document.getElementById('pipeline-message');

    badge.textContent = data.status.toUpperCase();
    progress.style.width = (data.progress || 0) + '%';
    message.textContent = data.message || 'No active backtest.';

    if (data.running) {
        badge.className = 'badge badge-running';
    } else {
        badge.className = data.status === 'error' ? 'badge badge-magenta' : 'badge badge-green';
        if (backtestPollingId) {
            clearInterval(backtestPollingId);
            backtestPollingId = null;
        }
        appendPipelineLog(`[BACKTEST] ${data.message || data.status}`);
        loadVisuals();
    }
}

// ══════════════════════════════════════════════════════════════════
// ML Training Controls
// ══════════════════════════════════════════════════════════════════

async function triggerMLTrain() {
    const btn = document.getElementById('btn-ml-train');
    btn.disabled = true;

    try {
        const res = await fetch(`${API}/api/ml/train`, { method: 'POST' });
        const data = await res.json();

        if (res.ok) {
            document.getElementById('ml-message').textContent = data.message || 'Training started...';
            document.getElementById('ml-badge').textContent = 'TRAINING';
            document.getElementById('ml-badge').className = 'badge badge-running';
            startMLPolling();
        } else {
            document.getElementById('ml-message').textContent = data.detail || 'Failed to start';
            document.getElementById('ml-badge').textContent = 'ERROR';
            document.getElementById('ml-badge').className = 'badge badge-magenta';
            btn.disabled = false;
        }
    } catch (e) {
        document.getElementById('ml-message').textContent = `Error: ${e.message}`;
        btn.disabled = false;
    }
}

function startMLPolling() {
    if (mlPollingId) return;
    mlPollingId = setInterval(pollMLStatus, 2000);
}

async function pollMLStatus() {
    const data = await fetchJSON('/api/ml/status');
    if (!data) return;

    const badge = document.getElementById('ml-badge');
    const progress = document.getElementById('ml-progress');
    const message = document.getElementById('ml-message');

    badge.textContent = data.status.toUpperCase();
    progress.style.width = (data.progress || 0) + '%';
    message.textContent = data.message || 'No active ML training.';

    if (data.running) {
        badge.className = 'badge badge-running';
    } else {
        // Training finished
        badge.className = data.status === 'error' ? 'badge badge-magenta' : 'badge badge-green';
        if (mlPollingId) {
            clearInterval(mlPollingId);
            mlPollingId = null;
        }
        document.getElementById('btn-ml-train').disabled = false;

        // Reload results
        if (data.status === 'done') {
            loadComparison();
            loadVisuals(); // New plots will appear
        }
    }
}

// Make trigger functions globally accessible
window.triggerScrape = triggerScrape;
window.triggerPipeline = triggerPipeline;
window.triggerBacktest = triggerBacktest;
window.triggerMLTrain = triggerMLTrain;

// ── Initialization ───────────────────────────────────────────────

async function init() {
    loadStats();
    loadLogs();
    setInterval(() => { loadStats(); loadLogs(); }, 10000);
}

init();
