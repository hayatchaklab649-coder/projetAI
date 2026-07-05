const state = {
  summary: null,
  dataset: [],
  metrics: null,
};

const colors = {
  positif: "#168a63",
  neutre: "#c89221",
  negatif: "#d2554c",
};

function qs(selector) {
  return document.querySelector(selector);
}

function qsa(selector) {
  return [...document.querySelectorAll(selector)];
}

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Erreur inconnue");
  }
  return data;
}

function setStatus(text) {
  qs("#status").textContent = text;
}

function activateTab(tabId) {
  qsa(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === tabId));
  qsa(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === tabId));
}

function drawBarChart(canvas, data, titleFormatter = (label) => label) {
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.width = canvas.clientWidth * ratio;
  const height = canvas.height = canvas.clientHeight * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const displayWidth = width / ratio;
  const displayHeight = height / ratio;
  ctx.clearRect(0, 0, displayWidth, displayHeight);

  const entries = Object.entries(data);
  const maxValue = Math.max(...entries.map(([, value]) => value), 1);
  const padding = 34;
  const gap = 18;
  const barWidth = (displayWidth - padding * 2 - gap * (entries.length - 1)) / entries.length;

  ctx.textAlign = "center";
  ctx.strokeStyle = "#e7eef4";
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i += 1) {
    const y = 24 + i * ((displayHeight - 86) / 3);
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(displayWidth - padding, y);
    ctx.stroke();
  }

  entries.forEach(([label, value], index) => {
    const x = padding + index * (barWidth + gap);
    const barHeight = (displayHeight - 80) * (value / maxValue);
    const y = displayHeight - 44 - barHeight;
    ctx.fillStyle = colors[label] || "#3b6ea8";
    roundRect(ctx, x, y, barWidth, barHeight, 7);
    ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,0.2)";
    roundRect(ctx, x + 5, y + 5, Math.max(4, barWidth - 10), 5, 4);
    ctx.fill();
    ctx.fillStyle = "#1f2623";
    ctx.font = "700 14px system-ui";
    ctx.fillText(String(value), x + barWidth / 2, y - 8);
    ctx.fillStyle = "#66716d";
    ctx.font = "13px system-ui";
    ctx.fillText(titleFormatter(label), x + barWidth / 2, displayHeight - 18);
  });
}

function roundRect(ctx, x, y, width, height, radius) {
  const safeRadius = Math.min(radius, width / 2, Math.max(0, height / 2));
  ctx.beginPath();
  ctx.moveTo(x + safeRadius, y);
  ctx.lineTo(x + width - safeRadius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + safeRadius);
  ctx.lineTo(x + width, y + height - safeRadius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - safeRadius, y + height);
  ctx.lineTo(x + safeRadius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - safeRadius);
  ctx.lineTo(x, y + safeRadius);
  ctx.quadraticCurveTo(x, y, x + safeRadius, y);
}

function renderSummary() {
  const summary = state.summary;
  qs("#rows").textContent = summary.rows;
  qs("#avgWords").textContent = summary.avg_words;
  qs("#avgChars").textContent = summary.avg_chars;
  qs("#classCount").textContent = Object.keys(summary.sentiment_distribution).length;
  drawBarChart(qs("#sentimentChart"), summary.sentiment_distribution);
  drawBarChart(qs("#lengthChart"), summary.length_by_sentiment);
  renderSources();
  renderWords("all");
}

function renderSources() {
  const maxValue = Math.max(...Object.values(state.summary.source_distribution), 1);
  qs("#sourceList").innerHTML = Object.entries(state.summary.source_distribution)
    .sort((a, b) => b[1] - a[1])
    .map(([source, count]) => `
      <div class="source-row">
        <span>${source}</span>
        <div class="bar"><span style="width:${(count / maxValue) * 100}%; background:#2f6fb5"></span></div>
        <strong>${count}</strong>
      </div>
    `).join("");
}

function renderWords(filter) {
  const source = filter === "all"
    ? state.summary.top_words
    : state.summary.top_words_by_sentiment[filter];
  qs("#wordCloud").innerHTML = source.map((item) => {
    const size = Math.min(22, 12 + item.count * 2);
    return `<span class="word-chip" style="font-size:${size}px">${item.word} <small>${item.count}</small></span>`;
  }).join("");
}

function sentimentBadge(sentiment) {
  return `<span class="sentiment-pill sentiment-${sentiment}">${sentiment}</span>`;
}

function renderDataset(records = state.dataset) {
  qs("#datasetBody").innerHTML = records.map((row) => `
    <tr>
      <td>${row.id}</td>
      <td>${row.text}</td>
      <td>${sentimentBadge(row.sentiment)}</td>
      <td>${row.source}</td>
      <td>${row.word_count}</td>
    </tr>
  `).join("");
}

function renderMetrics() {
  const metrics = state.metrics;
  qs("#accuracy").textContent = `${Math.round(metrics.accuracy * 100)} %`;
  qs("#accuracyRing").style.setProperty("--score-angle", `${metrics.accuracy * 360}deg`);
  qs("#trainSize").textContent = metrics.train_size;
  qs("#testSize").textContent = metrics.test_size;

  const labelHeader = ["", ...metrics.labels];
  const rows = [labelHeader, ...metrics.confusion_matrix.map((row, index) => [metrics.labels[index], ...row])];
  qs("#confusionMatrix").innerHTML = rows.map((row, rowIndex) => `
    <div class="matrix-row">
      ${row.map((cell, cellIndex) => `
        <div class="matrix-cell ${rowIndex === 0 || cellIndex === 0 ? "matrix-label" : ""}">${cell}</div>
      `).join("")}
    </div>
  `).join("");

  qs("#reportBody").innerHTML = metrics.labels.map((label) => {
    const row = metrics.classification_report[label];
    return `
      <tr>
        <td>${sentimentBadge(label)}</td>
        <td>${row.precision.toFixed(2)}</td>
        <td>${row.recall.toFixed(2)}</td>
        <td>${row["f1-score"].toFixed(2)}</td>
        <td>${row.support}</td>
      </tr>
    `;
  }).join("");
}

function renderPrediction(result) {
  const scores = Object.entries(result.scores).map(([label, value]) => `
    <div class="score-row">
      <span>${label}</span>
      <div class="bar"><span style="width:${value * 100}%; background:${colors[label] || "#3b6ea8"}"></span></div>
      <strong>${Math.round(value * 100)}%</strong>
    </div>
  `).join("");

  qs("#predictionResult").classList.remove("empty");
  qs("#predictionResult").innerHTML = `
    <p>Sentiment predit : ${sentimentBadge(result.sentiment)}</p>
    <p>Confiance : <strong>${Math.round(result.confidence * 100)}%</strong></p>
    <p>Texte nettoye : <code>${result.clean_text || "aucun mot significatif"}</code></p>
    <div class="score-bars">${scores}</div>
  `;
}

async function predict() {
  const text = qs("#textInput").value.trim();
  if (!text) {
    qs("#predictionResult").className = "result empty";
    qs("#predictionResult").textContent = "Le texte est obligatoire.";
    return;
  }
  qs("#predictionResult").className = "result empty";
  qs("#predictionResult").textContent = "Analyse en cours...";
  const result = await getJson("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  renderPrediction(result);
}

function bindEvents() {
  qsa(".tab").forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.tab));
  });
  qs("#wordFilter").addEventListener("change", (event) => renderWords(event.target.value));
  qs("#predictButton").addEventListener("click", predict);
  qs("#clearButton").addEventListener("click", () => {
    qs("#textInput").value = "";
    qs("#predictionResult").className = "result empty";
    qs("#predictionResult").textContent = "Saisis un avis puis lance l'analyse.";
  });
  qsa(".example").forEach((button) => {
    button.addEventListener("click", () => {
      qs("#textInput").value = button.dataset.text;
      predict();
    });
  });
  qs("#searchInput").addEventListener("input", (event) => {
    const query = event.target.value.toLowerCase();
    const filtered = state.dataset.filter((row) =>
      row.text.toLowerCase().includes(query) ||
      row.sentiment.toLowerCase().includes(query) ||
      row.source.toLowerCase().includes(query)
    );
    renderDataset(filtered);
  });
  window.addEventListener("resize", () => {
    if (state.summary) {
      drawBarChart(qs("#sentimentChart"), state.summary.sentiment_distribution);
      drawBarChart(qs("#lengthChart"), state.summary.length_by_sentiment);
    }
  });
}

async function init() {
  bindEvents();
  try {
    const [summary, dataset, metrics] = await Promise.all([
      getJson("/api/summary"),
      getJson("/api/dataset"),
      getJson("/api/metrics"),
    ]);
    state.summary = summary;
    state.dataset = dataset.records;
    state.metrics = metrics;
    renderSummary();
    renderDataset();
    renderMetrics();
    setStatus("Pret");
  } catch (error) {
    setStatus("Erreur");
    console.error(error);
  }
}

init();
