/* TelePredict — main.js */

const SEMI = 128.5; // half-circumference of gauge arc

document.getElementById("pForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = document.getElementById("predictBtn");
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Predicting…';

  try {
    const fd  = new FormData(e.target);
    const res = await fetch("/predict", { method: "POST", body: fd });
    const d   = await res.json();

    if (d.error) { alert(d.error); return; }

    // Show result panel
    document.getElementById("resultIdle").style.display = "none";
    document.getElementById("resultLive").style.display = "block";

    // Animate gauge
    const pct    = d.probability;
    const offset = SEMI - (pct / 100) * SEMI;
    const arc    = document.getElementById("gaugeArc");
    arc.style.transition = "stroke-dashoffset 1.1s cubic-bezier(.4,0,.2,1), stroke .3s";
    arc.style.stroke = d.risk_color;
    arc.style.strokeDashoffset = offset;

    // Labels
    document.getElementById("gaugePct").textContent   = pct + "%";
    document.getElementById("gaugePct").style.color    = d.risk_color;
    document.getElementById("gaugeLabel").textContent  = d.churn_label;
    document.getElementById("gaugeLabel").style.color  = d.risk_color;

    const badge = document.getElementById("riskBadge");
    badge.textContent             = d.risk_level;
    badge.style.background        = d.risk_color + "1A";
    badge.style.color             = d.risk_color;
    badge.style.borderColor       = d.risk_color + "55";

    document.getElementById("recBox").textContent      = d.recommendation;
    document.getElementById("verdict").textContent     = d.churn_label;
    document.getElementById("verdict").style.color     = d.risk_color;
    document.getElementById("confVal").textContent     = d.confidence + "%";

  } catch (err) {
    alert("Prediction request failed. Ensure the Flask app is running.");
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">⚡</span> Generate Prediction';
  }
});
