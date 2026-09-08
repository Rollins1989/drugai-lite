// ---------- nav ----------
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('view-' + btn.dataset.view).classList.add('active');
    if (btn.dataset.view === 'about') loadAbout();
  });
});

// ---------- helpers ----------
function badgeClass(label) {
  const l = (label || '').toLowerCase();
  if (['good', 'high', 'pass'].includes(l)) return 'badge-good';
  if (['moderate'].includes(l)) return 'badge-moderate';
  if (['poor', 'low', 'fail'].includes(l)) return 'badge-poor';
  return '';
}

function svgImg(b64) {
  return `<img src="data:image/svg+xml;base64,${b64}" alt="structure">`;
}

// ---------- analyze ----------
const analyzeForm = document.getElementById('analyze-form');
const analyzeResult = document.getElementById('analyze-result');

document.querySelectorAll('.chip[data-smi]').forEach(chip => {
  chip.addEventListener('click', () => {
    document.getElementById('smiles-input').value = chip.dataset.smi;
    analyzeForm.dispatchEvent(new Event('submit'));
  });
});

analyzeForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const smiles = document.getElementById('smiles-input').value.trim();
  if (!smiles) return;
  analyzeResult.classList.remove('hidden');
  analyzeResult.innerHTML = '<div class="loading">Computing descriptors and running ensemble models…</div>';
  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles })
    });
    if (!res.ok) {
      const err = await res.json();
      analyzeResult.innerHTML = `<div class="error-box">${err.detail || 'Could not parse this SMILES string.'}</div>`;
      return;
    }
    const data = await res.json();
    renderAnalysis(data);
  } catch (err) {
    analyzeResult.innerHTML = `<div class="error-box">Request failed: ${err}</div>`;
  }
});

function renderAnalysis(d) {
  const desc = d.descriptors;
  const descRows = Object.entries(desc).map(([k, v]) => `<div>${k}: <b>${v}</b></div>`).join('');
  const violations = d.lipinski.violations.length
    ? `<div class="violations">Violations: ${d.lipinski.violations.join(', ')}</div>` : '';
  const explainRows = d.explanation.map(e => `<li>${e}</li>`).join('');

  analyzeResult.innerHTML = `
    <div class="mol-card">
      <div>
        <div class="mol-struct">${svgImg(d.structure_svg_b64)}</div>
        <div class="overall-score">
          <div class="num">${d.overall_score}</div>
          <div class="lbl">Overall composite score</div>
        </div>
        <div class="section-label">Lipinski rule of five</div>
        <span class="badge ${badgeClass(d.lipinski.pass ? 'pass' : 'fail')}">${d.lipinski.pass ? 'PASS' : 'FAIL'} (${d.lipinski.n_violations} violation${d.lipinski.n_violations===1?'':'s'})</span>
        ${violations}
      </div>
      <div>
        <div class="metric-grid">
          <div class="metric">
            <div class="metric-label">Predicted solubility</div>
            <div class="metric-value">${d.solubility.log_solubility_mol_per_L} <span class="badge ${badgeClass(d.solubility.label)}">${d.solubility.label}</span></div>
          </div>
          <div class="metric">
            <div class="metric-label">Confidence</div>
            <div class="metric-value"><span class="badge ${badgeClass(d.solubility.confidence)}">${d.solubility.confidence}</span></div>
          </div>
          <div class="metric">
            <div class="metric-label">Toxicity probability (${d.toxicity.assay})</div>
            <div class="metric-value">${(d.toxicity.toxicity_probability*100).toFixed(1)}% <span class="badge ${badgeClass(d.toxicity.label)}">${d.toxicity.label}</span></div>
          </div>
          <div class="metric">
            <div class="metric-label">Confidence</div>
            <div class="metric-value"><span class="badge ${badgeClass(d.toxicity.confidence)}">${d.toxicity.confidence}</span></div>
          </div>
        </div>

        <div class="section-label">Why (top descriptor weights driving the model)</div>
        <ul class="explain-list">${explainRows}</ul>

        <div class="section-label">Molecular descriptors</div>
        <div class="desc-table">${descRows}</div>
      </div>
    </div>
  `;
}

// ---------- screening ----------
const screenForm = document.getElementById('screen-form');
const screenResult = document.getElementById('screen-result');

document.getElementById('sample-lib-btn').addEventListener('click', async () => {
  const sample = [
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "C1=CC2=C(C=C1O)C(=O)C3=C(O2)C=C(C=C3)O", "CC(=O)Nc1ccc(O)cc1", "CN1CCC[C@H]1c1cccnc1",
    "COc1cc2c(cc1OC)C(=O)C(CC1CCN(C)CC1)C2", "CC1=CC(=O)C=CC1=O", "O=C(O)c1ccccc1",
    "CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC", "Clc1ccccc1", "CC(C)NCC(O)COc1cccc2ccccc12",
    "COc1ccc2[nH]c(nc2c1)S(=O)Cc1ncc(C)c(OC)c1C", "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "Oc1ccc(cc1)C1CCNCC1", "CC12CCC3c4ccc(O)cc4CCC3C1CCC2O", "CN(C)CCC=C1c2ccccc2CCc2ccccc21",
    "CCCCCCCCCCCCCCCC(=O)O", "OC(=O)c1ccccc1O", "Nc1ccc(cc1)S(=O)(=O)Nc1nccs1", "CC(C)Cc1ccccc1"
  ].join('\n');
  const blob = new Blob([sample], { type: 'text/plain' });
  const file = new File([blob], 'sample_library.smi', { type: 'text/plain' });
  const dt = new DataTransfer();
  dt.items.add(file);
  document.getElementById('screen-file').files = dt.files;
  screenForm.dispatchEvent(new Event('submit'));
});

screenForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById('screen-file');
  if (!fileInput.files.length) return;
  screenResult.classList.remove('hidden');
  screenResult.innerHTML = '<div class="loading">Running screening funnel: validity → Lipinski → ML scoring → ranking…</div>';
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  try {
    const res = await fetch('/api/screen', { method: 'POST', body: fd });
    const data = await res.json();
    renderScreen(data);
  } catch (err) {
    screenResult.innerHTML = `<div class="error-box">Request failed: ${err}</div>`;
  }
});

function renderScreen(d) {
  const f = d.funnel;
  const funnelHtml = `
    <div class="funnel">
      <div class="funnel-step"><div class="n">${f.submitted}</div><div class="lbl">Submitted</div></div>
      <div class="funnel-step"><div class="n">${f.valid_structures}</div><div class="lbl">Valid structures</div></div>
      <div class="funnel-step"><div class="n">${f.passed_lipinski}</div><div class="lbl">Passed Lipinski</div></div>
      <div class="funnel-step"><div class="n">${f.scored}</div><div class="lbl">ML scored</div></div>
      <div class="funnel-step"><div class="n">${f.top_candidates}</div><div class="lbl">Top candidates shown</div></div>
    </div>
  `;
  const cards = d.candidates.map((c, i) => `
    <div class="cand-card">
      <div class="cand-struct">${svgImg(c.structure_svg_b64)}</div>
      <div>
        <div class="cand-smiles"><span class="rank-num">#${i+1}</span>${c.smiles}</div>
        <div class="cand-tags">
          <span class="badge ${badgeClass(c.solubility.label)}">Solubility ${c.solubility.label}</span>
          <span class="badge ${badgeClass(c.toxicity.label)}">Toxicity ${c.toxicity.label}</span>
          <span class="badge ${badgeClass(c.solubility.confidence)}">Confidence ${c.solubility.confidence}</span>
        </div>
      </div>
      <div class="cand-score">
        <div class="num">${c.overall_score}</div>
        <div class="lbl">score</div>
      </div>
    </div>
  `).join('');
  screenResult.innerHTML = funnelHtml + `<div class="candidate-list">${cards || '<div class="loading">No candidates passed the funnel.</div>'}</div>`;
}

// ---------- about / model cards ----------
async function loadAbout() {
  const box = document.getElementById('about-content');
  box.innerHTML = '<div class="loading">Loading model cards…</div>';
  const res = await fetch('/api/health');
  const data = await res.json();
  const m = data.metrics;
  box.innerHTML = `
    <div class="model-card">
      <h3>Solubility model — Gradient Boosting + Random Forest ensemble</h3>
      <div class="kv">
        <div>Dataset</div><div>${m.solubility.dataset}</div>
        <div>Train / test split</div><div>${m.solubility.n_train} / ${m.solubility.n_test} molecules</div>
        <div>Ensemble R²</div><div>${m.solubility.ensemble_r2}</div>
        <div>Ensemble RMSE</div><div>${m.solubility.ensemble_rmse} log units</div>
        <div>Individual models</div><div>RF R² ${m.solubility.rf_r2} · GB R² ${m.solubility.gb_r2}</div>
      </div>
    </div>
    <div class="model-card">
      <h3>Toxicity model — Gradient Boosting + Random Forest ensemble</h3>
      <div class="kv">
        <div>Dataset</div><div>${m.toxicity.dataset}</div>
        <div>Train / test split</div><div>${m.toxicity.n_train} / ${m.toxicity.n_test} molecules</div>
        <div>Ensemble ROC-AUC</div><div>${m.toxicity.ensemble_auc}</div>
        <div>Ensemble accuracy</div><div>${m.toxicity.ensemble_accuracy}</div>
        <div>Individual models</div><div>RF AUC ${m.toxicity.rf_auc} · GB AUC ${m.toxicity.gb_auc}</div>
      </div>
    </div>
    <div class="scope-box">
      <h3>What's real in this build vs. roadmap</h3>
      <div class="real">
        <b style="color:var(--good)">Real, working, evaluated:</b>
        <ul>
          <li>RDKit descriptor computation + 2D structure rendering</li>
          <li>Trained solubility &amp; toxicity ensemble models, on public MoleculeNet data</li>
          <li>Model-disagreement based confidence (RF vs GB spread)</li>
          <li>Lipinski / QED drug-likeness scoring</li>
          <li>Batch virtual screening funnel with real filtering at every stage</li>
        </ul>
      </div>
      <div class="roadmap">
        <b style="color:var(--moderate)">Explicitly out of scope for this build (roadmap):</b>
        <ul>
          <li>Protein–ligand docking &amp; 3D binding pose prediction</li>
          <li>GNN activity models trained on target-specific bioactivity (needs BindingDB/ChEMBL-scale training)</li>
          <li>Biomedical knowledge graph / literature RAG</li>
          <li>Multi-tenant auth, billing, job queues</li>
          <li>Molecular generation</li>
        </ul>
      </div>
    </div>
  `;
}
