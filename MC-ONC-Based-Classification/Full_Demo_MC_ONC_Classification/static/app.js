/**
 * app.js
 * ======
 * Frontend Application Logic for PediaMorph AI Local Demonstration.
 * Handles photo ingestion, sample selection, dual-tier API communication,
 * and dynamic rendering of morphometric visualizations & triage scores.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const btnBrowse = document.getElementById('btn-browse');
  const dropzonePrompt = document.getElementById('dropzone-prompt');
  const dropzonePreview = document.getElementById('dropzone-preview');
  const previewImg = document.getElementById('preview-img');
  const previewFilename = document.getElementById('preview-filename');
  const btnChangeImg = document.getElementById('btn-change-img');
  const samplesContainer = document.getElementById('samples-container');
  const btnRunAnalysis = document.getElementById('btn-run-analysis');
  const pipelineProgress = document.getElementById('pipeline-progress');
  const progressFill = document.getElementById('progress-fill');

  const resultsWrapper = document.getElementById('results-wrapper');
  const triageBanner = document.getElementById('triage-banner');
  const triageIcon = document.getElementById('triage-icon');
  const triageTierName = document.getElementById('triage-tier-name');
  const triageTime = document.getElementById('triage-time');
  const triageHeadline = document.getElementById('triage-headline');
  const triageAction = document.getElementById('triage-action');
  const triageScoreVal = document.getElementById('triage-score-val');
  const scoreFillMini = document.getElementById('score-fill-mini');

  const meterNeedle = document.getElementById('meter-needle');
  const ldaProbVal = document.getElementById('lda-prob-val');
  const xgbProbVal = document.getElementById('xgb-prob-val');
  const csVal = document.getElementById('cs-val');

  const egatProbVal = document.getElementById('egat-prob-val');
  const egatLocusVal = document.getElementById('egat-locus-val');
  const egatHeatmapImg = document.getElementById('egat-heatmap-img');
  const calipersContainer = document.getElementById('calipers-container');

  const multimodalCard = document.getElementById('multimodal-card');
  const multimodalForm = document.getElementById('multimodal-form');
  const mmSex = document.getElementById('mm-sex');
  const mmMuac = document.getElementById('mm-muac');
  const btnResolve = document.getElementById('btn-resolve');
  const multimodalResult = document.getElementById('multimodal-result');
  const mmResultBanner = document.getElementById('mm-result-banner');
  const mmIcon = document.getElementById('mm-icon');
  const mmTriageTitle = document.getElementById('mm-triage-title');
  const mmTriageSub = document.getElementById('mm-triage-sub');

  const bioArmLen = document.getElementById('bio-arm-len');
  const bioUpperLen = document.getElementById('bio-upper-len');
  const bioForearmLen = document.getElementById('bio-forearm-len');
  const bioMuac = document.getElementById('bio-muac');
  const visContourImg = document.getElementById('vis-contour-img');
  const visCartesianImg = document.getElementById('vis-cartesian-img');

  const btnToggleLandmarks = document.getElementById('btn-toggle-landmarks');
  const landmarksTableContainer = document.getElementById('landmarks-table-container');
  const landmarksTbody = document.getElementById('landmarks-tbody');

  // State
  let currentFile = null;
  let currentSampleFilename = null;
  let currentResiduals36 = null;

  // 1. Fetch & Render Clinical Samples
  fetchSamples();

  async function fetchSamples() {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      renderSamples(data.samples || []);
    } catch (err) {
      samplesContainer.innerHTML = '<div class="loading-samples text-muted">Failed to load samples</div>';
    }
  }

  function renderSamples(samples) {
    if (!samples.length) {
      samplesContainer.innerHTML = '<div class="loading-samples text-muted">No sample photos available.</div>';
      return;
    }

    samplesContainer.innerHTML = '';
    samples.forEach(s => {
      const card = document.createElement('div');
      card.className = 'sample-card';
      card.title = `${s.filename} (${s.size_kb} KB)`;
      card.innerHTML = `
        <img class="sample-thumb" src="${s.url}" alt="${s.title}">
        <span class="sample-info">${s.filename}</span>
      `;
      card.addEventListener('click', () => selectSample(s, card));
      samplesContainer.appendChild(card);
    });
  }

  function selectSample(sample, cardElement) {
    document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
    cardElement.classList.add('active');

    currentSampleFilename = sample.filename;
    currentFile = null;

    previewImg.src = sample.url;
    previewFilename.textContent = sample.filename;
    dropzonePrompt.classList.add('hidden');
    dropzonePreview.classList.remove('hidden');

    btnRunAnalysis.disabled = false;
  }

  // 2. Dropzone & File Ingestion
  btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener('click', () => {
    if (!dropzonePreview.classList.contains('hidden')) return;
    fileInput.click();
  });

  btnChangeImg.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleUserFile(e.target.files[0]);
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUserFile(e.dataTransfer.files[0]);
    }
  });

  function handleUserFile(file) {
    currentFile = file;
    currentSampleFilename = null;
    document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewFilename.textContent = file.name;
      dropzonePrompt.classList.add('hidden');
      dropzonePreview.classList.remove('hidden');
      btnRunAnalysis.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    currentFile = null;
    currentSampleFilename = null;
    fileInput.value = '';
    previewImg.src = '';
    dropzonePrompt.classList.remove('hidden');
    dropzonePreview.classList.add('hidden');
    btnRunAnalysis.disabled = true;
    document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
  }

  // 3. Run Pipeline Trigger
  btnRunAnalysis.addEventListener('click', async () => {
    if (!currentFile && !currentSampleFilename) return;

    btnRunAnalysis.disabled = true;
    pipelineProgress.classList.remove('hidden');
    resultsWrapper.classList.add('hidden');
    multimodalResult.classList.add('hidden');

    animateProgressSteps();

    try {
      let response;
      if (currentSampleFilename) {
        response = await fetch(`/api/analyze-sample/${encodeURIComponent(currentSampleFilename)}`, {
          method: 'POST'
        });
      } else {
        const formData = new FormData();
        formData.append('file', currentFile);
        response = await fetch('/api/analyze', {
          method: 'POST',
          body: formData
        });
      }

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: 'Unknown error occurred.' }));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const result = await response.json();
      finishProgress();
      renderFullResults(result);

    } catch (err) {
      alert(`Pipeline Execution Failed:\n${err.message}`);
      pipelineProgress.classList.add('hidden');
      btnRunAnalysis.disabled = false;
    }
  });

  let progressInterval = null;
  function animateProgressSteps() {
    let pct = 10;
    progressFill.style.width = '10%';
    const steps = [
      document.getElementById('pstep-1'),
      document.getElementById('pstep-2'),
      document.getElementById('pstep-3'),
      document.getElementById('pstep-4'),
      document.getElementById('pstep-5'),
      document.getElementById('pstep-6')
    ];

    steps.forEach((s, idx) => {
      s.className = 'step-label' + (idx === 0 ? ' active' : '');
    });

    let currentStep = 0;
    progressInterval = setInterval(() => {
      pct += 14;
      if (pct > 92) pct = 92;
      progressFill.style.width = pct + '%';

      currentStep = Math.min(Math.floor((pct / 92) * 5), 5);
      steps.forEach((s, idx) => {
        if (idx < currentStep) s.className = 'step-label done';
        else if (idx === currentStep) s.className = 'step-label active';
        else s.className = 'step-label';
      });
    }, 400);
  }

  function finishProgress() {
    clearInterval(progressInterval);
    progressFill.style.width = '100%';
    const steps = document.querySelectorAll('.progress-steps .step-label');
    steps.forEach(s => s.className = 'step-label done');

    setTimeout(() => {
      pipelineProgress.classList.add('hidden');
      btnRunAnalysis.disabled = false;
    }, 600);
  }

  // 4. Render Results
  function renderFullResults(data) {
    const p1 = data.phase1_vision;
    const p2 = data.phase2_classification;
    const t1 = p2.tier1_mobile_ensemble;
    const t2 = p2.tier2_cloud_egat;

    currentResiduals36 = p2.residuals_36;

    // Traffic Light Banner
    triageBanner.className = 'triage-banner';
    let tierClass = 'triage-green';
    let icon = '🟢';
    let needlePct = 15;

    if (t1.triage_tier === 'RED') {
      tierClass = 'triage-red';
      icon = '🔴';
      needlePct = 50 + (t1.probability_ensemble - 0.5) * 100;
    } else if (t1.triage_tier === 'YELLOW') {
      tierClass = 'triage-yellow';
      icon = '🟡';
      needlePct = 25 + ((t1.probability_ensemble - 0.25) / 0.25) * 25;
    } else {
      needlePct = (t1.probability_ensemble / 0.25) * 25;
    }

    triageBanner.classList.add(tierClass);
    triageIcon.textContent = icon;
    triageTierName.textContent = `TIER 1 ENSEMBLE: ${t1.triage_tier} ZONE`;
    triageTime.textContent = `Analyzed in ${data.timing_sec.total}s`;
    triageHeadline.textContent = t1.nutritional_category;
    triageAction.textContent = t1.clinical_action;
    triageScoreVal.textContent = `${(t1.probability_ensemble * 100).toFixed(1)}%`;
    scoreFillMini.style.width = `${Math.min(t1.probability_ensemble * 100, 100)}%`;
    scoreFillMini.style.background = (t1.triage_tier === 'RED') ? '#ef4444' : (t1.triage_tier === 'YELLOW') ? '#f59e0b' : '#10b981';

    // Tier 1 Mobile Ensemble Card
    meterNeedle.style.left = `${Math.max(2, Math.min(needlePct, 98))}%`;
    ldaProbVal.textContent = `${(t1.probability_lda * 100).toFixed(1)}%`;
    xgbProbVal.textContent = `${(t1.probability_xgboost * 100).toFixed(1)}%`;
    csVal.textContent = p2.centroid_size.toFixed(2);

    // Tier 2 Cloud EGAT Card
    egatProbVal.textContent = `${(t2.probability_egat * 100).toFixed(1)}%`;
    egatLocusVal.textContent = t2.primary_wasting_site;
    egatHeatmapImg.src = t2.attention_heatmap_base64;

    renderCalipersList(t2.calipers_attention);

    // Borderline Yellow Multimodal Resolver Card
    if (t1.triage_tier === 'YELLOW') {
      multimodalCard.classList.remove('hidden');
      if (p1.muac_cm) {
        mmMuac.value = p1.muac_cm;
      }
    } else {
      // Keep visible as optional secondary tool, but remove pulsing emphasis
      multimodalCard.classList.remove('hidden');
    }

    // Phase 1 Morphometric Vision Card
    bioArmLen.textContent = `${p1.total_arm_cm} cm`;
    bioUpperLen.textContent = `${p1.upper_arm_cm || '--'} cm`;
    bioForearmLen.textContent = `${p1.forearm_cm || '--'} cm`;
    bioMuac.textContent = `${p1.muac_cm || '--'} cm`;

    visContourImg.src = p1.contour_overlay_base64;
    visCartesianImg.src = p1.cartesian_plot_base64;

    renderLandmarksTable(p1.landmarks);

    // Show Results with smooth reveal
    resultsWrapper.classList.remove('hidden');
    resultsWrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderCalipersList(calipers) {
    calipersContainer.innerHTML = '';
    calipers.forEach(c => {
      const row = document.createElement('div');
      row.className = 'caliper-row' + (c.is_muac_site ? ' muac-row' : '');
      row.innerHTML = `
        <span class="caliper-name">${c.caliper_id}. ${c.name} (${c.pair}) ${c.is_muac_site ? '⭐' : ''}</span>
        <span class="caliper-weight">${c.attention_weight.toFixed(4)}</span>
      `;
      calipersContainer.appendChild(row);
    });
  }

  function renderLandmarksTable(landmarks) {
    landmarksTbody.innerHTML = '';
    landmarks.forEach(lm => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong style="color: ${lm.Color === 'red' ? '#ef4444' : '#eab308'}">L${lm.Landmark_ID}</strong></td>
        <td>${lm.Name}</td>
        <td>${lm.Category}</td>
        <td>${lm.X_cartesian_cm.toFixed(3)}</td>
        <td>${lm.Y_cartesian_cm.toFixed(3)}</td>
        <td>(${Math.round(lm.X_Pixel)}, ${Math.round(lm.Y_Pixel)})</td>
      `;
      landmarksTbody.appendChild(tr);
    });
  }

  // Accordion Toggle
  btnToggleLandmarks.addEventListener('click', () => {
    landmarksTableContainer.classList.toggle('hidden');
    btnToggleLandmarks.querySelector('.chevron-icon').classList.toggle('rotate');
  });

  // Multimodal Secondary Confirmation
  multimodalForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!currentResiduals36 || currentResiduals36.length !== 36) {
      alert('Shape residuals not available. Please run arm photo analysis first.');
      return;
    }

    const muacVal = parseFloat(mmMuac.value);
    if (isNaN(muacVal) || muacVal <= 0) {
      alert('Please enter a valid tape MUAC measurement (cm).');
      return;
    }

    btnResolve.disabled = true;
    btnResolve.textContent = 'Resolving...';

    try {
      const res = await fetch('/api/multimodal-confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          residuals_36: currentResiduals36,
          gender: mmSex.value,
          tape_muac_cm: muacVal
        })
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const mmData = await res.json();
      renderMultimodalResult(mmData);
    } catch (err) {
      alert(`Multimodal confirmation failed: ${err.message}`);
    } finally {
      btnResolve.disabled = false;
      btnResolve.textContent = 'Definitively Resolve Case';
    }
  });

  function renderMultimodalResult(data) {
    multimodalResult.classList.remove('hidden');
    const isMal = data.multimodal_probability >= 0.50;

    mmResultBanner.className = 'mm-result-banner ' + (isMal ? 'mm-malnourished' : 'mm-normal');
    mmIcon.textContent = isMal ? '🚨' : '✅';
    mmTriageTitle.textContent = isMal 
      ? 'Confirmed Malnourished: SAM / Urgent Referral'
      : 'Confirmed Healthy Thriving (Normal ONC)';
    
    mmTriageSub.textContent = `Multimodal 38D Probability: ${(data.multimodal_probability * 100).toFixed(1)}% • Tape MUAC: ${data.tape_muac_cm} cm • Protocol Verified (>89.7% Accuracy)`;
  }
});
