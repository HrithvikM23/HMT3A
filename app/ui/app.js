// Kinara Web Launcher Application Logic — Powered by UI/UX Pro Max & PyWebView Bridge
document.addEventListener('DOMContentLoaded', () => {

  // State Management
  const state = {
    sources: [],
    destination: '',
    peopleCount: 1,
    personColors: {},
    activeTab: 'presets',
    status: 'Idle',
    isCustomCommand: false,
    cameraFrames: {},
    workerFrames: {},
    activeCamera: 'CAM_0',
    activeWorker: 'ALL',
  };

  // DOM Elements
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  const btnStart = document.getElementById('btnStart');
  const btnCheck = document.getElementById('btnCheck');
  const btnStop = document.getElementById('btnStop');
  const btnKill = document.getElementById('btnKill');
  const btnResetDefaults = document.getElementById('btnResetDefaults');
  const btnUseCamera = document.getElementById('btnUseCamera');
  const btnAddFiles = document.getElementById('btnAddFiles');
  const btnClearFiles = document.getElementById('btnClearFiles');
  const btnBrowseDest = document.getElementById('btnBrowseDest');
  const btnBrowseTriangulation = document.getElementById('btnBrowseTriangulation');
  const btnCopyCmd = document.getElementById('btnCopyCmd');
  const btnClearLog = document.getElementById('btnClearLog');
  const cmdInput = document.getElementById('cmdInput');
  const logBox = document.getElementById('logBox');
  const livePreviewImg = document.getElementById('livePreviewImg');
  const previewPlaceholder = document.getElementById('previewPlaceholder');
  const fileSourceList = document.getElementById('fileSourceList');
  const destPathInput = document.getElementById('destPathInput');
  const peopleCountInput = document.getElementById('peopleCountInput');
  const peopleColorContainer = document.getElementById('peopleColorContainer');
  const chkCalibrateMode = document.getElementById('chkCalibrateMode');
  const chkInstantTriangulation = document.getElementById('chkInstantTriangulation');
  const chkRescueMode = document.getElementById('chkRescueMode');
  const chkEnableTriangulation = document.getElementById('chkEnableTriangulation');
  const triangulationPathInput = document.getElementById('triangulationPathInput');
  const paperSizeSelect = document.getElementById('paperSizeSelect');

  // Tab Navigation Handling
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabName = btn.getAttribute('data-tab');
      if (!tabName) return;

      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetTab = document.getElementById(`tab-${tabName}`);
      if (targetTab) {
        targetTab.classList.add('active');
        state.activeTab = tabName;
      }
    });
  });

  // Helper: Call Python API with Error Catching
  async function callPy(methodName, ...args) {
    if (window.pywebview && window.pywebview.api && window.pywebview.api[methodName]) {
      try {
        return await window.pywebview.api[methodName](...args);
      } catch (err) {
        appendLog(`Error calling ${methodName}: ${err}`, 'error');
      }
    } else {
      console.warn(`PyWebView API missing for call: ${methodName}`);
    }
  }

  // Helper: Append Formatted Log Lines
  function appendLog(message, type = 'normal') {
    if (!message || !logBox) return;
    const lines = String(message).split(/\r?\n/);
    lines.forEach(str => {
      const trimmed = str.trimEnd();
      if (!trimmed && lines.length > 1) return;
      const line = document.createElement('div');
      line.className = `log-line ${type}`;
      const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
      line.textContent = `[${timestamp}] ${trimmed}`;
      logBox.appendChild(line);
    });
    logBox.scrollTop = logBox.scrollHeight;
  }

  // Helper: Set Real-time Status Badge
  function setStatus(text, type = 'idle') {
    state.status = text;
    const statusTextEl = document.getElementById('statusText') || document.querySelector('.status-text');
    if (statusTextEl) {
      statusTextEl.textContent = text;
    }
    const statusBadgeEl = document.getElementById('statusBadge') || document.querySelector('.status-badge');
    if (statusBadgeEl) {
      statusBadgeEl.className = `status-badge ${type}`;
    }
  }

  // Copy CLI Command to Clipboard
  btnCopyCmd?.addEventListener('click', () => {
    if (cmdInput && cmdInput.value) {
      navigator.clipboard.writeText(cmdInput.value).then(() => {
        appendLog('Command copied to clipboard!', 'system');
      }).catch(err => {
        appendLog(`Failed to copy: ${err}`, 'error');
      });
    }
  });

  // Clear Terminal Logs
  btnClearLog?.addEventListener('click', () => {
    if (logBox) {
      logBox.innerHTML = '<div class="log-line system">&gt; Log console cleared. Waiting for events...</div>';
    }
  });

  // Preset Buttons
  document.getElementById('btnApplyMediaPipe')?.addEventListener('click', async () => {
    const pose = document.getElementById('mpPoseOption')?.value || 'Full';
    const hand = document.getElementById('mpHandWeight')?.value || 'Full';
    const res = await callPy('apply_preset', 'MediaPipe', pose, hand);
    if (res) updateCommandUI(res);
    setStatus('MediaPipe Preset Applied');
    appendLog(`Applied MediaPipe preset (Pose: ${pose}, Hand: ${hand})`, 'system');
  });

  document.getElementById('btnApplyRTMPose')?.addEventListener('click', async () => {
    const pose = document.getElementById('rtmPoseOption')?.value || 'Full';
    const hand = document.getElementById('rtmHandWeight')?.value || 'Heavy';
    const res = await callPy('apply_preset', 'RTMPose', pose, hand);
    if (res) updateCommandUI(res);
    setStatus('RTMPose Preset Applied');
    appendLog(`Applied RTMPose CUDA preset (Pose: ${pose}, Hand: ${hand})`, 'system');
  });

  document.getElementById('btnApplyONNX')?.addEventListener('click', async () => {
    const yolo = document.getElementById('onnxYoloOption')?.value || 'X-Large';
    const hand = document.getElementById('onnxHandWeight')?.value || 'Heavy';
    const res = await callPy('apply_preset', 'ONNX', yolo, hand);
    if (res) updateCommandUI(res);
    setStatus('ONNX YOLO Preset Applied');
    appendLog(`Applied ONNX YOLO preset (YOLO: ${yolo}, Hand: ${hand})`, 'system');
  });

  document.getElementById('btnApplyWholeBody')?.addEventListener('click', async () => {
    const mode = document.getElementById('wholebodyOption')?.value || 'Balanced';
    const res = await callPy('apply_preset', 'RTMPose WholeBody', mode, null);
    if (res) updateCommandUI(res);
    setStatus('WholeBody Preset Applied');
    appendLog(`Applied RTMPose WholeBody preset (Mode: ${mode})`, 'system');
  });

  document.getElementById('btnCalibA3')?.addEventListener('click', async () => {
    const selectedSize = paperSizeSelect ? paperSizeSelect.value : 'A3';
    const isRescue = chkRescueMode ? chkRescueMode.checked : false;
    const res = await callPy('apply_charuco_preset', isRescue ? 'Rescue' : 'A3');
    if (selectedSize && selectedSize !== 'A3') {
      await callPy('set_paper_size', selectedSize);
    }
    const cmd = await callPy('get_initial_command');
    if (cmd) updateCommandUI(cmd);
    setStatus('Calibration Applied');
    appendLog(`Applied ChArUco Calibration parameters (${selectedSize}${isRescue ? ', Rescue Mode' : ''})`, 'system');
  });

  // Action Buttons
  btnStart?.addEventListener('click', async () => {
    setStatus('Running...', 'running');
    if (previewPlaceholder) previewPlaceholder.style.display = 'none';
    if (livePreviewImg) livePreviewImg.style.display = 'block';
    appendLog('Starting Kinara Motion Tracking Pipeline...', 'system');
    const customCmd = state.isCustomCommand ? cmdInput?.value : null;
    const result = await callPy('start_run', customCmd);
    if (result && result.log) appendLog(result.log);
  });

  btnCheck?.addEventListener('click', async () => {
    setStatus('Checking...', 'checking');
    appendLog('Running local runtime environment verification...', 'system');
    const result = await callPy('check_runtime');
    if (result && result.log) appendLog(result.log);
  });

  btnStop?.addEventListener('click', async () => {
    setStatus('Stopping...', 'idle');
    appendLog('Requesting graceful stop signal...', 'system');
    await callPy('stop_run');
  });

  btnKill?.addEventListener('click', async () => {
    setStatus('Killed', 'error');
    appendLog('Sending emergency SIGKILL to runtime process...', 'error');
    await callPy('kill_run');
  });

  btnResetDefaults?.addEventListener('click', async () => {
    state.isCustomCommand = false;
    const tuneWorkersInput = document.getElementById('tuneWorkersInput');
    if (tuneWorkersInput) tuneWorkersInput.value = 0;
    const cmd = await callPy('reset_defaults');
    if (cmd) updateCommandUI(cmd);
    if (peopleCountInput) peopleCountInput.value = 1;
    state.peopleCount = 1;
    state.sources = [];
    state.personColors = {};
    updatePeopleColors();
    updateSourceListUI();
    if (destPathInput) destPathInput.value = '';
    if (triangulationPathInput) triangulationPathInput.value = '';
    if (chkCalibrateMode) chkCalibrateMode.checked = false;
    if (chkInstantTriangulation) chkInstantTriangulation.checked = false;
    if (chkRescueMode) chkRescueMode.checked = false;
    if (chkEnableTriangulation) chkEnableTriangulation.checked = false;
    if (paperSizeSelect) paperSizeSelect.value = 'A3';
    const tuneExecutionModeSelect = document.getElementById('tuneExecutionModeSelect');
    if (tuneExecutionModeSelect) tuneExecutionModeSelect.value = 'auto';
    setStatus('Defaults Restored');
    appendLog('All configuration parameters restored to default.', 'system');
  });

  // Sources & Files
  btnUseCamera?.addEventListener('click', async () => {
    state.sources = ['0'];
    updateSourceListUI();
    const cmd = await callPy('set_sources', state.sources);
    if (cmd) updateCommandUI(cmd);
    appendLog('Set input source to Camera 0 (Webcam)', 'system');
  });

  btnAddFiles?.addEventListener('click', async () => {
    const files = await callPy('browse_files');
    if (files && files.length > 0) {
      state.sources = [...state.sources, ...files];
      updateSourceListUI();
      const cmd = await callPy('set_sources', state.sources);
      if (cmd) updateCommandUI(cmd);
      appendLog(`Added ${files.length} video source file(s)`, 'system');
    }
  });

  btnClearFiles?.addEventListener('click', async () => {
    state.sources = [];
    updateSourceListUI();
    const cmd = await callPy('set_sources', []);
    if (cmd) updateCommandUI(cmd);
    appendLog('Cleared video source list', 'system');
  });

  btnBrowseDest?.addEventListener('click', async () => {
    const path = await callPy('browse_destination');
    if (path) {
      if (destPathInput) destPathInput.value = path;
      const cmd = await callPy('set_destination', path);
      if (cmd) updateCommandUI(cmd);
      appendLog(`Output destination updated: ${path}`, 'system');
    }
  });

  btnBrowseTriangulation?.addEventListener('click', async () => {
    const path = await callPy('browse_triangulation');
    if (path) {
      if (triangulationPathInput) triangulationPathInput.value = path;
      const cmd = await callPy('set_triangulation_path', path);
      if (cmd) updateCommandUI(cmd);
      appendLog(`Triangulation calibration file set: ${path}`, 'system');
    }
  });

  // People Count & Identity Colors
  peopleCountInput?.addEventListener('input', () => {
    let val = parseInt(peopleCountInput.value) || 1;
    val = Math.max(1, Math.min(12, val));
    state.peopleCount = val;
    updatePeopleColors();
    callPy('set_people_count', val).then(cmd => { if (cmd) updateCommandUI(cmd); });
  });

  function updatePeopleColors() {
    if (!peopleColorContainer) return;
    peopleColorContainer.innerHTML = '';
    if (state.peopleCount >= 2) {
      const presets = ['black,orange', 'white,blue', 'red,black', 'green,white', 'yellow,black', 'blue,red'];
      for (let i = 1; i <= state.peopleCount; i++) {
        const div = document.createElement('div');
        div.className = 'form-group';
        const colorVal = state.personColors[`person${i}`] || presets[(i - 1) % presets.length];
        div.innerHTML = `
          <label class="form-label">Person ${i} Clothing Hint</label>
          <input type="text" class="form-control" value="${colorVal}" id="color_person${i}" placeholder="e.g. black,orange" />
        `;
        peopleColorContainer.appendChild(div);

        document.getElementById(`color_person${i}`)?.addEventListener('change', (e) => {
          state.personColors[`person${i}`] = e.target.value;
          callPy('set_person_color', `person${i}`, e.target.value).then(cmd => { if (cmd) updateCommandUI(cmd); });
        });
      }
    }
  }

  // Checkbox Event Listeners
  chkCalibrateMode?.addEventListener('change', (e) => {
    callPy('set_checkbox', 'calibrate_cameras', e.target.checked).then(cmd => { if (cmd) updateCommandUI(cmd); });
  });

  chkInstantTriangulation?.addEventListener('change', (e) => {
    callPy('set_checkbox', 'triangulate_after_calibration', e.target.checked).then(cmd => { if (cmd) updateCommandUI(cmd); });
  });

  chkRescueMode?.addEventListener('change', (e) => {
    callPy('set_checkbox', 'charuco_rescue_mode', e.target.checked).then(cmd => { if (cmd) updateCommandUI(cmd); });
    if (e.target.checked) {
      appendLog('Enabled Calibration Rescue Mode (Lenient ChArUco thresholds)', 'system');
    } else {
      appendLog('Disabled Calibration Rescue Mode', 'system');
    }
  });

  chkEnableTriangulation?.addEventListener('change', (e) => {
    callPy('set_checkbox', 'triangulate_3d', e.target.checked).then(cmd => { if (cmd) updateCommandUI(cmd); });
  });

  paperSizeSelect?.addEventListener('change', (e) => {
    callPy('set_paper_size', e.target.value).then(cmd => { if (cmd) updateCommandUI(cmd); });
    const tunePaperSizeSelect = document.getElementById('tunePaperSizeSelect');
    if (tunePaperSizeSelect && tunePaperSizeSelect.value !== e.target.value) {
      tunePaperSizeSelect.value = e.target.value;
      tunePaperSizeSelect.dispatchEvent(new Event('change'));
    }
  });

  const tunePaperSizeSelect = document.getElementById('tunePaperSizeSelect');
  tunePaperSizeSelect?.addEventListener('change', (e) => {
    callPy('set_paper_size', e.target.value).then(cmd => { if (cmd) updateCommandUI(cmd); });
    if (paperSizeSelect && paperSizeSelect.value !== e.target.value) {
      paperSizeSelect.value = e.target.value;
      paperSizeSelect.dispatchEvent(new Event('change'));
    }
  });

  const tuneBackendSelect = document.getElementById('tuneBackendSelect');
  const tuneModelVariantSelect = document.getElementById('tuneModelVariantSelect');
  const tuneHandBackendSelect = document.getElementById('tuneHandBackendSelect');
  const tuneHandWeightSelect = document.getElementById('tuneHandWeightSelect');
  const tuneRtmposeModeSelect = document.getElementById('tuneRtmposeModeSelect');
  const tuneRtmposeDeviceSelect = document.getElementById('tuneRtmposeDeviceSelect');
  const tuneProcessingWidthSelect = document.getElementById('tuneProcessingWidthSelect');
  const tuneProfileSelect = document.getElementById('tuneProfileSelect');
  const tuneFallbackChk = document.getElementById('tuneFallbackChk');
  const tuneYoloHalfChk = document.getElementById('tuneYoloHalfChk');
  const tuneExecutionModeSelect = document.getElementById('tuneExecutionModeSelect');

  const tuneMinCutoffInput = document.getElementById('tuneMinCutoffInput');
  const tuneBetaInput = document.getElementById('tuneBetaInput');
  const tuneDerivCutoffInput = document.getElementById('tuneDerivCutoffInput');

  const tuneBodyConfInput = document.getElementById('tuneBodyConfInput');
  const tuneHandDetConfInput = document.getElementById('tuneHandDetConfInput');
  const tuneHandConfInput = document.getElementById('tuneHandConfInput');
  const tuneHandBoxScaleInput = document.getElementById('tuneHandBoxScaleInput');
  const tuneSkeletonThicknessSelect = document.getElementById('tuneSkeletonThicknessSelect');

  const tunePersonBoxScaleInput = document.getElementById('tunePersonBoxScaleInput');
  const tuneTrackHoldFramesInput = document.getElementById('tuneTrackHoldFramesInput');
  const tuneTrackMatchInput = document.getElementById('tuneTrackMatchInput');

  const tuneEnableTriangulationChk = document.getElementById('tuneEnableTriangulationChk');
  const tuneTriangulationMinCamsInput = document.getElementById('tuneTriangulationMinCamsInput');
  const tuneTriangulationOutlierChk = document.getElementById('tuneTriangulationOutlierChk');
  const tuneTriangulationAlphaInput = document.getElementById('tuneTriangulationAlphaInput');
  const tuneCameraSyncInput = document.getElementById('tuneCameraSyncInput');

  const tuneCalibrateModeChk = document.getElementById('tuneCalibrateModeChk');
  const tuneSquaresXInput = document.getElementById('tuneSquaresXInput');
  const tuneSquaresYInput = document.getElementById('tuneSquaresYInput');
  const tuneSquareSizeInput = document.getElementById('tuneSquareSizeInput');
  const tuneCharucoStrictnessSelect = document.getElementById('tuneCharucoStrictnessSelect');
  const tuneCharucoSharpenChk = document.getElementById('tuneCharucoSharpenChk');

  const cameraTabsContainer = document.getElementById('cameraTabsContainer');
  const activeCamTitle = document.getElementById('activeCamTitle');

  // Helper for setting CLI options in Python host
  function syncOption(key, val) {
    callPy('set_advanced_option', key, val).then(cmd => { if (cmd) updateCommandUI(cmd); });
  }

  tuneBackendSelect?.addEventListener('change', (e) => syncOption('body_backend', e.target.value));
  tuneModelVariantSelect?.addEventListener('change', (e) => syncOption('body_model_variant', e.target.value));
  tuneHandBackendSelect?.addEventListener('change', (e) => syncOption('hand_backend', e.target.value));
  tuneHandWeightSelect?.addEventListener('change', (e) => callPy('apply_preset_weight', e.target.value).then(cmd => { if (cmd) updateCommandUI(cmd); }));
  tuneRtmposeModeSelect?.addEventListener('change', (e) => syncOption('rtmpose_mode', e.target.value));
  tuneRtmposeDeviceSelect?.addEventListener('change', (e) => syncOption('rtmpose_device', e.target.value));
  tuneProcessingWidthSelect?.addEventListener('change', (e) => syncOption('processing_width', e.target.value));
  tuneProfileSelect?.addEventListener('change', (e) => callPy('set_profile', e.target.value).then(cmd => { if (cmd) updateCommandUI(cmd); }));

  const tuneWorkersInput = document.getElementById('tuneWorkersInput');
  const tuneMaxCpuInput = document.getElementById('tuneMaxCpuInput');

  cmdInput?.addEventListener('input', () => {
    state.isCustomCommand = true;
  });

  tuneWorkersInput?.addEventListener('input', (e) => {
    state.isCustomCommand = false;
    let val = parseInt(e.target.value) || 0;
    const maxVal = parseInt(e.target.max) || 64;
    if (val > maxVal) {
      val = maxVal;
      e.target.value = val;
    }
    callPy('set_parallel_workers', val).then(cmd => { if (cmd) updateCommandUI(cmd); });
  });

  tuneMaxCpuInput?.addEventListener('input', (e) => {
    state.isCustomCommand = false;
    let val = parseFloat(e.target.value) || 60.0;
    if (val > 100.0) {
      val = 100.0;
      e.target.value = 100;
    }
    val = Math.max(10.0, val);
    callPy('set_max_cpu_percent', val).then(cmd => { if (cmd) updateCommandUI(cmd); });
  });

  tuneMaxCpuInput?.addEventListener('blur', (e) => {
    let val = parseFloat(e.target.value) || 60.0;
    if (val > 100.0) {
      val = 100.0;
      e.target.value = 100;
    } else if (val < 10.0) {
      val = 10.0;
      e.target.value = 10;
    }
  });

  tuneFallbackChk?.addEventListener('change', (e) => { state.isCustomCommand = false; syncOption('backend_fallbacks', e.target.checked); });
  tuneYoloHalfChk?.addEventListener('change', (e) => { state.isCustomCommand = false; syncOption('yolo_half', e.target.checked); });
  tuneExecutionModeSelect?.addEventListener('change', (e) => { state.isCustomCommand = false; callPy('set_execution_mode', e.target.value).then(cmd => { if (cmd) updateCommandUI(cmd); }); });

  tuneMinCutoffInput?.addEventListener('input', (e) => {
    const val = e.target.value;
    const label = document.getElementById('minCutoffVal');
    if (label) label.textContent = `${val} Hz`;
    syncOption('one_euro_min_cutoff', val);
  });

  tuneBetaInput?.addEventListener('input', (e) => {
    const val = e.target.value;
    const label = document.getElementById('betaVal');
    if (label) label.textContent = val;
    syncOption('one_euro_beta', val);
  });

  tuneDerivCutoffInput?.addEventListener('input', (e) => {
    const val = e.target.value;
    const label = document.getElementById('derivCutoffVal');
    if (label) label.textContent = `${val} Hz`;
    syncOption('one_euro_d_cutoff', val);
  });

  tuneBodyConfInput?.addEventListener('input', (e) => {
    const val = e.target.value;
    const label = document.getElementById('bodyConfVal');
    if (label) label.textContent = parseFloat(val).toFixed(2);
    syncOption('body_conf_threshold', val);
  });

  tuneHandDetConfInput?.addEventListener('input', (e) => {
    const val = e.target.value;
    const label = document.getElementById('handDetConfVal');
    if (label) label.textContent = parseFloat(val).toFixed(2);
    syncOption('hand_det_threshold', val);
  });

  tuneHandConfInput?.addEventListener('input', (e) => {
    const val = e.target.value;
    const label = document.getElementById('handConfVal');
    if (label) label.textContent = parseFloat(val).toFixed(2);
    syncOption('hand_kp_threshold', val);
  });

  tuneHandBoxScaleInput?.addEventListener('input', (e) => {
    const val = e.target.value;
    const label = document.getElementById('handBoxScaleVal');
    if (label) label.textContent = `${parseFloat(val).toFixed(1)}x`;
    syncOption('hand_box_scale', val);
  });

  tuneSkeletonThicknessSelect?.addEventListener('change', (e) => syncOption('skeleton_thickness', e.target.value));

  tunePersonBoxScaleInput?.addEventListener('input', (e) => {
    const val = e.target.value;
    const label = document.getElementById('personBoxScaleVal');
    if (label) label.textContent = `${parseFloat(val).toFixed(2)}x`;
    syncOption('person_box_scale', val);
  });

  tuneTrackHoldFramesInput?.addEventListener('input', (e) => {
    const val = e.target.value;
    const label = document.getElementById('trackHoldVal');
    if (label) label.textContent = `${val} frames`;
    syncOption('person_track_hold_frames', val);
  });

  tuneTrackMatchInput?.addEventListener('input', (e) => {
    const val = e.target.value;
    const label = document.getElementById('trackMatchVal');
    if (label) label.textContent = parseFloat(val).toFixed(2);
    syncOption('person_match_threshold', val);
  });

  tuneEnableTriangulationChk?.addEventListener('change', (e) => {
    callPy('set_workflow_option', 'enable_triangulation', e.target.checked).then(cmd => { if (cmd) updateCommandUI(cmd); });
  });

  tuneTriangulationMinCamsInput?.addEventListener('change', (e) => syncOption('triangulation_min_cameras', e.target.value));
  tuneTriangulationOutlierChk?.addEventListener('change', (e) => syncOption('triangulation_use_outlier_rejection', e.target.checked));

  tuneTriangulationAlphaInput?.addEventListener('input', (e) => {
    const val = e.target.value;
    const label = document.getElementById('triangulationAlphaVal');
    if (label) label.textContent = parseFloat(val).toFixed(2);
    syncOption('triangulation_smoothing_alpha', val);
  });

  tuneCameraSyncInput?.addEventListener('change', (e) => syncOption('sync_offsets', e.target.value));

  tuneCalibrateModeChk?.addEventListener('change', (e) => {
    callPy('set_workflow_option', 'calibrate_mode', e.target.checked).then(cmd => { if (cmd) updateCommandUI(cmd); });
  });

  const tuneOscEnabledChk = document.getElementById('tuneOscEnabledChk');
  const tuneOscHostInput = document.getElementById('tuneOscHostInput');
  const tuneOscPortInput = document.getElementById('tuneOscPortInput');

  tuneSquaresXInput?.addEventListener('change', (e) => syncOption('charuco_squares_x', e.target.value));
  tuneSquaresYInput?.addEventListener('change', (e) => syncOption('charuco_squares_y', e.target.value));
  tuneSquareSizeInput?.addEventListener('change', (e) => syncOption('charuco_square_size', e.target.value));
  tuneCharucoStrictnessSelect?.addEventListener('change', (e) => syncOption('charuco_detection_strictness', e.target.value));
  tuneCharucoSharpenChk?.addEventListener('change', (e) => syncOption('charuco_retry_sharpen', e.target.checked));

  tuneOscEnabledChk?.addEventListener('change', (e) => syncOption('osc_enabled', e.target.checked));
  tuneOscHostInput?.addEventListener('change', (e) => syncOption('osc_host', e.target.value));
  tuneOscPortInput?.addEventListener('change', (e) => syncOption('osc_port', parseInt(e.target.value, 10) || 9000));

  // --------------------------------------------------
  // Single-Expandable Accordion Deck Logic for TUNE Tab
  // --------------------------------------------------
  const accordionHeaders = document.querySelectorAll('#tab-tune .accordion-header');
  accordionHeaders.forEach((header) => {
    header.addEventListener('click', () => {
      const parentCard = header.closest('.accordion-card');
      if (!parentCard) return;

      const isCurrentlyOpen = parentCard.classList.contains('open');

      // Close all accordion cards in #tab-tune
      document.querySelectorAll('#tab-tune .accordion-card').forEach((card) => {
        card.classList.remove('open');
      });

      // If clicked card was closed, open it exclusively
      if (!isCurrentlyOpen) {
        parentCard.classList.add('open');
      }
    });
  });

  function updateCameraTabsUI() {
    if (!cameraTabsContainer) return;
    cameraTabsContainer.innerHTML = '';

    const cams = state.sources.length > 0 ? state.sources.map((_, idx) => `CAM_${idx}`) : ['CAM_0'];
    const workerKeys = Object.keys(state.workerFrames).sort();

    if (!cams.includes(state.activeCamera)) {
      state.activeCamera = cams[0];
    }

    if (activeCamTitle) {
      activeCamTitle.textContent = state.activeWorker === 'ALL' ? 'WORKER GRID VIEW' : (state.activeWorker !== 'MAIN' ? state.activeWorker.replace('_', ' ') : state.activeCamera);
    }

    cams.forEach((camId) => {
      const btn = document.createElement('button');
      btn.className = `cam-tab ${camId === state.activeCamera && state.activeWorker !== 'ALL' ? 'active' : ''}`;
      btn.setAttribute('data-cam', camId);
      btn.textContent = camId;
      btn.addEventListener('click', () => {
        state.activeCamera = camId;
        state.activeWorker = 'MAIN';
        updateCameraTabsUI();
        resetToSingleTileView();
        displayActiveCameraFrame();
      });
      cameraTabsContainer.appendChild(btn);
    });

    if (workerKeys.length > 1) {
      const gridBtn = document.createElement('button');
      gridBtn.className = `cam-tab ${state.activeWorker === 'ALL' ? 'active' : ''}`;
      gridBtn.textContent = 'GRID VIEW';
      gridBtn.addEventListener('click', () => {
        state.activeWorker = 'ALL';
        updateCameraTabsUI();
        renderWorkerGrid();
      });
      cameraTabsContainer.appendChild(gridBtn);

      workerKeys.forEach((wId) => {
        const btn = document.createElement('button');
        btn.className = `cam-tab ${state.activeWorker === wId ? 'active' : ''}`;
        btn.textContent = wId.replace('_', ' ');
        btn.addEventListener('click', () => {
          state.activeWorker = wId;
          updateCameraTabsUI();
          resetToSingleTileView();
          const frame = state.workerFrames[wId];
          if (frame) {
            const liveImg = document.getElementById('livePreviewImg');
            const placeholder = document.getElementById('previewPlaceholder');
            if (liveImg) {
              liveImg.src = `data:image/jpeg;base64,${frame}`;
              if (placeholder) placeholder.style.display = 'none';
              liveImg.style.display = 'block';
            }
          }
        });
        cameraTabsContainer.appendChild(btn);
      });
    }
  }

  function resetToSingleTileView() {
    const videoGrid = document.getElementById('videoGrid');
    if (!videoGrid) return;
    videoGrid.className = 'video-grid';
    if (!document.getElementById('tileMain')) {
      videoGrid.innerHTML = `
        <div class="video-tile" id="tileMain">
          <div class="tile-header">
            <span class="pulse-indicator"></span>
            <span id="activeCamTitle">${state.activeWorker === 'MAIN' ? state.activeCamera : state.activeWorker.replace('_', ' ')}</span>
            <span class="tile-badge">LIVE STREAM</span>
            <span id="hudFrame" class="tile-badge" style="font-family: var(--font-mono);">00000</span>
          </div>
          <div class="tile-placeholder" id="previewPlaceholder">
            <div class="radar-scan"></div>
            <svg viewBox="0 0 24 24" width="42" height="42" fill="none" stroke="currentColor" stroke-width="1.5" class="placeholder-icon">
              <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
            </svg>
            <div class="placeholder-text nothing-font">Awaiting Live Motion Stream</div>
            <div class="placeholder-sub">Processed camera frames &amp; skeleton overlays stream live in real time</div>
          </div>
          <img id="livePreviewImg" class="live-preview-img" style="display:none;" alt="Live Motion Frame" />
        </div>`;
    }
  }

  function renderWorkerGrid() {
    const workerKeys = Object.keys(state.workerFrames);
    if (workerKeys.length <= 1 || state.activeWorker !== 'ALL') return;

    const videoGrid = document.getElementById('videoGrid');
    if (!videoGrid) return;

    videoGrid.className = 'video-grid grid-workers';
    videoGrid.innerHTML = '';

    workerKeys.sort().forEach((wId) => {
      const tile = document.createElement('div');
      tile.className = 'video-tile worker-tile';
      tile.id = `workerTile_${wId}`;
      const b64 = state.workerFrames[wId] || '';
      tile.innerHTML = `
        <div class="tile-header worker-tile-header">
          <span class="pulse-indicator"></span>
          <span>${wId.replace('_', ' ')}</span>
          <span class="tile-badge">CHUNK WORKER</span>
        </div>
        <img class="live-preview-img" style="display:${b64 ? 'block' : 'none'};" src="${b64 ? 'data:image/jpeg;base64,' + b64 : ''}" alt="${wId}" />
      `;
      videoGrid.appendChild(tile);
    });
  }

  function updateWorkerTile(workerId, base64Img) {
    const tile = document.getElementById(`workerTile_${workerId}`);
    if (tile) {
      const img = tile.querySelector('img');
      if (img) {
        img.src = `data:image/jpeg;base64,${base64Img}`;
        img.style.display = 'block';
      }
    }
  }

  function displayActiveCameraFrame() {
    const currentFrame = state.cameraFrames[state.activeCamera];
    if (currentFrame && livePreviewImg) {
      livePreviewImg.src = `data:image/jpeg;base64,${currentFrame}`;
      if (previewPlaceholder) previewPlaceholder.style.display = 'none';
      livePreviewImg.style.display = 'block';
    } else {
      if (livePreviewImg) {
        livePreviewImg.src = '';
        livePreviewImg.style.display = 'none';
      }
      if (previewPlaceholder) previewPlaceholder.style.display = 'flex';
    }
  }

  function updateSourceListUI() {
    if (!fileSourceList) return;
    fileSourceList.innerHTML = '';
    if (state.sources.length === 0) {
      fileSourceList.innerHTML = `
        <div class="source-item empty">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>
          <span>No video files added</span>
        </div>`;
      updateCameraTabsUI();
      return;
    }
    state.sources.forEach((src, idx) => {
      const item = document.createElement('div');
      item.className = 'source-item';
      item.innerHTML = `
        <span>CAM_${idx}: ${src}</span>
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
      `;
      fileSourceList.appendChild(item);
    });
    updateCameraTabsUI();
  }

  function updateCommandUI(commandString, force = true) {
    if (force || !state.isCustomCommand) {
      if (cmdInput) cmdInput.value = commandString;
    }
  }

  // PyWebView Event Listeners from Python Host
  window.addEventListener('pywebviewready', async () => {
    appendLog('PyWebView Native Desktop Bridge Ready', 'system');
    const initCmd = await callPy('get_initial_command');
    if (initCmd) updateCommandUI(initCmd);

    try {
      const sysInfo = await callPy('get_system_info');
      if (sysInfo && tuneWorkersInput) {
        tuneWorkersInput.max = sysInfo.cpu_count;
        tuneWorkersInput.placeholder = `0 = Auto (60% Cap: ${sysInfo.cpu_60_cap} Cores, Max: ${sysInfo.cpu_count})`;
      }
    } catch (e) {}
  });

  // Real-time log & status events emitted from Python Host
  window.onKinaraLog = (text) => {
    appendLog(text);
  };

  window.onKinaraStatus = (status, type) => {
    setStatus(status, type);
  };

  let frameCounter = 0;

  window.resetPreviewStage = () => {
    state.cameraFrames = {};
    state.workerFrames = {};
    state.activeWorker = 'ALL';
    frameCounter = 0;
    resetToSingleTileView();
    if (previewPlaceholder) {
      previewPlaceholder.style.display = 'flex';
    }
    const hudFrame = document.getElementById('hudFrame');
    if (hudFrame) hudFrame.textContent = '00000';
    updateCameraTabsUI();
  };

  window.onKinaraPreviewFrame = (base64Img, camId = 'CAM_0', workerId = 'WORKER_0') => {
    if (!base64Img) return;
    const targetCam = camId && camId.trim() ? camId : 'CAM_0';
    const targetWorker = workerId && workerId.trim() ? workerId : 'WORKER_0';

    const isNewWorker = !state.workerFrames[targetWorker];
    state.cameraFrames[targetCam] = base64Img;
    state.workerFrames[targetWorker] = base64Img;

    if (isNewWorker) {
      updateCameraTabsUI();
      if (state.activeWorker === 'ALL' && Object.keys(state.workerFrames).length > 1) {
        renderWorkerGrid();
      }
    } else {
      updateWorkerTile(targetWorker, base64Img);
    }

    if (targetCam === state.activeCamera || state.activeWorker === targetWorker) {
      const liveImg = document.getElementById('livePreviewImg');
      const placeholder = document.getElementById('previewPlaceholder');
      if (liveImg) {
        liveImg.src = `data:image/jpeg;base64,${base64Img}`;
        if (placeholder) placeholder.style.display = 'none';
        liveImg.style.display = 'block';

        frameCounter++;
        const hudFrame = document.getElementById('hudFrame');
        if (hudFrame) hudFrame.textContent = String(frameCounter).padStart(5, '0');
      }
    }
  };

  // --------------------------------------------------
  // Blender-Style Interactive Panel Resizing Engine
  // --------------------------------------------------
  const sidebarSection = document.getElementById('sidebarSection');
  const splitterHorizontal = document.getElementById('splitterHorizontal');
  const logConsoleContainer = document.querySelector('.log-console-container');
  const splitterVerticalLog = document.getElementById('splitterVerticalLog');

  // Load saved layout dimensions from localStorage
  const LAYOUT_KEY = 'kinara_layout_config';
  try {
    const savedLayout = JSON.parse(localStorage.getItem(LAYOUT_KEY) || '{}');
    if (savedLayout.sidebarWidth && sidebarSection) {
      sidebarSection.style.width = `${savedLayout.sidebarWidth}px`;
    }
    if (savedLayout.logHeight && logConsoleContainer) {
      logConsoleContainer.style.height = `${savedLayout.logHeight}px`;
    }
  } catch (e) {
    console.warn('Could not load saved layout preferences:', e);
  }

  function saveLayoutConfig(key, value) {
    try {
      const config = JSON.parse(localStorage.getItem(LAYOUT_KEY) || '{}');
      config[key] = value;
      localStorage.setItem(LAYOUT_KEY, JSON.stringify(config));
    } catch (e) {}
  }

  // 1. Horizontal Splitter (Sidebar Width Resizer)
  if (splitterHorizontal && sidebarSection) {
    let isDraggingH = false;
    let startX = 0;
    let startWidth = 0;

    splitterHorizontal.addEventListener('mousedown', (e) => {
      isDraggingH = true;
      startX = e.clientX;
      startWidth = sidebarSection.getBoundingClientRect().width;
      splitterHorizontal.classList.add('is-dragging');
      document.body.classList.add('resizing-h');
      e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
      if (!isDraggingH) return;
      const dx = startX - e.clientX; // dragging left expands right sidebar width
      let newWidth = Math.round(Math.max(260, Math.min(650, startWidth + dx)));
      sidebarSection.style.width = `${newWidth}px`;

      // Update dimension tooltip
      let tt = splitterHorizontal.querySelector('.splitter-tooltip');
      if (!tt) {
        tt = document.createElement('div');
        tt.className = 'splitter-tooltip';
        splitterHorizontal.appendChild(tt);
      }
      tt.textContent = `${newWidth}px`;
    });

    document.addEventListener('mouseup', () => {
      if (isDraggingH) {
        isDraggingH = false;
        splitterHorizontal.classList.remove('is-dragging');
        document.body.classList.remove('resizing-h');
        const tt = splitterHorizontal.querySelector('.splitter-tooltip');
        if (tt) tt.remove();
        saveLayoutConfig('sidebarWidth', sidebarSection.getBoundingClientRect().width);
      }
    });

    // Double-click to reset sidebar width to default (380px)
    splitterHorizontal.addEventListener('dblclick', () => {
      sidebarSection.style.width = '380px';
      saveLayoutConfig('sidebarWidth', 380);
      appendLog('Reset sidebar width to default (380px)', 'system');
    });
  }

  // 2. Vertical Splitter (Terminal Console Height Resizer)
  if (splitterVerticalLog && logConsoleContainer) {
    let isDraggingV = false;
    let startY = 0;
    let startHeight = 0;

    splitterVerticalLog.addEventListener('mousedown', (e) => {
      isDraggingV = true;
      startY = e.clientY;
      startHeight = logConsoleContainer.getBoundingClientRect().height;
      splitterVerticalLog.classList.add('is-dragging');
      document.body.classList.add('resizing-v');
      e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
      if (!isDraggingV) return;
      const dy = startY - e.clientY; // dragging up expands console height
      let newHeight = Math.round(Math.max(70, Math.min(500, startHeight + dy)));
      logConsoleContainer.style.height = `${newHeight}px`;

      // Update dimension tooltip
      let tt = splitterVerticalLog.querySelector('.splitter-tooltip');
      if (!tt) {
        tt = document.createElement('div');
        tt.className = 'splitter-tooltip';
        splitterVerticalLog.appendChild(tt);
      }
      tt.textContent = `${newHeight}px`;
    });

    document.addEventListener('mouseup', () => {
      if (isDraggingV) {
        isDraggingV = false;
        splitterVerticalLog.classList.remove('is-dragging');
        document.body.classList.remove('resizing-v');
        const tt = splitterVerticalLog.querySelector('.splitter-tooltip');
        if (tt) tt.remove();
        saveLayoutConfig('logHeight', logConsoleContainer.getBoundingClientRect().height);
      }
    });

    // Double-click to reset log console height to default (150px)
    splitterVerticalLog.addEventListener('dblclick', () => {
      logConsoleContainer.style.height = '150px';
      saveLayoutConfig('logHeight', 150);
      appendLog('Reset terminal log height to default (150px)', 'system');
    });
  }

  // --------------------------------------------------
  // Dual Theme Engine (Dark & Light Mode)
  // --------------------------------------------------
  const THEME_KEY = 'kinara_theme_preference';
  const btnThemeToggle = document.getElementById('btnThemeToggle');
  const themeIconSun = document.getElementById('themeIconSun');
  const themeIconMoon = document.getElementById('themeIconMoon');
  const themeText = document.getElementById('themeText');

  let currentTheme = localStorage.getItem(THEME_KEY) || 'dark';
  applyTheme(currentTheme);

  function applyTheme(theme) {
    currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);

    if (theme === 'light') {
      if (themeIconSun) themeIconSun.style.display = 'none';
      if (themeIconMoon) themeIconMoon.style.display = 'inline-block';
      if (themeText) themeText.textContent = 'Light';
    } else {
      if (themeIconSun) themeIconSun.style.display = 'inline-block';
      if (themeIconMoon) themeIconMoon.style.display = 'none';
      if (themeText) themeText.textContent = 'Dark';
    }
  }

  btnThemeToggle?.addEventListener('click', () => {
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(nextTheme);
    appendLog(`Switched theme to ${nextTheme.toUpperCase()} mode`, 'system');
  });

  // --------------------------------------------------
  // Raycast-Style Interactive Command Palette (Ctrl + K)
  // --------------------------------------------------
  const cmdPaletteOverlay = document.getElementById('cmdPaletteOverlay');
  const cmdPaletteInput = document.getElementById('cmdPaletteInput');
  const cmdPaletteResults = document.getElementById('cmdPaletteResults');
  const btnOpenCmdPalette = document.getElementById('btnOpenCmdPalette');

  const COMMAND_ACTIONS = [
    { title: '▶ Start Pipeline', category: 'Execution', action: () => btnStart?.click() },
    { title: '⚙ Check Runtime Environment', category: 'Execution', action: () => btnCheck?.click() },
    { title: '⏹ Stop Execution', category: 'Execution', action: () => btnStop?.click() },
    { title: '⚡ Emergency Kill', category: 'Execution', action: () => btnKill?.click() },
    { title: '🎬 Replay Intro Splash Screen', category: 'Visuals', action: () => replayIntroSplash() },
    { title: '🌓 Toggle Light / Dark Theme', category: 'System', action: () => btnThemeToggle?.click() },
    { title: '⚡ Apply RTMPose CUDA Preset', category: 'Presets', action: () => document.getElementById('btnApplyRTMPose')?.click() },
    { title: '⚡ Apply MediaPipe Pose Preset', category: 'Presets', action: () => document.getElementById('btnApplyMediaPipe')?.click() },
    { title: '⚡ Apply ONNX YOLO Preset', category: 'Presets', action: () => document.getElementById('btnApplyONNX')?.click() },
    { title: '⚡ Apply RTMPose WholeBody Preset', category: 'Presets', action: () => document.getElementById('btnApplyWholeBody')?.click() },
    { title: '📷 Use Webcam Input (Camera 0)', category: 'Capture', action: () => document.getElementById('btnUseCamera')?.click() },
    { title: '📁 Add Video Files...', category: 'Files', action: () => document.getElementById('btnAddFiles')?.click() },
    { title: '📐 Apply Calibration', category: 'Calibration', action: () => document.getElementById('btnCalibA3')?.click() },
    { title: '🎛 Open Engine Tuning Controls', category: 'Tune', action: () => document.querySelector('.tab-btn[data-tab="tune"]')?.click() },
    { title: '🧹 Clear Terminal Console Logs', category: 'Console', action: () => btnClearLog?.click() },
    { title: '↻ Reset Default Pipeline Settings', category: 'System', action: () => btnResetDefaults?.click() },
    // Execution Mode
    { title: '🔀 Processing Mode: Auto', category: 'Tune', action: () => { const el = document.getElementById('tuneExecutionModeSelect'); if (el) { el.value = 'auto'; el.dispatchEvent(new Event('change')); } } },
    { title: '🔀 Processing Mode: Serial', category: 'Tune', action: () => { const el = document.getElementById('tuneExecutionModeSelect'); if (el) { el.value = 'serial'; el.dispatchEvent(new Event('change')); } } },
    { title: '🔀 Processing Mode: Parallel', category: 'Tune', action: () => { const el = document.getElementById('tuneExecutionModeSelect'); if (el) { el.value = 'parallel'; el.dispatchEvent(new Event('change')); } } },
    { title: '🔀 Processing Mode: Pipeline Parallel', category: 'Tune', action: () => { const el = document.getElementById('tuneExecutionModeSelect'); if (el) { el.value = 'pipeline-parallel'; el.dispatchEvent(new Event('change')); } } },
    // Body Backend
    { title: '🦴 Body Backend: MediaPipe', category: 'Tune', action: () => { const el = document.getElementById('tuneBackendSelect'); if (el) { el.value = 'mediapipe'; el.dispatchEvent(new Event('change')); } } },
    { title: '🦴 Body Backend: RTMPose', category: 'Tune', action: () => { const el = document.getElementById('tuneBackendSelect'); if (el) { el.value = 'rtmpose'; el.dispatchEvent(new Event('change')); } } },
    { title: '🦴 Body Backend: YOLO', category: 'Tune', action: () => { const el = document.getElementById('tuneBackendSelect'); if (el) { el.value = 'yolo'; el.dispatchEvent(new Event('change')); } } },
    { title: '🦴 Body Backend: WholeBody', category: 'Tune', action: () => { const el = document.getElementById('tuneBackendSelect'); if (el) { el.value = 'wholebody'; el.dispatchEvent(new Event('change')); } } },
    // Hand Backend
    { title: '✋ Hand Backend: MediaPipe', category: 'Tune', action: () => { const el = document.getElementById('tuneHandBackendSelect'); if (el) { el.value = 'mediapipe'; el.dispatchEvent(new Event('change')); } } },
    { title: '✋ Hand Backend: ONNX', category: 'Tune', action: () => { const el = document.getElementById('tuneHandBackendSelect'); if (el) { el.value = 'onnx'; el.dispatchEvent(new Event('change')); } } },
    { title: '✋ Hand Backend: RTMPose WholeBody', category: 'Tune', action: () => { const el = document.getElementById('tuneHandBackendSelect'); if (el) { el.value = 'rtmpose-wholebody'; el.dispatchEvent(new Event('change')); } } },
    { title: '✋ Hand Backend: Disabled', category: 'Tune', action: () => { const el = document.getElementById('tuneHandBackendSelect'); if (el) { el.value = 'none'; el.dispatchEvent(new Event('change')); } } },
    // Inference Device
    { title: '🖥 Inference Device: CUDA GPU', category: 'Tune', action: () => { const el = document.getElementById('tuneRtmposeDeviceSelect'); if (el) { el.value = 'cuda'; el.dispatchEvent(new Event('change')); } } },
    { title: '🖥 Inference Device: CPU', category: 'Tune', action: () => { const el = document.getElementById('tuneRtmposeDeviceSelect'); if (el) { el.value = 'cpu'; el.dispatchEvent(new Event('change')); } } },
    // Performance Profile
    { title: '⚡ Profile: Fastest', category: 'Tune', action: () => { const el = document.getElementById('tuneProfileSelect'); if (el) { el.value = 'fastest'; el.dispatchEvent(new Event('change')); } } },
    { title: '⚡ Profile: Mid (Balanced)', category: 'Tune', action: () => { const el = document.getElementById('tuneProfileSelect'); if (el) { el.value = 'mid'; el.dispatchEvent(new Event('change')); } } },
    { title: '⚡ Profile: Quality', category: 'Tune', action: () => { const el = document.getElementById('tuneProfileSelect'); if (el) { el.value = 'quality'; el.dispatchEvent(new Event('change')); } } },
    // Processing Resolution
    { title: '📐 Resolution: Original', category: 'Tune', action: () => { const el = document.getElementById('tuneProcessingWidthSelect'); if (el) { el.value = ''; el.dispatchEvent(new Event('change')); } } },
    { title: '📐 Resolution: 1280px HD', category: 'Tune', action: () => { const el = document.getElementById('tuneProcessingWidthSelect'); if (el) { el.value = '1280'; el.dispatchEvent(new Event('change')); } } },
    { title: '📐 Resolution: 640px Ultra Fast', category: 'Tune', action: () => { const el = document.getElementById('tuneProcessingWidthSelect'); if (el) { el.value = '640'; el.dispatchEvent(new Event('change')); } } },
    // Toggles
    { title: '🔁 Toggle CPU Backend Fallback', category: 'Tune', action: () => { const el = document.getElementById('tuneFallbackChk'); if (el) { el.checked = !el.checked; el.dispatchEvent(new Event('change')); } } },
    { title: '🔁 Toggle CUDA FP16 Half-Precision', category: 'Tune', action: () => { const el = document.getElementById('tuneYoloHalfChk'); if (el) { el.checked = !el.checked; el.dispatchEvent(new Event('change')); } } },
    // Tab Navigation shortcuts
    { title: '📋 Go to Capture Tab', category: 'Navigation', action: () => document.querySelector('.tab-btn[data-tab="capture"]')?.click() },
    { title: '📋 Go to Preview Tab', category: 'Navigation', action: () => document.querySelector('.tab-btn[data-tab="preview"]')?.click() },
    { title: '📋 Go to Console Tab', category: 'Navigation', action: () => document.querySelector('.tab-btn[data-tab="console"]')?.click() },
    { title: '📋 Go to Sources Tab', category: 'Navigation', action: () => document.querySelector('.tab-btn[data-tab="sources"]')?.click() },
    { title: '📋 Go to People Tab', category: 'Navigation', action: () => document.querySelector('.tab-btn[data-tab="people"]')?.click() },
    { title: '📋 Go to Tune Tab', category: 'Navigation', action: () => document.querySelector('.tab-btn[data-tab="tune"]')?.click() },
    { title: '📋 Go to Calibration Tab', category: 'Navigation', action: () => document.querySelector('.tab-btn[data-tab="calibration"]')?.click() },
  ];

  let selectedIndex = 0;

  function toggleCommandPalette(show) {
    if (!cmdPaletteOverlay) return;
    const isVisible = cmdPaletteOverlay.style.display !== 'none';
    const nextState = show !== undefined ? show : !isVisible;

    if (nextState) {
      cmdPaletteOverlay.style.display = 'flex';
      if (cmdPaletteInput) {
        cmdPaletteInput.value = '';
        cmdPaletteInput.focus();
      }
      renderCommandResults('');
    } else {
      cmdPaletteOverlay.style.display = 'none';
    }
  }

  function renderCommandResults(query) {
    if (!cmdPaletteResults) return;
    const q = (query || '').toLowerCase().trim();
    const filtered = COMMAND_ACTIONS.filter(c => c.title.toLowerCase().includes(q) || c.category.toLowerCase().includes(q));

    selectedIndex = Math.min(selectedIndex, Math.max(0, filtered.length - 1));
    cmdPaletteResults.innerHTML = '';

    if (filtered.length === 0) {
      cmdPaletteResults.innerHTML = '<div class="cmd-item" style="color:var(--text-muted);">No matching commands found</div>';
      return;
    }

    filtered.forEach((cmd, idx) => {
      const item = document.createElement('div');
      item.className = `cmd-item ${idx === selectedIndex ? 'selected' : ''}`;
      item.innerHTML = `
        <div class="cmd-item-left">
          <svg class="cmd-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
          <span>${cmd.title}</span>
        </div>
        <kbd>${cmd.category}</kbd>
      `;
      item.addEventListener('click', () => {
        cmd.action();
        toggleCommandPalette(false);
      });
      cmdPaletteResults.appendChild(item);
    });
  }

  btnOpenCmdPalette?.addEventListener('click', () => toggleCommandPalette(true));

  cmdPaletteInput?.addEventListener('input', (e) => {
    selectedIndex = 0;
    renderCommandResults(e.target.value);
  });

  cmdPaletteOverlay?.addEventListener('click', (e) => {
    if (e.target === cmdPaletteOverlay) toggleCommandPalette(false);
  });

  // Global Keyboard Shortcuts Engine
  document.addEventListener('keydown', (e) => {
    // Ctrl + K for Command Palette
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      toggleCommandPalette();
      return;
    }

    // Esc to close palette
    if (e.key === 'Escape') {
      if (cmdPaletteOverlay && cmdPaletteOverlay.style.display !== 'none') {
        toggleCommandPalette(false);
        return;
      }
    }

    // Command Palette Arrow Navigation & Enter Execution
    if (cmdPaletteOverlay && cmdPaletteOverlay.style.display !== 'none') {
      const items = cmdPaletteResults?.querySelectorAll('.cmd-item');
      if (!items || items.length === 0) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        selectedIndex = (selectedIndex + 1) % items.length;
        renderCommandResults(cmdPaletteInput?.value || '');
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        selectedIndex = (selectedIndex - 1 + items.length) % items.length;
        renderCommandResults(cmdPaletteInput?.value || '');
      } else if (e.key === 'Enter') {
        e.preventDefault();
        const q = (cmdPaletteInput?.value || '').toLowerCase().trim();
        const filtered = COMMAND_ACTIONS.filter(c => c.title.toLowerCase().includes(q) || c.category.toLowerCase().includes(q));
        if (filtered[selectedIndex]) {
          filtered[selectedIndex].action();
          toggleCommandPalette(false);
        }
      }
      return;
    }

    // Hotkey Number Keys 1-7 for Tabs (when not focused on text inputs)
    if (!['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
      if (['1', '2', '3', '4', '5', '6', '7'].includes(e.key)) {
        const idx = parseInt(e.key) - 1;
        const btns = document.querySelectorAll('.tab-btn');
        if (btns[idx]) {
          btns[idx].click();
          appendLog(`Switched to tab: ${btns[idx].textContent.trim()}`, 'system');
        }
      }
    }
  });

  // --------------------------------------------------
  // Nothing OS Interactive Dotted Matrix Background Engine
  // --------------------------------------------------
  (function initParallaxCanvas() {
    const canvas = document.getElementById('parallaxCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    });

    const mouse = { x: width / 2, y: height / 2, targetX: width / 2, targetY: height / 2 };

    window.addEventListener('mousemove', (e) => {
      mouse.targetX = e.clientX;
      mouse.targetY = e.clientY;
    });

    const SPACING = 30; // Grid cell spacing in pixels

    function drawDottedGrid() {
      ctx.clearRect(0, 0, width, height);

      // Lerp mouse coordinates for smooth 2.5D parallax floating physics
      mouse.x += (mouse.targetX - mouse.x) * 0.06;
      mouse.y += (mouse.targetY - mouse.y) * 0.06;

      const isLight = document.documentElement.getAttribute('data-theme') === 'light';
      const offsetX = (mouse.x - width / 2) * 0.025;
      const offsetY = (mouse.y - height / 2) * 0.025;

      const cols = Math.ceil(width / SPACING) + 2;
      const rows = Math.ceil(height / SPACING) + 2;
      const INTERACTION_RADIUS = 260;

      for (let c = -1; c < cols; c++) {
        for (let r = -1; r < rows; r++) {
          const baseX = c * SPACING + offsetX;
          const baseY = r * SPACING + offsetY;

          // Compute distance to current mouse cursor position
          const dist = Math.hypot(baseX - mouse.x, baseY - mouse.y);
          let dotRadius = 1.5;
          let dotAlpha = isLight ? 0.30 : 0.38;
          let dotColor = isLight ? `rgba(100, 116, 139, ${dotAlpha})` : `rgba(255, 255, 255, ${dotAlpha})`;

          let dispX = 0;
          let dispY = 0;

          let shadowBlur = 0;
          let shadowColor = 'transparent';

          // Interactive Mouse Wave: Dark Mode = Ball Glow, Light Mode = Ball Dim / Darken
          if (dist < INTERACTION_RADIUS) {
            const factor = (1 - dist / INTERACTION_RADIUS);
            
            // Spring displacement away from cursor
            const angle = Math.atan2(baseY - mouse.y, baseX - mouse.x);
            dispX = Math.cos(angle) * factor * 16;
            dispY = Math.sin(angle) * factor * 16;

            if (isLight) {
              // --------------------------------------------------
              // Light Mode: Grey, Royal Purple & Deep Indigo Balls
              // --------------------------------------------------
              dotRadius = 1.5 + factor * 3.2;

              if ((c + r) % 5 === 0) {
                // Royal Purple Accent Ball
                dotColor = `rgba(124, 58, 237, ${0.70 + factor * 0.30})`;
              } else if ((c + r) % 3 === 0) {
                // Deep Indigo Violet Ball
                dotColor = `rgba(67, 56, 202, ${0.70 + factor * 0.30})`;
              } else {
                // Slate Grey Ball
                dotColor = `rgba(71, 85, 105, ${0.65 + factor * 0.35})`;
              }
            } else {
              // --------------------------------------------------
              // Dark Mode: Signature Red, Cyber Cyan Blue & Pure White Glowing Balls
              // --------------------------------------------------
              dotRadius = 1.5 + factor * 3.8;
              shadowBlur = Math.round(14 * factor);

              if ((c + r) % 5 === 0) {
                // Signature Glowing Red Ball
                dotColor = `rgba(255, 0, 85, ${0.75 + factor * 0.25})`;
                shadowColor = `rgba(255, 0, 85, ${0.90 * factor})`;
              } else if ((c + r) % 3 === 0) {
                // Neon Blue / Cyber Cyan Glowing Ball
                dotColor = `rgba(0, 242, 254, ${0.70 + factor * 0.30})`;
                shadowColor = `rgba(0, 242, 254, ${0.85 * factor})`;
              } else {
                // Pure Luminous White Glowing Ball
                dotColor = `rgba(255, 255, 255, ${0.80 + factor * 0.20})`;
                shadowColor = `rgba(255, 255, 255, ${0.85 * factor})`;
              }
            }
          }

          ctx.save();
          if (shadowBlur > 0) {
            ctx.shadowBlur = shadowBlur;
            ctx.shadowColor = shadowColor;
          }

          ctx.fillStyle = dotColor;
          ctx.beginPath();
          ctx.arc(baseX + dispX, baseY + dispY, dotRadius, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }
      }

      requestAnimationFrame(drawDottedGrid);
    }

    drawDottedGrid();
  })();

  // --------------------------------------------------
  // Nothing OS Pixelated Ripple Intro Splash Engine
  // --------------------------------------------------
  const introSplash = document.getElementById('introSplash');
  const introRippleCanvas = document.getElementById('introRippleCanvas');
  const brandLogoWrapper = document.querySelector('.brand-logo-wrapper');

  let splashTimeout = null;
  let startRippleAnimation = null;

  function replayIntroSplash() {
    if (!introSplash) return;
    introSplash.style.display = 'flex';
    introSplash.classList.remove('fade-out');

    // Reset progress animation
    const progress = introSplash.querySelector('.intro-loader-progress');
    if (progress) {
      progress.style.animation = 'none';
      progress.offsetHeight; // trigger reflow
      progress.style.animation = 'intro-progress 2.2s cubic-bezier(0.16, 1, 0.3, 1) forwards';
    }

    if (typeof startRippleAnimation === 'function') {
      startRippleAnimation();
    }

    if (splashTimeout) clearTimeout(splashTimeout);

    // Auto-fade after 2.4s
    splashTimeout = setTimeout(() => {
      introSplash.classList.add('fade-out');
      setTimeout(() => {
        introSplash.style.display = 'none';
      }, 600);
    }, 2400);
  }

  // Click brand logo in header to replay intro
  brandLogoWrapper?.addEventListener('click', () => {
    replayIntroSplash();
    appendLog('Replaying Nothing OS Intro Splash', 'system');
  });

  (function initIntroRippleCanvas() {
    if (!introRippleCanvas || !introSplash) return;
    const ctx = introRippleCanvas.getContext('2d');
    if (!ctx) return;

    let w = introRippleCanvas.width = window.innerWidth;
    let h = introRippleCanvas.height = window.innerHeight;

    const CELL_SIZE = 14;
    let cols = Math.ceil(w / CELL_SIZE);
    let rows = Math.ceil(h / CELL_SIZE);

    let buffer1 = new Float32Array(cols * rows);
    let buffer2 = new Float32Array(cols * rows);

    window.addEventListener('resize', () => {
      w = introRippleCanvas.width = window.innerWidth;
      h = introRippleCanvas.height = window.innerHeight;
      cols = Math.ceil(w / CELL_SIZE);
      rows = Math.ceil(h / CELL_SIZE);
      buffer1 = new Float32Array(cols * rows);
      buffer2 = new Float32Array(cols * rows);
    });

    let prevMouse = { x: w / 2, y: h / 2 };
    let isAnimating = false;

    window.addEventListener('mousemove', (e) => {
      if (introSplash.style.display === 'none') return;
      const mx = e.clientX;
      const my = e.clientY;
      const dx = mx - prevMouse.x;
      const dy = my - prevMouse.y;
      const speed = Math.hypot(dx, dy);

      if (speed > 0.5) {
        const mc = Math.floor(mx / CELL_SIZE);
        const mr = Math.floor(my / CELL_SIZE);
        const radius = 2;
        for (let r = -radius; r <= radius; r++) {
          for (let c = -radius; c <= radius; c++) {
            const nc = mc + c;
            const nr = mr + r;
            if (nc >= 1 && nc < cols - 1 && nr >= 1 && nr < rows - 1) {
              const dist = Math.hypot(c, r);
              if (dist <= radius) {
                const idx = nr * cols + nc;
                buffer1[idx] += (1 - dist / radius) * Math.min(speed * 2.2, 50);
              }
            }
          }
        }
      }
      prevMouse.x = mx;
      prevMouse.y = my;

      if (!isAnimating) {
        isAnimating = true;
        drawWaterFlow();
      }
    });

    function drawWaterFlow() {
      ctx.fillStyle = '#050505';
      ctx.fillRect(0, 0, w, h);

      let activeEnergy = 0;
      for (let r = 1; r < rows - 1; r++) {
        const rowIdx = r * cols;
        for (let c = 1; c < cols - 1; c++) {
          const i = rowIdx + c;
          let val = (
            buffer1[i - 1] +
            buffer1[i + 1] +
            buffer1[i - cols] +
            buffer1[i + cols]
          ) / 2 - buffer2[i];

          val *= 0.90; // Faster damping for subtle liquid movement
          buffer2[i] = val;
          activeEnergy += Math.abs(val);
        }
      }

      const temp = buffer1;
      buffer1 = buffer2;
      buffer2 = temp;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const i = r * cols + c;
          const heightVal = buffer1[i];

          const cx = c * CELL_SIZE + CELL_SIZE / 2;
          const cy = r * CELL_SIZE + CELL_SIZE / 2;

          let dx = 0;
          let dy = 0;
          if (c > 0 && c < cols - 1 && r > 0 && r < rows - 1) {
            dx = (buffer1[i - 1] - buffer1[i + 1]) * 0.15;
            dy = (buffer1[i - cols] - buffer1[i + cols]) * 0.15;
          }

          const waveMag = Math.abs(heightVal);
          const baseRadius = 1.4;
          const dotRadius = baseRadius + Math.min(waveMag * 0.02, 1.8);
          const alpha = 0.18 + Math.min(waveMag * 0.01, 0.45);

          ctx.fillStyle = waveMag > 3.0
            ? `rgba(255, 0, 55, ${alpha})`
            : `rgba(255, 255, 255, ${alpha * 0.4})`;

          ctx.beginPath();
          ctx.arc(cx + dx, cy + dy, dotRadius, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      if (introSplash.style.display !== 'none' || activeEnergy > 2.0) {
        requestAnimationFrame(drawWaterFlow);
      } else {
        isAnimating = false;
      }
    }

    startRippleAnimation = function() {
      const mc = Math.floor(cols / 2);
      const mr = Math.floor(rows / 2);
      for (let r = -4; r <= 4; r++) {
        for (let c = -4; c <= 4; c++) {
          const nc = mc + c;
          const nr = mr + r;
          if (nc >= 1 && nc < cols - 1 && nr >= 1 && nr < rows - 1) {
            const dist = Math.hypot(c, r);
            if (dist <= 4) {
              buffer1[nr * cols + nc] = (1 - dist / 4) * 260;
            }
          }
        }
      }
      if (!isAnimating) {
        isAnimating = true;
        drawWaterFlow();
      }
    };

    replayIntroSplash();
  })();

  // ==========================================================================
  // Cyber-Glass Dropdown System Generator & Synchronizer
  // ==========================================================================
  function initCustomGlassDropdowns() {
    const selects = document.querySelectorAll('select.form-control, select');

    selects.forEach((select) => {
      if (select.dataset.glassSelectInit === 'true') {
        if (typeof select._syncGlassSelect === 'function') {
          select._syncGlassSelect();
        }
        return;
      }

      select.dataset.glassSelectInit = 'true';

      // Hide native select element visually while retaining standard form accessibility & events
      select.style.position = 'absolute';
      select.style.opacity = '0';
      select.style.pointerEvents = 'none';
      select.style.width = '0';
      select.style.height = '0';
      select.style.overflow = 'hidden';

      // Create glass wrapper container
      const wrapper = document.createElement('div');
      wrapper.className = 'glass-select-wrapper';
      select.parentNode.insertBefore(wrapper, select);
      wrapper.appendChild(select);

      // Create glass trigger button
      const trigger = document.createElement('div');
      trigger.className = 'glass-select-trigger';
      trigger.tabIndex = 0;
      trigger.setAttribute('role', 'combobox');
      trigger.setAttribute('aria-expanded', 'false');
      trigger.setAttribute('aria-haspopup', 'listbox');

      const labelSpan = document.createElement('span');
      labelSpan.className = 'glass-select-label';

      const arrowSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      arrowSvg.setAttribute('class', 'glass-select-arrow');
      arrowSvg.setAttribute('viewBox', '0 0 24 24');
      arrowSvg.setAttribute('fill', 'none');
      arrowSvg.setAttribute('stroke', 'currentColor');
      arrowSvg.setAttribute('stroke-width', '2.5');
      arrowSvg.setAttribute('stroke-linecap', 'round');
      arrowSvg.setAttribute('stroke-linejoin', 'round');

      const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
      polyline.setAttribute('points', '6 9 12 15 18 9');
      arrowSvg.appendChild(polyline);

      trigger.appendChild(labelSpan);
      trigger.appendChild(arrowSvg);
      wrapper.appendChild(trigger);

      // Create floating glass dropdown listbox panel
      const dropdownMenu = document.createElement('div');
      dropdownMenu.className = 'glass-select-dropdown';
      dropdownMenu.setAttribute('role', 'listbox');
      wrapper.appendChild(dropdownMenu);

      function renderOptions() {
        dropdownMenu.innerHTML = '';
        const options = Array.from(select.options);
        const currentVal = select.value;

        options.forEach((opt) => {
          const optDiv = document.createElement('div');
          const isSelected = opt.value === currentVal || (opt.selected && !currentVal);
          optDiv.className = `glass-select-option ${isSelected ? 'selected' : ''}`;
          optDiv.setAttribute('role', 'option');
          optDiv.setAttribute('data-value', opt.value);

          const optText = document.createElement('span');
          optText.textContent = opt.textContent;
          optDiv.appendChild(optText);

          // Neon Checkmark SVG
          const checkSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
          checkSvg.setAttribute('class', 'check-icon');
          checkSvg.setAttribute('viewBox', '0 0 24 24');
          checkSvg.setAttribute('fill', 'none');
          checkSvg.setAttribute('stroke', 'currentColor');
          checkSvg.setAttribute('stroke-width', '3');
          checkSvg.setAttribute('stroke-linecap', 'round');
          checkSvg.setAttribute('stroke-linejoin', 'round');

          const checkPoly = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
          checkPoly.setAttribute('points', '20 6 9 17 4 12');
          checkSvg.appendChild(checkPoly);
          optDiv.appendChild(checkSvg);

          optDiv.addEventListener('click', (e) => {
            e.stopPropagation();
            if (select.value !== opt.value) {
              select.value = opt.value;
              select.dispatchEvent(new Event('change', { bubbles: true }));
            }
            updateTriggerLabel();
            closeDropdown();
          });

          dropdownMenu.appendChild(optDiv);
        });

        updateTriggerLabel();
      }

      function updateTriggerLabel() {
        const selectedOpt = select.options[select.selectedIndex] || select.options[0];
        labelSpan.textContent = selectedOpt ? selectedOpt.textContent : '';

        const currentVal = select.value;
        const optElements = dropdownMenu.querySelectorAll('.glass-select-option');
        optElements.forEach((el) => {
          if (el.getAttribute('data-value') === currentVal) {
            el.classList.add('selected');
          } else {
            el.classList.remove('selected');
          }
        });
      }

      function updateDropdownPosition() {
        if (!wrapper.classList.contains('open')) return;
        const rect = trigger.getBoundingClientRect();
        dropdownMenu.style.position = 'fixed';
        dropdownMenu.style.left = `${rect.left}px`;
        dropdownMenu.style.width = `${rect.width}px`;
        dropdownMenu.style.zIndex = '9999999';

        const dropdownHeight = Math.min(250, select.options.length * 38 + 16);
        const spaceBelow = window.innerHeight - rect.bottom;

        if (spaceBelow < dropdownHeight && rect.top > dropdownHeight) {
          dropdownMenu.style.top = 'auto';
          dropdownMenu.style.bottom = `${window.innerHeight - rect.top + 6}px`;
        } else {
          dropdownMenu.style.bottom = 'auto';
          dropdownMenu.style.top = `${rect.bottom + 6}px`;
        }
      }

      function openDropdown() {
        // Close all other open dropdown wrappers
        document.querySelectorAll('.glass-select-wrapper.open').forEach((w) => {
          if (w !== wrapper) {
            w.classList.remove('open');
            const d = document.querySelector(`.glass-select-dropdown[data-for="${w.dataset.glassId}"]`);
            if (d) {
              d.classList.remove('open');
              if (d.parentNode === document.body) wrapper.appendChild(d);
            }
          }
        });

        const glassId = select.id || Math.random().toString(36).substring(2, 9);
        wrapper.dataset.glassId = glassId;
        dropdownMenu.dataset.for = glassId;

        wrapper.classList.add('open');
        trigger.setAttribute('aria-expanded', 'true');

        if (dropdownMenu.parentNode !== document.body) {
          document.body.appendChild(dropdownMenu);
        }

        updateDropdownPosition();
        dropdownMenu.classList.add('open');

        window.addEventListener('scroll', updateDropdownPosition, true);
        window.addEventListener('resize', updateDropdownPosition, true);
      }

      function closeDropdown() {
        wrapper.classList.remove('open');
        trigger.setAttribute('aria-expanded', 'false');
        dropdownMenu.classList.remove('open');

        window.removeEventListener('scroll', updateDropdownPosition, true);
        window.removeEventListener('resize', updateDropdownPosition, true);

        if (dropdownMenu.parentNode === document.body) {
          wrapper.appendChild(dropdownMenu);
        }
      }

      function toggleDropdown() {
        if (wrapper.classList.contains('open')) {
          closeDropdown();
        } else {
          openDropdown();
        }
      }

      trigger.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleDropdown();
      });

      trigger.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown') {
          e.preventDefault();
          openDropdown();
        } else if (e.key === 'Escape') {
          closeDropdown();
        }
      });

      select._syncGlassSelect = () => {
        renderOptions();
      };

      select.addEventListener('change', () => {
        updateTriggerLabel();
      });

      const observer = new MutationObserver(() => {
        renderOptions();
      });
      observer.observe(select, { childList: true, subtree: true, attributes: true });

      renderOptions();
    });
  }

  // Close open glass dropdowns when clicking outside
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.glass-select-wrapper') && !e.target.closest('.glass-select-dropdown')) {
      document.querySelectorAll('.glass-select-wrapper.open').forEach((w) => {
        w.classList.remove('open');
        const glassId = w.dataset.glassId;
        const d = document.querySelector(`.glass-select-dropdown[data-for="${glassId}"]`);
        if (d) {
          d.classList.remove('open');
          if (d.parentNode === document.body) {
            w.appendChild(d);
          }
        }
      });
    }
  });

  // Initialize all custom Cyber-Glass dropdown controls
  initCustomGlassDropdowns();

});




