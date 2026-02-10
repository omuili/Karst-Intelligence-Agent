/**
 * Main application controller for Sinkhole Scanner
 * REAL DATA VERSION - No pre-loading, requires scan to see data
 */

class SinkholeApp {
  constructor() {
    this.map = null;
    this.scanner = null;
    this.config = null;

    // Bind methods
    this.init = this.init.bind(this);
    this.handleMapClick = this.handleMapClick.bind(this);
    this.handleScanToggle = this.handleScanToggle.bind(this);
  }

  /**
   * Initialize the application
   */
  async init() {
    try {
      console.log('Initializing Karst Intelligence Agent (Real Data Mode)...');

      // Load configuration (use default if API fails so map always shows)
      try {
        this.config = await window.api.getConfig();
        console.log('Config loaded:', this.config);
      } catch (configErr) {
        console.warn('Config fetch failed, using default:', configErr);
        const bbox = [-87.6, 24.5, -80.0, 31.0]; // [minLon, minLat, maxLon, maxLat]
        this.config = {
          aoi: {
            name: 'Florida',
            center: [27.8, -81.8],
            area_km2: 170000,
            bbox: bbox,
            geojson: {
              type: 'Polygon',
              coordinates: [[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]],
            },
          },
          map: { minZoom: 6, maxZoom: 18, defaultZoom: 7, tileSize: 512 },
          features: { geminiEnabled: false, modelTrained: false },
        };
      }

      // Update AOI info
      this._updateAOIInfo();

      // Initialize map (NO heatmap initially) – always run so map is never "gone"
      this.map = new SinkholeMap('map');
      await this.map.init(this.config);
      this.map.onMapClick = this.handleMapClick;
      // Force Leaflet to recalc size (fixes blank map when container is in grid)
      var self = this;
      setTimeout(function () {
        if (self.map && self.map.map) self.map.map.invalidateSize();
      }, 300);

      // Initialize scanner with map reference
      this.scanner = new ScannerController();
      this.scanner.setMap(this.map);
      this._setupScannerCallbacks();

      // Set up UI event handlers
      this._setupEventHandlers();

      // Connect WebSocket
      try { window.api.connectWebSocket(); } catch (_) {}

      // Check Gemini status
      this._checkGeminiStatus();

      console.log('Karst Intelligence Agent initialized');
      console.log('Click START AGENT to run full pipeline (scan → AI Analysis → monitoring)');
    } catch (error) {
      console.error('Failed to initialize app:', error);
      this._showError(
        'Failed to initialize application. Please refresh the page.',
      );
    }
  }

  /**
   * Check Gemini 3 API status
   */
  async _checkGeminiStatus() {
    const statusBadge = document.getElementById('gemini-available');
    const analysisBtn = document.getElementById('start-agent-analysis');
    
    if (statusBadge) {
      statusBadge.textContent = 'Checking...';
      statusBadge.className = 'status-badge checking';
    }
    
    try {
      const status = await window.api.getGeminiStatus();
      console.log('Gemini status:', status);
      
      if (statusBadge) {
        if (status.available) {
          statusBadge.textContent = 'Available';
          statusBadge.className = 'status-badge available';
          if (analysisBtn) analysisBtn.disabled = false;
        } else {
          statusBadge.textContent = status.error ? 'Error' : 'Unavailable';
          statusBadge.className = 'status-badge unavailable';
          if (analysisBtn) analysisBtn.disabled = true;
        }
      }
    } catch (error) {
      console.error('Failed to check Gemini status:', error);
      if (statusBadge) {
        statusBadge.textContent = 'Error';
        statusBadge.className = 'status-badge unavailable';
      }
    }
  }

  /**
   * Load model metrics from API
   */
  async _loadModelMetrics() {
    try {
      const response = await fetch('/api/analysis/model/metrics');
      const data = await response.json();

      if (!data.metrics) {
        console.log('Model not trained yet - train the model first');
        return;
      }

      console.log('Loaded model metrics:', data);
      this._displayModelMetrics(data);
      this._displayFeatureImportance(data.feature_importance);
    } catch (error) {
      console.log('Could not load model metrics:', error);
    }
  }

  /**
   * Display model evaluation metrics in UI
   * Supports new Spatial CV format with fold results
   */
  _displayModelMetrics(data) {
    const container = document.getElementById('model-metrics-container');
    if (!container) return;

    // Defensive checks for undefined data
    if (!data || !data.metrics) {
      container.innerHTML =
        '<div class="no-data">Model metrics not available</div>';
      return;
    }

    const metrics = data.metrics;
    const cm = data.confusion_matrix || {};
    const training = data.training || {};
    // Helper function for safe number formatting
    const safeFixed = (val, digits = 3) => {
      const num = parseFloat(val);
      return isNaN(num) ? '0.000' : num.toFixed(digits);
    };

    // Calculate fold stats if available
    let foldInfo = '';
    if (training.fold_results && training.fold_results.length > 0) {
      const folds = training.fold_results;
      const aucValues = folds.map((f) => f.auc || 0);
      const minAuc = Math.min(...aucValues);
      const maxAuc = Math.max(...aucValues);

      foldInfo = `
                <div class="fold-details">
                    <span class="fold-stat">Folds: ${folds.length}</span>
                    <span class="fold-stat">AUC range: ${safeFixed(minAuc, 2)} - ${safeFixed(maxAuc, 2)}</span>
                </div>
            `;
    }

    const aucRoc = metrics.auc_roc || metrics.auc || 0;
    const precision = metrics.precision || 0;
    const recall = metrics.recall || 0;
    const f1Score = metrics.f1_score || metrics.f1 || 0;
    const accuracy = metrics.accuracy || 0;

    container.innerHTML = `
            <div class="metrics-grid">
                <div class="metric-card ${aucRoc > 0.7 ? 'good' : aucRoc > 0.5 ? 'moderate' : 'poor'}">
                    <span class="metric-value">${safeFixed(aucRoc)}</span>
                    <span class="metric-label">AUC-ROC</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value">${safeFixed(precision)}</span>
                    <span class="metric-label">Precision</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value">${safeFixed(recall)}</span>
                    <span class="metric-label">Recall</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value">${safeFixed(f1Score)}</span>
                    <span class="metric-label">F1 Score</span>
                </div>
            </div>
            
            ${foldInfo}
            
            <div class="confusion-matrix">
                <h4 class="matrix-title">Confusion Matrix (Test Set)</h4>
                <div class="matrix-grid">
                    <div class="matrix-corner"></div>
                    <div class="matrix-header">Pred: No</div>
                    <div class="matrix-header">Pred: Yes</div>
                    <div class="matrix-label">Actual: No</div>
                    <div class="matrix-cell tn">${cm.true_negative || cm.tn || 0}</div>
                    <div class="matrix-cell fp">${cm.false_positive || cm.fp || 0}</div>
                    <div class="matrix-label">Actual: Yes</div>
                    <div class="matrix-cell fn">${cm.false_negative || cm.fn || 0}</div>
                    <div class="matrix-cell tp">${cm.true_positive || cm.tp || 0}</div>
                </div>
            </div>
            
            <div class="training-info">
                <div class="info-row">
                    <span class="info-label">Test samples:</span>
                    <span class="info-value">${training.total_test_samples || training.test_samples || 'N/A'}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Model:</span>
                    <span class="info-value">${data.model_type || 'XGBoost'}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Accuracy:</span>
                    <span class="info-value">${safeFixed(accuracy * 100, 1)}%</span>
                </div>
            </div>
        `;
  }

  /**
   * Display feature importance in UI
   */
  _displayFeatureImportance(featureImportance) {
    const container = document.getElementById('input-features-container');
    if (!container || !featureImportance) return;

    // Feature name mapping for display (includes Sentinel and ground movement used in inference)
    const featureLabels = {
      dist_to_sinkhole: 'Distance to Sinkholes',
      ground_displacement: 'Ground Displacement (InSAR)',
      sentinel_optical: 'Sentinel-2 Optical (NDVI/NDWI)',
      curvature: 'Terrain Curvature',
      elevation: 'Elevation (DEM)',
      sink_depth: 'Sink/Depression Depth',
      slope: 'Slope',
      dist_to_water: 'Distance to Water',
      karst_presence: 'Karst Presence',
      sinkhole_proximity: 'Sinkhole Proximity',
      karst_geology: 'Karst Geology',
      terrain: 'Terrain Features',
      water: 'Water Proximity',
    };

    // Sort by importance
    const sorted = Object.entries(featureImportance).sort(
      (a, b) => b[1] - a[1],
    );

    // Colors for bars (gradient from cyan to dark blue)
    const colors = [
      '#00ffc8',
      '#00b4d8',
      '#0096c7',
      '#0077b6',
      '#023e8a',
      '#03045e',
      '#370617',
    ];

    let html = '';
    sorted.forEach(([name, importance], idx) => {
      const percent = (importance * 100).toFixed(2);
      const label = featureLabels[name] || name;
      const color = colors[Math.min(idx, colors.length - 1)];

      html += `
                <div class="input-feature">
                    <span class="feature-bar" style="width: ${percent}%; background: ${color};"></span>
                    <span class="feature-label">${label}</span>
                    <span class="feature-importance">${percent}%</span>
                </div>
            `;
    });

    container.innerHTML = html;
  }

  /**
   * Update AOI information display
   */
  _updateAOIInfo() {
    const areaEl = document.getElementById('aoi-area');
    if (areaEl && this.config) {
      areaEl.textContent = `${this.config.aoi.area_km2.toFixed(2)} km²`;
    }
    const locationLabel = document.getElementById('location-label');
    if (locationLabel && this.config && this.config.aoi) {
      locationLabel.textContent = this.config.aoi.name;
    }
  }

  /**
   * Train model only (no scan) - called from Train model button
   */
  async _startTrainingOnly() {
    const base = (typeof window !== 'undefined' && window.api && window.api.baseUrl) ? window.api.baseUrl : '';
    const startTraining = typeof window.api?.startTraining === 'function'
      ? () => window.api.startTraining()
      : async () => {
          const r = await fetch(`${base}/api/analysis/model/train`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
          if (!r.ok) throw new Error('Failed to start training');
          return r.json();
        };
    const getTrainingStatus = typeof window.api?.getTrainingStatus === 'function'
      ? (id) => window.api.getTrainingStatus(id)
      : async (id) => {
          const r = await fetch(`${base}/api/analysis/model/train/status/${id}`);
          if (!r.ok) throw new Error('Failed to get status');
          return r.json();
        };
    try {
      this._showNotification('Starting model training...');
      const { training_id, status } = await startTraining();
      this._showNotification(`Training started (${training_id}). Check scan progress or wait for completion.`);
      const poll = async () => {
        try {
          const s = await getTrainingStatus(training_id);
          if (s.status === 'complete') {
            this._showNotification('Training complete! Model saved.');
            if (this._displayModelMetrics && s.metrics) {
              this._displayModelMetrics(s.metrics);
            }
            return;
          }
          if (s.status === 'failed') {
            this._showNotification(`Training failed: ${s.message || 'Unknown error'}`);
            return;
          }
          setTimeout(poll, 3000);
        } catch (e) {
          this._showNotification(`Training status check failed: ${e.message}`);
        }
      };
      setTimeout(poll, 2000);
    } catch (error) {
      this._showNotification(`Failed to start training: ${error.message}`);
    }
  }

  /**
   * Set up scanner callbacks
   */
  _setupScannerCallbacks() {
    this.scanner.onScanStart = () => {
      console.log('[App] Scan started');
      const mode = this.scanner.getScanMode();
      const modeText =
        mode === 'quick'
          ? 'Quick Agent (pretrained model)'
          : 'Train + Agent (full pipeline)';
      this._showNotification(`Starting ${modeText}...`);
    };

    this.scanner.onScanProgress = (status) => {
      // Progress is shown in UI automatically via scanner.js
    };

    this.scanner.onScanComplete = async (status) => {
      console.log('[App] Scan complete:', status);
      const seconds = (status.elapsed_ms / 1000).toFixed(1);
      const modeText = status.mode === 'quick' ? 'Quick Agent' : 'Train + Agent';
      this._showNotification(
        `${modeText} complete! ${status.features_found} sinkholes found. Starting AI Analysis...`,
      );

      // Activate heatmap toggle
      const toggleHeatmap = document.getElementById('toggle-heatmap');
      if (toggleHeatmap) {
        toggleHeatmap.classList.add('active');
      }
      
      // AUTONOMOUS: Run AI Analysis automatically (no user click)
      try {
        console.log('[App] Auto-starting AI Analysis...');
        await this._startAgentAnalysis();
      } catch (e) {
        console.warn('[App] Auto AI Analysis failed:', e);
        this._showNotification('AI Analysis could not start automatically. You can run it manually.');
      }
    };

    this.scanner.onScanError = (error) => {
      console.error('[App] Scan error:', error);
      this._showError(`Scan failed: ${error.message}`);
    };

    // Handle metrics loaded from scanner (during model phase)
    this.scanner.onMetricsLoaded = (data) => {
      console.log('[App] Metrics loaded from scanner:', data);
      this._displayModelMetrics(data);
      this._displayFeatureImportance(data.feature_importance);
    };
  }

  /**
   * Set up UI event handlers
   */
  _setupEventHandlers() {
    // Scan button
    const scanBtn = document.getElementById('start-scan-btn');
    if (scanBtn) {
      scanBtn.addEventListener('click', this.handleScanToggle);
    }

    // Map controls
    const toggleHeatmap = document.getElementById('toggle-heatmap');
    const toggleFeatures = document.getElementById('toggle-features');
    const toggleBasemap = document.getElementById('toggle-basemap');

    if (toggleHeatmap) {
      // Start inactive (heatmap not loaded until scan)
      toggleHeatmap.classList.remove('active');
      toggleHeatmap.addEventListener('click', () => {
        if (!this.map.scanComplete) {
          this._showNotification('Run a scan first to generate the heatmap');
          return;
        }
        const isActive = toggleHeatmap.classList.toggle('active');
        this.map.setHeatmapVisible(isActive);
      });
    }

    if (toggleFeatures) {
      toggleFeatures.classList.add('active');
      toggleFeatures.addEventListener('click', () => {
        const isActive = toggleFeatures.classList.toggle('active');
        // Toggle sinkhole markers visibility
        if (this.map.sinkholeMarkers) {
          if (isActive) {
            this.map.sinkholeMarkers.addTo(this.map.map);
          } else {
            this.map.map.removeLayer(this.map.sinkholeMarkers);
          }
        }
      });
    }

    if (toggleBasemap) {
      toggleBasemap.addEventListener('click', () => {
        const basemap = this.map.cycleBasemap();
        console.log('Switched to basemap:', basemap);
      });
    }

    // Gemini AI Analysis button
    const agentAnalysisBtn = document.getElementById('start-agent-analysis');
    if (agentAnalysisBtn) {
      agentAnalysisBtn.addEventListener('click', () => this._startAgentAnalysis());
    }
    
    // Early Warning Monitoring buttons
    const monitorBtn = document.getElementById('btn-start-monitoring');
    if (monitorBtn) {
      monitorBtn.addEventListener('click', () => this._toggleMonitoring());
    }
    
    const checkNowBtn = document.getElementById('btn-check-now');
    if (checkNowBtn) {
      checkNowBtn.addEventListener('click', () => this._triggerMonitoringCheck());
    }

    // Train model only (no scan)
    const trainModelBtn = document.getElementById('train-model-btn');
    if (trainModelBtn) {
      trainModelBtn.addEventListener('click', () => this._startTrainingOnly());
    }

    // Load Terrain button is attached at end of script (works even if init fails)
    
    // Check monitoring status on load
    this._checkMonitoringStatus();
  }
  
  /**
   * Check current monitoring status
   */
  async _checkMonitoringStatus() {
    try {
      const base = (typeof window !== 'undefined' && window.api && window.api.baseUrl) ? window.api.baseUrl : '';
      const response = await fetch(`${base}/api/analysis/monitoring/status`);
      const status = await response.json();
      
      this._updateMonitoringUI(status);
      
      // Update data source statuses
      if (status.data_sources) {
        // GPS data source
        if (status.data_sources.gps?.available) {
          this._setDataSourceStatus('src-gps', 'loaded', 
            `${status.data_sources.gps.station} connected`);
        }
      }
      
      // Update combined risk display if available
      if (status.latest_measurements?.combined_risk) {
        this._updateCombinedRiskDisplay(status.latest_measurements.combined_risk);
      }
      
      // Update GPS panel if available
      if (status.latest_measurements?.gps) {
        this._updateGPSPanel(status.latest_measurements.gps);
      }
    } catch (error) {
      console.log('Could not check monitoring status:', error);
    }
  }
  
  /**
   * Start monitoring only if not already active (used by autonomous pipeline)
   */
  async _startMonitoringIfInactive() {
    const base = (typeof window !== 'undefined' && window.api && window.api.baseUrl) ? window.api.baseUrl : '';
    const statusResp = await fetch(`${base}/api/analysis/monitoring/status`);
    const status = await statusResp.json();
    if (status.active) return;
    this._addMonitoringLog('info', 'Starting early warning agent (autonomous)...');
    const response = await fetch(`${base}/api/analysis/monitoring/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ check_interval_minutes: 5 }),
    });
    await response.json();
    this._addMonitoringLog('success', 'Agent active - checking ground movement every 5 min');
    this._updateMonitoringUI({ active: true });
    const checkBtn = document.getElementById('btn-check-now');
    if (checkBtn) checkBtn.disabled = false;
    const riskDisplay = document.getElementById('combined-risk-display');
    const gpsPanel = document.getElementById('gps-data-panel');
    if (riskDisplay) riskDisplay.style.display = 'block';
    if (gpsPanel) gpsPanel.style.display = 'block';
    this._triggerMonitoringCheck();
  }

  /**
   * Toggle monitoring on/off
   */
  async _toggleMonitoring() {
    const btn = document.getElementById('btn-start-monitoring');
    const checkBtn = document.getElementById('btn-check-now');
    
    try {
      // Check current state
      const base = (typeof window !== 'undefined' && window.api && window.api.baseUrl) ? window.api.baseUrl : '';
      const statusResp = await fetch(`${base}/api/analysis/monitoring/status`);
      const status = await statusResp.json();
      
      if (status.active) {
        // Stop monitoring
        const response = await fetch(`${base}/api/analysis/monitoring/stop`, { method: 'POST' });
        const result = await response.json();
        this._addMonitoringLog('info', 'Agent stopped - autonomous monitoring disabled');
        this._updateMonitoringUI({ active: false });
        if (checkBtn) checkBtn.disabled = true;
      } else {
        // Start monitoring
        this._addMonitoringLog('info', 'Starting early warning agent...');
        const response = await fetch(`${base}/api/analysis/monitoring/start`, { 
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ check_interval_minutes: 5 })  // Check every 5 min for demo
        });
        const result = await response.json();
        this._addMonitoringLog('success', 'Agent active - checking ground movement every 5 min');
        this._updateMonitoringUI({ active: true });
        if (checkBtn) checkBtn.disabled = false;
        
        // Show panels
        const riskDisplay = document.getElementById('combined-risk-display');
        const gpsPanel = document.getElementById('gps-data-panel');
        if (riskDisplay) riskDisplay.style.display = 'block';
        if (gpsPanel) gpsPanel.style.display = 'block';
        
        // Trigger immediate check
        this._triggerMonitoringCheck();
      }
    } catch (error) {
      console.error('Monitoring toggle error:', error);
      this._addMonitoringLog('alert', `Error: ${error.message}`);
    }
  }
  
  /**
   * Trigger an immediate monitoring check (agentic system)
   */
  async _triggerMonitoringCheck() {
    const checkBtn = document.getElementById('btn-check-now');
    if (checkBtn) {
      checkBtn.disabled = true;
      checkBtn.innerHTML = '<span class="btn-icon spinning">⟳</span> Checking...';
    }
    this._addMonitoringLog('info', 'Running agentic ground movement check...');
    
    try {
      const base = (typeof window !== 'undefined' && window.api && window.api.baseUrl) ? window.api.baseUrl : '';
      const response = await fetch(`${base}/api/analysis/monitoring/check-now`, { method: 'POST' });
      const result = await response.json();
      
      console.log('Monitoring check result:', result);
      
      // Update GPS data source and panel
      if (result.measurements?.gps && !result.measurements.gps.error) {
        this._setDataSourceStatus('src-gps', 'loaded', 
          `${result.measurements.gps.station} - ${result.measurements.gps.vertical_velocity_mm_year?.toFixed(2)} mm/yr`);
        this._updateGPSPanel(result.measurements.gps);
      } else {
        this._setDataSourceStatus('src-gps', 'error', 'GPS unavailable');
      }
      
      // Update combined risk display
      if (result.combined_risk) {
        this._updateCombinedRiskDisplay(result.combined_risk);
      }
      
      // Process alerts
      if (result.alerts_generated && result.alerts_generated.length > 0) {
        for (const alert of result.alerts_generated) {
          const rs = alert.combined_risk_score;
          this._addMonitoringLog('alert', 
            `🚨 ${alert.level}: ${alert.trigger_name} - Risk score: ${rs != null ? rs.toFixed(2) : 'N/A'}`);
          
          // Show the full Gemini-drafted alert message (ready to send to authority)
          if (alert.gemini_message) {
            this._addMonitoringLog('alert', `--- ALERT MESSAGE (review & send) ---`);
            this._showAlertMessage(alert);
          }
        }
      } else if (result.status === 'normal') {
        const s = result.combined_risk?.score;
        this._addMonitoringLog('success', 
          `Normal - Combined risk: ${s != null ? s.toFixed(2) : 'N/A'} (${result.combined_risk?.level || 'N/A'})`);
      } else if (result.status === 'elevated') {
        const s = result.combined_risk?.score;
        this._addMonitoringLog('warning', 
          `Elevated risk: ${s != null ? s.toFixed(2) : 'N/A'} - monitoring closely`);
      }
      
      // Log trigger evaluations
      if (result.triggers_evaluated) {
        const firedTriggers = result.triggers_evaluated.filter(t => t.fired);
        if (firedTriggers.length > 0) {
          console.log('Fired triggers:', firedTriggers);
        }
      }
      
    } catch (error) {
      this._addMonitoringLog('alert', `Check failed: ${error.message}`);
      console.error('Monitoring check error:', error);
    }
  }
  
  /**
   * Update GPS panel with latest data
   */
  _updateGPSPanel(gpsData) {
    const panel = document.getElementById('gps-data-panel');
    if (!panel || !gpsData) return;
    
    panel.style.display = 'block';
    
    const stationId = document.getElementById('gps-station-id');
    const velocity = document.getElementById('gps-velocity');
    const distance = document.getElementById('gps-distance');
    const dataAge = document.getElementById('gps-data-age');
    const status = document.getElementById('gps-status');
    
    if (stationId) stationId.textContent = gpsData.station || 'FLOL';
    if (velocity) {
      const vel = gpsData.vertical_velocity_mm_year;
      if (vel != null && typeof vel === 'number') {
        velocity.textContent = vel.toFixed(2);
        velocity.className = vel < -3 ? 'metric-value warning' : 'metric-value';
      } else {
        velocity.textContent = '—';
        velocity.className = 'metric-value';
      }
    }
    if (distance) distance.textContent = gpsData.distance_to_aoi_km != null ? Number(gpsData.distance_to_aoi_km).toFixed(1) : '—';
    if (dataAge) dataAge.textContent = gpsData.date_range || '—';
    if (status) {
      status.textContent = gpsData.is_subsiding ? 'SUBSIDING' : 'STABLE';
      status.className = gpsData.is_subsiding ? 'source-status warning' : 'source-status live';
    }
  }
  
  /**
   * Update combined risk score display
   */
  _updateCombinedRiskDisplay(riskData) {
    const display = document.getElementById('combined-risk-display');
    if (!display || !riskData) return;
    
    display.style.display = 'block';
    
    const gaugeFill = document.getElementById('risk-gauge-fill');
    const scoreEl = document.getElementById('combined-risk-score');
    const levelBadge = document.getElementById('risk-level-badge');
    const staticRisk = document.getElementById('static-risk');
    const movementRisk = document.getElementById('movement-risk');
    
    const score = riskData.score != null ? riskData.score : null;
    const level = riskData.level || 'NORMAL';
    const baseSusc = riskData.components?.base_susceptibility;
    const movementR = riskData.components?.movement_risk;
    
    if (gaugeFill) {
      gaugeFill.style.width = score != null ? `${score * 100}%` : '0%';
      gaugeFill.className = `risk-gauge-fill ${level.toLowerCase().replace('_', '-')}`;
    }
    if (scoreEl) scoreEl.textContent = score != null ? score.toFixed(2) : '—';
    if (levelBadge) {
      levelBadge.textContent = level;
      levelBadge.className = `risk-level-badge ${level.toLowerCase().replace('_', '-')}`;
    }
    if (staticRisk) staticRisk.textContent = baseSusc != null ? baseSusc.toFixed(2) : '—';
    if (movementRisk) movementRisk.textContent = movementR != null ? movementR.toFixed(2) : '—';
  }
  
  /**
   * Update monitoring UI based on status
   */
  _updateMonitoringUI(status) {
    const btn = document.getElementById('btn-start-monitoring');
    const statusBlock = document.getElementById('monitoring-status');
    const indicator = document.querySelector('#monitoring-status .status-indicator');
    const statusText = document.querySelector('#monitoring-status .status-text');
    
    if (status.active) {
      if (btn) {
        btn.innerHTML = '<span class="btn-icon">⏹️</span> Stop Monitoring';
        btn.classList.add('monitoring');
        btn.disabled = false;
      }
      if (statusBlock) statusBlock.classList.add('monitoring-active');
      if (indicator) {
        indicator.className = 'status-indicator active rolling';
      }
      if (statusText) {
        statusText.textContent = 'Monitoring…';
      }
    } else {
      if (btn) {
        btn.innerHTML = '<span class="btn-icon">📡</span> Start Monitoring';
        btn.classList.remove('monitoring');
        btn.disabled = false;
      }
      if (statusBlock) statusBlock.classList.remove('monitoring-active');
      if (indicator) {
        indicator.className = 'status-indicator inactive';
      }
      if (statusText) {
        statusText.textContent = 'Agent Inactive';
      }
    }
  }
  
  /**
   * Add entry to monitoring log
   */
  _addMonitoringLog(type, message) {
    const log = document.getElementById('monitoring-log');
    if (!log) return;
    
    const timestamp = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${timestamp}] ${message}`;
    
    log.insertBefore(entry, log.firstChild);
    
    // Keep only last 30 entries (more when alerts are shown)
    while (log.children.length > 30) {
      log.removeChild(log.lastChild);
    }
  }
  
  /**
   * Show full Gemini-drafted alert message in the monitoring log (ready to send to authority)
   */
  _showAlertMessage(alert) {
    const log = document.getElementById('monitoring-log');
    if (!log || !alert.gemini_message) return;
    
    const block = document.createElement('div');
    block.className = 'log-entry alert alert-message-block';
    block.style.whiteSpace = 'pre-wrap';
    block.style.maxHeight = '200px';
    block.style.overflowY = 'auto';
    block.style.marginTop = '4px';
    block.style.padding = '8px';
    block.textContent = alert.gemini_message.trim();
    
    log.insertBefore(block, log.firstChild);
    
    // Keep only last 30 entries
    while (log.children.length > 30) {
      log.removeChild(log.lastChild);
    }
  }
  
  /**
   * Helper to set data source status from app.js
   */
  _setDataSourceStatus(sourceId, status, detail) {
    const sourceEl = document.getElementById(sourceId);
    if (!sourceEl) return;
    
    const statusEl = sourceEl.querySelector('.source-status');
    const detailEl = sourceEl.querySelector('.source-detail');
    
    if (statusEl) {
      statusEl.className = `source-status ${status}`;
      statusEl.textContent = status === 'loaded' ? '●' : status === 'loading' ? '○' : status === 'error' ? '✗' : '○';
    }
    if (detailEl && detail) {
      detailEl.textContent = detail;
    }
  }

  /**
   * Start Gemini 3 agentic analysis
   */
  async _startAgentAnalysis() {
    const btn = document.getElementById('start-agent-analysis');
    const resultsDiv = document.getElementById('gemini-results');
    
    if (btn) {
      btn.disabled = true;
      btn.classList.add('running');
      btn.innerHTML = '<span class="btn-icon">⏳</span> Analyzing...';
    }
    
    this._showNotification('Starting Gemini 3 AI analysis...');
    
    try {
      // Start the analysis
      const response = await window.api.startAgentAnalysis({
        includeSatellite: true,
        thinkingLevel: 'high'
      });
      
      console.log('Agent analysis started:', response);
      
      // Poll for completion
      await this._pollAgentAnalysis(response.analysis_id);
      
    } catch (error) {
      console.error('Agent analysis failed:', error);
      this._showError(`AI analysis failed: ${error.message}`);
      
      if (btn) {
        btn.disabled = false;
        btn.classList.remove('running');
        btn.innerHTML = '<span class="btn-icon">🧠</span> Run AI Analysis';
      }
    }
  }

  /**
   * Poll for agent analysis completion
   */
  async _pollAgentAnalysis(analysisId) {
    const btn = document.getElementById('start-agent-analysis');
    const maxAttempts = 120; // 10 minutes max
    let attempts = 0;
    
    while (attempts < maxAttempts) {
      try {
        const status = await window.api.getAgentStatus(analysisId);
        console.log('Agent status:', status);
        
        if (status.status === 'completed') {
          // Get the full report
          const report = await window.api.getAgentReport(analysisId);
          console.log('Agent report:', report);
          
          this._displayGeminiResults(report);
          this._showNotification(`AI analysis complete! Starting monitoring...`);
          
          if (btn) {
            btn.disabled = false;
            btn.classList.remove('running');
            btn.innerHTML = '<span class="btn-icon">🧠</span> Run AI Analysis';
          }
          // AUTONOMOUS: Start monitoring automatically (no user click)
          try {
            await this._startMonitoringIfInactive();
          } catch (e) {
            console.warn('[App] Auto-start monitoring failed:', e);
          }
          return;
          
        } else if (status.status === 'failed') {
          const errMsg = status.error || 'Analysis failed';
          this._displayGeminiError(errMsg);
          throw new Error(errMsg);
        }
        
        // Update UI with current step
        if (status.current_step) {
          if (btn) {
            btn.innerHTML = `<span class="btn-icon">⏳</span> ${status.current_step}...`;
          }
        }
        
        // Wait before next poll
        await new Promise(resolve => setTimeout(resolve, 5000));
        attempts++;
        
      } catch (error) {
        console.error('Poll error:', error);
        throw error;
      }
    }
    
    throw new Error('Analysis timed out');
  }

  /**
   * Display Gemini analysis error (e.g. failed to start or backend error)
   */
  _displayGeminiError(message) {
    const resultsDiv = document.getElementById('gemini-results');
    const riskLevel = document.getElementById('gemini-risk-level');
    const confidence = document.getElementById('gemini-confidence');
    const reasoningText = document.getElementById('gemini-reasoning-text');
    const recommendationsList = document.getElementById('gemini-recommendations-list');
    if (resultsDiv) resultsDiv.style.display = 'block';
    if (riskLevel) {
      riskLevel.textContent = 'ERROR';
      riskLevel.className = 'risk-value gemini-risk error';
    }
    if (confidence) confidence.textContent = '0% confidence';
    if (reasoningText) reasoningText.textContent = message;
    if (recommendationsList) recommendationsList.innerHTML = '<li>Check Gemini configuration (Vertex AI or GEMINI_API_KEY) and try again.</li>';
  }

  /**
   * Display Gemini analysis results
   */
  _displayGeminiResults(report) {
    const resultsDiv = document.getElementById('gemini-results');
    const riskLevel = document.getElementById('gemini-risk-level');
    const confidence = document.getElementById('gemini-confidence');
    const reasoningText = document.getElementById('gemini-reasoning-text');
    const recommendationsList = document.getElementById('gemini-recommendations-list');
    
    if (resultsDiv) {
      resultsDiv.style.display = 'block';
    }
    
    const level = report.risk_assessment?.level || 'unknown';
    const conf = report.risk_assessment?.confidence ?? 0;
    const reasoning = report.analysis_reasoning || '';
    const noResult = (level === 'unknown' && conf === 0 && !reasoning.trim());
    
    // Risk level
    if (riskLevel) {
      riskLevel.textContent = level.toUpperCase().replace('_', ' ');
      riskLevel.className = `risk-value gemini-risk ${level.toLowerCase().replace('_', '-')}`;
    }
    
    // Confidence
    if (confidence) {
      confidence.textContent = `${Math.round(conf * 100)}% confidence`;
    }
    
    // Reasoning
    if (reasoningText) {
      if (noResult) {
        reasoningText.textContent = 'Analysis did not produce a result. Check that Gemini is configured (Vertex AI or GEMINI_API_KEY) and try "Run AI Analysis" again.';
      } else {
        reasoningText.textContent = reasoning.substring(0, 500) + (reasoning.length > 500 ? '...' : '');
      }
    }
    
    // Recommendations
    if (recommendationsList) {
      recommendationsList.innerHTML = '';
      const recs = report.recommendations || [];
      if (recs.length === 0 && noResult) {
        const li = document.createElement('li');
        li.textContent = 'Ensure USE_VERTEX_AI=true and GOOGLE_CLOUD_PROJECT (and credentials) are set, or set GEMINI_API_KEY for AI Studio.';
        recommendationsList.appendChild(li);
      } else {
        recs.slice(0, 5).forEach(rec => {
          const li = document.createElement('li');
          li.textContent = typeof rec === 'string' ? rec : String(rec);
          if (typeof rec === 'string' && rec.includes('[HIGH]')) li.className = 'high-priority';
          recommendationsList.appendChild(li);
        });
      }
    }
  }

  /**
   * Handle scan button toggle
   */
  handleScanToggle() {
    if (this.scanner.isScanning) {
      this.scanner.stopScan();
      this._showNotification('Scan stopped');
    } else {
      this.scanner.startScan({
        zoom: 14,
        includeGemini: this.config?.features?.geminiEnabled,
      });
    }
  }

  /**
   * Handle map click for point query
   */
  async handleMapClick(latlng) {
    if (this.scanner.isScanning) {
      return; // Don't query while scanning
    }

    try {
      // Check if point is in AOI
      const bbox = this.config.aoi.bbox;
      if (
        latlng.lng < bbox[0] ||
        latlng.lng > bbox[2] ||
        latlng.lat < bbox[1] ||
        latlng.lat > bbox[3]
      ) {
        this._showNotification(
          'Click inside the Winter Park AOI to query susceptibility.',
        );
        return;
      }

      // Set marker
      this.map.setQueryMarker(latlng);

      // Query API
      const result = await window.api.queryPoint(latlng.lat, latlng.lng);

      // Update UI
      this._displayQueryResult(result);
    } catch (error) {
      console.error('Failed to query point:', error);
      this._showError('Failed to query location. Please try again.');
    }
  }

  /**
   * Display point query result
   */
  _displayQueryResult(result) {
    const container = document.getElementById('query-result');
    if (!container) return;

    container.style.display = 'block';

    // Coordinates
    const coordsEl = document.getElementById('query-coords');
    if (coordsEl) {
      coordsEl.textContent = `${result.lat.toFixed(6)}, ${result.lon.toFixed(6)}`;
    }

    // Risk level
    const riskEl = document.getElementById('query-risk-value');
    if (riskEl) {
      riskEl.textContent = result.risk_level;
      riskEl.className =
        'risk-value ' + result.risk_level.toLowerCase().replace(/[- ]/g, '-');
    }

    // Susceptibility bar
    const susceptFill = document.getElementById('query-suscept-fill');
    const susceptValue = document.getElementById('query-suscept-value');

    if (susceptFill) {
      const percent = result.susceptibility * 100;
      susceptFill.style.width = `${percent}%`;

      // Color based on value
      const color = this._getSusceptibilityColor(result.susceptibility);
      susceptFill.style.background = color;
    }

    if (susceptValue) {
      susceptValue.textContent = `${(result.susceptibility * 100).toFixed(1)}%`;
    }
  }

  /**
   * Get color for susceptibility value
   */
  _getSusceptibilityColor(value) {
    if (value < 0.2) return '#22c55e';
    if (value < 0.4) return '#84cc16';
    if (value < 0.6) return '#eab308';
    if (value < 0.8) return '#f97316';
    return '#dc2626';
  }

  /**
   * Show notification
   */
  _showNotification(message) {
    console.log('Notification:', message);

    const toast = document.createElement('div');
    toast.style.cssText = `
            position: fixed;
            bottom: 80px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(10, 15, 20, 0.95);
            border: 1px solid #00ffc8;
            color: #e0e0e0;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 0.9rem;
            z-index: 10000;
            font-family: 'JetBrains Mono', monospace;
            animation: fadeInUp 0.3s ease;
        `;
    toast.textContent = message;

    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 0.3s';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  /**
   * Show error
   */
  _showError(message) {
    console.error('Error:', message);

    const toast = document.createElement('div');
    toast.style.cssText = `
            position: fixed;
            bottom: 80px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(220, 38, 38, 0.95);
            border: 1px solid #dc2626;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 0.9rem;
            z-index: 10000;
            font-family: 'JetBrains Mono', monospace;
            animation: fadeInUp 0.3s ease;
        `;
    toast.textContent = message;

    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 0.3s';
      setTimeout(() => toast.remove(), 300);
    }, 5000);
  }
  
  /**
   * HYBRID SYSTEM: Get Gemini AI validation of ML scan results
   */
  async _getGeminiValidation() {
    try {
      const response = await fetch('/api/analysis/validate-current', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (!response.ok) {
        throw new Error(`Validation failed: ${response.status}`);
      }
      const result = await response.json();
      
      if (result.ground_displacement) {
        this._addMonitoringLog('success', 'Ground displacement data loaded');
      }
      
      // Check for alerts from the validation
      if (result.hybrid_assessment?.alert_recommended) {
        this._addMonitoringLog('alert', 
          `ALERT ${result.hybrid_assessment.alert_level}: ${result.hybrid_assessment.alert_message || 'Check recommended'}`);
      }
      
      return result;
    } catch (e) {
      console.error('[App] Gemini validation error:', e);
      return null;
    }
  }
  
  /**
   * Display hybrid ML + Gemini results
   */
  _displayHybridResults(validation) {
    console.log('[App] Displaying hybrid results:', validation);
    
    // Update the Gemini results section
    const geminiSection = document.getElementById('gemini-results');
    if (!geminiSection) return;
    
    const hybrid = validation.hybrid_assessment || {};
    const mlMetrics = validation.ml_metrics || {};
    const geminiValidation = validation.gemini_validation || {};
    
    // Build the results HTML
    const riskCategory = geminiValidation.risk_category || 'unknown';
    const confidence = geminiValidation.confidence_percent || 0;
    const assessment = geminiValidation.overall_assessment || hybrid.combined_interpretation || '';
    const recommendations = geminiValidation.recommendations || [];
    
    // Risk level styling
    const riskColors = {
      'low': '#22c55e',
      'moderate': '#eab308', 
      'elevated': '#f97316',
      'high': '#ef4444',
      'very_high': '#dc2626'
    };
    const riskColor = riskColors[riskCategory] || '#6b7280';
    
    // Build recommendations HTML
    let recsHtml = '';
    if (recommendations.length > 0) {
      recsHtml = '<div class="gemini-recommendations"><h4>AI Recommendations</h4><ul>';
      recommendations.forEach(rec => {
        const priority = rec.priority || 'medium';
        const priorityClass = priority === 'high' ? 'priority-high' : 
                             priority === 'low' ? 'priority-low' : 'priority-medium';
        recsHtml += `<li class="${priorityClass}">
          <strong>[${priority.toUpperCase()}]</strong> ${rec.recommendation || rec}
        </li>`;
      });
      recsHtml += '</ul></div>';
    }
    
    // Key findings
    let findingsHtml = '';
    const findings = geminiValidation.key_findings || [];
    if (findings.length > 0) {
      findingsHtml = '<div class="gemini-findings"><h4>Key Findings</h4><ul>';
      findings.forEach(f => {
        findingsHtml += `<li>${f}</li>`;
      });
      findingsHtml += '</ul></div>';
    }
    
    geminiSection.innerHTML = `
      <div class="hybrid-header">
        <span class="hybrid-badge">HYBRID ML + GEMINI 3</span>
      </div>
      
      <div class="hybrid-risk-assessment">
        <div class="risk-level" style="color: ${riskColor}">
          ${riskCategory.replace('_', ' ').toUpperCase()}
        </div>
        <div class="risk-confidence">${confidence}% confidence</div>
      </div>
      
      <div class="hybrid-metrics">
        <div class="metric">
          <span class="metric-label">ML Avg Susceptibility</span>
          <span class="metric-value">${(mlMetrics.avg_susceptibility * 100).toFixed(1)}%</span>
        </div>
        <div class="metric">
          <span class="metric-label">High Risk Tiles</span>
          <span class="metric-value">${mlMetrics.high_risk_tiles || 0}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Tiles Analyzed</span>
          <span class="metric-value">${mlMetrics.total_tiles || 0}</span>
        </div>
      </div>
      
      <div class="hybrid-reasoning">
        <h4>AI Analysis</h4>
        <p>${assessment}</p>
      </div>
      
      ${findingsHtml}
      ${recsHtml}
      
      <div class="hybrid-model-info">
        <small>
          ML: XGBoost | AI: ${validation.model_info?.ai_validator || 'gemini-3-pro-preview'}
        </small>
      </div>
    `;
    
    // Make sure the section is visible
    geminiSection.style.display = 'block';
  }
}

// Add animation keyframes
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateX(-50%) translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }
    }
    
    .scanner-rect {
        animation: scanPulse 0.5s ease-in-out infinite alternate;
    }
    
    @keyframes scanPulse {
        from {
            stroke-opacity: 0.8;
            fill-opacity: 0.1;
        }
        to {
            stroke-opacity: 1;
            fill-opacity: 0.2;
        }
    }
`;
document.head.appendChild(style);

// Load Terrain button – attach immediately so it always works (does not depend on app init)
(function () {
  var loadTerrainBtn = document.getElementById('load-terrain-btn');
  if (!loadTerrainBtn) return;
  function onLoadTerrainClick() {
    if (typeof window.loadTerrainData === 'function') {
      window.loadTerrainData();
      return;
    }
    if (typeof window.loadTerrainData === 'function') {
      window.loadTerrainData();
    }
  }
  loadTerrainBtn.addEventListener('click', onLoadTerrainClick);
})();

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new SinkholeApp();
  app.init();

  // Make app available globally for debugging
  window.sinkholeApp = app;
});
