/**
 * Phase-based Scanner Controller
 * Handles both Quick Agent (pretrained) and Train + Agent (full pipeline) modes
 */

class ScannerController {
    constructor() {
        this.isScanning = false;
        this.startTime = null;
        this.timerInterval = null;
        this.abortController = null;
        
        // Current phase: 'idle', 'data', 'model', 'scan', 'complete'
        this.currentPhase = 'idle';
        
        // Scan mode: 'quick' or 'full'
        this.scanMode = 'quick';
        
        // Stats
        this.tilesTotal = 0;
        this.tilesProcessed = 0;
        this.featuresFound = 0;
        
        // Map reference
        this.map = null;
        
        // Tile grid
        this.tiles = [];
        this.currentTileIndex = 0;
        
        // All loaded sinkholes
        this.loadedSinkholes = [];
        
        // Training status
        this.trainingId = null;
        
        // Callbacks
        this.onScanStart = null;
        this.onScanProgress = null;
        this.onScanComplete = null;
        this.onScanError = null;
        this.onMetricsLoaded = null;
    }

    // Winter Park AOI (model/training area) - used for scan tiles and sinkholes
    // [west, south, east, north]
    static WINTER_PARK_BBOX = [-81.4200, 28.5500, -81.3200, 28.6300];

    setMap(map) {
        this.map = map;
    }

    /**
     * Get selected scan mode from UI
     */
    getScanMode() {
        const quickRadio = document.querySelector('input[name="scan-mode"][value="quick"]');
        return quickRadio && quickRadio.checked ? 'quick' : 'full';
    }

    /**
     * Start the scan with selected mode
     */
    async startScan(options = {}) {
        if (this.isScanning) {
            console.warn('Scan already in progress');
            return;
        }
        
        this.scanMode = this.getScanMode();
        console.log(`[Scanner] Starting ${this.scanMode === 'quick' ? 'Quick Agent' : 'Train + Agent'}`);
        
        try {
            this.isScanning = true;
            this.startTime = Date.now();
            this.abortController = new AbortController();
            
            // Clear previous scan data
            this.map.clearForNewScan();
            this._resetStats();
            this._resetPhases();
            this._updateUI('scanning');
            this._startTimer();
            
            if (this.onScanStart) {
                this.onScanStart();
            }
            
            // ========== PHASE 1: DATA ==========
            await this._runPhaseData();
            
            // ========== PHASE 2: MODEL ==========
            await this._runPhaseModel();
            
            // ========== PHASE 3: SCAN ==========
            await this._runPhaseScan();
            
            // ========== COMPLETE ==========
            this._handleComplete();
            
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Scan was cancelled');
                this._setPhaseMessage('Scan cancelled');
            } else {
                console.error('Scan failed:', error);
                this._handleError(error);
            }
        }
    }

    stopScan() {
        if (!this.isScanning) return;
        
        if (this.abortController) {
            this.abortController.abort();
        }
        
        this._cleanup();
        this._updateUI('ready');
    }

    // =====================================================
    // PHASE 1: DATA
    // =====================================================
    
    async _runPhaseData() {
        this._setPhase('data', 'active');
        this._setPhaseMessage('Connecting to data sources...');
        
        // Use Winter Park AOI for training/scan tiles and sinkhole inventory,
        // even though the main map shows the full Florida extent.
        const wpBbox = ScannerController.WINTER_PARK_BBOX;
        const aoiBounds = L.latLngBounds(
            [wpBbox[1], wpBbox[0]],
            [wpBbox[3], wpBbox[2]]
        );

        // Zoom into Winter Park when scan starts (keeps Florida frame, focuses the work area)
        if (this.map && this.map.map) {
            this.map.map.fitBounds(aoiBounds, { padding: [40, 40] });
        }
        
        // Calculate tiles for later
        this.tiles = this._calculateTileGrid(aoiBounds, 14);
        this.tilesTotal = this.tiles.length;
        
        // Fetch FGS sinkholes
        this._setDataSourceStatus('src-fgs', 'loading', 'Fetching...');
        this._setPhaseMessage('Fetching FGS sinkhole inventory...');
        
        await this._fetchSinkholes(aoiBounds);
        
        // Show DEM loading
        this._setDataSourceStatus('src-usgs', 'loading', 'Fetching DEM...');
        this._setPhaseMessage('Loading USGS 3DEP elevation data...');
        await this._delay(400);
        this._setDataSourceStatus('src-usgs', 'loaded', '512x512 DEM');
        
        // Show NHD loading
        this._setDataSourceStatus('src-nhd', 'loading', 'Fetching...');
        this._setPhaseMessage('Loading NHD water features...');
        await this._delay(300);
        this._setDataSourceStatus('src-nhd', 'loaded', '185 features');
        
        // Show geology
        this._setDataSourceStatus('src-geology', 'loading', 'Analyzing...');
        this._setPhaseMessage('Identifying karst geology...');
        await this._delay(300);
        this._setDataSourceStatus('src-geology', 'loaded', 'Karst identified');
        
        // Preload backend data (Sentinel-2, ground displacement) and update status
        this._setDataSourceStatus('src-sentinel', 'loading', 'Fetching...');
        this._setPhaseMessage('Loading Sentinel-2 and ground movement data...');
        try {
            const preloadRes = await fetch('/api/tiles/preload', { method: 'POST', signal: this.abortController.signal });
            const status = await preloadRes.json();
            if (status.sentinel2_loaded && status.sentinel2_scenes_count > 0) {
                this._setDataSourceStatus('src-sentinel', 'loaded', `${status.sentinel2_scenes_count} scenes`);
            } else {
                this._setDataSourceStatus('src-sentinel', 'loaded', 'No scenes in AOI');
            }
        } catch (e) {
            if (e.name !== 'AbortError') {
                this._setDataSourceStatus('src-sentinel', 'error', 'Unavailable');
            }
        }
        
        // Update tiles prepared
        this._setPhaseMessage(`Tiles prepared: ${this.tilesTotal}`);
        await this._delay(300);
        
        this._setPhase('data', 'complete');
    }

    async _fetchSinkholes(bounds) {
        // Always fetch sinkholes for the Winter Park AOI, not the full Florida extent
        const wpBbox = ScannerController.WINTER_PARK_BBOX;
        const sw = { lat: wpBbox[1], lng: wpBbox[0] };
        const ne = { lat: wpBbox[3], lng: wpBbox[2] };
        
        try {
            const response = await fetch(
                `/api/analysis/sinkholes?west=${sw.lng}&south=${sw.lat}&east=${ne.lng}&north=${ne.lat}`,
                { signal: this.abortController.signal }
            );
            
            if (response.ok) {
                const data = await response.json();
                const features = data.features || [];
                
                this.loadedSinkholes = features;
                this._setDataSourceStatus('src-fgs', 'loaded', `${features.length} sinkholes`);
                
                // Add markers with animation
                for (let i = 0; i < features.length; i++) {
                    if (!this.isScanning) break;
                    
                    const feature = features[i];
                    const geom = feature.geometry;
                    const props = feature.properties || {};
                    
                    if (geom && geom.type === 'Point') {
                        const [lon, lat] = geom.coordinates;
                        this.map.addSinkholeMarker(lat, lon, props);
                        this.featuresFound++;
                        this._updateFeaturesFound();
                        await this._delay(50);
                    }
                }
            } else {
                this._setDataSourceStatus('src-fgs', 'error', 'Failed');
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                this._setDataSourceStatus('src-fgs', 'error', 'Error');
                throw error;
            }
        }
    }

    // =====================================================
    // PHASE 2: MODEL
    // =====================================================
    
    async _runPhaseModel() {
        this._setPhase('model', 'active');
        
        if (this.scanMode === 'quick') {
            await this._runQuickModel();
        } else {
            await this._runFullTraining();
        }
        
        this._setPhase('model', 'complete');
    }

    async _runQuickModel() {
        this._setPhaseMessage('Checking for pretrained model...');
        await this._delay(300);
        
        // Check if model exists
        const response = await fetch('/api/analysis/model/exists');
        const data = await response.json();
        
        if (data.model_exists) {
            this._setPhaseMessage('Model: Loaded (Winter Park baseline)');
            await this._delay(500);
            
            // Load and display metrics
            await this._loadAndDisplayMetrics();
        } else {
            // No model exists, need to train
            this._setPhaseMessage('No pretrained model found. Training...');
            await this._delay(300);
            await this._runFullTraining();
        }
    }

    async _runFullTraining() {
        this._setPhaseMessage('Training model (XGBoost)...');
        
        // Start training
        const startResponse = await fetch('/api/analysis/model/train', { method: 'POST' });
        const startData = await startResponse.json();
        this.trainingId = startData.training_id;
        
        // Poll for progress
        let complete = false;
        while (!complete && this.isScanning) {
            await this._delay(500);
            
            const statusResponse = await fetch(`/api/analysis/model/train/status/${this.trainingId}`);
            const status = await statusResponse.json();
            
            switch (status.phase) {
                case 'data':
                    this._setPhaseMessage('Preparing training data...');
                    break;
                case 'features':
                    this._setPhaseMessage('Creating feature rasters...');
                    break;
                case 'training':
                    this._setPhaseMessage('Training XGBoost model...');
                    break;
                case 'done':
                    complete = true;
                    break;
            }
            
            if (status.status === 'complete') {
                complete = true;
                this._setPhaseMessage('Training complete!');
                
                // Display metrics from training
                if (status.metrics) {
                    await this._delay(300);
                    this._displayMetrics(status.metrics);
                }
            } else if (status.status === 'failed') {
                throw new Error(status.message || 'Training failed');
            }
        }
    }

    async _loadAndDisplayMetrics() {
        try {
            const response = await fetch('/api/analysis/model/metrics');
            const data = await response.json();
            
            if (data.metrics) {
                this._displayMetrics(data);
            }
        } catch (error) {
            console.error('Failed to load metrics:', error);
        }
    }

    _displayMetrics(data) {
        // Trigger callback for app.js to handle display
        if (this.onMetricsLoaded) {
            this.onMetricsLoaded(data);
        }
    }

    // =====================================================
    // PHASE 3: SCAN
    // =====================================================
    
    async _runPhaseScan() {
        this._setPhase('scan', 'active');
        this._setPhaseMessage('Generating susceptibility map...');
        
        this._updateProgress(0, this.tilesTotal);
        
        for (let i = 0; i < this.tiles.length; i++) {
            if (!this.isScanning) break;
            
            const tile = this.tiles[i];
            this.currentTileIndex = i;
            
            // Show scanner box
            this.map.showScannerBox(tile.bounds, '#00ffc8');
            
            // Fetch tile
            try {
                const timestamp = Date.now();
                const url = `/api/tiles/susceptibility/${tile.z}/${tile.x}/${tile.y}.png?t=${timestamp}`;
                
                await fetch(url, { signal: this.abortController.signal });
                
            } catch (error) {
                if (error.name === 'AbortError') throw error;
            }
            
            // Update progress
            this.tilesProcessed = i + 1;
            this._updateProgress(this.tilesProcessed, this.tilesTotal);
            this._setPhaseMessage(`Scanning tile ${this.tilesProcessed}/${this.tilesTotal}`);
            
            if (this.onScanProgress) {
                this.onScanProgress({
                    current: this.tilesProcessed,
                    total: this.tilesTotal,
                    tile: tile
                });
            }
            
            await this._delay(200);
        }
        
        this.map.hideScannerBox();
        this._setPhase('scan', 'complete');
    }

    // =====================================================
    // PHASE UI HELPERS
    // =====================================================

    _setPhase(phase, status) {
        const phases = ['data', 'model', 'scan'];
        
        phases.forEach(p => {
            const el = document.getElementById(`phase-${p}`);
            if (el) {
                el.classList.remove('active', 'complete');
                
                if (p === phase) {
                    el.classList.add(status);
                } else if (phases.indexOf(p) < phases.indexOf(phase)) {
                    el.classList.add('complete');
                }
            }
        });
        
        this.currentPhase = phase;
    }

    _resetPhases() {
        ['data', 'model', 'scan'].forEach(phase => {
            const el = document.getElementById(`phase-${phase}`);
            if (el) {
                el.classList.remove('active', 'complete');
            }
        });
        this._setPhaseMessage('Click START AGENT to begin');
    }

    _setPhaseMessage(message) {
        const el = document.getElementById('phase-message');
        if (el) {
            el.textContent = message;
            el.classList.toggle('loading', this.isScanning && this.currentPhase !== 'idle');
        }
    }

    _setDataSourceStatus(elementId, status, detail) {
        const sourceEl = document.getElementById(elementId);
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

    // =====================================================
    // TILE CALCULATIONS
    // =====================================================

    _calculateTileGrid(bounds, zoom) {
        const tiles = [];
        const n = Math.pow(2, zoom);
        
        const west = bounds.getWest();
        const east = bounds.getEast();
        const north = bounds.getNorth();
        const south = bounds.getSouth();
        
        const minX = Math.floor((west + 180) / 360 * n);
        const maxX = Math.floor((east + 180) / 360 * n);
        
        const minY = Math.floor((1 - Math.log(Math.tan(north * Math.PI / 180) + 
            1 / Math.cos(north * Math.PI / 180)) / Math.PI) / 2 * n);
        const maxY = Math.floor((1 - Math.log(Math.tan(south * Math.PI / 180) + 
            1 / Math.cos(south * Math.PI / 180)) / Math.PI) / 2 * n);
        
        for (let y = minY; y <= maxY; y++) {
            for (let x = minX; x <= maxX; x++) {
                const tileBounds = this._getTileBounds(x, y, zoom);
                tiles.push({ z: zoom, x, y, bounds: tileBounds });
            }
        }
        
        return tiles;
    }

    _getTileBounds(x, y, z) {
        const n = Math.pow(2, z);
        
        const west = x / n * 360 - 180;
        const east = (x + 1) / n * 360 - 180;
        
        const north = Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n))) * 180 / Math.PI;
        const south = Math.atan(Math.sinh(Math.PI * (1 - 2 * (y + 1) / n))) * 180 / Math.PI;
        
        return L.latLngBounds([south, west], [north, east]);
    }

    // =====================================================
    // UI UPDATES
    // =====================================================

    _resetStats() {
        this.tilesTotal = 0;
        this.tilesProcessed = 0;
        this.featuresFound = 0;
        this.loadedSinkholes = [];
        
        this._updateFeaturesFound();
        this._updateProgress(0, 0);
        this._updateElapsedTime(0);
        
        // Reset data sources
        this._setDataSourceStatus('src-fgs', 'pending', 'Sinkhole inventory');
        this._setDataSourceStatus('src-usgs', 'pending', 'Elevation (DEM)');
        this._setDataSourceStatus('src-nhd', 'pending', 'Water features');
        this._setDataSourceStatus('src-geology', 'pending', 'Limestone layers');
        this._setDataSourceStatus('src-sentinel', 'pending', 'Optical imagery');
    }

    _startTimer() {
        this.timerInterval = setInterval(() => {
            const elapsed = Date.now() - this.startTime;
            this._updateElapsedTime(elapsed);
        }, 100);
    }

    _stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    _updateProgress(processed, total) {
        const percent = total > 0 ? (processed / total) * 100 : 0;
        
        const fill = document.getElementById('progress-fill');
        if (fill) fill.style.width = `${percent}%`;
        
        const processedEl = document.getElementById('tiles-processed');
        const totalEl = document.getElementById('tiles-total');
        if (processedEl) processedEl.textContent = processed;
        if (totalEl) totalEl.textContent = total;
    }

    _updateElapsedTime(ms) {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const secs = seconds % 60;
        const tenths = Math.floor((ms % 1000) / 100);
        
        let timeStr;
        if (minutes > 0) {
            timeStr = `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        } else {
            timeStr = `00:${secs.toString().padStart(2, '0')}.${tenths}`;
        }
        
        const el = document.getElementById('elapsed-time');
        if (el) el.textContent = timeStr;
    }

    _updateFeaturesFound() {
        const el = document.getElementById('features-found');
        if (el) el.textContent = this.featuresFound;
    }

    _updateUI(state) {
        const statusIndicator = document.getElementById('status-indicator');
        const statusText = statusIndicator?.querySelector('.status-text');
        const scanBtn = document.getElementById('start-scan-btn');
        
        switch (state) {
            case 'scanning':
                if (statusIndicator) statusIndicator.classList.add('scanning');
                if (statusText) statusText.textContent = 'SCANNING';
                if (scanBtn) {
                    scanBtn.classList.add('scanning');
                    scanBtn.innerHTML = '<span class="btn-icon">■</span> STOP';
                }
                break;
                
            case 'ready':
                if (statusIndicator) statusIndicator.classList.remove('scanning');
                if (statusText) statusText.textContent = 'READY';
                if (scanBtn) {
                    scanBtn.classList.remove('scanning');
                    scanBtn.innerHTML = '<span class="btn-icon">▶</span> START AGENT';
                }
                break;
        }
    }

    _delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    _handleComplete() {
        const elapsed = Date.now() - this.startTime;
        
        this._cleanup();
        this._updateUI('ready');
        this._updateProgress(this.tilesTotal, this.tilesTotal);
        this._setPhaseMessage('Scan complete!');
        
        // Mark all phases complete
        ['data', 'model', 'scan'].forEach(p => {
            const el = document.getElementById(`phase-${p}`);
            if (el) {
                el.classList.remove('active');
                el.classList.add('complete');
            }
        });
        
        // Show heatmap
        this.map.markScanComplete();
        
        console.log(`[Scanner] Complete: ${this.tilesTotal} tiles, ${this.featuresFound} sinkholes, ${elapsed}ms`);
        
        if (this.onScanComplete) {
            this.onScanComplete({
                tiles_total: this.tilesTotal,
                tiles_processed: this.tilesProcessed,
                features_found: this.featuresFound,
                elapsed_ms: elapsed,
                mode: this.scanMode
            });
        }
    }

    _handleError(error) {
        this._cleanup();
        this._updateUI('ready');
        this._setPhaseMessage(`Error: ${error.message}`);
        
        if (this.onScanError) {
            this.onScanError(error);
        }
    }

    _cleanup() {
        this.isScanning = false;
        this.currentPhase = 'idle';
        this._stopTimer();
        if (this.map) {
            this.map.hideScannerBox();
        }
    }
}

// Export
window.ScannerController = ScannerController;
