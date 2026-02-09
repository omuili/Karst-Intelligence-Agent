/**
 * API Client for Sinkhole Scanner
 */

class SinkholeAPI {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.wsConnection = null;
        this.onProgress = null;
    }

    /**
     * Get application configuration
     */
    async getConfig() {
        const response = await fetch(`${this.baseUrl}/api/config`);
        if (!response.ok) throw new Error('Failed to fetch config');
        return response.json();
    }

    /**
     * Get statistics for the AOI
     */
    async getStatistics() {
        const response = await fetch(`${this.baseUrl}/api/analysis/statistics`);
        if (!response.ok) throw new Error('Failed to fetch statistics');
        return response.json();
    }

    /**
     * Start a new scan
     */
    async startScan(options = {}) {
        const response = await fetch(`${this.baseUrl}/api/analysis/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                zoom: options.zoom || 14,
                include_gemini: options.includeGemini !== false,
                bbox: options.bbox || null,
            }),
        });
        if (!response.ok) throw new Error('Failed to start scan');
        return response.json();
    }

    /**
     * Get scan status
     */
    async getScanStatus(scanId) {
        const response = await fetch(`${this.baseUrl}/api/analysis/scan/${scanId}`);
        if (!response.ok) throw new Error('Failed to get scan status');
        return response.json();
    }

    /**
     * Cancel a running scan
     */
    async cancelScan(scanId) {
        const response = await fetch(`${this.baseUrl}/api/analysis/scan/${scanId}`, {
            method: 'DELETE',
        });
        if (!response.ok) throw new Error('Failed to cancel scan');
        return response.json();
    }

    /**
     * Get scan results
     */
    async getScanResults(scanId) {
        const response = await fetch(`${this.baseUrl}/api/analysis/results/${scanId}`);
        if (!response.ok) throw new Error('Failed to get scan results');
        return response.json();
    }

    /**
     * Query susceptibility at a point
     */
    async queryPoint(lat, lon) {
        const response = await fetch(`${this.baseUrl}/api/analysis/query/point`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lon }),
        });
        if (!response.ok) throw new Error('Failed to query point');
        return response.json();
    }

    /**
     * Get tile info
     */
    async getTileInfo(z, x, y) {
        const response = await fetch(`${this.baseUrl}/api/tiles/info/${z}/${x}/${y}`);
        if (!response.ok) throw new Error('Failed to get tile info');
        return response.json();
    }

    /**
     * Get susceptibility tile URL
     */
    getSusceptibilityTileUrl(z, x, y) {
        return `${this.baseUrl}/api/tiles/susceptibility/${z}/${x}/${y}.png`;
    }

    /**
     * Get feature tile URL
     */
    getFeatureTileUrl(z, x, y) {
        return `${this.baseUrl}/api/tiles/features/${z}/${x}/${y}.json`;
    }

    /**
     * List tiles for a zoom level
     */
    async listTiles(zoom = 14) {
        const response = await fetch(`${this.baseUrl}/api/analysis/tiles/list?zoom=${zoom}`);
        if (!response.ok) throw new Error('Failed to list tiles');
        return response.json();
    }

    /**
     * Connect to WebSocket for progress updates
     */
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/progress`;
        
        this.wsConnection = new WebSocket(wsUrl);
        
        this.wsConnection.onopen = () => {
            console.log('WebSocket connected');
            this._updateConnectionStatus(true);
        };
        
        this.wsConnection.onclose = () => {
            console.log('WebSocket disconnected');
            this._updateConnectionStatus(false);
            // Attempt reconnection after delay
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.wsConnection.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        this.wsConnection.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (this.onProgress && data.type === 'progress') {
                    this.onProgress(data);
                }
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };
    }

    /**
     * Send message through WebSocket
     */
    sendMessage(data) {
        if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
            this.wsConnection.send(JSON.stringify(data));
        }
    }

    /**
     * Update connection status indicator
     */
    _updateConnectionStatus(connected) {
        const statusEl = document.getElementById('connection-status');
        if (statusEl) {
            statusEl.textContent = connected ? '● Connected' : '○ Disconnected';
            statusEl.style.color = connected ? 'var(--accent-success)' : 'var(--accent-danger)';
        }
    }

    // ========================================================================
    // GEMINI 3 AGENTIC ANALYSIS API
    // ========================================================================

    /**
     * Check Gemini 3 API status
     */
    async getGeminiStatus() {
        const response = await fetch(`${this.baseUrl}/api/analysis/agent/gemini-status`);
        if (!response.ok) throw new Error('Failed to get Gemini status');
        return response.json();
    }

    /**
     * Start Gemini 3 agentic analysis
     */
    async startAgentAnalysis(options = {}) {
        const response = await fetch(`${this.baseUrl}/api/analysis/agent/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                bbox: options.bbox || null,
                include_satellite: options.includeSatellite !== false,
                thinking_level: options.thinkingLevel || 'high',
            }),
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to start agent analysis');
        }
        return response.json();
    }

    /**
     * Get agent analysis status
     */
    async getAgentStatus(analysisId) {
        const response = await fetch(`${this.baseUrl}/api/analysis/agent/status/${analysisId}`);
        if (!response.ok) throw new Error('Failed to get agent status');
        return response.json();
    }

    /**
     * Get agent analysis report
     */
    async getAgentReport(analysisId) {
        const response = await fetch(`${this.baseUrl}/api/analysis/agent/report/${analysisId}`);
        if (!response.ok) throw new Error('Failed to get agent report');
        return response.json();
    }

    /**
     * Start model training (train only, no scan)
     */
    async startTraining() {
        const response = await fetch(`${this.baseUrl}/api/analysis/model/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
        if (!response.ok) throw new Error('Failed to start training');
        return response.json();
    }

    /**
     * Get training job status
     */
    async getTrainingStatus(trainingId) {
        const response = await fetch(`${this.baseUrl}/api/analysis/model/train/status/${trainingId}`);
        if (!response.ok) throw new Error('Failed to get training status');
        return response.json();
    }

    /**
     * Quick risk assessment using Gemini
     */
    async quickAssess(lat, lon) {
        const response = await fetch(`${this.baseUrl}/api/analysis/agent/quick-assess`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lon }),
        });
        if (!response.ok) throw new Error('Failed to perform quick assessment');
        return response.json();
    }
}

// Export singleton instance
window.api = new SinkholeAPI();

