/**
 * Map initialization and management for Sinkhole Scanner
 * REAL DATA VERSION - Heatmap ONLY shows after scan completes
 */

class SinkholeMap {
    constructor(containerId = 'map') {
        this.containerId = containerId;
        this.map = null;
        this.config = null;
        
        // Layers
        this.baseLayers = {};
        this.susceptibilityLayer = null;
        this.featuresLayer = null;
        this.aoiBoundaryLayer = null;
        this.queryMarker = null;
        this.scannerLayer = null;
        this.sinkholeMarkers = null;
        this.processedTilesLayer = null;  // Shows scanned tiles
        
        // State - NO HEATMAP until scan completes
        this.showHeatmapEnabled = false;
        this.showFeatures = true;
        this.currentBasemap = 'light';  // Light default for risk maps (Fix 5)
        this.scanComplete = false;
        this.scanStarted = false;
        
        // Callbacks
        this.onMapClick = null;
        this.onTileLoad = null;
    }

    /**
     * Initialize the map with configuration
     */
    async init(config) {
        this.config = config;
        
        // Create map
        this.map = L.map(this.containerId, {
            center: config.aoi.center,
            zoom: config.map.defaultZoom,
            minZoom: config.map.minZoom,
            maxZoom: config.map.maxZoom,
            zoomControl: true,
            attributionControl: true,
        });

        // Add base layers
        this._initBaseLayers();
        
        // Add AOI boundary
        this._addAOIBoundary();
        
        // Legend/scale elements (bottom-right on map)
        this._scaleEl = document.getElementById('map-scale');
        this._legendEl = document.getElementById('map-legend');
        this._updateScaleText();
        this.map.on('zoomend moveend', () => this._updateScaleText());
        
        // Create empty layer groups - NO susceptibility layer yet
        this.featuresLayer = L.layerGroup();
        this.sinkholeMarkers = L.layerGroup().addTo(this.map);
        this.scannerLayer = L.layerGroup().addTo(this.map);
        this.processedTilesLayer = L.layerGroup().addTo(this.map);
        
        // Set up event handlers
        this._setupEventHandlers();

        return this;
    }

    /**
     * Initialize base map layers (Fix 5: light default for susceptibility)
     */
    _initBaseLayers() {
        // Light basemap (default for risk maps - overlay reads clearly)
        this.baseLayers.light = L.tileLayer(
            'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
                subdomains: 'abc',
                maxZoom: 19,
            }
        );

        // Satellite layer (Esri World Imagery)
        this.baseLayers.satellite = L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            {
                attribution: 'Esri, Maxar, Earthstar Geographics',
                maxZoom: 19,
            }
        );

        // Dark map layer
        this.baseLayers.dark = L.tileLayer(
            'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
            {
                attribution: '&copy; OpenStreetMap &copy; CARTO',
                subdomains: 'abcd',
                maxZoom: 20,
            }
        );

        // Default: light (Fix 5 - professional risk map)
        this.baseLayers.light.addTo(this.map);
    }

    /**
     * Add AOI boundary layer (skipped if no geojson so map still displays)
     */
    _addAOIBoundary() {
        const geojson = this.config.aoi && this.config.aoi.geojson;
        if (!geojson) return;
        this.aoiBoundaryLayer = L.geoJSON(geojson, {
            style: {
                color: '#00ffc8',
                weight: 2,
                opacity: 0.8,
                fillColor: '#00ffc8',
                fillOpacity: 0.05,
                dashArray: '5, 10',
            },
        }).addTo(this.map);
    }

    /**
     * Create susceptibility layer as XYZ tiles (Fix 2: tile-based, not single overlay)
     * Opacity 0.55 so overlay supports basemap (Fix 4/5)
     */
    _createSusceptibilityLayer() {
        const timestamp = Date.now();
        // Backend caches tiles at z=14; serves that or scaled parent/child for other zooms.
        this.susceptibilityLayer = L.tileLayer(
            `/api/tiles/susceptibility/{z}/{x}/{y}.png?t=${timestamp}&live=true`,
            {
                opacity: 0.55,
                maxNativeZoom: 14,
                maxZoom: this.config.map.maxZoom,
                tileSize: 256,
                zIndex: 100,
            }
        );
    }

    /**
     * Set up map event handlers
     */
    _updateScaleText() {
        const el = this._scaleEl;
        if (!el || !this.map) return;
        const zoom = this.map.getZoom();
        const center = this.map.getCenter();
        const lat = center.lat * Math.PI / 180;
        const metersPerPx = 40075000 * Math.cos(lat) / (256 * Math.pow(2, zoom));
        const kmPer100Px = (metersPerPx * 100 / 1000);
        let scaleText = 'Scale: —';
        if (kmPer100Px >= 1000) {
            scaleText = `Scale: ~${(kmPer100Px / 1000).toFixed(0)} Mm`;
        } else if (kmPer100Px >= 1) {
            scaleText = `Scale: ~${kmPer100Px.toFixed(1)} km`;
        } else {
            scaleText = `Scale: ~${(kmPer100Px * 1000).toFixed(0)} m`;
        }
        el.textContent = scaleText;
    }

    _setupEventHandlers() {
        this.map.on('mousemove', (e) => {
            this._updateCoordinates(e.latlng);
        });

        this.map.on('click', async (e) => {
            if (this.onMapClick) {
                this.onMapClick(e.latlng);
            }
        });
    }

    /**
     * Update coordinates display
     */
    _updateCoordinates(latlng) {
        const latEl = document.getElementById('coord-lat');
        const lonEl = document.getElementById('coord-lon');
        
        if (latEl && lonEl) {
            latEl.textContent = latlng.lat.toFixed(6);
            lonEl.textContent = latlng.lng.toFixed(6);
        }
    }

    /**
     * Clear everything for a fresh scan
     */
    clearForNewScan() {
        // Remove susceptibility layer if exists
        if (this.susceptibilityLayer && this.map.hasLayer(this.susceptibilityLayer)) {
            this.map.removeLayer(this.susceptibilityLayer);
        }
        this.susceptibilityLayer = null;
        
        // Clear processed tiles visual
        this.processedTilesLayer.clearLayers();
        
        // Clear sinkhole markers
        this.sinkholeMarkers.clearLayers();
        
        // Reset state
        this.scanComplete = false;
        this.scanStarted = true;
        this.showHeatmapEnabled = false;
    }

    /**
     * Show the scanner box at specific tile bounds with pulsing animation
     */
    showScannerBox(bounds, color = '#00ffc8') {
        this.scannerLayer.clearLayers();
        
        // Main scanner rectangle
        const rect = L.rectangle(bounds, {
            color: color,
            weight: 4,
            opacity: 1,
            fillColor: color,
            fillOpacity: 0.2,
            className: 'scanner-rect-active'
        });
        
        // Add scan line effect
        const center = bounds.getCenter();
        const scanLine = L.polyline([
            [bounds.getSouth(), bounds.getWest()],
            [bounds.getSouth(), bounds.getEast()]
        ], {
            color: '#ffffff',
            weight: 2,
            opacity: 0.8,
            className: 'scan-line'
        });
        
        this.scannerLayer.addLayer(rect);
        this.scannerLayer.addLayer(scanLine);
        
        // Animate scan line moving down
        this._animateScanLine(scanLine, bounds);
    }

    /**
     * Animate scan line moving across the tile
     */
    _animateScanLine(scanLine, bounds) {
        const south = bounds.getSouth();
        const north = bounds.getNorth();
        const west = bounds.getWest();
        const east = bounds.getEast();
        const duration = 300;  // ms
        const steps = 15;
        const stepDuration = duration / steps;
        
        let step = 0;
        const animate = () => {
            if (step >= steps || !this.scannerLayer.hasLayer(scanLine)) return;
            
            const progress = step / steps;
            const lat = north - (north - south) * progress;
            
            scanLine.setLatLngs([
                [lat, west],
                [lat, east]
            ]);
            
            step++;
            setTimeout(animate, stepDuration);
        };
        
        animate();
    }

    /**
     * Mark a tile as processed (show fading overlay)
     */
    markTileProcessed(bounds) {
        const rect = L.rectangle(bounds, {
            color: '#00ffc8',
            weight: 1,
            opacity: 0.3,
            fillColor: '#00ffc8',
            fillOpacity: 0.1,
            interactive: false,
        });
        
        this.processedTilesLayer.addLayer(rect);
    }

    /**
     * Hide the scanner box
     */
    hideScannerBox() {
        this.scannerLayer.clearLayers();
    }

    /**
     * Add a sinkhole marker with animation
     */
    addSinkholeMarker(lat, lon, properties = {}) {
        const icon = L.divIcon({
            className: 'sinkhole-marker',
            html: `<div class="sinkhole-dot" style="
                width: 6px;
                height: 6px;
                background: #dc2626;
                border: none;
                border-radius: 50%;
                box-shadow: none;
            "></div>`,
            iconSize: [6, 6],
            iconAnchor: [3, 3],
        });
        
        const marker = L.marker([lat, lon], { icon });
        
        // Popup with real FGS data
        const type = properties.SINKHOLE_T || properties.type || 'Unknown';
        const diameter = properties.DIAMETER_F || properties.diameter_ft || '?';
        const source = properties.SOURCE || 'Florida Geological Survey';
        const date = properties.REPORTED_D || '';
        
        const popupContent = `
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; min-width: 180px;">
                <div style="color: #ef4444; font-weight: bold; margin-bottom: 8px; font-size: 12px;">
                    ⚠ HISTORICAL SINKHOLE
                </div>
                <div style="display: grid; grid-template-columns: auto 1fr; gap: 4px 10px;">
                    <span style="color: #888;">Type:</span>
                    <span style="color: #fff;">${type}</span>
                    <span style="color: #888;">Diameter:</span>
                    <span style="color: #fff;">${diameter} ft</span>
                    ${date ? `<span style="color: #888;">Reported:</span><span style="color: #fff;">${date}</span>` : ''}
                    <span style="color: #888;">Source:</span>
                    <span style="color: #fff;">${source}</span>
                </div>
            </div>
        `;
        marker.bindPopup(popupContent, {
            className: 'sinkhole-popup'
        });
        
        this.sinkholeMarkers.addLayer(marker);
        return marker;
    }

    /**
     * Clear all sinkhole markers
     */
    clearSinkholeMarkers() {
        this.sinkholeMarkers.clearLayers();
    }

    /**
     * Mark scan as complete and reveal heatmap
     */
    markScanComplete() {
        this.scanComplete = true;
        this.showHeatmapEnabled = true;
        
        // Clear the processed tiles visual
        this.processedTilesLayer.clearLayers();
        
        // Now add the susceptibility layer
        this._createSusceptibilityLayer();
        this.susceptibilityLayer.addTo(this.map);
        
        // Bring sinkhole markers to front
        if (this.sinkholeMarkers) {
            this.sinkholeMarkers.eachLayer(layer => {
                if (layer.bringToFront) layer.bringToFront();
            });
        }
    }

    /**
     * Toggle heatmap visibility
     */
    setHeatmapVisible(visible) {
        if (!this.scanComplete) return;  // Can't show until scan complete
        
        this.showHeatmapEnabled = visible;
        if (visible) {
            if (!this.susceptibilityLayer) {
                this._createSusceptibilityLayer();
            }
            if (!this.map.hasLayer(this.susceptibilityLayer)) {
                this.susceptibilityLayer.addTo(this.map);
            }
        } else if (this.susceptibilityLayer && this.map.hasLayer(this.susceptibilityLayer)) {
            this.map.removeLayer(this.susceptibilityLayer);
        }
    }

    /**
     * Cycle through basemaps: Light (default) → Satellite → Dark (Fix 5)
     */
    cycleBasemap() {
        const basemapOrder = ['light', 'satellite', 'dark'];
        const currentIndex = basemapOrder.indexOf(this.currentBasemap);
        const nextIndex = (currentIndex + 1) % basemapOrder.length;
        const nextBasemap = basemapOrder[nextIndex];
        
        this.map.removeLayer(this.baseLayers[this.currentBasemap]);
        this.baseLayers[nextBasemap].addTo(this.map);
        this.baseLayers[nextBasemap].bringToBack();
        
        this.currentBasemap = nextBasemap;
        return nextBasemap;
    }

    /**
     * Add a query marker at location
     */
    setQueryMarker(latlng) {
        if (this.queryMarker) {
            this.map.removeLayer(this.queryMarker);
        }
        
        const icon = L.divIcon({
            className: 'query-marker',
            html: `<div style="
                width: 24px;
                height: 24px;
                background: radial-gradient(circle, #00ffc8 0%, transparent 70%);
                border: 2px solid #00ffc8;
                border-radius: 50%;
                box-shadow: 0 0 10px #00ffc8;
            "></div>`,
            iconSize: [24, 24],
            iconAnchor: [12, 12],
        });
        
        this.queryMarker = L.marker(latlng, { icon }).addTo(this.map);
    }

    /**
     * Get AOI bounds
     */
    getAOIBounds() {
        const bbox = this.config.aoi.bbox;
        return L.latLngBounds(
            [bbox[1], bbox[0]],
            [bbox[3], bbox[2]]
        );
    }

    /**
     * Fit map to AOI bounds
     */
    fitToAOI() {
        this.map.fitBounds(this.getAOIBounds(), { padding: [20, 20] });
    }

    /**
     * Get current zoom level
     */
    getZoom() {
        return this.map.getZoom();
    }
}

// Export
window.SinkholeMap = SinkholeMap;
