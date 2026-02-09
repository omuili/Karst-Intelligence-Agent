/**
 * 3D Terrain (DEM) viewer - OpenTopography + Three.js
 * Uses global THREE and THREE.OrbitControls set by terrain3d-bootstrap.js
 */
(function () {
  var THREE = window.THREE;
  if (!THREE || !THREE.OrbitControls) {
    console.warn('[3D Terrain] THREE not ready yet; Load Terrain will work after bootstrap.');
    return;
  }

  var API_BASE = window.location.origin || '';

  var scene, camera, renderer, controls, terrainMesh;
  var terrain3dInitialized = false;
  var heightExaggeration = 3;
  var terrainData = null;

  function initTerrain3D() {
    if (terrain3dInitialized) return;

    var container = document.getElementById('terrain3d');
    var parent = document.getElementById('terrain3d-container');
    if (!container || !parent) return;

    var width = parent.clientWidth || 400;
    var height = 320;
    width = Math.max(width, 280);
    height = Math.max(height, 280);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0f14);
    scene.fog = new THREE.Fog(0x0a0f14, 80, 350);

    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(0, 80, 100);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2.1;
    controls.minDistance = 20;
    controls.maxDistance = 300;

    scene.add(new THREE.AmbientLight(0x404060, 0.5));
    var dirLight = new THREE.DirectionalLight(0xffffff, 1);
    dirLight.position.set(50, 100, 50);
    scene.add(dirLight);
    var fillLight = new THREE.DirectionalLight(0x00b4d8, 0.25);
    fillLight.position.set(-50, 50, -50);
    scene.add(fillLight);

    var gridHelper = new THREE.GridHelper(200, 50, 0x1a2530, 0x151d26);
    gridHelper.position.y = -1;
    scene.add(gridHelper);

    function animate() {
      requestAnimationFrame(animate);
      if (controls) controls.update();
      if (renderer && scene && camera) renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', function () {
      if (!parent || !renderer || !camera) return;
      var w = parent.clientWidth || width;
      var h = 320;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });

    terrain3dInitialized = true;
  }

  function createTerrainMesh(elevationData, metadata) {
    if (!scene) return;

    if (terrainMesh) {
      scene.remove(terrainMesh);
      terrainMesh.geometry.dispose();
      terrainMesh.material.dispose();
    }

    var rows = elevationData.length;
    var cols = elevationData[0].length;
    var geometry = new THREE.PlaneGeometry(100, 100, cols - 1, rows - 1);
    var positions = geometry.attributes.position.array;

    var minElev = metadata.elevation_stats.min_m;
    var maxElev = metadata.elevation_stats.max_m;
    var elevRange = (maxElev - minElev) || 1;

    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        var vertexIndex = (i * cols + j) * 3;
        var elevation = elevationData[i][j];
        var normalizedHeight = ((elevation - minElev) / elevRange) * heightExaggeration * 10;
        positions[vertexIndex + 2] = normalizedHeight;
      }
    }

    geometry.computeVertexNormals();

    var colors = new Float32Array(rows * cols * 3);
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        var idx = (i * cols + j) * 3;
        var elevation = elevationData[i][j];
        var t = (elevation - minElev) / elevRange;
        var r, g, b;
        if (t < 0.2) {
          r = 0.12 + t * 0.4; g = 0.23 + t * 0.6; b = 0.37 + t * 0.2;
        } else if (t < 0.5) {
          var t2 = (t - 0.2) / 0.3;
          r = 0.18 + t2 * 0.1; g = 0.35 + t2 * 0.15; b = 0.23 - t2 * 0.1;
        } else if (t < 0.8) {
          var t2b = (t - 0.5) / 0.3;
          r = 0.29 + t2b * 0.25; g = 0.49 - t2b * 0.1; b = 0.23 + t2b * 0.1;
        } else {
          var t2c = (t - 0.8) / 0.2;
          r = 0.55 + t2c * 0.08; g = 0.45 + t2c * 0.05; b = 0.35 + t2c * 0.03;
        }
        colors[idx] = r; colors[idx + 1] = g; colors[idx + 2] = b;
      }
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    var material = new THREE.MeshLambertMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
    });
    terrainMesh = new THREE.Mesh(geometry, material);
    terrainMesh.rotation.x = -Math.PI / 2;
    terrainMesh.position.y = 0;
    scene.add(terrainMesh);

    resetTerrainView();
  }

  function resetTerrainView() {
    if (!camera || !controls) return;
    camera.position.set(0, 80, 100);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
  }

  async function loadTerrainData() {
    var loading = document.getElementById('terrain-loading');
    var container = document.getElementById('terrain3d-container');
    var info = document.getElementById('terrain-info');
    var resetBtn = document.getElementById('reset-terrain-view-btn');

    if (loading) loading.style.display = 'flex';

    try {
      if (!window.THREE || !window.THREE.OrbitControls) {
        throw new Error('3D library not ready. Refresh the page and try again.');
      }

      var regionsRes = await fetch(API_BASE + '/api/terrain/regions');
      var regionsJson = await regionsRes.json();
      var regions = regionsJson.regions || [];
      var center = regions.length
        ? { lat: regions[0].center.lat, lng: regions[0].center.lng }
        : { lat: 28.5983, lng: -81.351 };

      var response = await fetch(API_BASE + '/api/terrain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat: center.lat,
          lng: center.lng,
          size_km: 2,
          resolution: '10m',
        }),
      });

      if (!response.ok) {
        var errText = await response.text();
        throw new Error(errText || 'Failed to load terrain');
      }

      var data = await response.json();
      terrainData = data;

      if (!terrain3dInitialized) {
        initTerrain3D();
        if (!terrain3dInitialized) throw new Error('Failed to initialize Three.js');
      }

      createTerrainMesh(data.elevation, data.metadata);

      if (info) {
        document.getElementById('terrain-resolution').textContent = data.metadata.resolution || '—';
        document.getElementById('terrain-grid-size').textContent =
          data.metadata.grid_size.rows + ' × ' + data.metadata.grid_size.cols;
        document.getElementById('terrain-min-elev').textContent =
          data.metadata.elevation_stats.min_m.toFixed(1) + ' m';
        document.getElementById('terrain-max-elev').textContent =
          data.metadata.elevation_stats.max_m.toFixed(1) + ' m';
        document.getElementById('terrain-range').textContent =
          data.metadata.elevation_stats.range_m.toFixed(1) + ' m';
        info.style.display = 'block';
      }
      if (container) container.style.display = 'block';
      if (resetBtn) resetBtn.style.display = 'inline-block';
    } catch (err) {
      console.error('[3D Terrain]', err);
      var hint = /OrbitControls|constructor|not ready/i.test(err.message)
        ? 'Refresh the page and try again.'
        : 'Set OPENTOPOGRAPHY_API_KEY in .env if not set.';
      alert('Terrain load failed: ' + err.message + '\n\n' + hint);
    } finally {
      if (loading) loading.style.display = 'none';
    }
  }

  function onDomReady() {
    var loadBtn = document.getElementById('load-terrain-btn');
    var resetBtn = document.getElementById('reset-terrain-view-btn');
    if (loadBtn) loadBtn.addEventListener('click', loadTerrainData);
    if (resetBtn) resetBtn.addEventListener('click', resetTerrainView);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', onDomReady);
  } else {
    onDomReady();
  }

  window.loadTerrainData = loadTerrainData;
  window.resetTerrainView = resetTerrainView;
  window.initTerrain3D = initTerrain3D;
})();
