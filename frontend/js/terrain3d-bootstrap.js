/**
 * Load Three.js and OrbitControls, then inject the main terrain script.
 * This runs as a module so imports work; the main script runs as a plain script with global THREE.
 */
(async function () {
  try {
    const three = await import('https://unpkg.com/three@0.160.0/build/three.module.js');
    const controls = await import('https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js');
    window.THREE = three.default;
    window.THREE.OrbitControls = controls.OrbitControls;
    const script = document.createElement('script');
    script.src = '/static/js/terrain3d-main.js?v=3';
    script.onerror = function () {
      console.error('[3D Terrain] Failed to load terrain3d-main.js');
    };
    document.body.appendChild(script);
  } catch (e) {
    console.error('[3D Terrain] Bootstrap failed:', e);
  }
})();
