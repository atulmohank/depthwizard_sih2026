/* DepthWizard app bridge - app/server.py injects this after the script of the generated viewer.html
   (viewer_city_template.html, unchanged). Classic scripts share one global scope, so we can use the template's
   globals: D, S, HS, HC, scene, camera, renderer, group, canopy, surface, photo, tip, ex, applyAll,
   setBlocks, setHeat, heatMat. The parent page talks to us with postMessage. */
(function () {
  'use strict';
  const panel = document.querySelector('.panel');
  if (panel) panel.style.display = 'none';                    // the app sidebar replaces the template's panel
  const send = m => { if (parent !== window) parent.postMessage(m, location.origin); };

  let LUT = null, HMAX = Math.max(5, S.cap_m), mode = 'blocks', heatTex = null;
  const rgb = t => {
    t = Math.min(1, Math.max(0, t));
    if (!LUT) return new THREE.Color().setHSL(0.66 - 0.66 * t, 0.75, 0.52);   // template's ramp until config arrives
    const k = Math.round(t * 255) * 3;
    return new THREE.Color(LUT[k] / 255, LUT[k + 1] / 255, LUT[k + 2] / 255);
  };

  // Colour by height: same turbo ramp + scale as the 2D height map, so one legend fits both
  let heatMats = {};
  heatMat = function (h) {                                    // replaces the template's global function
    const k = Math.round(Math.min(1, h / HMAX) * 40);
    return heatMats[k] || (heatMats[k] = new THREE.MeshStandardMaterial({ color: rgb(k / 40), roughness: 0.8 }));
  };
  function heatTexture(H) {                                   // ground/tree layer coloured by height
    const c = document.createElement('canvas'); c.width = D.gw; c.height = D.gh;
    const x = c.getContext('2d'), im = x.createImageData(D.gw, D.gh);
    for (let i = 0; i < H.length; i++) {
      const col = rgb(H[i] / HMAX);
      im.data[i * 4] = col.r * 255; im.data[i * 4 + 1] = col.g * 255; im.data[i * 4 + 2] = col.b * 255; im.data[i * 4 + 3] = 255;
    }
    x.putImageData(im, 0, 0);
    const t = new THREE.CanvasTexture(c); t.anisotropy = renderer.capabilities.getMaxAnisotropy(); return t;
  }

  function setMode(m) {
    mode = m;
    if (m === 'surface') { setBlocks(false); setHeat(false); }
    else { setBlocks(true); setHeat(m === 'heat'); }
    const map = m === 'heat' ? (heatTex || (heatTex = heatTexture(HC))) : photo;
    if (canopy.material.map !== map) { canopy.material.map = map; canopy.material.needsUpdate = true; }
    tip.style.display = 'none';
  }

  function setEx(v) {
    ex = v; $('ex').value = v; applyAll();
    for (const m of [canopy, surface]) m.geometry.boundingSphere = null;   // heights changed -> refresh raycast bounds
  }

  addEventListener('message', e => {
    if (e.origin !== location.origin || e.source !== parent) return;
    const d = e.data || {};
    switch (d.type) {
      case 'config':
        LUT = d.lut; HMAX = d.hmax || HMAX; heatMats = {};
        if (heatTex) { heatTex.dispose(); heatTex = null; }
        setMode(mode); break;
      case 'mode': setMode(d.mode); break;
      case 'ex': setEx(+d.v); break;
      case 'fly': $('bFly').click(); break;
      case 'reset': $('bReset').click(); break;
      case 'shot':                                            // read the canvas in the same task as the render
        renderer.render(scene, camera);
        send({ type: 'shot', url: renderer.domElement.toDataURL('image/png') }); break;
    }
  });

  // Hover: buildings -> their block height (template shows the tooltip); anything else -> surface height from HS
  const ray = new THREE.Raycaster(), p = new THREE.Vector2();
  let pending = null, raf = 0;
  function sampleHS(x, z) {
    const c = Math.round((x + D.SX / 2) / D.SX * (D.gw - 1)), r = Math.round((z + D.SZ / 2) / D.SZ * (D.gh - 1));
    if (c < 0 || r < 0 || c >= D.gw || r >= D.gh) return null;
    return HS[r * D.gw + c];
  }
  function hover() {
    raf = 0;
    const e = pending;
    p.set(e.clientX / innerWidth * 2 - 1, -e.clientY / innerHeight * 2 + 1);
    ray.setFromCamera(p, camera);
    if (group.visible) {
      const hit = ray.intersectObjects(group.children)[0];
      if (hit) { const b = hit.object.userData; send({ type: 'hover', h: b.h, capped: b.c, kind: 'building' }); return; }
    }
    const field = surface.visible ? surface : canopy;
    const hit = ray.intersectObject(field)[0];
    const h = hit ? sampleHS(hit.point.x, hit.point.z) : null;
    send({ type: 'hover', h, kind: 'surface' });
    if (h == null) { tip.style.display = 'none'; return; }
    tip.innerHTML = `<b>${h.toFixed(1)} m</b> surface height`;
    tip.style.display = 'block';
    tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 14) + 'px';
  }
  const cnv = renderer.domElement;
  cnv.addEventListener('pointermove', e => { pending = e; if (!raf) raf = requestAnimationFrame(hover); });
  cnv.addEventListener('pointerleave', () => { tip.style.display = 'none'; send({ type: 'hover', h: null }); });
})();
