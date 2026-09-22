(function () {
  'use strict';
  var root = document.documentElement;
  var reasons = [];
  var css = window.CSS;
  function supports(property, value) {
    return !!(css && css.supports && css.supports(property, value));
  }
  var style = document.createElement('style');
  var probe = document.createElement('div');
  probe.id = 'render-compat-probe';
  probe.style.cssText = 'position:absolute;visibility:hidden;pointer-events:none';
  style.textContent = '@layer compatProbe { #render-compat-probe { width:13px; } }';
  try {
    document.head.appendChild(style);
    root.appendChild(probe);
    if (window.getComputedStyle(probe).width !== '13px') reasons.push('cascade-layers');
  } catch (error) {
    reasons.push('cascade-layers');
  } finally {
    if (probe.parentNode) probe.parentNode.removeChild(probe);
    if (style.parentNode) style.parentNode.removeChild(style);
  }
  if (!supports('color', 'color-mix(in srgb, red, blue)')) reasons.push('color-mix');
  if (!supports('color', 'oklch(50% 0.1 120)')) reasons.push('oklch');
  if (!supports('aspect-ratio', '1 / 1')) reasons.push('aspect-ratio');
  if (!css || !css.registerProperty) reasons.push('registered-properties');
  root.setAttribute('data-css-compat', reasons.length ? 'loading' : 'modern');
  window.__RENDER_COMPAT__ = { mode: reasons.length ? 'compat' : 'modern', reasons: reasons };
  if (!reasons.length) return;

  // The built main page supplies a fingerprinted full stylesheet. Dev and live
  // keep using the smaller hand-maintained fallback.
  window.__loadRenderCompatFallback = function () {
    var link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = '/compat/render-compat.css?v=2';
    link.onload = function () { root.setAttribute('data-css-compat', 'ready'); };
    link.onerror = function () { root.setAttribute('data-css-compat', 'failed'); };
    document.head.appendChild(link);
  };
  if (!window.__RENDER_COMPAT_CSS_URL__) {
    document.write('<link id="render-compat-css" rel="stylesheet" href="/compat/render-compat.css?v=2" onload="document.documentElement.setAttribute(\'data-css-compat\',\'ready\')" onerror="document.documentElement.setAttribute(\'data-css-compat\',\'failed\')">');
    return;
  }
  document.write('<link id="render-compat-css" rel="stylesheet" href="' + window.__RENDER_COMPAT_CSS_URL__ + '" onload="document.documentElement.setAttribute(\'data-css-compat\',\'ready\')" onerror="window.__loadRenderCompatFallback()">');
}());
