(function () {
  'use strict';
  var started = Date.now();
  var mounted = false;
  var failed = false;
  var count = 0;
  var seen = {};
  var timer;
  function chinese() { return /^zh/i.test(document.documentElement.lang || navigator.language || ''); }
  function text(zh, en) { return chinese() ? zh : en; }
  function resourcePath(value) {
    if (!value) return '';
    var link = document.createElement('a');
    link.href = String(value);
    if (link.host !== window.location.host) return 'external';
    var path = link.pathname;
    return /^\/(?:assets|compat|src)\//.test(path) || /\/startup-recovery\.js$/.test(path) ? path.slice(0, 200) : '';
  }
  function report(kind, details) {
    details = details || {};
    var resource = resourcePath(details.resource);
    var key = kind + ':' + resource;
    if (seen[key] || count >= 10) return;
    seen[key] = true;
    count++;
    var meta = document.querySelector('meta[name="build-id"]');
    var record = {
      kind: String(kind).replace(/[^a-z_-]/g, '').slice(0, 40),
      phase: mounted ? 'page' : 'startup', resource: resource,
      build: meta ? meta.content.slice(0, 80) : 'development',
      elapsed_ms: Math.min(Date.now() - started, 3600000),
      at: new Date().toISOString(),
      site: document.documentElement.getAttribute('data-site') === 'tables' ? 'tables'
        : document.getElementById('human-app') ? 'play' : document.getElementById('live-app') ? 'live' : 'main',
      compat: document.documentElement.getAttribute('data-css-compat') || 'unknown'
    };
    try {
      var records = JSON.parse(window.localStorage.getItem('2048tables:load-failures') || '[]');
      if (!Array.isArray(records)) records = [];
      records.push(record);
      window.localStorage.setItem('2048tables:load-failures', JSON.stringify(records.slice(-20)));
    } catch (_) {}
    if (window.console && window.console.warn) window.console.warn('[2048 load]', record);
    var endpoint = document.querySelector('meta[name="load-diagnostics"]');
    if (!endpoint) return;
    try {
      var request = new XMLHttpRequest();
      request.open('POST', endpoint.content, true);
      request.timeout = 5000;
      request.setRequestHeader('Content-Type', 'application/json');
      request.send(JSON.stringify(record));
    } catch (_) {}
  }
  function render() {
    var panel = document.getElementById('app-startup');
    if (!panel || mounted) return;
    var slow = failed || Date.now() - started >= 12000;
    panel.querySelector('[data-startup-message]').textContent = failed
      ? text('\u9875\u9762\u52a0\u8f7d\u5931\u8d25', 'Unable to load the page')
      : slow ? text('\u52a0\u8f7d\u65f6\u95f4\u8f83\u957f\uff0c\u4ecd\u5728\u5c1d\u8bd5\u2026', 'Loading is taking longer than expected...')
        : text('\u6b63\u5728\u52a0\u8f7d\u2026', 'Loading...');
    var button = panel.querySelector('button');
    button.textContent = text('\u91cd\u65b0\u52a0\u8f7d', 'Reload page');
    button.style.display = slow ? 'inline-block' : 'none';
    button.onclick = function () { window.location.reload(); };
  }
  window.__APP_RECOVERY__ = {
    report: report,
    render: render,
    fail: function (kind, details) { failed = true; report(kind, details); render(); },
    mounted: function () {
      mounted = true;
      window.clearTimeout(timer);
      var panel = document.getElementById('app-startup');
      if (panel && panel.parentNode) panel.parentNode.removeChild(panel);
    }
  };
  window.addEventListener('error', function (event) {
    var target = event.target;
    if (target && (target.tagName === 'SCRIPT' || target.tagName === 'LINK')) {
      var url = target.src || target.href;
      var path = resourcePath(url);
      if (!path || path === 'external') return;
      // Preload failures are recorded by the actual script/style loader.
      if (target.rel === 'preload' || target.rel === 'modulepreload') return;
      report('resource_error', { resource: url });
      if (!mounted) { failed = true; render(); }
    } else if (!mounted) {
      window.__APP_RECOVERY__.fail('startup_error', { resource: event.filename });
    }
  }, true);
  window.addEventListener('vite:preloadError', function (event) {
    var match = String(event.payload && event.payload.message || '').match(/https?:\/\/[^\s]+/);
    report('chunk_error', { resource: match ? match[0] : '' });
  });
  window.addEventListener('unhandledrejection', function () {
    if (!mounted) window.__APP_RECOVERY__.fail('startup_rejection');
  });
  document.addEventListener('DOMContentLoaded', render);
  timer = window.setTimeout(function () {
    if (!mounted) { report('startup_timeout'); render(); }
  }, 12000);
}());
