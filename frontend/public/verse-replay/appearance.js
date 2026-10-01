(function () {
  'use strict';
  var root = document.documentElement;
  var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  var style = document.createElement('style');
  style.id = 'replay-tile-theme';
  document.head.appendChild(style);
  var revision = 0;
  var lastRefresh = 0;
  var inFlight = false;
  var activePreferences = {};
  var localSignature = '';
  var resolvedThemeId = 0;
  var resolvedTheme = null;
  var colorPattern = /^#[\da-f]{6}(?:[\da-f]{2}|\s*\/\s*(?:0(?:\.\d+)?|1(?:\.0+)?|\d{1,3}%))?$/i;

  function read(key, kind) {
    try { return JSON.parse(window[kind || 'localStorage'].getItem(key) || 'null'); }
    catch (_) { return null; }
  }
  function cookie(name) {
    try {
      var items = document.cookie.split(';');
      for (var i = 0; i < items.length; i++) {
        var item = items[i].trim();
        if (item.indexOf(name + '=') === 0) return JSON.parse(decodeURIComponent(item.slice(name.length + 1)));
      }
    } catch (_) { /* Optional shared palette or fallback session. */ }
    return null;
  }
  function localPreferences() {
    var saved = read('2048tables:user-preferences');
    return saved && saved.version === 1 && saved.value && typeof saved.value === 'object' ? saved.value : {};
  }
  function pendingPreferences() {
    var owner = read('2048tables:account-preferences-owner');
    var pending = read('2048tables:account-preferences-pending');
    return owner && pending && pending.value ? pending.value[owner.value] || {} : {};
  }
  function validColor(value) { return typeof value === 'string' && colorPattern.test(value.trim()); }
  function placeholder(palette) {
    return palette.length && palette.every(function (item) {
      return /^(#000|#000000|black)$/i.test(typeof item === 'string' ? item : item.background);
    });
  }
  // Matches the main site's resolveTileColors, including its isolated light-tile rule.
  function customPalette(colors) {
    if (!Array.isArray(colors) || !colors.length || placeholder(colors)) return [];
    if (!colors.every(function (color) { return /^#[\da-f]{6}$/i.test(color); })) return [];
    var flags = colors.map(function (color) {
      var rgb = [1, 3, 5].map(function (offset) { return parseInt(color.slice(offset, offset + 2), 16); });
      return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.299 * 238 + 0.587 * 218 + 0.114 * 179;
    });
    var filled = flags.slice();
    for (var i = 1; i < flags.length - 1; i++) if (!flags[i] && flags[i - 1] && flags[i + 1]) filled[i] = true;
    return colors.map(function (color, index) { return { background: color, color: filled[index] ? '#f9f6f2' : '#776e65' }; });
  }
  function applyPalette(palette, superTile) {
    var rules = [];
    function rule(className, tile) {
      if (!tile || !validColor(tile.background) || !validColor(tile.color)) return;
      var shadow = validColor(tile.shadow) ? tile.shadow : '#00000000';
      var outline = validColor(tile.outline) ? tile.outline : '#ffffff22';
      rules.push(':root .tile.' + className + ',:root .node-badge.' + className + '{background:' + tile.background + ';color:' + tile.color + ';box-shadow:0 0 10px ' + shadow + ',inset 0 0 0 1px ' + outline + ';}');
    }
    palette.slice(0, 36).forEach(function (tile, index) { rule('value-' + Math.pow(2, index + 1), tile); });
    rule('value-super', superTile);
    style.textContent = rules.join('\n');
  }
  function applySaved(theme, dark) {
    if (!theme || typeof theme !== 'object') return false;
    var mode = dark ? theme.dark || theme.light : theme.light || theme.dark;
    if (!mode) return false;
    function tile(value) {
      var entry = mode[value];
      return entry && { background: entry['--tile-background'], color: entry['--tile-color'], shadow: entry['--tile-shadow-color'], outline: entry['--tile-outline-color'] };
    }
    var palette = Array.from({ length: 16 }, function (_, i) { return tile(Math.pow(2, i + 1)); });
    if (!palette.every(function (entry) { return entry && validColor(entry.background) && validColor(entry.color); })) return false;
    applyPalette(palette, tile('Super'));
    return true;
  }
  function apply(preferences) {
    activePreferences = preferences;
    var dark = typeof preferences.dark_mode === 'boolean' ? preferences.dark_mode : !!(media && media.matches);
    root.setAttribute('data-theme', dark ? 'dark' : 'light');
    var palette = preferences.use_custom_theme ? customPalette(preferences.custom_colors) : (window.ReplayThemeCatalog || {})[preferences.theme];
    if (!palette || !palette.length) palette = cookie('2048tables-tile-palette') || customPalette(preferences.colors);
    if (!Array.isArray(palette) || placeholder(palette)) palette = [];
    applyPalette(palette);
    var cache = read('saved-vth-theme-cache-v1');
    var saved = cache && cache[preferences.saved_theme_id];
    if (Number(preferences.saved_theme_id) > 0) {
      if (Number(preferences.saved_theme_id) === resolvedThemeId) applySaved(resolvedTheme, dark);
      else if (saved) applySaved(saved.theme, dark);
    }
  }
  function headers() {
    var session = read('2048tables:device-session-token') || read('2048tables:device-session-token', 'sessionStorage') || cookie('tb_device_session_fallback');
    return session && session.token && (!session.expires_at || Date.parse(session.expires_at) > Date.now()) ? { Authorization: 'Bearer ' + session.token } : {};
  }
  function get(path) {
    var controller = typeof AbortController === 'function' ? new AbortController() : null;
    var timer = controller ? setTimeout(function () { controller.abort(); }, 5000) : null;
    return fetch(path, { credentials: 'include', cache: 'no-store', headers: headers(), signal: controller ? controller.signal : undefined })
      .then(function (response) { if (!response.ok) throw new Error('Appearance unavailable'); return response.json(); })
      .then(function (result) { clearTimeout(timer); return result; }, function (error) { clearTimeout(timer); throw error; });
  }
  function refresh() {
    if (inFlight || Date.now() - lastRefresh < 30000 || typeof fetch !== 'function') return;
    inFlight = true;
    lastRefresh = Date.now();
    var serial = revision;
    get('/api/profile/preferences').then(function (result) {
      if (serial !== revision || !result.preferences) return;
      var preferences = Object.assign({}, result.preferences, pendingPreferences());
      apply(preferences);
      if (Number(preferences.saved_theme_id) > 0) {
        return get('/api/profile/themes/' + Number(preferences.saved_theme_id)).then(function (item) {
          if (serial === revision) {
            resolvedThemeId = Number(preferences.saved_theme_id);
            resolvedTheme = item.theme;
            applySaved(item.theme, root.getAttribute('data-theme') === 'dark');
          }
        });
      }
    }).catch(function () { /* Keep the cached appearance; playback must never depend on this request. */ })
      .then(function () {
        inFlight = false;
        if (serial !== revision) { lastRefresh = 0; refresh(); }
      });
  }
  function resume() {
    if (document.visibilityState === 'hidden') return;
    var local = localPreferences();
    if (JSON.stringify(local) !== localSignature) {
      revision++;
      localSignature = JSON.stringify(local);
      apply(local);
    }
    refresh();
  }
  localSignature = JSON.stringify(localPreferences());
  apply(localPreferences());
  refresh();
  window.addEventListener('storage', function (event) {
    if (['2048tables:user-preferences', '2048tables:account-preferences-owner', '2048tables:account-preferences-pending', '2048tables:device-session-token', 'saved-vth-theme-cache-v1'].indexOf(event.key) !== -1 || event.key === null) {
      revision++;
      localSignature = JSON.stringify(localPreferences());
      apply(localPreferences());
    }
  });
  window.addEventListener('focus', resume);
  document.addEventListener('visibilitychange', resume);
  function systemChanged() { if (typeof activePreferences.dark_mode !== 'boolean') apply(activePreferences); }
  if (media && media.addEventListener) media.addEventListener('change', systemChanged);
  else if (media && media.addListener) media.addListener(systemChanged);
})();
