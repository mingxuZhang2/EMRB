/* EMRB 评测页交互：标签切换、榜单渲染（精简/完整）、排序、热力单元格与悬浮提示。 */
(function () {
  'use strict';

  var D = window.EMRB;

  /* 蓝色顺序色阶 100 至 700，与 styles.css 中的 --seq-* 保持一致。 */
  var RAMP = [
    { s: 100, hex: '#cde2fb' }, { s: 150, hex: '#b7d3f6' }, { s: 200, hex: '#9ec5f4' },
    { s: 250, hex: '#86b6ef' }, { s: 300, hex: '#6da7ec' }, { s: 350, hex: '#5598e7' },
    { s: 400, hex: '#3987e5' }, { s: 450, hex: '#2a78d6' }, { s: 500, hex: '#256abf' },
    { s: 550, hex: '#1c5cab' }, { s: 600, hex: '#184f95' }, { s: 650, hex: '#104281' },
    { s: 700, hex: '#0d366b' }
  ];

  /* 得分率 0 至 100 映射到色阶。500 步及更深用白字，保证对比度。 */
  function scale(v) {
    if (v === null || v === undefined) return { bg: '#f7f8fb', fg: 'var(--ink-3)' };
    var i = Math.min(RAMP.length - 1, Math.max(0, Math.round((v / 100) * (RAMP.length - 1))));
    return { bg: RAMP[i].hex, fg: RAMP[i].s >= 500 ? '#ffffff' : '#12233f' };
  }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }

  function fmt(v, dp) {
    return v === null || v === undefined ? '无' : v.toFixed(dp === undefined ? 1 : dp);
  }

  /* ---------- 标签切换 ---------- */

  function initTabs() {
    var tabs = [].slice.call(document.querySelectorAll('.tab'));

    function select(t, scroll) {
      tabs.forEach(function (o) {
        var on = o === t;
        o.setAttribute('aria-selected', on ? 'true' : 'false');
        document.getElementById(o.dataset.panel).hidden = !on;
      });
      if (scroll) {
        var bar = document.querySelector('.tabbar');
        var y = bar.getBoundingClientRect().top + window.pageYOffset - 70;
        window.scrollTo({ top: y, behavior: 'smooth' });
      }
    }

    /* 支持 #board / #arch 直接定位到某个标签页 */
    var hash = (window.location && window.location.hash || '').replace('#', '');
    var byHash = tabs.filter(function (t) { return t.dataset.panel === 'panel-' + hash; })[0];
    if (byHash) select(byHash, false);

    tabs.forEach(function (t) {
      t.addEventListener('click', function () {
        select(t, true);
      });
    });
    document.querySelectorAll('[data-goto]').forEach(function (a) {
      a.addEventListener('click', function (e) {
        e.preventDefault();
        var t = document.querySelector('.tab[data-panel="' + a.dataset.goto + '"]');
        if (t) t.click();
      });
    });
  }

  /* ---------- 悬浮提示 ---------- */

  var tip = el('div', 'tip');
  document.body.appendChild(tip);

  function showTip(html, ev) {
    tip.innerHTML = html;
    tip.dataset.show = '1';
    var pad = 14;
    var r = tip.getBoundingClientRect();
    var x = Math.min(ev.clientX + pad, window.innerWidth - r.width - 8);
    var y = ev.clientY - r.height - pad;
    if (y < 8) y = ev.clientY + pad;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }

  function hideTip() { tip.dataset.show = '0'; }

  /* ---------- 榜单 ---------- */

  var state = { mode: 'level', sortKey: 'avg', sortDir: 'desc' };

  /* 支持 ?mode=type 直接进入完整榜 */
  if (/[?&]mode=type/.test(window.location ? window.location.search : '')) {
    state.mode = 'type';
  }

  function columns() {
    if (state.mode === 'level') {
      return D.levels.map(function (l, i) {
        return { key: 'lv' + i, group: l.key, label: l.key, full: l.key + '　' + l.title, get: function (m) { return m.levels[i]; } };
      });
    }
    var cols = [], n = 0;
    D.groups.forEach(function (g) {
      g.types.forEach(function (t) {
        var idx = n++;
        cols.push({ key: 't' + idx, group: g.key, label: t, full: g.key + '　' + t, get: function (m) { return m.types[idx]; } });
      });
    });
    return cols;
  }

  function sortedModels(cols) {
    var rows = D.models.slice();
    var col = cols.filter(function (c) { return c.key === state.sortKey; })[0];
    var val = state.sortKey === 'avg'
      ? function (m) { return m.avg; }
      : state.sortKey === 'name'
        ? null
        : (col ? col.get : function (m) { return m.avg; });
    rows.sort(function (a, b) {
      if (state.sortKey === 'name') {
        return a.name.localeCompare(b.name) * (state.sortDir === 'asc' ? 1 : -1);
      }
      var x = val(a), y = val(b);
      if (x === null || x === undefined) x = -1;
      if (y === null || y === undefined) y = -1;
      return (x - y) * (state.sortDir === 'asc' ? 1 : -1);
    });
    return rows;
  }

  function render() {
    var cols = columns();
    var rows = sortedModels(cols);
    var host = document.getElementById('lb');
    host.textContent = '';

    var byType = state.mode === 'type';
    var table = el('table', 'lb ' + (byType ? 'lb--type' : 'lb--level'));
    var thead = el('thead');

    /* 固定列表头：序号、模型、源类型 */
    function fixedHeads(row) {
      var thRank = el('th', 'fixed-head stick1 h-rank', '#');
      var thName = el('th', 'fixed-head stick2 h-model sortable', '模型');
      thName.dataset.k = 'name';
      var thSrc = el('th', 'fixed-head h-src', '源类型');
      [thRank, thName, thSrc].forEach(function (th) { row.appendChild(th); });
    }

    function avgHead(row) {
      var th = el('th', 'sortable', '总体得分');
      th.dataset.k = 'avg';
      row.appendChild(th);
    }

    if (byType) {
      /* 上级表头：五个能力层次分组，固定列与总体得分列在此行留空 */
      var grp = el('tr', 'grp');
      ['fixed-head stick1', 'fixed-head stick2', 'fixed-head'].forEach(function (cls) {
        grp.appendChild(el('th', cls, ''));
      });
      D.groups.forEach(function (g, gi) {
        var th = el('th', 'grp-cell', g.key + '　' + g.types.length + ' 类');
        th.colSpan = g.types.length;
        th.title = D.levels[gi].title;
        grp.appendChild(th);
      });
      grp.appendChild(el('th', 'fixed-head', ''));
      thead.appendChild(grp);

      /* 下级表头：27 个题型，竖排 */
      var hdr = el('tr', 'hdr');
      fixedHeads(hdr);
      cols.forEach(function (c) {
        var th = el('th', 'sortable rot');
        th.dataset.k = c.key;
        th.title = c.full;
        th.appendChild(el('span', null, c.label));
        hdr.appendChild(th);
      });
      avgHead(hdr);
      thead.appendChild(hdr);
    } else {
      /* 精简榜只有一行表头 */
      var row = el('tr', 'hdr');
      fixedHeads(row);
      cols.forEach(function (c, i) {
        var th = el('th', 'sortable');
        th.dataset.k = c.key;
        th.title = D.levels[i].title;
        th.textContent = c.label;
        row.appendChild(th);
      });
      avgHead(row);
      thead.appendChild(row);
    }
    table.appendChild(thead);

    /* 排序标记与点击 */
    thead.querySelectorAll('th.sortable').forEach(function (th) {
      if (th.dataset.k === state.sortKey) th.dataset.sorted = state.sortDir;
      th.addEventListener('click', function () {
        if (state.sortKey === th.dataset.k) {
          state.sortDir = state.sortDir === 'desc' ? 'asc' : 'desc';
        } else {
          state.sortKey = th.dataset.k;
          state.sortDir = th.dataset.k === 'name' ? 'asc' : 'desc';
        }
        render();
      });
    });

    var tbody = el('tbody');
    rows.forEach(function (m, ri) {
      var tr = el('tr');
      tr.appendChild(el('td', 'c-rank stick1', String(ri + 1)));

      var tdm = el('td', 'c-model stick2');
      tdm.appendChild(document.createTextNode(m.name));
      tdm.appendChild(el('span', 'vendor', m.vendor));
      tr.appendChild(tdm);

      var tds = el('td', 'c-src');
      tds.appendChild(el('span', 'tag ' + (m.src === '开源' ? 'open' : 'closed'), m.src));
      tr.appendChild(tds);

      cols.forEach(function (c) {
        var v = c.get(m);
        var td = el('td', 'cell', fmt(v, state.mode === 'level' ? 1 : 1));
        var s = scale(v);
        td.style.background = s.bg;
        td.style.color = s.fg;
        td.addEventListener('mousemove', function (ev) {
          showTip('<b>' + m.name + '</b><br>' + c.full + '<br>得分率 ' + fmt(v, 2) + '%', ev);
        });
        td.addEventListener('mouseleave', hideTip);
        tr.appendChild(td);
      });

      var tda = el('td', 'cell avg', m.avg.toFixed(1));
      var sa = scale(m.avg);
      tda.style.background = sa.bg;
      tda.style.color = sa.fg;
      tr.appendChild(tda);
      tbody.appendChild(tr);
    });

    /* 跨模型均值行 */
    var mr = el('tr', 'means');
    mr.appendChild(el('td', 'c-rank stick1', ''));
    mr.appendChild(el('td', 'c-model stick2', '跨模型均值'));
    mr.appendChild(el('td', 'c-src', ''));
    var means = state.mode === 'level' ? D.levelMeans : D.typeMeans;
    means.forEach(function (v) { mr.appendChild(el('td', 'cell', v.toFixed(1))); });
    var all = D.models.reduce(function (a, m) { return a + m.avg; }, 0) / D.models.length;
    mr.appendChild(el('td', 'cell avg', all.toFixed(1)));
    tbody.appendChild(mr);

    table.appendChild(tbody);
    host.appendChild(table);
  }

  function initLeaderboard() {
    document.querySelectorAll('#lb-mode button').forEach(function (b) {
      b.setAttribute('aria-pressed', b.dataset.mode === state.mode ? 'true' : 'false');
      b.addEventListener('click', function () {
        document.querySelectorAll('#lb-mode button').forEach(function (o) {
          o.setAttribute('aria-pressed', o === b ? 'true' : 'false');
        });
        state.mode = b.dataset.mode;
        state.sortKey = 'avg';
        state.sortDir = 'desc';
        render();
      });
    });
    render();
  }

  /* ---------- 指标卡 ---------- */

  function initStats() {
    var host = document.getElementById('stats');
    if (!host) return;
    [
      ['200', '道题目', ''],
      ['5', '个难度等级', ''],
      ['27', '类题型', ''],
      ['11', '种信号类型', ''],
      ['920', '个子问题', ''],
      ['24.1–78.9', '总分区间', '%']
    ].forEach(function (s) {
      var c = el('div', 'stat');
      var v = el('div', 'v', s[0]);
      if (s[2]) v.appendChild(el('small', null, s[2]));
      c.appendChild(v);
      c.appendChild(el('div', 'k', s[1]));
      host.appendChild(c);
    });
  }

  /* ---------- 复制 BibTeX ---------- */

  function initCopy() {
    document.querySelectorAll('.bib-wrap .copy').forEach(function (b) {
      b.addEventListener('click', function () {
        var txt = b.parentNode.querySelector('pre').textContent;
        var done = function () {
          var old = b.textContent;
          b.textContent = '已复制';
          setTimeout(function () { b.textContent = old; }, 1600);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(txt).then(done, done);
        } else {
          var ta = document.createElement('textarea');
          ta.value = txt;
          document.body.appendChild(ta);
          ta.select();
          try { document.execCommand('copy'); } catch (e) { /* 忽略 */ }
          document.body.removeChild(ta);
          done();
        }
      });
    });
  }

  initTabs();
  initStats();
  initLeaderboard();
  initCopy();
})();
