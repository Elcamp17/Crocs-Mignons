(function () {
  const PANEL_ID = "repro-pairs-panel";
  const STATE = {
    rawPayload: null,
    workingPayload: null,
  };

  function escapeHtml(v) {
    return String(v ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function cloneDeep(obj) {
    try {
      return JSON.parse(JSON.stringify(obj));
    } catch {
      return obj;
    }
  }

  function normalizeRoot(payload) {
    // Support several shapes: { repro_report }, { report }, or direct report object
    if (!payload || typeof payload !== "object") return { root: null, report: null };
    const report =
      payload.repro_report ||
      payload.report ||
      payload.data?.repro_report ||
      payload.data?.report ||
      payload;
    return { root: payload, report };
  }

  function getSpecimens(report) {
    return Array.isArray(report?.specimens) ? report.specimens : [];
  }

  function createEmptyRecommendations() {
    return {
      top_pairs: [],
      mirror_pairs: [],
      global_pool_max: {},
      summary: {},
    };
  }

  function computeReproPairsFromSpecimens(specimens) {
    const active = (specimens || []).filter((s) => !s?._excluded);
    const females = active.filter((s) => String(s?.sex || "").toLowerCase() === "f");
    const males = active.filter((s) => String(s?.sex || "").toLowerCase() === "m");

    if (!females.length || !males.length) {
      return {
        ...createEmptyRecommendations(),
        summary: {
          note: "Pas assez de mâles/femelles après corrections (ou tout a été exclu).",
          active_count: active.length,
          excluded_count: (specimens || []).length - active.length,
        },
      };
    }

    const statKeys = ["health", "stamina", "weight", "oxygen", "food", "melee"];
    const poolMax = {};
    for (const key of statKeys) {
      const vals = active
        .map((s) => Number(s?.stats?.[key]))
        .filter((n) => Number.isFinite(n) && n >= 0);
      poolMax[key] = vals.length ? Math.max(...vals) : null;
    }

    const pairScores = [];
    for (const f of females) {
      for (const m of males) {
        let covered = 0;
        const coveredKeys = [];
        const missingKeys = [];

        for (const key of statKeys) {
          const target = poolMax[key];
          const fv = Number(f?.stats?.[key]);
          const mv = Number(m?.stats?.[key]);
          const ok =
            Number.isFinite(target) &&
            ((Number.isFinite(fv) && fv === target) || (Number.isFinite(mv) && mv === target));
          if (ok) {
            covered++;
            coveredKeys.push(key);
          } else {
            missingKeys.push(key);
          }
        }

        const pairConfidence = Math.round(
          (
            (((Number(f?.confidence) || 0) + (Number(m?.confidence) || 0)) / 2) *
            100
          )
        );

        pairScores.push({
          female_id: f.id ?? null,
          male_id: m.id ?? null,
          female_level: f.level ?? null,
          male_level: m.level ?? null,
          female_name: f.name || f.species || "-",
          male_name: m.name || m.species || "-",
          female_stats: f.stats || {},
          male_stats: m.stats || {},
          coverage_count: covered,
          coverage_total: statKeys.length,
          coverage_ratio: covered / statKeys.length,
          covered_keys: coveredKeys,
          missing_keys: missingKeys,
          confidence_pct: pairConfidence,
          explain_short:
            covered === statKeys.length
              ? "Très bon couple (toutes les stats max du pool sont couvertes)"
              : `Couvre ${covered}/${statKeys.length} stats max du pool`,
        });
      }
    }

    pairScores.sort((a, b) => {
      if (b.coverage_count !== a.coverage_count) return b.coverage_count - a.coverage_count;
      return (b.confidence_pct || 0) - (a.confidence_pct || 0);
    });

    const mirrors = [];
    for (let i = 0; i < active.length; i++) {
      for (let j = i + 1; j < active.length; j++) {
        const a = active[i], b = active[j];
        const sameSex = String(a?.sex || "").toLowerCase() === String(b?.sex || "").toLowerCase();
        if (sameSex) continue;
        const sameStats = JSON.stringify(a?.stats || {}) === JSON.stringify(b?.stats || {});
        if (sameStats) {
          mirrors.push({
            a_id: a.id ?? null,
            b_id: b.id ?? null,
            a_label: `${String(a?.sex || "?").toUpperCase()}${a?.level ?? "?"}`,
            b_label: `${String(b?.sex || "?").toUpperCase()}${b?.level ?? "?"}`,
          });
        }
      }
    }

    return {
      top_pairs: pairScores.slice(0, 6),
      mirror_pairs: mirrors.slice(0, 8),
      global_pool_max: poolMax,
      summary: {
        active_count: active.length,
        excluded_count: (specimens || []).length - active.length,
      },
    };
  }

  function statLabel(key) {
    return (
      {
        health: "Vie",
        stamina: "Endu",
        weight: "Poids",
        oxygen: "Oxy",
        food: "Nour",
        melee: "Att",
      }[key] || key
    );
  }

  function formatStats(stats) {
    if (!stats || typeof stats !== "object") return "-";
    const order = ["health", "stamina", "weight", "oxygen", "food", "melee"];
    return order
      .map((k) => `${statLabel(k)}:${Number.isFinite(Number(stats[k])) ? Number(stats[k]) : "-"}`)
      .join(" · ");
  }

  function ensurePanel() {
    const holder = document.getElementById("reproPairsMountPoint");
    if (!holder) return null;

    let panel = document.getElementById(PANEL_ID);
    if (panel) return panel;

    panel = document.createElement("div");
    panel.id = PANEL_ID;
    panel.className = "repro-pairs-panel";
    holder.appendChild(panel);
    return panel;
  }

  function renderPairsSection(container, payload) {
    const { report } = normalizeRoot(payload);
    const reco = report?.recommendations || createEmptyRecommendations();
    const topPairs = Array.isArray(reco.top_pairs) ? reco.top_pairs : [];
    const mirrors = Array.isArray(reco.mirror_pairs) ? reco.mirror_pairs : [];
    const summary = reco.summary || {};

    const best = topPairs[0];
    const bestHtml = best
      ? `
        <div class="repro-card success">
          <div class="repro-card-title">✅ Meilleur couple à lancer maintenant</div>
          <div class="repro-best-pair">M${escapeHtml(best.male_level)} × F${escapeHtml(best.female_level)}</div>
          <div class="repro-best-meta">
            Couverture : ${escapeHtml(best.coverage_count)}/${escapeHtml(best.coverage_total)}
            • ${escapeHtml(best.explain_short || "")}
          </div>
        </div>`
      : `
        <div class="repro-card warning">
          <div class="repro-card-title">⚠️ Aucune paire exploitable</div>
          <div class="repro-best-meta">Pas assez de dinos valides (mâle + femelle) après corrections.</div>
        </div>`;

    const mirrorsHtml = mirrors.length
      ? `
        <div class="repro-card">
          <div class="repro-card-title">💡 Paires miroir détectées</div>
          <ul class="repro-mirror-list">
            ${mirrors.map((m) => `<li>${escapeHtml(m.a_label)} = ${escapeHtml(m.b_label)}</li>`).join("")}
          </ul>
        </div>`
      : "";

    const pairsListHtml = topPairs.length
      ? `
        <div class="repro-card">
          <div class="repro-card-title">📌 Top couples (après corrections)</div>
          <div class="repro-list">
            ${topPairs
              .map(
                (p, idx) => `
              <div class="repro-list-item">
                <div class="repro-list-head">
                  <span class="rank">#${idx + 1}</span>
                  <span class="pair">M${escapeHtml(p.male_level)} × F${escapeHtml(p.female_level)}</span>
                  <span class="coverage">${escapeHtml(p.coverage_count)}/${escapeHtml(p.coverage_total)}</span>
                </div>
                <div class="repro-list-sub">${escapeHtml(p.explain_short || "")}</div>
                ${
                  Array.isArray(p.missing_keys) && p.missing_keys.length
                    ? `<div class="repro-list-missing">Manque: ${escapeHtml(
                        p.missing_keys.map(statLabel).join(", ")
                      )}</div>`
                    : `<div class="repro-list-missing ok">Toutes les stats max du pool sont couvertes</div>`
                }
              </div>`
              )
              .join("")}
          </div>
        </div>`
      : "";

    const summaryHtml = `
      <div class="repro-card">
        <div class="repro-card-title">🧮 Résumé corrections</div>
        <div class="repro-summary-grid">
          <div><span>Pris en compte</span><strong>${escapeHtml(summary.active_count ?? "-")}</strong></div>
          <div><span>Exclus</span><strong>${escapeHtml(summary.excluded_count ?? 0)}</strong></div>
        </div>
      </div>`;

    container.innerHTML = `
      <div class="repro-pairs-header">
        <h3>Couples conseillés (repro)</h3>
        <p>Calcul local basé sur les dinos détectés, avec corrections manuelles possibles.</p>
      </div>
      ${summaryHtml}
      ${bestHtml}
      ${mirrorsHtml}
      ${pairsListHtml}
      <div id="reproEditorPanel"></div>
    `;
  }

  function getWorkingSpecimens() {
    const { report } = normalizeRoot(STATE.workingPayload || {});
    return getSpecimens(report);
  }

  function parseIntOrKeep(input, fallback) {
    if (input === null) return fallback;
    const txt = String(input).trim();
    if (!txt) return fallback;
    const n = Number(txt.replace(",", "."));
    return Number.isFinite(n) ? Math.round(n) : fallback;
  }

  function editSpecimenByPrompt(idx) {
    const specimens = getWorkingSpecimens();
    const s = specimens[idx];
    if (!s) return;

    const sex = prompt("Sexe (m / f / -)", s.sex ?? "-");
    if (sex === null) return;

    const level = prompt("Niveau", s.level ?? "");
    if (level === null) return;

    const stage = prompt("Stage (baby/adult/-)", s.stage ?? "-");
    if (stage === null) return;

    const currentStats = [
      s?.stats?.health ?? "",
      s?.stats?.stamina ?? "",
      s?.stats?.weight ?? "",
      s?.stats?.oxygen ?? "",
      s?.stats?.food ?? "",
      s?.stats?.melee ?? "",
    ].join(",");
    const statsLine = prompt(
      "Stats (ordre: Vie,Endu,Poids,Oxy,Nour,Att) séparées par virgules",
      currentStats
    );
    if (statsLine === null) return;

    const conf = prompt("Confiance (%)", Math.round((Number(s.confidence) || 0) * 100));
    if (conf === null) return;

    const parts = String(statsLine)
      .split(/[;,]/)
      .map((x) => x.trim());

    const [h, st, w, o, f, m] = parts;
    s.sex = (String(sex).trim().toLowerCase() || s.sex || "-").slice(0, 1);
    s.level = parseIntOrKeep(level, s.level);
    s.stage = String(stage).trim() || s.stage || "-";
    s.stats = s.stats || {};
    s.stats.health = parseIntOrKeep(h, s.stats.health);
    s.stats.stamina = parseIntOrKeep(st, s.stats.stamina);
    s.stats.weight = parseIntOrKeep(w, s.stats.weight);
    s.stats.oxygen = parseIntOrKeep(o, s.stats.oxygen);
    s.stats.food = parseIntOrKeep(f, s.stats.food);
    s.stats.melee = parseIntOrKeep(m, s.stats.melee);

    const confPct = parseIntOrKeep(conf, Math.round((Number(s.confidence) || 0) * 100));
    s.confidence = Math.max(0, Math.min(1, (Number(confPct) || 0) / 100));
    s._edited = true;
    rerenderFromWorking();
  }

  function toggleExcludeSpecimen(idx) {
    const specimens = getWorkingSpecimens();
    const s = specimens[idx];
    if (!s) return;
    s._excluded = !s._excluded;
    rerenderFromWorking();
  }

  function markFalseSpecimen(idx) {
    const specimens = getWorkingSpecimens();
    const s = specimens[idx];
    if (!s) return;
    s._excluded = true;
    s._flag_false = true;
    rerenderFromWorking();
  }

  function relaunchFullScanFromRow() {
    const btn =
      document.querySelector("#analyzeVideoBtn, [data-action='analyze-video'], .btn-analyze-video") ||
      Array.from(document.querySelectorAll("button")).find((b) =>
        /analyser la vidéo/i.test((b.textContent || "").trim())
      );
    if (btn) {
      alert("La relecture IA par dino n'est pas encore supportée côté backend.\nJe relance le scan complet pour l'instant.");
      btn.click();
    } else {
      alert("Relecture par dino non disponible côté backend pour l'instant. Tu peux modifier/exclure la ligne puis relancer le scan complet.");
    }
  }

  function resetCorrections() {
    if (!STATE.rawPayload) return;
    STATE.workingPayload = cloneDeep(STATE.rawPayload);
    rerenderFromWorking();
  }

  function applyRecomputedRecommendationsOnWorking() {
    const { report } = normalizeRoot(STATE.workingPayload || {});
    if (!report) return;
    const specs = getSpecimens(report);
    report.recommendations = computeReproPairsFromSpecimens(specs);
  }

  function renderEditorPanel() {
    const host = document.getElementById("reproEditorPanel");
    if (!host) return;

    const { report } = normalizeRoot(STATE.workingPayload || {});
    const specimens = getSpecimens(report);
    if (!specimens.length) {
      host.innerHTML = "";
      return;
    }

    const rowsHtml = specimens
      .map((s, idx) => {
        const excluded = !!s._excluded;
        const cls = [
          "repro-edit-row",
          excluded ? "excluded" : "",
          s._edited ? "edited" : "",
          s._flag_false ? "flagfalse" : "",
        ]
          .filter(Boolean)
          .join(" ");

        return `
          <tr class="${cls}" data-idx="${idx}">
            <td>${idx + 1}</td>
            <td>${escapeHtml(s.sex || "-")}</td>
            <td>${escapeHtml(s.name || s.species || "-")}</td>
            <td>${escapeHtml(s.level ?? "-")}</td>
            <td>${escapeHtml(s.stage || "-")}</td>
            <td class="stats-cell">${escapeHtml(formatStats(s.stats || {}))}</td>
            <td>${escapeHtml(Math.round((Number(s.confidence) || 0) * 100))}%</td>
            <td class="actions">
              <button type="button" class="repro-mini-btn" data-rp-action="edit" data-rp-idx="${idx}">✏️ Modifier</button>
              <button type="button" class="repro-mini-btn ${excluded ? "" : "danger"}" data-rp-action="toggle-exclude" data-rp-idx="${idx}">
                ${excluded ? "↩️ Inclure" : "🗑 Exclure"}
              </button>
              <button type="button" class="repro-mini-btn warn" data-rp-action="mark-false" data-rp-idx="${idx}">⚠️ Faux</button>
              <button type="button" class="repro-mini-btn" data-rp-action="relaunch" data-rp-idx="${idx}">🔄 Refaire IA</button>
            </td>
          </tr>
        `;
      })
      .join("");

    host.innerHTML = `
      <div class="repro-card repro-edit-card">
        <div class="repro-card-title">🛠️ Correction manuelle des dinos détectés</div>
        <div class="repro-edit-help">
          Si une ligne est mal lue (ex: stats à 249 partout), tu peux la <strong>modifier</strong> ou l’<strong>exclure</strong>.
          Le calcul des couples ci-dessus est recalculé automatiquement avec tes corrections.
        </div>
        <div class="repro-edit-toolbar">
          <button type="button" class="repro-mini-btn" data-rp-action="reset-all">♻️ Réinitialiser les corrections</button>
          <button type="button" class="repro-mini-btn" data-rp-action="relaunch-all">🚀 Relancer le scan complet</button>
        </div>
        <div class="repro-edit-table-wrap">
          <table class="repro-edit-table">
            <thead>
              <tr>
                <th>#</th><th>Sexe</th><th>Nom</th><th>Niveau</th><th>Stage</th><th>Stats</th><th>Conf</th><th>Actions</th>
              </tr>
            </thead>
            <tbody>${rowsHtml}</tbody>
          </table>
        </div>
      </div>
    `;
  }

  function bindEditorEvents() {
    const panel = document.getElementById(PANEL_ID);
    if (!panel || panel.__rpBound) return;
    panel.__rpBound = true;

    panel.addEventListener("click", (ev) => {
      const btn = ev.target.closest("[data-rp-action]");
      if (!btn) return;
      const action = btn.getAttribute("data-rp-action");
      const idx = Number(btn.getAttribute("data-rp-idx"));

      if (action === "edit") return editSpecimenByPrompt(idx);
      if (action === "toggle-exclude") return toggleExcludeSpecimen(idx);
      if (action === "mark-false") return markFalseSpecimen(idx);
      if (action === "relaunch") return relaunchFullScanFromRow();
      if (action === "reset-all") return resetCorrections();
      if (action === "relaunch-all") return relaunchFullScanFromRow();
    });
  }

  function rerenderFromWorking() {
    const panel = ensurePanel();
    if (!panel || !STATE.workingPayload) return;
    applyRecomputedRecommendationsOnWorking();
    renderPairsSection(panel, STATE.workingPayload);
    bindEditorEvents();
    renderEditorPanel();
  }

  function renderReproPairsPanel(payload) {
    if (!payload) {
      resetReproPairsPanel();
      return;
    }
    STATE.rawPayload = cloneDeep(payload);
    STATE.workingPayload = cloneDeep(payload);
    rerenderFromWorking();
  }

  function resetReproPairsPanel() {
    STATE.rawPayload = null;
    STATE.workingPayload = null;
    const panel = document.getElementById(PANEL_ID);
    if (panel) {
      panel.remove();
    }
  }

  window.renderReproPairsPanel = renderReproPairsPanel;
  window.resetReproPairsPanel = resetReproPairsPanel;
})();