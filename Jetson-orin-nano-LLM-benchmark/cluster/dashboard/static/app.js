const state = {
  token: "",
  nodes: [],
  status: [],
  models: [],
  selectedNodes: new Set(),
  runs: [],
  experimentGroups: [],
  actions: [],
  activeExperiment: null,
  onboarding: {},
  settings: { worker_api_auth: false, dashboard_token_auth: false },
  environment: [],
  environmentBusy: false,
  environmentActionIds: new Set(),
  onboardingProbe: null,
  devices: [],
  eventSource: null,
  metricHistory: new Map(),
  detailNode: "",
  selectedModels: [],
  suites: [],
  chartModels: new Map(),
  chartInteraction: new Map(),
  publicationChartId: "",
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function getToken() {
  const fromUrl = new URLSearchParams(location.search).get("token");
  if (fromUrl) {
    sessionStorage.setItem("clusterToken", fromUrl);
    const clean = new URL(location.href);
    clean.searchParams.delete("token");
    history.replaceState({}, "", clean.pathname + clean.search + clean.hash);
  }
  return fromUrl || sessionStorage.getItem("clusterToken") || "";
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (state.token) headers["X-Cluster-Token"] = state.token;
  if (options.body && typeof options.body !== "string") {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (response.status === 401) {
    const dialog = $("#authDialog");
    if (dialog && !dialog.open) dialog.showModal();
  }
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  return data;
}

function toast(title, message = "", kind = "success") {
  const item = document.createElement("div");
  item.className = `toast ${kind === "error" ? "error" : ""}`;
  item.innerHTML = `<strong>${escapeHtml(title)}</strong>${message ? `<span>${escapeHtml(message)}</span>` : ""}`;
  $("#toastStack").append(item);
  setTimeout(() => item.remove(), 4800);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function finite(value) {
  return value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
}

const fmt = (value, digits = 1, fallback = "—") => finite(value) ? Number(value).toFixed(digits) : fallback;
const pct = value => finite(value) ? `${fmt(value, 0)}%` : "—";
const DASHBOARD_COLORS = ["#718f17", "#e57c38", "#163126", "#0072b2", "#cc79a7", "#f0e442"];
const PUBLICATION_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"];
const platformName = value => ({ jetson: "NVIDIA Jetson", "raspberry-pi": "Raspberry Pi 5", auto: "자동 감지", "generic-linux": "Linux" }[value] || value || "미확인");
const STRATEGIES = {
  single_node: { label: "단일 노드 기준선", short: "SINGLE" },
  replicated_round_robin: { label: "복제 · 요청 분산", short: "ROUND ROBIN" },
  broadcast_compare: { label: "전체 동시 전송", short: "BROADCAST" },
  node_sweep: { label: "노드 수 스윕", short: "NODE SWEEP" },
  model_parallel_rpc: { label: "모델 분할 · RPC", short: "MODEL RPC", experimental: true },
  legacy: { label: "이전 방식 · 복제 요청 분산", short: "LEGACY RR" },
};

function selectedStrategy() {
  return $('input[name="execution_strategy"]:checked')?.value || "replicated_round_robin";
}

function plannedNodeNames() {
  return [...state.selectedNodes];
}

function runStrategy(run) {
  return run.execution_strategy || run.strategy || run.config?.execution_strategy || run.definition?.default_config?.execution_strategy || "legacy";
}

function runModelId(run) {
  const actual = Array.isArray(run.actual_model_config) ? run.actual_model_config[0] : run.actual_model_config;
  return run.benchmark_parameters?.model_id || run.model_id || actual?.model_id || actual?.model || "unknown-model";
}

function shortModelName(value) {
  return String(value || "unknown-model").split("/").pop().replace(/\.gguf$/i, "");
}

function runSuiteLabel(run) {
  if (!run.suite_id) return "";
  const position = finite(run.model_index) && finite(run.model_count) ? ` · ${run.model_index}/${run.model_count}` : "";
  return `${String(run.suite_id).slice(-10)}${position}`;
}

function sweepIsCumulative(scenarios) {
  if (!scenarios?.length) return false;
  if (scenarios.some(item => String(item.scenario_id || "").startsWith("nodes-") || String(item.label || "").startsWith("누적"))) return true;
  if (scenarios.some(item => String(item.scenario_id || "").startsWith("node-") || String(item.label || "").startsWith("개별"))) return false;
  const counts = scenarios.map(item => (item.nodes || []).length);
  return counts.some((count, index) => index > 0 && count > counts[index - 1]) && counts.every((count, index) => index === 0 || count >= counts[index - 1]);
}

function runDisplayThroughput(run) {
  if (runStrategy(run) !== "node_sweep") return run.cluster_tokens_per_s;
  const scenarios = Array.isArray(run.scenario_summaries) ? run.scenario_summaries.filter(item => finite(item.cluster_tokens_per_s)) : [];
  if (!scenarios.length) return null;
  const cumulative = sweepIsCumulative(scenarios);
  return cumulative ? scenarios.at(-1).cluster_tokens_per_s : Math.max(...scenarios.map(item => Number(item.cluster_tokens_per_s)));
}

function runThroughputCell(run) {
  const value = runDisplayThroughput(run);
  if (!finite(value)) return "—";
  if (runStrategy(run) === "broadcast_compare") return `${fmt(value)} replica tok/s<br><small>복제본 합산</small>`;
  if (runStrategy(run) === "node_sweep") return `${fmt(value)} tok/s<br><small>대표 시나리오</small>`;
  return `${fmt(value)} tok/s`;
}

function strategyMeta(value) {
  return STRATEGIES[value] || { label: value || "알 수 없음", short: String(value || "UNKNOWN").toUpperCase() };
}

function parseRpcTensorSplit(strict = false) {
  const raw = $("#rpcTensorSplitInput").value.trim();
  if (!raw) {
    if (strict && $("#rpcSplitPolicySelect").value === "custom") throw new Error("직접 분할을 선택했다면 노드별 비율을 입력하세요.");
    return [];
  }
  let values;
  try {
    values = raw.startsWith("[") ? JSON.parse(raw) : raw.split(",").map(value => Number(value.trim()));
  } catch (_error) {
    if (strict) throw new Error("노드별 비율은 1, 1, 2 같은 숫자 목록이어야 합니다.");
    return [];
  }
  const valid = Array.isArray(values) && values.length && values.every(value => Number.isFinite(Number(value)) && Number(value) > 0);
  if (!valid) {
    if (strict) throw new Error("노드별 분할 비율에는 0보다 큰 숫자만 사용할 수 있습니다.");
    return [];
  }
  const normalized = values.map(Number);
  if (strict && normalized.length !== state.selectedNodes.size) throw new Error(`선택 노드 ${state.selectedNodes.size}대와 같은 개수의 분할 비율이 필요합니다.`);
  return normalized;
}

function formatUptime(seconds) {
  if (!finite(seconds)) return "—";
  const total = Number(seconds);
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return `${days ? `${days}일 ` : ""}${hours}시간 ${minutes}분`;
}

function statusFor(nodeName) {
  return state.status.find(item => item.name === nodeName) || {};
}

function environmentFor(nodeName) {
  return state.environment.find(item => item.node === nodeName || item.name === nodeName) || {
    node: nodeName,
    status: "unknown",
    checks: [],
    missing_system_packages: [],
    manual_commands: [],
  };
}

const READINESS = {
  ready: { label: "READY", detail: "LLM 런타임 구성 정상" },
  repairable: { label: "AUTO FIX", detail: "자동 구성 가능" },
  manual: { label: "MANUAL", detail: "수동 작업 필요" },
  blocked: { label: "BLOCKED", detail: "환경 구성 차단" },
  checking: { label: "CHECKING", detail: "환경 확인 중" },
  unknown: { label: "UNCHECKED", detail: "점검하지 않음" },
};

function readinessMeta(status) {
  const aliases = { needs_setup: "repairable", unavailable: "blocked", failed: "blocked", not_checked: "unknown" };
  const source = String(status || "unknown").toLowerCase();
  const normalized = aliases[source] || source;
  return { status: READINESS[normalized] ? normalized : "unknown", ...(READINESS[normalized] || READINESS.unknown) };
}

function checkMeta(status) {
  const normalized = String(status || "unknown").toLowerCase();
  if (["ok", "pass", "passed", "ready", "installed"].includes(normalized)) return { status: "pass", icon: "✓", label: "정상" };
  if (["warning", "warn", "repairable", "missing"].includes(normalized)) return { status: "warning", icon: "!", label: "구성 필요" };
  if (["manual"].includes(normalized)) return { status: "manual", icon: "↗", label: "수동 작업" };
  if (["failed", "fail", "error", "blocked"].includes(normalized)) return { status: "failed", icon: "×", label: "실패" };
  if (["checking", "running", "pending"].includes(normalized)) return { status: "checking", icon: "…", label: "확인 중" };
  return { status: "unknown", icon: "·", label: "미확인" };
}

function checkedAtLabel(value) {
  if (!value) return "아직 점검하지 않음";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : `${date.toLocaleDateString("ko-KR")} ${date.toLocaleTimeString("ko-KR")}`;
}

function normalizedEnvironmentItems(payload) {
  if (Array.isArray(payload)) return payload;
  if (payload?.report) return [payload.report];
  if (payload?.node && payload?.status) return [payload];
  return payload?.environment || payload?.node_readiness || payload?.reports || [];
}

function setEnvironmentReports(payload) {
  const reports = normalizedEnvironmentItems(payload);
  if (!Array.isArray(reports)) return;
  const normalized = reports.map(report => ({
    ...report,
    node: report.node || report.name || "",
    status: readinessMeta(report.status).status,
    checks: Array.isArray(report.checks) ? report.checks : [],
    missing_system_packages: Array.isArray(report.missing_system_packages) ? report.missing_system_packages : [],
    manual_commands: Array.isArray(report.manual_commands) ? report.manual_commands : [],
  })).filter(report => report.node);
  const partial = Boolean(payload?.report || (payload?.node && payload?.status));
  if (partial) {
    const merged = new Map(state.environment.map(report => [report.node, report]));
    normalized.forEach(report => merged.set(report.node, report));
    state.environment = [...merged.values()];
  } else {
    state.environment = normalized;
  }
  renderNodes();
  renderNodeDetail();
}

function actualPlatform(node) {
  return statusFor(node.name).profile?.platform_kind || statusFor(node.name).node_info?.platform_kind || node.platform || "auto";
}

function runExperimentId(run) {
  if (run.experiment_id) return run.experiment_id;
  const slug = String(run.name || "unnamed").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "experiment";
  return `legacy-${slug}`;
}

function ingestStatus(items) {
  items.forEach(item => {
    const metrics = item.metrics || {};
    if (!metrics.sampled_at) return;
    const history = state.metricHistory.get(item.name) || [];
    if (history.at(-1)?.sampled_at === metrics.sampled_at) return;
    history.push({
      sampled_at: metrics.sampled_at,
      cpu: finite(metrics.cpu_pct) ? Number(metrics.cpu_pct) : null,
      gpu: finite(metrics.gpu_pct) ? Number(metrics.gpu_pct) : null,
      ram: finite(metrics.ram_pct) ? Number(metrics.ram_pct) : null,
      power: finite(metrics.power_w) ? Number(metrics.power_w) : null,
      temperature: finite(metrics.gpu_temp_c) ? Number(metrics.gpu_temp_c) : finite(metrics.cpu_temp_c) ? Number(metrics.cpu_temp_c) : null,
    });
    if (history.length > 120) history.splice(0, history.length - 120);
    state.metricHistory.set(item.name, history);
  });
}

function initializeSelection() {
  if (state.selectedNodes.size) return;
  state.nodes.filter(node => node.enabled).slice(0, 4).forEach(node => state.selectedNodes.add(node.name));
}

function renderNodes() {
  initializeSelection();
  const grid = $("#nodeGrid");
  if (!state.nodes.length) {
    grid.innerHTML = `<div class="empty-result"><strong>등록된 노드가 없습니다.</strong></div>`;
    return;
  }
  grid.innerHTML = state.nodes.map(node => {
    const live = statusFor(node.name);
    const online = Boolean(live.api);
    const metrics = live.metrics || {};
    const model = live.current?.model_id || "모델 로드 안 됨";
    const selected = state.selectedNodes.has(node.name);
    const kind = actualPlatform(node);
    const roleLabel = node.role === "head" ? "HEAD · CONTROL + INFERENCE" : `WORKER · ${platformName(kind).toUpperCase()}`;
    const error = live.error && live.error !== "disabled" ? live.error : "";
    const environment = environmentFor(node.name);
    const readiness = readinessMeta(environment.status);
    return `
      <article class="node-card ${selected ? "selected" : ""} ${node.enabled ? "" : "disabled"}" data-node-card="${escapeHtml(node.name)}">
        <div class="node-card-head">
          <label class="node-select" title="실험 참여 여부">
            <input type="checkbox" data-node-select="${escapeHtml(node.name)}" ${selected ? "checked" : ""} ${node.enabled ? "" : "disabled"}>
            <span></span>
          </label>
          <div class="node-title"><strong>${escapeHtml(node.name)}</strong><small>${escapeHtml(roleLabel)}<br>${escapeHtml(node.host)}:${node.api_port}</small></div>
          <div class="node-status-stack">
            <span class="status-pill ${online ? "online" : ""}"><i></i>${online ? "ONLINE" : node.enabled ? "OFFLINE" : "DISABLED"}</span>
            <span class="readiness-pill ${readiness.status}" title="${escapeHtml(readiness.detail)}"><i></i>${readiness.label}</span>
          </div>
        </div>
        <div class="node-model"><span>ACTIVE MODEL · ${live.model_count || 0} AVAILABLE</span><strong title="${escapeHtml(model)}">${escapeHtml(model)}</strong></div>
        <div class="node-metrics">
          <div><small>CPU</small><strong>${pct(metrics.cpu_pct)}</strong></div>
          <div><small>${kind === "jetson" ? "GPU" : "RAM"}</small><strong>${kind === "jetson" ? pct(metrics.gpu_pct) : pct(metrics.ram_pct)}</strong></div>
          <div><small>POWER</small><strong>${finite(metrics.power_w) ? `${fmt(metrics.power_w)}W` : "N/A"}</strong></div>
          <div><small>TEMP</small><strong>${finite(metrics.gpu_temp_c ?? metrics.cpu_temp_c) ? `${fmt(metrics.gpu_temp_c ?? metrics.cpu_temp_c, 0)}°` : "—"}</strong></div>
        </div>
        <button class="node-detail-button" type="button" data-node-detail="${escapeHtml(node.name)}">상세 상태</button>
        ${error ? `<span class="node-card-menu" title="${escapeHtml(error)}">!</span>` : ""}
      </article>`;
  }).join("");

  $$('[data-node-select]').forEach(input => input.addEventListener("change", event => {
    const name = event.currentTarget.dataset.nodeSelect;
    if (event.currentTarget.checked) {
      if (state.selectedNodes.size >= 4) {
        event.currentTarget.checked = false;
        return toast("최대 4대", "한 실험에는 최대 네 대까지 참여할 수 있습니다.", "error");
      }
      state.selectedNodes.add(name);
    } else {
      state.selectedNodes.delete(name);
    }
    if (!state.selectedNodes.size) {
      event.currentTarget.checked = true;
      state.selectedNodes.add(name);
      toast("노드 선택 필요", "실험에는 최소 한 대가 필요합니다.", "error");
    }
    renderNodes();
  }));
  $$('[data-node-detail]').forEach(button => button.addEventListener("click", () => openNodeDetail(button.dataset.nodeDetail)));
  updateSummary();
}

function renderEnvironmentSummary() {
  const selected = [...state.selectedNodes];
  const reports = selected.map(environmentFor);
  const counts = reports.reduce((acc, report) => {
    const status = readinessMeta(report.status).status;
    acc[status] = (acc[status] || 0) + 1;
    return acc;
  }, {});
  const ready = counts.ready || 0;
  const actionNeeded = reports.length - ready;
  const summary = $("#environmentSummary");
  if (summary) {
    summary.innerHTML = `<span><strong>${ready}</strong> READY</span><span><strong>${actionNeeded}</strong> ACTION NEEDED</span>`;
  }
  const hasSelection = selected.length > 0;
  ["#checkEnvironmentButton", "#installEnvironmentButton", "#environmentCheckAllButton", "#environmentInstallAllButton"].forEach(selector => {
    const button = $(selector);
    if (button) button.disabled = !hasSelection || state.environmentBusy;
  });
}

function updateSummary() {
  const enabled = state.nodes.filter(node => node.enabled);
  const online = enabled.filter(node => statusFor(node.name).api);
  const selected = [...state.selectedNodes];
  const powers = online.map(node => Number(statusFor(node.name).metrics?.power_w)).filter(Number.isFinite);
  $("#onlineCount").textContent = online.length;
  $("#enabledCount").textContent = enabled.length;
  $("#selectedCount").textContent = selected.length;
  $("#runNodes").textContent = selected.length;
  $("#selectionSummary").textContent = `선택 노드 ${selected.length}대 · ${selected.join(", ")}`;
  $("#modelCount").textContent = state.models.length || "—";
  $("#averagePower").textContent = powers.length ? fmt(powers.reduce((a, b) => a + b, 0)) : "—";
  const head = enabled.find(node => node.role === "head");
  $("#headStatus").textContent = head && statusFor(head.name).api ? "ONLINE" : "OFFLINE";
  $$('.satellite').forEach((element, index) => {
    const worker = enabled.filter(node => node.role === "worker")[index];
    element.classList.toggle("online", Boolean(worker && statusFor(worker.name).api));
  });
  const latest = state.runs.find(run => run.status === "completed");
  $("#recentThroughput").textContent = latest ? fmt(latest.cluster_tokens_per_s) : "—";
  updateStrategyGuidance();
  updatePlatformGuidance();
  updateModelAvailability();
  renderEnvironmentSummary();
}

function updatePlatformGuidance() {
  const selectedKinds = plannedNodeNames().map(name => {
    const node = state.nodes.find(item => item.name === name);
    return node ? actualPlatform(node) : "auto";
  });
  const hasPi = selectedKinds.includes("raspberry-pi");
  const layers = $("#layersInput");
  const rpc = selectedStrategy() === "model_parallel_rpc";
  layers.max = hasPi && !rpc ? "0" : "120";
  if (hasPi && !rpc && Number(layers.value) !== 0) layers.value = "0";
  layers.disabled = rpc;
  layers.title = rpc ? "RPC 모델 분할은 coordinator와 원격 장치의 전체 가속 가능 레이어를 사용합니다." : "";
  $("#uniformInput").disabled = rpc;
  $("#configValidity").textContent = rpc
    ? ($("#rpcAcknowledgeInput").checked ? "RPC 실험 준비됨" : "RPC 위험 확인 필요")
    : hasPi ? "Pi 포함 · CPU 모드" : "설정 준비됨";
}

function updateStrategyGuidance() {
  const strategy = selectedStrategy();
  const meta = strategyMeta(strategy);
  const nodeCount = Math.max(1, state.selectedNodes.size);
  const modelCount = Math.max(1, selectedModelIds().length);
  const requests = Math.max(0, Number($("#requestsInput").value) || 0);
  const sweepMode = $("#sweepModeSelect").value;
  const splitPolicy = $("#rpcSplitPolicySelect").value;
  $("#sweepOptions").hidden = strategy !== "node_sweep";
  $("#rpcOptions").hidden = strategy !== "model_parallel_rpc";
  $("#rpcTensorField").hidden = splitPolicy !== "custom";
  $$(".strategy-card").forEach(card => card.classList.toggle("selected", card.querySelector("input")?.checked));
  $("#strategyFormulaTitle").textContent = meta.label;
  $("#runStrategyBadge").textContent = meta.label;
  $("#runNodes").textContent = strategy === "single_node" ? 1 : nodeCount;

  let modelCopies = nodeCount;
  let nodesPerAnswer = 1;
  let logicalRequests = requests;
  let physicalCalls = requests;
  let explanation = "모델을 노드마다 복제하고 사용자 요청은 한 노드씩 분배합니다.";
  if (strategy === "single_node") {
    modelCopies = 1;
    explanation = nodeCount === 1
      ? `선택한 노드(${[...state.selectedNodes][0] || "미선택"}) 하나에서 기준 성능을 측정합니다.`
      : "정확한 기준선을 위해 노드 한 대만 선택해야 합니다.";
  } else if (strategy === "broadcast_compare") {
    physicalCalls = requests * nodeCount;
    explanation = `논리 요청 ${requests}개를 ${nodeCount}대 모두에 보내 서로 독립된 답변 ${physicalCalls}개를 비교합니다.`;
  } else if (strategy === "node_sweep") {
    physicalCalls = requests * nodeCount;
    logicalRequests = physicalCalls;
    explanation = sweepMode === "cumulative"
      ? `1대부터 ${nodeCount}대까지 ${nodeCount}개 단계에서 요청 ${requests}개씩 실행합니다. 각 단계 안에서는 선택된 노드가 요청을 나눕니다.`
      : `${nodeCount}대를 하나씩 분리해 각 노드에 요청 ${requests}개를 실행합니다.`;
  } else if (strategy === "model_parallel_rpc") {
    modelCopies = "1 · 분할";
    nodesPerAnswer = nodeCount;
    explanation = `모델 하나를 ${nodeCount}대에 나누고 한 답변 계산에 모두 참여시킵니다. 물리 호출 수에는 내부 텐서 RPC 통신을 포함하지 않습니다.`;
  }
  $("#formulaModelCopies").textContent = modelCopies;
  $("#formulaModels").textContent = modelCount;
  $("#formulaNodesPerAnswer").textContent = nodesPerAnswer;
  $("#formulaLogicalRequests").textContent = logicalRequests;
  $("#formulaPhysicalCalls").textContent = physicalCalls;
  $("#formulaTotalCalls").textContent = physicalCalls * modelCount;
  $("#strategyFormulaExplanation").textContent = explanation;
}

function selectedModelIds() {
  return state.selectedModels.filter(id => state.models.some(model => model.id === id));
}

function syncLegacyModelSelect() {
  const select = $("#modelSelect");
  const ids = selectedModelIds();
  select.innerHTML = ids.map(id => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`).join("");
  if (ids.length) select.value = ids[0];
}

function setSelectedModels(ids) {
  state.selectedModels = [...new Set(ids)].filter(id => state.models.some(model => model.id === id));
  syncLegacyModelSelect();
  renderModelPicker();
  updateModelAvailability();
  updateStrategyGuidance();
}

function renderModelPicker() {
  const query = ($("#modelSearchInput")?.value || "").trim().toLowerCase();
  const ids = selectedModelIds();
  const visible = state.models.filter(model => !query || `${model.id} ${model.size_gb}`.toLowerCase().includes(query));
  $("#modelSelectionCount").textContent = `${ids.length}개 선택`;
  $("#modelChips").innerHTML = ids.map(id => `<span class="model-chip"><span title="${escapeHtml(id)}">${escapeHtml(id)}</span><button type="button" data-remove-model="${escapeHtml(id)}" aria-label="${escapeHtml(id)} 선택 해제">×</button></span>`).join("");
  $("#modelChecklist").innerHTML = visible.length ? visible.map(model => `<label class="model-option"><input type="checkbox" data-model-option="${escapeHtml(model.id)}" ${ids.includes(model.id) ? "checked" : ""}><span title="${escapeHtml(model.id)}">${escapeHtml(model.id)}</span><small>${fmt(model.size_gb, 2)} GB</small></label>`).join("") : `<div class="model-empty">검색 결과가 없습니다.</div>`;
}

function updateModelAvailability() {
  const modelIds = selectedModelIds();
  if (!modelIds.length) {
    $("#modelHint").textContent = "벤치마크할 모델을 한 개 이상 선택하세요.";
    $("#modelHint").style.color = "var(--orange)";
    return;
  }
  const plannedNames = plannedNodeNames();
  const placementNames = selectedStrategy() === "model_parallel_rpc"
    ? plannedNames.filter(name => state.nodes.find(node => node.name === name)?.role === "head")
    : plannedNames;
  const missing = placementNames.filter(name => {
    const live = statusFor(name);
    return live.api && Array.isArray(live.model_ids) && modelIds.some(id => !live.model_ids.includes(id));
  });
  const selectedKinds = plannedNames.map(name => {
    const node = state.nodes.find(item => item.name === name);
    return node ? actualPlatform(node) : "auto";
  });
  const hint = $("#modelHint");
  if (missing.length) {
    hint.textContent = `${missing.join(", ")}에 선택 모델 일부가 없습니다. 실행 전에 모델 동기화가 필요합니다.`;
    hint.style.color = "var(--orange)";
  } else {
    const totalSize = modelIds.reduce((sum, id) => sum + Number(state.models.find(item => item.id === id)?.size_gb || 0), 0);
    const piNote = selectedKinds.includes("raspberry-pi")
      ? selectedStrategy() === "model_parallel_rpc" ? " · Pi는 RPC CPU 장치로 참여" : " · Pi CPU/OpenBLAS · GPU 레이어 0"
      : "";
    const placementNote = selectedStrategy() === "model_parallel_rpc"
      ? " · GGUF는 head에만 필요, worker는 텐서 수신"
      : " · 선택 노드 모델 상태 정상";
    hint.textContent = `${modelIds.length}개 · 합계 ${fmt(totalSize, 2)} GB${placementNote}${piNote}`;
    hint.style.color = "";
  }
}

function renderModels(defaults = {}) {
  const requested = Array.isArray(defaults.model_ids) && defaults.model_ids.length
    ? defaults.model_ids
    : state.selectedModels.length ? state.selectedModels : [defaults.model_id || state.models[0]?.id].filter(Boolean);
  state.selectedModels = [...new Set(requested)].filter(id => state.models.some(model => model.id === id));
  syncLegacyModelSelect();
  renderModelPicker();
  updateModelAvailability();
}

function renderSettings() {
  const workerEnabled = Boolean(state.settings.worker_api_auth);
  const dashboardEnabled = Boolean(state.settings.dashboard_token_auth);
  $("#workerAuthInput").checked = workerEnabled;
  $("#dashboardAuthInput").checked = dashboardEnabled;
  $("#dashboardTokenInput").value = "";
  const workerNotice = $("#workerAuthNotice");
  workerNotice.classList.toggle("enabled", workerEnabled);
  workerNotice.textContent = workerEnabled
    ? "현재 켜짐 · worker API 토큰 인증 모드"
    : "현재 꺼짐 · 신뢰 LAN 전용 모드";
  updateDashboardAuthGuidance();
}

function updateDashboardAuthGuidance() {
  const current = Boolean(state.settings.dashboard_token_auth);
  const desired = $("#dashboardAuthInput").checked;
  const enabling = desired && !current;
  const field = $("#dashboardTokenField");
  field.hidden = !enabling;
  $("#dashboardTokenInput").required = enabling;
  const notice = $("#dashboardAuthNotice");
  notice.classList.toggle("enabled", desired);
  notice.textContent = current === desired
    ? desired ? "현재 켜짐 · 접속 시 대시보드 토큰 필요" : "현재 꺼짐 · 토큰 없이 대시보드 접속"
    : desired ? "저장 후 켜짐 · 현재 대시보드 토큰 확인 필요" : "저장 후 꺼짐 · 토큰 없이 접속";
}

function applyConfig(defaults, includeName = true) {
  const mapping = {
    requests: "#requestsInput", concurrency: "#concurrencyInput", max_tokens: "#maxTokensInput",
    n_ctx: "#contextInput", n_gpu_layers: "#layersInput", warmup_requests: "#warmupInput",
    temperature: "#temperatureInput", top_p: "#topPInput", seed: "#seedInput", prompt: "#promptInput",
  };
  if (includeName && defaults.name !== undefined) $("#experimentName").value = defaults.name;
  Object.entries(mapping).forEach(([key, selector]) => { if (defaults[key] !== undefined) $(selector).value = defaults[key]; });
  if (defaults.require_uniform_config !== undefined) $("#uniformInput").checked = defaults.require_uniform_config !== false;
  const strategy = defaults.execution_strategy || defaults.strategy;
  const strategyInput = strategy ? $$('input[name="execution_strategy"]').find(input => input.value === strategy) : null;
  if (strategyInput) strategyInput.checked = true;
  if (defaults.sweep_mode !== undefined) $("#sweepModeSelect").value = defaults.sweep_mode;
  if (defaults.rpc_split_mode !== undefined) $("#rpcSplitModeSelect").value = defaults.rpc_split_mode;
  if (defaults.rpc_split_policy !== undefined) $("#rpcSplitPolicySelect").value = defaults.rpc_split_policy;
  if (Array.isArray(defaults.rpc_tensor_split)) $("#rpcTensorSplitInput").value = defaults.rpc_tensor_split.join(", ");
  if (defaults.acknowledge_experimental_rpc !== undefined) $("#rpcAcknowledgeInput").checked = Boolean(defaults.acknowledge_experimental_rpc);
  const configModels = Array.isArray(defaults.model_ids) && defaults.model_ids.length ? defaults.model_ids : [defaults.model_id].filter(Boolean);
  if (configModels.length) state.selectedModels = configModels.filter(id => state.models.some(model => model.id === id));
  if (defaults.model_cooldown_s !== undefined) $("#modelCooldownInput").value = defaults.model_cooldown_s;
  if (defaults.continue_on_model_error !== undefined) $("#continueModelErrorInput").checked = defaults.continue_on_model_error !== false;
  syncLegacyModelSelect(); renderModelPicker();
  if (Array.isArray(defaults.node_names)) {
    const available = defaults.node_names.filter(name => state.nodes.some(node => node.name === name && node.enabled));
    if (available.length) state.selectedNodes = new Set(available.slice(0, 4));
  }
  updateFormMirrors();
  renderNodes();
}

function applyDefaults(defaults) {
  $("#experimentName").value = defaults.name || "cluster-load-test";
  renderModels(defaults);
  applyConfig({ ...defaults, require_uniform_config: defaults.require_uniform_config !== false });
}

function updateFormMirrors() {
  $("#temperatureValue").textContent = Number($("#temperatureInput").value).toFixed(1);
  $("#topPValue").textContent = Number($("#topPInput").value).toFixed(2).replace(/0$/, "");
  $("#promptLength").textContent = $("#promptInput").value.length;
  $("#runRequests").textContent = $("#requestsInput").value;
  $("#runConcurrency").textContent = $("#concurrencyInput").value;
  updateStrategyGuidance();
  updatePlatformGuidance();
  updateModelAvailability();
}

function renderExperimentGroups() {
  const formSelect = $("#experimentGroupSelect");
  const resultSelect = $("#resultExperimentFilter");
  const currentForm = formSelect.value;
  const currentResult = resultSelect.value || "all";
  const options = state.experimentGroups.map(group => {
    const strategy = group.default_config?.execution_strategy;
    const strategySuffix = strategy ? ` · ${strategyMeta(strategy).label}` : "";
    return `<option value="${escapeHtml(group.experiment_id)}">${escapeHtml(group.name)} · ${group.run_count || 0}회${escapeHtml(strategySuffix)}${group.legacy ? " · 이전 기록" : ""}</option>`;
  }).join("");
  formSelect.innerHTML = `<option value="">+ 새 실험 만들기</option>${options}`;
  resultSelect.innerHTML = `<option value="all">전체 실험</option>${options}`;
  if ([...formSelect.options].some(option => option.value === currentForm)) formSelect.value = currentForm;
  if ([...resultSelect.options].some(option => option.value === currentResult)) resultSelect.value = currentResult;
}

function filteredRuns() {
  const filter = $("#resultExperimentFilter").value;
  if (filter === "all") return state.runs;
  const group = state.experimentGroups.find(item => item.experiment_id === filter);
  return group?.runs || state.runs.filter(run => runExperimentId(run) === filter);
}

function filteredSuites() {
  const filter = $("#resultExperimentFilter").value;
  return state.suites.filter(suite => filter === "all" || suite.experiment_id === filter);
}

function artifactTimestamp(item) {
  const value = item?.finished_at || item?.updated_at || item?.started_at || item?.run_id || "";
  const parsed = Date.parse(value);
  if (Number.isFinite(parsed)) return parsed;
  const compact = String(value).match(/(20\d{2})(\d{2})(\d{2})[_T-]?(\d{2})(\d{2})(\d{2})/);
  return compact ? Date.UTC(...compact.slice(1).map(Number).map((part, index) => index === 1 ? part - 1 : part)) : 0;
}

function latestResultArtifact(runs, suites) {
  const newestRun = runs[0];
  const newestSuite = suites[0];
  if (!newestSuite) return { run: newestRun, suite: null };
  if (!newestRun) return { run: null, suite: newestSuite };
  return artifactTimestamp(newestSuite) >= artifactTimestamp(newestRun)
    ? { run: runs.find(run => run.suite_id === newestSuite.suite_id) || null, suite: newestSuite }
    : { run: newestRun, suite: null };
}

function suiteOutcomeRuns(suite, actualRuns) {
  const persisted = Array.isArray(suite?.models) ? suite.models : [];
  if (!persisted.length) return actualRuns;
  return persisted.map(record => {
    const actual = actualRuns.find(run => Number(run.model_index) === Number(record.model_index))
      || actualRuns.find(run => runModelId(run) === record.model_id);
    const errors = Array.isArray(record.errors) ? record.errors : [];
    if (actual) return {
      ...actual,
      suite_status: suite.status,
      cleanup_status: record.cleanup_status,
      suite_model_errors: errors,
    };
    return {
      suite_id: suite.suite_id,
      suite_status: suite.status,
      experiment_id: suite.experiment_id,
      name: suite.name,
      model_id: record.model_id,
      model_index: record.model_index,
      model_count: persisted.length,
      status: record.status || "unrun",
      cleanup_status: record.cleanup_status,
      error: errors.map(item => item.error).filter(Boolean).join("; "),
      suite_model_placeholder: true,
    };
  });
}

function omittedModelLabel(run) {
  const status = String(run.status || "unknown").toUpperCase();
  const cleanup = run.cleanup_status === "failed" ? " · unload 실패" : "";
  const reason = run.error ? ` · ${ellipsis(run.error, 36)}` : "";
  return `${shortModelName(runModelId(run))} (${status}${cleanup}${reason})`;
}

function renderRuns() {
  const runs = filteredRuns();
  const suites = filteredSuites();
  const filter = $("#resultExperimentFilter").value;
  const group = state.experimentGroups.find(item => item.experiment_id === filter);
  const completedStrategies = new Set(runs.filter(run => run.status === "completed" && finite(run.cluster_tokens_per_s)).map(runStrategy));
  const mixedAllStrategies = completedStrategies.size > 1;
  const baseContext = group
    ? `${group.name} · ${group.run_count || 0}회 실행 · ${strategyMeta(group.default_config?.execution_strategy || runStrategy(runs[0] || {})).label} · experiment_id ${group.experiment_id}`
    : `모든 실험 ${state.experimentGroups.length}개 · 실행 ${state.runs.length}회`;
  $("#experimentContext").textContent = mixedAllStrategies
    ? `${baseContext} · 서로 다른 실행 방식이 섞여 있어 그래프를 숨겼습니다. 방식별 실험 묶음을 따로 만들어 비교하세요.`
    : baseContext;
  const table = $("#runsTable");
  if (!runs.length && !suites.length) {
    table.innerHTML = `<tr><td colspan="9" class="empty-cell">이 실험의 실행 기록 없음</td></tr>`;
    $("#resultHighlight").innerHTML = `<div class="empty-result"><strong>연결된 벤치마크 결과가 없습니다.</strong><span>이 실험을 실행하면 결과가 같은 experiment_id에 누적됩니다.</span></div>`;
    $("#chartGrid").hidden = true;
    updateSummary();
    return;
  }
  table.innerHTML = runs.length ? runs.slice(0, 30).map(run => `
    <tr data-run-experiment="${escapeHtml(runExperimentId(run))}">
      <td><strong>${escapeHtml(run.name || run.run_id)}</strong><br><small>${escapeHtml(run.run_id || "")}</small></td>
      <td class="model-cell"><strong title="${escapeHtml(runModelId(run))}">${escapeHtml(shortModelName(runModelId(run)))}</strong>${run.suite_id ? `<span class="suite-badge">SUITE ${escapeHtml(runSuiteLabel(run))}</span>` : ""}</td>
      <td><span class="strategy-badge ${strategyMeta(runStrategy(run)).experimental ? "experimental" : ""}">${escapeHtml(strategyMeta(runStrategy(run)).label)}</span></td>
      <td>${Array.isArray(run.nodes) ? run.nodes.length : "—"}</td>
      <td title="${runStrategy(run) === "broadcast_compare" ? "모든 복제본이 성공한 논리 요청 비율" : "성공한 실제 호출 비율"}">${runStrategy(run) === "broadcast_compare" && run.all_replicas_success_rate !== undefined ? `${pct(Number(run.all_replicas_success_rate) * 100)}<br><small>all replicas</small>` : run.success_rate !== undefined ? pct(Number(run.success_rate) * 100) : "—"}</td>
      <td>${run.ttft_p50_s != null ? `${fmt(run.ttft_p50_s, 2)}s` : "—"}</td>
      <td>${run.e2e_p95_s != null ? `${fmt(run.e2e_p95_s, 2)}s` : "—"}</td>
      <td>${runThroughputCell(run)}</td>
      <td><span class="run-status ${run.status === "failed" || ["failed", "partial", "cancelled"].includes(run.suite_status) ? "failed" : ""}">${escapeHtml((run.status || "unknown").toUpperCase())}</span>${run.suite_status && run.suite_status !== "completed" ? `<br><small>SUITE ${escapeHtml(run.suite_status.toUpperCase())}</small>` : ""}</td>
    </tr>`).join("") : `<tr><td colspan="9" class="empty-cell">최근 suite는 모델 실행 전 종료되었습니다.</td></tr>`;
  const latestArtifact = latestResultArtifact(runs, suites);
  const newestSuite = latestArtifact.suite;
  const newest = newestSuite
    ? runs.find(run => run.suite_id === newestSuite.suite_id) || { suite_id: newestSuite.suite_id, suite_status: newestSuite.status, experiment_id: newestSuite.experiment_id, name: newestSuite.name, status: newestSuite.status }
    : latestArtifact.run;
  const newestSuiteActualRuns = newest?.suite_id
    ? runs.filter(run => run.suite_id === newest.suite_id).sort((a, b) => Number(a.model_index || 0) - Number(b.model_index || 0))
    : [newest];
  const newestSuiteRuns = newestSuite ? suiteOutcomeRuns(newestSuite, newestSuiteActualRuns) : newestSuiteActualRuns;
  const latest = newestSuiteRuns.filter(run => run?.status === "completed" && finite(run.cluster_tokens_per_s)).at(-1);
  if (latest && latest.cluster_tokens_per_s != null) {
    const latestStrategy = strategyMeta(runStrategy(latest));
    const strategy = runStrategy(latest);
    let displayedThroughput = runDisplayThroughput(latest);
    let throughputTitle = strategy === "model_parallel_rpc" ? "SHARDED MODEL THROUGHPUT" : "CLUSTER THROUGHPUT";
    let throughputDetail = shortModelName(runModelId(latest));
    if (strategy === "broadcast_compare") {
      throughputTitle = "REPLICA TOKEN AGGREGATE";
      throughputDetail = "사용자 처리량 아님 · 복제본 합산 생성량";
    } else if (strategy === "node_sweep") {
      const scenarios = (latest.scenario_summaries || []).filter(item => finite(item.cluster_tokens_per_s));
      const chosen = scenarios.find(item => Number(item.cluster_tokens_per_s) === Number(displayedThroughput)) || scenarios.at(-1);
      throughputTitle = "SWEEP SCENARIO THROUGHPUT";
      throughputDetail = `${chosen?.label || chosen?.scenario_id || "측정 단계"} · 순차 전체 평균 아님`;
    }
    const successValue = strategy === "broadcast_compare" && latest.all_replicas_success_rate != null
      ? Number(latest.all_replicas_success_rate)
      : Number(latest.success_rate);
    const successDetail = strategy === "broadcast_compare" && latest.answer_agreement_rate != null
      ? `모든 복제본 성공 · 답변 일치 ${pct(Number(latest.answer_agreement_rate) * 100)}`
      : `${latest.successful || 0} / ${latest.requests || 0} physical calls`;
    $("#resultHighlight").innerHTML = `<div class="result-cards">
      <article class="result-card primary-result"><span>${throughputTitle}</span><strong>${fmt(displayedThroughput)}<small> tok/s</small></strong><small>${escapeHtml(throughputDetail)}</small><i class="result-strategy">${escapeHtml(latestStrategy.label)}</i></article>
      <article class="result-card"><span>TTFT · P50</span><strong>${fmt(latest.ttft_p50_s, 2)}s</strong><small>${strategy === "broadcast_compare" ? "복제본 응답별 첫 토큰 지연" : "첫 토큰 지연"}</small></article>
      <article class="result-card"><span>E2E · P95</span><strong>${fmt(latest.e2e_p95_s, 2)}s</strong><small>${strategy === "broadcast_compare" ? "복제본 응답별 완료 지연" : "요청 완료 지연"}</small></article>
      <article class="result-card"><span>${strategy === "broadcast_compare" ? "ALL-REPLICA SUCCESS" : "SUCCESS"}</span><strong>${pct(successValue * 100)}</strong><small>${successDetail}</small></article>
    </div>`;
  } else {
    const statuses = newestSuiteRuns.filter(Boolean).map(run => `${shortModelName(runModelId(run))}: ${(run.status || "unknown").toUpperCase()}`);
    $("#resultHighlight").innerHTML = `<div class="empty-result"><strong>최근 ${newest?.suite_id ? "모델 suite에 " : "실행에 "}완료된 측정값이 없습니다.</strong><span>${escapeHtml(statuses.join(" · ") || "실행 기록 없음")}${newest?.error ? ` · ${escapeHtml(newest.error)}` : ""}</span></div>`;
  }
  const newestMetricScope = newest?.suite_id ? newestSuiteRuns : [newest];
  const hasCompletedMetrics = newestMetricScope.some(run => run?.status === "completed" && finite(run.cluster_tokens_per_s));
  $("#chartGrid").hidden = !hasCompletedMetrics || mixedAllStrategies;
  if (hasCompletedMetrics && !mixedAllStrategies) requestAnimationFrame(() => drawResultCharts(runs));
  $$('[data-run-experiment]').forEach(row => row.addEventListener("click", () => {
    $("#resultExperimentFilter").value = row.dataset.runExperiment;
    renderRuns();
  }));
  updateSummary();
}

function setupCanvas(canvas, options = {}) {
  const ratio = options.ratio || window.devicePixelRatio || 1;
  const width = options.width || Math.max(canvas.clientWidth, 280);
  const height = options.height || Number(canvas.getAttribute("height")) || 260;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = options.cssWidth || "100%";
  canvas.style.height = options.cssHeight || `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width, height, ratio };
}

function drawEmptyChart(canvas, message) {
  const { context, width, height } = setupCanvas(canvas);
  context.fillStyle = "#f8f7f1";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#92968f";
  context.font = "11px ui-monospace, monospace";
  context.textAlign = "center";
  context.fillText(message, width / 2, height / 2);
  state.chartModels.delete(canvas.id);
  const card = canvas.closest(".chart-card");
  if (card) $$('[data-chart-png], [data-paper-export]', card).forEach(button => { button.disabled = true; });
  const legend = $(`#${canvas.id}Legend`);
  if (legend) legend.innerHTML = "";
}

function chartInteraction(chartId) {
  if (!state.chartInteraction.has(chartId)) state.chartInteraction.set(chartId, { hidden: new Set(), hits: [], focusIndex: -1 });
  return state.chartInteraction.get(chartId);
}

function chartValue(value, unit = "") {
  if (!finite(value)) return "—";
  const digits = Math.abs(Number(value)) < 10 ? 2 : 1;
  return `${Number(value).toFixed(digits)}${unit ? ` ${unit}` : ""}`;
}

function ellipsis(text, limit) {
  const value = String(text || "");
  return value.length > limit ? `${value.slice(0, Math.max(1, limit - 1))}…` : value;
}

function drawChartCanvas(canvas, model, options = {}) {
  const { context: ctx, width, height } = setupCanvas(canvas, options);
  const hidden = options.hidden || new Set();
  const activeSeries = model.series.filter(series => !hidden.has(series.label));
  const header = options.header || 0;
  const pad = { left: width < 430 ? 50 : 62, right: 18, top: 16 + header, bottom: 48 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const all = activeSeries.flatMap(series => series.values).filter(finite).map(Number);
  const maximum = Math.max(...all, 1) * 1.12;
  const hits = [];
  ctx.fillStyle = options.background || "#f8f7f1";
  ctx.fillRect(0, 0, width, height);
  if (header) {
    ctx.fillStyle = "#16251e"; ctx.fillRect(0, 0, width, header - 4);
    ctx.fillStyle = "#c7f25b"; ctx.font = `700 ${Math.max(16, width / 52)}px ${getComputedStyle(document.documentElement).getPropertyValue("--sans")}`;
    ctx.fillText(model.title, 28, 34);
    ctx.fillStyle = "rgba(255,255,255,.58)"; ctx.font = `500 ${Math.max(10, width / 105)}px ui-monospace, monospace`;
    ctx.fillText(ellipsis(model.subtitle || strategyMeta(model.strategy).label, 120), 28, 55);
    let legendX = 28; let legendY = 79;
    activeSeries.forEach(series => {
      const label = ellipsis(series.label, 22); const itemWidth = 32 + ctx.measureText(label).width;
      if (legendX + itemWidth > width - 24) { legendX = 28; legendY += 16; }
      ctx.fillStyle = series.color; ctx.fillRect(legendX, legendY - 7, 16, 3);
      ctx.fillStyle = "rgba(255,255,255,.76)"; ctx.fillText(label, legendX + 22, legendY);
      legendX += itemWidth + 16;
    });
  }
  ctx.strokeStyle = "#dedbd0"; ctx.fillStyle = "#747a72"; ctx.font = "9px ui-monospace, monospace"; ctx.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = pad.top + plotHeight * tick / 4;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(chartValue(maximum * (4 - tick) / 4), pad.left - 8, y + 3);
  }
  ctx.save(); ctx.translate(13, pad.top + plotHeight / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillStyle = "#5d645d"; ctx.font = "600 8px ui-monospace, monospace"; ctx.fillText(`${model.yLabel}${model.unit ? ` (${model.unit})` : ""}`, 0, 0); ctx.restore();
  if (!activeSeries.length || !all.length) {
    ctx.fillStyle = "#92968f"; ctx.textAlign = "center"; ctx.fillText("표시할 계열을 범례에서 선택하세요", pad.left + plotWidth / 2, pad.top + plotHeight / 2);
    return { hits, width, height };
  }
  const groupWidth = plotWidth / Math.max(1, model.labels.length);
  if (model.type === "line") {
    activeSeries.forEach(series => {
      ctx.strokeStyle = series.color; ctx.lineWidth = 2.5; ctx.lineJoin = "round";
      let drawing = false; ctx.beginPath();
      series.values.forEach((value, index) => {
        if (!finite(value)) { drawing = false; return; }
        const x = model.labels.length > 1 ? pad.left + plotWidth * index / (model.labels.length - 1) : pad.left + plotWidth / 2;
        const y = pad.top + plotHeight - Number(value) / maximum * plotHeight;
        if (!drawing) { ctx.moveTo(x, y); drawing = true; } else ctx.lineTo(x, y);
      });
      ctx.stroke();
      series.values.forEach((value, index) => {
        if (!finite(value)) return;
        const x = model.labels.length > 1 ? pad.left + plotWidth * index / (model.labels.length - 1) : pad.left + plotWidth / 2;
        const y = pad.top + plotHeight - Number(value) / maximum * plotHeight;
        ctx.fillStyle = series.color; ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
        hits.push({ x, y, radius: 13, label: model.labels[index], series: series.label, value: Number(value), unit: model.unit, detail: series.details?.[index], color: series.color });
      });
    });
  } else {
    const barWidth = Math.min(42, groupWidth * .72 / activeSeries.length);
    activeSeries.forEach((series, seriesIndex) => series.values.forEach((value, groupIndex) => {
      if (!finite(value)) return;
      const center = pad.left + groupWidth * groupIndex + groupWidth / 2 + (seriesIndex - (activeSeries.length - 1) / 2) * barWidth;
      const barHeight = Number(value) / maximum * plotHeight;
      const x = center - barWidth * .40; const y = pad.top + plotHeight - barHeight; const barRenderWidth = barWidth * .80;
      ctx.fillStyle = series.color; ctx.fillRect(x, y, barRenderWidth, barHeight);
      hits.push({ x: center, y, width: Math.max(barRenderWidth, 16), height: Math.max(barHeight, 12), label: model.labels[groupIndex], series: series.label, value: Number(value), unit: model.unit, detail: series.details?.[groupIndex], color: series.color });
    }));
  }
  const labelEvery = Math.max(1, Math.ceil(model.labels.length / Math.max(3, Math.floor(plotWidth / 75))));
  ctx.fillStyle = "#747a72"; ctx.font = "8px ui-monospace, monospace"; ctx.textAlign = "center";
  model.labels.forEach((label, index) => {
    if (index % labelEvery && index !== model.labels.length - 1) return;
    const x = model.type === "line" && model.labels.length > 1 ? pad.left + plotWidth * index / (model.labels.length - 1) : pad.left + groupWidth * index + groupWidth / 2;
    ctx.fillText(ellipsis(label, width < 500 ? 9 : 14), x, height - 25);
  });
  if (model.xLabel) { ctx.fillStyle = "#5d645d"; ctx.font = "600 8px ui-monospace, monospace"; ctx.fillText(model.xLabel, pad.left + plotWidth / 2, height - 6); }
  return { hits, width, height };
}

function renderChartLegend(canvas, model) {
  const target = $(`#${canvas.id}Legend`);
  if (!target) return;
  const interaction = chartInteraction(canvas.id);
  target.innerHTML = model.series.map(series => `<button type="button" style="--series-color:${series.color}" data-chart-series="${escapeHtml(series.label)}" aria-pressed="${interaction.hidden.has(series.label) ? "false" : "true"}" title="${escapeHtml(series.label)} 계열 표시 전환">${escapeHtml(series.label)}</button>`).join("");
}

function showChartTooltip(canvas, hit, keyboard = false) {
  const tooltip = $(`#${canvas.id}Tooltip`);
  if (!tooltip || !hit) return;
  const left = Math.max(70, Math.min(canvas.clientWidth - 70, hit.x));
  const top = Math.max(66, hit.y);
  tooltip.style.left = `${left}px`; tooltip.style.top = `${top}px`;
  tooltip.innerHTML = `<strong>${escapeHtml(hit.label)}</strong><span>${escapeHtml(hit.series)} · ${escapeHtml(chartValue(hit.value, hit.unit))}</span>${hit.detail ? `<span>${escapeHtml(hit.detail)}</span>` : ""}`;
  tooltip.hidden = false;
  if (keyboard) canvas.setAttribute("aria-label", `${hit.label}, ${hit.series}, ${chartValue(hit.value, hit.unit)}${hit.detail ? `, ${hit.detail}` : ""}. 화살표 키로 다음 값을 탐색하세요.`);
}

function hideChartTooltip(canvas) {
  const tooltip = $(`#${canvas.id}Tooltip`);
  if (tooltip) tooltip.hidden = true;
}

function bindChartInteraction(canvas) {
  if (canvas.dataset.interactiveBound) return;
  canvas.dataset.interactiveBound = "true";
  const findHit = event => {
    const interaction = chartInteraction(canvas.id);
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * ((interaction.width || rect.width) / rect.width);
    const y = (event.clientY - rect.top) * ((interaction.height || rect.height) / rect.height);
    return interaction.hits.reduce((best, hit) => {
      const distance = Math.hypot(hit.x - x, Math.max(0, hit.y - y));
      return !best || distance < best.distance ? { hit, distance } : best;
    }, null);
  };
  canvas.addEventListener("pointermove", event => { const match = findHit(event); if (match && match.distance < 45) showChartTooltip(canvas, match.hit); else hideChartTooltip(canvas); });
  canvas.addEventListener("pointerdown", event => { const match = findHit(event); if (match && match.distance < 55) showChartTooltip(canvas, match.hit); });
  canvas.addEventListener("pointerleave", () => hideChartTooltip(canvas));
  canvas.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "Escape"].includes(event.key)) return;
    const interaction = chartInteraction(canvas.id);
    if (event.key === "Escape") { hideChartTooltip(canvas); return; }
    if (!interaction.hits.length) return;
    event.preventDefault();
    if (event.key === "Home") interaction.focusIndex = 0;
    else if (event.key === "End") interaction.focusIndex = interaction.hits.length - 1;
    else interaction.focusIndex = (interaction.focusIndex + (["ArrowRight", "ArrowDown"].includes(event.key) ? 1 : -1) + interaction.hits.length) % interaction.hits.length;
    showChartTooltip(canvas, interaction.hits[interaction.focusIndex], true);
  });
}

function setChartModel(chartId, model) {
  const canvas = $(`#${chartId}`);
  if (!canvas || !model?.series?.some(series => series.values.some(finite))) return drawEmptyChart(canvas, model?.emptyMessage || "표시할 측정값 없음");
  model.series = model.series.map((series, index) => ({ ...series, color: series.color || DASHBOARD_COLORS[index % DASHBOARD_COLORS.length] }));
  state.chartModels.set(chartId, model);
  const interaction = chartInteraction(chartId);
  const rendered = drawChartCanvas(canvas, model, { hidden: interaction.hidden });
  interaction.hits = rendered.hits; interaction.width = rendered.width; interaction.height = rendered.height;
  renderChartLegend(canvas, model); bindChartInteraction(canvas);
  const card = canvas.closest(".chart-card");
  if (card) $$('[data-chart-png], [data-paper-export]', card).forEach(button => { button.disabled = false; });
}

function scenarioModelsForRuns(runs) {
  const modelNames = runs.map(run => shortModelName(runModelId(run)));
  const labels = [...new Set(runs.flatMap(run => (run.scenario_summaries || []).map(item => item.label || item.scenario_id)))];
  return {
    labels,
    series: runs.map((run, index) => ({
      label: modelNames[index], color: DASHBOARD_COLORS[index % DASHBOARD_COLORS.length],
      values: labels.map(label => (run.scenario_summaries || []).find(item => (item.label || item.scenario_id) === label)?.cluster_tokens_per_s),
      details: labels.map(label => { const item = (run.scenario_summaries || []).find(entry => (entry.label || entry.scenario_id) === label); return item ? `${(item.nodes || []).length} nodes${finite(item.speedup_vs_baseline) ? ` · ${fmt(item.speedup_vs_baseline, 2)}×` : ""}` : ""; }),
    })),
  };
}

function drawResultCharts(runs) {
  const latestArtifact = latestResultArtifact(runs, filteredSuites());
  const newestSuite = latestArtifact.suite;
  const newestRun = newestSuite
    ? runs.find(run => run.suite_id === newestSuite.suite_id) || { suite_id: newestSuite.suite_id, model_count: newestSuite.model_count }
    : latestArtifact.run;
  if (!newestRun) return;
  const latestSuiteActual = newestRun.suite_id
    ? runs.filter(run => run.suite_id === newestRun.suite_id).sort((a, b) => Number(a.model_index || 0) - Number(b.model_index || 0))
    : [newestRun];
  const latestSuiteAll = newestSuite ? suiteOutcomeRuns(newestSuite, latestSuiteActual) : latestSuiteActual;
  const latestSuiteCompleted = latestSuiteActual.filter(run => run.status === "completed" && finite(run.cluster_tokens_per_s));
  const allCompleted = runs.filter(run => run.status === "completed" && finite(run.cluster_tokens_per_s));
  if (!latestSuiteCompleted.length) return;
  const latestRun = latestSuiteCompleted.at(-1);
  const strategy = runStrategy(latestRun);
  const completed = allCompleted.slice(0, 18).reverse();
  const isSuiteComparison = Boolean(newestRun.suite_id) && Number(newestRun.model_count || latestSuiteAll.length) > 1;
  const comparisonRuns = isSuiteComparison ? latestSuiteCompleted : [];
  const omittedModels = isSuiteComparison ? latestSuiteAll.filter(run => run.status !== "completed" || !finite(run.cluster_tokens_per_s)) : [];
  const strategyLabel = strategyMeta(strategy).label;
  const cleanupWarnings = latestSuiteAll.filter(run => run.cleanup_status === "failed");
  const omittedSuffix = `${omittedModels.length ? ` · 미완료 ${omittedModels.length}개 제외: ${omittedModels.map(omittedModelLabel).join(", ")}` : ""}${cleanupWarnings.length ? ` · 정리 실패: ${cleanupWarnings.map(run => shortModelName(runModelId(run))).join(", ")}` : ""}`;
  const throughputMeaning = strategy === "broadcast_compare" ? "복제본 합산 생성량 (사용자 처리량 아님)" : strategy === "node_sweep" ? "스윕 시나리오 처리량" : strategy === "model_parallel_rpc" ? "분할 모델 공동 처리량" : "클러스터 처리량";

  if (strategy === "node_sweep") {
    const scenarioSource = comparisonRuns.length ? comparisonRuns : [latestRun];
    const scenarioData = scenarioModelsForRuns(scenarioSource);
    setChartModel("throughputChart", { type: "bar", title: comparisonRuns.length ? "모델별 노드 스케일링" : "노드 수 스윕 처리량", subtitle: `${strategyLabel} · 순차 전체 평균 제외${omittedSuffix}`, xLabel: "Sweep scenario", yLabel: "Throughput", unit: "tok/s", strategy, labels: scenarioData.labels, series: scenarioData.series, runs: scenarioSource });
    const latencyLabels = scenarioData.labels;
    const latestScenarios = latestRun.scenario_summaries || [];
    setChartModel("latencyChart", { type: "bar", title: "스윕 단계별 지연", subtitle: shortModelName(runModelId(latestRun)), xLabel: "Sweep scenario", yLabel: "Latency", unit: "s", strategy, labels: latencyLabels, series: [
      { label: "TTFT p50", values: latencyLabels.map(label => latestScenarios.find(item => (item.label || item.scenario_id) === label)?.ttft_p50_s) },
      { label: "E2E p95", values: latencyLabels.map(label => latestScenarios.find(item => (item.label || item.scenario_id) === label)?.e2e_p95_s) },
    ], runs: [latestRun] });
    const cumulative = sweepIsCumulative(latestScenarios);
    $("#nodeChartTitle").textContent = cumulative ? "스케일링 속도 향상" : "개별 노드 처리량";
    if (cumulative) setChartModel("nodeChart", { type: "line", title: "기준 대비 속도 향상", subtitle: "첫 단계 = 1.0× · 효율은 툴팁에 표시", xLabel: "Sweep scenario", yLabel: "Speedup", unit: "×", strategy, labels: latencyLabels, series: [{ label: "Speedup", values: latestScenarios.map(item => item.speedup_vs_baseline), details: latestScenarios.map(item => finite(item.scaling_efficiency) ? `scaling efficiency ${fmt(Number(item.scaling_efficiency) * 100, 1)}%` : "") }], runs: [latestRun] });
    else setChartModel("nodeChart", { type: "bar", title: "개별 노드 처리량", subtitle: "개별 모드는 scaling efficiency를 표시하지 않음", xLabel: "Node", yLabel: "Throughput", unit: "tok/s", strategy, labels: latencyLabels, series: [{ label: "Node throughput", values: latestScenarios.map(item => item.cluster_tokens_per_s) }], runs: [latestRun] });
    return;
  }

  if (comparisonRuns.length) {
    const labels = comparisonRuns.map(run => shortModelName(runModelId(run)));
    setChartModel("throughputChart", { type: "bar", title: `모델별 ${throughputMeaning}`, subtitle: `${strategyLabel} · suite ${latestRun.suite_id}${omittedSuffix}`, xLabel: "Model", yLabel: throughputMeaning, unit: "tok/s", strategy, labels, series: [{ label: throughputMeaning, values: comparisonRuns.map(runDisplayThroughput) }], runs: comparisonRuns });
    setChartModel("latencyChart", { type: "bar", title: "모델별 응답 지연", subtitle: `${strategyLabel} · 동일 suite${strategy === "broadcast_compare" ? " · 복제본 응답별 분포" : ""}${omittedSuffix}`, xLabel: "Model", yLabel: "Latency", unit: "s", strategy, labels, series: [
      { label: "TTFT p50", values: comparisonRuns.map(run => run.ttft_p50_s) }, { label: "E2E p95", values: comparisonRuns.map(run => run.e2e_p95_s) },
    ], runs: comparisonRuns });
  } else {
    const labels = completed.map(run => String(run.run_id || "").slice(9, 15));
    const modelNames = [...new Set(completed.map(run => shortModelName(runModelId(run))))];
    setChartModel("throughputChart", { type: "line", title: throughputMeaning, subtitle: `${strategyLabel} · 실행별 추이`, xLabel: "Run", yLabel: throughputMeaning, unit: "tok/s", strategy, labels, series: modelNames.map(name => ({ label: name, values: completed.map(run => shortModelName(runModelId(run)) === name ? runDisplayThroughput(run) : null), details: completed.map(run => run.run_id) })), runs: completed });
    const latencyRuns = completed.slice(-8);
    setChartModel("latencyChart", { type: "bar", title: "TTFT / E2E 지연", subtitle: `${strategyLabel} · 최근 실행`, xLabel: "Run", yLabel: "Latency", unit: "s", strategy, labels: latencyRuns.map(run => String(run.run_id || "").slice(9, 15)), series: [
      { label: "TTFT p50", values: latencyRuns.map(run => run.ttft_p50_s), details: latencyRuns.map(run => shortModelName(runModelId(run))) },
      { label: "E2E p95", values: latencyRuns.map(run => run.e2e_p95_s), details: latencyRuns.map(run => shortModelName(runModelId(run))) },
    ], runs: latencyRuns });
  }

  if (strategy === "model_parallel_rpc") {
    $("#nodeChartTitle").textContent = "분할 모델 공동 처리량";
    const source = comparisonRuns.length ? comparisonRuns : [latestRun];
    setChartModel("nodeChart", { type: "bar", title: "RPC coordinator 공동 처리량", subtitle: "워커별 tok/s는 계산하지 않음", xLabel: "Sharded model", yLabel: "Coordinator throughput", unit: "tok/s", strategy, labels: source.map(run => shortModelName(runModelId(run))), series: [{ label: "Sharded model", values: source.map(run => run.cluster_tokens_per_s), details: source.map(run => `${run.topology?.participants?.length || run.nodes?.length || "—"} participants`) }], runs: source });
    return;
  }
  const detailRun = latestRun.per_node && Object.keys(latestRun.per_node).length ? latestRun : [...completed].reverse().find(run => run.per_node && Object.keys(run.per_node).length);
  if (!detailRun) return drawEmptyChart($("#nodeChart"), "노드별 지표가 있는 새 실행이 필요합니다");
  const nodeNames = Object.keys(detailRun.per_node);
  const broadcast = strategy === "broadcast_compare";
  $("#nodeChartTitle").textContent = broadcast ? "복제본별 생성률" : "노드별 기여도";
  setChartModel("nodeChart", { type: "bar", title: broadcast ? "복제본별 토큰 생성률" : "노드별 유효 처리량", subtitle: broadcast ? `답변 일치 ${pct(Number(detailRun.answer_agreement_rate || 0) * 100)} · 사용자 처리량 아님` : shortModelName(runModelId(detailRun)), xLabel: "Node", yLabel: broadcast ? "Replica generation rate" : "Effective throughput", unit: "tok/s", strategy, labels: nodeNames, series: [{ label: broadcast ? "Replica tok/s" : "Effective tok/s", values: nodeNames.map(name => detailRun.per_node[name].effective_tokens_per_s) }], runs: [detailRun] });
}

function safeFilename(value) {
  return String(value || "chart").normalize("NFKD").replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 90) || "chart";
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function pngBytesWithDpi(arrayBuffer, dpi) {
  const source = new Uint8Array(arrayBuffer);
  if (source.length < 33 || String.fromCharCode(...source.slice(1, 4)) !== "PNG") return source;
  const type = new TextEncoder().encode("pHYs");
  const data = new Uint8Array(9); const view = new DataView(data.buffer); const pixelsPerMeter = Math.round(Number(dpi) / .0254);
  view.setUint32(0, pixelsPerMeter); view.setUint32(4, pixelsPerMeter); data[8] = 1;
  const crcInput = new Uint8Array(13); crcInput.set(type); crcInput.set(data, 4);
  let crc = 0xffffffff;
  for (const byte of crcInput) { crc ^= byte; for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0); }
  crc = (crc ^ 0xffffffff) >>> 0;
  const chunk = new Uint8Array(21); const chunkView = new DataView(chunk.buffer); chunkView.setUint32(0, 9); chunk.set(type, 4); chunk.set(data, 8); chunkView.setUint32(17, crc);
  const ihdrEnd = 8 + 4 + 4 + 13 + 4;
  let insertAt = ihdrEnd; let existingEnd = -1;
  while (insertAt + 12 <= source.length) {
    const length = new DataView(source.buffer, source.byteOffset + insertAt, 4).getUint32(0);
    const kind = String.fromCharCode(...source.slice(insertAt + 4, insertAt + 8));
    if (kind === "pHYs") { existingEnd = insertAt + 12 + length; break; }
    if (kind === "IDAT" || kind === "IEND") break;
    insertAt += 12 + length;
  }
  const removed = existingEnd > 0 ? existingEnd - insertAt : 0;
  const output = new Uint8Array(source.length + chunk.length - removed); output.set(source.slice(0, insertAt)); output.set(chunk, insertAt); output.set(source.slice(existingEnd > 0 ? existingEnd : insertAt), insertAt + chunk.length);
  return output;
}

function xmlEscape(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[char]));
}

function canonicalExperimentId(run) {
  return run.experiment_id || `legacy-${String(run.name || "unnamed").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "experiment"}`;
}

function publicationComparisonSignature(run) {
  const parameters = run.benchmark_parameters || {};
  const actual = (Array.isArray(run.actual_model_config) ? run.actual_model_config : [run.actual_model_config || {}])
    .map(item => ({ node: item.node || "", n_ctx: item.n_ctx, n_gpu_layers: item.n_gpu_layers, n_batch: item.n_batch, inference_threads: item.inference_threads }))
    .sort((a, b) => a.node.localeCompare(b.node));
  const topology = run.topology || {};
  return JSON.stringify({
    strategy: runStrategy(run),
    prompt_sha256: parameters.prompt_sha256 || "legacy-unknown",
    n_ctx: parameters.n_ctx,
    effective_model_config: actual,
    requests: parameters.requests_per_scenario ?? parameters.requests,
    concurrency: parameters.concurrency,
    max_tokens: parameters.max_tokens,
    temperature: parameters.temperature,
    top_p: parameters.top_p,
    seed: parameters.seed,
    warmup_requests: parameters.warmup_requests,
    nodes: [...(run.nodes || [])].sort(),
    topology: {
      engine: topology.engine,
      runtime_commit: topology.runtime_commit,
      split_mode: topology.split_mode,
      split_policy: topology.split_policy,
      tensor_split: topology.tensor_split,
      resolved_device_order: topology.resolved_device_order,
    },
  });
}

function publicationMetadata(model) {
  const run = model.runs?.[0] || {};
  const parameters = run.benchmark_parameters || {};
  const actualConfigs = Array.isArray(run.actual_model_config) ? run.actual_model_config : [run.actual_model_config || {}];
  const actualValues = key => [...new Set(actualConfigs.map(item => item?.[key]).filter(value => value !== undefined && value !== null))];
  const actualLabel = (key, requested) => {
    const values = actualValues(key);
    if (values.length === 1) return String(values[0]);
    if (values.length > 1) return `${Math.min(...values.map(Number))}–${Math.max(...values.map(Number))}`;
    return requested !== undefined && requested !== null ? `${requested} requested` : "";
  };
  const modelId = parameters.model_id || runModelId(run);
  const ctxLabel = actualLabel("n_ctx", parameters.n_ctx);
  const gpuLayerLabel = runStrategy(run) === "model_parallel_rpc"
    ? String(parameters.effective_n_gpu_layers || run.topology?.requested_gpu_layers || "all")
    : actualLabel("n_gpu_layers", parameters.requested_n_gpu_layers ?? parameters.n_gpu_layers);
  const fields = [
    model.runs?.length > 1 ? `${model.runs.length} models/runs` : shortModelName(modelId),
    ctxLabel ? `ctx ${ctxLabel}` : "",
    gpuLayerLabel ? `GPU layers ${gpuLayerLabel}` : "",
    finite(parameters.requests_per_scenario ?? parameters.requests) ? `n=${parameters.requests_per_scenario ?? parameters.requests}` : "",
    finite(parameters.concurrency) ? `concurrency ${parameters.concurrency}` : "",
    finite(parameters.max_tokens) ? `max ${parameters.max_tokens} tokens` : "",
    finite(parameters.seed) ? `seed ${parameters.seed}` : "",
  ].filter(Boolean);
  return fields.join(" · ") || `${strategyMeta(model.strategy).label} · 이전 결과(상세 실행 파라미터 미기록)`;
}

function buildPublicationSvg(model, options = {}) {
  const width = Math.max(640, Math.round(Number(options.width) || 1004));
  const sizeMm = Number(options.sizeMm) || 0;
  const physicalWidthMm = sizeMm || 85;
  const unitsPerPoint = width / (physicalWidthMm / 25.4 * 72);
  const pointSize = points => (points * unitsPerPoint).toFixed(2);
  const baseHeight = Math.round(width * (options.aspectRatio || .64));
  const series = model.series.map((item, index) => ({ ...item, color: PUBLICATION_COLORS[index % PUBLICATION_COLORS.length] }));
  const legendFontSize = Number(pointSize(7.5));
  const legendAvailable = width * .86;
  const legendEntries = series.map(item => {
    const label = ellipsis(item.label, 28);
    return { item, label, width: Math.min(legendAvailable, width * .05 + label.length * legendFontSize * .58) };
  });
  let legendRows = 1; let legendCursor = 0;
  legendEntries.forEach(entry => {
    if (legendCursor > 0 && legendCursor + entry.width > legendAvailable) { legendRows += 1; legendCursor = 0; }
    legendCursor += entry.width;
  });
  const legendRowHeight = Number(pointSize(10));
  const legendExtraHeight = Math.max(0, legendRows - 1) * legendRowHeight;
  const height = Math.round(baseHeight + legendExtraHeight);
  const svgWidth = sizeMm ? `${sizeMm}mm` : String(width);
  const svgHeight = sizeMm ? `${(sizeMm * height / width).toFixed(2)}mm` : String(height);
  const includeTitle = options.includeTitle !== false;
  const showValues = Boolean(options.showValues);
  const title = options.title || model.title;
  const subtitle = options.subtitle || publicationMetadata(model);
  const pad = { left: Math.round(width * .105), right: Math.round(width * .035), top: (includeTitle ? Math.round(baseHeight * .18) : Math.round(baseHeight * .08)) + legendExtraHeight, bottom: Math.round(baseHeight * .16) };
  const plotWidth = width - pad.left - pad.right; const plotHeight = height - pad.top - pad.bottom;
  const all = series.flatMap(item => item.values).filter(finite).map(Number);
  const maximum = Math.max(...all, 1) * 1.12;
  const font = "Arial, Noto Sans CJK KR, Noto Sans KR, sans-serif";
  const elements = [`<svg xmlns="http://www.w3.org/2000/svg" width="${svgWidth}" height="${svgHeight}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="figure-title figure-desc">`, `<title id="figure-title">${xmlEscape(title)}</title>`, `<desc id="figure-desc">${xmlEscape(`${subtitle}. ${model.yLabel} by ${model.xLabel}.`)}</desc>`, `<rect width="100%" height="100%" fill="#ffffff"/>`];
  if (includeTitle) {
    elements.push(`<text x="${pad.left}" y="${Math.round(baseHeight * .07)}" font-family="${font}" font-size="${pointSize(11)}" font-weight="700" fill="#111111">${xmlEscape(title)}</text>`);
    elements.push(`<text x="${pad.left}" y="${Math.round(baseHeight * .115)}" font-family="${font}" font-size="${pointSize(7.5)}" fill="#444444">${xmlEscape(ellipsis(subtitle, width < 900 ? 120 : 180))}</text>`);
  }
  for (let tick = 0; tick <= 5; tick += 1) {
    const y = pad.top + plotHeight * tick / 5; const value = maximum * (5 - tick) / 5;
    elements.push(`<line x1="${pad.left}" y1="${y.toFixed(2)}" x2="${width - pad.right}" y2="${y.toFixed(2)}" stroke="${tick === 5 ? "#333333" : "#dddddd"}" stroke-width="${tick === 5 ? 1.4 : 1}"/>`);
    elements.push(`<text x="${pad.left - 12}" y="${(y + 4).toFixed(2)}" text-anchor="end" font-family="${font}" font-size="${pointSize(7.5)}" fill="#222222">${xmlEscape(chartValue(value))}</text>`);
  }
  elements.push(`<text transform="translate(${Math.round(width * .026)},${pad.top + plotHeight / 2}) rotate(-90)" text-anchor="middle" font-family="${font}" font-size="${pointSize(8)}" font-weight="600" fill="#111111">${xmlEscape(`${model.yLabel}${model.unit ? ` (${model.unit})` : ""}`)}</text>`);
  const groupWidth = plotWidth / Math.max(1, model.labels.length);
  if (model.type === "line") {
    series.forEach((item, seriesIndex) => {
      let path = ""; let active = false;
      item.values.forEach((value, index) => {
        if (!finite(value)) { active = false; return; }
        const x = model.labels.length > 1 ? pad.left + plotWidth * index / (model.labels.length - 1) : pad.left + plotWidth / 2;
        const y = pad.top + plotHeight - Number(value) / maximum * plotHeight;
        path += `${active ? " L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`; active = true;
      });
      elements.push(`<path d="${path}" fill="none" stroke="${item.color}" stroke-width="${Math.max(2, width * .003)}"/>`);
      item.values.forEach((value, index) => {
        if (!finite(value)) return;
        const x = model.labels.length > 1 ? pad.left + plotWidth * index / (model.labels.length - 1) : pad.left + plotWidth / 2;
        const y = pad.top + plotHeight - Number(value) / maximum * plotHeight;
        elements.push(`<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${Math.max(3, width * .005)}" fill="${item.color}" stroke="#ffffff" stroke-width="1.5"/>`);
        if (showValues) elements.push(`<text x="${x.toFixed(2)}" y="${(y - 10).toFixed(2)}" text-anchor="middle" font-family="${font}" font-size="${pointSize(7.5)}" fill="#111111">${xmlEscape(chartValue(value))}</text>`);
      });
    });
  } else {
    const barWidth = Math.min(width * .065, groupWidth * .72 / series.length);
    series.forEach((item, seriesIndex) => item.values.forEach((value, groupIndex) => {
      if (!finite(value)) return;
      const center = pad.left + groupWidth * groupIndex + groupWidth / 2 + (seriesIndex - (series.length - 1) / 2) * barWidth;
      const barHeight = Number(value) / maximum * plotHeight; const x = center - barWidth * .4; const y = pad.top + plotHeight - barHeight;
      elements.push(`<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${(barWidth * .8).toFixed(2)}" height="${barHeight.toFixed(2)}" fill="${item.color}"/>`);
      if (showValues) elements.push(`<text x="${center.toFixed(2)}" y="${(y - 9).toFixed(2)}" text-anchor="middle" font-family="${font}" font-size="${pointSize(7.5)}" fill="#111111">${xmlEscape(chartValue(value))}</text>`);
    }));
  }
  const every = Math.max(1, Math.ceil(model.labels.length / Math.max(3, Math.floor(plotWidth / (width * .11)))));
  model.labels.forEach((label, index) => {
    if (index % every && index !== model.labels.length - 1) return;
    const x = model.type === "line" && model.labels.length > 1 ? pad.left + plotWidth * index / (model.labels.length - 1) : pad.left + groupWidth * index + groupWidth / 2;
    elements.push(`<text x="${x.toFixed(2)}" y="${height - pad.bottom + Math.round(baseHeight * .045)}" text-anchor="middle" font-family="${font}" font-size="${pointSize(7)}" fill="#222222">${xmlEscape(ellipsis(label, width < 900 ? 11 : 18))}</text>`);
  });
  elements.push(`<text x="${pad.left + plotWidth / 2}" y="${height - Math.round(baseHeight * .025)}" text-anchor="middle" font-family="${font}" font-size="${pointSize(8)}" font-weight="600" fill="#111111">${xmlEscape(model.xLabel || "")}</text>`);
  const legendStartY = includeTitle ? Math.round(baseHeight * .147) : Math.round(baseHeight * .038); let legendX = pad.left; let legendY = legendStartY;
  legendEntries.forEach(entry => {
    if (legendX > pad.left && legendX + entry.width > width - pad.right) { legendX = pad.left; legendY += legendRowHeight; }
    elements.push(`<rect x="${legendX}" y="${legendY - width * .008}" width="${width * .022}" height="${width * .006}" fill="${entry.item.color}"/><text x="${legendX + width * .028}" y="${legendY}" font-family="${font}" font-size="${pointSize(7.5)}" fill="#222222">${xmlEscape(entry.label)}</text>`);
    legendX += entry.width;
  });
  elements.push(`<metadata>${xmlEscape(JSON.stringify({ strategy: model.strategy, title, subtitle, generated_at: options.generatedAt || "", models: model.runs?.map(runModelId) || [] }))}</metadata>`, `</svg>`);
  return { svg: elements.join(""), width, height, sizeMm };
}

function downloadDashboardPng(chartId) {
  const model = state.chartModels.get(chartId); if (!model) return;
  const source = $(`#${chartId}`); const width = Math.max(1000, Math.round(source.clientWidth * 2)); const height = Math.round(width * .52);
  const canvas = document.createElement("canvas");
  drawChartCanvas(canvas, model, { width, height, ratio: 1, header: 104, background: "#f8f7f1", hidden: chartInteraction(chartId).hidden, cssWidth: `${width}px`, cssHeight: `${height}px` });
  canvas.toBlob(blob => blob && downloadBlob(blob, `${safeFilename(model.title)}-dashboard.png`), "image/png");
}

function updatePublicationSpec() {
  const sizeMm = $("#publicationSizeSelect").value === "two" ? 180 : 85;
  const format = $("#publicationFormatSelect").value;
  const dpi = Number($("#publicationDpiSelect").value);
  const pixels = Math.round(sizeMm / 25.4 * dpi);
  $("#publicationDpiField").hidden = format !== "png";
  $("#publicationOutputSpec").textContent = `${sizeMm} mm · ${format.toUpperCase()}`;
  $("#publicationPixelSpec").textContent = format === "png" ? `${pixels} px · ${dpi} DPI` : "벡터 해상도";
}

function openPublicationDialog(chartId) {
  const model = state.chartModels.get(chartId); if (!model) return;
  if ($("#resultExperimentFilter").value === "all") return toast("실험 선택 필요", "논문 그래프는 한 실험 묶음만 선택한 뒤 다운로드하세요.", "error");
  const strategies = new Set((model.runs || []).map(runStrategy));
  const experiments = new Set((model.runs || []).map(canonicalExperimentId));
  if (strategies.size > 1 || experiments.size > 1) return toast("결과 의미가 섞여 있음", "동일 experiment_id와 실행 방식의 결과만 논문 그래프로 만들 수 있습니다.", "error");
  const signatures = new Set((model.runs || []).map(publicationComparisonSignature));
  if (signatures.size > 1) return toast("비교 조건이 서로 다름", "프롬프트·노드·컨텍스트·동시성·생성·샘플링·실제 런타임 구성이 같은 결과만 논문 그래프로 내보낼 수 있습니다.", "error");
  state.publicationChartId = chartId;
  $("#publicationChartName").textContent = model.title;
  $("#publicationStrategy").textContent = `${strategyMeta(model.strategy).label} · ${publicationMetadata(model)}`;
  $("#publicationTitleInput").value = model.title;
  updatePublicationSpec(); $("#publicationDialog").showModal();
}

async function exportPublicationChart() {
  const model = state.chartModels.get(state.publicationChartId); if (!model) throw new Error("내보낼 그래프가 없습니다.");
  const format = $("#publicationFormatSelect").value; const dpi = Number($("#publicationDpiSelect").value); const sizeMm = $("#publicationSizeSelect").value === "two" ? 180 : 85;
  const width = format === "png" ? Math.round(sizeMm / 25.4 * dpi) : (sizeMm === 180 ? 1800 : 1004);
  const built = buildPublicationSvg(model, { width, sizeMm, title: $("#publicationTitleInput").value.trim() || model.title, includeTitle: $("#publicationIncludeTitle").checked, showValues: $("#publicationShowValues").checked, generatedAt: new Date().toISOString() });
  const base = `${safeFilename(sizeMm === 180 ? "two-column" : "one-column")}-${safeFilename(model.title)}-paper`;
  if (format === "svg") return downloadBlob(new Blob([built.svg], { type: "image/svg+xml;charset=utf-8" }), `${base}.svg`);
  const url = URL.createObjectURL(new Blob([built.svg], { type: "image/svg+xml;charset=utf-8" }));
  const image = new Image();
  await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = url; });
  const canvas = document.createElement("canvas"); canvas.width = built.width; canvas.height = built.height;
  const ctx = canvas.getContext("2d"); ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, canvas.width, canvas.height); ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  URL.revokeObjectURL(url);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
  if (blob) {
    const withDpi = pngBytesWithDpi(await blob.arrayBuffer(), dpi);
    downloadBlob(new Blob([withDpi], { type: "image/png" }), `${base}-${dpi}dpi.png`);
  }
}

function metricBar(label, value, detail = "") {
  const safe = finite(value) ? Math.min(100, Math.max(0, Number(value))) : 0;
  return `<div class="telemetry-bar"><div><span>${escapeHtml(label)}</span><strong>${finite(value) ? pct(value) : "N/A"}</strong></div><i><b style="width:${safe}%"></b></i>${detail ? `<small>${escapeHtml(detail)}</small>` : ""}</div>`;
}

function renderEnvironmentDetail(report) {
  const readiness = readinessMeta(report.status);
  const checks = Array.isArray(report.checks) ? report.checks : [];
  const missingPackages = Array.isArray(report.missing_system_packages) ? report.missing_system_packages : [];
  const manualCommands = Array.isArray(report.manual_commands) ? report.manual_commands : [];
  const backend = typeof report.backend === "object" ? report.backend?.kind : report.backend;
  const canInstall = !state.environmentBusy && !["ready", "checking", "blocked"].includes(readiness.status);
  return `
    <section class="readiness-detail" aria-labelledby="readinessDetailTitle">
      <div class="readiness-detail-head">
        <div>
          <span>LLM EXECUTION ENVIRONMENT</span>
          <h3 id="readinessDetailTitle">LLM 런타임 준비도</h3>
          <p>${escapeHtml(readiness.detail)} · 마지막 점검 ${escapeHtml(checkedAtLabel(report.checked_at))}</p>
        </div>
        <span class="readiness-hero-status ${readiness.status}"><i></i>${readiness.label}</span>
      </div>
      <div class="readiness-path-grid">
        <div><span>PLATFORM</span><strong>${escapeHtml(platformName(report.platform))}</strong><small>${escapeHtml([report.architecture, backend || "backend 미확인"].filter(Boolean).join(" · "))}</small></div>
        <div><span>PROJECT</span><strong title="${escapeHtml(report.project_dir || "")}">${escapeHtml(report.project_dir || "미확인")}</strong><small>프로젝트 전용 작업 경로</small></div>
        <div><span>VIRTUAL ENV</span><strong title="${escapeHtml(report.venv_path || "")}">${escapeHtml(report.venv_path || "미구성")}</strong><small>Python 패키지 격리 설치</small></div>
        <div><span>MODELS</span><strong>${finite(report.model_count) ? Number(report.model_count) : "—"}</strong><small>${finite(report.disk_free_gb) ? `disk ${fmt(report.disk_free_gb, 1)} GB free` : "발견한 로컬 GGUF"}</small></div>
      </div>
      <div class="readiness-checks">
        <div class="readiness-subhead"><strong>환경 체크리스트</strong><span>${checks.length}개 항목</span></div>
        <ul>${checks.length ? checks.map(check => {
          const meta = checkMeta(check.status);
          return `<li class="${meta.status}"><i aria-hidden="true">${meta.icon}</i><div><strong>${escapeHtml(check.label || check.id || "환경 항목")}</strong><small>${escapeHtml(check.detail || meta.label)}${check.auto_fixable ? " · 자동 구성 가능" : ""}</small></div><span>${meta.label}</span></li>`;
        }).join("") : `<li class="unknown"><i aria-hidden="true">·</i><div><strong>점검 결과 없음</strong><small>다시 점검을 실행하면 항목별 상태가 표시됩니다.</small></div><span>미확인</span></li>`}</ul>
      </div>
      ${missingPackages.length ? `<div class="readiness-guidance warning"><strong>시스템 패키지 필요</strong><p>${missingPackages.map(item => `<code>${escapeHtml(item)}</code>`).join(" ")}</p><small>passwordless sudo가 가능하면 고정 허용 목록만 설치하고, 불가하면 정확한 수동 명령을 안내합니다.</small></div>` : ""}
      ${manualCommands.length ? `<div class="readiness-guidance manual"><strong>해당 노드에서 직접 실행</strong>${manualCommands.map(command => `<code>${escapeHtml(command)}</code>`).join("")}</div>` : ""}
      <div class="readiness-detail-actions">
        <button class="button ghost compact" type="button" data-environment-node-action="environment-check" ${state.environmentBusy ? "disabled" : ""}>다시 점검</button>
        <button class="button secondary compact" type="button" data-environment-node-action="environment-install" ${canInstall ? "" : "disabled"}>자동 구성</button>
      </div>
    </section>`;
}

function renderNodeDetail() {
  if (!state.detailNode || !$("#nodeDetailDialog").open) return;
  const node = state.nodes.find(item => item.name === state.detailNode);
  if (!node) return;
  const live = statusFor(node.name);
  const metrics = live.metrics || {};
  const profile = live.profile || {};
  const kind = profile.platform_kind || node.platform;
  $("#detailNodeName").textContent = node.name;
  const cores = metrics.cpu?.cores_pct || [];
  const temperatures = metrics.temperatures_c || {};
  const rails = metrics.power?.rails_w || {};
  const engines = metrics.accelerator?.engines || {};
  const fans = metrics.fans || {};
  const environment = environmentFor(node.name);
  $("#nodeDetailContent").innerHTML = `
    <div class="detail-identity">
      <div><span>PLATFORM</span><strong>${escapeHtml(platformName(kind))}</strong><small>${escapeHtml(profile.board_model || "미확인")}</small></div>
      <div><span>OS / KERNEL</span><strong>${escapeHtml(profile.os || "—")}</strong><small>${escapeHtml(profile.l4t || profile.kernel || "")}</small></div>
      <div><span>BACKEND</span><strong>${escapeHtml(profile.runtime_backend?.kind || "—")}</strong><small>${profile.runtime_backend?.verified ? `검증됨 · ${escapeHtml(profile.cuda || "native")}` : "검증 안 됨"}</small></div>
      <div><span>UPTIME</span><strong>${formatUptime(metrics.uptime_s)}</strong><small>${escapeHtml(metrics.power?.mode || "")}</small></div>
    </div>
    <div class="detail-grid">
      <section class="telemetry-panel"><div class="telemetry-title"><span>CPU</span><strong>${pct(metrics.cpu_pct)}</strong></div>
        <div class="core-grid">${cores.length ? cores.map((value, index) => `<div><span>C${index}</span><i><b style="height:${Math.min(100, Number(value) || 0)}%"></b></i><small>${fmt(value, 0)}%</small></div>`).join("") : "<small>코어 데이터 없음</small>"}</div>
        <p>${fmt(metrics.cpu?.frequency_mhz, 0)} MHz · load ${fmt(metrics.cpu?.load_1m, 2)} / ${fmt(metrics.cpu?.load_5m, 2)} / ${fmt(metrics.cpu?.load_15m, 2)}</p>
      </section>
      <section class="telemetry-panel"><div class="telemetry-title"><span>MEMORY / STORAGE</span><strong>${pct(metrics.ram_pct)}</strong></div>
        ${metricBar("RAM", metrics.memory?.percent, `${fmt(metrics.memory?.used_mb, 0)} / ${fmt(metrics.memory?.total_mb, 0)} MB`)}
        ${metricBar("SWAP", metrics.swap?.percent, `${fmt(metrics.swap?.used_mb, 0)} / ${fmt(metrics.swap?.total_mb, 0)} MB`)}
        ${metricBar("DISK", metrics.disk?.percent, `${fmt(metrics.disk?.free_gb)} GB free`)}
      </section>
      <section class="telemetry-panel"><div class="telemetry-title"><span>ACCELERATOR / ENGINES</span><strong>${finite(metrics.gpu_pct) ? pct(metrics.gpu_pct) : "CPU ONLY"}</strong></div>
        <div class="tag-metrics">${Object.keys(engines).length ? Object.entries(engines).map(([name, value]) => `<span>${escapeHtml(name)} <strong>${finite(value) ? pct(value) : "—"}</strong></span>`).join("") : "<span>전용 가속기 지표 없음</span>"}</div>
      </section>
      <section class="telemetry-panel"><div class="telemetry-title"><span>THERMAL / POWER</span><strong>${finite(metrics.power_w) ? `${fmt(metrics.power_w, 2)} W` : "N/A"}</strong></div>
        <div class="tag-metrics">${Object.entries(temperatures).map(([name, value]) => `<span>${escapeHtml(name)} <strong>${fmt(value, 1)}°C</strong></span>`).join("") || "<span>온도 센서 없음</span>"}</div>
        <div class="rail-list">${Object.entries(rails).map(([name, value]) => `<div><span>${escapeHtml(name)}</span><strong>${fmt(value, 2)} W</strong></div>`).join("") || "<small>전력 레일 데이터 없음</small>"}</div>
        <p>FAN ${Object.entries(fans).map(([name, value]) => `${name} ${fmt(value, 0)}%`).join(" · ") || "N/A"} · RX ${fmt((metrics.network?.receive_bytes_s || 0) / 1024, 1)} KB/s · TX ${fmt((metrics.network?.send_bytes_s || 0) / 1024, 1)} KB/s</p>
      </section>
    </div>
    <section class="telemetry-history"><div><span>LIVE HISTORY</span><strong>최근 ${state.metricHistory.get(node.name)?.length || 0}개 표본 · CPU / GPU / RAM</strong></div><canvas id="telemetryChart" height="230" aria-label="노드 CPU GPU RAM 실시간 사용률 그래프"></canvas></section>
    ${renderEnvironmentDetail(environment)}
    ${metrics.sampler_error ? `<div class="telemetry-warning">${escapeHtml(metrics.sampler_error)}</div>` : ""}`;
  $$('[data-environment-node-action]', $("#nodeDetailContent")).forEach(button => button.addEventListener("click", () => {
    runEnvironmentAction(button.dataset.environmentNodeAction, [node.name]);
  }));
  requestAnimationFrame(drawTelemetryChart);
}

function drawTelemetryChart() {
  const canvas = $("#telemetryChart");
  if (!canvas || !state.detailNode) return;
  const history = (state.metricHistory.get(state.detailNode) || []).slice(-60);
  if (!history.length) return drawEmptyChart(canvas, "표본 수집 중");
  const { context: ctx, width, height } = setupCanvas(canvas);
  const pad = { left: 34, right: 12, top: 30, bottom: 24 };
  ctx.strokeStyle = "#d9d6ca"; ctx.fillStyle = "#858980"; ctx.font = "9px ui-monospace, monospace";
  [0, 25, 50, 75, 100].forEach(value => {
    const y = height - pad.bottom - value / 100 * (height - pad.top - pad.bottom);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    ctx.fillText(`${value}`, 4, y + 3);
  });
  const series = [
    { key: "cpu", label: "CPU", color: "#718f17" },
    { key: "gpu", label: "GPU", color: "#e57c38" },
    { key: "ram", label: "RAM", color: "#163126" },
  ];
  const step = history.length > 1 ? (width - pad.left - pad.right) / (history.length - 1) : 0;
  series.forEach((item, seriesIndex) => {
    ctx.strokeStyle = item.color; ctx.lineWidth = 2; ctx.beginPath(); let started = false;
    history.forEach((sample, index) => {
      if (!finite(sample[item.key])) return;
      const x = pad.left + step * index;
      const y = height - pad.bottom - Number(sample[item.key]) / 100 * (height - pad.top - pad.bottom);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = item.color; ctx.fillRect(pad.left + seriesIndex * 64, 8, 9, 9);
    ctx.fillStyle = "#5f645d"; ctx.fillText(item.label, pad.left + 13 + seriesIndex * 64, 16);
  });
}

function openNodeDetail(nodeName) {
  state.detailNode = nodeName;
  $("#nodeDetailDialog").showModal();
  renderNodeDetail();
}

function setRunState(active) {
  state.activeExperiment = active;
  const running = active && ["queued", "running"].includes(active.status);
  $("#runButton").classList.toggle("hidden", running);
  $("#cancelButton").classList.toggle("hidden", !running);
  $("#runStateDot").classList.toggle("running", running);
  if (!active) return;
  if (active.execution_strategy || active.strategy) $("#runStrategyBadge").textContent = strategyMeta(active.execution_strategy || active.strategy).label;
  if (active.current_model) $("#runStrategyBadge").textContent = `${strategyMeta(active.execution_strategy || active.strategy).label} · ${shortModelName(active.current_model)} ${finite(active.model_index) && Number(active.model_index) > 0 && finite(active.model_count) ? `${active.model_index}/${active.model_count}` : ""}`;
  const total = Number(active.total || 0);
  const completed = Number(active.completed || 0);
  const progress = total ? Math.min(100, Math.round(completed / total * 100)) : 0;
  $("#runProgressBar").style.width = `${progress}%`;
  $("#runProgressText").textContent = `${progress}%`;
  $("#runPhase").textContent = ({
    queued: "실행 준비", suite: "모델 Suite 준비", model_starting: "다음 모델 준비",
    loading_model: "모델 로드", warmup: "워밍업", measurement: "부하 측정",
    model_cleanup: "모델 메모리 정리", model_cooldown: "모델 교체 대기",
    cancelling: "취소 요청", finished: "Suite 종료",
  })[active.phase] || active.status || "실행 중";
  if (active.error) logLine("ERROR", active.error);
}

function logLine(label, message) {
  const log = $("#consoleLog");
  const line = document.createElement("p");
  line.innerHTML = `<time>${escapeHtml(label)}</time>${escapeHtml(message)}`;
  log.append(line);
  while (log.children.length > 80) log.firstElementChild.remove();
  log.scrollTop = log.scrollHeight;
}

function environmentLogLine(label, message) {
  const log = $("#environmentLog");
  if (!log) return;
  const line = document.createElement("p");
  line.innerHTML = `<time>${escapeHtml(label)}</time>${escapeHtml(message)}`;
  log.append(line);
  while (log.children.length > 80) log.firstElementChild.remove();
  log.scrollTop = log.scrollHeight;
}

function actionName(value) {
  if (typeof value === "string") return value;
  return value?.action || value?.name || value?.action_name || "";
}

function actionId(value) {
  if (!value || typeof value === "string") return "";
  return String(value.id || value.action_id || value.task_id || "");
}

function environmentActionFromMessage(message) {
  const name = actionName(message.action) || message.action_name || message.name || "";
  const id = actionId(message.action) || String(message.action_id || message.task_id || "");
  return {
    name,
    id,
    matches: name.startsWith("environment-") || Boolean(id && state.environmentActionIds.has(id)),
  };
}

function setEnvironmentBusy(busy, phase = "환경 점검 대기") {
  state.environmentBusy = busy;
  const dot = $("#environmentStateDot");
  if (dot) dot.classList.toggle("running", busy);
  const label = $("#environmentPhase");
  if (label) label.textContent = phase;
  renderEnvironmentSummary();
  renderNodeDetail();
}

async function refreshEnvironmentReports() {
  const data = await api("/api/environment");
  setEnvironmentReports(data);
  return normalizedEnvironmentItems(data);
}

async function runEnvironmentAction(action, nodeNames = [...state.selectedNodes]) {
  const nodes = [...new Set(nodeNames)].filter(name => state.nodes.some(node => node.name === name && node.enabled));
  if (!nodes.length) return toast("노드 선택 필요", "환경을 점검할 노드를 한 대 이상 선택하세요.", "error");
  if (state.environmentBusy) return toast("환경 작업 진행 중", "현재 작업이 끝난 뒤 다시 시도하세요.", "error");
  const installing = action === "environment-install";
  if (installing && !confirm(`선택한 ${nodes.length}대의 LLM 실행 환경을 자동 구성합니다. Python 패키지는 각 프로젝트의 가상환경에 설치합니다. passwordless sudo가 가능하면 고정된 시스템 패키지만 설치하고, 불가하면 수동 명령만 안내합니다. 계속할까요?`)) return null;
  const previous = state.environment.map(report => ({ ...report }));
  const known = new Map(state.environment.map(report => [report.node, report]));
  nodes.forEach(node => known.set(node, { ...environmentFor(node), node, status: "checking" }));
  state.environment = [...known.values()];
  renderNodes();
  setEnvironmentBusy(true, installing ? "선택 노드 자동 구성 중" : "선택 노드 환경 점검 중");
  environmentLogLine(installing ? "INSTALL" : "CHECK", `${nodes.join(", ")} · 작업 요청`);
  try {
    const options = installing ? { confirmed: true, models: selectedModelIds() } : {};
    const result = await api("/api/actions", { method: "POST", body: { action, node_names: nodes, options } });
    const created = result.action || result;
    const id = actionId(created);
    if (id && state.environmentBusy) state.environmentActionIds.add(id);
    toast(installing ? "환경 자동 구성 시작" : "환경 점검 시작", nodes.join(", "));
    return result;
  } catch (error) {
    state.environment = previous;
    setEnvironmentBusy(false, "환경 작업 시작 실패");
    renderNodes();
    environmentLogLine("ERROR", error.message);
    toast("환경 작업 시작 실패", error.message, "error");
    return null;
  }
}

async function bootstrap() {
  try {
    const data = await api("/api/bootstrap");
    state.nodes = data.nodes || [];
    state.status = data.status || [];
    state.models = data.models || [];
    state.runs = data.runs || [];
    state.suites = data.suites || [];
    state.experimentGroups = data.experiment_groups || [];
    state.actions = data.actions || [];
    state.environment = [];
    setEnvironmentReports(data.environment || data.node_readiness || []);
    state.onboarding = data.onboarding || {};
    state.settings = data.settings || { worker_api_auth: false, dashboard_token_auth: false };
    if (!state.settings.dashboard_token_auth && state.token) {
      state.token = "";
      sessionStorage.removeItem("clusterToken");
    }
    ingestStatus(state.status);
    applyDefaults(data.defaults || {});
    renderExperimentGroups();
    renderNodes();
    renderRuns();
    setRunState(data.active_experiment);
    $("#publicKey").textContent = state.onboarding.public_key || "키가 아직 생성되지 않았습니다.";
    renderSettings();
    state.environmentActionIds.clear();
    const runningEnvironmentActions = state.actions.filter(action => actionName(action).startsWith("environment-") && ["queued", "running"].includes(action.status));
    runningEnvironmentActions.forEach(action => { const id = actionId(action); if (id) state.environmentActionIds.add(id); });
    setEnvironmentBusy(Boolean(runningEnvironmentActions.length), runningEnvironmentActions.length ? "노드 환경 작업 진행 중" : "환경 점검 대기");
    connectEvents();
    if (!location.hash) requestAnimationFrame(() => window.scrollTo({ top: 0, left: 0, behavior: "instant" }));
  } catch (error) {
    if (/401|token|invalid|missing/i.test(error.message)) {
      if (!$("#authDialog").open) $("#authDialog").showModal();
    }
    else toast("대시보드 초기화 실패", error.message, "error");
  }
}

async function refreshExperimentData() {
  const data = await api("/api/experiments");
  state.runs = data.runs || [];
  state.suites = data.suites || [];
  state.experimentGroups = data.experiment_groups || [];
  renderExperimentGroups();
  renderRuns();
}

function connectEvents() {
  if (state.eventSource) state.eventSource.close();
  const eventUrl = state.token ? `/api/events?token=${encodeURIComponent(state.token)}` : "/api/events";
  state.eventSource = new EventSource(eventUrl);
  state.eventSource.onopen = () => {
    $(".live-indicator").classList.add("connected");
    $("#streamLabel").textContent = "실시간 연결됨";
  };
  state.eventSource.onerror = () => {
    $(".live-indicator").classList.remove("connected");
    $("#streamLabel").textContent = "재연결 중";
    if (state.settings.dashboard_token_auth && !$("#authDialog").open) $("#authDialog").showModal();
  };
  state.eventSource.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.type === "cluster_status") {
      state.status = message.nodes || [];
      ingestStatus(state.status);
      $("#lastUpdated").textContent = `UPDATED ${new Date(message.at).toLocaleTimeString("ko-KR")}`;
      renderNodes();
      renderNodeDetail();
    } else if (message.type === "inventory_changed") {
      state.nodes = message.nodes || state.nodes;
      renderNodes();
    } else if (message.type === "environment_changed") {
      setEnvironmentReports(message.environment || message.reports || message.node_readiness || message.report || message);
      const changed = normalizedEnvironmentItems(message.environment || message.reports || message.node_readiness || message.report || message);
      if (changed.length) environmentLogLine("STATUS", `${changed.map(report => report.node || report.name).filter(Boolean).join(", ")} · 환경 상태 갱신`);
    } else if (message.type === "settings_changed") {
      state.settings = message.settings || state.settings;
      renderSettings();
      if (state.settings.dashboard_token_auth && !state.token && !$("#authDialog").open) {
        $("#authDialog").showModal();
      } else if (!state.settings.dashboard_token_auth && state.token) {
        state.token = "";
        sessionStorage.removeItem("clusterToken");
        setTimeout(connectEvents, 0);
      }
      if (message.action) toast("보안 설정 적용 중", "연결된 worker API를 재시작합니다.");
    } else if (message.type === "auth_required") {
      state.eventSource?.close();
      if (!$("#authDialog").open) $("#authDialog").showModal();
    } else if (message.type === "action_started") {
      const environmentAction = environmentActionFromMessage(message);
      if (environmentAction.matches) {
        if (environmentAction.id) state.environmentActionIds.add(environmentAction.id);
        setEnvironmentBusy(true, environmentAction.name === "environment-install" ? "선택 노드 자동 구성 중" : "선택 노드 환경 점검 중");
        environmentLogLine("START", `${environmentAction.name || "environment"} 작업 시작`);
      }
    } else if (message.type === "action_log") {
      const environmentAction = environmentActionFromMessage(message);
      if (environmentAction.matches) {
        const line = message.line || message.message || "환경 작업 진행 중";
        if (!String(line).startsWith("CLUSTER_ENVIRONMENT_JSON=")) environmentLogLine("TASK", line);
      }
      else logLine("TASK", message.line);
    } else if (message.type === "action_finished") {
      const action = message.action;
      const environmentAction = environmentActionFromMessage(message);
      if (environmentAction.matches) {
        const id = environmentAction.id || actionId(action);
        if (id) state.environmentActionIds.delete(id);
        const status = action?.status || message.status || "completed";
        const nodes = Array.isArray(action?.nodes) ? action.nodes : Array.isArray(message.nodes) ? message.nodes : [];
        environmentLogLine(status === "completed" ? "DONE" : "ERROR", `${environmentAction.name || "환경 작업"} · ${nodes.join(", ") || "선택 노드"} · ${status}`);
        setEnvironmentBusy(state.environmentActionIds.size > 0, state.environmentActionIds.size ? "노드 환경 작업 진행 중" : "환경 작업 완료");
        refreshEnvironmentReports().catch(error => environmentLogLine("WARN", `최신 상태 조회 실패 · ${error.message}`));
        toast(status === "completed" ? "환경 작업 완료" : "환경 작업 실패", nodes.join(", ") || environmentAction.name, status === "completed" ? "success" : "error");
      } else {
        const status = action?.status || message.status;
        const nodes = Array.isArray(action?.nodes) ? action.nodes : [];
        toast(status === "completed" ? "작업 완료" : "작업 실패", `${actionName(action)} · ${nodes.join(", ")}`, status === "completed" ? "success" : "error");
      }
    } else if (message.type === "experiment_event") {
      const inner = message.event || {};
      setRunState(message.active);
      if (inner.type === "phase") logLine("PHASE", inner.message || inner.phase);
      if (inner.type === "suite_started") logLine("SUITE", `${inner.model_count || message.active?.model_count || "—"} models 시작`);
      if (inner.type === "model_started") logLine("MODEL", `${shortModelName(inner.model_id)} · ${inner.model_index || "—"}/${inner.model_count || message.active?.model_count || "—"}`);
      if (inner.type === "model_finished") logLine("MODEL", `${shortModelName(inner.model_id)} 완료 · ${fmt(runDisplayThroughput(inner.summary || {}))} tok/s`);
      if (inner.type === "model_failed") logLine("ERROR", `${shortModelName(inner.model_id)} 실패 · ${inner.error || "unknown error"}`);
      if (inner.type === "node_model_loaded") logLine("MODEL", `${inner.node} · ${inner.actual?.model_id || "loaded"}`);
      if (inner.type === "request_completed") logLine("RUN", `${inner.completed}/${inner.total} · ${inner.result?.node} · ${inner.result?.ok ? "OK" : "FAIL"}`);
      if (inner.type === "warning") logLine("WARN", inner.message);
      if (inner.type === "run_finished") {
        refreshExperimentData().catch(error => toast("결과 갱신 실패", error.message, "error"));
        if (!(message.active?.model_count > 1 || inner.summary?.suite_id)) toast("벤치마크 완료", `${fmt(inner.summary.cluster_tokens_per_s)} tok/s · ${inner.summary.nodes.length} nodes`);
      }
      if (inner.type === "suite_finished") {
        refreshExperimentData().catch(error => toast("결과 갱신 실패", error.message, "error"));
        const completedModels = inner.completed_models ?? message.active?.completed_models ?? 0;
        const failedModels = Array.isArray(inner.errors) ? inner.errors.length : message.active?.errors?.length || 0;
        const suiteStatus = inner.status || message.active?.status || "completed";
        toast(
          suiteStatus === "completed" ? "모델 Suite 완료" : `모델 Suite ${suiteStatus}`,
          `${completedModels}개 완료 · ${failedModels}개 오류`,
          suiteStatus === "completed" ? "success" : "error",
        );
      }
    } else if (message.type === "experiment_failed") {
      setRunState(message.active);
      refreshExperimentData().catch(() => {});
      toast("벤치마크 실패", message.message, "error");
    }
  };
}

async function runActionOnNodes(action, nodeNames, options = {}) {
  if (!nodeNames.length) return toast("노드 선택 필요", "하나 이상의 노드를 선택하세요.", "error");
  try {
    const result = await api("/api/actions", { method: "POST", body: { action, node_names: nodeNames, options } });
    logLine("TASK", `${action} 시작 · ${result.action.nodes.join(", ")}`);
    toast("작업 시작", action);
    return result;
  } catch (error) {
    toast("작업 시작 실패", error.message, "error");
    return null;
  }
}

async function runAction(action, options = {}) {
  return runActionOnNodes(action, [...state.selectedNodes], options);
}

function experimentPayload() {
  const executionStrategy = selectedStrategy();
  const selectedNodeNames = [...state.selectedNodes];
  const modelIds = selectedModelIds();
  if (!modelIds.length) throw new Error("벤치마크할 모델을 한 개 이상 선택하세요.");
  if (executionStrategy === "single_node" && selectedNodeNames.length !== 1) {
    throw new Error("단일 노드 기준선은 정확히 한 대만 선택해야 합니다.");
  }
  if (["broadcast_compare", "node_sweep"].includes(executionStrategy) && selectedNodeNames.length < 2) {
    throw new Error("선택한 실행 방식에는 최소 두 대의 노드가 필요합니다.");
  }
  if (executionStrategy === "model_parallel_rpc") {
    const selectedNodes = selectedNodeNames.map(name => state.nodes.find(node => node.name === name)).filter(Boolean);
    if (selectedNodes.filter(node => node.role === "head").length !== 1 || !selectedNodes.some(node => node.role === "worker")) {
      throw new Error("모델 분할 RPC에는 coordinator인 head 1대와 worker 1대 이상을 선택해야 합니다.");
    }
    if (!$("#rpcAcknowledgeInput").checked) throw new Error("모델 분할 RPC의 실험적 특성과 위험을 먼저 확인하세요.");
  }
  const tensorSplit = executionStrategy === "model_parallel_rpc" && $("#rpcSplitPolicySelect").value === "custom" ? parseRpcTensorSplit(true) : [];
  return {
    experiment_id: $("#experimentGroupSelect").value,
    name: $("#experimentName").value.trim(),
    node_names: plannedNodeNames(),
    model_id: modelIds[0],
    model_ids: modelIds,
    continue_on_model_error: $("#continueModelErrorInput").checked,
    model_cooldown_s: Number($("#modelCooldownInput").value),
    execution_strategy: executionStrategy,
    sweep_mode: $("#sweepModeSelect").value,
    rpc_split_mode: $("#rpcSplitModeSelect").value,
    rpc_split_policy: $("#rpcSplitPolicySelect").value,
    rpc_tensor_split: tensorSplit,
    acknowledge_experimental_rpc: $("#rpcAcknowledgeInput").checked,
    requests: Number($("#requestsInput").value),
    concurrency: Number($("#concurrencyInput").value),
    max_tokens: Number($("#maxTokensInput").value),
    n_ctx: Number($("#contextInput").value),
    n_gpu_layers: Number($("#layersInput").value),
    warmup_requests: Number($("#warmupInput").value),
    temperature: Number($("#temperatureInput").value),
    top_p: Number($("#topPInput").value),
    seed: Number($("#seedInput").value),
    require_uniform_config: $("#uniformInput").checked,
    prompt: $("#promptInput").value,
  };
}

function candidatePayload() {
  return {
    name: $("#nodeName").value.trim(), role: "worker", host: $("#nodeHost").value.trim(),
    user: $("#nodeUser").value.trim(), ssh_port: Number($("#nodeSshPort").value), api_port: Number($("#nodeApiPort").value),
    project_dir: $("#nodeProjectDir").value.trim(), enabled: true, identity_file: "", platform: $("#nodePlatform").value,
  };
}

function renderDevices(scan) {
  state.devices = scan.devices || [];
  const networks = (scan.networks || []).map(item => `${item.interface} · ${item.network}`).join(", ");
  $("#scanStatus").textContent = `${networks || "사설 LAN 없음"} · SSH 기기 ${state.devices.length}대`;
  const list = $("#deviceList");
  if (!state.devices.length) {
    list.innerHTML = `<div class="device-empty">SSH 포트가 열린 기기를 찾지 못했습니다. 워커의 SSH 서비스를 확인하세요.</div>`;
    return;
  }
  list.innerHTML = state.devices.map(device => `
    <button type="button" class="device-card ${device.is_head ? "head-device" : ""}" data-device-host="${escapeHtml(device.host)}" ${device.is_head ? "disabled" : ""}>
      <i></i><span><strong>${escapeHtml(device.known_node || device.host)}</strong><small>${escapeHtml(device.host)} · SSH ${device.ssh_port}${device.is_head ? " · HEAD" : device.known_node ? " · 등록됨" : " · 새 기기"}</small><code>${escapeHtml(device.fingerprint || "fingerprint 확인 불가")}</code></span><b>선택</b>
    </button>`).join("");
  $$('[data-device-host]').forEach(button => button.addEventListener("click", () => {
    const device = state.devices.find(item => item.host === button.dataset.deviceHost);
    $("#nodeHost").value = device.host;
    if (device.known_node) {
      const known = state.nodes.find(node => node.name === device.known_node);
      $("#nodeName").value = device.known_node;
      if (known) {
        $("#nodeUser").value = known.user;
        $("#nodeSshPort").value = known.ssh_port;
        $("#nodeApiPort").value = known.api_port;
        $("#nodeProjectDir").value = known.project_dir;
        $("#nodePlatform").value = known.platform || "auto";
      }
    }
    else if (!$("#nodeName").value) {
      const used = new Set(state.nodes.map(node => node.name));
      const available = [1, 2, 3].find(index => !used.has(`edge-worker-0${index}`)) || state.nodes.length;
      $("#nodeName").value = `edge-worker-0${available}`;
    }
    $$('.device-card').forEach(item => item.classList.toggle("selected", item === button));
    state.onboardingProbe = null;
    $("#probeResult").hidden = true;
  }));
}

async function scanNetwork(force = false) {
  $("#scanStatus").textContent = "사설 LAN에서 SSH 기기를 검색하는 중…";
  $("#deviceList").innerHTML = `<div class="device-empty scanning">최대 /24 범위 · SSH 포트만 확인합니다.</div>`;
  try {
    const result = await api(`/api/network/scan?force=${force ? "true" : "false"}`, { method: "POST" });
    renderDevices(result);
  } catch (error) {
    $("#scanStatus").textContent = "검색 실패";
    $("#deviceList").innerHTML = `<div class="device-empty">${escapeHtml(error.message)}</div>`;
  }
}

function renderProbe(result) {
  state.onboardingProbe = result;
  const panel = $("#probeResult");
  panel.hidden = false;
  const discovery = result.discovery || {};
  if (!result.ok) {
    const paired = result.ssh_ok;
    panel.className = "probe-result failed";
    panel.innerHTML = `<strong>${paired ? "지원하지 않는 환경" : "SSH 공개 키 인증 필요"}</strong><span>${paired ? "Jetson 또는 Raspberry Pi의 64-bit ARM OS가 필요합니다." : "Head 공개 키를 선택한 기기의 authorized_keys에 등록한 뒤 다시 확인하세요."}</span><small>${escapeHtml((result.warnings || []).join(" · ") || discovery.error || "연결할 수 없음")}</small>`;
    return;
  }
  const missing = discovery.missing_packages || [];
  const manual = missing.length && !discovery.sudo_nopasswd;
  panel.className = `probe-result ${manual ? "warning" : "ready"}`;
  panel.innerHTML = `
    <strong>${manual ? "수동 sudo 1회 필요" : "자동 준비 가능"}</strong>
    <span>${escapeHtml(platformName(discovery.platform_kind))} · ${escapeHtml(discovery.board_model || "")} · ${escapeHtml(discovery.architecture || "")} · ${escapeHtml(discovery.os || "")}</span>
    <small>프로젝트 ${discovery.project ? "있음" : "신규 설치"} · 디스크 ${fmt(discovery.disk_free_gb)} GB · NTP ${escapeHtml(discovery.ntp_synchronized || "미확인")}</small>
    ${missing.length ? `<code>sudo apt-get update &amp;&amp; sudo apt-get install -y ${escapeHtml(missing.join(" "))}</code>` : ""}
    ${(result.warnings || []).map(warning => `<em>${escapeHtml(warning)}</em>`).join("")}`;
  if (["jetson", "raspberry-pi"].includes(discovery.platform_kind)) $("#nodePlatform").value = discovery.platform_kind;
}

async function probeCandidate() {
  const payload = candidatePayload();
  if (!payload.host || !payload.name || !payload.user) throw new Error("기기와 SSH 계정 정보를 먼저 입력하세요.");
  $("#probeResult").hidden = false;
  $("#probeResult").className = "probe-result";
  $("#probeResult").innerHTML = `<strong>SSH 및 환경 확인 중…</strong>`;
  const result = await api("/api/nodes/probe", { method: "POST", body: payload });
  renderProbe(result);
  return result;
}

function resetNodeForm() {
  $("#nodeForm").reset();
  $("#nodeUser").value = "jetson_orin_nano";
  $("#nodeProjectDir").value = "/home/jetson_orin_nano/project/llm/local_llm_bench";
  $("#nodePlatform").value = "auto";
  $("#probeResult").hidden = true;
  state.onboardingProbe = null;
}

function bindEvents() {
  $("#authForm").addEventListener("submit", event => {
    event.preventDefault();
    state.token = $("#tokenInput").value.trim();
    sessionStorage.setItem("clusterToken", state.token);
    $("#authDialog").close();
    bootstrap();
  });
  $$('[data-close-dialog]').forEach(button => button.addEventListener("click", () => button.closest("dialog").close()));
  $$('[data-chart-png]').forEach(button => button.addEventListener("click", () => downloadDashboardPng(button.dataset.chartPng)));
  $$('[data-paper-export]').forEach(button => button.addEventListener("click", () => openPublicationDialog(button.dataset.paperExport)));
  $("#publicationFormatSelect").addEventListener("change", updatePublicationSpec);
  $("#publicationSizeSelect").addEventListener("change", updatePublicationSpec);
  $("#publicationDpiSelect").addEventListener("change", updatePublicationSpec);
  $("#publicationForm").addEventListener("submit", async event => {
    event.preventDefault();
    try { await exportPublicationChart(); $("#publicationDialog").close(); toast("논문용 그래프 다운로드", "선택한 규격으로 그래프를 생성했습니다."); }
    catch (error) { toast("그래프 생성 실패", error.message, "error"); }
  });
  $$(".chart-legend").forEach(legend => legend.addEventListener("click", event => {
    const button = event.target.closest("[data-chart-series]"); if (!button) return;
    const chartId = legend.id.replace(/Legend$/, ""); const model = state.chartModels.get(chartId); if (!model) return;
    const interaction = chartInteraction(chartId); const label = button.dataset.chartSeries;
    if (interaction.hidden.has(label)) interaction.hidden.delete(label); else interaction.hidden.add(label);
    setChartModel(chartId, model);
  }));
  $("#addNodeButton").addEventListener("click", () => {
    $("#nodeDialog").showModal();
    scanNetwork();
  });
  $("#settingsButton").addEventListener("click", () => {
    renderSettings();
    $("#settingsDialog").showModal();
  });
  $("#workerAuthInput").addEventListener("change", event => {
    $("#workerAuthNotice").textContent = event.currentTarget.checked
      ? "저장 후 켜짐 · 연결 노드 재시작 필요"
      : "저장 후 꺼짐 · 신뢰 LAN 전용 모드";
  });
  $("#dashboardAuthInput").addEventListener("change", updateDashboardAuthGuidance);
  $("#settingsForm").addEventListener("submit", async event => {
    event.preventDefault();
    try {
      const dashboardAuth = $("#dashboardAuthInput").checked;
      const enablingDashboardAuth = dashboardAuth && !state.settings.dashboard_token_auth;
      const dashboardToken = enablingDashboardAuth ? $("#dashboardTokenInput").value.trim() : state.token;
      if (enablingDashboardAuth && !dashboardToken) {
        return toast("대시보드 토큰 필요", "head의 .run/cluster/dashboard.token 값을 입력하세요.", "error");
      }
      const data = await api("/api/settings", {
        method: "PUT",
        body: {
          worker_api_auth: $("#workerAuthInput").checked,
          dashboard_token_auth: dashboardAuth,
          dashboard_token: dashboardToken,
        },
      });
      if (dashboardAuth) {
        state.token = dashboardToken;
        sessionStorage.setItem("clusterToken", dashboardToken);
        if ($("#authDialog").open) $("#authDialog").close();
      } else {
        state.token = "";
        sessionStorage.removeItem("clusterToken");
      }
      state.settings = data.settings;
      renderSettings();
      connectEvents();
      $("#settingsDialog").close();
      toast("설정 저장 완료", data.action ? "워커 API 재시작 작업을 시작했습니다." : "보안 설정을 즉시 적용했습니다.");
    } catch (error) {
      renderSettings();
      connectEvents();
      toast("설정 적용 실패", error.message, "error");
    }
  });
  $("#scanNetworkButton").addEventListener("click", () => scanNetwork(true));
  $("#probeNodeButton").addEventListener("click", async () => {
    try { await probeCandidate(); }
    catch (error) { toast("환경 확인 실패", error.message, "error"); }
  });
  $("#nodeForm").addEventListener("submit", async event => {
    event.preventDefault();
    try {
      const probe = await probeCandidate();
      if (!probe.ok) return toast("SSH 키 등록 필요", "공개 키를 워커에 등록한 뒤 다시 확인하세요.", "error");
      const discovery = probe.discovery || {};
      if ((discovery.missing_packages || []).length && !discovery.sudo_nopasswd) {
        return toast("수동 패키지 설치 필요", "표시된 sudo 명령을 워커에서 한 번 실행한 뒤 다시 확인하세요.", "error");
      }
      const payload = candidatePayload();
      const result = await api("/api/nodes", { method: "POST", body: payload });
      const index = state.nodes.findIndex(node => node.name === result.node.name);
      if (index >= 0) state.nodes[index] = result.node; else state.nodes.push(result.node);
      state.selectedNodes.add(result.node.name);
      renderNodes();
      $("#nodeDialog").close();
      resetNodeForm();
      await runActionOnNodes("prepare", [result.node.name], { confirmed: true, models: selectedModelIds() });
      toast("워커 등록 완료", "환경 구성, 모델 동기화와 API 시작 작업을 진행합니다.");
    } catch (error) { toast("워커 등록 실패", error.message, "error"); }
  });
  $("#copyKeyButton").addEventListener("click", async () => {
    if (!state.onboarding.public_key) return toast("SSH 키 없음", "head에서 키 생성 스크립트를 실행하세요.", "error");
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(state.onboarding.public_key);
      else {
        const helper = document.createElement("textarea");
        helper.value = state.onboarding.public_key;
        helper.style.position = "fixed"; helper.style.opacity = "0";
        document.body.append(helper); helper.select();
        if (!document.execCommand("copy")) throw new Error("copy unsupported");
        helper.remove();
      }
      toast("SSH 공개 키 복사됨");
    } catch (_error) {
      const range = document.createRange();
      range.selectNodeContents($("#publicKey"));
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      toast("키를 선택했습니다", "복사가 차단되어 있습니다. 선택된 키를 직접 복사하세요.", "error");
    }
  });
  $("#refreshButton").addEventListener("click", () => api("/api/status/refresh", { method: "POST" }).catch(error => toast("새로고침 실패", error.message, "error")));
  ["#quickHealthButton", "#checkEnvironmentButton", "#environmentCheckAllButton"].forEach(selector => {
    $(selector).addEventListener("click", () => runEnvironmentAction("environment-check"));
  });
  ["#installEnvironmentButton", "#environmentInstallAllButton"].forEach(selector => {
    $(selector).addEventListener("click", () => runEnvironmentAction("environment-install"));
  });
  $("#prepareRpcButton").addEventListener("click", () => {
    const nodes = [...state.selectedNodes];
    const records = nodes.map(name => state.nodes.find(node => node.name === name)).filter(Boolean);
    if (records.filter(node => node.role === "head").length !== 1 || !records.some(node => node.role === "worker")) {
      return toast("노드 구성 확인", "RPC coordinator인 head 1대와 worker 1대 이상을 선택하세요.", "error");
    }
    if (confirm(`선택한 ${nodes.length}대에 모델 분할 RPC 실행 환경을 준비합니다. 실제 성능은 네트워크 상태에 따라 저하될 수 있습니다. 계속할까요?`)) {
      runActionOnNodes("prepare-rpc", nodes, { confirmed: true });
    }
  });
  $$('[data-cluster-action]').forEach(button => button.addEventListener("click", () => {
    const action = button.dataset.clusterAction;
    const options = action === "sync-models" ? { models: selectedModelIds() } : {};
    runAction(action, options);
  }));
  $$('.segmented button').forEach(button => button.addEventListener("click", () => {
    const count = Number(button.dataset.preset);
    const candidates = state.nodes.filter(node => node.enabled).slice(0, count);
    state.selectedNodes = new Set(candidates.map(node => node.name));
    $$('.segmented button').forEach(item => item.classList.toggle("active", item === button));
    renderNodes();
  }));
  $("#experimentGroupSelect").addEventListener("change", event => {
    const group = state.experimentGroups.find(item => item.experiment_id === event.currentTarget.value);
    if (!group) return;
    $("#experimentName").value = group.name;
    applyConfig(group.default_config || {}, false);
    $("#resultExperimentFilter").value = group.experiment_id;
    renderRuns();
  });
  $("#resultExperimentFilter").addEventListener("change", renderRuns);
  $("#experimentForm").addEventListener("input", updateFormMirrors);
  $("#modelSearchInput").addEventListener("input", renderModelPicker);
  $("#modelChecklist").addEventListener("change", event => {
    const input = event.target.closest("[data-model-option]");
    if (!input) return;
    setSelectedModels(input.checked ? [...selectedModelIds(), input.dataset.modelOption] : selectedModelIds().filter(id => id !== input.dataset.modelOption));
  });
  $("#modelChips").addEventListener("click", event => {
    const button = event.target.closest("[data-remove-model]");
    if (button) setSelectedModels(selectedModelIds().filter(id => id !== button.dataset.removeModel));
  });
  $("#selectAllModelsButton").addEventListener("click", () => setSelectedModels([...selectedModelIds(), ...$$('[data-model-option]').map(input => input.dataset.modelOption)]));
  $("#clearModelsButton").addEventListener("click", () => setSelectedModels([]));
  $("#experimentForm").addEventListener("submit", async event => {
    event.preventDefault();
    const participatingNodes = plannedNodeNames();
    const missingOffline = participatingNodes.filter(name => !statusFor(name).api);
    if (missingOffline.length) return toast("노드 API 오프라인", `${missingOffline.join(", ")} 서버를 먼저 시작하세요.`, "error");
    try {
      const data = await api("/api/experiments", { method: "POST", body: experimentPayload() });
      $("#consoleLog").innerHTML = "";
      logLine("START", `${data.experiment.name} · ${data.experiment.nodes.join(", ")}`);
      setRunState(data.experiment);
      await refreshExperimentData();
      $("#experimentGroupSelect").value = data.definition.experiment_id;
      $("#resultExperimentFilter").value = data.definition.experiment_id;
      renderRuns();
    } catch (error) { toast("실험 시작 실패", error.message, "error"); }
  });
  $("#cancelButton").addEventListener("click", async () => {
    try { const data = await api("/api/experiments/cancel", { method: "POST" }); setRunState(data.experiment); }
    catch (error) { toast("취소 실패", error.message, "error"); }
  });
  const observer = new IntersectionObserver(entries => entries.forEach(entry => {
    if (entry.isIntersecting) $$('.nav-link').forEach(link => link.classList.toggle("active", link.dataset.section === entry.target.id));
  }), { rootMargin: "-30% 0px -60%" });
  $$('.section').forEach(section => observer.observe(section));
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { renderRuns(); renderNodeDetail(); }, 140);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  if ("scrollRestoration" in history) history.scrollRestoration = "manual";
  state.token = getToken();
  bindEvents();
  bootstrap();
});
