/* Local conversation UI with durable, per-conversation context. */
(function () {
  "use strict";

  const form = document.getElementById("form");
  const question = document.getElementById("question");
  const submit = document.getElementById("submit");
  const newChat = document.getElementById("new-chat");
  const welcome = document.getElementById("welcome");
  const messages = document.getElementById("messages");
  const conversation = document.getElementById("conversation");
  const menuToggle = document.getElementById("menu-toggle");
  const backdrop = document.getElementById("sidebar-backdrop");
  const historyList = document.getElementById("history-list");
  const historyMore = document.getElementById("history-more");
  const historyMessage = document.getElementById("history-message");
  let dataReady = false;
  let historyReady = false;
  let busy = false;
  let conversationId = null;
  let selectedRegionId = localStorage.getItem("maritimeAgentRegionId") || "";
  let baseCatalog = null;
  let regionDatasets = [];
  let historyOffset = 0;
  const historyPageSize = 40;
  const memoryKey = "maritimeAgentConversationId";
  const sourceSyncKey = "maritimeDataSourcesChanged";

  function updateSubmit() {
    submit.disabled = !dataReady || !historyReady || busy || !question.value.trim();
    newChat.disabled = busy || !historyReady;
  }

  function resizeQuestion() {
    question.style.height = "auto";
    question.style.height = Math.min(question.scrollHeight, 180) + "px";
  }

  function scrollToLatest() {
    conversation.scrollTop = conversation.scrollHeight;
  }

  function closeSidebar() {
    document.body.classList.remove("sidebar-open");
    backdrop.hidden = true;
    menuToggle.setAttribute("aria-expanded", "false");
  }

  function formatHour(value) {
    return value ? value.replace("T", " ").slice(0, 16) : "—";
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function renderSelectedSource() {
    const dataset = regionDatasets.find(item => item.region_id === selectedRegionId);
    const deleteButton = document.getElementById("delete-dataset");
    const actionStatus = document.getElementById("dataset-action-status");
    actionStatus.hidden = true;
    actionStatus.textContent = "";
    if (dataset) {
      document.getElementById("dataset-name").textContent = dataset.canonical_name;
      document.getElementById("dataset-id").textContent = dataset.region_id;
      document.getElementById("dataset-range").textContent = dataset.first_utc ? `${formatHour(dataset.first_utc)} — ${formatHour(dataset.last_utc)}` : "暂无统一时间范围";
      document.getElementById("stat-hours").textContent = dataset.source_count.toLocaleString();
      document.getElementById("stat-zones").textContent = dataset.row_count.toLocaleString();
      document.getElementById("stat-events").textContent = dataset.kind_count.toLocaleString();
      document.getElementById("stat-hours-label").textContent = "接入来源";
      document.getElementById("stat-zones-label").textContent = "记录/要素";
      document.getElementById("stat-events-label").textContent = "数据类型";
      document.getElementById("dataset-state").textContent = "多源研究数据集";
      deleteButton.hidden = false;
    } else if (baseCatalog) {
      selectedRegionId = "";
      localStorage.removeItem("maritimeAgentRegionId");
      document.getElementById("dataset-name").textContent = baseCatalog.dataset || baseCatalog.dataset_id || "未命名数据集";
      document.getElementById("dataset-id").textContent = baseCatalog.dataset_id || "";
      document.getElementById("dataset-range").textContent = (baseCatalog.time_range_utc || []).map(formatHour).join(" — ") || "暂无时间范围";
      document.getElementById("stat-hours").textContent = baseCatalog.port_hour_rows ?? "—";
      document.getElementById("stat-zones").textContent = baseCatalog.zones?.length ?? "—";
      document.getElementById("stat-events").textContent = baseCatalog.confirmed_event_count ?? "—";
      document.getElementById("stat-hours-label").textContent = "港口小时";
      document.getElementById("stat-zones-label").textContent = "分区";
      document.getElementById("stat-events-label").textContent = "确认事件";
      document.getElementById("dataset-state").textContent = "数据库已连接";
      deleteButton.hidden = true;
    }
  }

  async function loadCatalog() {
    const state = document.getElementById("dataset-state");
    const status = document.getElementById("topbar-status");
    try {
      const [response,regionResponse,sourceResponse] = await Promise.all([
        fetch("/catalog"),fetch("/regions"),fetch("/data-sources?limit=100")]);
      const data = await response.json();
      const regionData=await regionResponse.json();
      const sourceData=await sourceResponse.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      if (!regionResponse.ok) throw new Error(regionData.detail || `HTTP ${regionResponse.status}`);
      if (!sourceResponse.ok) throw new Error(sourceData.detail || `HTTP ${sourceResponse.status}`);
      const isSqlite = data.backend === "sqlite";
      baseCatalog = data;
      const allSources=sourceData.sources||[];
      regionDatasets=(regionData.regions||[]).map(region=>{
        const items=allSources.filter(source=>source.region_id===region.region_id);
        const starts=items.map(item=>item.first_utc).filter(Boolean).sort();
        const ends=items.map(item=>item.last_utc).filter(Boolean).sort();
        return {...region,source_count:items.length,row_count:items.reduce((sum,item)=>sum+Number(item.row_count_total||0),0),kind_count:new Set(items.map(item=>item.kind)).size,first_utc:starts[0]||null,last_utc:ends.at(-1)||null};
      }).filter(region=>region.source_count>0);
      const selector = document.getElementById("dataset-selector");
      selector.replaceChildren();
      const baseOption = document.createElement("option");
      baseOption.value = ""; baseOption.textContent = `${data.dataset || data.dataset_id || "Houston"} · 基础数据集`; selector.append(baseOption);
      for (const dataset of regionDatasets) {
        const option = document.createElement("option"); option.value = dataset.region_id;
        option.textContent = `${dataset.canonical_name} · ${dataset.source_count} 个来源 · ${dataset.kind_count} 类数据`;
        selector.append(option);
      }
      selector.value = regionDatasets.some(item => item.region_id === selectedRegionId) ? selectedRegionId : "";
      selectedRegionId = selector.value;
      renderSelectedSource();
      state.textContent = selectedRegionId ? "多源研究数据集" : isSqlite ? "数据库已连接" : "CSV 数据已连接";
      status.lastElementChild.textContent = isSqlite ? "SQLite 已连接" : "CSV 已连接";
      document.getElementById("storage-footer").textContent = isSqlite ? "本地只读数据 · SQLite" : "本地只读数据 · CSV";
      dataReady = true;
    } catch (error) {
      state.textContent = "数据集不可用";
      status.lastElementChild.textContent = "数据不可用";
      document.getElementById("dataset-range").textContent = String(error.message || error);
      document.getElementById("dataset-card").classList.add("is-error");
      status.classList.add("is-error");
      dataReady = false;
    }
    updateSubmit();
  }

  document.getElementById("dataset-selector").addEventListener("change", event => {
    selectedRegionId = event.target.value;
    if (selectedRegionId) localStorage.setItem("maritimeAgentRegionId", selectedRegionId);
    else localStorage.removeItem("maritimeAgentRegionId");
    // A dataset switch is a new analysis scope.  Start a fresh visible
    // conversation so earlier Houston answers are not mistaken for evidence
    // about the newly selected region.
    conversationId = null;
    try { localStorage.removeItem(memoryKey); } catch (_) { /* Ignore. */ }
    messages.replaceChildren();
    welcome.hidden = false;
    markActiveHistory();
    renderSelectedSource();
  });
  window.addEventListener("storage",event=>{if(event.key===sourceSyncKey)loadCatalog()});

  function addUserMessage(text) {
    const article = document.createElement("article");
    article.className = "message message-user";
    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = text;
    article.append(body);
    messages.append(article);
  }

  function createLiveThoughtTracker(container, userQuestion = "") {
    const startTime = performance.now();
    const thoughtBox = document.createElement("div");
    thoughtBox.className = "thought-container is-thinking";

    const header = document.createElement("div");
    header.className = "thought-header";

    const headerLeft = document.createElement("div");
    headerLeft.className = "thought-header-left";

    const spinner = document.createElement("span");
    spinner.className = "thinking-spinner";

    const title = document.createElement("span");
    title.className = "thought-title";
    title.textContent = "深度思考中...";

    const timer = document.createElement("span");
    timer.className = "thought-time";
    timer.textContent = "0.0s";

    headerLeft.append(spinner, title, timer);
    header.append(headerLeft);

    const body = document.createElement("div");
    body.className = "thought-body";

    const cursor = document.createElement("span");
    cursor.className = "thinking-cursor";
    cursor.textContent = "●";

    const streamMsg = document.createElement("span");
    streamMsg.className = "thinking-stream-msg";

    const q = (userQuestion || "").toLowerCase();
    let steps = [
      "正在调用 search_ingested_data 检索海区数据资产与时空边界...",
      "正在调用专业分析工具链执行空间与动力学校准...",
      "正在提取可信离散观测报文并核算特征...",
      "正在核验证据账本（Verifier）并执行防幻觉审查..."
    ];
    if (q.includes("全维度") || q.includes("深度任务") || q.includes("交叉核验") || q.includes("多源模态") || (q.includes("航迹") && q.includes("会遇")) || (q.includes("走锚") && q.includes("遥感"))) {
      steps = [
        "正在调度底层穿透引擎 get_target_vessel_dossier 调取全维度时空态势档案...",
        "正在核算 3,200+ 条逐报文 AIS 链路完整性、15分钟缺口与 SOG 极大/均值...",
        "正在扫描局部切平面相对速度矢量与 COLREGs 1972 极限逼近会遇态势...",
        "正在交叉核验 Sentinel-1 遥感雷达切片、133米匹配与海洋气象水文场...",
        "正在执行 Verifier 独立证据对账与全要素结构化态势呈报..."
      ];
    } else if (q.includes("数据") || q.includes("覆盖") || q.includes("有哪些") || /20\d{2}[-/.]\d{1,2}/.test(q)) {
      steps = [
        "正在调用 search_ingested_data 扫描海区多源情报资产目录...",
        "正在调取 AIS 轨迹、气象水文与卫星遥感数据源覆盖元数据...",
        "正在比对目标日期与各模态有效观测时间窗口...",
        "正在核验证据账本凭证并生成多源数据覆盖清单..."
      ];
    } else if (q.includes("走锚") || q.includes("漂移") || q.includes("停留") || q.includes("航迹") || q.includes("轨迹")) {
      steps = [
        "正在解析目标船舶 MMSI 与时间，调用 search_ingested_data 定位数据源...",
        "正在调用 get_ingested_vessel_track 调取逐报文离散经纬度与 SOG 观测点...",
        "正在核算航速分布、转向连续性与锚泊流体力学受力比...",
        "正在核验证据账本（Verifier）并组织高精细航迹与走锚排查报告..."
      ];
    } else if (q.includes("换") || q.includes("另一艘") || q.includes("再换")) {
      steps = [
        "正在核对会话历史已研判船舶，调用 search_ingested_tracks 轮转活跃船舶...",
        "正在调用 get_ingested_vessel_track 获取新目标逐报文清洗轨迹...",
        "正在核算该船动力学 SOG 航速谱与流体锚泊工况...",
        "正在核验证据账本（Verifier）并生成换船比对结论..."
      ];
    } else if (q.includes("最近") || q.includes("多远") || q.includes("会遇") || q.includes("cpa") || q.includes("碰撞") || q.includes("避碰")) {
      steps = [
        "正在调用 query_vessels_snapshot 扫描全天各小时船舶时空分布快照...",
        "正在调用 assess_collision_risk 解算局部切平面相对速度矢量与 DCPA/TCPA...",
        "正在根据 COLREGs 1972 国际海事规则判定会遇态势与避碰责任...",
        "正在核验证据账本凭据并输出极限最近距离与量化相遇态势..."
      ];
    } else if (q.includes("变化") || q.includes("对比") || q.includes("流量") || q.includes("日变化")) {
      steps = [
        "正在调用 search_ingested_data 定位基准日与目标日海区数据源...",
        "正在调用 ingested_ais_daily_change 核算小时级业务流量与活跃船数...",
        "正在对比移动船舶（SOG > 0.5 节）与停泊船舶动态流动比率...",
        "正在核验证据账本并生成结构化业务日变化对比报告..."
      ];
    }

    streamMsg.textContent = steps[0];
    body.append(cursor, streamMsg);

    const agentPanel = document.createElement("div");
    agentPanel.className = "agent-team-panel";
    agentPanel.hidden = true;

    const agentHeader = document.createElement("div");
    agentHeader.className = "agent-team-header";
    agentHeader.textContent = "多智能体协同研判团队";
    const agentBadges = document.createElement("div");
    agentBadges.className = "agent-badges";
    agentPanel.append(agentHeader, agentBadges);
    body.append(agentPanel);

    thoughtBox.append(header, body);
    container.replaceChildren(thoughtBox);

    let hasCustomMessage = false;
    const intervalId = setInterval(() => {
      const elapsed = (performance.now() - startTime) / 1000;
      timer.textContent = elapsed.toFixed(1) + "s";

      if (!hasCustomMessage) {
        if (elapsed > 3.0) {
          streamMsg.textContent = steps[3] || steps[steps.length - 1];
        } else if (elapsed > 1.8) {
          streamMsg.textContent = steps[2] || steps[1];
        } else if (elapsed > 0.8) {
          streamMsg.textContent = steps[1] || steps[0];
        }
      }
    }, 100);

    const badgeMap = new Map();
    const roleIcons = {
      "ais": "⚓",
      "gis": "🗺️",
      "events": "📢",
      "remote_sensing": "🛰️",
      "intake": "📥",
      "synthesizer": "🧠",
      "verifier": "🛡️",
    };

    return {
      updateMessage(text) {
        if (text) {
          hasCustomMessage = true;
          streamMsg.textContent = text;
        }
      },
      renderAgentTeam(roles, roleNames) {
        if (!roles || !roles.length) return;
        agentPanel.hidden = false;
        agentBadges.replaceChildren();
        badgeMap.clear();

        roles.forEach((role, idx) => {
          const name = (roleNames && roleNames[idx]) ? roleNames[idx] : role;
          const badge = document.createElement("div");
          badge.className = "agent-badge is-running";
          const icon = document.createElement("span");
          icon.textContent = roleIcons[role] || "🤖";
          const label = document.createElement("span");
          label.textContent = name;
          const status = document.createElement("span");
          status.className = "badge-status";
          status.textContent = "· 正在调取数据";
          badge.append(icon, label, status);
          agentBadges.append(badge);
          badgeMap.set(role, { badge, status, label });
        });
      },
      updateAgentStatus(role, statusText, isDone = false) {
        const item = badgeMap.get(role);
        if (item) {
          item.status.textContent = statusText ? `· ${statusText}` : "";
          if (isDone) {
            item.badge.classList.remove("is-running");
            item.badge.classList.add("is-done");
          }
        }
      },
      stop() {
        clearInterval(intervalId);
        return (performance.now() - startTime) / 1000;
      }
    };
  }

  function renderCompletedThought(container, data, elapsedSeconds = null) {
    if (!container) return;
    const thoughtText = (data.thought || data.reasoning_content || "").trim();
    if (!thoughtText) {
      container.replaceChildren();
      return;
    }

    const thoughtBox = document.createElement("div");
    thoughtBox.className = "thought-container";

    const header = document.createElement("div");
    header.className = "thought-header";
    header.setAttribute("role", "button");
    header.setAttribute("tabindex", "0");
    header.setAttribute("aria-expanded", "true");

    const headerLeft = document.createElement("div");
    headerLeft.className = "thought-header-left";

    const icon = document.createElement("span");
    icon.className = "thought-icon";
    icon.textContent = "💭";

    const title = document.createElement("span");
    title.className = "thought-title";
    title.textContent = "思考过程";

    const time = document.createElement("span");
    time.className = "thought-time";
    time.textContent = elapsedSeconds ? `· 耗时 ${elapsedSeconds.toFixed(1)} 秒` : "";

    headerLeft.append(icon, title, time);

    const toggleIndicator = document.createElement("span");
    toggleIndicator.className = "thought-toggle-indicator";
    toggleIndicator.textContent = "▲ 收起";

    header.append(headerLeft, toggleIndicator);

    const body = document.createElement("div");
    body.className = "thought-body";
    if (typeof window.renderAnswerMarkdown === "function") {
      window.renderAnswerMarkdown(body, thoughtText);
    } else {
      body.textContent = thoughtText;
    }

    let isExpanded = true;
    function toggle() {
      isExpanded = !isExpanded;
      body.style.display = isExpanded ? "block" : "none";
      toggleIndicator.textContent = isExpanded ? "▲ 收起" : "▼ 展开";
      header.setAttribute("aria-expanded", String(isExpanded));
    }

    header.addEventListener("click", toggle);
    header.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggle();
      }
    });

    thoughtBox.append(header, body);
    container.replaceChildren(thoughtBox);
  }

  function addAssistantMessage(showLoading = true, userQuestion = "") {
    const article = document.createElement("article");
    article.className = "message message-assistant";
    const avatar = document.createElement("div");
    avatar.className = "assistant-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "海";
    const content = document.createElement("div");
    content.className = "assistant-content";
    const name = document.createElement("div");
    name.className = "assistant-name";
    name.textContent = "海事态势智能体";

    // 思考过程卡片容器（置于正式回答的上方）
    const thinkingBlock = document.createElement("div");
    thinkingBlock.className = "thinking-block";

    const answer = document.createElement("div");
    answer.className = "answer";
    answer.setAttribute("aria-live", "polite");

    let liveTracker = null;
    if (showLoading) {
      liveTracker = createLiveThoughtTracker(thinkingBlock, userQuestion);
    }

    content.append(name, thinkingBlock, answer);
    article.append(avatar, content);
    messages.append(article);
    return { answer, content, thinkingBlock, liveTracker };
  }

  function addEvidence(content, data) {
    const evidence = Array.isArray(data.evidence) ? data.evidence : [];
    const trace = Array.isArray(data.tool_trace) ? data.tool_trace : [];
    const details = document.createElement("details");
    details.className = "evidence-panel";
    const summary = document.createElement("summary");
    summary.textContent = `查看证据与工具调用 · ${evidence.length} 条证据 / ${trace.length} 次调用`;
    const grid = document.createElement("div");
    grid.className = "evidence-grid";
    const citationNumbers = new Map();
    const evidenceItems = new Map();
    const evidenceBox = document.createElement("div");
    evidenceBox.className = "evidence-box";
    const evidenceTitle = document.createElement("strong");
    evidenceTitle.textContent = "数据证据";
    const evidenceList = document.createElement("div");
    evidenceList.className = "evidence-list";
    evidence.forEach((item, index) => {
      const id = item?.evidence_id;
      const card = document.createElement("div");
      card.className = "evidence-item";
      const heading = document.createElement("div");
      heading.className = "evidence-item-heading";
      const badge = document.createElement("span");
      badge.className = "evidence-number";
      badge.textContent = `[${index + 1}]`;
      const label = document.createElement("code");
      label.className = "evidence-id";
      label.textContent = typeof id === "string" ? id : "未编号证据";
      heading.append(badge, label);
      const code = document.createElement("pre");
      code.textContent = JSON.stringify(item, null, 2);
      card.append(heading, code);
      evidenceList.append(card);
      if (typeof id === "string" && !citationNumbers.has(id)) {
        citationNumbers.set(id, index + 1);
        evidenceItems.set(id, card);
      }
    });
    if (!evidence.length) evidenceList.textContent = "暂无数据证据";
    evidenceBox.append(evidenceTitle, evidenceList);
    const traceBox = document.createElement("div");
    traceBox.className = "evidence-box";
    const traceTitle = document.createElement("strong");
    traceTitle.textContent = "工具调用";
    const traceCode = document.createElement("pre");
    traceCode.textContent = JSON.stringify(trace, null, 2);
    traceBox.append(traceTitle, traceCode);
    grid.append(evidenceBox, traceBox);
    details.append(summary, grid);
    content.append(details);
    const warnings = Array.isArray(data.warnings) ? data.warnings.filter(Boolean) : [];
    if (warnings.length) {
      const note = document.createElement("p");
      note.className = "warning-note";
      note.textContent = warnings.join("；");
      content.append(note);
    }
    return {
      citationNumbers,
      onCitation(id) {
        const card = evidenceItems.get(id);
        if (!card) return;
        details.open = true;
        for (const selected of evidenceList.querySelectorAll(".is-highlighted")) {
          selected.classList.remove("is-highlighted");
        }
        card.classList.add("is-highlighted");
        card.scrollIntoView({ behavior: "smooth", block: "center" });
      },
    };
  }

  function addCognitivePanel(content, reasoningState) {
    if (!reasoningState) return;
    const facts = Array.isArray(reasoningState.established_facts) ? reasoningState.established_facts : [];
    const refuted = Array.isArray(reasoningState.refuted_hypotheses) ? reasoningState.refuted_hypotheses : [];
    const questions = Array.isArray(reasoningState.unresolved_questions) ? reasoningState.unresolved_questions : [];
    const entities = reasoningState.target_entities || {};

    const entityKeys = Object.keys(entities).filter(k => entities[k]);
    if (!facts.length && !refuted.length && !questions.length && !entityKeys.length) return;

    const details = document.createElement("details");
    details.className = "cognitive-panel";
    const summary = document.createElement("summary");
    const summaryParts = [];
    if (entities.mmsi) summaryParts.push(`船舶: ${entities.mmsi}`);
    if (entities.region_id) summaryParts.push(`海区: ${entities.region_id}`);
    if (facts.length) summaryParts.push(`${facts.length} 条已证事实`);
    if (refuted.length) summaryParts.push(`${refuted.length} 条排除假说`);
    summary.textContent = `🧠 态势认知黑板与思维链断点 · ${summaryParts.join(" · ") || "查看状态"}`;

    const container = document.createElement("div");
    container.className = "cognitive-content";

    if (entityKeys.length) {
      const entBox = document.createElement("div");
      entBox.className = "cognitive-entities";
      entityKeys.forEach(k => {
        const tag = document.createElement("span");
        tag.className = "cognitive-tag";
        tag.textContent = `${k}: ${entities[k]}`;
        entBox.append(tag);
      });
      container.append(entBox);
    }

    if (facts.length) {
      const title = document.createElement("div");
      title.className = "cognitive-section-title";
      title.textContent = "已证实态势公理 (Established Facts)";
      const ul = document.createElement("ul");
      ul.className = "cognitive-list";
      facts.forEach(f => {
        const li = document.createElement("li");
        li.textContent = f;
        ul.append(li);
      });
      container.append(title, ul);
    }

    if (refuted.length) {
      const title = document.createElement("div");
      title.className = "cognitive-section-title";
      title.textContent = "已排除假说 (Refuted Hypotheses)";
      const ul = document.createElement("ul");
      ul.className = "cognitive-list";
      refuted.forEach(r => {
        const li = document.createElement("li");
        li.textContent = r;
        ul.append(li);
      });
      container.append(title, ul);
    }

    if (questions.length) {
      const title = document.createElement("div");
      title.className = "cognitive-section-title";
      title.textContent = "未决疑点 / 待深挖盲区 (Unresolved Questions)";
      const ul = document.createElement("ul");
      ul.className = "cognitive-list";
      questions.forEach(q => {
        const li = document.createElement("li");
        li.textContent = q;
        ul.append(li);
      });
      container.append(title, ul);
    }

    details.append(summary, container);
    content.append(details);
  }

  function addFeedbackBar(content, data) {
    const bar = document.createElement("div");
    bar.className = "feedback-bar";

    const thumbBtn = document.createElement("button");
    thumbBtn.type = "button";
    thumbBtn.className = "feedback-btn";
    thumbBtn.textContent = "👍 研判准确";
    thumbBtn.onclick = () => {
      thumbBtn.textContent = "✔ 已反馈准确";
      thumbBtn.disabled = true;
    };

    const critiqueBtn = document.createElement("button");
    critiqueBtn.type = "button";
    critiqueBtn.className = "feedback-btn";
    critiqueBtn.textContent = "💡 指出误区 / 纠偏自进化";

    const critiqueBox = document.createElement("div");
    critiqueBox.className = "critique-box";
    critiqueBox.hidden = true;

    const input = document.createElement("input");
    input.type = "text";
    input.className = "critique-input";
    input.placeholder = "例如：对于气象风速，请同时给出节(kts)和米每秒(m/s)双单位换算...";

    const submitBtn = document.createElement("button");
    submitBtn.type = "button";
    submitBtn.className = "critique-submit-btn";
    submitBtn.textContent = "沉淀自进化规约";

    const statusMsg = document.createElement("span");
    statusMsg.className = "critique-status-msg";

    critiqueBtn.onclick = () => {
      critiqueBox.hidden = !critiqueBox.hidden;
      if (!critiqueBox.hidden) input.focus();
    };

    submitBtn.onclick = async () => {
      const lesson = input.value.trim();
      if (!lesson) return;
      submitBtn.disabled = true;
      submitBtn.textContent = "沉淀中...";
      try {
        const res = await fetch("/api/memory/critiques", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            domain: "general",
            trigger_pattern: lesson,
            failed_path: "用户人工反馈纠偏",
            corrected_lesson: lesson,
            confidence: 1.0,
          }),
        });
        if (!res.ok) throw new Error("保存失败");
        statusMsg.textContent = "✨ 规约已沉淀入自进化知识库，后续将自动生效！";
        input.value = "";
        submitBtn.hidden = true;
      } catch (err) {
        statusMsg.textContent = "保存异常: " + (err.message || err);
        submitBtn.disabled = false;
        submitBtn.textContent = "沉淀自进化规约";
      }
    };

    critiqueBox.append(input, submitBtn, statusMsg);
    bar.append(thumbBtn, critiqueBtn, critiqueBox);
    content.append(bar);
  }

  function showAssistantReply(assistant, data, durationSeconds = null) {
    if (assistant.liveTracker) {
      const realDuration = assistant.liveTracker.stop();
      if (durationSeconds === null) durationSeconds = realDuration;
    }
    renderCompletedThought(assistant.thinkingBlock, data, durationSeconds);
    const citations = addEvidence(assistant.content, data);
    addCognitivePanel(assistant.content, data.reasoning_state);
    if (typeof window.renderAnswerMarkdown === "function") {
      window.renderAnswerMarkdown(assistant.answer, data.answer, citations);
    } else {
      assistant.answer.textContent = data.answer || "";
    }
    if (typeof window.renderAnswerVisualizations === "function") {
      window.renderAnswerVisualizations(assistant.answer, data);
    }
    addFeedbackBar(assistant.content, data);
  }

  function showConversation(data) {
    messages.replaceChildren();
    const turns = Array.isArray(data.turns) ? data.turns : [];
    for (const turn of turns) {
      addUserMessage(turn.question);
      showAssistantReply(addAssistantMessage(false), turn);
    }
    welcome.hidden = turns.length > 0;
    scrollToLatest();
  }

  function markActiveHistory() {
    for (const button of historyList.querySelectorAll(".history-item")) {
      const active = button.dataset.conversationId === conversationId;
      button.classList.toggle("is-active", active);
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    }
  }

  async function loadHistory(reset = false) {
    if (reset) {
      historyOffset = 0;
      historyList.replaceChildren();
    }
    historyMore.disabled = true;
    historyMessage.hidden = true;
    try {
      const response = await fetch(`/conversations?limit=${historyPageSize}&offset=${historyOffset}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      for (const item of data.conversations || []) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "history-item";
        button.dataset.conversationId = item.conversation_id;
        button.title = item.title || "未命名对话";
        const title = document.createElement("span");
        title.className = "history-item-title";
        title.textContent = item.title || "未命名对话";
        const meta = document.createElement("span");
        meta.className = "history-item-meta";
        const when = new Date(item.updated_at_utc);
        meta.textContent = `${Number.isNaN(when.getTime()) ? "" : when.toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })} · ${item.turn_count} 轮`;
        button.append(title, meta);
        button.addEventListener("click", () => openConversation(item.conversation_id));
        historyList.append(button);
      }
      historyOffset += (data.conversations || []).length;
      historyMore.hidden = historyOffset >= (data.total || 0);
      if (!historyOffset) {
        const empty = document.createElement("p");
        empty.className = "history-empty";
        empty.textContent = "暂无历史对话";
        historyList.append(empty);
      }
      markActiveHistory();
    } catch (_) {
      historyMessage.textContent = "历史记录暂时无法读取";
      historyMessage.hidden = false;
      historyMore.hidden = true;
    } finally {
      historyMore.disabled = false;
    }
  }

  async function openConversation(id) {
    if (busy || !historyReady || id === conversationId) return;
    historyReady = false;
    updateSubmit();
    historyMessage.hidden = true;
    try {
      const response = await fetch(`/conversations/${encodeURIComponent(id)}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      conversationId = data.conversation_id;
      try { localStorage.setItem(memoryKey, conversationId); } catch (_) { /* Ignore. */ }
      showConversation(data);
      question.value = "";
      resizeQuestion();
      markActiveHistory();
      closeSidebar();
    } catch (_) {
      historyMessage.textContent = "无法打开这条对话，请稍后重试";
      historyMessage.hidden = false;
    } finally {
      historyReady = true;
      updateSubmit();
    }
  }

  async function restoreConversation() {
    let saved = null;
    try { saved = localStorage.getItem(memoryKey); } catch (_) { /* Storage may be disabled. */ }
    if (!saved) {
      historyReady = true;
      updateSubmit();
      return;
    }
    try {
      const response = await fetch(`/conversations/${encodeURIComponent(saved)}`);
      if (response.status === 404 || response.status === 422) {
        try { localStorage.removeItem(memoryKey); } catch (_) { /* Ignore. */ }
        return;
      }
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      conversationId = data.conversation_id;
      showConversation(data);
    } catch (_) {
      // Keep the saved ID so a temporary server outage does not erase the session.
      conversationId = saved;
    } finally {
      historyReady = true;
      updateSubmit();
    }
  }

  async function submitQuestion() {
    const text = question.value.trim();
    if (!text || busy || !dataReady || !historyReady) return;
    busy = true;
    updateSubmit();
    welcome.hidden = true;
    question.value = "";
    resizeQuestion();
    addUserMessage(text);
    const submitStart = performance.now();
    const assistant = addAssistantMessage(true, text);
    scrollToLatest();

    let accumulatedAnswer = "";
    let finalDoneData = null;

    try {
      const response = await fetch("/ask/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream"
        },
        body: JSON.stringify({
          question: text,
          ...(conversationId ? { conversation_id: conversationId } : {}),
          ...(selectedRegionId ? { region_id: selectedRegionId } : {})
        }),
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `HTTP ${response.status}`);
      }

      if (!response.body) {
        throw new Error("浏览器环境不支持流式响应");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();

        for (const part of parts) {
          const trimmed = part.trim();
          if (!trimmed) continue;

          let eventType = "message";
          let eventDataText = "";

          const lines = trimmed.split("\n");
          for (const line of lines) {
            if (line.startsWith("event:")) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              eventDataText = line.slice(5).trim();
            }
          }

          if (!eventDataText) continue;
          let eventData = {};
          try {
            eventData = JSON.parse(eventDataText);
          } catch (_) {
            eventData = { raw: eventDataText };
          }

          if (eventType === "error") {
            throw new Error(eventData.detail || "服务端分析失败");
          } else if (eventType === "routing") {
            const rolesStr = (eventData.role_names || []).join("、");
            if (assistant.liveTracker) {
              if (eventData.roles && eventData.roles.length) {
                assistant.liveTracker.renderAgentTeam(eventData.roles, eventData.role_names);
              }
              if (rolesStr) {
                assistant.liveTracker.updateMessage(`已调度专业分析团队：【${rolesStr}】`);
              }
            }
          } else if (eventType === "specialist_start") {
            if (assistant.liveTracker) {
              assistant.liveTracker.updateAgentStatus(eventData.role, "正在检索情报...", false);
              if (eventData.message) {
                assistant.liveTracker.updateMessage(eventData.message);
              }
            }
          } else if (eventType === "specialist_done") {
            if (assistant.liveTracker) {
              const count = eventData.evidence_count != null ? eventData.evidence_count : 0;
              assistant.liveTracker.updateAgentStatus(eventData.role, `✔ 已获 ${count} 项凭据`, true);
              if (eventData.message) {
                assistant.liveTracker.updateMessage(eventData.message);
              }
            }
          } else if (eventType === "status") {
            if (eventData.message && assistant.liveTracker) {
              assistant.liveTracker.updateMessage(eventData.message);
            }
          } else if (eventType === "token") {
            if (eventData.delta) {
              accumulatedAnswer += eventData.delta;
              assistant.answer.innerHTML = escapeHtml(accumulatedAnswer) + '<span class="typing-cursor"></span>';
              scrollToLatest();
            }
          } else if (eventType === "verifier") {
            if (eventData.message && assistant.liveTracker) {
              assistant.liveTracker.updateMessage(eventData.message);
            }
          } else if (eventType === "done") {
            finalDoneData = eventData;
          }
        }
      }

      const durationSeconds = (performance.now() - submitStart) / 1000;
      if (finalDoneData) {
        conversationId = finalDoneData.conversation_id;
        try { localStorage.setItem(memoryKey, conversationId); } catch (_) { /* Ignore. */ }
        if (assistant.liveTracker) assistant.liveTracker.stop();
        showAssistantReply(assistant, finalDoneData, durationSeconds);
        await loadHistory(true);
      } else if (accumulatedAnswer) {
        if (assistant.liveTracker) assistant.liveTracker.stop();
        assistant.answer.innerHTML = escapeHtml(accumulatedAnswer);
      }
    } catch (error) {
      if (assistant.liveTracker) assistant.liveTracker.stop();
      assistant.answer.classList.add("is-error");
      assistant.answer.textContent = `请求失败：${error.message || error}`;
    } finally {
      busy = false;
      updateSubmit();
      scrollToLatest();
      question.focus();
    }
  }

  form.addEventListener("submit", event => {
    event.preventDefault();
    submitQuestion();
  });
  question.addEventListener("input", () => { resizeQuestion(); updateSubmit(); });
  question.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submitQuestion();
    }
  });
  document.querySelectorAll("[data-question]").forEach(button => {
    button.addEventListener("click", () => {
      question.value = button.dataset.question;
      resizeQuestion();
      updateSubmit();
      question.focus();
    });
  });
  newChat.addEventListener("click", () => {
    if (busy || !historyReady) return;
    conversationId = null;
    try { localStorage.removeItem(memoryKey); } catch (_) { /* Ignore. */ }
    messages.replaceChildren();
    welcome.hidden = false;
    question.value = "";
    resizeQuestion();
    updateSubmit();
    closeSidebar();
    question.focus();
    conversation.scrollTop = 0;
    markActiveHistory();
  });
  historyMore.addEventListener("click", () => loadHistory(false));
  menuToggle.addEventListener("click", () => {
    const open = !document.body.classList.contains("sidebar-open");
    document.body.classList.toggle("sidebar-open", open);
    backdrop.hidden = !open;
    menuToggle.setAttribute("aria-expanded", String(open));
  });
  backdrop.addEventListener("click", closeSidebar);
  document.addEventListener("keydown", event => { if (event.key === "Escape") closeSidebar(); });

  const openCritiquesBtn = document.getElementById("open-critiques-btn");
  const closeCritiquesBtn = document.getElementById("close-critiques-btn");
  const critiquesModal = document.getElementById("critiques-modal");
  const critiquesList = document.getElementById("critiques-list");

  async function loadCritiques() {
    critiquesList.replaceChildren();
    const loading = document.createElement("p");
    loading.style.cssText = "text-align:center;color:#94a3b8;font-size:13px";
    loading.textContent = "正在读取自进化反思库…";
    critiquesList.append(loading);
    try {
      const res = await fetch("/api/memory/critiques?limit=50");
      const data = await res.json();
      const list = data.critiques || [];
      critiquesList.replaceChildren();
      if (!list.length) {
        const empty = document.createElement("p");
        empty.style.cssText = "text-align:center;color:#94a3b8;font-size:13px;padding:20px 0";
        empty.textContent = "暂无自进化规则。当你在对话中纠偏或点击“纠偏自进化”时，系统将自动沉淀经验。";
        critiquesList.append(empty);
        return;
      }
      list.forEach(c => {
        const card = document.createElement("div");
        card.className = "critique-card";

        const top = document.createElement("div");
        top.className = "critique-card-top";
        const badge = document.createElement("span");
        badge.className = "critique-domain-badge";
        badge.textContent = c.domain || "general";
        const conf = document.createElement("span");
        conf.style.cssText = "font-size:11px;color:#708b88";
        conf.textContent = `置信度: ${Number(c.confidence || 1).toFixed(1)}`;
        top.append(badge, conf);

        const rule = document.createElement("div");
        rule.className = "critique-rule-text";
        rule.textContent = c.corrected_lesson;

        const trig = document.createElement("div");
        trig.className = "critique-trigger-text";
        trig.textContent = `触发语义: ${c.trigger_pattern}`;

        const actions = document.createElement("div");
        actions.className = "critique-actions";

        const upBtn = document.createElement("button");
        upBtn.type = "button";
        upBtn.className = "critique-action-btn";
        upBtn.textContent = "👍 赞成 (+0.2)";
        upBtn.onclick = async () => {
          await fetch(`/api/memory/critiques/${c.critique_id}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "upvote" }),
          });
          loadCritiques();
        };

        const downBtn = document.createElement("button");
        downBtn.type = "button";
        downBtn.className = "critique-action-btn";
        downBtn.textContent = "👎 踩 (-0.3)";
        downBtn.onclick = async () => {
          await fetch(`/api/memory/critiques/${c.critique_id}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "downvote" }),
          });
          loadCritiques();
        };

        const delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.className = "critique-action-btn critique-delete-btn";
        delBtn.textContent = "🗑 删除";
        delBtn.onclick = async () => {
          if (!confirm("确认删除这条自进化规则？")) return;
          await fetch(`/api/memory/critiques/${c.critique_id}`, { method: "DELETE" });
          loadCritiques();
        };

        actions.append(upBtn, downBtn, delBtn);
        card.append(top, rule, trig, actions);
        critiquesList.append(card);
      });
    } catch (err) {
      critiquesList.replaceChildren();
      const errEl = document.createElement("p");
      errEl.style.cssText = "color:#a0473d;font-size:13px";
      errEl.textContent = "加载失败: " + (err.message || err);
      critiquesList.append(errEl);
    }
  }

  if (openCritiquesBtn && critiquesModal) {
    openCritiquesBtn.addEventListener("click", () => {
      critiquesModal.showModal();
      loadCritiques();
    });
  }
  if (closeCritiquesBtn && critiquesModal) {
    closeCritiquesBtn.addEventListener("click", () => {
      critiquesModal.close();
    });
  }
  if (critiquesModal) {
    critiquesModal.addEventListener("click", (e) => {
      if (e.target === critiquesModal) critiquesModal.close();
    });
  }

  loadCatalog();
  restoreConversation().then(() => loadHistory(true));
  updateSubmit();
})();
