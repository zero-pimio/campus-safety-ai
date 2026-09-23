"use strict";
(() => {
  const byId = id => document.getElementById(id);
  const player = byId("player");
  const playButton = byId("play-button");
  const seek = byId("seek");
  const statusLabels = {available: "可在线播放", local_only: "当前仅本地演示", design_pending: "设计阶段 · 待实现"};
  let catalog = null;
  let current = null;
  let selectionVersion = 0;

  function safeURL(value, local = false) {
    if (typeof value !== "string" || !value.trim()) return null;
    try {
      const url = new URL(value, document.baseURI);
      if (!["http:", "https:"].includes(url.protocol)) return null;
      if (local && url.origin !== location.origin) return null;
      return url.href;
    } catch { return null; }
  }

  function text(id, value) { byId(id).textContent = value || ""; }
  function clock(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
    const value = Math.floor(seconds);
    return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
  }
  function duration(clip) {
    const values = [];
    if (Number.isFinite(clip.duration_seconds)) values.push(`${Number(clip.duration_seconds.toFixed(2))} 秒`);
    if (Number.isInteger(clip.frames)) values.push(`${clip.frames} 帧`);
    return values.join(" · ");
  }
  function attributed(id, title, url, fallback) {
    const node = byId(id);
    node.replaceChildren();
    const href = safeURL(url);
    if (href) {
      const link = document.createElement("a");
      link.href = href;
      link.textContent = title || fallback;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      node.append(link);
    } else node.textContent = title || fallback;
  }
  function showError(message) {
    text("player-error", message);
    byId("player-error").hidden = false;
  }
  function updatePlayback() {
    playButton.textContent = player.paused || player.ended ? "播放" : "暂停";
    const total = Number.isFinite(player.duration) ? player.duration : 0;
    text("playback-time", `${clock(player.currentTime)} / ${clock(total)}`);
    seek.disabled = total <= 0;
    seek.max = total || 100;
    seek.value = Math.min(player.currentTime || 0, total);
    seek.setAttribute("aria-valuetext", `${clock(player.currentTime)}，共 ${clock(total)}`);
  }
  function hashId() {
    try { return decodeURIComponent(location.hash.slice(1)); } catch { return ""; }
  }
  function shareURL() {
    const url = new URL(location.href);
    url.hash = current ? encodeURIComponent(current.id) : "";
    return url.href;
  }

  function selectClip(clip, updateHash = true) {
    selectionVersion += 1;
    player.pause();
    player.removeAttribute("src");
    player.removeAttribute("poster");
    current = clip;
    byId("player-error").hidden = true;
    text("share-status", "");
    byId("share-button").disabled = false;
    text("clip-title", clip.title);
    text("clip-meta", clip.status === "available" ? duration(clip) : statusLabels[clip.status]);
    text("clip-description", clip.description);
    text("clip-badge", clip.badge || statusLabels[clip.status]);
    text("clip-limitation", clip.limitation || "尚无完整验收说明，请勿据此推断正式识别性能。");
    text("verification-scope", clip.status === "design_pending" ? "当前仅有设计说明，尚无真实运行结果或视频验收。" : "已有视频展示预先计算的结果。单段演示不能代表校园精度、设备实时性能或完整事件验收。");
    text("clip-model", clip.model || (clip.status === "design_pending" ? "尚未实现运行模块" : "见本片段说明"));
    attributed("clip-source", clip.source_title, clip.source_url, clip.status === "design_pending" ? "暂无运行素材" : "尚未提供来源说明");
    attributed("clip-license", clip.license_title, clip.license_url, "未声明公开再分发许可");
    byId("clip-details").hidden = false;
    const available = clip.status === "available";
    byId("video-stage").hidden = !available;
    byId("playback-controls").hidden = !available;
    byId("empty-stage").hidden = available;
    byId("download-link").hidden = !available;
    playButton.disabled = !available;
    if (available) {
      player.src = clip.src;
      if (clip.poster) player.poster = clip.poster;
      player.setAttribute("aria-label", `${clip.title}，实际识别回放`);
      byId("download-link").href = clip.src;
      byId("download-link").download = `${clip.id}.mp4`;
    } else {
      byId("download-link").removeAttribute("href");
      text("empty-title", statusLabels[clip.status]);
      text("empty-description", clip.limitation || clip.description || "该模块暂无可公开播放的视频。");
      const repository = safeURL(catalog.repository_url);
      const reproduce = byId("reproduce-link");
      reproduce.hidden = !repository;
      if (repository) {
        reproduce.href = `${repository.replace(/\/$/, "")}/blob/main/docs/demos/README.md`;
        reproduce.textContent = clip.status === "design_pending" ? "查看设计与交付进展 ↗" : "查看源码与复现说明 ↗";
      }
    }
    player.load();
    updatePlayback();
    document.querySelectorAll("[data-clip]").forEach(button => {
      const selected = button.dataset.clip === clip.id;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    if (updateHash) {
      const url = new URL(location.href);
      url.hash = encodeURIComponent(clip.id);
      history.replaceState(null, "", url);
    }
    byId("share-link").href = shareURL();
    byId("share-link").hidden = false;
  }

  function renderList() {
    const list = byId("clip-list");
    list.replaceChildren();
    catalog.videos.forEach((clip, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "clip-button";
      button.dataset.clip = clip.id;
      button.setAttribute("aria-controls", "viewer");
      button.setAttribute("aria-pressed", "false");
      const placeholder = document.createElement("span");
      placeholder.className = "clip-thumbnail clip-placeholder";
      placeholder.setAttribute("aria-hidden", "true");
      placeholder.textContent = clip.status === "design_pending" ? "◇" : "⌁";
      if (clip.poster) {
        const image = document.createElement("img");
        image.src = clip.poster;
        image.alt = "";
        image.loading = "lazy";
        image.className = "clip-thumbnail";
        image.addEventListener("error", () => image.replaceWith(placeholder), {once: true});
        button.append(image);
      } else button.append(placeholder);
      const label = document.createElement("span");
      label.className = "clip-text";
      const number = document.createElement("span");
      number.className = "clip-index";
      number.textContent = `${String(index + 1).padStart(2, "0")} ${clip.status === "available" ? duration(clip) : ""}`;
      const title = document.createElement("strong");
      title.textContent = clip.title;
      const status = document.createElement("span");
      status.className = `clip-status ${clip.status}`;
      status.textContent = statusLabels[clip.status];
      label.append(number, title, status);
      button.append(label);
      button.addEventListener("click", () => {
        byId("selection-notice").hidden = true;
        if (current?.id !== clip.id) selectClip(clip);
      });
      list.append(button);
    });
  }

  async function loadCatalog() {
    byId("catalog-error").hidden = true;
    byId("retry-button").disabled = true;
    byId("viewer").setAttribute("aria-busy", "true");
    const abort = new AbortController();
    const timeout = setTimeout(() => abort.abort(), 12000);
    try {
      const response = await fetch("./catalog.json", {signal: abort.signal});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const value = await response.json();
      if (!Array.isArray(value.videos) || !value.videos.length) throw new Error("清单没有可显示的条目");
      const ids = new Set();
      for (const clip of value.videos) {
        if (typeof clip.id !== "string" || !clip.id || ids.has(clip.id) || !clip.title || !Object.hasOwn(statusLabels, clip.status)) throw new Error("清单条目格式无效");
        ids.add(clip.id);
        const src = safeURL(clip.src, true);
        if (clip.status === "available" && !src) throw new Error("可播放条目缺少有效的本站视频地址");
        clip.src = clip.status === "available" ? src : null;
        clip.poster = safeURL(clip.poster, true);
      }
      catalog = value;
      text("available-count", String(value.videos.filter(clip => clip.status === "available").length).padStart(2, "0"));
      text("catalog-count", `${value.videos.length} 个条目`);
      text("updated-at", value.updated ? `更新于 ${value.updated}` : "");
      const repository = safeURL(value.repository_url);
      byId("repository-link").hidden = !repository;
      if (repository) byId("repository-link").href = repository;
      renderList();
      const requested = hashId();
      const matched = value.videos.find(clip => clip.id === requested);
      if (requested && !matched) {
        text("selection-notice", "链接中的片段未收录，已显示当前清单的首个可播放片段。");
        byId("selection-notice").hidden = false;
      }
      selectClip(matched || value.videos.find(clip => clip.status === "available") || value.videos[0]);
    } catch (error) {
      text("catalog-error-detail", location.protocol === "file:" ? "浏览器无法直接读取本地清单。请访问已发布的 GitHub Pages，或通过本地 HTTP 服务打开此目录。" : `无法读取 catalog.json（${error.name === "AbortError" ? "请求超时" : error.message}）。请检查网络后重试。`);
      byId("catalog-error").hidden = false;
      text("clip-title", "演示清单暂时不可用");
      text("empty-title", "加载未完成");
      text("empty-description", "请点击上方“重新加载”。视频不会自动播放。");
    } finally {
      clearTimeout(timeout);
      byId("retry-button").disabled = false;
      byId("viewer").setAttribute("aria-busy", "false");
    }
  }

  playButton.addEventListener("click", async () => {
    if (current?.status !== "available") return;
    if (!player.paused && !player.ended) { player.pause(); return; }
    const version = selectionVersion;
    try {
      byId("player-error").hidden = true;
      if (player.ended) player.currentTime = 0;
      await player.play();
    } catch (error) {
      if (version === selectionVersion && error.name !== "AbortError") showError("播放未能开始。请再次点击“播放”，或下载当前 MP4 后观看。");
    }
  });
  ["play", "pause", "ended", "timeupdate", "loadedmetadata", "durationchange", "emptied"].forEach(event => player.addEventListener(event, updatePlayback));
  player.addEventListener("error", () => {
    if (current?.status === "available" && player.error) showError("当前视频加载失败，或浏览器不支持此编码。可以下载 MP4 后观看，或选择其他片段。");
  });
  seek.addEventListener("input", () => {
    if (Number.isFinite(player.duration)) player.currentTime = Math.min(Number(seek.value), player.duration);
  });
  byId("fullscreen-button").addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (byId("video-stage").requestFullscreen) await byId("video-stage").requestFullscreen();
      else if (player.webkitEnterFullscreen) player.webkitEnterFullscreen();
      else showError("此浏览器不支持全屏，可下载视频后使用本地播放器观看。");
    } catch { showError("浏览器未能进入全屏，请在普通视图中播放。"); }
  });
  document.addEventListener("fullscreenchange", () => {
    text("fullscreen-button", document.fullscreenElement === byId("video-stage") ? "退出全屏" : "全屏");
  });
  byId("share-button").addEventListener("click", async () => {
    const url = shareURL();
    try { await navigator.clipboard.writeText(url); text("share-status", "当前片段链接已复制，可直接分享。"); }
    catch { text("share-status", `请复制当前片段链接：${url}`); }
  });
  window.addEventListener("hashchange", () => {
    const clip = catalog?.videos.find(item => item.id === hashId());
    if (clip && clip.id !== current?.id) selectClip(clip, false);
  });
  byId("retry-button").addEventListener("click", loadCatalog);
  loadCatalog();
})();
