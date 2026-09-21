const FRANCE_24_HLS =
  "https://live.france24.com/hls/live/2037218/F24_EN_HI_HLS/master_5000.m3u8";

const elements = {
  form: document.querySelector("#source-form"),
  source: document.querySelector("#source-url"),
  media: document.querySelector("#media"),
  message: document.querySelector("#media-message"),
  meta: document.querySelector("#stream-meta"),
  dot: document.querySelector("#connection-dot"),
  label: document.querySelector("#connection-label"),
  error: document.querySelector("#error-banner"),
};

let hls = null;

function setState(state, label, message = "") {
  elements.dot.dataset.state = state;
  elements.label.textContent = label;
  elements.message.textContent = message;
  elements.message.hidden = !message;
}

function showError(message) {
  elements.error.textContent = message;
  elements.error.hidden = false;
  setState("failed", "Playback failed", "Unable to play this stream.");
}

function clearError() {
  elements.error.hidden = true;
  elements.error.textContent = "";
}

function dispose() {
  if (hls) hls.destroy();
  hls = null;
  elements.media.pause();
  elements.media.removeAttribute("src");
  elements.media.load();
}

async function beginPlayback() {
  try {
    await elements.media.play();
  } catch (_error) {
    setState("connected", "Ready", "Press play to start the live stream.");
  }
}

function updateMetadata(level = null, audioTrackCount = null) {
  const video = elements.media;
  const dimensions = video.videoWidth && video.videoHeight
    ? `${video.videoWidth}×${video.videoHeight}`
    : level?.width && level?.height
      ? `${level.width}×${level.height}`
      : "live video";
  const audio = audioTrackCount === null
    ? "audio included"
    : `${audioTrackCount} audio track${audioTrackCount === 1 ? "" : "s"}`;
  elements.meta.textContent = `${dimensions} · ${audio} · direct HLS`;
}

function loadNative(url) {
  elements.media.src = url;
  elements.media.addEventListener("loadedmetadata", () => {
    updateMetadata();
    setState("connected", "Live", "");
    beginPlayback();
  }, { once: true });
}

function loadWithHlsJs(url) {
  hls = new window.Hls({
    enableWorker: true,
    lowLatencyMode: true,
    backBufferLength: 60,
  });
  hls.on(window.Hls.Events.MEDIA_ATTACHED, () => hls.loadSource(url));
  hls.on(window.Hls.Events.MANIFEST_PARSED, (_event, data) => {
    updateMetadata(data.levels?.[0], hls.audioTracks.length);
    setState("connected", "Live", "");
    beginPlayback();
  });
  hls.on(window.Hls.Events.LEVEL_SWITCHED, (_event, data) => {
    updateMetadata(hls.levels[data.level], hls.audioTracks.length);
  });
  hls.on(window.Hls.Events.ERROR, (_event, data) => {
    if (!data.fatal) return;
    if (data.type === window.Hls.ErrorTypes.NETWORK_ERROR) {
      setState("", "Reconnecting", "Waiting for the live stream…");
      hls.startLoad();
    } else if (data.type === window.Hls.ErrorTypes.MEDIA_ERROR) {
      hls.recoverMediaError();
    } else {
      showError(`HLS playback failed: ${data.details || data.type}`);
    }
  });
  hls.attachMedia(elements.media);
}

function loadStream(value) {
  let url;
  try {
    url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol)) throw new Error();
  } catch (_error) {
    showError("Enter an absolute HTTP or HTTPS HLS manifest URL.");
    return;
  }

  clearError();
  dispose();
  elements.source.value = url.toString();
  const pageUrl = new URL(window.location.href);
  pageUrl.searchParams.set("url", url.toString());
  try {
    window.history.replaceState(null, "", pageUrl);
  } catch (_error) {
    // Some browsers restrict history changes for pages opened from file://.
  }
  setState("", "Connecting", "Loading live stream…");

  if (elements.media.canPlayType("application/vnd.apple.mpegurl")) {
    loadNative(url.toString());
  } else if (window.Hls?.isSupported()) {
    loadWithHlsJs(url.toString());
  } else {
    showError("This browser does not support HLS playback or Media Source Extensions.");
  }
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  loadStream(elements.source.value);
});
elements.media.addEventListener("playing", () => setState("connected", "Live", ""));
elements.media.addEventListener("waiting", () => setState("", "Buffering", "Buffering live video…"));
elements.media.addEventListener("error", () => {
  if (!hls) showError("The browser could not load this HLS stream.");
});

const requested = new URLSearchParams(window.location.search).get("url");
loadStream(requested || FRANCE_24_HLS);
