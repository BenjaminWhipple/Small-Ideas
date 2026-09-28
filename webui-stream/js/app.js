(function () {
  "use strict";

  var data = window.STREAM_HUB_DATA;
  var state = { selectedSource: data.sources[0].id, selectedRecording: data.recordings[0].id, paused: false, lineIndex: 0 };
  var sourceGrid = document.getElementById("source-grid");
  var transcriptList = document.getElementById("transcript-list");
  var transcriptHeading = document.getElementById("transcript-heading");
  var archiveGroups = document.getElementById("archive-groups");
  var archivePreview = document.getElementById("archive-preview");
  var pauseButton = document.getElementById("pause-transcript");

  function escapeHtml(value) {
    return String(value).replace(/[&<>'"]/g, function (character) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", "\"": "&quot;" }[character];
    });
  }

  function sourceById(id) {
    return data.sources.find(function (source) { return source.id === id; });
  }

  function latestCaption(sourceId) {
    var lines = data.transcripts[sourceId] || [];
    return lines.length ? lines[lines.length - 1].text : "No transcript events yet.";
  }

  function mediaMarkup(source) {
    if (source.streamUrl) {
      return '<video class="stream-video" src="' + escapeHtml(source.streamUrl) + '" autoplay muted playsinline></video>';
    }
    return '<div class="source-glyph" aria-hidden="true"></div><span class="stage-label">No stream source configured</span>';
  }

  function renderSources() {
    sourceGrid.innerHTML = data.sources.map(function (source) {
      var selected = source.id === state.selectedSource;
      return '<article class="source-tile' + (selected ? ' is-selected' : '') + '" tabindex="0" role="button" aria-pressed="' + selected + '" data-source-id="' + source.id + '">' +
        '<div class="media-stage">' + mediaMarkup(source) + '</div>' +
        '<div class="tile-meta"><div class="tile-title-row"><span class="tile-title">' + escapeHtml(source.label) + ' · ' + escapeHtml(source.location) + '</span>' +
        '<span class="badge ' + source.status + '">' + escapeHtml(source.status) + '</span></div>' +
        '<p class="caption">' + escapeHtml(latestCaption(source.id)) + '</p></div></article>';
    }).join("");

    sourceGrid.querySelectorAll(".source-tile").forEach(function (tile) {
      function select() { state.selectedSource = tile.dataset.sourceId; renderSources(); renderTranscript(); }
      tile.addEventListener("click", select);
      tile.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); }
      });
    });
  }

  function renderTranscript() {
    var source = sourceById(state.selectedSource);
    var lines = data.transcripts[state.selectedSource] || [];
    transcriptHeading.textContent = source.label + " · " + source.location;
    transcriptList.innerHTML = lines.map(function (line) {
      return '<div class="transcript-line"><time class="transcript-time">' + escapeHtml(line.time) + '</time><p class="transcript-text">' + escapeHtml(line.text) + '</p></div>';
    }).join("");
    transcriptList.scrollTop = transcriptList.scrollHeight;
  }

  function switchView(viewName) {
    document.querySelectorAll(".view").forEach(function (view) {
      var active = view.id === viewName + "-view";
      view.hidden = !active;
      view.classList.toggle("is-active", active);
    });
    document.querySelectorAll(".tab").forEach(function (tab) {
      var active = tab.dataset.view === viewName;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    window.location.hash = viewName;
  }

  function renderArchive() {
    archiveGroups.innerHTML = data.sources.map(function (source) {
      var recordings = data.recordings.filter(function (recording) { return recording.sourceId === source.id; });
      var rows = recordings.map(function (recording) {
        return '<button class="recording-row' + (recording.id === state.selectedRecording ? ' is-selected' : '') + '" type="button" data-recording-id="' + recording.id + '">' +
          '<span class="recording-name">' + escapeHtml(recording.filename) + '</span>' +
          '<span class="recording-cell">' + escapeHtml(recording.recordedAt) + '</span>' +
          '<span class="recording-cell">' + escapeHtml(recording.size) + '</span>' +
          '<span class="recording-cell">' + escapeHtml(recording.duration) + '</span></button>';
      }).join("");
      return '<section class="archive-group"><div class="group-heading"><h3>' + escapeHtml(source.label) + ' · ' + escapeHtml(source.location) + '</h3><span>' + recordings.length + ' files</span></div>' + rows + '</section>';
    }).join("");

    archiveGroups.querySelectorAll(".recording-row").forEach(function (row) {
      row.addEventListener("click", function () { state.selectedRecording = row.dataset.recordingId; renderArchive(); renderArchivePreview(); });
    });
  }

  function renderArchivePreview() {
    var recording = data.recordings.find(function (item) { return item.id === state.selectedRecording; });
    var source = sourceById(recording.sourceId);
    var media = recording.mediaUrl
      ? '<video src="' + escapeHtml(recording.mediaUrl) + '" controls playsinline></video>'
      : '<div class="preview-empty"><div class="source-glyph" aria-hidden="true"></div><strong>Recording preview unavailable</strong><p>Add a local or remote URL to this record in data.js to enable playback.</p></div>';
    archivePreview.innerHTML = '<div class="preview-stage">' + media + '</div><div class="preview-details"><h2>' + escapeHtml(source.label) + ' · ' + escapeHtml(recording.duration) + '</h2><p>' + escapeHtml(recording.filename) + '</p><p>' + escapeHtml(recording.recordedAt) + ' · ' + escapeHtml(recording.size) + '</p></div>';
  }

  function appendTranscript() {
    if (state.paused) return;
    var now = new Date();
    var time = [now.getHours(), now.getMinutes(), now.getSeconds()].map(function (part) { return String(part).padStart(2, "0"); }).join(":");
    data.transcripts[state.selectedSource].push({ time: time, text: data.simulatedLines[state.lineIndex % data.simulatedLines.length] });
    state.lineIndex += 1;
    if (data.transcripts[state.selectedSource].length > 12) data.transcripts[state.selectedSource].shift();
    renderSources();
    renderTranscript();
  }

  document.querySelectorAll(".tab").forEach(function (tab) {
    tab.addEventListener("click", function () { switchView(tab.dataset.view); });
  });

  pauseButton.addEventListener("click", function () {
    state.paused = !state.paused;
    pauseButton.textContent = state.paused ? "▶" : "Ⅱ";
    pauseButton.setAttribute("aria-label", state.paused ? "Resume transcript simulation" : "Pause transcript simulation");
    pauseButton.title = pauseButton.getAttribute("aria-label");
  });

  document.getElementById("stream-summary").textContent = data.sources.filter(function (source) { return source.status === "online"; }).length + " of " + data.sources.length + " sources online";
  document.getElementById("archive-summary").textContent = data.recordings.length + " recordings · " + data.sources.length + " sources";
  renderSources();
  renderTranscript();
  renderArchive();
  renderArchivePreview();
  switchView(window.location.hash === "#archive" ? "archive" : "streams");
  window.setInterval(appendTranscript, 6500);
}());
