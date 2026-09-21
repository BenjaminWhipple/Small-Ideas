# Live Stream Viewer

This directory is a standalone HTML/CSS/JavaScript HLS player.

Open `index.html` directly in a browser. The page defaults to the France 24
English live feed. Paste another direct `.m3u8` manifest into the URL field and
select **Load stream** to switch sources.

The page contains:

- `index.html`: the complete page structure and pinned hls.js browser loader;
- `src/style.css`: neutral layout and standard browser interactions;
- `src/direct-hls.js`: source selection, native HLS/hls.js playback, status,
  retry, and teardown behavior.

Safari uses native HLS. Other modern desktop browsers use the pinned hls.js
1.7.3 bundle from jsDelivr, with Subresource Integrity verification. The video
starts muted because browsers reject unsolicited audio autoplay; use the normal
video control to unmute.

Static hosting is supported but not required. There are currently no caption,
translation, session, or marking controls and no backend calls in this prototype.
