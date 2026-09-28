(function () {
  "use strict";

  window.STREAM_HUB_DATA = {
    sources: [
      { id: "source-01", label: "Source 01", location: "Main entrance", status: "online", streamUrl: null },
      { id: "source-02", label: "Source 02", location: "Workshop floor", status: "online", streamUrl: null },
      { id: "source-03", label: "Source 03", location: "Loading bay", status: "standby", streamUrl: null },
      { id: "source-04", label: "Source 04", location: "Rear corridor", status: "offline", streamUrl: null }
    ],
    transcripts: {
      "source-01": [
        { time: "09:41:02", text: "Delivery team has arrived at the main entrance." },
        { time: "09:41:18", text: "The access code is being confirmed." },
        { time: "09:42:07", text: "Door opened. Two people entered the lobby." }
      ],
      "source-02": [
        { time: "09:39:44", text: "Machine three is ready for the next material run." },
        { time: "09:40:31", text: "Confirming the measurements before startup." }
      ],
      "source-03": [
        { time: "09:35:12", text: "Bay is clear. Waiting for the afternoon pickup." }
      ],
      "source-04": [
        { time: "09:28:09", text: "Audio source disconnected." }
      ]
    },
    simulatedLines: [
      "Copy that. I can see the update now.",
      "The area is clear and normal activity has resumed.",
      "Please hold at the entrance for confirmation.",
      "Equipment check is complete.",
      "Movement detected near the edge of the frame.",
      "The delivery has been logged and received."
    ],
    recordings: [
      { id: "r1", sourceId: "source-01", filename: "source-01_2026-09-28_084500.mp4", recordedAt: "Sep 28, 2026 · 08:45", size: "182 MB", duration: "12:04", mediaUrl: null },
      { id: "r2", sourceId: "source-01", filename: "source-01_2026-09-27_163012.mp4", recordedAt: "Sep 27, 2026 · 16:30", size: "96 MB", duration: "06:21", mediaUrl: null },
      { id: "r3", sourceId: "source-02", filename: "source-02_2026-09-28_071805.mp4", recordedAt: "Sep 28, 2026 · 07:18", size: "241 MB", duration: "15:58", mediaUrl: null },
      { id: "r4", sourceId: "source-02", filename: "source-02_2026-09-27_114410.mp4", recordedAt: "Sep 27, 2026 · 11:44", size: "78 MB", duration: "05:09", mediaUrl: null },
      { id: "r5", sourceId: "source-03", filename: "source-03_2026-09-27_181223.mp4", recordedAt: "Sep 27, 2026 · 18:12", size: "155 MB", duration: "10:16", mediaUrl: null },
      { id: "r6", sourceId: "source-04", filename: "source-04_2026-09-26_220006.mp4", recordedAt: "Sep 26, 2026 · 22:00", size: "63 MB", duration: "04:11", mediaUrl: null }
    ]
  };
}());
