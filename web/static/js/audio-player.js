/**
 * Audio Player Component
 *
 * Minimal wrapper around HTML5 <audio> elements and download links.
 * The native audio controls handle play/pause, seek bar, and duration
 * display — this module provides helpers to wire up sources and
 * download URLs, plus a factory for dynamically created result players.
 *
 * Requirements: 5.3, 6.3, 6.4
 *
 * Exposed as window.AudioPlayer
 */
(function () {
  'use strict';

  /**
   * Set the source of an <audio> element by ID.
   *
   * @param {string} audioElementId - DOM id of the <audio> element.
   * @param {string} src            - URL to set as the audio source.
   */
  function setAudioSource(audioElementId, src) {
    var audio = document.getElementById(audioElementId);
    if (!audio) return;
    audio.src = src;
    audio.load();
  }

  /**
   * Set the href and download filename on a download link element.
   *
   * @param {string} linkElementId - DOM id of the <a> element.
   * @param {string} url           - Download URL.
   * @param {string} [filename]    - Suggested filename for the download attribute.
   */
  function setDownloadLink(linkElementId, url, filename) {
    var link = document.getElementById(linkElementId);
    if (!link) return;
    link.href = url;
    if (filename) {
      link.setAttribute('download', filename);
    }
  }

  /**
   * Create a result audio player element and append it to a container.
   * Useful for pipeline results where the number of output files is
   * not known ahead of time.
   *
   * The created markup mirrors the static players already in index.html
   * so the existing component CSS applies automatically.
   *
   * @param {string} containerId - DOM id of the parent container.
   * @param {string} label       - Display label (e.g. "Speaker 1").
   * @param {string} audioSrc    - URL for the audio source.
   * @param {string} downloadUrl - URL for the download button.
   * @returns {HTMLElement|null} The created wrapper element, or null if
   *   the container was not found.
   */
  function createResultPlayer(containerId, label, audioSrc, downloadUrl) {
    var container = document.getElementById(containerId);
    if (!container) return null;

    // Card wrapper
    var card = document.createElement('div');
    card.className = 'card';

    // Title
    var title = document.createElement('h3');
    title.className = 'card__title';
    title.textContent = label;
    card.appendChild(title);

    // Audio player row
    var player = document.createElement('div');
    player.className = 'audio-player';

    var labelSpan = document.createElement('span');
    labelSpan.className = 'audio-player__label';
    labelSpan.textContent = label;
    player.appendChild(labelSpan);

    var audio = document.createElement('audio');
    audio.controls = true;
    audio.setAttribute('aria-label', label + ' audio');
    audio.src = audioSrc;
    player.appendChild(audio);

    var downloadLink = document.createElement('a');
    downloadLink.className = 'btn btn--sm btn--secondary audio-player__download';
    downloadLink.href = downloadUrl;
    downloadLink.setAttribute('download', '');
    downloadLink.setAttribute('aria-label', 'Download ' + label + ' audio');
    downloadLink.textContent = 'Download';
    player.appendChild(downloadLink);

    card.appendChild(player);
    container.appendChild(card);

    return card;
  }

  // ── Expose on window ────────────────────────────────────

  window.AudioPlayer = {
    setAudioSource: setAudioSource,
    setDownloadLink: setDownloadLink,
    createResultPlayer: createResultPlayer
  };
})();
