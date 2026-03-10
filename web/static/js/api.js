/**
 * Voice Separator API Client
 *
 * Provides functions for communicating with the backend REST API:
 * - uploadSeparate / uploadPipeline: submit files for processing
 * - getStatus: check job progress
 * - getDownloadUrl: build download links
 * - pollStatus: poll until job completes or fails (with timeout)
 *
 * Exposed on window.VoiceSeparatorAPI
 */
(function () {
  'use strict';

  var DEFAULT_POLL_INTERVAL = 2000;   // 2 seconds
  var DEFAULT_POLL_TIMEOUT  = 600000; // 10 minutes

  // ── Helpers ──────────────────────────────────────────────

  /**
   * Parse a fetch Response, throwing on non-2xx status.
   * Attempts to read the JSON body for an error message.
   */
  async function handleResponse(response) {
    if (response.ok) {
      return response.json();
    }

    var errorBody;
    try {
      errorBody = await response.json();
    } catch (_) {
      errorBody = null;
    }

    var message = (errorBody && (errorBody.error || errorBody.detail))
      ? (errorBody.error || errorBody.detail)
      : 'Request failed with status ' + response.status;

    throw new Error(message);
  }

  /**
   * Wrapper around fetch that converts network errors into
   * descriptive Error instances.
   */
  async function safeFetch(url, options) {
    try {
      return await fetch(url, options);
    } catch (err) {
      throw new Error('Network error: ' + (err.message || 'Unable to reach the server'));
    }
  }

  // ── Public API ───────────────────────────────────────────

  /**
   * Upload a file for voice separation.
   * POST /api/separate
   *
   * @param {File} file - The audio file to separate.
   * @returns {Promise<{job_id: string, status: string}>}
   */
  async function uploadSeparate(file) {
    var formData = new FormData();
    formData.append('file', file);

    var response = await safeFetch('/api/separate', {
      method: 'POST',
      body: formData
    });

    return handleResponse(response);
  }

  /**
   * Upload a file for full pipeline processing.
   * POST /api/pipeline
   *
   * @param {File} file - The audio/video file to process.
   * @returns {Promise<{job_id: string, status: string}>}
   */
  async function uploadPipeline(file) {
    var formData = new FormData();
    formData.append('file', file);

    var response = await safeFetch('/api/pipeline', {
      method: 'POST',
      body: formData
    });

    return handleResponse(response);
  }

  /**
   * Fetch the current status of a job.
   * GET /api/status/{jobId}
   *
   * @param {string} jobId
   * @returns {Promise<Object>} Job status response
   */
  async function getStatus(jobId) {
    var response = await safeFetch('/api/status/' + encodeURIComponent(jobId));
    return handleResponse(response);
  }

  /**
   * Build the download URL for a result file.
   *
   * @param {string} fileId
   * @returns {string} URL path
   */
  function getDownloadUrl(fileId) {
    return '/api/download/' + encodeURIComponent(fileId);
  }

  /**
   * Poll job status at a regular interval until the job reaches
   * a terminal state (completed / failed) or the timeout expires.
   *
   * Transient network errors during polling are silently retried
   * on the next interval rather than rejecting immediately.
   *
   * @param {string}   jobId    - The job to poll.
   * @param {Function} onUpdate - Called with each status response.
   * @param {Object}   [options]
   * @param {number}   [options.interval] - Polling interval in ms (default 2 000).
   * @param {number}   [options.timeout]  - Max wait in ms (default 600 000 = 10 min).
   * @returns {Promise<Object>} Resolves with the final status response.
   */
  function pollStatus(jobId, onUpdate, options) {
    var interval = (options && options.interval) || DEFAULT_POLL_INTERVAL;
    var timeout  = (options && options.timeout)  || DEFAULT_POLL_TIMEOUT;

    return new Promise(function (resolve, reject) {
      var timerId  = null;
      var timeoutId = null;
      var stopped = false;

      function cleanup() {
        stopped = true;
        if (timerId)  clearTimeout(timerId);
        if (timeoutId) clearTimeout(timeoutId);
      }

      // Timeout guard
      timeoutId = setTimeout(function () {
        cleanup();
        reject(new Error('Polling timed out after ' + (timeout / 1000) + ' seconds'));
      }, timeout);

      async function tick() {
        if (stopped) return;

        try {
          var status = await getStatus(jobId);

          if (typeof onUpdate === 'function') {
            onUpdate(status);
          }

          if (status.status === 'completed' || status.status === 'failed') {
            cleanup();
            resolve(status);
            return;
          }
        } catch (_err) {
          // Transient error — retry on next tick
        }

        if (!stopped) {
          timerId = setTimeout(tick, interval);
        }
      }

      // Kick off the first poll
      tick();
    });
  }

  // ── Expose on window ────────────────────────────────────

  window.VoiceSeparatorAPI = {
    uploadSeparate: uploadSeparate,
    uploadPipeline: uploadPipeline,
    getStatus: getStatus,
    getDownloadUrl: getDownloadUrl,
    pollStatus: pollStatus
  };
})();
