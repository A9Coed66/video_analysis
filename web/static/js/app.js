/**
 * Main App Logic
 *
 * Wires together navigation, Voice Separator flow, Pipeline flow,
 * error handling, and keyboard navigation on DOMContentLoaded.
 *
 * Requirements: 1.4, 4.4, 6.2, 6.3, 6.4, 6.5, 7.4, 7.5, 7.6, 7.7, 9.2, 9.3, 9.4, 9.5
 */
(function () {
  'use strict';

  // ── Pipeline step order ──────────────────────────────────
  var PIPELINE_STEPS = ['chunking', 'diarization', 'separation', 'transcription', 'unification'];

  // ── Helpers ──────────────────────────────────────────────

  /**
   * Format seconds into mm:ss.ms display.
   * @param {number} seconds
   * @returns {string}
   */
  function formatTimestamp(seconds) {
    if (seconds == null || isNaN(seconds)) return '0:00';
    var mins = Math.floor(seconds / 60);
    var secs = seconds % 60;
    var secsStr = secs < 10 ? '0' + secs.toFixed(2) : secs.toFixed(2);
    return mins + ':' + secsStr;
  }

  /**
   * Show an element by removing the hidden attribute.
   * @param {HTMLElement|null} el
   */
  function show(el) {
    if (el) el.hidden = false;
  }

  /**
   * Hide an element by setting the hidden attribute.
   * @param {HTMLElement|null} el
   */
  function hide(el) {
    if (el) el.hidden = true;
  }

  // ── 1. Navigation ───────────────────────────────────────

  function initNavigation() {
    // Smooth scroll for all internal anchor links
    var navLinks = document.querySelectorAll('.navbar__links a[href^="#"]');
    var heroLinks = document.querySelectorAll('.hero__actions a[href^="#"]');
    var allInternalLinks = document.querySelectorAll('a[href^="#"]');

    allInternalLinks.forEach(function (link) {
      link.addEventListener('click', function (e) {
        var targetId = link.getAttribute('href');
        if (!targetId || targetId === '#') return;
        var target = document.querySelector(targetId);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth' });
          // Close mobile menu if open
          closeMobileMenu();
        }
      });
    });

    // Active state tracking via IntersectionObserver
    var sections = document.querySelectorAll('section[id], header.hero-section');
    if (sections.length > 0 && 'IntersectionObserver' in window) {
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            var id = entry.target.id;
            navLinks.forEach(function (link) {
              var href = link.getAttribute('href');
              if (href === '#' + id) {
                link.classList.add('active');
              } else {
                link.classList.remove('active');
              }
            });
          }
        });
      }, {
        rootMargin: '-20% 0px -60% 0px',
        threshold: 0
      });

      sections.forEach(function (section) {
        observer.observe(section);
      });
    }

    // Hamburger menu toggle
    var toggle = document.querySelector('.navbar__toggle');
    var navLinksContainer = document.getElementById('navbar-links');

    if (toggle && navLinksContainer) {
      toggle.addEventListener('click', function () {
        var isOpen = navLinksContainer.classList.toggle('open');
        toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      });

      // Close menu when a nav link is clicked (mobile)
      navLinks.forEach(function (link) {
        link.addEventListener('click', function () {
          closeMobileMenu();
        });
      });
    }

    function closeMobileMenu() {
      if (navLinksContainer) navLinksContainer.classList.remove('open');
      if (toggle) toggle.setAttribute('aria-expanded', 'false');
    }
  }

  // ── 2. Voice Separator Flow ─────────────────────────────

  function initVoiceSeparator() {
    var separateBtn = document.getElementById('vs-separate-btn');
    var progressEl  = document.getElementById('vs-progress');
    var resultsEl   = document.getElementById('vs-results');
    var selectedFile = null;

    // Initialize upload zone
    var uploadZone = new window.UploadZone({
      zoneId: 'vs-upload-zone',
      fileInputId: 'vs-file-input',
      fileInfoId: 'vs-file-info',
      fileNameId: 'vs-file-name',
      fileSizeId: 'vs-file-size',
      audioPreviewId: 'vs-audio-preview',
      audioElementId: 'vs-audio-element',
      errorId: 'vs-error',
      acceptedExtensions: ['wav', 'mp3', 'flac', 'ogg'],
      onFileSelected: function (file) {
        selectedFile = file;
        if (separateBtn) separateBtn.disabled = false;
        // Reset previous results/errors
        hide(resultsEl);
        hide(progressEl);
        removeError('voice-separator');
      }
    });

    if (!separateBtn) return;

    separateBtn.addEventListener('click', function () {
      if (!selectedFile) return;
      startSeparation(selectedFile);
    });

    // Keyboard: Enter on button
    separateBtn.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        separateBtn.click();
      }
    });

    function startSeparation(file) {
      // Disable button, show progress, hide previous results
      separateBtn.disabled = true;
      hide(resultsEl);
      removeError('voice-separator');
      show(progressEl);

      window.VoiceSeparatorAPI.uploadSeparate(file)
        .then(function (response) {
          var jobId = response.job_id;
          return window.VoiceSeparatorAPI.pollStatus(jobId, function (status) {
            // onUpdate — nothing special for single-step progress
          });
        })
        .then(function (finalStatus) {
          hide(progressEl);

          if (finalStatus.status === 'completed' && finalStatus.result) {
            showSeparationResults(finalStatus.result);
          } else if (finalStatus.status === 'failed') {
            showError('voice-separator', finalStatus.error || 'Separation failed. Please try again.');
            separateBtn.disabled = false;
          }
        })
        .catch(function (err) {
          hide(progressEl);
          showError('voice-separator', err.message || 'An error occurred. Please try again.');
          separateBtn.disabled = false;
        });
    }

    function showSeparationResults(result) {
      var files = result.files || [];

      // Expect 2 files: Speaker 1 and Speaker 2
      var spk1 = files[0];
      var spk2 = files[1];

      if (spk1) {
        var url1 = window.VoiceSeparatorAPI.getDownloadUrl(spk1.file_id);
        window.AudioPlayer.setAudioSource('vs-speaker1-audio', url1);
        window.AudioPlayer.setDownloadLink('vs-speaker1-download', url1, spk1.filename || 'speaker1.wav');
      }

      if (spk2) {
        var url2 = window.VoiceSeparatorAPI.getDownloadUrl(spk2.file_id);
        window.AudioPlayer.setAudioSource('vs-speaker2-audio', url2);
        window.AudioPlayer.setDownloadLink('vs-speaker2-download', url2, spk2.filename || 'speaker2.wav');
      }

      show(resultsEl);
    }

    function resetVoiceSeparator() {
      uploadZone.reset();
      selectedFile = null;
      separateBtn.disabled = true;
      hide(progressEl);
      hide(resultsEl);
      removeError('voice-separator');
    }

    // Expose reset for retry
    window._vsReset = resetVoiceSeparator;
  }

  // ── 3. Pipeline Flow ────────────────────────────────────

  function initPipeline() {
    var processBtn    = document.getElementById('pl-process-btn');
    var progressEl    = document.getElementById('pl-progress');
    var progressLabel = document.getElementById('pl-progress-label');
    var resultsEl     = document.getElementById('pl-results');
    var transcriptBody = document.getElementById('pl-transcript-body');
    var downloadBtn   = document.getElementById('pl-download-btn');
    var selectedFile  = null;

    // Initialize upload zone
    var uploadZone = new window.UploadZone({
      zoneId: 'pl-upload-zone',
      fileInputId: 'pl-file-input',
      fileInfoId: 'pl-file-info',
      fileNameId: 'pl-file-name',
      fileSizeId: 'pl-file-size',
      audioPreviewId: 'pl-audio-preview',
      audioElementId: 'pl-audio-element',
      errorId: 'pl-error',
      acceptedExtensions: ['wav', 'mp3', 'mp4', 'flac', 'ogg'],
      onFileSelected: function (file) {
        selectedFile = file;
        if (processBtn) processBtn.disabled = false;
        // Reset previous results/errors
        hide(resultsEl);
        hide(progressEl);
        removeError('pipeline');
      }
    });

    if (!processBtn) return;

    processBtn.addEventListener('click', function () {
      if (!selectedFile) return;
      startPipeline(selectedFile);
    });

    processBtn.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        processBtn.click();
      }
    });

    function startPipeline(file) {
      processBtn.disabled = true;
      hide(resultsEl);
      removeError('pipeline');
      resetPipelineSteps();
      show(progressEl);

      window.VoiceSeparatorAPI.uploadPipeline(file)
        .then(function (response) {
          var jobId = response.job_id;
          return window.VoiceSeparatorAPI.pollStatus(jobId, function (status) {
            // Update pipeline step indicators
            if (status.current_step) {
              updatePipelineSteps(status.current_step);
              if (progressLabel) {
                progressLabel.textContent = 'Processing: ' + status.current_step + '…';
              }
            }
          });
        })
        .then(function (finalStatus) {
          hide(progressEl);

          if (finalStatus.status === 'completed' && finalStatus.result) {
            showPipelineResults(finalStatus.result);
          } else if (finalStatus.status === 'failed') {
            showError('pipeline', finalStatus.error || 'Pipeline processing failed. Please try again.');
            processBtn.disabled = false;
          }
        })
        .catch(function (err) {
          hide(progressEl);
          showError('pipeline', err.message || 'An error occurred. Please try again.');
          processBtn.disabled = false;
        });
    }

    /**
     * Update pipeline step indicators based on current_step.
     * Steps before current are marked completed, current is active.
     */
    function updatePipelineSteps(currentStep) {
      var currentIndex = PIPELINE_STEPS.indexOf(currentStep);
      if (currentIndex === -1) return;

      PIPELINE_STEPS.forEach(function (step, index) {
        var stepEl = document.querySelector('.progress__step[data-step="' + step + '"]');
        if (!stepEl) return;

        stepEl.classList.remove('progress__step--active', 'progress__step--completed');

        if (index < currentIndex) {
          stepEl.classList.add('progress__step--completed');
        } else if (index === currentIndex) {
          stepEl.classList.add('progress__step--active');
        }
      });

      // Update connectors
      var connectors = document.querySelectorAll('.progress__step-connector');
      connectors.forEach(function (connector, index) {
        connector.classList.remove('progress__step-connector--active');
        if (index < currentIndex) {
          connector.classList.add('progress__step-connector--active');
        }
      });
    }

    /**
     * Reset all pipeline step indicators to default state.
     */
    function resetPipelineSteps() {
      PIPELINE_STEPS.forEach(function (step) {
        var stepEl = document.querySelector('.progress__step[data-step="' + step + '"]');
        if (stepEl) {
          stepEl.classList.remove('progress__step--active', 'progress__step--completed');
        }
      });
      var connectors = document.querySelectorAll('.progress__step-connector');
      connectors.forEach(function (connector) {
        connector.classList.remove('progress__step-connector--active');
      });
      if (progressLabel) progressLabel.textContent = 'Processing…';
    }

    function showPipelineResults(result) {
      // Populate transcript table
      if (transcriptBody) {
        transcriptBody.innerHTML = '';
        var transcript = result.transcript || [];
        transcript.forEach(function (entry) {
          var row = document.createElement('tr');

          var speakerCell = document.createElement('td');
          speakerCell.textContent = entry.speaker_label || '';
          row.appendChild(speakerCell);

          var startCell = document.createElement('td');
          startCell.textContent = formatTimestamp(entry.start_time);
          row.appendChild(startCell);

          var endCell = document.createElement('td');
          endCell.textContent = formatTimestamp(entry.end_time);
          row.appendChild(endCell);

          var textCell = document.createElement('td');
          textCell.textContent = entry.text || '';
          row.appendChild(textCell);

          transcriptBody.appendChild(row);
        });
      }

      // Show download button if files available
      if (downloadBtn && result.files && result.files.length > 0) {
        var firstFile = result.files[0];
        var downloadUrl = window.VoiceSeparatorAPI.getDownloadUrl(firstFile.file_id);
        downloadBtn.href = downloadUrl;
        if (firstFile.filename) {
          downloadBtn.setAttribute('download', firstFile.filename);
        }
        show(downloadBtn);
      }

      show(resultsEl);
    }

    function resetPipeline() {
      uploadZone.reset();
      selectedFile = null;
      processBtn.disabled = true;
      hide(progressEl);
      hide(resultsEl);
      if (downloadBtn) hide(downloadBtn);
      if (transcriptBody) transcriptBody.innerHTML = '';
      resetPipelineSteps();
      removeError('pipeline');
    }

    // Expose reset for retry
    window._plReset = resetPipeline;
  }

  // ── 4. Error Handling ───────────────────────────────────

  /**
   * Show an error message with a retry button inside the given section.
   * @param {string} sectionId - 'voice-separator' or 'pipeline'
   * @param {string} message
   */
  function showError(sectionId, message) {
    var section = document.getElementById(sectionId);
    if (!section) return;

    // Remove any existing error in this section
    removeError(sectionId);

    var errorDiv = document.createElement('div');
    errorDiv.className = 'results__error';
    errorDiv.setAttribute('role', 'alert');
    errorDiv.setAttribute('data-error-for', sectionId);

    var msgP = document.createElement('p');
    msgP.className = 'results__error-message';
    msgP.textContent = message;
    errorDiv.appendChild(msgP);

    var retryBtn = document.createElement('button');
    retryBtn.className = 'btn btn--primary btn--sm';
    retryBtn.textContent = 'Try Again';
    retryBtn.setAttribute('aria-label', 'Retry ' + (sectionId === 'voice-separator' ? 'voice separation' : 'pipeline processing'));
    retryBtn.addEventListener('click', function () {
      if (sectionId === 'voice-separator' && window._vsReset) {
        window._vsReset();
      } else if (sectionId === 'pipeline' && window._plReset) {
        window._plReset();
      }
    });
    // Keyboard support for retry button
    retryBtn.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        retryBtn.click();
      }
    });
    errorDiv.appendChild(retryBtn);

    // Insert error after the progress element or at end of container
    var container = section.querySelector('.container');
    if (container) {
      container.appendChild(errorDiv);
    }
  }

  /**
   * Remove error message from a section.
   * @param {string} sectionId
   */
  function removeError(sectionId) {
    var existing = document.querySelector('[data-error-for="' + sectionId + '"]');
    if (existing && existing.parentNode) {
      existing.parentNode.removeChild(existing);
    }
  }

  // ── Init on DOMContentLoaded ────────────────────────────

  document.addEventListener('DOMContentLoaded', function () {
    initNavigation();
    initVoiceSeparator();
    initPipeline();
  });
})();
