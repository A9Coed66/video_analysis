/**
 * Upload Zone Component
 *
 * Reusable drag-and-drop / click-to-select file upload with client-side
 * validation, file info display, and audio preview.
 *
 * Used by both Voice Separator and Pipeline sections — they differ only
 * in accepted file extensions.
 *
 * Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 7.1, 7.2
 *
 * Exposed as window.UploadZone
 */
(function () {
  'use strict';

  var DEFAULT_MAX_SIZE = 500 * 1024 * 1024; // 500 MB

  // ── Helpers ──────────────────────────────────────────────

  /**
   * Format bytes into a human-readable string (KB, MB, GB).
   * @param {number} bytes
   * @returns {string}
   */
  function formatFileSize(bytes) {
    if (bytes == null || bytes < 0) return '0 B';
    if (bytes === 0) return '0 B';

    var units = ['B', 'KB', 'MB', 'GB'];
    var i = 0;
    var size = bytes;

    while (size >= 1024 && i < units.length - 1) {
      size /= 1024;
      i++;
    }

    // Show integers when possible, otherwise 2 decimal places
    var formatted = size % 1 === 0 ? size.toString() : size.toFixed(2);
    return formatted + ' ' + units[i];
  }

  /**
   * Extract the lowercase file extension (without dot) from a filename.
   * @param {string} filename
   * @returns {string}
   */
  function getExtension(filename) {
    if (!filename || typeof filename !== 'string') return '';
    var parts = filename.split('.');
    if (parts.length < 2) return '';
    return parts[parts.length - 1].toLowerCase();
  }

  // ── UploadZone ───────────────────────────────────────────

  /**
   * Create and initialise an upload zone.
   *
   * @param {Object} config
   * @param {string} config.zoneId            - ID of the drop-zone element
   * @param {string} config.fileInputId       - ID of the hidden <input type="file">
   * @param {string} config.fileInfoId        - ID of the file-info container
   * @param {string} config.fileNameId        - ID of the file-name span
   * @param {string} config.fileSizeId        - ID of the file-size span
   * @param {string} config.audioPreviewId    - ID of the audio preview container
   * @param {string} config.audioElementId    - ID of the <audio> element
   * @param {string} config.errorId           - ID of the error message element
   * @param {string[]} config.acceptedExtensions - e.g. ['wav','mp3','flac','ogg']
   * @param {number}  [config.maxSize]        - Max file size in bytes (default 500 MB)
   * @param {Function} [config.onFileSelected] - Callback receiving the valid File
   */
  function UploadZone(config) {
    if (!config) throw new Error('UploadZone: config is required');

    this.config = config;
    this.maxSize = config.maxSize != null ? config.maxSize : DEFAULT_MAX_SIZE;
    this.acceptedExtensions = (config.acceptedExtensions || []).map(function (e) {
      return e.toLowerCase().replace(/^\./, '');
    });
    this.selectedFile = null;

    // Cache DOM references
    this.zone         = document.getElementById(config.zoneId);
    this.fileInput    = document.getElementById(config.fileInputId);
    this.fileInfo     = document.getElementById(config.fileInfoId);
    this.fileName     = document.getElementById(config.fileNameId);
    this.fileSize     = document.getElementById(config.fileSizeId);
    this.audioPreview = document.getElementById(config.audioPreviewId);
    this.audioElement = document.getElementById(config.audioElementId);
    this.error        = document.getElementById(config.errorId);

    this._dragCounter = 0;
    this._objectUrl   = null;

    this._bindEvents();
  }

  // ── Event binding ────────────────────────────────────────

  UploadZone.prototype._bindEvents = function () {
    var self = this;

    if (!this.zone || !this.fileInput) return;

    // Click zone → trigger hidden file input
    this.zone.addEventListener('click', function () {
      self.fileInput.click();
    });

    // Keyboard: Enter / Space also triggers file input
    this.zone.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        self.fileInput.click();
      }
    });

    // File input change
    this.fileInput.addEventListener('change', function () {
      if (self.fileInput.files && self.fileInput.files.length > 0) {
        self._handleFile(self.fileInput.files[0]);
      }
    });

    // Drag-and-drop events
    this.zone.addEventListener('dragenter', function (e) {
      e.preventDefault();
      e.stopPropagation();
      self._dragCounter++;
      self.zone.classList.add('upload-zone--dragover');
    });

    this.zone.addEventListener('dragover', function (e) {
      e.preventDefault();
      e.stopPropagation();
    });

    this.zone.addEventListener('dragleave', function (e) {
      e.preventDefault();
      e.stopPropagation();
      self._dragCounter--;
      if (self._dragCounter <= 0) {
        self._dragCounter = 0;
        self.zone.classList.remove('upload-zone--dragover');
      }
    });

    this.zone.addEventListener('drop', function (e) {
      e.preventDefault();
      e.stopPropagation();
      self._dragCounter = 0;
      self.zone.classList.remove('upload-zone--dragover');

      var files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length > 0) {
        self._handleFile(files[0]);
      }
    });
  };

  // ── File handling ────────────────────────────────────────

  /**
   * Validate and process a selected file.
   * @param {File} file
   */
  UploadZone.prototype._handleFile = function (file) {
    // Reset previous state
    this._hideError();
    this._hideFileInfo();
    this._hideAudioPreview();
    this.selectedFile = null;

    // Validate extension
    var ext = getExtension(file.name);
    if (this.acceptedExtensions.length > 0 && this.acceptedExtensions.indexOf(ext) === -1) {
      var formatsStr = this.acceptedExtensions.map(function (e) {
        return e.toUpperCase();
      }).join(', ');
      this._showError('Unsupported file format. Accepted formats: ' + formatsStr);
      return;
    }

    // Validate size
    if (file.size > this.maxSize) {
      this._showError('File is too large. Maximum size: ' + formatFileSize(this.maxSize));
      return;
    }

    // Valid file
    this.selectedFile = file;
    this._showFileInfo(file);
    this._showAudioPreview(file);

    if (typeof this.config.onFileSelected === 'function') {
      this.config.onFileSelected(file);
    }
  };

  // ── UI helpers ───────────────────────────────────────────

  UploadZone.prototype._showFileInfo = function (file) {
    if (this.fileName) this.fileName.textContent = file.name;
    if (this.fileSize) this.fileSize.textContent = formatFileSize(file.size);
    if (this.fileInfo) this.fileInfo.hidden = false;
  };

  UploadZone.prototype._hideFileInfo = function () {
    if (this.fileInfo) this.fileInfo.hidden = true;
    if (this.fileName) this.fileName.textContent = '';
    if (this.fileSize) this.fileSize.textContent = '';
  };

  UploadZone.prototype._showAudioPreview = function (file) {
    if (!this.audioElement || !this.audioPreview) return;

    // Revoke previous object URL to avoid memory leaks
    if (this._objectUrl) {
      URL.revokeObjectURL(this._objectUrl);
    }

    this._objectUrl = URL.createObjectURL(file);
    this.audioElement.src = this._objectUrl;
    this.audioPreview.hidden = false;
  };

  UploadZone.prototype._hideAudioPreview = function () {
    if (!this.audioElement || !this.audioPreview) return;

    if (this._objectUrl) {
      URL.revokeObjectURL(this._objectUrl);
      this._objectUrl = null;
    }
    this.audioElement.removeAttribute('src');
    this.audioElement.load(); // reset player
    this.audioPreview.hidden = true;
  };

  UploadZone.prototype._showError = function (message) {
    if (!this.error) return;
    this.error.textContent = message;
    this.error.hidden = false;
  };

  UploadZone.prototype._hideError = function () {
    if (!this.error) return;
    this.error.textContent = '';
    this.error.hidden = true;
  };

  /**
   * Programmatically reset the upload zone to its initial state.
   */
  UploadZone.prototype.reset = function () {
    this._hideError();
    this._hideFileInfo();
    this._hideAudioPreview();
    this.selectedFile = null;
    if (this.fileInput) this.fileInput.value = '';
  };

  // ── Static utility exposed for testing / reuse ──────────

  UploadZone.formatFileSize = formatFileSize;
  UploadZone.getExtension   = getExtension;

  // ── Expose on window ────────────────────────────────────

  window.UploadZone = UploadZone;
})();
