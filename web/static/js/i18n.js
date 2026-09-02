/**
 * Smart Eye Diagnosis System - Internationalization (i18n) Engine
 * 
 * Features:
 * - Global persistence: saves language choice to localStorage
 * - Automatic restoration on page load
 * - Seamless translation across all pages in the workspace
 * - Reactive UI updates when language changes
 */

class I18nEngine {
  constructor() {
    this.supportedLanguages = ['ko', 'en', 'zh', 'vi', 'ru', 'ja'];
    // Load saved language or detect from browser
    this.currentLanguage = this.loadLanguage();
    this.listeners = [];
    
    console.log(`[i18n] Initialized with language: ${this.currentLanguage}`);
    
    // Initialize immediately and on DOMContentLoaded
    this.init();
  }

  /**
   * Initialize i18n engine - applies translations when DOM is ready
   */
  init() {
    // If DOM not ready yet, wait for it
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        console.log('[i18n] DOMContentLoaded - applying translations');
        this.applyTranslations();
      });
    } else {
      // DOM already loaded
      console.log('[i18n] Document already loaded - applying translations immediately');
      this.applyTranslations();
    }

    // Reapply after short delay to ensure all elements are rendered
    setTimeout(() => {
      console.log('[i18n] Applying translations after 300ms delay');
      this.applyTranslations();
    }, 300);

    // Watch for dynamically added elements with data-i18n attributes
    if (document.body) {
      this.observeDOMChanges();
    } else {
      document.addEventListener('DOMContentLoaded', () => this.observeDOMChanges(), { once: true });
    }
  }

  /**
   * Observe DOM for dynamically added elements with data-i18n attributes
   */
  observeDOMChanges() {
    try {
      const observer = new MutationObserver((mutations) => {
        let hasNewI18nElements = false;
        for (const mutation of mutations) {
          if (mutation.addedNodes.length > 0) {
            for (const node of mutation.addedNodes) {
              if (node.nodeType === 1 && (node.hasAttribute('data-i18n') || node.querySelector('[data-i18n]'))) {
                hasNewI18nElements = true;
                break;
              }
            }
          }
        }
        if (hasNewI18nElements) {
          this.applyTranslations();
        }
      });

      observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: false,
        characterData: false
      });
    } catch (e) {
      console.warn('[i18n] MutationObserver not available:', e);
    }
  }

  /**
   * Load language preference from localStorage with fallbacks
   * @returns {string} Language code (ko, en, zh, vi, ru, ja)
   */
  loadLanguage() {
    try {
      // First check localStorage for saved language
      const saved = localStorage.getItem('i18n_language') || localStorage.getItem('selectedLanguage');
      if (saved && this.supportedLanguages && this.supportedLanguages.includes(saved)) {
        console.log(`[i18n.loadLanguage] Found saved language: ${saved}`);
        return saved;
      }
    } catch (e) {
      console.warn('[i18n.loadLanguage] localStorage access error:', e);
    }

    // Try to detect from browser language
    try {
      const browserLang = navigator.language?.split('-')[0]?.toLowerCase();
      if (browserLang && this.supportedLanguages && this.supportedLanguages.includes(browserLang)) {
        console.log(`[i18n.loadLanguage] Using browser language: ${browserLang}`);
        return browserLang;
      }
    } catch (e) {
      console.warn('[i18n.loadLanguage] Browser language detection failed:', e);
    }

    // Default to Korean
    console.log('[i18n.loadLanguage] Defaulting to Korean (ko)');
    return 'ko';
  }

  /**
   * Get the current active language
   * @returns {string} Current language code
   */
  getCurrentLanguage() {
    return this.currentLanguage;
  }

  /**
   * Get a translation string by key
   * @param {string} key - Translation key
   * @param {object} params - Optional parameters for template replacement
   * @returns {string} Translated string
   */
  getTranslation(key, params = {}) {
    // Check if translations object exists and has current language
    if (typeof translations === 'undefined' || !translations) {
      console.warn(`[i18n.getTranslation] Translations object not available for key: ${key}`);
      return key;
    }

    const langDict = translations[this.currentLanguage];
    let text = langDict && langDict[key] ? langDict[key] : key;

    // Replace template parameters (e.g., {count}, {phone})
    Object.keys(params).forEach(paramKey => {
      text = text.replace(new RegExp(`\\{${paramKey}\\}`, 'g'), params[paramKey]);
    });

    return text;
  }

  /**
   * Change the current language and update DOM
   * @param {string} lang - Language code (ko, en, zh, vi, ru, ja)
   */
  changeLanguage(lang) {
    if (!this.supportedLanguages.includes(lang)) {
      console.warn(`[i18n.changeLanguage] Unsupported language: ${lang}`);
      return;
    }

    console.log(`[i18n.changeLanguage] Changing language from ${this.currentLanguage} to ${lang}`);
    this.currentLanguage = lang;

    // Save to localStorage (both keys for compatibility)
    try {
      localStorage.setItem('i18n_language', lang);
      localStorage.setItem('selectedLanguage', lang);
      console.log(`[i18n.changeLanguage] Saved language to localStorage: ${lang}`);
    } catch (e) {
      console.warn('[i18n.changeLanguage] Failed to save to localStorage:', e);
    }

    // Update DOM with new translations
    this.applyTranslations();

    // Notify all registered listeners
    console.log(`[i18n.changeLanguage] Notifying ${this.listeners.length} listeners`);
    this.listeners.forEach(callback => {
      try {
        callback(lang);
      } catch (e) {
        console.warn('[i18n.changeLanguage] Listener error:', e);
      }
    });
  }

  /**
   * Register a callback to be called when language changes
   * @param {function} callback - Function to call on language change
   */
  onLanguageChange(callback) {
    if (typeof callback === 'function') {
      this.listeners.push(callback);
    }
  }

  /**
   * Apply translations to all elements with data-i18n attribute
   * This method scans the entire DOM and updates all translatable elements
   */
  applyTranslations() {
    // Update document language attribute
    document.documentElement.lang = this.currentLanguage;

    // Find all elements with data-i18n attribute
    const elements = document.querySelectorAll('[data-i18n]');
    console.log(`[i18n.applyTranslations] Found ${elements.length} elements with data-i18n attribute for language: ${this.currentLanguage}`);

    let updateCount = 0;
    let errorCount = 0;

    elements.forEach(element => {
      try {
        const key = element.getAttribute('data-i18n');
        const params = this.extractParamsFromElement(element);
        const text = this.getTranslation(key, params);

        // Handle different element types
        if (element.tagName === 'INPUT') {
          // For input elements, check for placeholder attribute
          if (element.hasAttribute('data-i18n-placeholder')) {
            element.placeholder = text;
          } else {
            element.value = text;
          }
          updateCount++;
        } else if (element.tagName === 'TEXTAREA') {
          // For textarea, handle placeholder or value
          if (element.hasAttribute('data-i18n-placeholder')) {
            element.placeholder = text;
          } else {
            element.value = text;
          }
          updateCount++;
        } else {
          // For other elements (div, button, span, p, etc.)
          if (element.hasAttribute('data-i18n-html')) {
            // If HTML flag is set, use innerHTML
            element.innerHTML = text;
          } else {
            // Otherwise use textContent (safer)
            element.textContent = text;
          }
          updateCount++;
        }
      } catch (e) {
        console.warn(`[i18n.applyTranslations] Error updating element:`, e);
        errorCount++;
      }
    });

    console.log(`[i18n.applyTranslations] Updated ${updateCount} elements (${errorCount} errors)`);
    // Apply fallback translations to common selectors when templates lack data-i18n
    try {
      this.applyFallbackTranslations();
    } catch (e) {
      console.warn('[i18n.applyTranslations] applyFallbackTranslations error:', e);
    }
  }

  /**
   * Apply translations to common selectors if they lack data-i18n attributes.
   * This helps translate pages that haven't been fully annotated with data-i18n.
   */
  applyFallbackTranslations() {
    const mapping = [
      { selector: 'h1', key: 'main_title' },
      { selector: '.helper-text', key: 'helper_disclaimer' },
      { selector: '.btn-start', key: 'btn_start_diagnosis' },
      { selector: '.btn-kakao-link', key: 'btn_kakao_link' },
      { selector: '.btn-mobile-access', key: 'btn_mobile_access' },
      { selector: '.mobile-o2o-title', key: 'mobile_o2o_title' },
      { selector: '.kakao-link-title', key: 'kakao_link_title' },
      { selector: '#mobile-connect-status', key: 'mobile_o2o_waiting' },
      { selector: '.theme-toggle-floating [data-role="label"]', key: 'theme_dark_mode' }
    ];

    mapping.forEach((item) => {
      try {
        const el = document.querySelector(item.selector);
        if (!el) return;
        // Skip if element already has data-i18n
        if (el.hasAttribute && el.hasAttribute('data-i18n')) return;
        const text = this.getTranslation(item.key);
        if (!text) return;
        // Respect HTML flag for known keys
        if (item.key && (item.key.endsWith('_line1') || item.key.endsWith('_line2') || item.key.startsWith('subtitle'))) {
          el.innerHTML = text;
        } else {
          el.textContent = text;
        }
      } catch (e) {
        // continue on error
      }
    });
  }

  /**
   * Extract parameters from element attributes for translation replacement
   * @param {HTMLElement} element - Element to extract params from
   * @returns {object} Parameters object
   */
  extractParamsFromElement(element) {
    const params = {};
    const attributes = element.attributes;

    for (let i = 0; i < attributes.length; i++) {
      const attr = attributes[i];
      if (attr.name.startsWith('data-param-')) {
        const paramKey = attr.name.substring(11); // Remove 'data-param-' prefix
        params[paramKey] = attr.value;
      }
    }

    return params;
  }

  /**
   * Utility: Get all supported languages
   * @returns {array} Array of language codes
   */
  getSupportedLanguages() {
    return [...this.supportedLanguages];
  }

  /**
   * Utility: Get language name in current language
   * @param {string} lang - Language code
   * @returns {string} Language name
   */
  getLanguageName(lang) {
    const key = `lang_${lang}`;
    return this.getTranslation(key);
  }
}

/**
 * Initialize and expose i18n globally
 */
const i18n = new I18nEngine();

// Make i18n globally accessible
window.i18n = i18n;

/**
 * ============================================================================
 * LLM INTEGRATION - ACTIVE FUNCTIONS
 * ============================================================================
 */

/**
 * ============================================================================
 * Step 1: API Helper Functions for LLM Integration
 * ============================================================================
 */

/**
 * Enhanced fetch that automatically adds language to request body
 * @param {string} url - API endpoint
 * @param {object} options - Fetch options (method, body, etc.)
 * @returns {Promise<Response>} Fetch response
 */
i18n.apiFetch = function(url, options = {}) {
  const defaultOptions = {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      'Accept-Language': this.currentLanguage
    }
  };

  const mergedOptions = { ...defaultOptions, ...options };

  // If body exists, inject language field
  if (mergedOptions.body && typeof mergedOptions.body === 'string') {
    try {
      const bodyObj = JSON.parse(mergedOptions.body);
      bodyObj.language = this.currentLanguage;  // Step 2: Add language to request
      mergedOptions.body = JSON.stringify(bodyObj);
    } catch (e) {
      // If body is not JSON, leave as is
    }
  }

  return fetch(url, mergedOptions);
};

/**
 * Diagnose eye image with language context
 * @param {Blob|FormData} imageData - Image file or FormData
 * @param {object} metadata - Additional metadata (optional)
 * @returns {Promise<object>} Diagnosis result with language
 */
i18n.diagnosisWithLanguage = async function(imageData, metadata = {}) {
  try {
    const formData = imageData instanceof FormData ? imageData : new FormData();
    if (!(imageData instanceof FormData)) {
      formData.append('image', imageData);
    }
    
    // Step 2: Add language to request
    formData.append('language', this.currentLanguage);
    formData.append('metadata', JSON.stringify(metadata));

    const response = await fetch('/api/diagnose', {
      method: 'POST',
      body: formData,
      headers: {
        'Accept-Language': this.currentLanguage
      }
    });

    const result = await response.json();
    // Step 3: Backend should return language-aware response
    return {
      ...result,
      language: this.currentLanguage,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    console.error('Diagnosis API error:', error);
    throw error;
  }
};

/**
 * Generate medical report with LLM (language-aware)
 * @param {object} diagnosisResult - Diagnosis data
 * @param {object} patientInfo - Patient metadata
 * @returns {Promise<object>} Generated report
 */
i18n.generateMedicalReport = async function(diagnosisResult, patientInfo = {}) {
  try {
    const payload = {
      diagnosis_result: diagnosisResult,
      patient_info: patientInfo,
      language: this.currentLanguage  // Step 2: Include language
    };

    // Step 3: Backend receives language field
    const response = await this.apiFetch('/api/generate_report', {
      method: 'POST',
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`Report generation failed: ${response.statusText}`);
    }

    const report = await response.json();
    
    // Step 4: LLM generates report in selected language
    return {
      ...report,
      language: this.currentLanguage,
      generated_at: new Date().toISOString()
    };
  } catch (error) {
    console.error('Report generation error:', error);
    throw error;
  }
};

/**
 * Generic API call with automatic language injection
 * @param {string} endpoint - API endpoint path
 * @param {object} data - Request data
 * @param {string} method - HTTP method (default: POST)
 * @returns {Promise<object>} API response
 */
i18n.callAPI = async function(endpoint, data = {}, method = 'POST') {
  try {
    const payload = {
      ...data,
      language: this.currentLanguage  // Step 2: Always include language
    };

    const response = await this.apiFetch(endpoint, {
      method: method,
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`API call failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API call error (${endpoint}):`, error);
    throw error;
  }
};

/**
 * ============================================================================
 * BACKEND INTEGRATION CHECKLIST
 * ============================================================================
 * 
 * Step 1: ✓ i18n.getCurrentLanguage()
 *   - Returns current language code: 'ko' | 'en' | 'zh' | 'vi' | 'ru' | 'ja'
 * 
 * Step 2: ✓ API Request with Language
 *   - Use i18n.diagnosisWithLanguage(image) or i18n.callAPI(endpoint, data)
 *   - Automatic language injection into request body
 * 
 * Step 3: Backend Receives Language
 *   - In Flask route: language = request.json.get('language', 'ko')
 *   - Store in database or pass to LLM
 * 
 * Step 4: LLM Generates Report
 *   - Python backend example:
 *     @app.route('/api/generate_report', methods=['POST'])
 *     def generate_report():
 *         language = request.json.get('language', 'ko')
 *         diagnosis = request.json.get('diagnosis_result')
 *         
 *         # LLM prompt template
 *         prompt = f"Generate a medical diagnosis report in {language} language..."
 *         report = llm_client.generate(prompt, diagnosis)
 *         
 *         return {
 *           'report': report,
 *           'language': language,
 *           'model_version': '1.0'
 *         }
 * 
 * ============================================================================
 */

/**
 * Export i18n for use in other scripts (ES6 modules)
 */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = i18n;
}
