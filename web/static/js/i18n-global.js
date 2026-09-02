/**
 * Global i18n Language Selector Management
 * Used across all pages in the application
 * 
 * Features:
 * - Global language FAB in bottom-right corner
 * - Persistent language selection across page navigation
 * - Works with all pages that link to i18n.js
 */

(() => {
'use strict';

const languageMetadata = {
    'ko': { flag: '🇰🇷', name: '한국어' },
    'en': { flag: '🇺🇸', name: 'English' },
    'zh': { flag: '🇨🇳', name: '中文' },
    'vi': { flag: '🇻🇳', name: 'Tiếng Việt' },
    'ru': { flag: '🇷🇺', name: 'Русский' },
    'ja': { flag: '🇯🇵', name: '日本語' }
};

/**
 * Initialize global language selector FAB
 * This function should be called after i18n and DOM are ready
 */
function initializeGlobalLanguageSelector() {
    // Wait for i18n to be available
    if (typeof i18n === 'undefined') {
        console.warn('[Global i18n] i18n object not available, retrying...');
        setTimeout(initializeGlobalLanguageSelector, 100);
        return;
    }

    console.log('[Global i18n] Initializing global language selector');

    // Create FAB if it doesn't exist
    if (!document.querySelector('.language-selector-fab')) {
        createLanguageSelectorFab();
    }

    // Update FAB to show current language
    updateGlobalLanguageFab();

    // Register listener for language changes
    if (i18n.onLanguageChange) {
        i18n.onLanguageChange((lang) => {
            console.log(`[Global i18n] Language changed to: ${lang}`);
            updateGlobalLanguageFab();
        });
    }
}

/**
 * Create the language selector FAB element
 */
function createLanguageSelectorFab() {
    // Check if FAB already exists
    if (document.querySelector('.language-selector-fab')) {
        return;
    }

    const fab = document.createElement('div');
    fab.className = 'language-selector-fab';
    fab.id = 'language-selector-fab';

    // Create menu
    const menu = document.createElement('div');
    menu.className = 'language-fab-menu';
    menu.id = 'language-fab-menu';

    // Add language options
    Object.entries(languageMetadata).forEach(([lang, meta]) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'language-fab-option';
        btn.dataset.lang = lang;
        btn.textContent = meta.flag;
        btn.title = meta.name;
        btn.setAttribute('aria-label', meta.name);
        btn.onclick = () => selectLanguageFromGlobalFab(lang);
        menu.appendChild(btn);
    });

    // Create toggle button
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'language-fab-toggle';
    toggle.id = 'language-fab-toggle';
    toggle.title = 'Change Language';
    toggle.setAttribute('aria-label', 'Language Selector');
    toggle.onclick = () => toggleGlobalLanguageFab();

    const flag = document.createElement('span');
    flag.id = 'language-fab-flag';
    flag.textContent = '🇰🇷';
    toggle.appendChild(flag);

    fab.appendChild(menu);
    fab.appendChild(toggle);

    document.body.appendChild(fab);

    console.log('[Global i18n] Language selector FAB created');

    // Add click outside listener
    document.addEventListener('click', (event) => {
        if (!fab.contains(event.target)) {
            closeGlobalLanguageFab();
        }
    });
}

/**
 * Toggle language FAB menu
 */
function toggleGlobalLanguageFab() {
    const menu = document.getElementById('language-fab-menu');
    if (menu) {
        menu.classList.toggle('open');
        console.log('[Global i18n] FAB menu toggled');
    }
}

/**
 * Close language FAB menu
 */
function closeGlobalLanguageFab() {
    const menu = document.getElementById('language-fab-menu');
    if (menu) {
        menu.classList.remove('open');
    }
}

/**
 * Select language from global FAB
 */
function selectLanguageFromGlobalFab(lang) {
    if (!['ko', 'en', 'zh', 'vi', 'ru', 'ja'].includes(lang)) {
        console.warn(`[Global i18n] Invalid language: ${lang}`);
        return;
    }

    console.log(`[Global i18n] Selected language: ${lang}`);

    // Call i18n to change language
    if (typeof i18n !== 'undefined' && i18n.changeLanguage) {
        i18n.changeLanguage(lang);
    }

    // Close menu
    closeGlobalLanguageFab();

    // Update FAB display
    updateGlobalLanguageFab();
}

/**
 * Update global FAB display with current language
 */
function updateGlobalLanguageFab() {
    if (typeof i18n === 'undefined') {
        return;
    }

    const currentLang = i18n.getCurrentLanguage();
    const metadata = languageMetadata[currentLang];

    if (!metadata) {
        console.warn(`[Global i18n] Unknown language: ${currentLang}`);
        return;
    }

    // Update flag in toggle button
    const flagEl = document.getElementById('language-fab-flag');
    if (flagEl) {
        flagEl.textContent = metadata.flag;
    }

    // Update selected state in menu
    document.querySelectorAll('.language-fab-option').forEach(btn => {
        if (btn.dataset.lang === currentLang) {
            btn.classList.add('selected');
        } else {
            btn.classList.remove('selected');
        }
    });

    console.log(`[Global i18n] FAB updated to show: ${currentLang}`);
}

/**
 * Auto-initialize when document is ready
 */
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(initializeGlobalLanguageSelector, 500);
    });
} else {
    // DOM is already loaded
    setTimeout(initializeGlobalLanguageSelector, 100);
}

})();
