(function () {
    'use strict';

    var allowedPaths = ['/', '/index', '/report', '/result'];
    var currentPath = (window.location.pathname || '').toLowerCase();
    var canRender = allowedPaths.some(function (p) {
        return currentPath === p || currentPath.indexOf(p + '/') === 0;
    });

    if (!canRender) {
        return;
    }

    if (document.getElementById('ai-chat-root')) {
        return;
    }

    // Shared variables
    var root, panel, messages, input, sendBtn, header, title, closeBtn, fab;
    var greetingRow = null;
    var apiKeyConfigured = false;

    function getTranslation(key) {
        // Try to get from i18n engine first
        if (typeof window.i18n !== 'undefined' && window.i18n.getTranslation) {
            var trans = window.i18n.getTranslation(key);
            if (trans) return trans;
        }
        
        // Fallback: try to get directly from translations object
        if (typeof window.translations !== 'undefined' && window.translations) {
            var lang = 'ko'; // default
            try {
                // Get language from localStorage (same way as i18n does)
                var saved = localStorage.getItem('i18n_language') || localStorage.getItem('selectedLanguage') || 'ko';
                lang = String(saved).slice(0, 2).toLowerCase();
            } catch (e) {}
            
            if (window.translations[lang] && window.translations[lang][key]) {
                return window.translations[lang][key];
            }
        }
        
        return '';
    }

    function scrollToBottom() {
        messages.scrollTop = messages.scrollHeight;
    }

    function appendMessage(role, text, opts) {
        var options = opts || {};
        var row = document.createElement('div');
        row.className = 'ai-chat-row ' + role;

        var bubble = document.createElement('div');
        bubble.className = 'ai-chat-bubble';

        if (options.typing) {
            var typing = document.createElement('span');
            typing.className = 'ai-chat-typing';
            typing.innerHTML = '<span class="ai-chat-dot"></span><span class="ai-chat-dot"></span><span class="ai-chat-dot"></span>';
            bubble.appendChild(typing);
        } else {
            bubble.textContent = text;
        }

        row.appendChild(bubble);
        messages.appendChild(row);
        scrollToBottom();

        return row;
    }

    function parseStoredJson(key) {
        var raw = sessionStorage.getItem(key);
        if (raw === null) {
            raw = localStorage.getItem(key);
            if (raw !== null) {
                sessionStorage.setItem(key, raw);
                localStorage.removeItem(key);
            }
        }
        if (!raw) return null;
        try {
            return JSON.parse(raw);
        } catch (e) {
            return null;
        }
    }

    function buildDiagnosisResultPayload() {
        var bilateral = parseStoredJson('bilateral_analysis') || {};
        var leftEye = parseStoredJson('eye_analysis_L');
        var rightEye = parseStoredJson('eye_analysis_R');

        if (leftEye && !bilateral.left_eye) bilateral.left_eye = leftEye;
        if (rightEye && !bilateral.right_eye) bilateral.right_eye = rightEye;

        if (!bilateral.left_eye && !bilateral.right_eye && !bilateral.guide) {
            return { disease: '정보 없음', confidence: null };
        }
        return bilateral;
    }

    async function checkApiKeyStatus() {
        try {
            var response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_message: 'test',
                    diagnosis_result: buildDiagnosisResultPayload()
                })
            });

            var data = {};
            try {
                data = await response.json();
            } catch (e) {}

            // API key is configured if response is not 503 (SERVICE UNAVAILABLE)
            // or if response is 200 (success)
            apiKeyConfigured = response.status !== 503 && response.status !== 500;
            return apiKeyConfigured;
        } catch (e) {
            apiKeyConfigured = false;
            return false;
        }
    }

    function buildAiReply(userText) {
        var text = (userText || '').toLowerCase();

        if (text.indexOf('충혈') >= 0) {
            return '충혈 증상이 반복되면 조명, 수면, 렌즈 착용 시간을 먼저 점검해 보세요. 통증/시력저하가 동반되면 가까운 안과 진료를 권장드립니다.';
        }
        if (text.indexOf('통증') >= 0 || text.indexOf('아파') >= 0) {
            return '통증이 있는 경우는 단순 피로가 아닐 수 있습니다. 증상이 지속되면 즉시 전문의 진료를 받아주세요.';
        }
        if (text.indexOf('렌즈') >= 0) {
            return '렌즈는 권장 착용 시간을 넘기지 않는 것이 중요합니다. 건조감이나 이물감이 있으면 즉시 제거 후 휴식을 권장합니다.';
        }

        return '문의 주신 내용 확인했습니다. 현재 정보는 참고용이며, 정확한 진단은 전문의 진료를 통해 확인해 주세요.';
    }

    async function requestLlmReply(userText) {
        var response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_message: userText,
                diagnosis_result: buildDiagnosisResultPayload()
            })
        });

        var data = {};
        try {
            data = await response.json();
        } catch (e) {
            data = {};
        }

        if (!response.ok || data.status !== 'ok' || !data.reply) {
            var message = data.message || ('chat api failed: ' + response.status);
            throw new Error(message);
        }

        return String(data.reply).trim();
    }

    function setOpenState(open) {
        panel.hidden = !open;
        fab.setAttribute('aria-expanded', open ? 'true' : 'false');

        if (open) {
            input.focus();
            scrollToBottom();

            // Clear old messages and reinitialize with current language
            messages.innerHTML = '';
            
            // Add fresh greeting in current language
            var currentGreeting = getTranslation('chat_greeting') || '안녕하세요. AI 의료 상담입니다. 현재 불편한 증상이나 궁금한 점을 입력해 주세요.';
            greetingRow = appendMessage('ai', currentGreeting);
            
            // Check API key status when opening chat
            checkApiKeyStatus().then(function (hasKey) {
                if (!hasKey) {
                    var apiMsg = getTranslation('chat_api_unconfigured') || '⚠️ API 키가 연결되어 있지 않습니다. 관리자에게 문의해주세요. OpenAI API 키를 설정한 후 서비스를 이용할 수 있습니다.';
                    appendMessage('ai', apiMsg);
                    input.disabled = true;
                    sendBtn.disabled = true;
                } else {
                    input.disabled = false;
                    sendBtn.disabled = false;
                }
            });
        }
    }

    function sendMessage() {
        var text = (input.value || '').trim();
        if (!text) return;

        appendMessage('user', text);
        input.value = '';

        var typingRow = appendMessage('ai', '', { typing: true });
        sendBtn.disabled = true;

        window.setTimeout(async function () {
            try {
                var reply = await requestLlmReply(text);
                typingRow.remove();
                appendMessage('ai', reply || buildAiReply(text));
            } catch (err) {
                console.warn('[chat-widget] LLM API failed, fallback reply used:', err);
                typingRow.remove();
                appendMessage('ai', buildAiReply(text));
            } finally {
                sendBtn.disabled = false;
                input.focus();
            }
        }, 700 + Math.floor(Math.random() * 500));
    }

    function maybeOffsetForFloatingThemeButton() {
        var floatingThemeBtn = document.querySelector('.theme-toggle-floating');
        if (!floatingThemeBtn) return;

        var styles = window.getComputedStyle(floatingThemeBtn);
        var isFixed = styles.position === 'fixed';
        var hasBottom = styles.bottom !== 'auto';
        var hasRight = styles.right !== 'auto';

        if (isFixed && hasBottom && hasRight) {
            root.classList.add('offset-for-theme');
        }
    }

    // Initialize DOM when document is ready
    function initializeChat() {
        root = document.createElement('div');
        root.id = 'ai-chat-root';
        root.className = 'ai-chat-root';

        panel = document.createElement('section');
        panel.className = 'ai-chat-panel';
        panel.hidden = true;
        panel.setAttribute('aria-label', 'AI 의료 상담 창');

        header = document.createElement('div');
        header.className = 'ai-chat-header';

        title = document.createElement('div');
        title.className = 'ai-chat-title';
        title.textContent = getTranslation('chat_title') || 'AI 의료 상담';

        closeBtn = document.createElement('button');
        closeBtn.className = 'ai-chat-close';
        closeBtn.type = 'button';
        closeBtn.setAttribute('aria-label', getTranslation('chat_close_btn') || '채팅 닫기');
        closeBtn.textContent = '×';

        header.appendChild(title);
        header.appendChild(closeBtn);

        messages = document.createElement('div');
        messages.className = 'ai-chat-messages';

        var inputWrap = document.createElement('div');
        inputWrap.className = 'ai-chat-input-wrap';

        input = document.createElement('input');
        input.className = 'ai-chat-input';
        input.type = 'text';
        input.placeholder = getTranslation('chat_placeholder') || '증상이나 궁금한 점을 입력하세요.';
        input.setAttribute('maxlength', '300');

        sendBtn = document.createElement('button');
        sendBtn.className = 'ai-chat-send';
        sendBtn.type = 'button';
        sendBtn.textContent = 'Send';

        inputWrap.appendChild(input);
        inputWrap.appendChild(sendBtn);

        panel.appendChild(header);
        panel.appendChild(messages);
        panel.appendChild(inputWrap);

        fab = document.createElement('button');
        fab.className = 'ai-chat-fab';
        fab.type = 'button';
        fab.setAttribute('aria-label', 'AI 의료 상담 열기');
        fab.setAttribute('aria-expanded', 'false');
        fab.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3C6.48 3 2 6.94 2 11.8c0 2.8 1.52 5.28 3.89 6.9V22l3.17-1.73c.93.24 1.91.37 2.94.37 5.52 0 10-3.94 10-8.8S17.52 3 12 3zm0 15.2c-.93 0-1.81-.13-2.62-.36l-.75-.22-1.86 1.02.03-1.98-.63-.42C4.15 14.84 3.2 13.39 3.2 11.8c0-4.18 3.94-7.6 8.8-7.6s8.8 3.42 8.8 7.6-3.94 7.6-8.8 7.6z"></path></svg>';

        root.appendChild(panel);
        root.appendChild(fab);
        document.body.appendChild(root);

        // Event listeners
        fab.addEventListener('click', function () {
            setOpenState(panel.hidden);
        });

        closeBtn.addEventListener('click', function () {
            setOpenState(false);
        });

        sendBtn.addEventListener('click', sendMessage);

        input.addEventListener('keyup', function (event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        });

        // Listen for language changes to update UI text
        if (typeof window.i18n !== 'undefined' && window.i18n.onLanguageChange) {
            window.i18n.onLanguageChange(function (newLang) {
                // Update title
                var newTitle = getTranslation('chat_title') || 'AI 의료 상담';
                title.textContent = newTitle;
                
                // Update placeholder
                var newPlaceholder = getTranslation('chat_placeholder') || '증상이나 궁금한 점을 입력하세요.';
                input.placeholder = newPlaceholder;
                
                // Update close button label
                var newCloseLabel = getTranslation('chat_close_btn') || '채팅 닫기';
                closeBtn.setAttribute('aria-label', newCloseLabel);
                
                // If chat panel is open, refresh the messages in the new language
                if (!panel.hidden) {
                    messages.innerHTML = '';
                    
                    // Add fresh greeting in new language
                    var newGreeting = getTranslation('chat_greeting') || '안녕하세요. AI 의료 상담입니다. 현재 불편한 증상이나 궁금한 점을 입력해 주세요.';
                    greetingRow = appendMessage('ai', newGreeting);
                    
                    // Check API key status and show message if needed
                    if (!apiKeyConfigured) {
                        var newApiMsg = getTranslation('chat_api_unconfigured') || '⚠️ API 키가 연결되어 있지 않습니다. 관리자에게 문의해주세요. OpenAI API 키를 설정한 후 서비스를 이용할 수 있습니다.';
                        appendMessage('ai', newApiMsg);
                    }
                }
            });
        }

        maybeOffsetForFloatingThemeButton();
    }

    // Wait for DOM to be ready before initializing
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeChat);
    } else {
        initializeChat();
    }
})();
