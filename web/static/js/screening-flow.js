(function () {
    'use strict';

    const ORDER = ['eye', 'skin', 'scalp'];
    const LABELS = { eye: '안구', skin: '피부', scalp: '두피' };
    const KEYS = {
        plan: 'screening_plan',
        completed: 'screening_completed',
        active: 'screening_active_modality',
        config: 'screening_config_cache'
    };

    function readJson(key, fallback) {
        try {
            const value = sessionStorage.getItem(key);
            return value ? JSON.parse(value) : fallback;
        } catch (error) {
            return fallback;
        }
    }

    function normalize(values) {
        const requested = Array.isArray(values) ? values : [];
        return ORDER.filter((id) => requested.includes(id));
    }

    function getPlan() {
        return normalize(readJson(KEYS.plan, []));
    }

    function setPlan(values) {
        const plan = normalize(values);
        sessionStorage.setItem(KEYS.plan, JSON.stringify(plan));
        return plan;
    }

    function getCompleted() {
        return normalize(readJson(KEYS.completed, []));
    }

    function setCompleted(values) {
        const completed = normalize(values);
        sessionStorage.setItem(KEYS.completed, JSON.stringify(completed));
        return completed;
    }

    function getActive() {
        const active = sessionStorage.getItem(KEYS.active) || '';
        return ORDER.includes(active) ? active : '';
    }

    function setActive(modality) {
        if (ORDER.includes(modality)) {
            sessionStorage.setItem(KEYS.active, modality);
        } else {
            sessionStorage.removeItem(KEYS.active);
        }
    }

    function begin(values) {
        const plan = setPlan(values);
        setCompleted([]);
        setActive(plan[0] || '');
        ORDER.forEach((id) => {
            sessionStorage.removeItem(`screening_capture_${id}`);
            sessionStorage.removeItem(`screening_capture_source_${id}`);
            sessionStorage.removeItem(`screening_result_${id}_status`);
        });
        return plan;
    }

    function markCompleted(modality) {
        if (!ORDER.includes(modality)) return getCompleted();
        const completed = getCompleted();
        if (!completed.includes(modality)) completed.push(modality);
        return setCompleted(completed);
    }

    function nextPending() {
        const completed = getCompleted();
        return getPlan().find((id) => !completed.includes(id)) || '';
    }

    function routeFor(modality) {
        return modality === 'eye' ? '/capture' : '/screening/capture';
    }

    function goTo(modality) {
        if (!ORDER.includes(modality)) {
            window.location.href = '/screening/summary';
            return;
        }
        setActive(modality);
        window.location.href = routeFor(modality);
    }

    function continueFlow() {
        const next = nextPending();
        if (next) {
            goTo(next);
        } else {
            setActive('');
            window.location.href = '/screening/summary';
        }
    }

    async function getConfig() {
        try {
            const response = await fetch('/api/screening/config', { cache: 'no-store' });
            const payload = await response.json();
            if (!response.ok || payload.status !== 'ok') throw new Error('config unavailable');
            sessionStorage.setItem(KEYS.config, JSON.stringify(payload));
            return payload;
        } catch (error) {
            const cached = readJson(KEYS.config, null);
            if (cached) return cached;
            throw error;
        }
    }

    window.ScreeningFlow = {
        ORDER,
        LABELS,
        begin,
        continueFlow,
        getActive,
        getCompleted,
        getConfig,
        getPlan,
        goTo,
        markCompleted,
        nextPending,
        routeFor,
        setActive
    };
})();
