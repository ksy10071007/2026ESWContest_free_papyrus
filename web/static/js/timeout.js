// Unified centralized screensaver / timeout logic
(function(){
  if (window.__timeoutScriptLoaded) return; // prevent double-load
  window.__timeoutScriptLoaded = true;

  const IDLE_MS = 180000; // 3 minutes
  const SCREENSAVER_SECONDS = 30; // 30s countdown
  const CACHE_KEYS = ['last_mobile_upload_capture', 'user_id'];

  // expose session init globally
  window.initializeMasterSession = function() {
    try { sessionStorage.clear(); } catch (e) { console.warn('[Session Init] sessionStorage.clear failed', e); }
    CACHE_KEYS.forEach(k=>{ try { localStorage.removeItem(k); } catch(e){ console.warn('[Session Init] remove failed', k, e); } });
    console.log('[Session Init] session cleared');
  };

  // create/ inject overlay DOM once with improved UI
  function injectOverlay(){
    if (document.getElementById('screensaver-overlay')) return document.getElementById('screensaver-overlay');

    const overlay = document.createElement('div');
    overlay.id = 'screensaver-overlay';
    overlay.style.cssText = 'display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(15,23,42,0.93); z-index:9999; justify-content:center; align-items:center;';

    overlay.innerHTML = `
      <div style="text-align: center; max-width: 560px; width: 92%; padding: 28px; background: rgba(30,41,59,0.75); border-radius: 14px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.06); color: #fff;">
        <div style="font-size: 2.6rem; margin-bottom: 8px;">⏳</div>
        <h2 style="font-size: 1.25rem; margin: 0 0 10px 0; font-weight: 700;">터치하지 않으면 처음화면으로 돌아갑니다</h2>
        <p style="color: #94a3b8; margin: 0 0 18px 0; font-size: 0.95rem;">화면을 터치하면 진단이 계속됩니다.</p>
        <div style="width: 100%; height: 8px; background: #334155; border-radius: 999px; overflow: hidden; margin-bottom: 14px;">
          <div id="screensaver-progress" style="width: 100%; height: 100%; background: linear-gradient(90deg, #ef4444, #f43f5e); border-radius: 999px; transition: width 1s linear;"></div>
        </div>
        <div id="screensaver-countdown" style="font-size: 2.2rem; font-weight: 800; color: #f43f5e; margin-bottom: 16px;">${SCREENSAVER_SECONDS}</div>
        <button id="screensaver-cancel" style="width:100%; padding:12px 18px; font-size:1.05rem; font-weight:700; color:#fff; background:#10b981; border:none; border-radius:10px; cursor:pointer;">진단 계속하기</button>
      </div>
    `;

    document.body.appendChild(overlay);

    const cancelBtn = document.getElementById('screensaver-cancel');
    if (cancelBtn) cancelBtn.addEventListener('click', (e)=>{ e.stopPropagation(); hideScreensaver(); resetIdle(); });

    return overlay;
  }

  let idleTimer = null;
  let countdownTimer = null;
  let remaining = SCREENSAVER_SECONDS;
  injectOverlay();

  function updateCountdown(){
    const el = document.getElementById('screensaver-countdown');
    const progress = document.getElementById('screensaver-progress');
    if(el) el.textContent = String(remaining);
    if(progress) { const pct = (remaining / SCREENSAVER_SECONDS) * 100; progress.style.width = pct + '%'; }
  }

  function showScreensaver(){
    const overlay = document.getElementById('screensaver-overlay');
    if(!overlay) return;
    overlay.style.display = 'flex';
    remaining = SCREENSAVER_SECONDS; updateCountdown();
    if(countdownTimer) clearInterval(countdownTimer);
    countdownTimer = setInterval(()=>{
      remaining -= 1;
      if(remaining <= 0){
        clearInterval(countdownTimer); countdownTimer = null;
        try{ window.initializeMasterSession(); }catch(e){}
        window.location.href = '/';
        return;
      }
      updateCountdown();
    }, 1000);
  }

  function hideScreensaver(){
    const overlay = document.getElementById('screensaver-overlay');
    if(!overlay) return;
    overlay.style.display = 'none';
    if(countdownTimer){ clearInterval(countdownTimer); countdownTimer = null; }
    remaining = SCREENSAVER_SECONDS; updateCountdown();
  }

  function resetIdle(){ if(idleTimer) clearTimeout(idleTimer); hideScreensaver(); idleTimer = setTimeout(showScreensaver, IDLE_MS); }

  ['mousemove','mousedown','touchstart','keydown','scroll','touchmove'].forEach(evt=>{ window.addEventListener(evt, resetIdle, {passive:true}); });

  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(resetIdle, 200);
  } else {
    window.addEventListener('DOMContentLoaded', ()=> setTimeout(resetIdle,200));
  }

})();
