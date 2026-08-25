let messageCount = 0;
let uptimeStartMs = null;

function clearEmptyState() {
  const empty = document.getElementById('empty-state');
  if (empty) empty.remove();
}

function scrollToBottom() {
  const el = document.getElementById('messages');
  el.scrollTop = el.scrollHeight;
}

function formatTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function hideTypingIndicator() {
  const existing = document.getElementById('typing-indicator');
  if (existing) existing.remove();
}

function showTypingIndicator() {
  if (document.getElementById('typing-indicator')) return;
  clearEmptyState();
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'typing-dots';
  div.id = 'typing-indicator';
  div.innerHTML = '<span></span><span></span><span></span>';
  container.appendChild(div);
  scrollToBottom();
}

function addMessage(role, cssClass, text) {
  hideTypingIndicator();
  clearEmptyState();
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg ' + cssClass;
  div.innerHTML =
    '<div class="role">' + role + '</div>' +
    escapeHtml(text) +
    '<div class="timestamp">' + formatTime() + '</div>';
  container.appendChild(div);
  scrollToBottom();
}

function addUserMessage(text) {
  addMessage('You', 'user', text);
  messageCount += 1;
  document.getElementById('message-count').textContent = String(messageCount);
}

function addBtMessage(text) {
  addMessage('BT', 'bt', text);
}

function addErrorMessage(text) {
  addMessage('BT', 'error', text);
}

function setSessionInfo(wakePhrase, modelName) {
  document.getElementById('wake-phrase').textContent = wakePhrase;
  document.getElementById('model-name').textContent = modelName;
}

function setWeather(city, temp, description) {
  document.getElementById('weather-temp').textContent = temp;
  document.getElementById('weather-desc').textContent = description;
  document.getElementById('weather-city').textContent = city;
  document.getElementById('header-weather-temp').textContent = temp;
  document.getElementById('header-weather-city').textContent = city;
}

function setWeatherUnavailable() {
  document.getElementById('weather-temp').textContent = '—';
  document.getElementById('weather-desc').textContent = 'Unavailable';
  document.getElementById('weather-city').textContent = 'Check your internet connection';
  document.getElementById('header-weather-temp').textContent = '—';
  document.getElementById('header-weather-city').textContent = 'Unavailable';
}

function tickHeaderClock() {
  const now = new Date();
  const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const date = now.toLocaleDateString([], { month: 'long', day: 'numeric', year: 'numeric' });
  document.getElementById('header-clock').textContent = time + '  |  ' + date;
}

function openConfigFolder() {
  if (bridgeReady) {
    window.pywebview.api.open_config_folder();
  }
}

const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 26;

function setGauge(id, percent) {
  const fill = document.getElementById(id);
  const offset = GAUGE_CIRCUMFERENCE * (1 - Math.max(0, Math.min(100, percent)) / 100);
  fill.style.strokeDasharray = GAUGE_CIRCUMFERENCE;
  fill.style.strokeDashoffset = offset;
}

function setSystemStats(cpuPercent, ramPercent, ramDetail, diskPercent, diskDetail) {
  document.getElementById('cpu-value').textContent = cpuPercent + '%';
  setGauge('cpu-gauge', cpuPercent);
  document.getElementById('ram-value').textContent = ramPercent + '%';
  setGauge('ram-gauge', ramPercent);
  document.getElementById('ram-detail').textContent = ramDetail;
  document.getElementById('disk-value').textContent = diskPercent + '%';
  setGauge('disk-gauge', diskPercent);
  document.getElementById('disk-detail').textContent = diskDetail;
}

function clearConversation() {
  const container = document.getElementById('messages');
  container.innerHTML = '<div id="empty-state">Type a message, or say &quot;Hey Jarvis&quot;, to talk to BT.</div>';
  messageCount = 0;
  document.getElementById('message-count').textContent = '0';
}

function formatUptime(ms) {
  const totalSeconds = Math.floor(ms / 1000);
  const h = String(Math.floor(totalSeconds / 3600)).padStart(2, '0');
  const m = String(Math.floor((totalSeconds % 3600) / 60)).padStart(2, '0');
  const s = String(totalSeconds % 60).padStart(2, '0');
  return h + ':' + m + ':' + s;
}

function tickUptime() {
  if (uptimeStartMs === null) return;
  document.getElementById('uptime').textContent = formatUptime(Date.now() - uptimeStartMs);
}

function setStatus(status) {
  const dot = document.getElementById('status-dot');
  const label = document.getElementById('status-label');
  dot.className = 'orb ' + status;
  const labels = {
    idle: 'Idle',
    listening: 'Listening for wake word…',
    thinking: 'Thinking…',
    speaking: 'Speaking…',
  };
  label.textContent = labels[status] || status;

  if (status === 'thinking') {
    showTypingIndicator();
  } else {
    hideTypingIndicator();
  }

  const idle = status === 'idle';
  const micBtn = document.getElementById('mic-btn');
  const textInput = document.getElementById('text-input');
  const sendBtn = document.getElementById('send-btn');
  if (bridgeReady) {
    micBtn.disabled = !idle;
    textInput.disabled = !idle;
    sendBtn.disabled = !idle;
    textInput.placeholder = idle ? 'Type a message...' : (labels[status] || status);
  }
  micBtn.className = 'icon-btn' + (idle ? '' : ' ' + status);
  micBtn.title = idle ? 'Start listening' : (labels[status] || status);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

let bridgeReady = false;

function startListening() {
  if (bridgeReady) {
    window.pywebview.api.start_listening();
  }
}

function sendText(event) {
  event.preventDefault();
  const input = document.getElementById('text-input');
  const text = input.value.trim();
  if (!text || !bridgeReady) {
    return false;
  }
  input.value = '';
  window.pywebview.api.send_text_message(text);
  return false;
}

window.addEventListener('pywebviewready', function () {
  bridgeReady = true;
  uptimeStartMs = Date.now();
  setInterval(tickUptime, 1000);
  tickUptime();
  const badge = document.getElementById('connection-badge');
  badge.classList.add('online');
  document.getElementById('connection-label').textContent = 'Online';
  setStatus('idle');
});

setInterval(tickHeaderClock, 1000);
tickHeaderClock();
