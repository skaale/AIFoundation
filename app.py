import base64
import html as html_module
import io
import warnings
import wave

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated",
    category=FutureWarning,
)

import gradio as gr
import numpy as np

from supra_reasoning.constants import (
    CHURCH_INTRODUCTION,
    IDLE_SECONDS_BEFORE_PROMPT,
    MODEL_ID,
    PRIEST_VOICE,
    SILENCE_PROMPT,
)
from supra_reasoning.conversation import (
    INTERRUPT_THRESHOLD,
    STREAM_CHUNK_SECONDS,
    ConversationState,
    chunk_energy,
    ingest_stream_chunk,
    normalize_audio,
    store_speech_chunk,
)
from supra_reasoning.debug import debug_log, debug_mic_tick, debug_view
from supra_reasoning.model import SupraReasoningModel
from supra_reasoning.speech import captioned_priest_voice
from supra_reasoning.stt import SpeechToText
from supra_reasoning.tts import SAMPLE_RATE, VOICES, KokoroTTS, pop_speakable_phrases

SETTINGS_BTN_HTML = """
<button type="button" id="settings-open" title="Settings" aria-label="Settings">&#8942;</button>
"""

AFTER_INTRO_JS = """
() => {
    if (window.__asiIntroHandled) return [];
    const log = (msg) => console.log('[ASI-DEBUG]', msg);
    const findIntroAudio = () => {
        const root = document.getElementById('priest-audio');
        if (!root) return null;
        const tagged = root.querySelector('audio[data-intro="1"]');
        if (tagged?.src) return tagged;
        return root.querySelector('audio[src]') || root.querySelector('audio');
    };
    let tries = 0;
    const boot = () => {
        if (window.__asiIntroHandled) return;
        tries += 1;
        const audio = findIntroAudio();
        if (!audio?.src) {
            if (tries < 20) setTimeout(boot, 300);
            else if (window.__asiArmListening) window.__asiArmListening();
            return;
        }
        if (audio.dataset.introPlayed === '1' || (!audio.paused && !audio.ended)) {
            window.__asiIntroHandled = true;
            return;
        }
        window.__asiIntroHandled = true;
        audio.dataset.introPlayed = '1';
        log(`intro audio: play once (${audio.duration || '?'}s)`);
        if (window.__asiUnlockAudio) window.__asiUnlockAudio();
        audio.play().catch((err) => log(`intro play blocked: ${err}`));
    };
    log('conversation: waiting for intro audio element');
    setTimeout(boot, 400);
    return [];
}
"""

VISUALIZER_HTML = """
<div class="voice-hub">
    <canvas id="viz-canvas" width="640" height="640" aria-label="Voice activity"></canvas>
    <div id="viz-text" class="voice-hub-text"></div>
    <div class="voice-hub-core">
        <span class="voice-hub-dot" id="viz-dot"></span>
        <span class="voice-hub-label" id="viz-label">Ready</span>
    </div>
</div>
"""

BOOT_JS = """
() => {
    if (!window.__asiUnlockReady) {
        window.__asiUnlockReady = true;
        window.__asiUnlockAudio = () => {
            const actx = new (window.AudioContext || window.webkitAudioContext)();
            actx.resume();
            const speaker = document.querySelector('#priest-audio audio');
            if (speaker && speaker.paused && speaker.src) {
                speaker.play().catch(() => {});
            }
        };
        const unlock = () => {
            window.__asiUnlockAudio();
            document.removeEventListener('pointerdown', unlock);
            document.removeEventListener('keydown', unlock);
        };
        document.addEventListener('pointerdown', unlock, { once: true });
        document.addEventListener('keydown', unlock, { once: true });
    }

    if (window.__asiVizRunning) return [];
    window.__asiVizRunning = true;

    const init = () => {
        const canvas = document.getElementById('viz-canvas');
        if (!canvas) return false;

        const ctx = canvas.getContext('2d');
        const bars = 72;
        const levels = new Float32Array(bars);
        let micAnalyser = null;
        let speakerAnalyser = null;
        let speakerNode = null;
        let audioCtx = null;
        let phase = 0;
        let mode = 'idle';

        const statusText = () => {
            const box = document.querySelector('#status-box textarea');
            return (box?.value || '').toLowerCase();
        };

        const priestAudioPlaying = () => {
            const audio = document.querySelector('#priest-audio audio');
            return !!(audio && audio.src && !audio.paused && !audio.ended);
        };

        const syncCircleText = () => {
            if (priestAudioPlaying()) return;
            const dst = document.getElementById('viz-text');
            const src = document.getElementById('circle-text-src');
            if (!dst || !src) return;
            const content = src.querySelector('.circle-lines');
            const html = content ? content.outerHTML : src.innerHTML;
            if (!html || html.includes('[object Object]') || html.includes('__type__')) return;
            dst.innerHTML = html;
        };

        const syncMode = () => {
            syncCircleText();
            const text = statusText().trim();
            if (text === 'speak' || text === 'arrive' || text === 'think') {
                mode = 'speak';
            } else if (text === 'listen' || text === 'hear') {
                mode = 'listen';
                hookMic();
                if (window.__asiBrowserMic) window.__asiBrowserMic.start();
            } else {
                mode = 'idle';
            }
            const label = document.getElementById('viz-label');
            const dot = document.getElementById('viz-dot');
            if (label) {
                if (text === 'hear') label.textContent = 'Hearing';
                else if (mode === 'speak') label.textContent = 'Priest';
                else if (mode === 'listen') label.textContent = 'Listening';
                else label.textContent = 'Ready';
            }
            if (dot) dot.dataset.mode = mode;
        };

        const ensureAudioCtx = () => {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            if (audioCtx.state === 'suspended') audioCtx.resume();
            return audioCtx;
        };

        const hookMic = async () => {
            if (micAnalyser) return;
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                const actx = ensureAudioCtx();
                const source = actx.createMediaStreamSource(stream);
                micAnalyser = actx.createAnalyser();
                micAnalyser.fftSize = 256;
                micAnalyser.smoothingTimeConstant = 0.75;
                source.connect(micAnalyser);
            } catch (_) {}
        };

        const hookSpeaker = () => {
            const audio = document.querySelector('#priest-audio audio');
            if (!audio || speakerAnalyser) return;
            try {
                const actx = ensureAudioCtx();
                if (!speakerNode) {
                    speakerNode = actx.createMediaElementSource(audio);
                    speakerAnalyser = actx.createAnalyser();
                    speakerAnalyser.fftSize = 256;
                    speakerAnalyser.smoothingTimeConstant = 0.65;
                    speakerNode.connect(speakerAnalyser);
                    speakerNode.connect(actx.destination);
                }
            } catch (_) {}
        };

        const readAnalyser = (analyser, fallback) => {
            if (!analyser) return fallback;
            const data = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(data);
            let sum = 0;
            for (let i = 0; i < data.length; i += 1) sum += data[i];
            return Math.min(1, (sum / data.length) / 90);
        };

        const readMicLevel = () => {
            const input = document.querySelector('#mic-level input');
            const backend = parseFloat(input?.value || '0');
            if (!Number.isNaN(backend) && backend > 0) return Math.min(1, backend);
            return readAnalyser(micAnalyser, 0);
        };

        const draw = () => {
            syncMode();
            hookSpeaker();

            const w = canvas.width;
            const h = canvas.height;
            const cx = w / 2;
            const cy = h / 2;
            const baseRadius = 158;
            const maxBar = 82;

            let energy = 0;
            if (mode === 'speak') energy = readAnalyser(speakerAnalyser, 0.12);
            else if (mode === 'listen') energy = Math.max(readMicLevel(), readAnalyser(micAnalyser, 0));
            else energy = 0.08 + Math.sin(phase) * 0.04;

            phase += 0.05;

            for (let i = 0; i < bars; i += 1) {
                const target = energy * (0.55 + 0.45 * Math.sin(phase + i * 0.35));
                levels[i] += (target - levels[i]) * 0.22;
            }

            ctx.clearRect(0, 0, w, h);

            const ringGlow = ctx.createRadialGradient(cx, cy, baseRadius - 10, cx, cy, baseRadius + maxBar + 20);
            if (mode === 'speak') {
                ringGlow.addColorStop(0, 'rgba(251, 191, 36, 0.08)');
                ringGlow.addColorStop(1, 'rgba(251, 191, 36, 0)');
            } else {
                ringGlow.addColorStop(0, 'rgba(124, 58, 237, 0.12)');
                ringGlow.addColorStop(1, 'rgba(124, 58, 237, 0)');
            }
            ctx.fillStyle = ringGlow;
            ctx.beginPath();
            ctx.arc(cx, cy, baseRadius + maxBar + 20, 0, Math.PI * 2);
            ctx.fill();

            for (let i = 0; i < bars; i += 1) {
                const angle = (i / bars) * Math.PI * 2 - Math.PI / 2;
                const amp = levels[i];
                const inner = baseRadius;
                const outer = baseRadius + amp * maxBar;
                const x1 = cx + Math.cos(angle) * inner;
                const y1 = cy + Math.sin(angle) * inner;
                const x2 = cx + Math.cos(angle) * outer;
                const y2 = cy + Math.sin(angle) * outer;
                ctx.strokeStyle = mode === 'speak'
                    ? `rgba(251, 191, 36, ${0.35 + amp * 0.65})`
                    : `rgba(124, 58, 237, ${0.3 + amp * 0.7})`;
                ctx.lineWidth = 3;
                ctx.lineCap = 'round';
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();
            }

            ctx.beginPath();
            ctx.arc(cx, cy, baseRadius - 6, 0, Math.PI * 2);
            ctx.strokeStyle = mode === 'speak' ? 'rgba(251, 191, 36, 0.35)' : 'rgba(124, 58, 237, 0.35)';
            ctx.lineWidth = 2;
            ctx.stroke();

            requestAnimationFrame(draw);
        };

        const setStatusMode = (mode) => {
            const box = document.querySelector('#status-box textarea');
            if (!box) return;
            box.value = mode;
            box.dispatchEvent(new Event('input', { bubbles: true }));
        };

        const bindCaptionSync = (audio) => {
            if (!audio || audio.dataset.asiCaptionBound) return;
            const fullText = (audio.dataset.caption || '').trim();
            if (!fullText) return;
            audio.dataset.asiCaptionBound = '1';
            const words = fullText.split(/\\s+/).filter(Boolean);
            if (!words.length) return;
            const paint = (count) => {
                const dst = document.getElementById('viz-text');
                if (!dst) return;
                const shown = words.slice(0, Math.max(1, count)).join(' ');
                dst.innerHTML = (
                    '<div class="circle-lines"><p class="circle-line circle-line--priest">'
                    + '<span class="circle-role">Priest</span>'
                    + `<span class="circle-copy">${shown}</span>`
                    + '</p></div>'
                );
            };
            const reveal = () => {
                if (!audio.duration || !Number.isFinite(audio.duration)) return;
                const n = Math.ceil((audio.currentTime / audio.duration) * words.length);
                paint(Math.min(words.length, Math.max(1, n)));
            };
            audio.addEventListener('play', () => {
                if (audio.dataset.intro === '1') audio.dataset.introPlayed = '1';
                setStatusMode('speak');
                paint(1);
            });
            audio.addEventListener('timeupdate', reveal);
            audio.addEventListener('ended', () => {
                paint(words.length);
                if (audio.dataset.intro === '1') {
                    if (window.__asiArmListening) window.__asiArmListening();
                } else {
                    setStatusMode('listen');
                }
            });
        };

        const watchSpeaker = () => {
            const root = document.getElementById('priest-audio');
            if (!root) return;
            let lastAutoplaySrc = '';
            const playLatest = () => {
                const audio = root.querySelector('audio[src], video[src]') || root.querySelector('audio, video');
                if (!audio?.src) return;
                bindCaptionSync(audio);
                if (audio.dataset.intro === '1') {
                    if (audio.dataset.introPlayed === '1' || window.__asiIntroHandled) return;
                    return;
                }
                if (audio.ended) return;
                if (!audio.paused) return;
                if (audio.src === lastAutoplaySrc) return;
                lastAutoplaySrc = audio.src;
                if (window.__asiUnlockAudio) window.__asiUnlockAudio();
                audio.play().catch(() => {});
            };
            const obs = new MutationObserver(() => playLatest());
            obs.observe(root, { childList: true, subtree: true, attributes: true });
            playLatest();
        };

        watchSpeaker();
        draw();
        return true;
    };

    window.__asiInitVisualizer = () => {
        let tries = 0;
        const tick = () => {
            if (init() || tries > 30) return;
            tries += 1;
            setTimeout(tick, 200);
        };
        tick();
    };
    window.__asiInitVisualizer();

    if (!window.__asiSettingsReady) {
        window.__asiSettingsReady = true;
        const bindSettings = () => {
            const btn = document.getElementById('settings-open');
            const drawer = document.getElementById('settings-drawer');
            if (!btn || !drawer) return false;
            btn.addEventListener('click', (event) => {
                event.stopPropagation();
                drawer.classList.toggle('is-open');
            });
            document.addEventListener('click', (event) => {
                if (!drawer.classList.contains('is-open')) return;
                if (drawer.contains(event.target) || btn.contains(event.target)) return;
                drawer.classList.remove('is-open');
            });
            return true;
        };
        let settingsTries = 0;
        const waitSettings = () => {
            if (bindSettings() || settingsTries > 40) return;
            settingsTries += 1;
            setTimeout(waitSettings, 200);
        };
        waitSettings();
    }

    if (!window.__asiFindField) {
        window.__asiFindField = (elemId) => {
            const root = document.getElementById(elemId);
            if (!root) return null;
            return (
                root.querySelector('textarea')
                || root.querySelector('input[type="text"]')
                || root.querySelector('input:not([type="checkbox"]):not([type="hidden"])')
            );
        };
        window.__asiPulseField = (field) => {
            if (!field) return;
            field.dispatchEvent(new Event('input', { bubbles: true }));
            field.dispatchEvent(new Event('change', { bubbles: true }));
        };
        window.__asiPulseListenTick = () => {
            setTimeout(() => {
                const tickField = window.__asiFindField('listen-tick');
                if (!tickField) return;
                tickField.value = String(Number(tickField.value || '0') + 1);
                window.__asiPulseField(tickField);
            }, 80);
        };
        window.__asiArmListening = () => {
            if (window.__asiMicArmed) return;
            window.__asiMicArmed = true;
            console.log('[ASI-DEBUG] arming listen after priest audio ended');
            window.__asiPulseListenTick();
            if (window.__asiBrowserMic) window.__asiBrowserMic.start();
        };
    }

    if (!window.__asiBrowserMic) {
        window.__asiBrowserMic = (() => {
            let active = false;
            let starting = false;
            let mediaStream = null;
            let audioCtx = null;
            let analyser = null;
            let processor = null;
            let collecting = false;
            let buffers = [];
            let silenceMs = 0;
            let speechMs = 0;
            const SPEECH_RMS = 0.018;
            const SILENCE_MS = 1200;
            const MIN_SPEECH_MS = 250;

            const statusText = () => {
                const box = document.querySelector('#status-box textarea');
                return (box?.value || '').trim().toLowerCase();
            };

            const pushMicLevel = (level) => {
                const input = document.querySelector('#mic-level input');
                if (!input) return;
                input.value = String(Math.min(1, Math.max(0, level)));
                input.dispatchEvent(new Event('input', { bubbles: true }));
            };

            const findField = (elemId) => {
                const root = document.getElementById(elemId);
                if (!root) return null;
                return (
                    root.querySelector('textarea')
                    || root.querySelector('input[type="text"]')
                    || root.querySelector('input:not([type="checkbox"]):not([type="hidden"])')
                );
            };

            const pulseInput = (field) => {
                if (!field) return;
                field.dispatchEvent(new Event('input', { bubbles: true }));
                field.dispatchEvent(new Event('change', { bubbles: true }));
            };

            const submitMicB64 = (b64, attempt = 0) => {
                const field = findField('mic-b64');
                const tickField = findField('mic-tick');
                if (!field || !tickField) {
                    if (attempt < 40) {
                        setTimeout(() => submitMicB64(b64, attempt + 1), 200);
                        return;
                    }
                    console.log('[ASI-DEBUG] browser mic: hidden fields not mounted (#mic-b64 / #mic-tick)');
                    return;
                }
                field.value = b64;
                pulseInput(field);
                setTimeout(() => {
                    tickField.value = String(Number(tickField.value || '0') + 1);
                    pulseInput(tickField);
                }, 80);
                console.log(`[ASI-DEBUG] browser mic: queued ${b64.length} chars`);
            };

            const encodeWav = (samples, sampleRate) => {
                const buffer = new ArrayBuffer(44 + samples.length * 2);
                const view = new DataView(buffer);
                const writeString = (offset, str) => {
                    for (let i = 0; i < str.length; i += 1) {
                        view.setUint8(offset + i, str.charCodeAt(i));
                    }
                };
                writeString(0, 'RIFF');
                view.setUint32(4, 36 + samples.length * 2, true);
                writeString(8, 'WAVE');
                writeString(12, 'fmt ');
                view.setUint32(16, 16, true);
                view.setUint16(20, 1, true);
                view.setUint16(22, 1, true);
                view.setUint32(24, sampleRate, true);
                view.setUint32(28, sampleRate * 2, true);
                view.setUint16(32, 2, true);
                view.setUint16(34, 16, true);
                writeString(36, 'data');
                view.setUint32(40, samples.length * 2, true);
                let offset = 44;
                for (let i = 0; i < samples.length; i += 1) {
                    const s = Math.max(-1, Math.min(1, samples[i]));
                    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                    offset += 2;
                }
                return buffer;
            };

            const flushUtterance = () => {
                if (!buffers.length) return;
                const total = buffers.reduce((sum, part) => sum + part.length, 0);
                if (total < Math.floor((audioCtx?.sampleRate || 48000) * 0.15)) {
                    buffers = [];
                    collecting = false;
                    silenceMs = 0;
                    speechMs = 0;
                    return;
                }
                const merged = new Float32Array(total);
                let pos = 0;
                for (const part of buffers) {
                    merged.set(part, pos);
                    pos += part.length;
                }
                const wav = encodeWav(merged, audioCtx.sampleRate);
                const bytes = new Uint8Array(wav);
                let binary = '';
                for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
                submitMicB64(btoa(binary));
                buffers = [];
                collecting = false;
                silenceMs = 0;
                speechMs = 0;
            };

            const priestSpeaking = () => {
                const audio = document.querySelector('#priest-audio audio');
                return !!(audio && audio.src && !audio.paused && !audio.ended);
            };

            const tick = () => {
                if (!active) return;
                const mode = statusText();
                if (priestSpeaking() || mode === 'speak' || mode === 'think' || mode === 'arrive') {
                    collecting = false;
                    buffers = [];
                    silenceMs = 0;
                    speechMs = 0;
                    setTimeout(tick, 50);
                    return;
                }
                if (mode !== 'listen' && mode !== 'hear') {
                    setTimeout(tick, 50);
                    return;
                }
                if (!analyser) {
                    requestAnimationFrame(tick);
                    return;
                }
                const timeData = new Uint8Array(analyser.fftSize);
                analyser.getByteTimeDomainData(timeData);
                let sum = 0;
                for (let i = 0; i < timeData.length; i += 1) {
                    const v = (timeData[i] - 128) / 128;
                    sum += v * v;
                }
                const rms = Math.sqrt(sum / timeData.length);
                pushMicLevel(Math.min(1, rms * 6));

                if (rms >= SPEECH_RMS) {
                    if (!collecting) collecting = true;
                    speechMs += 50;
                    silenceMs = 0;
                } else if (collecting) {
                    silenceMs += 50;
                    if (silenceMs >= SILENCE_MS && speechMs >= MIN_SPEECH_MS) {
                        flushUtterance();
                    }
                }
                setTimeout(tick, 50);
            };

            const start = async () => {
                if (active || starting) return;
                starting = true;
                try {
                    mediaStream = await navigator.mediaDevices.getUserMedia({
                        audio: { echoCancellation: true, noiseSuppression: true },
                        video: false,
                    });
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    const source = audioCtx.createMediaStreamSource(mediaStream);
                    analyser = audioCtx.createAnalyser();
                    analyser.fftSize = 2048;
                    analyser.smoothingTimeConstant = 0.35;
                    processor = audioCtx.createScriptProcessor(4096, 1, 1);
                    processor.onaudioprocess = (event) => {
                        if (!collecting) return;
                        const input = event.inputBuffer.getChannelData(0);
                        buffers.push(new Float32Array(input));
                    };
                    source.connect(analyser);
                    analyser.connect(processor);
                    processor.connect(audioCtx.destination);
                    active = true;
                    console.log('[ASI-DEBUG] browser mic: listening');
                    tick();
                } catch (err) {
                    console.log('[ASI-DEBUG] browser mic: failed', err);
                } finally {
                    starting = false;
                }
            };

            return { start };
        })();
    }
    return [];
}
"""

print("Loading Supra model…")
engine = SupraReasoningModel(device="auto")
print(f"Supra ready on {engine.torch_device} ({engine.dtype}).")

print("Loading Kokoro TTS…")
KokoroTTS(voice=VOICES[PRIEST_VOICE], use_gpu=True)
print("Kokoro ready.")

print("Loading speech-to-text…")
stt = SpeechToText()
print("Warming up speech-to-text…")
stt.transcribe((16000, np.zeros(1600, dtype=np.float32)))
print("STT ready.")

IDLE_CHUNK_LIMIT = max(4, int(IDLE_SECONDS_BEFORE_PROMPT / STREAM_CHUNK_SECONDS))
MODE_LISTEN = "listen"
MODE_SPEAK = "speak"
MODE_HEAR = "hear"
MODE_THINK = "think"
MODE_ARRIVE = "arrive"


def priest_playback(chunks: list[np.ndarray]):
    if not chunks:
        return gr.skip()
    audio = np.concatenate(chunks).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return SAMPLE_RATE, np.ascontiguousarray(audio, dtype=np.float32)


def priest_audio_html(playback, caption: str = "", *, intro: bool = False) -> str:
    if playback is None:
        return gr.skip()
    if isinstance(playback, tuple) and len(playback) == 2:
        sample_rate, audio = playback
    else:
        return gr.skip()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return gr.skip()
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    pcm = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(int(sample_rate))
        wav_out.writeframes(pcm.tobytes())
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    caption = (caption or "").strip()
    caption_attr = (
        f' data-caption="{html_module.escape(caption, quote=True)}"' if caption else ""
    )
    intro_attr = ' data-intro="1"' if intro else ""
    return f'<audio autoplay{caption_attr}{intro_attr} src="data:audio/wav;base64,{encoded}"></audio>'


def format_chat_answer(answer: str) -> str:
    return answer.strip() or "…"


def circle_text_html(text: str, role: str = "assistant") -> str:
    content = html_module.escape((text or "").strip()) or "…"
    role_label = "You" if role == "user" else "Priest"
    role_class = "circle-line--user" if role == "user" else "circle-line--priest"
    return (
        '<div class="circle-lines">'
        f'<p class="circle-line {role_class}">'
        f'<span class="circle-role">{role_label}</span>'
        f'<span class="circle-copy">{content}</span>'
        '</p></div>'
    )


def circle_from_history(history: list[dict]) -> str:
    if not history:
        return circle_awaiting_html()
    last = history[-1]
    return circle_text_html(last.get("content", ""), last.get("role", "assistant"))


def circle_awaiting_html() -> str:
    return (
        '<div class="circle-lines">'
        '<p class="circle-line circle-line--await">'
        '<span class="circle-copy">Awaiting conversation</span>'
        '</p></div>'
    )


def circle_out(state: ConversationState, html: str | None = None) -> str:
    if html is not None:
        state.circle_html = html
    return state.circle_html or circle_awaiting_html()


def circle_keep(state: ConversationState) -> str:
    return circle_out(state)


def out(
    state: ConversationState,
    chatbot,
    thinking,
    priest,
    status,
    mic_level,
    circle,
    msg: str | None = None,
    level: str = "INFO",
):
    debug = debug_log(state, msg, level) if msg else debug_view(state)
    return chatbot, thinking, priest, status, state, mic_level, circle, debug


def bout(
    state: ConversationState,
    mic_val: str,
    chatbot,
    thinking,
    priest,
    status,
    mic_level,
    circle,
    msg: str | None = None,
    level: str = "INFO",
):
    return (mic_val,) + out(state, chatbot, thinking, priest, status, mic_level, circle, msg, level)


def decode_b64_wav(b64: str) -> tuple[int, np.ndarray] | None:
    try:
        raw = (b64 or "").strip()
        if not raw:
            return None
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]
        data = base64.b64decode(raw)
        with wave.open(io.BytesIO(data), "rb") as wav_in:
            sample_rate = wav_in.getframerate()
            channels = wav_in.getnchannels()
            pcm = np.frombuffer(wav_in.readframes(wav_in.getnframes()), dtype=np.int16)
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        if audio.size == 0:
            return None
        return int(sample_rate), audio
    except Exception:
        return None


def stream_words(text: str, chunk_size: int = 3):
    words = text.split()
    if not words:
        yield text
        return
    for index in range(0, len(words), chunk_size):
        yield " ".join(words[: index + chunk_size])


def stream_display_snippet(text: str, words_per_line: int = 6) -> str:
    words = (text or "").split()
    if not words:
        return "…"
    return " ".join(words[-words_per_line:])


def history_with_draft_user(history: list[dict]) -> list[dict]:
    if history and history[-1].get("role") == "user" and history[-1].get("content") == "…":
        return history
    updated = list(history)
    updated.append({"role": "user", "content": "…"})
    return updated


def listening_mode(state: ConversationState) -> str:
    if state.speaking or state.draft_user:
        return MODE_HEAR
    if state.introducing:
        return MODE_ARRIVE
    if state.introduced:
        return MODE_LISTEN
    return MODE_ARRIVE


def priest_tts(voice_label: str, speed: float) -> KokoroTTS:
    return KokoroTTS(voice=VOICES[voice_label], speed=speed, use_gpu=True)


def speak_priest_message(
    state: ConversationState,
    text: str,
    voice_label: str,
    speed: float,
    enable_tts: bool,
    mode: str,
):
    state.awaiting_response = True
    history = list(state.history)
    history.append({"role": "assistant", "content": text})
    state.history = history

    yield out(state, history, "", gr.skip(), mode, 0.0, circle_out(state, circle_text_html(text, "assistant")), "priest: speaking")

    last_playback = None
    if enable_tts:
        tts = priest_tts(voice_label, speed)
        for playback, caption in captioned_priest_voice(tts, text, words_per_step=2):
            if state.interrupted:
                break
            last_playback = playback
            yield out(state, history, "", gr.skip(), MODE_SPEAK, 0.0, circle_out(state, circle_text_html(caption, "assistant")))

    state.awaiting_response = False
    state.idle_chunks = 0
    state.reset_utterance()
    yield out(
        state,
        history,
        "",
        priest_audio_html(last_playback, caption=text),
        MODE_SPEAK,
        0.0,
        circle_out(state, circle_text_html(text, "assistant")),
        "priest: playing audio",
    )


def deliver_introduction(
    state: ConversationState,
    voice_label: str,
    speed: float,
    enable_tts: bool,
):
    if state.introduced:
        debug_log(state, "intro: skip replay (session already introduced)", "INTRO")
        if not state.mic_enabled:
            yield activate_listening(state)
        else:
            yield out(
                state,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                MODE_LISTEN,
                0.0,
                circle_keep(state),
                "intro: skip — already listening",
            )
        return

    if state.introducing:
        debug_log(state, "intro: duplicate load blocked", "WARN")
        yield out(
            state,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            MODE_SPEAK,
            0.0,
            circle_keep(state),
            "intro: already running",
        )
        return

    state.introducing = True
    state.awaiting_response = True
    state.mic_enabled = False
    state.reset_utterance()
    state.draft_user = False
    debug_log(state, "intro: started (mic disabled)", "INTRO")

    intro = CHURCH_INTRODUCTION.strip()
    history = [{"role": "assistant", "content": intro}]
    state.history = history

    yield out(
        state,
        history,
        "",
        gr.skip(),
        MODE_ARRIVE,
        0.0,
        circle_out(state, circle_text_html("Peace be with you.", "assistant")),
        "intro: waiting for priest voice",
    )

    last_playback = None
    if enable_tts:
        tts = priest_tts(voice_label, speed)
        debug_log(state, "intro: TTS generating audio…", "INTRO")
        for playback, caption in captioned_priest_voice(tts, intro, words_per_step=2):
            last_playback = playback
            yield out(
                state,
                history,
                "",
                gr.skip(),
                MODE_SPEAK,
                0.0,
                circle_out(state, circle_text_html(caption, "assistant")),
            )
        if last_playback is not None:
            sr, audio = last_playback
            debug_log(
                state,
                f"intro: sending speaker audio {len(audio) / sr:.1f}s @ {sr}Hz",
                "INTRO",
            )
            yield out(
                state,
                history,
                "",
                priest_audio_html(last_playback, caption=intro, intro=True),
                MODE_SPEAK,
                0.0,
                circle_out(state, circle_text_html(intro, "assistant")),
                "intro: priest audio ready (listening arms when audio ends)",
                "INTRO",
            )
    else:
        debug_log(state, "intro: no TTS audio (enable_tts off or TTS failed)", "WARN")
        yield out(
            state,
            history,
            "",
            gr.skip(),
            MODE_SPEAK,
            0.0,
            circle_out(state, circle_text_html(intro, "assistant")),
            "intro: text only",
            "INTRO",
        )

    state.introduced = True
    state.introducing = False
    state.awaiting_response = False
    state.mic_enabled = False
    state.idle_chunks = 0
    state.interrupted = False
    history = [
        {
            "role": "assistant",
            "content": "Peace be with you. Speak, and I will answer as your priest.",
        }
    ]
    state.history = history

    yield out(
        state,
        history,
        "",
        gr.skip(),
        MODE_SPEAK,
        0.0,
        circle_keep(state),
        "intro: complete — listening starts after audio ends",
    )


def _arm_listening_state(state: ConversationState) -> None:
    state.mic_enabled = True
    state.mic_seen = False
    state.idle_chunks = 0
    state.interrupted = False
    state.reset_utterance()


def activate_listening(state: ConversationState):
    _arm_listening_state(state)
    return out(
        state,
        gr.skip(),
        gr.skip(),
        gr.skip(),
        MODE_LISTEN,
        0.0,
        circle_out(state, circle_awaiting_html()),
        "listen: ACTIVATED — browser mic captures speech (speak now)",
        "OK",
    )


def activate_listening_light(state: ConversationState):
    """Update listen state without touching priest audio (avoids intro replay)."""
    _arm_listening_state(state)
    return (
        MODE_LISTEN,
        state,
        0.0,
        circle_out(state, circle_awaiting_html()),
        debug_log(
            state,
            "listen: ACTIVATED — browser mic captures speech (speak now)",
            "OK",
        ),
    )


def priest_idle_prompt(
    state: ConversationState,
    voice_label: str,
    speed: float,
    enable_tts: bool,
):
    yield from speak_priest_message(
        state,
        SILENCE_PROMPT,
        voice_label,
        speed,
        enable_tts,
        MODE_SPEAK,
    )


def capture_pending_audio(state: ConversationState) -> None:
    captured = state.combined_audio()
    if captured is None:
        state.pending_audio = None
        state.pending_turn = False
        debug_log(state, "capture: no audio in speech_chunks", "WARN")
        return
    sample_rate, audio = captured
    duration = audio.size / max(sample_rate, 1)
    if audio.size < int(sample_rate * 0.15):
        state.pending_audio = None
        state.pending_turn = False
        debug_log(
            state,
            f"capture: audio too short ({duration:.2f}s, need >=0.15s)",
            "WARN",
        )
        return
    state.pending_audio = (int(sample_rate), audio.astype(np.float32).tolist())
    state.pending_turn = True
    debug_log(
        state,
        f"capture: OK {duration:.2f}s @ {sample_rate}Hz, {len(state.speech_chunks)} chunks",
        "OK",
    )


def resolve_turn_audio(state: ConversationState) -> tuple[int, np.ndarray] | None:
    if state.pending_audio is not None:
        sample_rate, data = state.pending_audio
        state.pending_audio = None
        audio = np.asarray(data, dtype=np.float32)
        if audio.size == 0:
            return None
        return int(sample_rate), audio
    return state.combined_audio()


def run_assistant_turn(
    state: ConversationState,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    enable_tts: bool,
    voice_label: str,
    speed: float,
):
    if not state.introduced:
        yield out(state, state.history, "", gr.skip(), MODE_ARRIVE, 0.0, circle_keep(state), "assistant: blocked (not introduced)", "WARN")
        return

    state.pending_turn = False
    audio_input = resolve_turn_audio(state)
    state.reset_utterance()
    state.awaiting_response = True
    state.idle_chunks = 0

    if audio_input is None:
        state.awaiting_response = False
        yield out(state, state.history, "", gr.skip(), MODE_LISTEN, 0.0, circle_keep(state), "assistant: no audio input", "WARN")
        return

    sr, audio = audio_input
    debug_log(state, f"STT: transcribing {audio.size / sr:.2f}s @ {sr}Hz", "STT")
    try:
        text = stt.transcribe(audio_input)
    except Exception as exc:
        debug_log(state, f"STT: FAILED — {exc}", "ERR")
        state.awaiting_response = False
        state.draft_user = False
        history = list(state.history)
        if history and history[-1].get("role") == "user" and history[-1].get("content") == "…":
            history.pop()
        history.append({"role": "assistant", "content": f"I could not hear you clearly. ({exc})"})
        state.history = history
        yield out(state, history, "", gr.skip(), MODE_LISTEN, 0.0, circle_out(state, circle_text_html(history[-1]["content"], "assistant")), "STT error shown to user", "ERR")
        return

    if not text:
        debug_log(state, "STT: empty transcription", "WARN")
        state.awaiting_response = False
        state.draft_user = False
        history = list(state.history)
        if history and history[-1].get("role") == "user" and history[-1].get("content") == "…":
            history.pop()
        history.append(
            {
                "role": "assistant",
                "content": "Forgive me, I could not hear your words. Please speak again.",
            }
        )
        state.history = history
        yield out(state, history, "", gr.skip(), MODE_LISTEN, 0.0, circle_out(state, circle_text_html(history[-1]["content"], "assistant")), "STT: empty — ask user to repeat", "WARN")
        return

    debug_log(state, f'STT: heard "{text}"', "OK")
    history = list(state.history)
    if history and history[-1].get("role") == "user" and history[-1].get("content") == "…":
        history[-1] = {"role": "user", "content": text}
    else:
        history.append({"role": "user", "content": text})
    state.draft_user = False
    history.append({"role": "assistant", "content": "…"})
    state.history = history

    yield out(state, history, "", gr.skip(), MODE_HEAR, 0.0, circle_out(state, circle_text_html(text, "user")), f'user said: "{text}"', "OK")
    yield out(state, history, "", gr.skip(), MODE_THINK, 0.0, circle_out(state, circle_text_html("…", "assistant")), "LLM: generating reply…", "LLM")

    prior = history[:-2]
    tts = priest_tts(voice_label, speed) if enable_tts else None

    consumed_answer_len = 0
    phrase_buffer = ""
    thought = ""
    spoken_chunks: list[np.ndarray] = []

    try:
        for update in engine.generate_stream(
            text,
            history=prior,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ):
            if state.interrupted:
                break

            thought = update["thought"]
            answer = update["answer"]
            history[-1]["content"] = format_chat_answer(answer)
            state.history = history

            mode = MODE_SPEAK if answer else MODE_THINK
            display = stream_display_snippet(answer)
            yield out(state, history, thought, gr.skip(), mode, 0.0, circle_out(state, circle_text_html(display, "assistant")))

            if not enable_tts or not tts or not answer:
                continue

            if len(answer) < consumed_answer_len:
                consumed_answer_len = 0
                phrase_buffer = ""

            phrase_buffer += answer[consumed_answer_len:]
            consumed_answer_len = len(answer)

            phrases, phrase_buffer = pop_speakable_phrases(phrase_buffer)
            spoken_words = 0
            answer_words = format_chat_answer(answer).split()
            for phrase in phrases:
                if state.interrupted:
                    break
                for _, chunk in tts.stream(phrase):
                    if state.interrupted:
                        break
                    spoken_chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
                    spoken_words = min(len(answer_words), spoken_words + 2)
                    caption = " ".join(answer_words[:spoken_words]) or stream_display_snippet(answer)
                    yield out(state, history, thought, gr.skip(), MODE_SPEAK, 0.0, circle_out(state, circle_text_html(caption, "assistant")))

        if not state.interrupted and enable_tts and tts and phrase_buffer.strip():
            answer_words = format_chat_answer(history[-1]["content"]).split()
            spoken_words = max(0, len(answer_words) - len(phrase_buffer.split()))
            for _, chunk in tts.stream(phrase_buffer.strip()):
                if state.interrupted:
                    break
                spoken_chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
                spoken_words = min(len(answer_words), spoken_words + 2)
                caption = " ".join(answer_words[:spoken_words]) or stream_display_snippet(history[-1]["content"])
                yield out(state, history, thought, gr.skip(), MODE_SPEAK, 0.0, circle_out(state, circle_text_html(caption, "assistant")))
    except Exception as exc:
        debug_log(state, f"LLM/TTS: FAILED — {exc}", "ERR")
        history[-1]["content"] = f"I must pause for a moment. ({exc})"
        state.history = history
        final_caption = history[-1]["content"]
        final_audio = (
            priest_audio_html(priest_playback(spoken_chunks), caption=final_caption)
            if spoken_chunks
            else gr.skip()
        )
        yield out(
            state,
            history,
            thought,
            final_audio,
            MODE_SPEAK,
            0.0,
            circle_out(state, circle_text_html(final_caption, "assistant")),
            "reply failed",
            "ERR",
        )
    else:
        reply = history[-1]["content"]
        debug_log(state, f'LLM: reply ready ({len(reply)} chars), TTS chunks={len(spoken_chunks)}', "OK")
        final_audio = (
            priest_audio_html(priest_playback(spoken_chunks), caption=reply)
            if spoken_chunks
            else gr.skip()
        )
        mode = MODE_SPEAK if spoken_chunks else MODE_LISTEN
        yield out(
            state,
            history,
            thought,
            final_audio,
            mode,
            0.0,
            circle_out(state, circle_text_html(reply, "assistant")),
            "reply: playing priest audio",
            "OK",
        )
    finally:
        state.awaiting_response = False
        state.idle_chunks = 0
        state.draft_user = False
        state.reset_utterance()


def on_audio_stream(
    audio,
    state: ConversationState,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    enable_tts: bool,
    voice_label: str,
    speed: float,
):
    mic_level = 0.0
    energy = 0.0
    chunk = None
    if audio is not None:
        _, data = audio
        chunk = np.asarray(data, dtype=np.float32)
        energy = chunk_energy(chunk)
        mic_level = min(1.0, energy * 28)

    sample_count = int(chunk.size) if chunk is not None else 0

    if state.mic_enabled and not state.mic_seen:
        state.mic_seen = True
        debug_log(
            state,
            f"first mic packet: samples={sample_count} energy={energy:.5f}",
            "OK",
        )

    if state.mic_enabled and audio is None:
        debug_log(state, f"mic stream fired but audio=None", "WARN")

    if not state.introduced or state.introducing:
        tick = debug_mic_tick(state, energy, mic_level, sample_count, "ignored: intro phase")
        yield out(state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), 0.0, gr.skip(), tick)
        return

    if not state.mic_enabled:
        tick = debug_mic_tick(state, energy, mic_level, sample_count, "ignored: mic_enabled=False")
        yield out(state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), 0.0, gr.skip(), tick)
        return

    if (
        state.awaiting_response
        and not state.introducing
        and energy >= INTERRUPT_THRESHOLD
    ):
        state.interrupted = True
        state.awaiting_response = False
        state.pending_turn = False
        state.pending_audio = None
        state.introducing = False
        state.reset_utterance()
        state.speaking = True
        state.draft_user = True
        if chunk is not None and chunk.size > 0:
            store_speech_chunk(state, chunk)
        state.history = history_with_draft_user(state.history)
        yield out(
            state,
            state.history,
            gr.skip(),
            gr.skip(),
            MODE_HEAR,
            float(mic_level),
            circle_out(state, circle_text_html("…", "user")),
            f"interrupt: user spoke over priest (E={energy:.4f})",
            "WARN",
        )
        return

    if state.awaiting_response:
        tick = debug_mic_tick(state, energy, mic_level, sample_count, "ignored: awaiting priest reply")
        yield out(state, gr.skip(), gr.skip(), gr.skip(), MODE_SPEAK, float(mic_level), circle_keep(state), tick)
        return

    state, turn_ended = ingest_stream_chunk(audio, state)

    if state.speaking and not state.draft_user:
        state.draft_user = True
        state.history = history_with_draft_user(state.history)
    elif not state.speaking and state.draft_user and not turn_ended:
        state.draft_user = False
        history = list(state.history)
        if history and history[-1].get("role") == "user" and history[-1].get("content") == "…":
            history.pop()
        state.history = history

    if state.interrupted and state.speaking:
        state.interrupted = False

    if turn_ended:
        debug_log(state, f"turn: speech ended (E={energy:.4f})", "TURN")
        capture_pending_audio(state)
        state.reset_utterance()
        if not state.pending_turn:
            state.awaiting_response = False
            yield out(
                state,
                state.history,
                gr.skip(),
                gr.skip(),
                MODE_LISTEN,
                float(mic_level),
                circle_out(state, circle_text_html("I did not catch that. Please speak again.", "assistant")),
                "turn: no valid audio captured",
                "WARN",
            )
            return
        state.awaiting_response = True
        yield out(
            state,
            state.history,
            gr.skip(),
            gr.skip(),
            MODE_HEAR,
            float(mic_level),
            circle_out(state, circle_text_html("…", "user")),
            "turn: running STT → LLM → TTS",
            "TURN",
        )
        yield from run_assistant_turn(
            state,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            enable_tts,
            voice_label,
            speed,
        )
        return

    if (
        state.introduced
        and not state.awaiting_response
        and not state.speaking
        and state.idle_chunks >= IDLE_CHUNK_LIMIT
    ):
        state.idle_chunks = 0
        yield from priest_idle_prompt(state, voice_label, speed, enable_tts)
        return

    mode = listening_mode(state)
    if state.speaking or state.draft_user:
        tick = debug_mic_tick(state, energy, mic_level, sample_count, "user speaking…")
        yield out(
            state,
            state.history,
            gr.skip(),
            gr.skip(),
            mode,
            float(mic_level),
            circle_out(state, circle_text_html("…", "user")),
            tick,
        )
    elif mode == MODE_LISTEN:
        tick = debug_mic_tick(state, energy, mic_level, sample_count, "listening idle")
        yield out(
            state,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            mode,
            float(mic_level),
            circle_out(state, circle_awaiting_html()),
            tick,
        )
    else:
        tick = debug_mic_tick(state, energy, mic_level, sample_count)
        yield out(state, gr.skip(), gr.skip(), gr.skip(), mode, float(mic_level), circle_keep(state), tick)


def on_browser_audio(
    _tick: float,
    b64: str,
    state: ConversationState,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    enable_tts: bool,
    voice_label: str,
    speed: float,
):
    b64 = (b64 or "").strip()
    if not b64:
        yield bout(state, "", gr.skip(), gr.skip(), gr.skip(), gr.skip(), 0.0, gr.skip())
        return

    if not state.introduced or state.introducing or not state.mic_enabled:
        yield bout(
            state,
            "",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            0.0,
            gr.skip(),
            "browser: ignored (not listening)",
            "WARN",
        )
        return

    if state.awaiting_response:
        yield bout(
            state,
            "",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            0.0,
            gr.skip(),
            "browser: ignored (priest turn in progress)",
            "WARN",
        )
        return

    decoded = decode_b64_wav(b64)
    if decoded is None:
        yield bout(
            state,
            "",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            0.0,
            gr.skip(),
            "browser: WAV decode failed",
            "ERR",
        )
        return

    sample_rate, audio = decoded
    duration = audio.size / max(sample_rate, 1)
    state.debug_tick += 1
    debug_log(state, f"browser: received {duration:.2f}s @ {sample_rate}Hz", "OK")

    if audio.size < int(sample_rate * 0.15):
        yield bout(
            state,
            "",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            MODE_LISTEN,
            0.0,
            circle_keep(state),
            f"browser: audio too short ({duration:.2f}s)",
            "WARN",
        )
        return

    state.pending_audio = (sample_rate, audio.astype(np.float32).tolist())
    state.pending_turn = True
    state.reset_utterance()
    state.awaiting_response = True
    state.history = history_with_draft_user(state.history)

    yield bout(
        state,
        "",
        state.history,
        gr.skip(),
        gr.skip(),
        MODE_HEAR,
        0.0,
        circle_out(state, circle_text_html("…", "user")),
        "browser: STT → Supra → circle text",
        "TURN",
    )

    for item in run_assistant_turn(
        state,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        enable_tts,
        voice_label,
        speed,
    ):
        yield ("",) + item


custom_css = """
footer { display: none !important; }
.gradio-container, .main, .contain, .app, .panel-wrap {
    max-width: 100% !important;
    padding: 0 !important;
    background: transparent !important;
}
#voice-wrap .html-container, #circle-text-src .html-container,
#voice-viz .html-container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
#mic-level, #status-box, #chat-panel, #guide-links, #mic-b64, #mic-tick, #listen-tick { display: none !important; }
#priest-audio {
    position: fixed !important;
    left: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    opacity: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    pointer-events: none !important;
}
#center-stage {
    min-height: 100vh !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#voice-wrap, #voice-wrap > .form, #voice-wrap .block,
#voice-viz, #voice-viz > .form, #voice-viz .block,
#circle-text-src, #circle-text-src > .form, #circle-text-src .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
#voice-wrap {
    position: relative;
    width: min(92vw, 680px);
    height: min(92vw, 680px);
    margin: 0 auto !important;
}
#voice-viz { display: flex !important; justify-content: center !important; margin: 0 !important; width: 100% !important; height: 100% !important; }
.voice-hub {
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}
#circle-text-src {
    position: fixed !important;
    left: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    opacity: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
}
.voice-hub-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 34%;
    max-height: 28%;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    z-index: 5;
    pointer-events: none;
    overflow: hidden;
}
.voice-hub-text .circle-lines {
    width: 100%;
    max-height: 100%;
    overflow-y: auto;
    scrollbar-width: none;
    text-align: center !important;
}
.voice-hub-text .circle-lines::-webkit-scrollbar { display: none; }
.circle-lines { text-align: center !important; width: 100%; }
.circle-line {
    margin: 0 auto;
    max-width: 100%;
    font-size: clamp(0.45rem, 1.05vw, 0.56rem);
    line-height: 1.35;
    color: #f3e8ff;
    text-align: center !important;
}
.circle-role {
    display: block;
    margin-bottom: 0.2rem;
    font-size: 0.48rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #c4b5fd;
}
.circle-line--user .circle-role { color: #93c5fd; }
.circle-line--priest .circle-role { color: #fbbf24; }
.circle-line--await .circle-copy {
    color: #a78bfa;
    font-size: clamp(0.44rem, 1vw, 0.54rem);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.85;
}
.circle-copy {
    display: block;
    text-align: center;
    word-wrap: break-word;
    overflow-wrap: anywhere;
    hyphens: auto;
}
#viz-canvas {
    display: block;
    width: 100% !important;
    height: 100% !important;
}
.voice-hub-core {
    position: absolute;
    bottom: 10%;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.35rem;
    pointer-events: none;
    z-index: 4;
}
.voice-hub-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #a78bfa;
    box-shadow: 0 0 16px rgba(167, 139, 250, 0.8);
}
.voice-hub-dot[data-mode="speak"] {
    background: #fbbf24;
    box-shadow: 0 0 16px rgba(251, 191, 36, 0.8);
}
.voice-hub-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #c4b5fd;
}
#settings-open {
    position: fixed;
    top: 18px;
    right: 18px;
    z-index: 200;
    width: 42px;
    height: 42px;
    border-radius: 999px;
    border: 1px solid rgba(167, 139, 250, 0.35);
    background: rgba(15, 10, 30, 0.55);
    color: #ddd6fe;
    font-size: 1.35rem;
    line-height: 1;
    cursor: pointer;
    backdrop-filter: blur(8px);
}
#settings-open:hover { border-color: rgba(167, 139, 250, 0.8); color: #fff; }
#settings-drawer {
    position: fixed !important;
    top: 0 !important;
    right: 0 !important;
    width: min(360px, 92vw) !important;
    height: 100vh !important;
    transform: translateX(100%);
    transition: transform 0.28s ease;
    z-index: 150 !important;
    background: rgba(12, 8, 24, 0.96) !important;
    border-left: 1px solid rgba(167, 139, 250, 0.25) !important;
    padding: 1.25rem !important;
    overflow-y: auto !important;
    box-shadow: -12px 0 40px rgba(0, 0, 0, 0.35) !important;
}
#settings-drawer.is-open { transform: translateX(0); }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="violet"),
    title="ASI Foundation Church",
    css=custom_css,
    js=BOOT_JS,
    fill_height=True,
) as demo:
    state = gr.State(ConversationState())

    gr.HTML(SETTINGS_BTN_HTML)

    with gr.Column(elem_id="center-stage"):
        with gr.Column(elem_id="voice-wrap"):
            gr.HTML(VISUALIZER_HTML, elem_id="voice-viz", container=False)
            circle_text = gr.HTML(
                circle_awaiting_html(),
                elem_id="circle-text-src",
                visible="hidden",
            )

    chatbot = gr.Chatbot(
        label="Sacred conversation",
        height=360,
        type="messages",
        show_label=False,
        elem_id="chat-panel",
        visible="hidden",
    )

    status = gr.Textbox(
        value=MODE_ARRIVE,
        interactive=False,
        elem_id="status-box",
        visible="hidden",
    )

    priest_player = gr.HTML("", elem_id="priest-audio", container=False)

    mic_b64 = gr.Textbox(
        value="",
        elem_id="mic-b64",
        lines=1,
        max_lines=1,
        show_label=False,
        visible="hidden",
    )
    mic_tick = gr.Number(value=0, elem_id="mic-tick", show_label=False, visible="hidden")
    listen_tick = gr.Number(value=0, elem_id="listen-tick", show_label=False, visible="hidden")

    mic_level = gr.Number(value=0.0, elem_id="mic-level", show_label=False, visible="hidden")

    with gr.Column(elem_id="settings-drawer"):
        debug_panel = gr.Textbox(
            label="Debug log (intro · listen · STT · reply)",
            lines=14,
            max_lines=20,
            interactive=False,
            elem_id="debug-panel",
        )
        thinking = gr.Textbox(
            label="Priest reflection",
            lines=8,
            interactive=False,
        )
        enable_tts = gr.Checkbox(value=True, label="Priest speaks aloud")
        voice = gr.Dropdown(
            choices=list(VOICES.keys()),
            value=PRIEST_VOICE,
            label="Priest voice",
        )
        speed = gr.Slider(0.75, 1.15, value=0.92, step=0.05, label="Speech pace")
        max_tokens = gr.Slider(128, 512, value=256, step=32, label="Max reply length")
        temperature = gr.Slider(0.0, 1.2, value=0.7, step=0.05, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top-p")
        top_k = gr.Slider(1, 100, value=25, step=1, label="Top-k")
        gr.Markdown(
            f"**Guide:** [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) · "
            f"**Voice:** [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)",
            elem_id="guide-links",
        )

    settings = [max_tokens, temperature, top_p, top_k, enable_tts, voice, speed]
    turn_outputs = [chatbot, thinking, priest_player, status, state, mic_level, circle_text, debug_panel]
    listen_outputs = [status, state, mic_level, circle_text, debug_panel]
    browser_outputs = [mic_b64, *turn_outputs]

    demo.load(
        deliver_introduction,
        inputs=[state, voice, speed, enable_tts],
        outputs=turn_outputs,
        queue=True,
        show_progress="hidden",
        concurrency_limit=1,
    ).then(js=AFTER_INTRO_JS)

    listen_tick.change(
        activate_listening_light,
        inputs=[state],
        outputs=listen_outputs,
        queue=True,
        concurrency_limit=1,
    )

    mic_tick.change(
        on_browser_audio,
        inputs=[mic_tick, mic_b64, state, *settings],
        outputs=browser_outputs,
        queue=True,
        concurrency_limit=1,
    )

    def debug_heartbeat(state: ConversationState):
        if not state.mic_enabled:
            return debug_view(state)
        return debug_log(
            state,
            f"heartbeat: mic_enabled=True browser_utterances={state.debug_tick} (speak now)",
            "INFO",
        )

    gr.Timer(4).tick(
        debug_heartbeat,
        inputs=[state],
        outputs=[debug_panel],
        queue=True,
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(pwa=True)
