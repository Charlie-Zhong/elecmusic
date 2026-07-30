import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import re

# --- 修复 matplotlib 中文显示问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 核心物理与合成宏定义 (随心修改这里来改变音色质感)
# ==========================================
SAMPLING_RATE = 44100
A4_REF_FREQ = 440.0
SNAP_TOLERANCE = 30  
MAX_FREQ_DISPLAY = 5000 
BLOCK_SIZE = 1024

# 1. 物理刚性系数 (非谐波性)
STIFFNESS_B = 0 

# 2. 揉弦 (Vibrato) 参数
PORTAMENTO_TIME = 0.1  # 滑音时间常数(秒)
VIBRATO_RATE = 5.5      # 揉弦速度 (Hz)
VIBRATO_DEPTH = 0.0    # 揉弦最大幅度 (音分，20音分约等于0.2个半音)
VIBRATO_DELAY = 0.4     # 延迟揉弦时间(秒)
VIBRATO_FADE = 0.5      # 揉弦渐入时间(秒)

# 3. ADSR 包络参数 (单位：秒 或 比例)
ATTACK_TIME = 0.1    # 起音时间 
DECAY_TIME = 0.01    # 衰减时间
SUSTAIN_LEVEL = 0.9  # 持续音量 (0.0 到 1.0)
RELEASE_TIME = 0.3   # 释音时间 
REVERB_MIX = 0.45     
REVERB_DECAY = 0.8   
DELAY_TIMES = [1103, 1601, 2111]
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def get_inharmonic_ratio(n):
    return n * np.sqrt(1 + STIFFNESS_B * (n**2))

# --- 全局状态 ---
current_midi_note = 69 
target_base_freq = 440.0
current_base_freq = target_base_freq 
partial_ratios = {get_inharmonic_ratio(1): 1.0}  
partial_phases = {get_inharmonic_ratio(1): np.random.uniform(0, 2*np.pi)} # 随机初始相位
global_f0_phase = 0.0 
active_time = 0.0     
selected_ratio = None  
env_state = 0  
env_level = 0.0  

reverb_buffers = [np.zeros(d) for d in DELAY_TIMES]
reverb_ptrs = [0, 0, 0]

# --- 音频引擎 (带向量化混响) ---
def audio_callback(outdata, frames, time, status):
    global current_base_freq, global_f0_phase, active_time, env_state, env_level
    
    env = np.zeros(frames)
    
    if env_state != 0 or env_level > 0.0:
        attack_step = 1.0 / (ATTACK_TIME * SAMPLING_RATE)
        decay_step = (1.0 - SUSTAIN_LEVEL) / (DECAY_TIME * SAMPLING_RATE)
        release_step = SUSTAIN_LEVEL / (RELEASE_TIME * SAMPLING_RATE) if RELEASE_TIME > 0 else 1.0

        idx = 0  
        while idx < frames:
            if env_state == 0:
                break 
            elif env_state == 3:
                env[idx:] = SUSTAIN_LEVEL
                break
            elif env_state == 1: 
                samples_needed = max(0, int(np.ceil((1.0 - env_level) / attack_step)))
                n = min(samples_needed, frames - idx) 
                if n > 0:
                    env[idx : idx + n] = env_level + attack_step * np.arange(1, n + 1)
                    env_level = env[idx + n - 1]
                    idx += n
                if env_level >= 1.0 or n == samples_needed:
                    env_level, env_state = 1.0, 2
            elif env_state == 2: 
                samples_needed = max(0, int(np.ceil((env_level - SUSTAIN_LEVEL) / decay_step)))
                n = min(samples_needed, frames - idx)
                if n > 0:
                    env[idx : idx + n] = env_level - decay_step * np.arange(1, n + 1)
                    env_level = env[idx + n - 1]
                    idx += n
                if env_level <= SUSTAIN_LEVEL or n == samples_needed:
                    env_level, env_state = SUSTAIN_LEVEL, 3
            elif env_state == 4: 
                samples_needed = max(0, int(np.ceil(env_level / release_step)))
                n = min(samples_needed, frames - idx)
                if n > 0:
                    env[idx : idx + n] = env_level - release_step * np.arange(1, n + 1)
                    env_level = env[idx + n - 1]
                    idx += n
                if env_level <= 0.0 or n == samples_needed:
                    env_level, env_state = 0.0, 0

    wave = np.zeros(frames)
    
    if np.any(env > 0):
        beta = np.exp(-1.0 / (PORTAMENTO_TIME * SAMPLING_RATE))
        n_array = np.arange(1, frames + 1)
        beta_n = beta ** n_array
        f0_array = beta_n * current_base_freq + (1.0 - beta_n) * target_base_freq
        current_base_freq = f0_array[-1] 

        t_block = np.arange(frames) / SAMPLING_RATE
        current_times = active_time + t_block
        if env_state in [1, 2, 3, 4]:  
            active_time += frames / SAMPLING_RATE
            
        vib_depth_envelope = np.clip((current_times - VIBRATO_DELAY) / VIBRATO_FADE, 0.0, 1.0) * VIBRATO_DEPTH
        lfo = np.sin(2 * np.pi * VIBRATO_RATE * current_times)
        
        vib_multiplier = 2.0 ** ((vib_depth_envelope * lfo) / 1200.0)
        f0_array_vibrato = f0_array * vib_multiplier

        phase_diff = 2 * np.pi * f0_array_vibrato / SAMPLING_RATE
        f0_phases = global_f0_phase + np.cumsum(phase_diff)
        global_f0_phase = f0_phases[-1] % (2 * np.pi)

        ratios_snapshot = list(partial_ratios.items())
        for ratio, amp in ratios_snapshot:
            if amp > 0.01:
                # 叠加密码：加上该泛音专属的随机初始相位
                phase_offset = partial_phases.get(ratio, 0.0)
                wave += amp * np.sin(ratio * f0_phases + phase_offset)
                
        total_estimated = sum(a for r, a in ratios_snapshot)
        if total_estimated > 1.0: wave /= total_estimated
        wave *= env * 0.5  

    wet_signal = np.zeros(frames)
    for i, dlen in enumerate(DELAY_TIMES):
        ptr = reverb_ptrs[i]
        buf = reverb_buffers[i]
        delayed = np.zeros(frames)
        
        if ptr + frames <= dlen:
            delayed = buf[ptr : ptr + frames]
            buf[ptr : ptr + frames] = wave + delayed * REVERB_DECAY
            reverb_ptrs[i] = (ptr + frames) % dlen
        else:
            p1 = dlen - ptr
            p2 = frames - p1
            delayed[:p1] = buf[ptr:]
            delayed[p1:] = buf[:p2]
            
            buf[ptr:] = wave[:p1] + delayed[:p1] * REVERB_DECAY
            buf[:p2] = wave[p1:] + delayed[p1:] * REVERB_DECAY
            reverb_ptrs[i] = p2
            
        wet_signal += delayed

    wet_signal /= len(DELAY_TIMES)
    final_output = wave * (1.0 - REVERB_MIX) + wet_signal * REVERB_MIX
    outdata[:] = final_output.reshape(-1, 1).astype(np.float32)

stream = sd.OutputStream(samplerate=SAMPLING_RATE, blocksize=BLOCK_SIZE, channels=1, callback=audio_callback)
stream.start()

# --- UI 界面与可视化部分 ---
fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(left=0.08, bottom=0.2, right=0.95, hspace=0.35)
fig.canvas.manager.set_window_title("频谱与波形可视化")

line_wave, = ax_wave.plot([], [], lw=2, color='dodgerblue')
ax_wave.set_title("时域波形")
ax_wave.set_xlim(0, 0.02) 
ax_wave.set_ylim(-1.0, 1.0)
ax_wave.grid(True)

ax_spec.set_title("频域画布")
ax_spec.set_xlim(0, MAX_FREQ_DISPLAY)
ax_spec.set_ylim(0, 1.1)
ax_spec.set_xlabel("频率 (Hz)")
ax_spec.set_ylabel("振幅大小")
ax_spec.grid(True, alpha=0.3)

annot = ax_spec.annotate(
    "", xy=(0,0), xytext=(10, 15),
    textcoords="offset points",
    bbox=dict(boxstyle="round4,pad=0.5", fc="ivory", ec="gray", lw=1, alpha=0.9),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3")
)
annot.set_visible(False)

def update_visuals():
    t = np.linspace(0, 0.02, int(SAMPLING_RATE * 0.02), False)
    wave = np.zeros_like(t)
    for ratio, amp in partial_ratios.items():
        if amp > 0: wave += amp * np.sin(2 * np.pi * (ratio * target_base_freq) * t)
        
    total_est = sum(partial_ratios.values())
    if total_est > 1.0: wave /= total_est
    line_wave.set_data(t, wave)

    for line in reversed(ax_spec.lines): line.remove()
    
    max_harmonic = 1
    while True:
        fn = get_inharmonic_ratio(max_harmonic) * target_base_freq
        if fn > MAX_FREQ_DISPLAY: break
        ax_spec.axvline(x=fn, color='gray', linestyle='--', alpha=0.3, zorder=1)
        max_harmonic += 1

    ratios = list(partial_ratios.keys())
    amps = list(partial_ratios.values())
    if ratios:
        for ratio, a in zip(ratios, amps):
            f = ratio * target_base_freq
            is_harmonic = False
            for n in range(1, max_harmonic + 1):
                if abs(ratio - get_inharmonic_ratio(n)) < 0.05:
                    is_harmonic = True
                    break
            color = 'crimson' if is_harmonic else 'orange'
            ax_spec.plot([f, f], [0, a], color=color, lw=2, zorder=2)
            ax_spec.plot(f, a, 'o', color=color, markersize=6, zorder=3)

    fig.canvas.draw_idle()

update_visuals()

# ---------------------------------------------------------
# 🎛️ 核心改动：滚轮与按键调音逻辑
# ---------------------------------------------------------
def change_base_note(delta):
    global current_midi_note, target_base_freq
    current_midi_note = max(36, min(96, current_midi_note + delta))
    target_base_freq = 440.0 * (2.0 ** ((current_midi_note - 69) / 12.0))
    note_name = NOTE_NAMES[current_midi_note % 12]
    octave = (current_midi_note // 12) - 1
    txt_note_display.set_text(f"当前基音: {note_name}{octave} ({target_base_freq:.1f}Hz)")
    update_visuals()

def on_scroll(event):
    if event.step > 0:
        change_base_note(1)  
    elif event.step < 0:
        change_base_note(-1) 

def on_key_press(event):
    if event.key == 'up':
        change_base_note(1)
    elif event.key == 'down':
        change_base_note(-1)

fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('key_press_event', on_key_press)

# ---------------------------------------------------------
# 鼠标绘图逻辑 (已修改为音分对数吸附)
# ---------------------------------------------------------
def get_snapped_ratio(x):
    if x <= 0: return 1.0
    target_ratio = x / target_base_freq
    min_diff_cents = float('inf')
    best_rn = target_ratio
    for n in range(1, int(MAX_FREQ_DISPLAY / target_base_freq) + 5):
        rn = get_inharmonic_ratio(n)
        # 用对数公式计算音分距离: 1200 * |log2(f1/f2)|
        diff_cents = 1200.0 * abs(np.log2(target_ratio / rn))
        if diff_cents < min_diff_cents:
            min_diff_cents = diff_cents
            best_rn = rn
    # 这里的 SNAP_TOLERANCE (30) 现在代表 30 音分
    if min_diff_cents < SNAP_TOLERANCE:
        return best_rn
    return target_ratio

def find_closest_ratio(freq):
    if not partial_ratios or freq <= 0: return None
    target_ratio = freq / target_base_freq
    # 按照音分距离寻找最近的点
    closest_r = min(partial_ratios.keys(), key=lambda k: 1200.0 * abs(np.log2(k / target_ratio)))
    # 判定容差定为 40 音分
    if 1200.0 * abs(np.log2(closest_r / target_ratio)) < 40: 
        return closest_r
    return None

def on_mouse_press(event):
    global selected_ratio
    if event.inaxes != ax_spec or event.xdata is None or event.ydata is None: return
    x, y = event.xdata, max(0.0, min(1.0, event.ydata))

    if event.button == 1:  
        snapped_r = get_snapped_ratio(x)
        closest_r = find_closest_ratio(x)
        selected_ratio = closest_r if closest_r is not None else snapped_r
        partial_ratios[selected_ratio] = y
        update_visuals()
    elif event.button == 3:  
        closest_r = find_closest_ratio(x)
        if closest_r is not None:
            del partial_ratios[closest_r]
            if annot.get_visible(): annot.set_visible(False)
            update_visuals()

def on_mouse_move(event):
    global selected_ratio
    if event.inaxes != ax_spec or event.xdata is None:
        if annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()
        return
        
    x, y = event.xdata, max(0.0, min(1.0, event.ydata)) if event.ydata else 0.0

    if event.button == 1 and selected_ratio is not None:
        partial_ratios[selected_ratio] = y
        update_visuals()
        f = selected_ratio * target_base_freq
        annot.xy = (f, y)
        annot.set_text(f"频率: {f:.1f} Hz\n强度: {y:.2f}")
        if not annot.get_visible(): annot.set_visible(True)
        return

    closest_r = find_closest_ratio(x)
    if closest_r is not None:
        amp = partial_ratios[closest_r]
        f = closest_r * target_base_freq
        annot.xy = (f, amp)
        annot.set_text(f"频率: {f:.1f} Hz\n强度: {amp:.2f}")
        annot.set_visible(True)
        fig.canvas.draw_idle()
    else:
        if annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()

def on_mouse_release(event):
    global selected_ratio
    if event.button == 1: selected_ratio = None 

fig.canvas.mpl_connect('button_press_event', on_mouse_press)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_release_event', on_mouse_release)

# --- UI 控件调整 ---
ax_note_display = plt.axes([0.1, 0.05, 0.25, 0.06])
ax_note_display.axis('off') 
txt_note_display = ax_note_display.text(0.5, 0.5, '当前基音: A4 (440.0Hz)', 
                                        ha='center', va='center', fontsize=12, fontweight='bold',
                                        bbox=dict(facecolor='ivory', edgecolor='gray', boxstyle='round,pad=0.5'))

ax_btn_toggle = plt.axes([0.45, 0.05, 0.15, 0.06])
btn_toggle = Button(ax_btn_toggle, '按下琴键', color='lightgreen')

ax_btn_clear = plt.axes([0.65, 0.05, 0.15, 0.06])
btn_clear = Button(ax_btn_clear, '清空画布', color='lightcoral')

def toggle_note(event):
    global env_state, active_time
    if env_state in [0, 4]: 
        env_state = 1
        active_time = 0.0 
        btn_toggle.label.set_text('松开琴键')
        btn_toggle.color = 'salmon'
    else: 
        env_state = 4 
        btn_toggle.label.set_text('按下琴键')
        btn_toggle.color = 'lightgreen'
    fig.canvas.draw_idle()

def clear_canvas(event):
    partial_ratios.clear()
    if annot.get_visible(): annot.set_visible(False)
    update_visuals()

btn_toggle.on_clicked(toggle_note)
btn_clear.on_clicked(clear_canvas)

try:
    plt.show()
finally:
    stream.stop()
    stream.close()
    print("合成器已安全关闭。")