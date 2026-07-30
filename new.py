import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
import matplotlib.animation as animation
import re

# --- 修复 matplotlib 中文显示问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 物理建模引擎核心参数
# ==========================================
SAMPLING_RATE = 44100
MAX_DELAY = 8192  # 缓冲区大小，决定了能产生的最低频率 (约 5.3Hz)

# 1. 创建两个循环延迟线，模拟琴弦被弓子分成的两段
delay_neck = np.zeros(MAX_DELAY)   # 从弓子到琴枕 (左段)
delay_bridge = np.zeros(MAX_DELAY) # 从弓子到琴马 (右段)
write_ptr = 0

# 2. 全局物理交互变量 (受界面滑块实时控制)
current_f0 = 293.66  # 基频 (D4)
bow_velocity = 0.0   # 运弓速度 (0.0 则不发声)
bow_pressure = 3.0   # 弓子下压的力度
bow_position = 0.15  # 拉弓位置 (占全长的比例，通常靠近琴马)
noise_amount = 0.03  # 弓毛松香的随机摩擦底噪

# 3. 物理状态记忆
filter_state = 0.0   # 琴马低通滤波器状态 (模拟高频能量耗散)
dc_in = 0.0          # 直流阻塞器输入
dc_out = 0.0         # 直流阻塞器输出

# 用于界面可视化的全局波形缓存
latest_wave = np.zeros(2048)

NOTE_MAP = {
    'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3,
    'E': 4, 'F': 5, 'F#': 6, 'GB': 6, 'G': 7, 'G#': 8,
    'AB': 8, 'A': 9, 'A#': 10, 'BB': 10, 'B': 11
}

def note_to_freq(note_str):
    pattern = r"([A-Ga-g][#bB]?)([0-8])"
    match = re.match(pattern, note_str.strip())
    if not match: return 293.66
    name, octave = match.groups()
    semitone = NOTE_MAP[name.upper().replace('B', 'B')]
    midi_note = (int(octave) + 1) * 12 + semitone
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

# --- 音频引擎 (波导方程实时计算) ---
def audio_callback(outdata, frames, time, status):
    global write_ptr, filter_state, dc_in, dc_out, latest_wave

    wave = np.zeros(frames)
    
    # 根据当前音高和拉弓位置，计算两段琴弦的物理长度 (以采样点为单位)
    total_delay = SAMPLING_RATE / current_f0
    len_neck = total_delay * (1.0 - bow_position)
    len_bridge = total_delay * bow_position

    for i in range(frames):
        # ----------------------------------------------------
        # 步骤 1: 带线性插值的分数延迟线读取 (支持极致平滑的滑音)
        # ----------------------------------------------------
        # 读取琴枕方向传回来的波
        r_idx_n = write_ptr - len_neck
        if r_idx_n < 0: r_idx_n += MAX_DELAY
        idx_n = int(r_idx_n)
        frac_n = r_idx_n - idx_n
        y_from_neck = delay_neck[idx_n] * (1 - frac_n) + delay_neck[(idx_n + 1) % MAX_DELAY] * frac_n

        # 读取琴马方向传回来的波
        r_idx_b = write_ptr - len_bridge
        if r_idx_b < 0: r_idx_b += MAX_DELAY
        idx_b = int(r_idx_b)
        frac_b = r_idx_b - idx_b
        y_from_bridge = delay_bridge[idx_b] * (1 - frac_b) + delay_bridge[(idx_b + 1) % MAX_DELAY] * frac_b

        # ----------------------------------------------------
        # 步骤 2: 琴马的物理低通反射 (高频能量被琴马吸收)
        # ----------------------------------------------------
        filter_state = 0.85 * y_from_bridge + 0.15 * filter_state
        y_from_bridge_filt = filter_state

        # ----------------------------------------------------
        # 步骤 3: 核心魔法 —— 亥姆霍兹非线性摩擦表 (Smith 算法)
        # ----------------------------------------------------
        # 计算弓子与琴弦的相对速度差
        v_string = y_from_neck + y_from_bridge_filt
        v_delta = bow_velocity - v_string
        
        # 指数型摩擦力衰减曲线：速度差越小(咬死)，摩擦力越大；速度差越大(滑脱)，摩擦力剧减
        friction = bow_pressure * np.exp(-20.0 * (v_delta ** 2))
        y_bow = v_delta * friction

        # 只要运弓，就注入弓毛摩擦的白噪声
        if bow_velocity > 0.01:
            y_bow += np.random.uniform(-noise_amount, noise_amount) * bow_pressure * bow_velocity

        y_bow = np.clip(y_bow, -1.0, 1.0) # 物理极值保护

        # ----------------------------------------------------
        # 步骤 4: 波的散射与反向传播
        # ----------------------------------------------------
        y_to_neck = y_from_bridge_filt + y_bow
        y_to_bridge = y_from_neck + y_bow

        # 边界反射：撞击固定端(琴马/琴枕)会导致相位反转(-1)，乘以0.998模拟空气阻力衰减
        string_loss = 0.998
        delay_neck[write_ptr] = -y_to_neck * string_loss
        delay_bridge[write_ptr] = -y_to_bridge * string_loss

        out_val = y_to_bridge
        
        # ----------------------------------------------------
        # 步骤 5: 直流阻塞器 (消除波导算法常见的极化偏移积聚)
        # ----------------------------------------------------
        dc_out = out_val - dc_in + 0.995 * dc_out
        dc_in = out_val
        wave[i] = dc_out

        write_ptr = (write_ptr + 1) % MAX_DELAY

    # 输出并同步到可视化数组
    wave = np.clip(wave * 2.5, -1.0, 1.0) # 放大主音量
    latest_wave[:] = np.roll(latest_wave, -frames)
    latest_wave[-frames:] = wave
    outdata[:] = wave.reshape(-1, 1).astype(np.float32)

stream = sd.OutputStream(samplerate=SAMPLING_RATE, channels=1, callback=audio_callback)
stream.start()

# --- 界面与可视化构建 ---
fig = plt.figure(figsize=(12, 8))
fig.canvas.manager.set_window_title("J.O.Smith 物理波导提琴模拟")

# 波形图 (上)
ax_wave = plt.axes([0.1, 0.7, 0.8, 0.25])
line_wave, = ax_wave.plot(np.linspace(0, 2048/SAMPLING_RATE, 2048), np.zeros(2048), lw=2, color='teal')
ax_wave.set_title("实时琴弦震动轨迹 (观察由摩擦力产生的亥姆霍兹锯齿波)")
ax_wave.set_ylim(-1.0, 1.0)
ax_wave.grid(True)

# 频谱图 (中)
ax_spec = plt.axes([0.1, 0.4, 0.8, 0.22])
line_spec, = ax_spec.plot(np.fft.rfftfreq(2048, 1/SAMPLING_RATE), np.zeros(1025), color='crimson')
ax_spec.set_title("频域状态 (运弓压力会实时改变高频泛音的丰富度)")
ax_spec.set_xlim(0, 5000)
ax_spec.set_ylim(0, 50.0)
ax_spec.grid(True)

# 滑块与输入控件 (下)
ax_vel = plt.axes([0.15, 0.25, 0.65, 0.03])
slider_vel = Slider(ax_vel, '运弓速度 (拉动发声)', 0.0, 0.5, valinit=0.0, color='dodgerblue')

ax_pres = plt.axes([0.15, 0.18, 0.65, 0.03])
slider_pres = Slider(ax_pres, '弓子压力 (音色质感)', 1.0, 8.0, valinit=3.0, color='orange')

ax_pos = plt.axes([0.15, 0.11, 0.65, 0.03])
slider_pos = Slider(ax_pos, '拉弓位置 (梳状滤波)', 0.02, 0.4, valinit=0.15, color='mediumpurple')

ax_box = plt.axes([0.15, 0.03, 0.15, 0.05])
text_box = TextBox(ax_box, '左手按弦 (如D4/A4): ', initial='D4')

ax_btn_clear = plt.axes([0.35, 0.03, 0.15, 0.05])
btn_clear = Button(ax_btn_clear, '闷音 (清空琴弦阻尼)', color='lightgray')

# --- 交互回调事件 ---
def update_params(val):
    global bow_velocity, bow_pressure, bow_position
    bow_velocity = slider_vel.val
    bow_pressure = slider_pres.val
    bow_position = slider_pos.val
    
slider_vel.on_changed(update_params)
slider_pres.on_changed(update_params)
slider_pos.on_changed(update_params)

def apply_base_note(text):
    global current_f0
    current_f0 = note_to_freq(text)
text_box.on_submit(apply_base_note)

def damp_string(event):
    delay_neck.fill(0)
    delay_bridge.fill(0)
btn_clear.on_clicked(damp_string)

# 定时刷新图表
def animate(frame):
    # 更新波形图
    line_wave.set_ydata(latest_wave)
    # 更新频谱图
    spec = np.abs(np.fft.rfft(latest_wave))
    line_spec.set_ydata(spec)
    return line_wave, line_spec

ani = animation.FuncAnimation(fig, animate, interval=40, blit=False)

try:
    plt.show()
finally:
    stream.stop()
    stream.close()
    print("物理引擎已关闭。")