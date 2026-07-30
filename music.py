import numpy as np
import sounddevice as sd
import re

SAMPLING_RATE = 44100
A4_REF_FREQ = 440.0

NOTE_MAP = {
    'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3,
    'E': 4, 'F': 5, 'F#': 6, 'GB': 6, 'G': 7, 'G#': 8,
    'AB': 8, 'A': 9, 'A#': 10, 'BB': 10, 'B': 11
}

def note_to_freq(note_str):
    pattern = r"([A-Ga-g][#bB]?)([0-8])"
    match = re.match(pattern, note_str.strip())
    if not match: return A4_REF_FREQ
    name, octave = match.groups()
    semitone = NOTE_MAP[name.upper().replace('B', 'B')]
    midi_note = (int(octave) + 1) * 12 + semitone
    return A4_REF_FREQ * (2.0 ** ((midi_note - 69) / 12.0))

# ==========================================
# 1. 核心乐器类：泛音列合成器
# ==========================================
class OvertoneSynth:
    def __init__(self, partial_ratios, attack=0.01, decay=0.1, sustain=0.5, release=0.3, damping=0.0, pitch_drop_cents=0.0, pitch_drop_time=0.1, noise_amp=0.0):
        self.ratios = partial_ratios
        self.phases = {ratio: np.random.uniform(0, 2*np.pi) for ratio in partial_ratios.keys()}
        
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.damping = damping
        self.pitch_drop_cents = pitch_drop_cents
        self.pitch_drop_time = pitch_drop_time
        self.noise_amp = noise_amp  

    def generate_note(self, note_str, duration_sec, velocity=1.0, 
                      bend_cents=0.0, bend_delay=0.1, bend_time=0.3, 
                      vib_depth=0.0, vib_rate=5.5, vib_delay=0.2):
        """
        新增参数：
        :param bend_cents: 推拉音的目标音分变化 (例如 200 为推高一个全音)
        :param bend_delay: 拨弦后多久开始推弦 (秒)
        :param bend_time: 推弦动作持续的时间 (秒)
        :param vib_depth: 揉弦深度 (音分)
        :param vib_rate: 揉弦速度 (Hz)
        :param vib_delay: 拨弦后多久开始揉弦 (秒)
        """
        freq = note_to_freq(note_str)
        total_duration = duration_sec + self.release
        frames = int(total_duration * SAMPLING_RATE)
        t = np.linspace(0, total_duration, frames, endpoint=False)
        
        # --- 1. 生成基础 ADSR 包络 ---
        env = np.zeros(frames)
        attack_frames = int(self.attack * SAMPLING_RATE)
        decay_frames = int(self.decay * SAMPLING_RATE)
        release_frames = int(self.release * SAMPLING_RATE)
        hold_frames = int(duration_sec * SAMPLING_RATE) - attack_frames - decay_frames
        
        if hold_frames < 0:
            hold_frames = 0
            decay_frames = int(duration_sec * SAMPLING_RATE) - attack_frames
            
        current_idx = 0
        if attack_frames > 0:
            env[current_idx : current_idx + attack_frames] = np.linspace(0, 1.0, attack_frames)
            current_idx += attack_frames
        if decay_frames > 0:
            env[current_idx : current_idx + decay_frames] = np.linspace(1.0, self.sustain, decay_frames)
            current_idx += decay_frames
        if hold_frames > 0:
            env[current_idx : current_idx + hold_frames] = self.sustain
            current_idx += hold_frames
            
        release_start_val = env[current_idx - 1] if current_idx > 0 else 0
        if release_frames > 0 and current_idx + release_frames <= frames:
            env[current_idx : current_idx + release_frames] = np.linspace(release_start_val, 0, release_frames)
        
# --- 2. 复合音高包络 ---
        # 2.1 拨弦张力下坠
        if self.pitch_drop_cents > 0:
            attack_drop = self.pitch_drop_cents * velocity * np.exp(-t / self.pitch_drop_time)
        else:
            attack_drop = np.zeros(frames)
            
        # 2.2 推拉音 (平滑 S 型曲线)
        if abs(bend_cents) > 0:
            # 获取 0 到 1 的线性归一化时间
            norm_t = np.clip((t - bend_delay) / max(bend_time, 0.001), 0.0, 1.0)
            # 核心修改：利用余弦函数将线性转换为平滑的 S 曲线，完美模拟人类手指推弦的物理阻尼感
            smooth_curve = (1.0 - np.cos(np.pi * norm_t)) / 2.0
            bend_env = smooth_curve * bend_cents
        else:
            bend_env = np.zeros(frames)
            
        # 2.3 揉弦 (延迟渐入的 LFO)
        if vib_depth > 0:
            vib_fade_time = 0.2  
            vib_amp = np.clip((t - vib_delay) / vib_fade_time, 0.0, 1.0) * vib_depth
            vib_lfo = np.sin(2 * np.pi * vib_rate * t)
            vib_env = vib_amp * vib_lfo
        else:
            vib_env = np.zeros(frames)
            
        total_cents_shift = attack_drop + bend_env + vib_env
        freq_array = freq * (2.0 ** (total_cents_shift / 1200.0))

        base_phase = 2 * np.pi * np.cumsum(freq_array) / SAMPLING_RATE

        # --- 3. 叠加泛音生成波形 ---
        wave = np.zeros(frames)
        base_decay_rate = 4.6 / max(self.decay, 0.001) if self.damping > 0 else 0
            
        for ratio, amp in self.ratios.items():
            phase_offset = self.phases[ratio]
            if self.damping > 0:
                partial_decay_rate = base_decay_rate * (ratio ** self.damping)
                partial_env = np.exp(-partial_decay_rate * t)
                wave += amp * partial_env * np.sin(base_phase * ratio + phase_offset)
            else:
                wave += amp * np.sin(base_phase * ratio + phase_offset)
            
        total_amp = sum(self.ratios.values())
        if total_amp > 0: wave /= total_amp
            
        # --- 4. 混入指甲刮擦噪音 ---
        if self.noise_amp > 0:
            noise = np.random.normal(0, 1, frames)
            noise[1:] = noise[1:] - noise[:-1].copy()
            noise[0] = 0
            noise_decay = np.exp(-t * 80.0) 
            wave += noise * noise_decay * self.noise_amp

        return wave * env * velocity

# ==========================================
# 2. 进阶音轨序列器 (支持读取表情字典)
# ==========================================
def render_melody(synth, melody_data, bpm=120):
    seconds_per_beat = 60.0 / bpm
    
    current_beat = 0.0
    max_beat = 0.0
    for item in melody_data:
        # 解包支持 3 到 5 个参数
        dur_beats = item[1]
        step_beats = item[3] if len(item) > 3 else dur_beats
        max_beat = max(max_beat, current_beat + dur_beats)
        current_beat += step_beats
        
    total_seconds = max_beat * seconds_per_beat + synth.release
    total_frames = int(total_seconds * SAMPLING_RATE)
    master_track = np.zeros(total_frames)
    
    current_time_sec = 0.0
    for item in melody_data:
        note_item = item[0]
        dur_beats = item[1]
        vel = item[2]
        step_beats = item[3] if len(item) > 3 else dur_beats
        # 第五个参数是表情字典 (kwargs)
        articulations = item[4] if len(item) == 5 else {}
            
        duration_sec = dur_beats * seconds_per_beat
        step_sec = step_beats * seconds_per_beat
        
        notes_to_play = [note_item] if isinstance(note_item, str) else note_item
        
        for n_str in notes_to_play:
            if n_str.upper() != "REST":
                # 将字典解包传入生成器，实现专属的揉弦与推拉
                note_wave = synth.generate_note(n_str, duration_sec, velocity=vel, **articulations)
                
                start_idx = int(current_time_sec * SAMPLING_RATE)
                end_idx = start_idx + len(note_wave)
                if end_idx > total_frames:
                    end_idx = total_frames
                    note_wave = note_wave[:end_idx - start_idx]
                master_track[start_idx:end_idx] += note_wave
                
        current_time_sec += step_sec
        
    return master_track

# ==========================================
# 3. 创作与播放
# ==========================================
PLUCK_TIMBRE = {
    1.000: 0.25, 1.998: 0.65, 3.002: 0.75, 4.006: 0.75, 5.012: 0.75, 
    6.020: 0.55, 7.035: 0.38, 8.050: 0.30, 9.070: 0.55, 11.012: 0.35,
    12.04: 0.25, 13.015: 0.20, 14.023: 0.10, 15.030: 0.15, 16.040: 0.12,
    17.034: 0.18, 18.039: 0.10, 19.045: 0.08, 20.060: 0.05, 21.045: 0.05,
    27.067: 0.13 
}

my_synth = OvertoneSynth(
    partial_ratios=PLUCK_TIMBRE,
    attack=0.001, decay=1.2, sustain=0.8, release=0.5, damping=0.1,
    pitch_drop_cents=10.0, pitch_drop_time=0.1, noise_amp=0.01
)

# 🎶 全新《赛博琵琶》曲谱
# 格式解析：
# 1. ("D4", 1.0, 0.8) -> 普通单音
# 2. (["D3", "A3", "D4"], 1.0, 0.9) -> 三音和弦同时发声
# 3. ("D3", 2.0, 0.8, 0.05) -> 持续2拍，但仅过0.05拍就执行下一个音 (扫弦/琶音利器！)

# 🎶 一分钟完整乐章（利用列表动态组装）
cyberpunk_pipa = []

# --- 乐章一：散板引子 (变速起势) ---
# 利用步进拍数的不断缩短，完美模拟人类起手由慢到极快的加速拨弦 (Accelerando)
cyberpunk_pipa.extend([
    ("A2", 2.0, 0.9, 1.5), ("D4", 0.5, 0.6, 0.5),
    ("A2", 1.5, 0.9, 1.0), ("F4", 0.5, 0.7, 0.5),
    ("A2", 1.0, 1.0, 0.5), ("G4", 0.5, 0.8, 0.5),
    ("A2", 0.5, 1.0, 0.25), ("A4", 0.25, 0.9, 0.25),
    ("A2", 0.25, 1.0, 0.125), ("C5", 0.125, 1.0, 0.125),
    ("A2", 0.125, 1.0, 0.125), ("D5", 0.125, 1.0, 0.125),
    # 引子重音收尾，夸张推弦
    ("D5", 4.0, 1.0, 4.0, {"bend_cents": 150, "bend_time": 1.0, "vib_depth": 25, "vib_delay": 0.5}),
    ("REST", 1.0, 0)
])

# --- 乐章二：空弦 A2 律动主歌 (D minor) ---
# 包含强弱分明(Velocity变化)、附点与切分音的交织
for _ in range(2):
    cyberpunk_pipa.extend([
        ("A2", 0.5, 1.0, 0.25), ("D4", 0.5, 0.8, 0.25),
        ("A2", 0.25, 0.4, 0.25), ("F4", 0.5, 0.9, 0.25),
        ("A2", 0.25, 0.5, 0.25), ("E4", 0.5, 0.8, 0.25),
        ("A2", 0.5, 0.9, 0.5),
        # 复杂节奏：三连音 (3个音等分1拍)
        ("G4", 0.333, 0.9), ("F4", 0.333, 0.7), ("E4", 0.334, 0.6),
        ("D4", 1.0, 0.9, 1.0),
        
        # 变奏段：主旋律推高
        ("A2", 0.5, 1.0, 0.25), ("A4", 0.5, 0.8, 0.25),
        ("A2", 0.25, 0.4, 0.25), ("G4", 0.5, 0.9, 0.25),
        ("A2", 0.25, 0.5, 0.25), ("F4", 0.5, 0.8, 0.25),
        # 步进设为 0，让 A2 持续音与后面的三连音同时发生 (复调魔法)
        ("A2", 0.5, 0.9, 0.0), 
        ("E4", 0.333, 0.9), ("F4", 0.333, 0.7), ("G4", 0.334, 0.6),
        ("A4", 1.0, 0.9, 1.0, {"vib_depth": 15, "vib_delay": 0.2})
    ])

# --- 乐章三：多声部高潮 (双声部轮指 D minor) ---
tremolo_melody = [
    ("D3", 2.0), ("F3", 2.0), ("G3", 2.0), ("A3", 2.0),
    ("Bb3", 2.0), ("C4", 2.0), ("D4", 4.0)
]
for bass_note, duration in tremolo_melody:
    # 放置低频和声基底，步进时间 0
    cyberpunk_pipa.append((bass_note, duration, 0.9, 0.0))
    # 计算极速轮指的数量 (32分音符)
    num_tremolo_notes = int(duration / 0.125) 
    for i in range(num_tremolo_notes):
        # 表情细节：大拇指拨弦重(重音)，四指连滚轻，模拟真实物理轮指的层次感
        trem_vel = 0.95 if i % 4 == 0 else 0.45 + np.random.uniform(-0.1, 0.1)
        cyberpunk_pipa.append(("D5", 0.125, trem_vel, 0.125))

# --- 过渡段：准备转调 ---
cyberpunk_pipa.extend([
    (["A3", "E4", "A4"], 0.75, 1.0, 0.75), (["G3", "D4", "G4"], 0.25, 0.7, 0.25),
    (["F3", "C4", "F4"], 0.75, 0.9, 0.75), (["E3", "B3", "E4"], 0.25, 0.6, 0.25),
    # 疯狂的 A2 空弦滚奏加速打底
    ("A2", 0.5, 1.0, 0.25), ("A2", 0.5, 0.8, 0.25),
    ("A2", 0.5, 1.0, 0.25), ("A2", 0.5, 0.8, 0.25),
    ("A2", 0.25, 0.9, 0.125), ("A2", 0.25, 0.9, 0.125),
    ("A2", 0.25, 0.9, 0.125), ("A2", 0.25, 0.9, 0.125),
    ("A2", 0.25, 1.0, 0.125), ("A2", 0.25, 1.0, 0.125),
    ("A2", 0.25, 1.0, 0.125), ("A2", 0.25, 1.0, 0.125),
    ("REST", 0.5, 0)
])

# --- 乐章四：史诗转调 (Modulation to Eb Minor) ---
cyberpunk_pipa.extend([
    # 转调大扫弦 (步进极短，依次发声)
    ("Eb2", 2.0, 1.0, 0.05), ("Bb2", 1.95, 0.9, 0.05), ("Eb3", 1.9, 0.9, 0.05), 
    ("Gb3", 1.85, 0.9, 0.05), ("Bb3", 1.8, 1.0, 1.8),
    
    # 转调后充满张力的附点旋律
    ("Eb4", 0.75, 1.0, 0.75), ("Gb4", 0.25, 0.8, 0.25),
    ("Bb4", 1.0, 0.9, 1.0, {"vib_depth": 20, "vib_delay": 0.2}),
    
    ("Ab4", 0.333, 0.9), ("Gb4", 0.333, 0.8), ("F4", 0.334, 0.7),
    ("Eb4", 1.0, 0.9, 1.0),
    
    # 大跨度推弦
    ("Eb2", 2.0, 1.0, 0.05), ("Bb2", 1.95, 0.9, 0.05), ("Eb3", 1.9, 0.9, 0.05), 
    ("Gb3", 1.85, 0.9, 0.05), ("Db4", 1.8, 1.0, 1.8),
    
    ("Eb4", 0.75, 1.0, 0.75), ("F4", 0.25, 0.8, 0.25),
    ("Gb4", 1.0, 0.9, 1.0),
    
    ("F4", 0.333, 0.9), ("Eb4", 0.333, 0.8), ("Db4", 0.334, 0.7),
    ("Eb4", 2.0, 1.0, 2.0, {"bend_cents": 100, "bend_time": 0.5, "vib_depth": 15, "vib_delay": 0.3})
])

# --- 乐章五：降E小调双声部华彩 ---
tremolo_melody_eb = [
    ("Eb3", 2.0), ("Gb3", 2.0), ("Ab3", 2.0), ("Bb3", 2.0),
    ("B3", 2.0), ("Db4", 2.0), ("Eb4", 4.0)
]
for bass_note, duration in tremolo_melody_eb:
    cyberpunk_pipa.append((bass_note, duration, 0.9, 0.0))
    num_tremolo_notes = int(duration / 0.125)
    for i in range(num_tremolo_notes):
        # 高阶表情：加入正弦波控制音量，制造一波一波的“海浪感 (Swell)”
        wave_dyn = np.sin((i / num_tremolo_notes) * np.pi) * 0.3
        trem_vel = 0.9 if i % 4 == 0 else 0.4 + wave_dyn + np.random.uniform(-0.05, 0.05)
        trem_vel = np.clip(trem_vel, 0.0, 1.0)
        cyberpunk_pipa.append(("Eb5", 0.125, trem_vel, 0.125))

# --- 尾声：终局大扫弦 (Eb Minor 11) ---
cyberpunk_pipa.extend([
    ("REST", 1.0, 0),
    ("Eb2", 6.0, 0.9, 0.06), ("Bb2", 5.94, 0.9, 0.06), ("Eb3", 5.88, 0.9, 0.06), 
    ("Gb3", 5.82, 0.9, 0.06), ("Bb3", 5.76, 0.9, 0.06), ("Db4", 5.7, 0.9, 0.06),
    # 夸张的延音与揉弦结束整首曲子
    ("F4", 5.64, 1.0, 5.64, {"vib_depth": 30.0, "vib_rate": 6.0, "vib_delay": 0.5})
])

cyberpunk_pipa.extend([
    ("D2", 4.0, 0.7, 0.05), ("A2", 3.95, 0.7, 0.05), ("D3", 3.9, 0.8, 0.05), ("F3", 3.85, 0.8, 0.05), ("A3", 3.8, 0.9, 3.8),
    
    # 附点节奏，并在旋律中强调 C#4（D和声小调的导音），制造浓烈的异域感
    ("D4", 0.75, 0.8), ("E4", 0.25, 0.7), ("F4", 1.0, 0.9),
    ("G4", 0.75, 0.8), ("Bb4", 0.25, 0.7), ("A4", 1.0, 0.8),
    ("C#4", 0.5, 0.7), ("D4", 1.0, 0.9), ("E4", 0.5, 0.8), # C#4 和声小调色彩
    ("F4", 1.5, 0.9, 1.5, {"bend_cents": 100, "bend_delay":0.2, "bend_time":0.5}), ("E4", 0.5, 0.7),
])

# --- 乐章二：空弦 A2 律动主歌 (D Harmonic Minor) 约 32 拍 ---
for _ in range(2):
    cyberpunk_pipa.extend([
        # 空弦 A2 穿插的踏板音 (Pedal Point) 律动
        ("A2", 0.5, 1.0, 0.25), ("D4", 0.5, 0.8, 0.25),
        ("A2", 0.25, 0.4, 0.25), ("F4", 0.5, 0.9, 0.25),
        ("A2", 0.25, 0.5, 0.25), ("Bb4", 0.5, 0.8, 0.25),
        ("A2", 0.5, 0.9, 0.5),
        
        # 三连音下行
        ("A4", 0.333, 0.9), ("G4", 0.333, 0.7), ("F4", 0.334, 0.6),
        ("E4", 1.0, 0.9, 1.0),
        
        # 属和弦色彩 (A Major，利用 C#4 与 A2 空弦对撞)
        ("A2", 0.5, 1.0, 0.25), ("C#4", 0.5, 0.8, 0.25), 
        ("A2", 0.25, 0.4, 0.25), ("E4", 0.5, 0.9, 0.25),
        ("A2", 0.25, 0.5, 0.25), ("G4", 0.5, 0.8, 0.25),
        ("A2", 0.5, 0.9, 0.0), # 步进归零，让A2垫底
        
        ("F4", 0.333, 0.9), ("E4", 0.333, 0.7), ("C#4", 0.334, 0.6),
        ("D4", 1.0, 0.9, 1.0, {"vib_depth": 15, "vib_delay": 0.2})
    ])

# --- 乐章三：多声部轮指 (带强弱与旋律游走) 约 36 拍 ---
# 轮指不再是单一的音，而是带有旋律线的游走，低音则是和声走向
tremolo_melody_d = [
    # (低音和弦根音, 轮指旋律音, 持续拍数)
    ("D3", "D5", 2.0), ("D3", "E5", 2.0),
    ("Bb2", "F5", 2.0), ("Bb2", "D5", 2.0),
    ("G2", "Bb4", 2.0), ("G2", "G5", 2.0),
    ("A2", "F5", 2.0), ("A2", "E5", 2.0),
    
    # 更高把位的轮指激化
    ("D3", "A5", 2.0), ("D3", "G5", 2.0),
    ("Bb2", "F5", 2.0), ("Bb2", "E5", 2.0),
    ("A2", "D5", 2.0), ("A2", "C#5", 2.0), # 轮指打在导音 C# 上制造极强紧张感
    ("D3", "D5", 4.0)
]

for bass_note, trem_note, duration in tremolo_melody_d:
    # 放置低音，步进0
    cyberpunk_pipa.append((bass_note, duration, 0.9, 0.0))
    # 每拍 8 个音 (32分音符) 极速轮指
    num_tremolo_notes = int(duration / 0.125) 
    for i in range(num_tremolo_notes):
        # 轮指高阶动态：加入正弦波控制音量，制造一波一波的“海浪感 (Swell)”
        wave_dyn = np.sin((i / num_tremolo_notes) * np.pi) * 0.4
        
        # 强拍重音(重音)，次强拍(次重)，其他指头轻滚
        if i % 8 == 0:
            trem_vel = 0.95
        elif i % 4 == 0:
            trem_vel = 0.8
        else:
            trem_vel = 0.4 + wave_dyn + np.random.uniform(-0.05, 0.05)
            
        trem_vel = np.clip(trem_vel, 0.0, 1.0)
        cyberpunk_pipa.append((trem_note, 0.125, trem_vel, 0.125))

# --- 乐章四：桥段与转调爆发 (Modulation to Eb Harmonic Minor) 约 24 拍 ---
cyberpunk_pipa.extend([
    # 减七和弦急促爬升，极具史诗感
    (["C#4", "E4", "G4", "Bb4"], 1.0, 0.9, 1.0),
    (["E4", "G4", "Bb4", "C#5"], 1.0, 0.9, 1.0),
    (["G4", "Bb4", "C#5", "E5"], 1.0, 1.0, 1.0),
    (["Bb4", "C#5", "E5", "G5"], 1.0, 1.0, 1.0, {"bend_cents": 100, "bend_time": 1.0}),
    ("REST", 1.0, 0),
    
    # 转调降 E 和声小调！(加入 D natural 作为导音)
    ("Eb2", 2.0, 1.0, 0.05), ("Bb2", 1.95, 0.9, 0.05), ("Eb3", 1.9, 0.9, 0.05), 
    ("Gb3", 1.85, 0.9, 0.05), ("Bb3", 1.8, 1.0, 1.8),
    
    ("Eb4", 0.75, 1.0, 0.75), ("Gb4", 0.25, 0.8, 0.25),
    ("Bb4", 1.0, 0.9, 1.0, {"vib_depth": 20, "vib_delay": 0.2}),
    
    ("Ab4", 0.333, 0.9), ("Gb4", 0.333, 0.8), ("F4", 0.334, 0.7),
    ("Eb4", 1.0, 0.9, 1.0),
    
    # B Major 走向
    ("B2", 2.0, 1.0, 0.05), ("Gb3", 1.95, 0.9, 0.05), ("B3", 1.9, 0.9, 0.05), 
    ("Eb4", 1.85, 0.9, 0.05), ("Gb4", 1.8, 1.0, 1.8),
    
    ("F4", 0.333, 0.9), ("Eb4", 0.333, 0.8), ("D4", 0.334, 0.7), # D4 是 Eb 和声小调的核心灵魂！
    ("Eb4", 2.0, 1.0, 2.0, {"bend_cents": 100, "bend_time": 0.5, "vib_depth": 15, "vib_delay": 0.3})
])

# --- 乐章五：降E和声小调 终极狂暴轮指 约 36 拍 ---
tremolo_melody_eb = [
    ("Eb3", "Bb4", 2.0), ("Eb3", "Eb5", 2.0),
    ("B2", "Gb5", 2.0), ("B2", "F5", 2.0),
    ("Ab2", "Eb5", 2.0), ("Ab2", "B4", 2.0),
    ("Bb2", "D5", 2.0), ("Bb2", "F5", 2.0), # D5 和声小调色彩
    
    ("Eb3", "Bb5", 2.0), ("Eb3", "Ab5", 2.0),
    ("B2", "Gb5", 2.0), ("B2", "F5", 2.0),
    ("Bb2", "Eb5", 2.0), ("Bb2", "D5", 2.0),
    ("Eb3", "Eb5", 4.0)
]
for bass_note, trem_note, duration in tremolo_melody_eb:
    cyberpunk_pipa.append((bass_note, duration, 0.9, 0.0))
    num_tremolo_notes = int(duration / 0.125)
    for i in range(num_tremolo_notes):
        wave_dyn = np.sin((i / num_tremolo_notes) * np.pi) * 0.4
        if i % 8 == 0: trem_vel = 0.95
        elif i % 4 == 0: trem_vel = 0.85
        else: trem_vel = 0.45 + wave_dyn + np.random.uniform(-0.05, 0.05)
        trem_vel = np.clip(trem_vel, 0.0, 1.0)
        cyberpunk_pipa.append((trem_note, 0.125, trem_vel, 0.125))

# --- 尾声：悬浮终止式 (Suspended Cadence 回归 D 调) 约 24 拍 ---
cyberpunk_pipa.extend([
    ("REST", 1.0, 0),
    # 属和弦爆发，将耳朵硬拉回 D 调
    (["A1", "A2", "C#3", "E3", "G3", "A3"], 4.0, 1.0, 4.0, {"vib_depth": 10, "vib_delay": 1.0}),
    ("REST", 1.0, 0),
    
    # 终极挂留和弦 (Dsus2: D - E - A)
    # 缓慢的大琶音扫过，留下空灵的悬浮感与终止感
    ("D2", 12.0, 0.9, 0.08), 
    ("A2", 11.92, 0.9, 0.08), 
    ("D3", 11.84, 0.9, 0.08), 
    ("E3", 11.76, 0.9, 0.08), # E (Sus2 特征音)
    ("A3", 11.68, 0.9, 0.08), 
    ("E4", 11.60, 0.9, 0.08), 
    
    # 最后一个悬念高音 E4 挂留，伴随夸张的揉弦，并在最后慢慢消散
    ("A4", 11.52, 1.0, 11.52, {"vib_depth": 35.0, "vib_rate": 6.5, "vib_delay": 1.5})
])

print("正在渲染...")
# 约 82 拍，在 85 BPM 下时长约为 58 秒 + 尾音释放 = 完美一分钟
final_audio = render_melody(my_synth, cyberpunk_pipa, bpm=85)

# 为和声留出动态余量，防止多音符叠加爆音
final_audio *= 0.4

# 播放音频
print("开始播放...")
sd.play(final_audio, SAMPLING_RATE)
sd.wait()
print("播放结束。")