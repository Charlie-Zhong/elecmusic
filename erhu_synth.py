import numpy as np
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
# 核心二胡音色预设 (丰富的奇偶次泛音，模拟摩擦感)
# ==========================================
BOWED_TIMBRE = {
    1.00: 0.6, 2.00: 0.85, 3.00: 0.65, 4.00: 0.5, 5.00: 0.4, 
    6.00: 0.25, 7.00: 0.15, 8.00: 0.1, 9.00: 0.08, 10.00: 0.05
}

# ==========================================
# 二胡专属合成器类 (强化连弦与滑音逻辑)
# ==========================================
class ErhuSynth:
    def __init__(self, partial_ratios=BOWED_TIMBRE, attack=0.15, decay=0.1, sustain=0.9, release=0.4, noise_amp=0.02):
        self.ratios = partial_ratios
        self.phases = {ratio: np.random.uniform(0, 2*np.pi) for ratio in partial_ratios.keys()}
        
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.noise_amp = noise_amp  # 弓毛摩擦松香的底噪

    def generate_note(self, note_str, duration_sec, velocity=1.0, 
                      slide_from=None, slide_time=0.3,  # 二胡核心：滑音参数
                      vib_depth=15.0, vib_rate=5.5, vib_delay=0.2): # 默认带有轻微揉弦
        
        target_freq = note_to_freq(note_str)
        total_duration = duration_sec + self.release
        frames = int(total_duration * SAMPLING_RATE)
        t = np.linspace(0, total_duration, frames, endpoint=False)
        
        # --- 1. 生成拉弦乐器的 ADSR 包络 (起音慢，延音长) ---
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
            # 采用平滑释放，模拟弓子离开琴弦后的余音
            release_curve = np.linspace(1.0, 0.0, release_frames) ** 2
            env[current_idx : current_idx + release_frames] = release_start_val * release_curve
        
        # --- 2. 复合音高控制 (滑音 + 揉弦) ---
        slide_env = np.zeros(frames)
        if slide_from:
            start_freq = note_to_freq(slide_from)
            # 计算起始音和目标音之间的音分差
            cents_diff = 1200 * np.log2(start_freq / target_freq)
            
            slide_frames = int(slide_time * SAMPLING_RATE)
            slide_frames = min(slide_frames, frames)
            
            norm_t = np.linspace(0, 1.0, slide_frames)
            # 核心物理模拟：S型滑动曲线 (从 1 平滑过渡到 0)
            smooth_curve = (1.0 + np.cos(np.pi * norm_t)) / 2.0
            slide_env[:slide_frames] = cents_diff * smooth_curve
            
        # 揉弦 (左手按弦的周期性颤动)
        if vib_depth > 0:
            vib_fade_time = 0.3  # 揉弦渐入时间
            vib_amp = np.clip((t - vib_delay) / vib_fade_time, 0.0, 1.0) * vib_depth
            vib_lfo = np.sin(2 * np.pi * vib_rate * t)
            vib_env = vib_amp * vib_lfo
        else:
            vib_env = np.zeros(frames)
            
        total_cents_shift = slide_env + vib_env
        freq_array = target_freq * (2.0 ** (total_cents_shift / 1200.0))

        # 积分计算相位
        base_phase = 2 * np.pi * np.cumsum(freq_array) / SAMPLING_RATE

        # --- 3. 叠加泛音生成波形 ---
        wave = np.zeros(frames)
        for ratio, amp in self.ratios.items():
            phase_offset = self.phases[ratio]
            wave += amp * np.sin(base_phase * ratio + phase_offset)
            
        total_amp = sum(self.ratios.values())
        if total_amp > 0:
            wave /= total_amp
            
        # --- 4. 混入马尾弓毛与琴弦摩擦的宽带噪音 ---
        if self.noise_amp > 0:
            noise = np.random.normal(0, 1, frames)
            # 简单的高通滤波处理，保留沙沙声
            noise[1:] = noise[1:] - noise[:-1].copy() * 0.8
            wave += noise * self.noise_amp

        return wave * env * velocity

# ==========================================
# 二胡专属序列器
# ==========================================
def render_erhu(synth, melody_data, bpm=80):
    seconds_per_beat = 60.0 / bpm
    current_beat = 0.0
    max_beat = 0.0
    
    # 计算总时长
    for item in melody_data:
        dur_beats = item[1]
        step_beats = item[3] if len(item) > 3 else dur_beats
        max_beat = max(max_beat, current_beat + dur_beats)
        current_beat += step_beats
        
    total_seconds = max_beat * seconds_per_beat + synth.release
    total_frames = int(total_seconds * SAMPLING_RATE)
    master_track = np.zeros(total_frames)
    
    current_time_sec = 0.0
    for item in melody_data:
        note_str = item[0]
        dur_beats = item[1]
        vel = item[2]
        step_beats = item[3] if len(item) > 3 else dur_beats
        articulations = item[4] if len(item) == 5 else {}
            
        duration_sec = dur_beats * seconds_per_beat
        step_sec = step_beats * seconds_per_beat
        
        if note_str.upper() != "REST":
            # 动态传入 kwargs 实现每个音不同的滑音和揉弦策略
            note_wave = synth.generate_note(note_str, duration_sec, velocity=vel, **articulations)
            
            start_idx = int(current_time_sec * SAMPLING_RATE)
            end_idx = start_idx + len(note_wave)
            if end_idx > total_frames:
                end_idx = total_frames
                note_wave = note_wave[:end_idx - start_idx]
            master_track[start_idx:end_idx] += note_wave
                
        current_time_sec += step_sec
        
    return master_track