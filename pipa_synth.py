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
# 核心乐器类：泛音列合成器
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
        if self.pitch_drop_cents > 0:
            attack_drop = self.pitch_drop_cents * velocity * np.exp(-t / self.pitch_drop_time)
        else:
            attack_drop = np.zeros(frames)
            
        if abs(bend_cents) > 0:
            norm_t = np.clip((t - bend_delay) / max(bend_time, 0.001), 0.0, 1.0)
            smooth_curve = (1.0 - np.cos(np.pi * norm_t)) / 2.0
            bend_env = smooth_curve * bend_cents
        else:
            bend_env = np.zeros(frames)
            
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
# 进阶音轨序列器 
# ==========================================
def render_melody(synth, melody_data, bpm=120):
    seconds_per_beat = 60.0 / bpm
    
    current_beat = 0.0
    max_beat = 0.0
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
        note_item = item[0]
        dur_beats = item[1]
        vel = item[2]
        step_beats = item[3] if len(item) > 3 else dur_beats
        articulations = item[4] if len(item) == 5 else {}
            
        duration_sec = dur_beats * seconds_per_beat
        step_sec = step_beats * seconds_per_beat
        
        notes_to_play = [note_item] if isinstance(note_item, str) else note_item
        
        for n_str in notes_to_play:
            if n_str.upper() != "REST":
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
# 音色预设字典
# ==========================================
PLUCK_TIMBRE = {
    1.000: 0.25, 1.998: 0.65, 3.002: 0.75, 4.006: 0.75, 5.012: 0.75, 
    6.020: 0.55, 7.035: 0.38, 8.050: 0.30, 9.070: 0.55, 11.012: 0.35,
    12.04: 0.25, 13.015: 0.20, 14.023: 0.10, 15.030: 0.15, 16.040: 0.12,
    17.034: 0.18, 18.039: 0.10, 19.045: 0.08, 20.060: 0.05, 21.045: 0.05,
    27.067: 0.13 
}

pipa_synth = OvertoneSynth(
    partial_ratios=PLUCK_TIMBRE,
    attack=0.001, decay=1.2, sustain=0.8, release=0.5, damping=0.1,
    pitch_drop_cents=10.0, pitch_drop_time=0.1, noise_amp=0.01
)