import librosa


def get_harmonic_tempo_beats_song(mp3_filename):
    y, sr = librosa.load(mp3_filename)
    y_harmonic = librosa.effects.hpss(y)
    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

    return tempo, beats

def get_precursive_tempo_beats_song(mp3_filename):
    y, sr = librosa.load(mp3_filename)
    y_percussive = librosa.effects.hpss(y)
    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

    return tempo, beats
