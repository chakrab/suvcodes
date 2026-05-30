import numpy as np
import sounddevice as sd

class PitchPlayer:
    """
    Class to generate and play tones of specific frequencies
    """
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def make_tone(self, frequency_hz: float, duration_s: float, volume: float = 1.0) -> np.ndarray:
        """
        Generate a sine wave tone of a given frequency and duration.
        :param frequency_hz: Frequency of the tone in Hz
        :param duration_s: Duration of the tone in seconds
        :param volume: Volume of the tone (0.0 to 1.0)
        :return: Numpy array containing the audio samples
        """
        t = np.linspace(0, duration_s, int(self.sample_rate * duration_s), False)
        note = np.sin(frequency_hz * t * 2 * np.pi)
        # return float32 in range [-1.0, 1.0]
        return note.astype(np.float32)

    def play_tone(self, frequency_hz: float, duration_s: float, volume: float = 1.0):
        """
        Play a tone of a given frequency and duration.
        :param frequency_hz: Frequency of the tone in Hz
        :param duration_s: Duration of the tone in seconds
        :param volume: Volume of the tone (0.0 to 1.0)
        """
        try:
            audio = self.make_tone(frequency_hz, duration_s, volume)
            sd.play(audio, samplerate=self.sample_rate, blocking=True)
        except Exception as e:
            print("Error playing tone:", repr(e))
            raise e
        finally:
            sd.stop()

if __name__ == "__main__":
    player = PitchPlayer()
    N4_octet = [256, 288, 320, 341.3, 384, 426.7, 480]
    for note in N4_octet:
        print("Note:", note)
        #freq = wave_dict[note]
        print("Freq:", note)
        try:
            player.play_tone(note, 1.0, 1.0)
            print("Played OK")
        except Exception as e:
            print("Playback error:", repr(e))
            break
