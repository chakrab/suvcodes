class AudioTools:
    @staticmethod
    def _create_wave_dict():
        """
        Create a dictionary of note names to frequencies using following steps,
            - Read data/freq.txt and create a dictionary of note names to frequencies
            - The file has lines in following format:
                - Line 1: Note
                - Line 2: Frequency (Hz)
                - Line 3: Wavelength (cm)
        :return: Dictionary mapping note names to frequencies (Hz)
        """
        with open('../data/freq.txt', 'r') as f:
            lines = f.readlines()
            wave_dict = {}
            for i in range(0, len(lines), 3):
                note = lines[i].strip()
                freq = float(lines[i + 1].strip().replace('Hz', ''))
                wave_dict[note] = freq
            return wave_dict
        
if __name__ == "__main__":
    wave_dict = AudioTools._create_wave_dict()
    print(wave_dict)