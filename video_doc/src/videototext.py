import yt_dlp
from pathlib import Path
from faster_whisper import WhisperModel

class VideoToText:
    def __init__(self):
        self.model = WhisperModel("small", device="cpu", compute_type="int8",download_root="../models")

    def download_youtube_video(self, url, output_path):
        ydl_opts = {
            'extract_audio': True,
            'outtmpl': output_path,
            'format': 'bestaudio'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    def extract_text(self, video_path):
        print(f"Transcribing '{video_path}'...")
        segments, info = self.model.transcribe(video_path, beam_size=5, language="en", condition_on_previous_text=False)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        text = ""
        for segment in segments:
            print(f"  {segment.end/60.0:.3f} minute(s)", end="\r", flush=True)
            text += segment.text + " "
        print("\nTranscription complete.")
        return text
    
if __name__ == "__main__":
    yt_files = [
        ("https://www.youtube.com/watch?v=Qbvs1ncHcwo&pp=ygUDbmRl", "../data/Female_001.mp3", True),
        ("https://www.youtube.com/watch?v=Qbvs1ncHcwo", "../data/Female_002.mp3", True),
        ("https://www.youtube.com/watch?v=N_BHl3FV9NE", "../data/Female_003.mp3", True),
        ("https://www.youtube.com/watch?v=bGnUPGLJ3LA", "../data/Male_001.mp3", True),
        ("https://www.youtube.com/watch?v=5LYzsuQNHnI&pp=0gcJCaIKAYcqIYzv", "../data/Male_002.mp3", True),
        ("https://www.youtube.com/watch?v=V08LTgsJikY", "../data/Male_003.mp3", True)
    ]
    vtt = VideoToText()
    # Download videos if not already downloaded
    for url, audio_path, is_downloaded in yt_files:
        if not is_downloaded:
            vtt.download_youtube_video(url, audio_path)
        else:
            print(f"Audio file '{audio_path}' already exists. Skipping download.")
    
    # Transcribe audio files to text
    for url, audio_path, is_downloaded in yt_files:
        transcribed_path = f"{audio_path.rsplit('.', 1)[0] }.txt"
        if not Path(transcribed_path).exists():
            text = vtt.extract_text(audio_path)
            with open(transcribed_path, "w") as f:
                f.write(text)
        else:
            print(f"Transcribed text file '{transcribed_path}' already exists. Skipping transcription.")
