import os
import csv
from pytube import YouTube
from pydub import AudioSegment
import yt_dlp
import re

# Base directory
BASE_DIR = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Training/Datasets/test"

# List of 25 chord labels
CHORD_LABELS = [
    "N",
    "C",
    "C:min",
    "C#",
    "C#:min",
    "D",
    "D:min",
    "D#",
    "D#:min",
    "E",
    "E:min",
    "F",
    "F:min",
    "F#",
    "F#:min",
    "G",
    "G:min",
    "G#",
    "G#:min",
    "A",
    "A:min",
    "A#",
    "A#:min",
    "B",
    "B:min",
]


# Create folder for each chord label
def prepare_folders():
    for chord in CHORD_LABELS:
        os.makedirs(os.path.join(BASE_DIR, chord), exist_ok=True)


def download_audio(song_name, youtube_url, chord_label):
    try:
        print(f"▶️ Downloading: {song_name} ({chord_label})")

        safe_name = re.sub(r'[\\/*?:"<>|]', "", song_name)
        output_filename = f"[{chord_label}] - {safe_name}.wav"
        output_path = os.path.join(BASE_DIR, chord_label, output_filename)

        temp_download_path = f"temp_{safe_name}.%(ext)s"  # Let yt_dlp decide extension

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": temp_download_path,
            "quiet": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url.strip(), download=True)
            downloaded_filename = ydl.prepare_filename(info)
            downloaded_filename = os.path.splitext(downloaded_filename)[0] + ".wav"

        # Move file to correct folder
        os.rename(downloaded_filename, output_path)
        print(f"✅ Saved: {output_path}")

    except Exception as e:
        print(f"❌ Error processing {song_name}: {e}")


# Process song list
def process_song_list(song_list):
    prepare_folders()
    for song in song_list:
        song_name, url, chord = song
        if chord not in CHORD_LABELS:
            print(f"⚠️ Chord '{chord}' is invalid. Skipping song: {song_name}")
            continue
        download_audio(song_name.strip(), url.strip(), chord.strip())


# Example usage
if __name__ == "__main__":
    # Song list [Song name, YouTube URL, Main chord]
    songs = [
        # ["Mưa Hồng - Lê Hiếu", "https://www.youtube.com/watch?v=Xvbj-0gFfWQ", "C"],
        # ["Mưa Hồng - Lệ Quyên", "https://www.youtube.com/watch?v=FfRRBRfXAxk", "G"],
        # ["Mưa Hồng - Khánh Ly", "https://www.youtube.com/watch?v=LyksQotj-vY", "E"],
        # ["Mưa hồng - Elvis Phương", "https://youtu.be/RzElo0UvxMw", "B"],
        # ["Mưa hồng - Cao Duy", "https://youtu.be/te0YoW4Qk9E", "C"],
        # ["Mưa hồng - Khánh Ly", "https://youtu.be/LyksQotj-vY", "F"],
        # ["Mưa hồng - Ngọc Lan", "https://youtu.be/ZuLkmJcP6tE", "G"],
        # ["Mưa hồng - Kiều Nga", "https://youtu.be/HgsfaCPi9N0", "F"],
        # ["Mưa hồng - Thanh Hà", "https://youtu.be/i91or_7yAzY", "G"],
        # ["Mưa hồng - Thiên Kim", "https://youtu.be/fro9ZSp-lTo", "F"],
        # ["Mưa hồng - Hồng Nhung", "https://youtu.be/j3kpHVMEpv0", "F"],
        # ["Diễm xưa - Lệ Quyên", "https://www.youtube.com/watch?v=j08rG-aD9Lo", "B:min"],
        # ["Diễm xưa - Khánh Ly", "https://youtu.be/lSUvI-w36x0", "A:min"],
        # ["Diễm xưa - Như Mai", "https://youtu.be/HvaYTx2y7lc", "A:min"],
        # ["Diễm xưa - Tuấn Vũ", "https://youtu.be/TLSuLz_d6Xg", "B:min"],
        # ["Diễm xưa - Lê Thanh Huyền Trân", "https://youtu.be/bKnQAyMf8lA", "B:min"],
        # ["Diễm xưa - Don Hồ", "https://youtu.be/SISWructcso", "E:min"],
        # ["Diễm xưa - Trần Thu Hà", "https://youtu.be/n8uKwxDHlxo", "C:min"],
        # ["Diễm xưa - Phạm Thu Hà", "https://youtu.be/MHi2HJStOlM", "A:min"],
        # ["Diễm xưa - Mỹ Lệ", "https://youtu.be/ASJGu3XaRKI", "B:min"],
        # ["Mùa thu cho em - Ngọc Lan", "https://www.youtube.com/watch?v=7Gsdkt5sv6s", "C"],
        # ["Mùa thu cho em - Khánh Hà", "https://youtu.be/T-XBKhb8G2k", "C#"],
        # ["Mùa thu cho em - Xuân Phú", "https://youtu.be/Mpb3Y10UzwU", "G"],
        # ["Mùa thu cho em - Như Mai", "https://youtu.be/sW-bK0YWG7k", "D"],
        # ["Mùa thu cho em - Ngọc Lan", "https://youtu.be/TCxqRoL1SRk", "C#"],
        # ["Mùa thu cho em - Ý Lan", "https://youtu.be/C1KoD-S9ZwY", "C"],
        # ["Mùa thu cho em - Phương Vy", "https://youtu.be/zOy4ozLQ5bg", "E"],
        # ["Mùa thu cho em - Tùng Dương", "https://youtu.be/Os8cSBQfQk0", "A"],
        # ["Mùa thu cho em - Giang Hồng Ngọc", "https://youtu.be/NViPjXGXTVA", "D"],
        # ["Mùa thu cho em - Phi Khanh", "https://youtu.be/d1O-IWekDo8", "D"],
        # ["Mùa thu cho em - Đức Huy", "https://youtu.be/p7tJPiB1_IQ", "F"],
        # ["Mùa thu cho em - Phạm Thu Hà", "https://youtu.be/egHFIV5raOY", "C"],
        # ["Mùa thu cho em - Ngọc Huệ", "https://youtu.be/UhyaGLaizKc", "E"],
        # ["Lệ đá - Bằng Kiều", "https://www.youtube.com/watch?v=GeeH5xFHtHQ&t=3s", "D"],
        # ["Lệ đá - Lệ Thu", "https://youtu.be/wuZv3sxKNas", "E"],
        # ["Lệ đá - Ngọc Lan", "https://youtu.be/t3Oye24X9B4", "D"],
        # ["Lệ đá - Hương Lan", "https://youtu.be/xqfdnG3WtMk", "E"],
        # ["Lệ đá - Bảo Yến", "https://youtu.be/92sMq8VcwAw", "D"],
        # ["Lệ đá - Cẩm Ly", "https://youtu.be/Y8oMMoA33rI", "E"],
        # ["Lệ đá - Giang Hồng Ngọc", "https://youtu.be/UcUJVEvYnBw", "F"],
        # ["Dấu tình sầu - Ngọc Lan", "https://www.youtube.com/watch?v=GU_U_f9ZxDE", "E:min"],
        # ["Dấu tình sầu - Don Hồ", "https://youtu.be/emF9BQZbN54", "A:min"],
        # ["Dấu tình sầu - Chế Linh", "https://youtu.be/fLMPFsfTvHw", "A:min"],
        # ["Dấu tình sầu - Khánh Ly", "https://youtu.be/1Id1HNECoXs", "D:min"],
        # ["Dấu tình sầu - Kiều Nga", "https://youtu.be/D9Ora4Fuwjg", "D:min"],
        # ["Dấu tình sầu - Lệ Quyên", "https://youtu.be/OLNwvPq4cCg", "E:min"],
        # ["Dấu tình sầu - Quỳnh Lan", "https://youtu.be/E26vn58LCCA", "D:min"],
        # ["Dấu tình sầu - Cao Thái Sơn", "https://youtu.be/rVtOVBQmmpM", "C:min"],
        # ["Dấu tình sầu - Hương Lan", "https://youtu.be/4Wez_H4cFqU", "E:min"],
        # ["Dấu tình sầu - Giang Hồng Ngọc", "https://youtu.be/76CM4tNj83U", "E:min"],
        # ["Tuổi đá buồn - Khánh Ly", "https://www.youtube.com/watch?v=QiIwyXAM0W8", "A:min"],
        # ["Tuổi đá buồn - Quang Dũng", "https://www.youtube.com/watch?v=JoOpkcvgthw", "E:min"],
        # ["Tuổi đá buồn - Ngọc Lan", "https://youtu.be/QrtLPKZB4_I", "C:min"],
        # ["Tuổi đá buồn - Hồng Nhung", "https://youtu.be/VGxC_BO9cyA", "C:min"],
        # ["Tuổi đá buồn - Lệ Thu", "https://youtu.be/JxXalNWTdWs", "A:min"],
        # ["Tuổi đá buồn - Bảo Yến", "https://youtu.be/cPZCJPZ_yDc", "G:min"],
        # ["Tuổi đá buồn - Elvis Phương", "https://youtu.be/w6WoCNJeQjY", "E:min"],
        # ["Tuổi đá buồn - Lệ Hồng", "https://youtu.be/2-F-lScdJ3M", "B:min"],
        # ["Tuổi đá buồn - Như Mai", "https://youtu.be/sxOcL98urv8", "C:min"],
        # ["Biển tình - Quang Lê", "https://www.youtube.com/watch?v=DGAxiuoG9g8", "F"],
        # ["Biển tình - Lệ Quyên", "https://youtu.be/EKm-cYgOF9U", "C"],
        # ["Biển tình - Thanh Tuyền", "https://youtu.be/yYHg0IK-ay4", "C"],
        # ["Biển tình - Quang Lê", "https://youtu.be/zs3D2h_VTtM", "F"],
        # ["Biển tình - Đàm Vĩnh Hưng", "https://youtu.be/iHjFSq1fnhA", "F#"],
        # ["Biển tình - Dương Đình Trí", "https://youtu.be/2YgLFhNzfw4", "E"],
        # ["Biển tình - Dương Ngọc Thái", "https://youtu.be/mtiy4hibEIs", "F"],
        # ["Biển tình - Hương Lan", "https://youtu.be/PFabhx8lkRI", "B"],
        # ["Biển tình - Hạ Vy", "https://youtu.be/scTgt1I5dV8", "C"],
        # ["Biển tình - Hoàng Châu", "https://youtu.be/NIiAuhehNBo", "C"],
        # ["Biển tình - Tố My", "https://youtu.be/ioh17rCw4O8", "D"],
        # ["Biển tình - Chu Thuý Quỳnh", "https://youtu.be/5wX2bjIFLEI", "C"],
        # ["Phút cuối - Bằng Kiều", "https://www.youtube.com/watch?v=W8IbCUHfOnQ", "A"],
        # ["Phút cuối - Dương Ngọc Thái", "https://youtu.be/rEwTeD2Z9Jo", "F"],
        # ["Phút cuối - Sơn Tuyền", "https://youtu.be/DofJ7O03nus", "D"],
        # ["Phút cuối - Thiên Kim", "https://youtu.be/diA9XUUI1gw", "A"],
        # ["Phút cuối - Thiên Trang", "https://youtu.be/WoPl52uIrzU", "A"],
        # ["Phút cuối - Hạ Vy", "https://youtu.be/fZxCL2LVat8", "C"],
        # ["Phút cuối - Đàm Vĩnh Hưng", "https://youtu.be/w3oiXOmL6W8", "G"],
        # ["Phút cuối - Tuấn Vũ", "https://youtu.be/ByvDja7RM7E", "D#"],
        # ["Thành phố sương - Hà Anh Tuấn", "https://www.youtube.com/watch?v=AtBNDsPkq40", "E"],
        # ["Thành phố sương - Đình Nguyên", "https://www.youtube.com/watch?v=D6qkMckLdu4", "D"],
        # ["Mưa chiều kỷ niệm - Tuấn Vũ", "https://www.youtube.com/watch?v=jcwgz1ysUig", "A"],
        # ["Mưa chiều kỷ niệm - Cẩm Ly", "https://youtu.be/ebumyRkdm24", "E"],
        # ["Mưa chiều kỷ niệm - Hương Lan", "https://youtu.be/Xa7TDFWQmHc", "E"],
        # ["Mưa chiều kỷ niệm - Đông Đào", "https://youtu.be/HaGiiQdzFPM", "E"],
        # ["Mưa chiều kỷ niệm - Bảo Yến", "https://youtu.be/zPZvtxjEaiE", "D"],
        # ["Mưa chiều kỷ niệm - Khánh Ly", "https://youtu.be/YzTvSDSmACs", "D"],
        # ["Mưa chiều kỷ niệm - Đàm Vĩnh Hưng", "https://youtu.be/YzhkfBqItZs", "B"],
        # ["Tuổi xa người - Tuấn Ngọc", "https://www.youtube.com/watch?v=lsKPch0uLwI", "A"],
        # ["Tuổi xa người - Xuân Phú", "https://youtu.be/vEnVJeZvK7E", "A"],
        # ["Tuổi xa người - Duy Quang", "https://youtu.be/IGu6KNXYod8", "A"],
        # ["Tuổi xa người - Mai Khôi", "https://youtu.be/aR9_6mOtAx8", "F"],
        # ["Tuổi xa người - Trọng Thanh", "https://youtu.be/emJhaCpy8UQ", "A"],
        # ["Bây giờ tháng mấy - Trọng Tấn", "https://www.youtube.com/watch?v=5pBDJ-nAjKw", "C"],
        # ["Bây giờ tháng mấy - Chế Linh", "https://youtu.be/mOVQqmr_PDM", "G"],
        # ["Bây giờ tháng mấy - Đức Huy", "https://youtu.be/jmkU0yECBxI", "A"],
        # ["Bây giờ tháng mấy - Lê Hiếu", "https://youtu.be/V6LtmIBPmjw", "C"],
        # ["Bây giờ tháng mấy - Đàm Vĩnh Hưng", "https://youtu.be/WNY9huNnUSo", "C"],
        # ["Thương nhau ngày mưa - Tuấn Ngọc", "https://www.youtube.com/watch?v=nO0wp4pUMek", "A:min"],
        # ["Thương nhay ngày mưa - Elvis Phương", "https://youtu.be/XS4UEJwWIPU", "A:min"],
        # ["Biết đâu nguồn cội - Khánh Ly", "https://www.youtube.com/watch?v=yQdvFcZ0CsA", "A"],
        # ["Biết đâu nguồn cội - Tuấn Vũ", "https://www.youtube.com/watch?v=qOuD0D8lwFI", "D"],
        # ["Có những niềm riêng - Tuấn Ngọc", "https://www.youtube.com/watch?v=sBriCYfKhg0", "C#"],
        # ["Có những niềm riêng - Thanh Lan", "https://www.youtube.com/watch?v=vlhd7bpY2KQ", "G"],
        # ["Về đây nghe em - Tuấn Ngọc", "https://www.youtube.com/watch?v=hiIWViekNY0", "G"],
        # ["Ai đưa em về - Hương Lan", "https://www.youtube.com/watch?v=DxWh_ZAZmKU", "D"],
        # ["Ai đưa em về - Quỳnh Lan", "https://www.youtube.com/watch?v=IIBdFWw2AQw", "C"],
        # ["Hãy yêu nhau đi - Trần Thu Hà", "https://youtu.be/KjklEsVDat8", "A"],
        # ["Hãy yêu nhau đi - Lê Uyên", "https://youtu.be/JvKBhVUuUsQ", "A"],
        # ["Lâu đài tình ái - Elvis Phương", "https://youtu.be/6ClEVrySkwI", "D"],
        # ["Lâu đài tình ái - Duy Quang", "https://youtu.be/GUeSZc9A-0A", "D"],
        # ["Thao thức vì em - Cẩm Ly", "https://youtu.be/TFSN53ZKGaw", "C#"],
        # ["Thao thức vì em - Ngọc Lan", "https://youtu.be/Mi-NoG_Oq4g", "B"],
        # ["Thao thức vì em - Quang Linh", "https://youtu.be/No794WzALEA", "G"],
        # ["Thao thức vì em - Dương Ngọc Thái", "https://youtu.be/di6sYQ0IuLU", "F"],
        # ["Thao thức vì em - Elvis Phương", "https://youtu.be/o70mUJXW0w4", "F"],
        # ["Thao thức vì em - Bằng Kiều", "https://youtu.be/ya_wbCjD9q4", "A"],
        # ["Thao thức vì em - Hạ Vy", "https://youtu.be/ETNoeKabtw0", "D"].
        # ["Mưa nửa đêm - Duy Khánh", "https://youtu.be/McNK9PKX6TM", "C#:min"],
        # ["Mưa nửa đêm - Bảo Yến", "https://youtu.be/N6w5EGNphsw", "F#:min"],
        # ["Em gái mưa - Mr.Siro", "https://www.youtube.com/watch?v=f3GWjrHFZQU", "G#:min"],
        # ["Tình đơn phương - Hạ Vy", "https://youtu.be/vV-gSrWYbi0", "C#:min"],
        # ["Tình đơn phương - Huy Vũ", "https://youtu.be/svBYgKUGaCo", "F:min"]
        # ["Vùng lá me bay - Dương Hồng Loan", "https://youtu.be/qAZenHd28VM", "F#:min"],
        # ["Vùng lá me bay - Lưu Ánh Loan", "https://youtu.be/XwegaUTUBvg", "F#:min"],
        # ["Vùng lá me bay - Như Quỳnh", "https://youtu.be/z8DLkLnemFY", "F#:min"]
    ]
    process_song_list(songs)
