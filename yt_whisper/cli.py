import os
import whisper
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
import argparse
import warnings
import yt_dlp
from .utils import slugify, str2bool, write_srt, write_vtt
import tempfile

from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="video URLs to transcribe")
    parser.add_argument("--model", default="small",
                        choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--format", default="vtt",
                        choices=["vtt", "srt"], help="the subtitle format to output")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="directory to save the outputs")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="Whether to print out the progress and debug messages")
    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="language spoken in the audio, skip to perform language detection")

    parser.add_argument("--break-lines", type=int, default=0, 
                        help="Whether to break lines into a bottom-heavy pyramid shape if line length exceeds N characters. 0 disables line breaking.")
    
    parser.add_argument("--online", "-n", type=str2bool, default=False,
                        help="Whether to use OpenAI API instead of local model. Default is False. If True, make sure OPENAI_API_KEY environment variable is set with your OpenAI API key.")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    subtitles_format: str = args.pop("format")
    use_openai_api: bool = args.pop("online")
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"

    if use_openai_api:
        print("Using OpenAI API.")
        client = OpenAI()
    else:
        print("Using local model.")
        model = whisper.load_model(model_name)

    audios = get_audio(args.pop("video"))
    break_lines = args.pop("break_lines")

    for title, audio_path in audios.items():
        warnings.filterwarnings("ignore")
        if use_openai_api:
            file = open(audio_path, "rb")
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
            file.close()
            result_segments = result.segments
            result_text = result.text
        else:
            result = model.transcribe(audio_path, **args)
            result_segments = result["segments"]
            result_text = result["text"]
        
        print("Text:\n" + result_text)
        warnings.filterwarnings("default")

        if (subtitles_format == 'vtt'):
            vtt_path = os.path.join(output_dir, f"{slugify(title)}.vtt")
            with open(vtt_path, 'w', encoding="utf-8") as vtt:
                write_vtt(result_segments, file=vtt, line_length=break_lines)

            print("Saved VTT to", os.path.abspath(vtt_path))
        else:
            srt_path = os.path.join(output_dir, f"{slugify(title)}.srt")
            with open(srt_path, 'w', encoding="utf-8") as srt:
                write_srt(result_segments, file=srt, line_length=break_lines)

            print("Saved SRT to", os.path.abspath(srt_path))


def get_audio(urls):
    temp_dir = tempfile.gettempdir()

    ydl = yt_dlp.YoutubeDL({
        'quiet': True,
        'verbose': False,
        'format': 'bestaudio',
        "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
        'postprocessors': [{'preferredcodec': 'mp3', 'preferredquality': '192', 'key': 'FFmpegExtractAudio', }],
    })

    paths = {}

    for url in urls:
        result = ydl.extract_info(url, download=True)
        print(
            f"Downloaded video \"{result['title']}\". Generating subtitles..."
        )
        paths[result["title"]] = os.path.join(temp_dir, f"{result['id']}.mp3")

    return paths


if __name__ == '__main__':
    main()
