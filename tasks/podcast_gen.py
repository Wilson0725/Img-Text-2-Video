import json
from collections import defaultdict
from datetime import datetime, timedelta
import os
import time
from pydub import AudioSegment
from typing import Dict, OrderedDict as OrderedDictType, Tuple
import re

from piplines import KokoroSpeechGenerator
from utils import RemoteModelLoader
from collections import OrderedDict

class PodcastGen:
    def __init__(
        self,
        speech_buffer_path: str,
        lang_code: str = 'a',
        style: str=None,
        pacing: str=None,
        hosts: str=None,
        structure: str=None,
        audience: str=None,
        frequency: str=None,
        duration: str=None,
        vision: str=None,
        language: str=None,
        repo_id: str = 'hexgrad/Kokoro-82M'
    ):
        """
        :param lang_code: 
        'a' - English         e.g., am_puck, af_heart
        'z' - Chinese         e.g., zm_yunxi, zf_xiaoxiao
        'j' - Japanese        e.g., jf_alpha, jm_kumo
        'f' - French          e.g., ff_siwis
        'b' - British English e.g., bm_fable, bf_alice

        :parameters:
        title (str): Podcast title
        topic (str): Main theme or topic
        style (str): Tone or speaking style
        pacing (str): Pacing of the episode
        hosts (str): Host and guest setup
        structure (str): Episode content structure
        audience (str): Target audience
        frequency (str): Publishing frequency
        duration (str): Expected episode duration
        vision (str): Long-term vision or positioning
        language (str): Output language
        """
        self.speech_buffer_path = speech_buffer_path
        self.speech_generator = KokoroSpeechGenerator(
            speech_buffer_path=self.speech_buffer_path,
            repo_id=repo_id,
            lang_code=lang_code
        )
        self.styles = {
            "style": style,
            "pacing": pacing,
            "hosts": hosts,
            "structure": structure,
            "audience": audience,
            "frequency": frequency,
            "duration": duration,
            "vision": vision,
            "language": language
        }
        self.json_model = RemoteModelLoader.load_openai_json_model(
            model_name="gpt-4o",
            temperature=0
        )

    def drafting(
            self,
            ref_content: str,
            title: str,
            save: bool = False,
    ) -> str:

        prompt_variables = {
            "style": self.styles["style"],
            "pacing": self.styles["pacing"],
            "hosts": self.styles["hosts"],
            "structure": self.styles["structure"],
            "audience": self.styles["audience"],
            "frequency": self.styles["frequency"],
            "duration": self.styles["duration"],
            "vision": self.styles["vision"],
            "language": self.styles["language"],
            "ref": ref_content,
            "title": title
        }

        prompt = self.speech_generator._load_prompt(
            'PodcastGen', 
            prompt_variables
        )

        resp = self.json_model.invoke(prompt)

        print("original draft")
        print(resp.content)
                
        parsed_json = self.parse_podcast_json(resp.content)

        # 存檔（可選）
        if save:
            os.makedirs(f"{self.speech_buffer_path}/{title}", exist_ok=True)
            with open(f"{self.speech_buffer_path}/{title}/draft.json", "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, ensure_ascii=False, indent=2)

        return parsed_json
    
    def generate_speeches(
            self, 
            draft: dict, 
            voice_config: dict = None,
            speed: float = 1.2,
            save_name: str = None
        ) -> str:
        """
        Generate speech audio from the draft JSON content (multi-speaker format).
        All files will be saved directly into `self.speech_buffer_path` with filenames:
        <base_name>-<timestamp>-<speaker>.wav

        :param draft: JSON string (or dict) in multi-speaker format.
        :param save_name: Optional base name prefix for the output files.
        :param voice: The voice model to use. Default is 'am_puck'.
            Available Voice IDs and descriptions:

            English (US):
            - af_bella (Hannah): Female, reserved
            - af_nicole (Kaitlyn): Female, soft-spoken and casual
            - af_sarah (Lauren): Female, confident, educator style
            - af_sky (Sierra): Female, calm and composed
            - am_adam (Noah): Male, confident
            - am_michael (Daniel): Male, confident

            English (UK):
            - bf_emma (Chloe): Female
            - bf_isabella (Amelia): Female, calm and steady
            - bm_george (Edward): Male, mature
            - bm_lewis (Oliver): Male, confident

            Chinese (Mandarin):
            - zf_xiaobei (Mei): Female
            - zf_xiaoni (Lian): Female
            - zm_yunjian (Wei): Male

            Japanese:
            - jf_alpha (Sakura): Female
            - jf_gongitsune (Hana): Female
            - jm_kumo (Haruto): Male

            French:
            - ff_siwis (Élodie): Female, young

            Spanish:
            - ef_dora (Lucía): Female
            - em_alex (Mateo): Male

            Italian:
            - if_sara (Giulia): Female
            - im_nicola (Luca): Male

            Portuguese:
            - pf_dora (Camila): Female
            - pm_alex (Thiago): Male

            Hindi:
            - hf_alpha (Ananya): Female
            - hf_beta (Priya): Female
            - hm_omega (Arjun): Male

        :param speed: Speech speed. Default is 1.2.
        :param split_pattern: Regular expression for splitting the text. Default is '\\n+'.
        :return: Returns a dict where keys are slice IDs and values are paths to audio files.
        """

        # Parse draft if needed
        if isinstance(draft, str):
            try:
                draft_dict = json.loads(draft)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format in draft.")
        else:
            draft_dict = draft

        # Define base save path (flat folder)
        os.makedirs(self.speech_buffer_path, exist_ok=True)
        base_name = save_name or time.strftime("%Y%m%d-%H%M%S")

        audios = []
        # Iterate and generate flat audio files
        for speaker, lines in draft_dict.items():

            voice = voice_config.get(speaker, "am_puck") if voice_config else "am_puck"

            print(voice)
            speaker_clean = speaker.replace(" ", "_")
            for timestamp, sentence in lines.items():
                ts_str = timestamp.replace(":", "-")
                filename = f"{base_name}-{ts_str}-{speaker_clean}"
                file_path = os.path.join(self.speech_buffer_path, f'{filename}.wav')

                print(sentence)

                self.speech_generator.generate(
                    prompt=sentence,
                    save_name=filename,
                    voice=voice,
                    speed=speed
                )
                audios.append(file_path)

        return audios

    def merge_audios(self, audio_paths: list, output_name: str = None) -> str:
        """
        Merge a list of audio files into a single .wav file.

        :param audio_paths: List of .wav file paths to be merged.
        :param output_name: Optional output file name (without path).
        :return: Path to the merged .wav file.
        """
        combined = AudioSegment.empty()
        for path in sorted(audio_paths):
            audio = AudioSegment.from_wav(path)
            combined += audio

        output_file = output_name or f"{time.strftime('%Y%m%d-%H%M%S')}_merged.wav"
        output_path = os.path.join(self.speech_buffer_path, output_file)
        combined.export(output_path, format="wav")
        return output_path
    
    def generate_subtitles(self, draft: dict, save_name: str = "subtitle") -> str:
        """
        Generate SRT subtitles from the podcast draft.
        :param draft: JSON dict in multi-speaker format.
        :param save_name: Base name for saving the .srt file.
        :return: Path to the .srt file.
        """
        def to_srt_time(ts):
            h, m, s = map(int, ts.split(":"))
            return f"{h:02}:{m:02}:{s:02},000"

        srt_lines = []
        index = 1
        for speaker, lines in draft.items():
            for timestamp, sentence in sorted(lines.items()):
                start = to_srt_time(timestamp)
                # Simple guess: each line lasts 4 seconds
                end_time = time.strftime("%H:%M:%S", time.gmtime(time.mktime(time.strptime(timestamp, "%H:%M:%S")) + 4))
                end = to_srt_time(end_time)
                srt_lines.append(f"{index}\n{start} --> {end}\n{speaker}: {sentence}\n")
                index += 1

        output_path = os.path.join(self.speech_buffer_path, f"{save_name}.srt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))
        return output_path

    def determine_voice_config(self, draft: dict) -> str:
        """
        Determine the appropriate voice ID based on the draft content.
        This is a placeholder for future logic to select voice based on content.
        :param draft: JSON dict in multi-speaker format.
        :return: Default voice ID or determined voice ID.
        """

        prompt = self.speech_generator._load_prompt(
            'PodcastGenVoiceSelection', 
            {"draft": draft}
        )
        return json.loads(self.json_model.invoke(prompt).content)
    
    def run(
            self,
            ref_content: str, 
            title: str,
            speed: float = 1.2,
        ) -> str:

        """
        Generate speech audio from the draft JSON content (multi-speaker format).
        All files will be saved directly into `self.speech_buffer_path` with filenames:
        <base_name>-<timestamp>-<speaker>.wav

        :param draft: JSON string (or dict) in multi-speaker format.
        :param save_name: Optional base name prefix for the output files.
        :param voice: The voice model to use. Default is 'am_puck'.
            Available Voice IDs and descriptions:

            English (US):
            - af_bella (Hannah): Female, reserved
            - af_nicole (Kaitlyn): Female, soft-spoken and casual
            - af_sarah (Lauren): Female, confident, educator style
            - af_sky (Sierra): Female, calm and composed
            - am_adam (Noah): Male, confident
            - am_michael (Daniel): Male, confident

            English (UK):
            - bf_emma (Chloe): Female
            - bf_isabella (Amelia): Female, calm and steady
            - bm_george (Edward): Male, mature
            - bm_lewis (Oliver): Male, confident

            Chinese (Mandarin):
            - zf_xiaobei (Mei): Female
            - zf_xiaoni (Lian): Female
            - zm_yunjian (Wei): Male

            Japanese:
            - jf_alpha (Sakura): Female
            - jf_gongitsune (Hana): Female
            - jm_kumo (Haruto): Male

            French:
            - ff_siwis (Élodie): Female, young

            Spanish:
            - ef_dora (Lucía): Female
            - em_alex (Mateo): Male

            Italian:
            - if_sara (Giulia): Female
            - im_nicola (Luca): Male

            Portuguese:
            - pf_dora (Camila): Female
            - pm_alex (Thiago): Male

            Hindi:
            - hf_alpha (Ananya): Female
            - hf_beta (Priya): Female
            - hm_omega (Arjun): Male

        :param speed: Speech speed. Default is 1.2.
        :param split_pattern: Regular expression for splitting the text. Default is '\\n+'.
        :return: Returns a dict where keys are slice IDs and values are paths to audio files.
        """

        draft = self.drafting(ref_content=ref_content, title=title, save=True)
        print("=====================================")
        print(draft)
        print("=====================================")
        voice_config = self.determine_voice_config(draft)

        audio_paths = self.generate_speeches(
            draft=draft, 
            save_name=title,
            voice_config=voice_config,
            speed=speed
        )
        merged_audio = self.merge_audios(audio_paths, output_name=f"{title}_merged.wav")
        subtitle_path = self.generate_subtitles(draft, save_name=title)
        return {
            "draft": f"{self.speech_buffer_path}/{title}/draft.json",
            "audios": audio_paths,
            "merged_audio": merged_audio,
            "subtitles": subtitle_path
        }
    
    def parse_podcast_json(self, json_like_str: str, as_str: bool = False):
        """
        將模型輸出的 json-like 字串（可能含重複的 "Speaker X" 區塊）正規化為
        合法 JSON 的 B 類型結構：
        {
          "Speaker A": { "hh:mm:ss": "...", ... },
          "Speaker B": { "hh:mm:ss": "...", ... },
          ...
        }

        Args:
            json_like_str (str): 可能是 A 或 B 類型的 json-like 字串
            as_str (bool): True 則回傳 JSON 字串；False（預設）回傳 Python dict

        Returns:
            dict 或 str: 正規化後的結構（dict）或 JSON 字串
        """

        SPEAKER_BLOCK_RE = re.compile(r'"(Speaker\s+[A-Za-z0-9]+)"\s*:\s*\{(.*?)\}', re.DOTALL)
        LINE_RE = re.compile(r'"(\d{2}:\d{2}:\d{2})"\s*:\s*"(.*?)"', re.DOTALL)

        merged: Dict[str, "OrderedDictType[str, str]"] = {}

        # 1) 嘗試用 regex 擷取所有講者區塊（允許重複）
        blocks = list(SPEAKER_BLOCK_RE.finditer(json_like_str))

        for m in blocks:
            speaker = m.group(1)  # e.g. "Speaker A"
            body = m.group(2)     # 區塊內文（多組 "time": "text"）

            # 2) 解析區塊內的時間與文字
            lines = LINE_RE.findall(body)

            if speaker not in merged:
                merged[speaker] = OrderedDict()

            dest = merged[speaker]
            # 後者覆蓋前者（若同時間戳重複）
            for ts, text in lines:
                dest[ts] = text

        # 3) 若 regex 抓不到任何講者，可能原本已是合法 JSON（B 類型） → 直接 loads
        if not merged:
            try:
                obj = json.loads(json_like_str)
                # 確保結構為 {"Speaker X": {time: text, ...}, ...}
                # 若已符合，直接回傳；若不是你期待的結構，這裡可再加驗證/轉換
                return json.dumps(obj, ensure_ascii=False, indent=2) if as_str else obj
            except Exception:
                # 仍無法解析時，拋出錯誤
                raise ValueError("Unable to parse input as podcast JSON-like content.")

        # 4) 組裝最終物件（保持講者首次出現順序）
        final_obj = OrderedDict()
        for speaker, od in merged.items():
            final_obj[speaker] = OrderedDict(od)

        return json.dumps(final_obj, ensure_ascii=False, indent=2) if as_str else final_obj
