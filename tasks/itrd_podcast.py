import json
import os
import time
import requests
from .podcast_gen import PodcastGen

class PodcastGenVoiceITRD(PodcastGen):
    """
    PodcastGenVoiceITRD is a specialized version of PodcastGen for generating podcast voice scripts.
    It inherits from PodcastGen and is tailored for the ITRD project.

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.project = "ITRD"
        self.voice_style = "ITRD-specific style"
        self.language = "zh-tw"
    
    def generate_speeches(
            self, 
            draft: dict, 
            voice_config: dict = None,
            save_name: str = None
    ) -> str:

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
                )
                audios.append(file_path)

        return audios
    
    def determine_voice_config(self, draft: dict) -> str:
        """
        Determine the appropriate voice ID based on the draft content.
        This is a placeholder for future logic to select voice based on content.
        :param draft: JSON dict in multi-speaker format.
        :return: Default voice ID or determined voice ID.
        """

        prompt = self.speech_generator._load_prompt(
            'PodcastGenVoiceITRD', 
            {"draft": draft}
        )
        return json.loads(self.json_model.invoke(prompt).content)
    
    def generate_speeches(
        self,
        draft: dict,
        voice_config: dict = None,
        save_name: str = None
    ):
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

            voice = voice_config.get(speaker, "peter") if voice_config else "peter"

            print(voice)
            speaker_clean = speaker.replace(" ", "_")
            for timestamp, sentence in lines.items():
                ts_str = timestamp.replace(":", "-")
                filename = f"{base_name}-{ts_str}-{speaker_clean}"

                print(sentence)

                filepath = self.generate_speech(
                    text=sentence,
                    itrd_name=voice,
                    filename=filename,
                )
                audios.append(filepath)

        return audios
    
    def generate_speech(
        self,
        text: str,
        itrd_name: str = None,
        filename: str = None
    ) -> str:
        
        # index-tts port
        host = '210.59.241.250'
        port = 8887
        url = f'http://{host}:{port}/infer'

        headers = {"accept": "*/*"}
        itrd_map = {
            "jimmy": './test_data/jimmy_input.wav',
            "lily": "./test_data/lily_input.wav",
            "luka": "./test_data/luka_input.wav",
            "peter": "./test_data/peter_input.wav",
            "wilson": "./test_data/wilson.wav"
        }

        with open(itrd_map[itrd_name], "rb") as f:
            print(itrd_map[itrd_name])
            files = {"prompt_wav": f}
            data = {"text": text}

            print(files)

            response = requests.post(url, headers=headers, data=data, files=files)

        if response.status_code == 200:
            save_dir = "downloads"
            os.makedirs(save_dir, exist_ok=True)

            output_path = os.path.join(save_dir, f"{filename}.wav")
            with open(output_path, "wb") as f:
                f.write(response.content)

            print(f"✅ 音檔儲存成功：{output_path}")
            return output_path
        else:
            print(f"❌ 音檔請求失敗：{response.status_code}")
            return None

    def run(
            self,
            ref_content: str, 
            title: str,
        ) -> str:

        draft = self.drafting(ref_content=ref_content, title=title, save=True)
        print("=====================================")
        print(draft)
        print("=====================================")
        voice_config = self.determine_voice_config(draft)
        print(voice_config)
        print("=====================================")

        audio_paths = self.generate_speeches(
            draft=draft,
            voice_config=voice_config,
            save_name=title
        )
        merged_audio = self.merge_audios(audio_paths, output_name=f"{title}_merged_itrd.wav")
        subtitle_path = self.generate_subtitles(draft, save_name=title)

        return {
            "draft": f"{self.speech_buffer_path}/{title}/itrd/draft.json",
            "audios": audio_paths,
            "merged_audio": merged_audio,
            "subtitles": subtitle_path
        }