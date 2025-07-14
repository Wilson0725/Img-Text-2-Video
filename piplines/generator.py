import os 
import torch
import soundfile as sf

from utils import (
    PromptLoader, RemoteModelLoader, 
    Text2ImgModelLoader, Text2SpeechLoader
)
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from diffusers.utils import load_image

HF_TOKEN = os.getenv('HF_TOKEN')

class Generator:
    def __init__(self):
        self.openai = RemoteModelLoader.load_openai_model()
        self.json_openai = RemoteModelLoader.load_openai_json_model()
        self.prompthub_path = './docs/prompthub.yaml'

    def _load_prompt(self, prompt_name:str, prompt_variable:dict):
        prompt = PromptLoader.get_prompt(
            prompt_path = self.prompthub_path,
            prompt_name = prompt_name,
            prompt_variable = prompt_variable
        )
        return prompt
    
    def save_audio(
        self, 
        save_path: str,
        audio: bytes, 
        sample_rate: int = 24000
    ):
        sf.write(save_path, audio, sample_rate)
        return save_path
    
class ST3ImgGenerator(Generator):

    def __init__(
        self,
        img_buffer_path: str,
        model_path_or_id: str = "stabilityai/stable-diffusion-3.5",
        device_map: str = "balanced"
    ):
        super().__init__()

        self.img_buffer = img_buffer_path
        self.pipe = Text2ImgModelLoader.load_st3_pipline(
            model_path_or_id = model_path_or_id,
            device_map = device_map
        )

    def generate(
        self, 
        prompt: str, 
        n_steps: int, 
        guidance_scale: float, 
        img_name: str, 
        img_type: str = 'png', 
        width: int = 512, 
        height: int = 512
    ) -> str:

        img = self.pipe(
            prompt=prompt,
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        ).images[0]

        save_name = f"{img_name}.{img_type}"
        save_path = os.path.join(self.img_buffer, save_name)
        img.save(save_path)

        return save_path
    
class ControlNetImgGenerator(Generator):

    def __init__(
        self,
        img_buffer_path:str,
        controlnet_name: str = "jasperai/Flux.1-dev-Controlnet-Upscaler", 
        pipe_name =  'black-forest-labs/FLUX.1-dev',
        device:str="cuda"
    ):
        super().__init__()
        self.img_buffer = img_buffer_path

        self.controlnet = FluxControlNetModel.from_pretrained(
            controlnet_name,
            torch_dtype=torch.float32,
            use_auth_token=HF_TOKEN,
        )

        self.pipe = FluxControlNetPipeline.from_pretrained(
            pipe_name,
            controlnet=self.controlnet,
            torch_dtype=torch.float32,
            use_auth_token=HF_TOKEN,
        ).to(device)

    def generate(
        self,
        prompt: str,
        control_image_path: str,
        img_name: str,
        img_type: str = 'png',
        controlnet_conditioning_scale: float=0.6,
        num_inference_steps:int=28,
        guidance_scale: float=3.5,
    ) -> str:
        
        control_image = load_image(control_image_path)
        w, h = control_image.size

        image = self.pipe(
            prompt=prompt, 
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale,
            height=control_image.size[1],
            width=control_image.size[0]
        ).images[0]

        save_name = f"{img_name}.{img_type}"
        save_path = os.path.join(self.img_buffer, save_name)
        image.save(save_path)
        return save_path    

class KokoroSpeechGenerator(Generator):

    def __init__(
        self,
        speech_buffer_path: str,
        repo_id: str = 'hexgrad/Kokoro-82M',
        lang_code: str = 'a'
    ):
        """
        :param lang_code: 
        'a' - English         e.g., am_puck, af_heart
        'z' - Chinese         e.g., zm_yunxi, zf_xiaoxiao
        'j' - Japanese        e.g., jf_alpha, jm_kumo
        'f' - French          e.g., ff_siwis
        'b' - British English e.g., bm_fable, bf_alice
        """

        super().__init__()
        self.speech_buffer = speech_buffer_path
        self.pipe = Text2SpeechLoader.load_kokoro_pipline(
            repo_id=repo_id,
            lang_code=lang_code,
            device="cuda:6"
        ) 

    def generate(
        self, 
        prompt: str,
        save_name: str,
        voice: str = 'am_puck', 
        speed: float = 1.2, 
        split_pattern: str = r'\n+'
    ):
        """
        Convert input text into synthesized speech.

        :param text: The text content to synthesize.
        :param file_name: The name of the audio file to save (without extension).
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
        
        generator = self.pipe(
            text=prompt,
            voice=voice,
            speed=speed,
            split_pattern=split_pattern
        )

        audio_path = os.path.join(self.speech_buffer, f"{save_name}.wav")
        
        for gs, ps, audio in generator:
            audio_path = audio_path
            self.save_audio(audio_path, audio)

        return audio_path
        


