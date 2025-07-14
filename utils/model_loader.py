import json
import requests
import os
import torch
import requests
from kokoro import KPipeline

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from diffusers import FluxControlNetModel, StableDiffusion3Pipeline
from diffusers.pipelines import FluxControlNetPipeline
from diffusers.utils import load_image

HF_TOKEN = os.getenv('HF_TOKEN')

class ModelLoader:
    def __init__(self):
        pass

class ImgGenerationModelLoader(ModelLoader):
    def __init__(self):
        super().__init__()

class SpeechGenerationModelLoader(ModelLoader):
    def __init__(self):
        super().__init__()

class RemoteModelLoader(ModelLoader):

    @staticmethod
    def load_openai_model(model_name="gpt-4o", temperature=0):
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=False,
            timeout=600,
            verbose=False
        )
    
    @staticmethod
    def load_openai_json_model(model_name="gpt-4o", temperature=0):
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=False,
            timeout=600,
            model_kwargs={'response_format': {"type": "json_object"}},
            verbose=False
        )

    @staticmethod
    def get_perplexity_search(
        key_words: str,
        model = 'llama-3.1-sonar-small-128k-online',
        max_tokens: int = 1000, 
        temperature: float = 0.0,
        top_p: float = 0.9, 
        search_recency_filter: str = "week"
    )->json:

        url = "https://api.perplexity.ai/chat/completions"

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Be precise and concise."
                },
                {
                    "role": "user",
                    "content": key_words
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_citations": True,
            "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": search_recency_filter,
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1
        }

        headers = {
            "Authorization": f'Bearer {os.environ["PERPLEXITY_KEY"]}',
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)
        return response.json() 

class OllamaModelLoader(ModelLoader):
    
    @staticmethod
    def get_ollama_model(model_name: str, temperature=0):
        return ChatOllama(
            model=model_name,
            temperature=temperature
        )

class Text2ImgModelLoader(ImgGenerationModelLoader):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def load_st3_pipline(
        model_path_or_id: str = "stabilityai/stable-diffusion-3.5-large", 
        device_map="balanced"
    ):
        return StableDiffusion3Pipeline.from_pretrained(
            model_path_or_id,
            torch_dtype=torch.float32,
            device_map=device_map
        )

class Img2ImgLoader(ImgGenerationModelLoader):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def load_control_net_pipline(
        controlnet_name: str = "jasperai/Flux.1-dev-Controlnet-Upscaler", 
        pipe_name =  'black-forest-labs/FLUX.1-dev',
        device:str="cuda"
    ):

        controlnet = FluxControlNetModel.from_pretrained(
            controlnet_name,
            torch_dtype=torch.float32,
            use_auth_token=HF_TOKEN,
        )

        pipe = FluxControlNetPipeline.from_pretrained(
            pipe_name,
            controlnet=controlnet,
            torch_dtype=torch.float32,
            use_auth_token=HF_TOKEN,
        ).to(device)

        return pipe

class Text2SpeechLoader(SpeechGenerationModelLoader):

    def __init__(self):
        super().__init__()
    
    @staticmethod 
    def load_kokoro_pipline(
        repo_id: str = 'hexgrad/Kokoro-82M',
        lang_code: str = 'a', 
        device: str = None
    ):
        """
        KokoroService

        :param lang_code: 
                'a' - English         e.g., am_puck, af_heart
                'z' - Chinese         e.g., zm_yunxi, zf_xiaoxiao
                'j' - Japanese        e.g., jf_alpha, jm_kumo
                'f' - French          e.g., ff_siwis
                'b' - British English e.g., bm_fable, bf_alice

        :param buffer: 音訊暫存目錄，預設為 './buffer/audio'
        """        
        return KPipeline(lang_code=lang_code, repo_id=repo_id, device=device)