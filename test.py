# %%
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview")

# Enable memory savings
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

prompt = """A financial experts in a modern office space, 
engaged in an intense discussion about Tesla's stock performance and future prospects. 
The scene is captured using a wide-shot camera angle, showcasing the experts' body language 
and the office environment, emphasizing the gravity of the conversation."""

with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
      frames = pipe(prompt, num_frames=150).frames[0]

export_to_video(frames, "mochi.mp4", fps=30)

# %%
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video
import os
from PIL import Image

image1 = Image.open("test2.png").convert("RGB")  # 轉成 RGB 避免模式問題
os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16)
pipe.to("cuda")

prompt = "A man with short gray hair plays a red electric guitar." 
image = load_image(image1)

output = pipe(image=image).frames[0]
export_to_video(output, "CAPY.mp4", quality=10, fps=10)
# %%
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = DiffusionPipeline.from_pretrained("Skywork/SkyReels-A2", torch_dtype=torch.float16, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
pipe.to("cuda")

prompt = "A man with short gray hair plays a red electric guitar."
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guitar-man.png"
)

output = pipe(image=image, prompt=prompt).frames[0]
export_to_video(output, "output.mp4")
# %%
from piplines import ST3ImgGenerator, ControlNetImgGenerator, KokoroSpeechGenerator

gen = KokoroSpeechGenerator(
    speech_buffer_path="audio_buffer",
    repo_id="hexgrad/Kokoro-82M",
    lang_code = 'a'
)

prompt  = 'cyberpunk city, neon lights, futuristic architecture'
save_path = './docs/kokoro.mp3'
audio = gen.generate(
    prompt=prompt,
    save_name='test'
)
print(audio)
# %%
from tasks import PodcastGenVoiceITRD
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
structure = """
主持人peter完整解釋今日的台股變化後, luka 做一次發問
"""
gen = PodcastGenVoiceITRD(
    speech_buffer_path="audio_buffer",
    lang_code='z',
    style="像一位博學的說書人搭配一位好奇又搞笑的聽眾，以輕鬆、口語化的方式探討嚴肅或知識含量高的主題。主講人邏輯清晰、深入淺出，補充很多延伸知識；對談者則會提出一般人會有的疑問，偶爾開些無傷大雅的玩笑。",
    pacing="語速適中，遇到重點會放慢強調，轉場與舉例時節奏自然，能讓聽眾輕鬆吸收又不無聊。",
    hosts="主持人peter完整解釋今日的台股變化後, luka 做一次發問",
    structure=structure,
    audience="對世界充滿好奇的上班族、喜歡知識性內容又不想太學術的人、想用輕鬆方式長知識的聽眾",
    frequency="每週更新一次，適合週末晚上或通勤時收聽",
    duration="10–15 minutes",
    vision="打造一個像朋友說故事的知識型播客節目，讓人一邊笑一邊學，節目內容可當下飯知識也可當深夜陪伴。",
    language="繁體中文",
)

ref = """

tsla 今日新聞

特斯拉今日新聞分析
正面新聞（利多）
AI驅動的美國股票激增

新聞摘要：AI驅動的美國股票激增，導致單一股票槓桿ETF的興起，特斯拉是這些ETF的主要組成部分之一，吸引了投機者尋求放大AI相關的投資。
來源：Reuters
南韓公司在美國投資1500億美元

新聞摘要：南韓公司在美國投資1500億美元，涵蓋電池、芯片等關鍵行業，這可能促進特斯拉在美國的供應鏈和技術合作。
來源：Reuters
負面新聞（利空）
特朗普解雇美聯儲理事

新聞摘要：特朗普解雇美聯儲理事引發市場對央行獨立性的擔憂，可能影響貨幣政策前景，增加市場不確定性。
來源：Reuters
肯塔基州電動車電池廠的工人投票加入工會

新聞摘要：肯塔基州電動車電池廠的工人投票加入工會，可能影響特斯拉等電動車製造商的供應鏈和成本結構。
來源：Associated Press
短期影響分析
短期內，特斯拉的股價可能面臨壓力，主要由於市場對美聯儲獨立性的擔憂以及工會活動的潛在影響。然而，AI相關投資的增長可能為特斯拉提供一定的支撐。特斯拉作為高波動性股票，可能會受到市場波動的影響。

中期影響分析
中期來看，特斯拉在AI技術上的投入和應用可能會帶來顯著的競爭優勢。南韓在美國的投資可能進一步強化特斯拉的供應鏈，提升其在美國市場的競爭力。然而，工會活動可能增加運營成本，對特斯拉的利潤率構成挑戰。

長期影響分析
長期來看，特斯拉在AI技術和供應鏈上的優勢可能促進其市場地位的提升。隨著AI技術的進一步發展和供應鏈的強化，特斯拉有望在未來保持增長。南韓的投資支持可能進一步鞏固特斯拉在全球市場的領先地位。

相關新聞
Futures slip as Trump attack on Fed's Cook shakes investors

日期：2025-08-26
來源：Reuters
URL：點擊這裡
摘要：特朗普解雇美聯儲理事引發市場對央行獨立性的擔憂，可能影響貨幣政策前景。
Nvidia to set tone for booming AI leveraged ETF market

日期：2025-08-27
來源：Reuters
URL：點擊這裡
摘要：AI驅動的美國股票激增，導致單一股票槓桿ETF的興起，特斯拉是這些ETF的主要組成部分之一。
Unions seek broader foothold in the South as workers vote at an EV battery plant in Kentucky

日期：2025-08-27
來源：Associated Press
URL：點擊這裡
摘要：肯塔基州電動車電池廠的工人投票加入工會，可能影響特斯拉等電動車製造商的供應鏈和成本結構。
Factbox-South Korean firms pledge $150 billion in US investments at summit

日期：2025-08-26
來源：Reuters
URL：點擊這裡
摘要：南韓公司在美國投資1500億美元，涵蓋電池、芯片等關鍵行業，這可能促進特斯拉在美國的供應鏈和技術合作。
Alphabet Hits an All-Time High: More Rally Ahead for ETFs?

日期：2025-08-26
來源：Zacks
URL：點擊這裡
摘要：Alphabet的股票因AI和雲端計算的增長達到歷史新高，這可能對特斯拉等科技股產生影響。
Apple’s August Stock Revival Gives Hope to Concerned Investors

日期：2025-08-26
來源：Bloomberg
URL：點擊這裡
摘要：Apple的股票在8月復甦，這可能對特斯拉等科技股的市場情緒產生影響。
Why AMD (AMD) Stock Is Trading Up Today

日期：2025-08-26
來源：StockStory
URL：點擊這裡
摘要：AMD的股票因分析師升級和與IBM的合作而上漲，這可能對特斯拉等科技股產生影響。
China Is Pouring Exports Into Africa Faster Than Anywhere Else

日期：2025-08-26
來源：Bloomberg
URL：點擊這裡
摘要：中國對非洲的出口激增，這可能影響特斯拉在全球市場的競爭力。
GitLab (GTLB) Stock Trades Up, Here Is Why

日期：2025-08-27
來源：StockStory
URL：點擊這裡
摘要：GitLab的股票因看漲期權交易而上漲，這可能對特斯拉等科技股的市場情緒產生影響。
The Protocol: Bitcoin Mining Faces New Challenges as Power Costs Eat Profit

日期：2025-08-27
來源：CoinDesk
URL：點擊這裡
摘要：比特幣挖礦面臨新的挑戰，這可能對特斯拉等涉及區塊鏈技術的公司產生影響。
總結
特斯拉在今日的新聞中面臨著多重挑戰和機遇。AI技術的發展和南韓在美國的投資為特斯拉提供了潛在的增長動力。然而，市場對美聯儲獨立性的擔憂和工會活動可能對其短期股價造成壓力。長期來看，特斯拉在AI和供應鏈上的優勢可能促進其市場地位的提升。隨著全球市場的變化，特斯拉需要靈活應對，以保持其在電動車和AI技術領域的領先地位。
"""

audios = gen.run(
    ref_content=ref,
    title="P one GPT Daily",
)
print(audios)
# %%
# %%
from tasks import PodcastGenVoiceITRD

gen = PodcastGenVoiceITRD(
    speech_buffer_path="audio_buffer",
    lang_code='z',
    style="像一位博學的說書人搭配一位好奇又搞笑的聽眾，以輕鬆、口語化的方式探討嚴肅或知識含量高的主題。主講人邏輯清晰、深入淺出，補充很多延伸知識；對談者則會提出一般人會有的疑問，偶爾開些無傷大雅的玩笑。",
    pacing="語速適中，遇到重點會放慢強調，轉場與舉例時節奏自然，能讓聽眾輕鬆吸收又不無聊。",
    hosts="主持人Jimmy ：輕浮且不懂技術；來賓 Lily ：成熟女。",
    structure="開場（簡單寒暄、預告今天主題）→ 主題說明（背景、重點脈絡）→ 舉例與延伸 → 小莫提問互動 → 有趣冷知識或補充 → 結尾（總結、留下一個懸念）",
    audience="對世界充滿好奇的上班族、喜歡知識性內容又不想太學術的人、想用輕鬆方式長知識的聽眾",
    frequency="每週更新一次，適合週末晚上或通勤時收聽",
    duration="10–15 minutes",
    vision="打造一個像朋友說故事的知識型播客節目，讓人一邊笑一邊學，節目內容可當下飯知識也可當深夜陪伴。",
    language="繁體中文",
)

test = gen.generate_speech(
      text = "打造一個像朋友說故事的知識型播客節目，讓人一邊笑一邊學，節目內容可當下飯知識也可當深夜陪伴。",
      itrd_name = "lily",
      filename = "jimmy_test_2"
)
print(test)
# %%
