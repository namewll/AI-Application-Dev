import os
import time
import requests
from dotenv import load_dotenv
from zhipuai import ZhipuAI
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
client = ZhipuAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_prompt(idea,art_style):
    model = "glm-4-flash"
    system_prompt="""你是一名专业医学解剖绘图提示词专家，精通人体各器官结构、解剖规范、医学可视化标准、AIGC绘图提示工程。你的唯一任务：根据用户提供的器官名称、指定艺术风格，生成一段完整、严谨、可直接用于文生图模型（CogView）绘制人体器官解剖示意图的英文正向提示词，附带规避医学错误的反向负面提示词。
严格遵守以下全部规则：
1. 专业解剖约束
    1.1 严格遵循人体真实解剖结构，器官分层、血管、神经、组织位置完全符合医学教科书标准，禁止出现器官错位、形态畸形、解剖常识错误；
    1.2 区分宏观整体结构与微观组织，根据器官自动补充配套结构：如心脏需包含心房心室、主动脉、冠状动脉；肺部需包含肺泡、支气管；大脑需区分大脑皮层、小脑、脑干；
    1.3 描述器官色彩符合人体真实生理配色，血管区分动脉（鲜红色）、静脉（暗紫红色），软组织、骨骼、腺体色彩区分清晰。

2. 绘图风格标准化处理
    2.1 结合用户输入的艺术风格，统一画面构图、光影、线条逻辑；常见风格适配规则：
        - 扁平化教学风：纯色填充、清晰轮廓线、分层拆分、无复杂阴影、适合课本插图；
        - 3D写实解剖风：立体光影、真实人体肌理、柔和景深、半透明分层透视；
        - 线稿手绘风：干净黑色轮廓、淡灰色阴影、白底、无多余装饰；
        - 科普插画风：柔和配色、简约化结构、弱化恐怖写实感，面向医学生入门教学。
    2.2 构图统一为居中主体器官，画面干净无杂物，背景纯白/浅灰，不添加无关人物、风景、装饰元素。

3. 提示词输出格式规范
    输出分为两段，用【正向提示词】、【负面提示词】明确分隔，不要额外解释、不要多余中文：
    【正向提示词】：英文长句，包含：器官完整解剖结构描述+分层细节+光影构图+画质参数（8k, ultra detailed, high definition, medical illustration, sharp outline）+用户指定艺术风格；
    【负面提示词】：英文，规避解剖错误、劣质画面、违规元素，固定包含：malformation, wrong organ structure, deformed blood vessel, blurry, low resolution, ugly, disfigured, human face, landscape, clutter, text, watermark, patient wound, realistic bloody scene。

4. 额外专业要求
    4.1 自动补充医学绘图专业关键词，提升图像精准度，如anatomical cross-section（解剖横截面）、layered tissue（分层组织）、transparent perspective（半透明透视）；
    4.2 禁止生成血腥、创伤、病变病灶内容，仅输出健康正常人体器官标准示意图；
    4.3 若用户输入器官名称模糊，自动默认生成健康成人标准解剖结构，不追问用户；
    4.4 文字简洁紧凑，无冗余修饰，适配文生图模型识别逻辑，关键词权重合理排布。

最终只输出两段提示词，不输出任何对话、说明、解释文字。"""
    user_name=f"我的想法是:{idea},需要的艺术风格是:{art_style}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role":"system",
                    "content":system_prompt
                },{
                    "role":"user",
                    "content":user_name
                }
            ],
            temperature=0.7
        )
        prompt=response.choices[0].message.content
        return prompt
    except Exception as e:
        print(f"文本生成失败:{e}")
        return None

def generate_image(prompt,model="cogview-3-flash"):
    print("正在生成图片，请稍等片刻...")
    try:
        response=client.images.generations(
            model=model,
            prompt=prompt
        )
        image_url=response.data[0].url
        return image_url
    except Exception as e:
        print(f"图片生成失败:{e}")
        return None

def generate_video(image_url,model="CogVideoX-Flash"):
    try:
        response=client.videos.generations(
            model=model,
            image_url=image_url,
            prompt="让画面动起来",
            quality="quality",
            with_audio=True,
            size="1920x1080",
            fps=30
        )
        print(response)
        while True:
            result = client.videos.retrieve_videos_result(id=response.id)
            print(f"视频生成的状态是：{result.task_status}")
            if result.task_status=="SUCCESS":
                print("视频生成成功")
                print(f"视频的URL:{result.video_result}")
                video_url = result.video_result
                break
            elif result.task_status=="FAILED":
                print("视频生成失败")
                video_url = None
                break
            time.sleep(5)
    except Exception as e:
        print(f"视频生成失败:{e}")
        video_url = None
    return video_url

def create_organ(idea,art_style):
    print(f"\n🧠{'='*20}器官视频开始创作{'='*20}")
    prompt=generate_prompt(idea,art_style)
    if prompt==None:
        return
    image_url=generate_image(prompt,model="cogview-3-flash")
    if image_url==None:
        return
    video_url=generate_video(image_url,model="CogVideoX-Flash")
    if video_url==None:
        return
    return video_url

if __name__=="__main__":
    idea_list = [
        "完整成人心脏解剖，展示心房心室、冠状动脉与血液循环通路",
        "双肺完整解剖结构，清晰呈现支气管、肺泡与肺部血管分布",
        "人体肝脏完整解剖示意图，展示肝叶、门静脉、肝动脉与胆管分布结构",
        "肾脏完整解剖结构，清晰呈现肾皮质、肾髓质、肾盂、肾小球及输尿管走向",
        "人脑完整解剖结构，包含大脑皮层、小脑、脑干、脑室以及主要脑神经分布",
        "胃肠道系统解剖全貌，展示胃、十二指肠、小肠、大肠以及肠系膜血管分布",
        "脾脏解剖结构，清晰呈现脾门血管、脾小叶与毗邻脏器位置关系",
        "胰腺解剖示意图，展示胰头、胰体、胰尾以及胰管、周围血管分布",
        "人体胸腔纵隔解剖结构，展示心脏、大血管、气管、食管的位置毗邻关系",
        "膝关节解剖结构，清晰呈现股骨、胫骨、半月板、韧带与关节囊组织"
    ]
    style_list = [
        "医学课本扁平插画，纯色填充、清晰黑轮廓、纯白无阴影背景",
        "极简黑白教学线稿，单线条勾勒器官，浅灰分层底色",
        "3D半透明写实解剖风，柔和立体光影、人体真实肌理、浅灰色渐变背景",
        "科普柔和插画风，低饱和度莫兰迪配色，简约结构弱化血腥写实感，纯白背景",
        "学术期刊矢量医学插图，精细轮廓、分区色块标注、无多余装饰、浅灰背景",
        "Netter经典手绘解剖风格，细腻素描排线、柔和明暗渐变、复古白纸底色",
        "解剖横截面教学示意图，不同组织用区分色块填充、标注分区轮廓、纯白背景",
        "扁平化渐变医学插画，低饱和渐变色块、加粗黑色轮廓、极简干净构图",
        "水墨国风医学解剖插画，淡墨晕染分层、黑色简练轮廓、宣纸浅黄背景",
        "数字化教学立体解剖示意图，半透明多层叠加、高亮血管线条、浅灰纯色背景"
    ]
    with ThreadPoolExecutor(max_workers=5)as executor:
        video_urls=executor.map(create_organ,idea_list,style_list)
    video_urls=list(video_urls)
    if not os.path.exists('organ_videos'):
        os.mkdir('organ_videos')
    for (video_url,idea_name) in zip(video_urls,idea_list):
        print(video_url)
        res = requests.get(video_url[0].url)
        with open(f"organ_videos/{idea_name}.mp4","wb")as f:
            f.write(res.content)
        print(f'{idea_name}创建完成😁😍')


