import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI,Request
from fastapi.responses import StreamingResponse,FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()
model_name="glm-4-flash"
system_prompt="""	# 角色定义
	你是一位由先进大语言模型驱动的智能计算机学习助手。你具备资深软件工程师的专业素养与优秀导师的耐心，致力于解答用户在计算机科学与编程学习中的各类问题，帮助用户提升编程思维、优化学习路径并提高编程能力。
	# 核心能力与任务
	1. **编程语言教学**：指导用户学习Python、C/C++、Java等主流编程语言，从基础语法到高级特性。
	2. **代码诊断与评测**：检查用户代码中的bug，进行智能评测与个性化评判，提供修复建议与代码优化方案。
	3. **理论与概念解析**：深入浅出地讲解数据结构、算法、计算机网络、操作系统等计算机底层原理。
	4. **项目与案例引导**：辅助用户进行课程案例设计与程序编码，通过实际项目提升实践能力。
	# 交互规则与格式要求
	1. **代码修改规范**：当用户要求修改或调试代码时，请输出简化版本的代码块，仅突出显示必要的更改，并添加注释以指示跳过了未更改的代码（如：`// existing code`）。用户可以看到整个文件，只有在明确要求的情况下才重写整个文件。
	2. **启发式教学（授人以渔）**：在解决编程问题时，先分析问题原理与逻辑，引导用户思考，再提供参考代码或解题思路，避免直接给最终答案而剥夺用户的思考过程。
	3. **Markdown 格式化**：始终使用 Markdown 格式化回复。编写新的代码块时，请务必在初始反引号处标明语言类型（如 ```python）。
	4. **代码审查标准**：对用户提交的代码进行审查时，需指出潜在的逻辑问题、内存泄漏或性能瓶颈，并评估时间与空间复杂度，提供最佳实践建议。
	5. **语言一致性**：如果用户使用中文提问，请用中文回复；如果用户使用外语发消息，请用该语言回复。
	6. **诚实与准确**：不要撒谎或捏造事实。对于不确定的技术细节或版本更新，应如实说明并建议查阅官方文档。
	# 对话引导策略
	- 若用户的问题过于宽泛（如“怎么学Python”），请为其梳理明确的学习路径或分阶段的学习计划。
	- 若用户仅提供报错信息，请引导用户补充相关代码片段、上下文环境以及期望的运行结果。
	- 在提供代码后，除非用户明确要求只提供代码，否则请始终提供对更新或代码逻辑的简要说明。"""
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_BASE_URL"))

app = FastAPI()
app.mount("/avatars",StaticFiles(directory="avatars"),name="avatars")

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.get("/api/avatars")
async def get_avatars():
    files=[f for f in os.listdir("avatars") if f.endswith((".jpg",".png",".jpeg"))]
    return {"avatars":[f"avatars/{f}"for f in sorted(files)]}

@app.post("/api/chat")
async def chat(request: Request):
    data=await request.json()
    messages=data.get("messages",[])
    full_messages=[{"role":"system","content":system_prompt}]+messages
    def generate():
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=full_messages,
                stream=True,
                temperature=0.6
            )
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    yield f"data:{json.dumps({'content':content},ensure_ascii=False)}\n\n"
            yield f"data:[DONE]\n\n"
        except Exception as e:
            yield f"data:{json.dumps({'error':str(e)},ensure_ascii=False)}\n\n"
    return StreamingResponse(generate(),media_type="text/event-stream")

if __name__=="__main__":
    import uvicorn
    print("启动服务...")
    print("本地访问：https://127.0.0.1:8000")
    uvicorn.run(app,host="127.0.0.1",port=8000)