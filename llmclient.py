from openai import OpenAI
from typing import Optional, Dict, List


class DeepSeekChat:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",model="deepseek-reasoner"):
        """
        初始化 DeepSeekChat 客户端

        参数:
            api_key: DeepSeek API 密钥
            base_url: API 基础 URL (默认为 https://api.deepseek.com)
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url
        self.model = model

    def chat(
            self,
            messages: List[Dict[str, str]],
            stream: bool = False,
            **kwargs
    ) -> str:
        """
        与 DeepSeek Chat 进行交互

        参数:
            messages: 消息列表,格式为 [{"role": "user/system", "content": "消息内容"}, ...]
            model: 使用的模型 (默认为 deepseek-chat)
            stream: 是否使用流式响应 (默认为 False)
            **kwargs: 其他传递给 API 的参数

        返回:
            模型生成的回复内容
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            **kwargs
        )

        if stream:
            # 处理流式响应
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            return full_response
        else:
            # 处理普通响应
            return response.choices[0].message.content


# 使用示例
if __name__ == "__main__":
    # 替换为你的实际 API 密钥
    api_key = "sk-c1d3b6b0c7bf42b8924ae7966e2e69dd"

    # 创建客户端实例
    # deepseek = DeepSeekChat(api_key=, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="deepseek-r1")
    
    
    
    deepseek = DeepSeekChat(api_key="sk-c1d3b6b0c7bf42b8924ae7966e2e69dd")
    # 定义对话消息
    messages = [
        {"role": "system",
         "content": """你是自然语言专家,负责对自然语言进行综合分析,划分主题并截取话题结束的时间戳。输出可以为json解析的对象。"""},
        {"role": "user", "content": "你好,请介绍一下你自己"}
    ]

    # 发送请求并获取回复
    response = deepseek.chat(messages)
    print(response)