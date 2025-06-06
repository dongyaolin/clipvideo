
import os
import pickle
from functools import wraps
from typing import List, Dict, Any, Union
import time
import json
from pathlib import Path
from loggings import logger
from funasr import AutoModel
from openai import AsyncOpenAI
from main import api_key
from read_scence import TopicTimer
from fastapi.middleware.cors import CORSMiddleware
from ffmpegapi import to_hour_minutes_seconds
import jieba
from jieba import analyse
from sqlitemodule import TaskManager
import asyncio
import sys
import pika
import copy
import re
tm = TaskManager()

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"The execution time of the method {func.__name__} : {elapsed_time:.4f} s")
        return result
    return wrapper


topic_map = {"自适应": "", "大颗粒度": "1-3个", "中颗粒度": "4-6个", "小颗粒度": "7-12"}   
model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                          vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                          punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                          spk_model="cam++", spk_model_revision="v2.0.2",
                          disable_update=True
                          )

def paserllmout(llmout:str):
    
    logger.info(f"解析前：{llmout}")
    logger.info(f"解析前类型：{type(llmout)}")
    pattern = r'```json(.*?)```'
    matches = re.findall(pattern, llmout, re.DOTALL)
    if matches:
        return matches[0]
    return ''

def get_think(llmout:str):
    match = re.search(r'<think>(.*?)</think>', llmout, re.DOTALL)
    return match.group(1).strip()


def totxt(text, out):
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text)
    return out

def _mkdir(dir) -> bool:
    if not os.path.exists(dir):
        os.makedirs(dir)
    return True
parameters = pika.ConnectionParameters(
    host='localhost',
    heartbeat=5,
)

def connect_to_rabbitmq(parameters):
    while True:
        try:
            connection = pika.BlockingConnection(parameters)
            return connection
        except pika.exceptions.AMQPConnectionError as e:
            time.sleep(5)


def is_connection_open(connection):
    try:
        connection.process_data_events()
        return True
    except pika.exceptions.AMQPError:
        return False



class AudioHander:
    def __init__(self, src):
        self.src = src
        self.name = self.src.split("/")[-1].strip(".mp3")
        self.res = None
        self.subtitle_h = ""
        self.subitle = ""
        self.sentence_info = None
        
        
    def run(self):
        self.res = model.generate(self.src)
        self.sentence_info= self.res[0]["sentence_info"]
        
        spk_content = ""
        spk = [0]
        temp = ""
        start_end = []
        for i in self.sentence_info:
            cur_spk =  i["spk"]
            spk.append(cur_spk)
            if cur_spk != spk[-2]:
                if len(start_end)>0:
                    item3 = f'{start_end[0][0]}-{start_end[-1][0]}s:人员{cur_spk}说:{temp}\n'
                    self.subtitle_h += item3
                    # item4 = f'{start_end[0][1]}s-{start_end[-1][1]}s:人员{cur_spk}说:{temp}\n'
                    # self.subitle += item4 # ms
                start_end = []
                temp = ""
            
            else:
                start_end.append((to_hour_minutes_seconds(i["start"] / 1000), i["start"] / 1000))
                start_end.append((to_hour_minutes_seconds(i["end"] / 1000),i["end"] / 1000))
                temp+=i["text"]
        if temp:
            item = f"{start_end[0][0]}-{start_end[-1][0]}s:人员{spk[-1]}说:{temp}\n"
            self.subtitle_h += item

    
def get_tamp_text(messages=[
    {"role": "system", "content": """你是自然语言专家,负责对自然语言进行综合分析,划分主题并截取话题结束的时间戳.输出可以为json解析的对象.
         如：
         ```json
    {
    "topics": [
    {
      "topic": "AI模型竞争",
      "time_stamps": [
        [0.0, 8.98] ['Dipseek一出,OKAI就坐不住了,0也放出了最新的O3 mini模型,各项能力测评直接碾压Dipseek而已重回王者宝座,'],
        [9.26, 13.82] ['结果没几天,马斯克再放大招Galaxy 3横幅出世的综合实力再次吊打Dipseek.']
      ]
    },..."""},
    {"role": "user", "content": "你好,请介绍一下你自己"}
    ]):
    from llmclient import DeepSeekChat
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('API_KEY_DONG')
    # 创建客户端实例
    deepseek = DeepSeekChat(api_key)

    # 定义对话消息

    # 发送请求并获取回复
    response = deepseek.chat(messages)
    logger.info(response)
    return response



class TextHander:
    def __init__(self, audio_id, text:str, output_llm_dir=None,keli: str = "小颗粒度"):
        self.audio_id = audio_id
        self.keli = keli
        self.text = text
        self.text_lst = self.text.split('\n')
        self.section_lst = []
        self.total_lines_nums = len(self.text)
        self.split_nums = 0
        self._split_context()
        self.output_llm_dir = output_llm_dir
        if self.output_llm_dir is None:
            self.output_llm_dir = f"/root/llmoutjson/{self.keli}"
        if not os.path.exists(_mkdir(self.output_llm_dir)):
            _mkdir(self.output_llm_dir)
        self.message = '''[
        {"role": "system", "content": """你是自然语言专家,负责对直播间多人互动会话内容进行综合分析,划分{{keli}}主题并截取话题结束的时间戳(以秒/s为单位).
        注意：
        1.每段内容尽量翔实,内容包含主体和客体,有充足的铺垫和自然的收尾;
        2.内容包括但不局限于个人经历/讲故事/问答/唱歌/朗诵/才艺表演/粉丝互动 等内容;
        3.注意话题时间在30s～5min;
        4.如果长时间没有说话则认为是在表演获进行相关活动,此时保证当前话题结束再进行内容划分;
        5.注意“time_stamps”字段的值一定是python列表而不是字典;
        6.总结topic字段参考如下格式
            </topic>
            粉丝才艺惊人!唱歌涨粉1万,主播粉丝年龄跨度大,活动超吸睛!
            <topic>
        7.注意输出如下可被正确解析的json,形如：
            <example>
            ```json
            {"topics": [{"topic": "粉丝才艺惊人!唱歌涨粉1万,主播粉丝年龄跨度大,活动超吸睛!","time_stamps": [{"时间区间":
            [0.0, 8.98], "原始内容":'近日,某直播平台一位新人主播"音乐小糖"突然走红.令人惊讶的是,出圈的并非主播本人,而是其粉丝群体的惊人表现.',
            "时间区间":[9.26, 58.82], "原始内容":'在最近举办的"粉丝才艺擂台赛"活动中,多位粉丝展现出专业级演唱实力,其中@麦霸阿杰 翻唱的《孤勇者》视频单条播放量突破200万,直接带动主播账号3天内涨粉1.2万.',...,
            }
            },...]
            </example>"""},
        {"role": "user", "content": """{{question}}"""}
    ]'''


    def _split_context(self):
        self.split_nums = self.total_lines_nums // 6848
        if self.split_nums == 0:  # 小于11848 不分段
            self.section_lst = [self.text]
            return self

        elif self.total_lines_nums % self.split_nums == 0:  # 大于11848进行分段 split_nums > 0
            section_with =  self.total_lines_nums // self.split_nums
            for i in range(self.split_nums):
                section_context = self.text[int(i*0.95)*section_with:(i+1)*section_with]
                self.section_lst.append(section_context)
            return self

        else:
            section_with =  self.total_lines_nums // self.split_nums
            for i in range(self.split_nums):
                section_context = self.text[int(float(i*0.75)*section_with):(i+1)*section_with]
                self.section_lst.append(section_context)
            section_context = self.text[self.split_nums*section_with:][0]
            self.section_lst[-1]+=section_context
            return self
        

    def build_message_one(self, question) -> list[dict]:
        keli = topic_map[self.keli]
        if self.message:
            from jinja2 import Template
            template = Template(self.message)
            output = template.render(keli=keli, question=question)
            try:
                import ast
                output = ast.literal_eval(output)
                # output = eval(output)
            except:
                output = eval(output)
            return output


    async def get_topic_split_res_with_llm_async(self):
        """异步处理所有section并保存LLM输出结果"""
        tasks = []
        # 保存变量
        

        logger.info(f"分割文本：{self.section_lst}")
        for i, section in enumerate(self.section_lst):
            logger.info(f"开始处理第{i}部分")
            task = self._process_section_async(section, i)
            tasks.append(task)

        # 并发执行所有section处理
        await asyncio.gather(*tasks)

    async def get_temp_text_async(
        api_key: str = api_key,
        messages: List[Dict[str, str]] = None,
        model: str = "deepseek-reasoner",
        stream: bool = False,
    ) -> Union[str, dict]:
        """异步版本,参数同同步版"""
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        if not messages:
            messages = [
                {"role": "system", "content": "你擅长提取结构化数据"},
                {"role": "user", "content": ''}
            ]

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
            )

            if stream:
                full_response = ""
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        logger.info(content, end="", flush=True)
                        full_response += content
                return full_response
            else:
                result = response.choices[0].message.content
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return result

        except Exception as e:
            return {"error": str(e)}

    async def _process_section_async(self, section: Dict[str, Any], index: int):
        """异步处理单个section"""
        try:
            # 第一次请求LLM
            logger.info(f"输入:{section}")
            logger.info(f"构建消息")
            message = self.build_message_one(section)
            logger.info(f"构建消息成功：{message}")
            logger.info(f"开始大模型请求处理")
            count = 0
            
            res = await get_temp_text_async(messages=message)
            logger.info(f"大模型返回结果：{res}")

            
            
            # 可选的第二次请求（取消注释即可启用）
            # message = self.build_message_two(res)
            # res = await get_temp_text_async(messages=message)

            # 解析并保存结果
            logger.info(f"开始保存结果")
            json_string = await self._parse_and_save_async(res, index)
            logger.info(f"保存模型返回结果成功，返回json字符串")
            return json_string

        except Exception as e:
            logger.error(f"Processing section {index} Error: {str(e)}")
            return None


    async def _parse_and_save_async(self, llm_output: str, index: int) -> str:
        """异步解析LLM输出并保存到文件"""
        try:
            json_string = paserllmout(llm_output)
            if not json_string:
                return ""
            logger.info(f"待写入内容json_string:{json_string}")
            # 异步写入文件
            llmoutput_json_file_path = os.path.join(self.output_llm_dir, f"{str(index)}.json")
            logger.info(f"保存json文件路径:{llmoutput_json_file_path}")
            await self._async_write_file(llmoutput_json_file_path, json_string)
            logger.info(f"保存json完成:{llmoutput_json_file_path}")
            return json_string

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing faileded: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"An error occurred while saving the result.: {str(e)}")
            return ""
    async def process_video_async(self):
        try:
            logger.info(f"开始切分文本")
            await self.get_topic_split_res_with_llm_async()
            logger.info(f"切分完成")
        except Exception as e:
            logger.critical(f"Video processing faileded: {str(e)}")


    async def _async_write_file(self, path: Path, content: str):
        """异步写入文件"""
        path = Path(path)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,  # 使用默认线程池
            lambda: path.write_text(content, encoding='utf-8')
        )

def write_file(src, content):
    with open(src, "w") as f:
        json.dump(content, f, ensure_ascii=False)
        # json.dump(content, f, ensure_ascii=False)
from test_local_llm import LocalDeepSeekChat
local_ds = LocalDeepSeekChat()
def callback(ch, method, properties, body):
    try:
        audio_id = body.decode('utf-8')
        logger.info("audio_id", audio_id)
        audio_path, status = tm.get_audio_path(audio_id)
        if status !="completed":
            ah = AudioHander(audio_path)
            ah.run()
            try:
                logger.info(f"res2_input:{audio_id, ah.subtitle_h}")
                output_dir = f'/root/llmoutjson/小颗粒度/{audio_id}/'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                logger.info(f"实例化TextHander:")
                th = TextHander(audio_id, ah.subtitle_h, output_dir)
                logger.info(f"实例化TextHander完成")
                for index, section in enumerate(th.section_lst):
                    logger.info(f"开始处理第{index+1}部分：{section}")
                    message = th.build_message_one(section)
                    logger.info(f"构建第{index+1}个消息完成")
                    logger.info(f"思考中...")
                    llm_output = get_tamp_text(message)
                    # llm_output = local_ds.chat(message)
                    logger.info(f"大模型返回：{llm_output}")
                    if "<think>" in llm_output:
                        llm_output = llm_output.strip(f"<think>{get_think(llm_output)}</think>").strip()
                    if "json" in llm_output:
                        llm_output = paserllmout(llm_output)
                    logger.info(f"解析大模型返回：{llm_output}")
                    logger.info(f"待写入内容json_string")
                    llmoutput_json_file_path = os.path.join(output_dir, f"{str(index)}.json")
                    logger.info(f"保存json文件路径:{llmoutput_json_file_path}")
                    write_file(llmoutput_json_file_path,llm_output)
                    logger.info(f"保存json完成:{llmoutput_json_file_path}")

                logger.info(f"ah.sentence_info:{ah.sentence_info}")
                tmp = copy.deepcopy(ah.sentence_info)
                def filter_info(res:list) -> list:
                    for i in range(len(res)):
                        t = res[i]
                        t.pop("timestamp")
                filter_info(tmp)
                logger.info(f"过滤后ah.sentence_info:{tmp}")
                logger.info("开始处理大模型返回结果")
                tt = TopicTimer(output_dir,id=audio_id)
                logger.info(f"tt.topics.items():{tt.topics.items()}")
                res2 = [{"topic": i, "timestamp": j} for i, j in tt.topics.items()]
                res = {"subtile":tmp, "topics":res2}
                write_file(f"/root/res/{audio_id}.json", str(res))
                logger.info(f"更新任务状态")
                tm.update_audio_status(audio_id, "completed")
                logger.info("更新任务状态成功")

            except Exception as e:
                print(e)
                tm.update_audio_status(audio_id, "processing failed")
                
        print(audio_id)
    except Exception as e:
        tm.update_audio_status(audio_id, "processing failed")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        tm.update_audio_status(audio_id, "processing failed")
        print(e)


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', heartbeat=0))
    channel = connection.channel()
    channel.basic_consume(queue='task_queue', on_message_callback=callback, auto_ack=True)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)