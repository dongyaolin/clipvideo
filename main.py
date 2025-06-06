import argparse
import asyncio
import json
import os
import time
from functools import wraps
from pathlib import Path
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from funasr import AutoModel
from llmclient import DeepSeekChat
from loggings import logger
from openai import AsyncOpenAI

load_dotenv()
api_key = os.getenv('DASHSCOPE_API_KEY')


# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need


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


def totxt(text, out):
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text)
    return out


def readcontext(path) -> list:
    with open(path, 'r') as f:
        res = f.readlines()
    return res


def paserllmout(llmout):
    import re
    pattern = r'```json(.*?)```'
    matches = re.findall(pattern, llmout, re.DOTALL)
    if matches:
        return matches[0]
    return ''


def list2txt(lst: list[str]):
    res = ''
    for i in lst:
        res += i
    return res


@timeit
def split_context(n: int, context_path: str) -> list[str]:
    context_lst = readcontext(context_path)

    section_lst = []
    temp = ''
    if n == 1:
        for i in context_lst:
            temp += i
        section_lst.append(temp)
        return section_lst
    if len(context_lst) % n == 0:
        section_width = len(context_lst) // n
        for i in range(n):
            section_context = context_lst[int(i * 0.75 * section_width):(i + 1) * section_width]
            section_context = list2txt(section_context)
            section_lst.append(section_context)
        return section_lst
    else:
        section_width = len(context_lst) // n
        for i in range(n):
            section_context = context_lst[int(i * 0.75 * section_width):(i + 1) * section_width]
            section_context = list2txt(section_context)
            section_lst.append(section_context)
        section_context = context_lst[n * section_width:][0]
        section_lst[-1] += section_context
        return section_lst


async def get_temp_text_async(
        api_key: str = api_key,
        messages: List[Dict[str, str]] = None,
        model: str = "deepseek-chat",
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


@timeit
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


def _mkdir(dir) -> bool:
    if not os.path.exists(dir):
        os.makedirs(dir)
    return True


class VedioClip:
    topic = {"自适应": "", "大颗粒度": "1-3个", "中颗粒度": "4-6个", "小颗粒度": "7-12"}

    def __init__(self, vedio_path=None, keli: str = None, datasrc_dir=None, llmoutjson_dir=None, output_dir=None):
        self.video_path = vedio_path
        self.keli = keli
        self.datasrc_dir = datasrc_dir
        self.llmoutjson_dir = llmoutjson_dir
        self.output_dir = output_dir

        current_file_path = os.path.abspath(__file__)
        logger.info("当前脚本文件的绝对路径:", current_file_path)
        if self.video_path is None:
            self.video_path = r"/root/datasrc/xuejiagehomeinside.mp4"
        self.vidio_name = self.video_path.strip('.mp4').split('/')[-1]
        if self.datasrc_dir is None:
            self.datasrc_dir = '/root/datasrc'
        if self.llmoutjson_dir is None:
            self.llmoutjson_dir = '/root/llmoutjson'
        self.audio_name = f"{self.vidio_name}_output.wav"
        self.audio_path = os.path.join(self.datasrc_dir, self.audio_name)
        self.context_name = f'{self.vidio_name}_context.txt'
        self.context_path = os.path.join(self.datasrc_dir, self.context_name)
        self.context_name1 = f'{self.vidio_name}_context1.txt'
        self.context_path1 = os.path.join(self.datasrc_dir, self.context_name1)
        if self.keli is None:
            self.keli = "大颗粒度"
        self.output_llm_dir = llmoutjson_dir
        if self.output_llm_dir is None:
            self.output_llm_dir = "/root/llmoutjson"
        self.clip_vedio_output_name = f'vts_{self.vidio_name}_{self.keli}'
        self.clip_vedio_output_dir = output_dir
        if self.clip_vedio_output_dir is None:
            self.clip_vedio_output_dir = '/root/datadist/'
        self.output_dir = os.path.join(self.clip_vedio_output_dir, self.clip_vedio_output_name)

        _mkdir(self.datasrc_dir)
        _mkdir(self.clip_vedio_output_dir)
        _mkdir(self.output_dir)
        self.split_num = 0
        self.topic_res = os.path.join(self.clip_vedio_output_dir, f'{self.vidio_name}_topic.json')
        self.section_lst = []
        self.topic_obj = None
        self.context = ''
        self.context1 = ''
        self.message = '''[
        {"role": "system", "content": """你是自然语言专家,负责对直播间多人互动会话内容进行综合分析,划分{{keli}}主题并截取话题结束的时间戳.
        注意：
        1.每段内容尽量翔实,内容包含主体和客体,有充足的铺垫和自然的收尾;
        2.内容包括但不局限于个人经历/讲故事/问答/唱歌/朗诵/才艺表演/粉丝互动 等内容;
        3.注意话题时间在30s～5min;
        4.如果长时间没有说话则认为是在表演获进行相关活动,此时保证当前话题结束再进行内容划分;
        5.总结topic字段参考如下格式
            </topic>
            粉丝才艺惊人!唱歌涨粉1万,主播粉丝年龄跨度大,活动超吸睛!
            <topic>
        6.注意输出如下可被正确解析的json,形如：
            <example>
            ```json
            {"topics": [{"topic": "粉丝才艺惊人!唱歌涨粉1万,主播粉丝年龄跨度大,活动超吸睛!","time_stamps": {"时间区间":
            [0.0, 8.98], "原始内容":'近日,某直播平台一位新人主播"音乐小糖"突然走红.令人惊讶的是,出圈的并非主播本人,而是其粉丝群体的惊人表现.',
            "时间区间":[9.26, 58.82], "原始内容":'在最近举办的"粉丝才艺擂台赛"活动中,多位粉丝展现出专业级演唱实力,其中@麦霸阿杰 翻唱的《孤勇者》视频单条播放量突破200万,直接带动主播账号3天内涨粉1.2万.',...,
            }
            },...
            </example>"""},
        {"role": "user", "content": """{{question}}"""}
    ]'''

        self.message1 = '''[
                {"role": "system", "content": """你是自然语言专家,负责对话题和对应时间范围进行检查,合并一些相关话题,使得话题之间的差异更大化.输出可以为json解析的对象.
                    <example>
                    ```json
                {
            "topics": [
                {
                "topic": "AI模型竞争",
                "time_stamps": {"时间区间":
                    [0.0, 8.98], "原始内容":'Dipseek一出,OKAI就坐不住了,0也放出了最新的O3 mini模型,各项能力测评直接碾压Dipseek而已重回王者宝座,',
                    "时间区间":[9.26, 58.82], "原始内容":'结果没几天,马斯克再放大招Galaxy 3横幅出世的综合实力再次吊打Dipseek.'
                }
                },...
                </example>"""},
                {"role": "user", "content": """{{question}}"""}
            ]'''
        self._asr()
        self._split_context()

    @timeit
    def _asr(self):
        model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                          vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                          punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                          spk_model="cam++", spk_model_revision="v2.0.2",
                          disable_update=True
                          )
        # 音频提取
        from ffmpegapi import extract_audio_from_video
        extract_audio_from_video(self.video_path, self.audio_path)
        logger.info("Audio successfully extracted")
        logger.info("Start voice recognition")
        res = model.generate(input=self.audio_path,
                             batch_size_s=300,
                             hotword='雪茄哥')

        timestamp_content_speaker = []
        timestamp_speaker = []
        timestamp_content = []

        for i in res[0]['sentence_info']:
            item = {"timestamp": [i["start"], i["end"]], "speck_content": i["text"], "speaker": i["spk"]}
            timestamp_content_speaker.append(item)
            item1 = ([i["start"], i["end"]], i["spk"])
            item2 = ({(i["start"], i["end"]): i["text"]})
            from ffmpegapi import to_hour_minutes_seconds
            item3 = f'{to_hour_minutes_seconds(i["start"] / 1000)}-{to_hour_minutes_seconds(i["end"] / 1000)}s:人员{i["spk"]}说:{i["text"]}\n'
            self.context += item3
            # 保存另一份context输入大模型,因为测试发现s为单位描述时间生成的时间区间划分更合理
            item4 = f'{i["start"] / 1000}-{i["end"] / 1000}s:人员{i["spk"]}说:{i["text"]}\n'
            self.context1 += item4
            timestamp_speaker.append(item1)
            timestamp_content.append(item2)

        # 写入文本使用小时：分钟：秒这个形式,即context
        logger.info("Start voice text writing：{}".format(self.context_path))
        totxt(self.context, self.context_path)
        totxt(self.context1, self.context_path1)
        logger.info("Segment voice text to avoid exceeding the model's context window")

        # 输入大模型的内容选择以s为时间格式来描述事件的形式,即context1
        # 分割片段放入
        self.split_num = len(self.context1) // 11846  # 4
        if self.split_num == 0:
            self.split_num = 1

    @timeit
    def _split_context(self) -> list[str]:
        """context_path: 分割参考文本路径"""
        context_lst = readcontext(self.context_path1)
        temp = ''
        if self.split_num == 1:
            for i in context_lst:
                temp += i
            self.section_lst.append(temp)
        if len(context_lst) % self.split_num == 0:  # 
            section_width = len(context_lst) // self.split_num
            for i in range(self.split_num):
                section_context = context_lst[int(i * 0.75 * section_width):(i + 1) * section_width]
                section_context = list2txt(section_context)
                self.section_lst.append(section_context)
        else:
            section_width = len(context_lst) // self.split_num
            for i in range(self.split_num):
                section_context = context_lst[int(i * 0.75 * section_width):(i + 1) * section_width]
                section_context = list2txt(section_context)
                self.section_lst.append(section_context)
            section_context = context_lst[self.split_num * section_width:][0]
            self.section_lst[-1] += section_context

    def build_message_one(self, question) -> list[dict]:
        keli = VedioClip.topic.get(self.keli)
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

    def build_message_two(self, question) -> list[dict]:
        if self.message1:
            from jinja2 import Template
            template = Template(self.message1)
            output = template.render(question=question)
            try:
                import ast
                output = ast.literal_eval(output)
            except:
                output = eval(output)
            return output

    def _build_topic_obj(self, save=None):
        from read_scence import TopicTimer
        if save is None:
            save = os.path.join(self.output_llm_dir, f'{self.vidio_name}_{self.keli}_topic.json')
        self.topic_obj = TopicTimer(self.output_llm_dir, save=save, vedio_name=self.vidio_name).topics

    @timeit
    async def get_topic_split_res_with_llm_async(self):
        """异步处理所有section并保存LLM输出结果"""
        tasks = []
        for i, section in enumerate(self.section_lst):
            task = self._process_section_async(section, i)
            tasks.append(task)

        # 并发执行所有section处理
        await asyncio.gather(*tasks)

    async def _process_section_async(self, section: Dict[str, Any], index: int):
        """异步处理单个section"""
        try:
            # 第一次请求LLM
            message = self.build_message_one(section)
            res = await get_temp_text_async(messages=message)  # 假设已实现异步版本

            # 可选的第二次请求（取消注释即可启用）
            # message = self.build_message_two(res)
            # res = await get_temp_text_async(messages=message)

            # 解析并保存结果
            json_string = await self._parse_and_save_async(res, index)
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

            # 异步写入文件
            output_path = Path(f'{self.output_llm_dir}/{self.vidio_name}_{self.keli}_{index}_llmout.json')
            await self._async_write_file(output_path, json_string)
            return json_string

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"An error occurred while saving the result.: {str(e)}")
            return ""

    async def _async_write_file(self, path: Path, content: str):
        """异步写入文件"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,  # 使用默认线程池
            lambda: path.write_text(content, encoding='utf-8')
        )

    def get_topic_split_res_with_llm(self):
        # 分割部分后依次输入上下文
        for i, section in enumerate(self.section_lst):
            message = self.build_message_one(section)
            res = get_tamp_text(message)

            # twice ask
            # message = self.build_message_two(res)
            # res = get_tamp_text(message)
            try:
                second_res_json_string = paserllmout(res)
                if second_res_json_string:
                    # try:
                    #     second_res_json_dict = json.loads(second_res_json_string)
                    # except json.JSONDecodeError as e:
                    #     logger.info(f"JSON解析异常: {e}")
                    # logger.info(second_res_json_dict)
                    # second_res_json_string = json.dumps(second_res_json_dict, indent=4, ensure_ascii=False)
                    try:
                        totxt(second_res_json_string,
                              f'{self.output_llm_dir}/{self.vidio_name}_{self.keli}_{i}_llmout.json')
                    except Exception as e:
                        logger.error(f"An error occurred while writing to json: {e}")
            except Exception as e:
                logger.info("The output of the large model in JSON format is abnormal.")

    @timeit
    async def process_video_async(self):
        try:
            await self.get_topic_split_res_with_llm_async()
        except Exception as e:
            logger.critical(f"Video processing failed: {str(e)}")

    @timeit
    def split_with_llm(self):
        self._build_topic_obj()
        logger.info(f"Theme and time range:\n{self.topic_obj}")
        from ffmpegapi import splitvidiowithmultiintervals
        n = splitvidiowithmultiintervals(self.video_path, self.output_dir, self.topic_obj)
        logger.info(f"Total fragments numbers: {n}")


@timeit
def clear_folder(folder_path):
    import shutil
    try:
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            return (False, f"The folder does not exist.: {folder_path}")

        # 检查是否是文件夹
        if not os.path.isdir(folder_path):
            return (False, f"The path is not a folder.: {folder_path}")

        # 遍历并删除所有文件和子文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                return (False, f'Delete {file_path} failed: {e}')

        return (True, f"The folder has been successfully emptied: {folder_path}")

    except Exception as e:
        return (False, f"An error occurred while emptying the folder.: {e}")


class BatchProcesor:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *arg, **kwargs):
        self.count += 1
        self.func(*arg, **kwargs)
        return f"第{self.count}个视频处理完成"


@BatchProcesor
async def main(count):
    """---自定义路径---"""
    video_path = r'/home/waas/video_clip/datasrc/sourcevideos/source-C.mp4'  # 待分段视频路径. 
    datasrc_dir = rf'/home/waas/video_clip/datasrc/{count}/'  # 数据源目录 会保存一些分析过程文件
    llmoutjson_dir = rf'/home/waas/video_clip/llmoutjson/{count}/'  # 大模型分析结果保存目录 会保存大模型分段结果
    output_dir = rf'/home/waas/video_clip/output_dir/{count}'  # 分割视频目标目录 会保存分段后的视频片段
    kelidu = r'小颗粒度'
    clear_folder(llmoutjson_dir)
    while True:
        a = _mkdir(datasrc_dir)
        b = _mkdir(llmoutjson_dir)
        c = _mkdir(output_dir)
        if a and b and c:
            break
    logger.info("初始化路径成功")

    parser = argparse.ArgumentParser(description='视频主题分段程序')

    # 提供五个可选参数
    parser.add_argument('--keli', type=str, default=kelidu,
                        choices=["自适应", "大颗粒度", "中颗粒度", "小颗粒度"],
                        help='指定颗粒度,可选值为大颗粒度、中颗粒度、小颗粒度,默认为自适应')

    parser.add_argument('--video_path', type=str, default=video_path, required=False,
                        help='输入视频路径')
    parser.add_argument('--datasrc_dir', type=str, default=datasrc_dir, required=False,
                        help='音频和字幕文件保存目录')
    parser.add_argument('--llmoutjson_dir', type=str, default=llmoutjson_dir, required=False,
                        help='大模型处理中间文件路径')
    parser.add_argument('--output_dir', type=str, default=output_dir, required=False,
                        help='切片输出路径')

    args = parser.parse_args()
    s = time.time()
    vc = VedioClip(keli=args.keli, vedio_path=args.video_path, datasrc_dir=args.datasrc_dir,
                   llmoutjson_dir=args.llmoutjson_dir, output_dir=args.output_dir)
    try:
        # vc.get_topic_split_res_with_llm()
        await vc.process_video_async()
    except Exception as e:
        logger.info(f"LLM failed to obtain segmentation information:{e}")
    vc.split_with_llm()
    e = time.time()
    logger.info(f"Finished with {e - s:.4f}s")


if __name__ == "__main__":
    asyncio.run(main(1))
