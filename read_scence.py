import os
from sqlitemodule import TaskManager
from loggings import logger
print("导入模块:{}".format(__name__))
from main import DeepSeekChat,api_key
tm = TaskManager()

def llmandpaser(q)->dict:
    # 创建客户端实例
    deepseek = DeepSeekChat(api_key)
    # 定义对话消息
    messages = [
        {"role": "system",
         "content": """你是格式化专家，负责检查json格式， 如果存在格式错误请纠正。输出可以为json解析的对象。不进行额外说明或解释。"""},
         {"role":"user","content":q}
    ]

    # 发送请求并获取回复
    response = deepseek.chat(messages)
    
    import re
    pattern = r'```json(.*?)```'
    matches = re.findall(pattern, llmout, re.DOTALL)
    
    if matches:
        res = matches[0]
        logger.info(f"对错误格式json进行纠错后返回：{res}")
        return res
    return ''

class TopicTimer:
    def __init__(self, topic_dir=None, save=None, vedio_name=None, id=None):
        
        if id is not None:
            self.id = id
        self.vedio_name = vedio_name
        if vedio_name is None:
            self.vedio_name = 'topic'
        self.topic_dir = topic_dir
        if not self.topic_dir:
            self.topic_dir = "/home/waas/video_clip/llmoutjson"
        self.save = save  # topic.json路径
        if not os.path.exists(self.topic_dir):
            os.makedirs(self.topic_dir)
        self.json_path = os.listdir(self.topic_dir)
        self.topics: dict[str,list[tuple]]= {}
        self._build_topic_timer(self.save)
        

    def _build_topic_timer(self, save=None) -> dict:
        global json_file
        for jsonsrc_name in self.json_path:
            jsonsrc = os.path.join(self.topic_dir, jsonsrc_name)
            if isinstance(jsonsrc, str) and jsonsrc.endswith("json"):
                with open(jsonsrc, 'r', encoding = 'utf-8') as file:
                    import json
                    try:      
                        dic = json.load(file)
                    except:
                        logger.error("解析大模型输出数据错误")
                        tm.update_audio_status(self.id, "processing failed")
                        # max_try = 3
                        # for _ in range(max_try):
                        #     try:
                        #         out = llmandpaser(json_file)
                        #         dic = json.loads(out)
                        #         break
                        #     except Exception as e:
                        #         continue
                    
                    topic_set = set()
                    try:
                        dic = eval(dic)
                        for topic in dic.get("topics"):
                            cur_key = topic.get("topic")
                            timelist = topic.get("time_stamps")
                            start_list = []
                            end_list = []
                            if isinstance(timelist, list):
                                for timestampdic in timelist:
                                    start, end = timestampdic.get("时间区间")
                                    if abs(end-start)>30:
                                        start_list.append(start)
                                        end_list.append(end)
                                        if not start_list:
                                            start = min(start_list)
                                        if not end_list:
                                            end = max(end_list)


                                repeared = cur_key in topic_set
                                # start, end = (to_hour_minutes_seconds(i) for i in (start, end))

                                value = [(start, end)]
                                if not repeared:
                                    self.topics[cur_key] = value
                                    topic_set.add(cur_key)
                                else:
                                    # print(self.topics[cur_key])
                                    self.topics[cur_key] = self.topics[cur_key]+[(start, end)]  # extend原地操作
                            if self.topics is not None:
                                topics_list = sorted(self.topics.items(), key=lambda x: x[1][0][0])
                                self.topics = {i:j for i, j in topics_list}
                            elif isinstance(timelist, dict):
                                start, end = timelist.get("时间区间")
                                if abs(end-start)>30:
                                    repeared = cur_key in topic_set
                                    # start, end = (to_hour_minutes_seconds(i) for i in (start, end))

                                    value = [(start, end)]
                                    if not repeared:
                                        self.topics[cur_key] = value
                                        topic_set.add(cur_key)
                                    else:
                                        # print(self.topics[cur_key])
                                        self.topics[cur_key] = self.topics[cur_key]+[(start, end)]  # extend原地操作
                    except Exception as e:
                        logger.error("解析大模型输出数据错误")
                        tm.update_audio_status(self.id, "processing failed")
                        if self.vedio_name is not None:
                            
                            if isinstance(jsonsrc, str) and jsonsrc.endswith("topic.json"):
                                print("跳过非目标json解析")
                            else:
                                print("构建主题时间区间数据错误：{}".format(e))

        if save is not None and isinstance(save, str):
            import json
            topics = json.dumps(self.topics, indent=4, ensure_ascii=False)
            with open(save, 'w') as f:
                f.write(topics)


if __name__ == "__main__":
    res = TopicTimer("/root/llmoutjson/小颗粒度/75c52b27-648e-4e9b-95ad-dc33333d6490",vedio_name='source-C',id="df0892ac-501e-404a-a68f-af54bfd531ab")
    res1 = [{"topic": i, "timestamp": j} for i, j in res.topics.items()]
    print(res1)
    print(len(res1))