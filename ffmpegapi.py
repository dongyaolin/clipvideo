import ffmpeg
import os
from loggings import logger


def extract_audio_from_time_range(input_video, output_audio, start_time, end_time):
    '''从视频中提取特定时间段的音频'''
    try:
        (
            ffmpeg
            .input(input_video, ss=start_time, to=end_time)
            .output(output_audio, vn=None, acodec='libmp3lame')
            .run(overwrite_output=True)
        )
        logger.info(f"特定时间段音频提取成功,保存为 {output_audio}")
    except ffmpeg.Error as e:
        logger.error(f"特定时间段音频提取失败: {e.stderr.decode()}")

def extract_audio_from_video(input_video, outputaudio):
    """从视频中提取音频"""
    try:
        (
            ffmpeg
            .input(input_video)
            .output(outputaudio, ar=16000)
            .overwrite_output()
            .run()
        )
    except ffmpeg.Error as e:
        logger.exception("从视频中提取音频错误：{}".format(e))

    except Exception as e:
        logger.exception("从视频中提取音频错误：{}".format(e))


def extract_audio_from_video_pre(input_video, outputaudio):
    try:
        (
            ffmpeg
            .input(input_video)
            .output(
                outputaudio,
                **{
                    'ar': 16000,          # 采样率16kHz（语音可降至8kHz）
                    'ac': 1,              # 单声道（如果是语音）
                    'b:a': '16k',         # 音频比特率32kbps（可降至16kbps）
                    'acodec': 'libmp3lame',  # 使用opus编码（压缩率最高）wav 不支持
                    # 'acodec': 'pcm_s16le',
                    # 或者使用 'acodec': 'aac', 'profile:a': 'aac_he'  # 高效AAC
                    'compression_level': 10,  # 最大压缩opus有效）
                    'application': 'audio'    # 优化音频编码
                }
            )
            .overwrite_output()
            .run()
        )
    except ffmpeg.Error as e:
        logger.exception(f"音频提取压缩错误：{e}")
    except Exception as e:
        logger.exception(f"音频处理异常：{e}")


def extract_srt_from_video(input_video, srt):
    """提取视频的字幕"""
    try:
        (
        ffmpeg.input(input_video)
        .output(srt, map="0:s:0", c="srt") # map="0:s:1" 提取第二条字幕轨道 map="0:s:2" 提取第三条字幕轨道,以此类推
        .run()
        )
    except ffmpeg.Error as e:
        logger.error(f"extract_srt 提取字幕失败")

        
def ilocvideo(input_video, output_video, start, duration):
    """分割视频开始时间和持续时间"""
    '''start：s'''
    try:
        ffmpeg.input(input_video, ss=start, t=duration).output(output_video).run()
    except ffmpeg.Error as e:
        logger.error("ilocvideo 截取视频片段失败")



def split_video_by_intervals(input_file, output_dir, intervals:tuple|list):
    """
    按照指定的时间区间将视频拆分成多个片段。

    :param input_file: 输入视频文件的路径
    :param output_dir: 输出片段文件的目录
    :param intervals: 时间区间列表,每个区间是一个包含起始时间和结束时间（秒）的元组
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (start_time, end_time) in enumerate(intervals):
        output_file = os.path.join(output_dir, f'segment_{i:03d}.mp4')
        try:
            # 构建 FFmpeg 命令
            (
                ffmpeg
                .input(input_file, ss=start_time, to=end_time)
                .output(output_file, c='copy')
                .run(quiet=True)
            )
            logger.info(f"成功生成片段: {output_file}")
        except ffmpeg.Error as e:
            logger.error(f"生成片段 {output_file} 时出错: {e.stderr.decode()}")


def trim_and_add_text(input_video, output_video, start, duration, text, x, y, fontsize, fontcolor):
    """添加字幕"""
    try:
        (
            ffmpeg
            .input(input_video, ss=start, t=duration)
            .filter('drawtext', 
                    text=text, 
                    x=x, 
                    y=y, 
                    fontsize=fontsize, 
                    fontcolor=fontcolor,
                    # fontfile='arial.ttf'
                    )  # 指定字体文件,确保字体文件存在
            .output(output_video)
            .run()
        )
    except ffmpeg.Error as e:
        logger.error(f"切片添加文字失败 {e.stderr.decode()}")


def concat_videos(video_files: list[str], output_video):
    """拼接视频"""
    try:
        # 创建一个输入流列表
        input_streams = [ffmpeg.input(file) for file in video_files]
        
        # 使用 concat 过滤器合并视频
        merged = ffmpeg.concat(*input_streams, v=1, a=1)  # v=1 表示合并视频流,a=1 表示合并音频流
        
        # 输出到目标文件
        merged.output(output_video).run()
    except ffmpeg.Error as e:
        logger.error(f"合并失败：{e.stderr.decode()}")

    
def mute_video(input_video, output_video):
    """移除视频的音频"""
    # 打开输入视频
    input_stream = ffmpeg.input(input_video)
    
    # 移除音频流,只保留视频流
    video_stream = input_stream.video
    
    # 输出到目标文件,显式移除音频流
    ffmpeg.output(video_stream, output_video, acodec='anull').run()


# def add_water(input_video, output_video, watermark_path, opt):
#     (ffmpeg
#     .input(input_video, ss=start, t=duration)
#     .add_watermark(watermark_path)
#     .output(output_video)
#     .run()
#     )
    

def split_video_by_intervals(input_file, output_dir, intervals:tuple|list):
    """
    按照指定的时间区间将视频拆分成多个片段。

    :param input_file: 输入视频文件的路径
    :param output_dir: 输出片段文件的目录
    :param intervals: 时间区间列表,每个区间是一个包含起始时间和结束时间（秒）的元组
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (start_time, end_time) in enumerate(intervals):
        output_file = os.path.join(output_dir, f'segment_{i:03d}.mp4')
        try:
            # 构建 FFmpeg 命令
            (
                ffmpeg
                .input(input_file, ss=start_time, to=end_time)
                .output(output_file, c='copy')
                .run(quiet=True)
            )
            logger.info(f"成功生成片段: {output_file}")
        except ffmpeg.Error as e:
            logger.error(f"生成片段 {output_file} 时出错: {e.stderr.decode()}")

def splitvideo(input_file, output_file,start_time,end_time):
    try:
            # 构建 FFmpeg 命令
            (
                ffmpeg
                .input(input_file, ss=start_time, to=end_time)
                # .filter('drawtext', 
                #     text=text, 
                #     x='(w-text_w)/2',  # 水平居中
                #     y='(h-text_h)/4',
                #     fontsize=32, 
                #     fontcolor="black",
                #     fontfile='微软正黑体.ttf')
                .output(output_file, **{'c': 'copy'})
                .overwrite_output()
                .run(quiet=True)
            )
            logger.info(f"Successfully generated a fragment: {output_file}")
    except ffmpeg.Error as e:
        logger.error(f"An error has happened while generating {output_file}: {e.stderr.decode()}")

input_ = '''{'演员和粉丝互动': [(0.05, 24.475)], '现场互动和粉丝提问': [(25.15, 44.04)], '个人生活和习惯': [(44.06, 57.74)], '活动预告和嘉宾介绍': [(57.74, 74.12)], '设备问题和现场安排': [(74.64, 87.84)], '粉丝互动和现场氛围': [(87.98, 105.48)], '才艺展示和互动': [(105.48, 126.48)], '粉丝互动和礼物赠送': [(126.62, 141.56)], '个人经历和情感分享': [(141.72, 150.08)], '团队管理和未来计划': [(150.1, 163.46)], '直播互动与感谢': [(3505.42, 3556.15)], '粉丝互动与情感表达': [(3556.45, 3603.97)], '团队介绍与未来计划': [(3605.01, 3668.85)], '颁奖与感谢': [(3669.15, 3738.735)], '诗歌朗诵与表演': [(3740.65, 3878.64)], '离别与情感表达': [(3878.64, 4005.23)], '粉丝互动与拍照': [(4005.63, 4107.35), (4408.935, 4526.67), (5357.53, 5504.59), (5618.52, 5738.94)], '团队感言与感谢': [(4107.49, 4216.87), (4747.43, 4868.26), (5228.91, 5357.53), (5738.94, 5866.945), (5978.54, 6106.14), (6229.21, 6369.619), (6639.78, 6778.645)], '粉丝表演与互动': [(4216.87, 4320.5), (4868.26, 4973.16)], '奖金分配与互动': [(4320.5, 4408.935)], '音乐表演与互动': [(4526.67, 4652.58), (4973.16, 5096.55)], '粉丝互动与感谢': [(4652.58, 4747.43), (5096.55, 5228.91), (6106.14, 6229.21), (6506.09, 6639.78), (6778.645, 6897.93)], '情感表达与互动': [(5504.59, 5618.52)], '情感表达与未来计划': [(5866.945, 5978.54), (6369.619, 6506.09)]}'''
input_ = eval(input_)

def to_hour_minutes_seconds(t:int)-> str:
    try:
        t = int(t)
    except Exception as e:
        logger.exception("输入时间格式错误：{}".format(e))
    assert isinstance(t,int);"type of t must be int"
    
    h = t//3600
    m = (t-h*3600)//60
    s = t%60
    return f"{h:02}:{m:02}:{s:02}"

        





def tosrt(dir_path, title):
    import os
    dist = os.path.join(dir_path, title)
    dist = dist+".srt"
    s = '''
    1
    00:00:00,650 --> 01:00:32,090
    再用力
    '''
    s = s.replace('再用力', title)
    with open(dist, "w") as file:
        file.write(s)
    





def splitvidiowithmultiintervals(input_file: str, output_dir: str, topic_obj:dict[str,list[tuple]]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.listdir(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    # 如果是文件或符号链接,直接删除
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    # 如果是文件夹,使用 shutil.rmtree 递归删除
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.exception(f'Failed to delete {file_path}. Reason: {e}')
    inx = 1
    for index,(key, value) in enumerate(topic_obj.items()):
        length = len(value)
        if length > 1:
            for i, v in enumerate(value):
                start, end = (to_hour_minutes_seconds(i) for i in v)
                start1, end1 = (i for i in v)
                duration = float(end1) - float(start1)
                if duration < 30 or duration >300:
                    continue
                name = f"{inx:02}_{key}_{start}~{end}.mp4"
                output_file = os.path.join(output_dir, name)
                name_pre = name.split('.')[-2]
                inx+=1
                tosrt(output_dir,name_pre)
                splitvideo(input_file, output_file, start, end)
        else:
            start, end = (to_hour_minutes_seconds(i) for i in value[0])
            start1, end1 = (i for i in value[0])
            duration = float(end1) - float(start1)
            if duration < 30 or duration >300:
                continue
            name = f"{inx:02}_{key}_{start}~{end}.mp4"
            output_file = os.path.join(output_dir, name)
            name_pre = name.split('.')[-2]
            tosrt(output_dir, name_pre)
            inx+=1
            splitvideo(input_file, output_file, start, end)
    return inx-1
        


if __name__ == "__main__":
    pass


    # input_file = '/home/waas/video_clip/datasrc/sourcevideos/source-C.mp4'
    # output_file = 'output1.mp4'
    # # splitvideo(input_file, output_file, 1, 2)
    # output_dir = 'vts'
    # from read_scence import TopicTimer
    # topic_obj = TopicTimer(vedio_name='source-A').topics
    # print(f"主题和时间区间：\n{topic_obj}")
    # print("开始分割主题视频")
    # # exit()
    # n = splitvidiowithmultiintervals(input_file, output_dir, topic_obj)
    # logger.info(f"Total fragments numbers: {n}")
    # output_audio = "/root/workspace/paraformer/FunASR/vidiomp4/output1.wav"

    # extract_audio_from_video(input_file, output_audio)





    # 裁剪前十秒
    # input_file = "input.mp4"
    # output_file = "output.srt"
    # # 使用示例
    # input_video = '/root/workspace/checkspot/ffmpegdemo/test_data/in.mp4'
    # output_audio = 'output_audio.mp3'
    # output_video = 'outtrim.mp4'
    # start = 3
    # duration = 1
    # start_time = '00:01:00'
    # end_time = '00:02:00'
    # out_srt = 'out.srt'
    # extract_audio_from_time_range(input_video, output_audio, start_time, end_time)
    # extract_srt_from_video(input_video, out_srt)
    # ilocvideo(input_video, output_video, start, duration)
    # mute_video(input_video, output_video)
    # split_video_by_intervals(input_file, 'aaa.mp4', [start_time,end_time])    


