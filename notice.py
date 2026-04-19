from email.header import Header
from email.mime.text import MIMEText
from urllib.parse import quote
from urllib import request
import json


def wechat_notification(user_name, venue, venue_num, start_time, end_time, sckey):
    content = (
        f"学号：{user_name}\n"
        f"成功预约：{venue}\n"
        f"场地编号：{venue_num}\n"
        f"开始时间：{start_time}\n"
        f"结束时间：{end_time}\n"
        f"请及时去付款！"
    )
    with request.urlopen(
            quote(f'http://www.pushplus.plus/send?token={sckey}&title=成功预约&content={content}',
                  safe='/:?=&')) as response:
        response = json.loads(response.read().decode('utf-8'))
        if response.get('code') == 0 or response.get('msg') == 'success' or '执行成功' in str(response):
            print('微信通知成功')
        else:
            print('error: ' + str(response))
    return "微信通知成功\n"


if __name__ == '__main__':
    wechat_notification('', "羽毛球场测试", "", "", "")
