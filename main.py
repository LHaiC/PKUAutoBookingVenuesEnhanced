from configparser import ConfigParser
from os import stat
from time import sleep
import argparse
import datetime
import json
import os
import time

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as Chrome_Options
    from selenium.webdriver.chrome.service import Service as Chrome_Service
    from selenium.webdriver.firefox.options import Options as Firefox_Options
    from selenium.webdriver.firefox.service import Service as Firefox_Service
except ModuleNotFoundError:
    webdriver = None
    Chrome_Options = None
    Chrome_Service = None
    Firefox_Options = None
    Firefox_Service = None
import warnings
import sys
import multiprocessing as mp
from env_check import *
from page_func import *
from notice import *

warnings.filterwarnings('ignore')


def sys_path(browser):
    path = 'driver'
    if browser == "chrome":
        if sys.platform.startswith('win'):
            return os.path.join(path, 'chromedriver.exe')
        elif sys.platform.startswith('linux'):
            return os.path.join(path, 'chromedriver.bin')
        else:
            raise Exception('不支持该系统')
    elif browser == "firefox":
        if sys.platform.startswith('win'):
            return os.path.join(path, 'geckodriver.exe')
        elif sys.platform.startswith('linux'):
            return os.path.join(path, 'geckodriver.bin')
        else:
            raise Exception('不支持该系统')


def ensure_selenium_available():
    if webdriver is None or Chrome_Options is None or Firefox_Options is None:
        raise RuntimeError(
            "Selenium 未安装在当前 Python 环境中，请先运行: pip install selenium"
        )


def firefox_profile_root():
    root = os.path.abspath(os.path.join("models", ".selenium-profiles"))
    os.makedirs(root, exist_ok=True)
    return root


def firefox_binary_location():
    snap_binary = "/snap/firefox/current/usr/lib/firefox/firefox"
    if os.path.exists(snap_binary):
        return snap_binary
    return None


def build_driver(browser, headless=True):
    ensure_selenium_available()

    if browser == "chrome":
        chrome_options = Chrome_Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        driver_path = sys_path("chrome")
        if os.path.exists(driver_path):
            return webdriver.Chrome(
                service=Chrome_Service(executable_path=driver_path),
                options=chrome_options,
            )
        return webdriver.Chrome(options=chrome_options)

    if browser == "firefox":
        firefox_options = Firefox_Options()
        if headless:
            firefox_options.add_argument("--headless")
        binary_location = firefox_binary_location()
        if binary_location:
            firefox_options.binary_location = binary_location
        driver_path = sys_path("firefox")
        service_args = ["--profile-root", firefox_profile_root()]
        if os.path.exists(driver_path):
            return webdriver.Firefox(
                service=Firefox_Service(executable_path=driver_path, service_args=service_args),
                options=firefox_options,
            )
        return webdriver.Firefox(
            service=Firefox_Service(service_args=service_args),
            options=firefox_options,
        )

    raise Exception("不支持此类浏览器")


def load_config(config):
    conf = ConfigParser()
    conf.read(config, encoding='utf8')

    user_name = conf['login']['user_name']
    password = conf['login']['password']
    venue = conf.get('type', 'venue', fallback='')
    venue_num = conf.getint('type', 'venue_num', fallback=-1)
    start_time = conf.get('time', 'start_time', fallback='')
    end_time = conf.get('time', 'end_time', fallback='')
    wechat_notice = conf.getboolean('wechat', 'wechat_notice')
    sckey = conf['wechat']['SCKEY']
    username = conf['chaojiying']['username']
    pass_word = conf['chaojiying']['password']
    soft_id = conf['chaojiying']['soft_id']

    glm_enabled = conf.getboolean('glm_ocr', 'enabled') if conf.has_section('glm_ocr') else False
    glm_endpoint = conf['glm_ocr']['endpoint'] if conf.has_section('glm_ocr') else 'http://localhost:8000'
    glm_timeout = conf.getint('glm_ocr', 'timeout') if conf.has_option('glm_ocr', 'timeout') else 10
    allow_chaojiying_fallback = (
        conf.getboolean('glm_ocr', 'allow_chaojiying_fallback')
        if conf.has_option('glm_ocr', 'allow_chaojiying_fallback')
        else False
    )
    return (
        user_name, password, venue, venue_num, start_time, end_time, wechat_notice,
        sckey, username, pass_word, soft_id, glm_enabled, glm_endpoint, glm_timeout,
        allow_chaojiying_fallback,
    )


def log_status(config, start_time, log_str):
    print("记录日志")
    now = datetime.datetime.now()
    print(now)
    print('%s.log' % config.split('.')[0])
    with open('%s.log' % config.split('.')[0], 'a', encoding='utf-8') as fw:
        fw.write(str(now) + "\n")
        fw.write("%s\n" % str(start_time))
        fw.write(log_str + "\n")
    print("记录日志成功\n")


def parse_wait_until(value):
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.datetime.fromisoformat(value)


def wait_until_datetime(wait_until):
    release_time = parse_wait_until(wait_until)
    if release_time is None:
        return
    if release_time > datetime.datetime.now():
        print(f"等待预约开放至 {release_time.strftime('%Y-%m-%d %H:%M:%S')}")
    while True:
        remaining = (release_time - datetime.datetime.now()).total_seconds()
        if remaining <= 0:
            break
        time.sleep(min(remaining, 0.2))
    print("到达预约开放时间，开始进入预约页面")


def page(
    config,
    browser="chrome",
    wait_until=None,
    venue_override=None,
    venue_num_override=None,
    start_time_override=None,
    end_time_override=None,
):
    (
        user_name, password, venue, venue_num, start_time, end_time, wechat_notice,
        sckey, username, pass_word, soft_id, glm_enabled, glm_endpoint, glm_timeout,
        allow_chaojiying_fallback,
    ) = load_config(config)
    venue = venue_override or venue
    venue_num = int(venue_num_override) if venue_num_override not in (None, "") else venue_num
    start_time = start_time_override or start_time
    end_time = end_time_override or end_time
    if not venue or not start_time or not end_time:
        raise ValueError("缺少预约任务信息：需要 venue、start_time、end_time")

    log_str = ""
    status = True
    start_time_list_new, end_time_list_new, delta_day_list, log_exceeds = judge_exceeds_days_limit(start_time, end_time)
    log_str += log_exceeds
    if len(start_time_list_new) == 0:
        log_status(config, [start_time.split('/'), end_time.split('/')], log_exceeds)
        return False
    driver = build_driver(browser, headless=True)
    print(f'{browser} launched\n')

    if status:
        try:
            sleep(2)
            log_str += login(driver, user_name, password, retry=0)
        except:
            log_str += "登录失败\n"
            status = False
    if status:
        try:
            wait_until_datetime(wait_until)
            sleep(2)
            status, log_venue = go_to_venue(driver, venue)
            log_str += log_venue
        except:
            log_str += "进入预约 %s 界面失败\n" % venue
            status = False
    if status:
        try:
            sleep(2)
            status, log_book, start_time, end_time, venue_num = book(driver, start_time_list_new,
                                                                 end_time_list_new, delta_day_list, venue, venue_num)
            log_str += log_book
        except:
            log_str += "点击预约表格失败\n"
            print("点击预约表格失败\n")
            status = False

    if status:
        try:
            log_str += click_agree(driver)
        except:
            log_str += "点击同意失败\n"
            print("点击同意失败\n")
            status = False
    if status:
        try:
            log_str += click_book(driver)
        except:
            log_str += "确定预约失败\n"
            print("确定预约失败\n")
            status = False
    if status:
        try:
            log_str += verify(
                driver, glm_enabled, glm_endpoint, glm_timeout,
                allow_chaojiying_fallback, username, pass_word, soft_id,
            )
        except Exception as exc:
            log_str += f"安全验证失败: {exc}\n"
            print(f"安全验证失败: {exc}\n")
            status = False
    if status:
        try:
            log_str += click_submit_order(driver)
        except Exception as exc:
            log_str += f"提交订单失败: {exc}\n"
            print(f"提交订单失败: {exc}\n")
            status = False
    if status:
        try:
            log_str += click_pay(driver)
        except:
            log_str += "付款失败\n"
            print("付款失败\n")
            status = False
    if status and wechat_notice:
        try:
            log_str += wechat_notification(user_name,
                                           venue, venue_num, start_time, end_time, sckey)
        except:
            log_str += "微信通知失败\n"
            print("微信通知失败\n")
    time.sleep(10)

    # 写入状态文件供Dashboard读取
    status_data = {
        "status": "success" if status else "failed",
        "last_run": str(datetime.datetime.now()),
        "config": config,
        "venue": locals().get('venue'),
        "venue_num": locals().get('venue_num'),
    }
    try:
        with open('status.json', 'w', encoding='utf-8') as f:
            json.dump(status_data, f, ensure_ascii=False, indent=2)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    except Exception as e:
        print(f"写入status.json失败: {e}")

    driver.quit()
    log_status(config, [start_time_list_new, end_time_list_new], log_str)
    return status


def sequence_run(lst_conf, browser="chrome"):
    print("按序预约")
    for config in lst_conf:
        print("预约 %s" % config)
        page(config, browser)


def multi_run(lst_conf, browser="chrome"):
    parameter_list = []
    for i in range(len(lst_conf)):
        parameter_list.append((lst_conf[i], browser))
    print("并行预约")
    pool = mp.Pool()
    pool.starmap_async(page, parameter_list)
    pool.close()
    pool.join()


def run_cli():
    parser = argparse.ArgumentParser(description="PKU venue booking runner")
    parser.add_argument("--config", default="config.ini", help="config file path")
    parser.add_argument("--browser", default="firefox", choices=["firefox", "chrome"])
    parser.add_argument("--retries", type=int, default=3, help="stop after first success")
    parser.add_argument("--once", action="store_true", help="run one attempt only")
    parser.add_argument("--wait-until", default=None, help="wait until this local datetime after login before entering venue")
    parser.add_argument("--venue", default=None, help="booking venue, overrides config [type] venue")
    parser.add_argument("--venue-num", default=None, help="venue number, overrides config [type] venue_num")
    parser.add_argument("--start-time", default=None, help="booking start_time, overrides config [time] start_time")
    parser.add_argument("--end-time", default=None, help="booking end_time, overrides config [time] end_time")
    args = parser.parse_args()

    # lst_conf = env_check()
    # print(lst_conf)
    # multi_run(lst_conf, browser)
    # sequence_run(lst_conf, browser)
    retries = 1 if args.once else args.retries
    success = False
    for _ in range(retries):
        status_main = page(
            args.config,
            args.browser,
            wait_until=args.wait_until,
            venue_override=args.venue,
            venue_num_override=args.venue_num,
            start_time_override=args.start_time,
            end_time_override=args.end_time,
        )
        if status_main:
            success = True
            break
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(run_cli())
