try:
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
except ModuleNotFoundError:
    WebDriverWait = None
    EC = None
    By = None
from urllib.parse import quote
import time
import datetime
import warnings
from chaojiying import *
import base64

warnings.filterwarnings('ignore')


VENUE_ALIASES = {
    "五四羽毛球馆": ("五四体育中心", "羽毛球馆"),
    "五四体育中心-羽毛球馆": ("五四体育中心", "羽毛球馆"),
    "邱德拔羽毛球场": ("邱德拔体育馆", "羽毛球场"),
    "邱德拔体育馆-羽毛球场": ("邱德拔体育馆", "羽毛球场"),
}


def venue_parent_and_place(venue):
    venue = venue.strip()
    if venue in VENUE_ALIASES:
        return VENUE_ALIASES[venue]
    if "-" in venue:
        parent, place = venue.split("-", 1)
        return parent.strip(), place.strip()
    return None, venue


def venue_card_xpath(venue):
    parent, place = venue_parent_and_place(venue)
    if parent:
        return (
            f"//dl[.//dt[contains(@title, '{parent}')] "
            f"and .//dd[normalize-space()='{place}']]"
        )
    return f"//*[contains(text(), '{place}')]"


def sports_hall_place_xpath(venue):
    parent, place = venue_parent_and_place(venue)
    if not parent:
        return None
    return (
        f"//div[contains(@class, 'li') and .//h4[contains(normalize-space(), '{parent}')]]"
        f"//*[contains(concat(' ', normalize-space(@class), ' '), ' place-item ') "
        f"and normalize-space()='{place}']"
    )


def booking_venue_kind(venue):
    _parent, place = venue_parent_and_place(venue)
    if "羽毛球馆" in place:
        return "羽毛球馆"
    if "羽毛球场" in place:
        return "羽毛球场"
    return place


def booking_first_slot_hour(venue):
    venue = booking_venue_kind(venue)
    if venue == "羽毛球馆":
        return 6
    if venue == "羽毛球场":
        return 8
    return 0


def booking_slot_page_and_column(venue, start_time):
    first_hour = booking_first_slot_hour(venue)
    start = str(start_time).split()[1]
    start_hour = int(start[:2])
    slot_offset = start_hour - first_hour
    if slot_offset < 0:
        raise ValueError(f"预约开始时间早于场馆开放时间: {start}")
    return slot_offset // 5, slot_offset % 5 + 1


def click_venue_card(driver, venue):
    candidates = [venue_card_xpath(venue)]
    hall_xpath = sports_hall_place_xpath(venue)
    if hall_xpath:
        candidates.append(hall_xpath)

    last_error = None
    for xpath in candidates:
        try:
            element = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            driver.execute_script('arguments[0].scrollIntoView({block: "center"});', element)
            time.sleep(0.3)
            driver.execute_script('arguments[0].click();', element)
            return
        except Exception as exc:
            last_error = exc
    raise last_error


def reset_to_first_window(driver):
    handles = driver.window_handles
    if not handles:
        return
    first = handles[0]
    for handle in handles[1:]:
        try:
            driver.switch_to.window(handle)
            driver.close()
        except Exception:
            pass
    driver.switch_to.window(first)


def login(driver, user_name, password, retry=0):
    if retry == 3:
        return '门户登录失败\n'

    print('门户登录中...')

    appID = 'portal2017'
    iaaaUrl = 'https://iaaa.pku.edu.cn/iaaa/oauth.jsp'
    appName = quote('北京大学校内信息门户新版')
    redirectUrl = 'https://portal.pku.edu.cn/portal2017/ssoLogin.do'
    driver.get('https://portal.pku.edu.cn/portal2017/')
    driver.get(f'{iaaaUrl}?appID={appID}&appName={appName}&redirectUrl={redirectUrl}')
    time.sleep(2)
    WebDriverWait(driver, 10).until_not(
        EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
    WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, 'logon_button')))
    driver.find_element(By.ID, 'user_name').send_keys(user_name)
    WebDriverWait(driver, 10).until_not(
        EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
    time.sleep(0.2)
    driver.find_element(By.ID, 'password').send_keys(password)
    WebDriverWait(driver, 10).until_not(
        EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
    time.sleep(0.2)
    driver.find_element(By.ID, 'logon_button').click()
    try:
        WebDriverWait(driver, 50).until(EC.visibility_of_element_located((By.ID, 'all')))
        print('门户登录成功')
        return '门户登录成功\n'
    except:
        print('Retrying...')
        login(driver, user_name, password, retry + 1)


def go_to_venue(driver, venue, retry=0):
    if retry == 3:
        print("进入预约 %s 界面失败" % venue)
        log_str = "进入预约 %s 界面失败\n" % venue
        return False, log_str

    print("进入预约 %s 界面" % venue)
    log_str = "进入预约 %s 界面\n" % venue

    try:
        driver.switch_to.window(driver.window_handles[0])
        butt_all = driver.find_element(By.ID, 'all')
        driver.execute_script('arguments[0].click();', butt_all)
        WebDriverWait(driver, 10).until_not(
            EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'tag_s_venues')))
        time.sleep(0.5)
        driver.find_element(By.ID, 'tag_s_venues').click()
        while len(driver.window_handles) < 2:
            time.sleep(0.5)
        driver.switch_to.window(driver.window_handles[-1])
        WebDriverWait(driver, 10).until_not(
            EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
        time.sleep(5)
        click_venue_card(driver, venue)
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.XPATH, "//*[contains(text(), '当前位置')]")))
        status = True
        log_str += "进入预约 %s 界面成功\n" % venue
    except Exception as exc:
        print(f"retrying: {exc}")
        reset_to_first_window(driver)
        status, log_str = go_to_venue(driver, venue, retry + 1)
    return status, log_str


def click_agree(driver):
    print("点击同意")
    log_str = "点击同意\n"
    driver.switch_to.window(driver.window_handles[-1])
    WebDriverWait(driver, 10).until_not(
        EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CLASS_NAME, 'ivu-checkbox-wrapper')))
    driver.find_element(By.CLASS_NAME, 'ivu-checkbox-wrapper').click()
    print("点击同意成功\n")
    log_str += "点击同意成功\n"
    return log_str


def judge_exceeds_days_limit(start_time, end_time):
    start_time_list = start_time.split('/')
    end_time_list = end_time.split('/')
    print(start_time_list, end_time_list)
    now = datetime.datetime.today()
    today = datetime.datetime.strptime(str(now)[:10], "%Y-%m-%d")
    time_hour = datetime.datetime.strptime(
        str(now).split()[1][:-7], "%H:%M:%S")
    time_11_55 = datetime.datetime.strptime(
        "11:55:00", "%H:%M:%S")
    # time_11_55 = datetime.datetime.strptime(
    #     str(now).split()[1][:-7], "%H:%M:%S")

    start_time_list_new = []
    end_time_list_new = []
    delta_day_list = []
    log_str = ""
    for k in range(len(start_time_list)):
        start_time = start_time_list[k]
        end_time = end_time_list[k]
        if len(start_time) > 8:
            date = datetime.datetime.strptime(
                start_time.split('-')[0], "%Y%m%d")
            delta_day = (date - today).days
        else:
            delta_day = (int(start_time[0]) + 6 - today.weekday()) % 7
            date = today + datetime.timedelta(days=delta_day)
        print("日期:", str(date).split()[0])

        if delta_day > 3 or (delta_day == 3 and (time_hour < time_11_55)):
            print("只能在当天中午11:55后预约未来3天以内的场馆")
            log_str = "只能在当天中午11:55后预约未来3天以内的场馆\n"
            break
        else:
            start_time_list_new.append(start_time)
            end_time_list_new.append(end_time)
            delta_day_list.append(delta_day)
            print("在预约可预约日期范围内")
            log_str = "在预约可预约日期范围内\n"
    return start_time_list_new, end_time_list_new, delta_day_list, log_str


def book(driver, start_time_list, end_time_list, delta_day_list, venue, venue_num=-1):
    print("查找空闲场地")
    log_str = "查找空闲场地\n"

    def judge_close_to_time_12():
        now = datetime.datetime.today()
        time_hour = datetime.datetime.strptime(
            str(now).split()[1][:-7], "%H:%M:%S")
        time_11_55 = datetime.datetime.strptime(
            "11:55:00", "%H:%M:%S")
        time_12 = datetime.datetime.strptime(
            "12:00:00", "%H:%M:%S")
        if time_hour < time_11_55:
            return 0
        elif time_11_55 < time_hour < time_12:
            return 1
        elif time_hour > time_12:
            return 2

    def judge_in_time_range(start_time, end_time, venue_time_range):
        vt = venue_time_range.split('-')
        vt_start_time = datetime.datetime.strptime(vt[0], "%H:%M")
        vt_end_time = datetime.datetime.strptime(vt[1], "%H:%M")
        # print(vt_start_time)
        # print(start_time)
        if start_time <= vt_start_time and vt_end_time <= end_time:
            return True
        else:
            return False

    def move_to_date(delta_day):
        # for i in range(delta_day):
        WebDriverWait(driver, 10).until_not(
            EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
        driver.find_element(By.XPATH,
                            f'/html/body/div[1]/div/div/div[3]/div[2]/div/div[1]/div[2]/div[1]/div[2]/div[{delta_day + 1}]').click()
        # time.sleep(0.2)

    def next_page():
        # 如果第一页没有，就往后翻，直到不存在下一页
        WebDriverWait(driver, 10).until_not(
            EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
        # time.sleep(0.1)
        driver.find_element(By.XPATH,
                            '//*[@id="scrollTable"]/table/tbody/tr[last()]/td[last()]/div/i').click()

    def page_num(venue, start_time):
        return booking_slot_page_and_column(venue, start_time)

    def click_free(venue_num_click, time_num):
        WebDriverWait(driver, 5).until_not(
            EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
        trs = driver.find_elements(By.TAG_NAME, 'tr')
        trs = (driver.find_elements(By.TAG_NAME, 'tbody'))
        trs = trs[1].find_elements(By.TAG_NAME, 'tr')
        trs_list = []
        for i in range(0, len(trs) - 1):
            trs_list.append(trs[i].find_elements(By.TAG_NAME, 'td'))
        # print(len(trs_list))
        if len(trs_list) == 0:
            return False, -1, 0
        # print(len(trs_list[0]))
        # print(len(trs_list[1]))
        # print(len(trs_list[2]))
        # print(venue_num_click, time_num)
        if venue_num_click != -1:
            class_name = trs_list[venue_num_click - 1][time_num].find_element(By.TAG_NAME,
                                                                              'div').get_attribute("class")
            print(class_name)
            if class_name.split()[2] == 'free':
                trs_list[venue_num_click - 1][time_num].find_element(By.TAG_NAME, 'div').click()
                return True, venue_num_click

        else:
            # 随机点一列free的，防止每次都点第一列
            for i in range(len(trs_list) - 1):
                class_name = trs_list[i][time_num].find_element(By.TAG_NAME,
                                                                'div').get_attribute("class")
                print(class_name)
                if class_name.split()[2] == 'free':
                    venue_num_click = i
                    trs_list[i][time_num].find_element(By.TAG_NAME, 'div').click()
                    return True, venue_num_click

        return False, venue_num_click

    driver.switch_to.window(driver.window_handles[-1])
    WebDriverWait(driver, 10).until_not(
        EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
    # 若接近但是没到12点，停留在此页面
    flag = judge_close_to_time_12()
    if flag == 1:
        while True:
            flag = judge_close_to_time_12()
            if flag == 2:
                break
            else:
                time.sleep(0.5)
        driver.refresh()
        WebDriverWait(driver, 5).until_not(
            EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
    status = False
    for k in range(len(start_time_list)):
        start_time = start_time_list[k]
        end_time = end_time_list[k]
        delta_day = delta_day_list[k]

        if k != 0:
            driver.refresh()
            time.sleep(0.5)

        move_to_date(delta_day)

        start_time = datetime.datetime.strptime(
            start_time.split('-')[1], "%H%M")
        end_time = datetime.datetime.strptime(end_time.split('-')[1], "%H%M")
        interval_minutes = int((end_time - start_time).total_seconds() // 60)
        slot_count = max(1, interval_minutes // 60)
        print("开始时间:%s" % str(start_time).split()[1])
        print("结束时间:%s" % str(end_time).split()[1])
        page, time_num = page_num(venue, start_time)
        print(page, time_num)
        for _ in range(page):
            next_page()
        status, venue_num = click_free(venue_num, time_num)

        if status:
            log_str += "找到空闲场地，场地编号为%d\n" % venue_num
            print("找到空闲场地，场地编号为%d\n" % venue_num)
            if slot_count >= 2:
                # 尝试订第二个连续slot（同一场地，下一个小时）
                next_time_num = time_num + 1
                status2, venue_num2 = click_free(venue_num, next_time_num)
                if status2:
                    # 注意：不同时间槽可能分配到不同场地号，需验证一致性
                    if venue_num2 != venue_num:
                        log_str += f"[警告] 第二小时场地号({venue_num2})与第一小时({venue_num})不一致\n"
                        print(f"[警告] 第二小时场地号({venue_num2})与第一小时({venue_num})不一致\n")
                    else:
                        log_str += f"找到第二个连续空闲场地，场地编号为{venue_num2}\n"
                        print(f"找到第二个连续空闲场地，场地编号为{venue_num2}\n")
                    # 注意：本函数仅返回第一个场地的 venue_num。
                    # 若需同时通知两小时的场地/时间，需修改返回值接口或依赖上述日志。
            now = datetime.datetime.now()
            today = datetime.datetime.strptime(str(now)[:10], "%Y-%m-%d")
            date = today + datetime.timedelta(days=delta_day)
            return status, log_str, str(date)[:10] + str(start_time)[10:], str(date)[:10] + str(end_time)[
                                                                                            10:], venue_num
        else:
            log_str += "没有空余场地\n"
            print("没有空余场地\n")
    return status, log_str, None, None, None


def click_book(driver):
    print("确定预约")
    log_str = "确定预约\n"
    driver.switch_to.window(driver.window_handles[-1])
    WebDriverWait(driver, 10).until_not(
        EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
    # WebDriverWait(driver, 10).until(
    #     EC.visibility_of_element_located(
    #         (By.XPATH, '/html/body/div[1]/div/div/div[3]/div[2]/div/div[2]/div[5]/div/div[2]')))
    driver.find_element(By.XPATH,
                        '/html/body/div[1]/div/div/div[3]/div[2]/div/div[1]/div[2]/div[5]/div[2]/div[1]').click()
    print("确定预约成功")
    log_str += "确定预约成功\n"
    return log_str


def click_submit_order(driver):
    print("提交订单")
    log_str = "提交订单\n"
    driver.switch_to.window(driver.window_handles[-1])
    WebDriverWait(driver, 10).until_not(
        EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
    time.sleep(3)
    driver.find_element(By.XPATH,
                        '/html/body/div[1]/div/div/div[3]/div[2]/div/div[2]/div[2]/div[1]').click()
    # result = EC.alert_is_present()(driver)
    print("提交订单成功")
    log_str += "提交订单成功\n"
    return log_str


def verify(driver, glm_enabled, glm_endpoint, glm_timeout,
           allow_chaojiying_fallback, cy_username, cy_password, cy_soft_id):
    from captcha_solver import solve_captcha
    return solve_captcha(driver, glm_enabled, glm_endpoint, glm_timeout,
                         allow_chaojiying_fallback, cy_username, cy_password,
                         cy_soft_id)


def click_pay(driver):
    print("付款（校园卡）")
    log_str = "付款（校园卡）\n"
    time.sleep(30)
    print("需要用户自行付款")
    log_str += "需要用户自行付款\n"
    return log_str


if __name__ == '__main__':
    pass
