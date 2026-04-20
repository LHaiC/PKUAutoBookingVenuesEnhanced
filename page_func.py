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
import re

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


def time_column_from_rows(rows, start_time):
    expected_start = str(start_time).split()[1][:5]
    for row in rows:
        for idx, text in enumerate(row):
            match = re.search(r"(\d{2}:\d{2})\s*-\s*\d{2}:\d{2}", text)
            if match and match.group(1) == expected_start:
                return idx
    return None


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
            # scrollIntoView 后保留极短缓冲，避免 headless 模式下偶发点击抖动
            time.sleep(0.05)
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
    # time.sleep(2)  # disabled for perf
    WebDriverWait(driver, 10).until_not(
        EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
    WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, 'logon_button')))
    driver.find_element(By.ID, 'user_name').send_keys(user_name)
    WebDriverWait(driver, 10).until_not(
        EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
    # time.sleep(0.2)  # disabled for perf
    driver.find_element(By.ID, 'password').send_keys(password)
    WebDriverWait(driver, 10).until_not(
        EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
    # time.sleep(0.2)  # disabled for perf
    driver.find_element(By.ID, 'logon_button').click()
    try:
        WebDriverWait(driver, 50).until(EC.visibility_of_element_located((By.ID, 'all')))
        print('门户登录成功')
        return '门户登录成功\n'
    except:
        print('Retrying...')
        return login(driver, user_name, password, retry + 1)


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
        # time.sleep(0.5)  # disabled for perf
        driver.find_element(By.ID, 'tag_s_venues').click()
        WebDriverWait(driver, 10).until(lambda d: len(d.window_handles) >= 2)
        driver.switch_to.window(driver.window_handles[-1])
        WebDriverWait(driver, 10).until_not(
            EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
        # time.sleep(5)  # disabled for perf
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
        # time.sleep(0.2)  # disabled for perf

    def next_page():
        # 如果第一页没有，就往后翻，直到不存在下一页
        WebDriverWait(driver, 10).until_not(
            EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
        # time.sleep(0.1)
        driver.find_element(By.XPATH,
                            '//*[@id="scrollTable"]/table/tbody/tr[last()]/td[last()]/div/i').click()

    def visible_table_rows():
        table = driver.find_element(By.ID, 'scrollTable')
        rows = []
        for row in table.find_elements(By.TAG_NAME, 'tr'):
            cells = row.find_elements(By.TAG_NAME, 'td')
            if cells:
                rows.append([cell.text.strip() for cell in cells])
        return rows

    def time_num(start_time):
        for _ in range(8):
            column = time_column_from_rows(visible_table_rows(), start_time)
            if column is not None:
                return column
            try:
                next_page()
                # time.sleep(0.5)  # disabled for perf
            except Exception:
                break
        raise ValueError(f"页面上找不到预约时间段: {str(start_time).split()[1][:5]}")

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
            return False, -1
        # print(len(trs_list[0]))
        # print(len(trs_list[1]))
        # print(len(trs_list[2]))
        # print(venue_num_click, time_num)
        if venue_num_click != -1:
            if venue_num_click < 1:
                return False, venue_num_click
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
                    venue_num_click = i + 1
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
                time.sleep(0.05)  # 接近放号时提高轮询精度，减少错过窗口的概率
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

        move_to_date(delta_day)

        start_time = datetime.datetime.strptime(
            start_time.split('-')[1], "%H%M")
        end_time = datetime.datetime.strptime(end_time.split('-')[1], "%H%M")
        interval_minutes = int((end_time - start_time).total_seconds() // 60)
        slot_count = max(1, interval_minutes // 60)
        print("开始时间:%s" % str(start_time).split()[1])
        print("结束时间:%s" % str(end_time).split()[1])
        time_num_current = time_num(start_time)
        print("时间列:%d" % time_num_current)
        status, venue_num = click_free(venue_num, time_num_current)

        if status:
            selected_venue_num = venue_num
            log_str += "找到空闲场地，场地编号为%d\n" % selected_venue_num
            print("找到空闲场地，场地编号为%d\n" % selected_venue_num)
            for slot_offset in range(1, slot_count):
                status_next, venue_num_next = click_free(selected_venue_num, time_num_current + slot_offset)
                if not status_next or venue_num_next != selected_venue_num:
                    status = False
                    log_str += "连续%d小时场地不足\n" % slot_count
                    print("连续%d小时场地不足\n" % slot_count)
                    break
                log_str += "找到第%d个连续空闲场地，场地编号为%d\n" % (slot_offset + 1, selected_venue_num)
                print("找到第%d个连续空闲场地，场地编号为%d\n" % (slot_offset + 1, selected_venue_num))
            if not status:
                continue
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


def _element_is_displayed(element):
    try:
        return element.is_displayed()
    except Exception:
        return True


def _element_payment_text(element):
    parts = [getattr(element, "text", "") or ""]
    for attr in ("title", "aria-label", "value"):
        try:
            parts.append(element.get_attribute(attr) or "")
        except Exception:
            pass
    return " ".join(parts).strip()


def _element_area(element):
    size = getattr(element, "size", None) or {}
    rect = getattr(element, "rect", None) or {}
    width = size.get("width") or rect.get("width") or 0
    height = size.get("height") or rect.get("height") or 0
    return width * height


def submit_order_candidates(driver):
    locators = [
        "//*[self::button or self::div or self::a][normalize-space(.)='提交' and not(contains(@class, 'cancel'))]",
        '/html/body/div[1]/div/div/div[3]/div[2]/div/div[2]/div[2]/div[1]',
    ]
    candidates = []
    seen = set()
    for xpath in locators:
        try:
            elements = driver.find_elements(By.XPATH, xpath)
        except Exception:
            elements = []
        for element in elements:
            if id(element) in seen or not _element_is_displayed(element):
                continue
            text = _element_payment_text(element)
            if text and text != "提交":
                continue
            seen.add(id(element))
            candidates.append(element)
    candidates.sort(key=lambda element: _element_area(element) or 10**9)
    return candidates


def click_submit_order(driver, timeout=10, poll_interval=0.05):
    print("提交订单")
    log_str = "提交订单\n"
    driver.switch_to.window(driver.window_handles[-1])
    try:
        WebDriverWait(driver, 10).until_not(
            EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix")))
    except Exception:
        pass
    deadline = time.time() + timeout
    while True:
        candidates = submit_order_candidates(driver)
        if candidates:
            element = candidates[0]
            try:
                driver.execute_script('arguments[0].scrollIntoView({block: "center"});', element)
            except Exception:
                pass
            time.sleep(0.05)  # scroll 后做极短稳定，避免点击丢失
            try:
                element.click()
            except Exception:
                driver.execute_script('arguments[0].click();', element)
            break
        if time.time() >= deadline:
            raise RuntimeError("找不到提交订单按钮")
        time.sleep(poll_interval)  # disabled for perf
    # result = EC.alert_is_present()(driver)
    print("提交订单成功")
    log_str += "提交订单成功\n"
    return log_str


def verify(driver, glm_enabled, glm_endpoint, glm_timeout,
           allow_chaojiying_fallback, cy_username, cy_password, cy_soft_id,
           glm_proxy: str | None = None):
    from captcha_solver import solve_captcha
    return solve_captcha(driver, glm_enabled, glm_endpoint, glm_timeout,
                         allow_chaojiying_fallback, cy_username, cy_password,
                         cy_soft_id, glm_proxy=glm_proxy)


def print_page_visible_text(driver, label="页面可见文本"):
    """打印当前页面所有可见文本内容用于调试"""
    try:
        body = driver.find_element(By.TAG_NAME, "body")
        # 获取所有文本，包括嵌套元素的
        text = driver.execute_script("return document.body.innerText")
        preview = text[:3000] if len(text) > 3000 else text
        print(f"\n=== {label} ===")
        print(preview if preview.strip() else "[空]")
        print("=== END TEXT ===\n")
    except Exception as exc:
        print(f"获取页面文本失败: {exc}")


def screenshot_all_tabs(driver, prefix="tab"):
    """截图所有tab并保存，用于调试"""
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    handles = driver.window_handles
    for i, handle in enumerate(handles):
        driver.switch_to.window(handle)
        path = f"{prefix}-{stamp}-tab{i}.png"
        try:
            driver.save_screenshot(path)
            print(f"已保存 Tab {i} 截图: {path}, URL: {driver.current_url}")
        except Exception as exc:
            print(f"保存 Tab {i} 截图失败: {exc}")


def save_page_html(driver, path="page_debug.html", label="页面HTML"):
    """保存当前页面完整HTML到文件用于调试"""
    try:
        html = driver.page_source
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"已保存页面HTML到: {path}")
    except Exception as exc:
        print(f"保存页面HTML失败: {exc}")


def click_pay(driver):
    print("付款（电子校园卡）")
    log_str = "付款（电子校园卡）\n"

    # 检查是否有待处理的安全验证
    try:
        captcha_hint = driver.find_element(By.CLASS_NAME, "select-word").get_attribute("textContent")
        if captcha_hint and '请依次点击' in captcha_hint:
            print(f"发现待处理验证码: {captcha_hint}")
            # 提取目标字符
            import re
            targets = re.findall(r'[\u4e00-\u9fff]', captcha_hint)
            if targets:
                print(f"目标字符: {targets}")
                # 获取验证码图片并解决
                from captcha_solver import solve_captcha
                try:
                    solve_captcha(driver, True, 'http://localhost:8000', 30, False, '', '', '')
                    print("二次验证码完成")
                    log_str += "二次验证码完成\n"
                except Exception as exc:
                    print(f"二次验证码失败: {exc}")
                    log_str += f"二次验证码失败: {exc}\n"
    except Exception:
        pass

    # 提交订单后等待新窗口出现
    original_handles = driver.window_handles
    original_count = len(original_handles)
    try:
        WebDriverWait(driver, 15).until(
            lambda d: len(d.window_handles) > original_count
        )
    except Exception:
        print(f"未检测到新窗口，当前窗口数: {len(driver.window_handles)}")

    handles = driver.window_handles
    print(f"当前窗口数: {len(handles)}, 切换到: {handles[-1]}")
    driver.switch_to.window(handles[-1])

    # 等待 spinner 消失
    try:
        WebDriverWait(driver, 10).until_not(
            EC.visibility_of_element_located((By.CLASS_NAME, "loading.ivu-spin.ivu-spin-large.ivu-spin-fix"))
        )
    except Exception:
        pass

    # 尝试点击"支付"按钮（通用匹配：含"支付"文本的按钮）
    try:
        pay_button_xpath = "//button[contains(text(),'支付')]"
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, pay_button_xpath)))
        driver.find_element(By.XPATH, pay_button_xpath).click()
        print("已点击支付按钮")
        log_str += "已点击支付按钮\n"
    except Exception as exc:
        print(f"未找到支付按钮，尝试其他方式: {exc}")
        print_page_visible_text(driver, "支付页面内容")
        # 备用：点击任何包含"支付"的可点击元素
        try:
            alt_xpath = "//*[contains(text(),'支付')]"
            elems = driver.find_elements(By.XPATH, alt_xpath)
            for elem in elems:
                if elem.is_displayed() and elem.is_enabled():
                    try:
                        elem.click()
                        print(f"备用点击成功: {elem.text}")
                        log_str += f"备用点击: {elem.text}\n"
                        break
                    except Exception:
                        continue
        except Exception as e2:
            print(f"备用点击也失败: {e2}")
            log_str += f"支付按钮点击失败: {exc}\n"

    log_str += "付款完成\n"
    print("付款完成")
    return log_str


if __name__ == '__main__':
    pass