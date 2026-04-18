import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captcha_solver import CaptchaSolver
from main import load_config, sys_path
from page_func import (
    book,
    click_agree,
    click_book,
    go_to_venue,
    judge_exceeds_days_limit,
    login,
)


def build_driver(browser: str, headless: bool):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.firefox.service import Service as FirefoxService

    if browser == "firefox":
        options = FirefoxOptions()
        if headless:
            options.add_argument("--headless")
        driver_path = sys_path("firefox")
        if os.path.exists(driver_path):
            return webdriver.Firefox(service=FirefoxService(executable_path=driver_path), options=options)
        return webdriver.Firefox(options=options)

    if browser == "chrome":
        options = ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        driver_path = sys_path("chrome")
        if os.path.exists(driver_path):
            return webdriver.Chrome(service=ChromeService(executable_path=driver_path), options=options)
        return webdriver.Chrome(options=options)

    raise ValueError(f"Unsupported browser: {browser}")


def save_sample(image_content: bytes, order_words: list[str], output_dir: str) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    image_path = os.path.join(output_dir, f"captcha-{stamp}.png")
    meta_path = os.path.join(output_dir, f"captcha-{stamp}.json")

    with open(image_path, "wb") as f:
        f.write(image_content)

    metadata = {
        "targets": order_words,
        "image": os.path.basename(image_path),
        "source": "live_webpage_capture",
        "captured_at": stamp,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return image_path, meta_path


def capture(config_path: str, browser: str, headless: bool, output_dir: str) -> tuple[str, str]:
    (
        user_name,
        password,
        venue,
        venue_num,
        start_time,
        end_time,
        _wechat_notice,
        _sckey,
        cy_username,
        cy_password,
        cy_soft_id,
        glm_enabled,
        glm_endpoint,
        glm_timeout,
        allow_chaojiying_fallback,
        _auto_campus_card_pay,
    ) = load_config(config_path)

    driver = build_driver(browser, headless)
    try:
        login(driver, user_name, password, retry=0)
        time.sleep(1)
        status, log_venue = go_to_venue(driver, venue)
        if not status:
            raise RuntimeError(log_venue)

        start_list, end_list, delta_days, log_exceeds = judge_exceeds_days_limit(start_time, end_time)
        if not start_list:
            raise RuntimeError(log_exceeds)

        status, log_book, _actual_start, _actual_end, _actual_venue = book(
            driver,
            start_list,
            end_list,
            delta_days,
            venue,
            venue_num,
        )
        if not status:
            raise RuntimeError(log_book)

        click_agree(driver)
        click_book(driver)
        time.sleep(2)

        solver = CaptchaSolver(
            glm_enabled=glm_enabled,
            glm_endpoint=glm_endpoint,
            glm_timeout=glm_timeout,
            allow_chaojiying_fallback=allow_chaojiying_fallback,
            cy_username=cy_username,
            cy_password=cy_password,
            cy_soft_id=cy_soft_id,
        )
        _target_element, order_words, image_content = solver._get_captcha_info(driver)
        return save_sample(image_content, order_words, output_dir)
    finally:
        driver.quit()


def main():
    parser = argparse.ArgumentParser(description="Capture a live PKU venue captcha sample without solving it")
    parser.add_argument("--config", default="config.ini", help="Private runtime config path")
    parser.add_argument("--browser", choices=["firefox", "chrome"], default="firefox")
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    parser.add_argument("--output-dir", default="tests/captcha_samples")
    args = parser.parse_args()

    image_path, meta_path = capture(
        config_path=args.config,
        browser=args.browser,
        headless=not args.headed,
        output_dir=args.output_dir,
    )
    print(f"Saved captcha image: {image_path}")
    print(f"Saved captcha metadata: {meta_path}")


if __name__ == "__main__":
    main()
