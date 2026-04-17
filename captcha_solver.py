"""
Captcha Solver with dual-mode support:
1. GLM-OCR (local): Priority mode using local GLM-OCR service
2. Chaojiying (超级鹰): Fallback mode with 3 retries
"""

import base64
import time
import requests


def _strict_int(value):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("expected integer")
    return value


class CaptchaSolver:
    def __init__(self, glm_enabled, glm_endpoint, glm_timeout,
                 cy_username, cy_password, cy_soft_id, allow_chaojiying_fallback=False):
        self.glm_enabled = glm_enabled
        self.glm_endpoint = glm_endpoint.rstrip('/')
        self.glm_timeout = glm_timeout
        self.cy_username = cy_username
        self.cy_password = cy_password
        self.cy_soft_id = cy_soft_id
        self.allow_chaojiying_fallback = allow_chaojiying_fallback

    def _get_captcha_info(self, driver):
        """Extract captcha image base64 and target words from page."""
        from selenium.webdriver.common.by import By

        target_element = driver.find_element(By.XPATH,
            "/html/body/div[1]/div/div/div[3]/div[2]/div/div[1]/div[2]/div[4]/div[3]/div/div[2]/div/div[1]/div/img")
        order_element = driver.find_element(By.XPATH,
            "/html/body/div[1]/div/div/div[3]/div[2]/div/div[1]/div[2]/div[4]/div[3]/div/div[2]/div/div[2]/span")

        image_uri = target_element.get_attribute("src")
        order_str = order_element.text
        # order_words are the 3 characters to click, extracted from text like "请依次点击：北,京,大,学"
        order_words = order_str[-6:-1].split(",")

        # Extract base64 data from data URI
        data_start = image_uri.find(",") + 1
        image_base64 = image_uri[data_start:]
        image_content = base64.b64decode(image_base64)

        return target_element, order_words, image_content

    def _solve_with_glm(self, image_content, order_words):
        """
        Call local GLM-OCR service to get character positions.

        GLM-OCR actual API:
        - Endpoint: POST {endpoint}/glmocr/parse
        - Request: {"images": [base64_data_uri]}
        - Response: JSON with OCR results including character positions

        Returns: [[char, x, y], ...] for matched characters
        """
        # Prepare base64 data URI for GLM-OCR
        image_b64 = base64.b64encode(image_content).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{image_b64}"

        url = f"{self.glm_endpoint}/glmocr/parse"
        payload = {"images": [data_uri], "targets": order_words}

        try:
            response = requests.post(url, json=payload, timeout=self.glm_timeout)
            response.raise_for_status()
            result = response.json()
            return self._parse_glm_result(result, order_words)
        except requests.exceptions.Timeout:
            print(f"GLM-OCR timeout after {self.glm_timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            print(f"GLM-OCR request failed: {e}")
            return None
        except Exception as e:
            print(f"GLM-OCR error: {e}")
            return None

    def _parse_glm_result(self, result, order_words):
        """
        Parse GLM-OCR response to extract character positions.

        GLM-OCR response format (assumed):
        {
            "results": [
                {"text": "字符", "bbox": [x1, y1, x2, y2]},
                ...
            ]
        }

        Returns: [[char, x, y], ...] where x,y is center of bbox
        """
        try:
            # Try to extract results from GLM-OCR response
            # The actual response format may vary; handle common patterns
            if isinstance(result, dict):
                # Look for results in common locations
                items = result.get('results', [])
                if not items:
                    # Try alternative keys
                    items = result.get('data', [])
                if not items:
                    items = result.get('predictions', [])
                if not items:
                    items = result.get('output', [])
                if not items:
                    # If no structured results, try to interpret as raw OCR
                    return None

                words_loc = []
                for item in items:
                    # Handle different item formats
                    if isinstance(item, dict):
                        text = item.get('text', '')
                        if text and item.get('x') is not None and item.get('y') is not None:
                            try:
                                words_loc.append([text, _strict_int(item['x']), _strict_int(item['y'])])
                            except ValueError:
                                continue
                            continue

                        bbox = item.get('bbox', [])
                        if text and bbox and len(bbox) >= 4:
                            # Calculate center of bbox
                            try:
                                x1, y1, x2, y2 = [_strict_int(value) for value in bbox[:4]]
                            except ValueError:
                                continue
                            x = (x1 + x2) // 2
                            y = (y1 + y2) // 2
                            words_loc.append([text, x, y])

                # Match with order_words
                matched = []
                for target in order_words:
                    for wl in words_loc:
                        if wl[0] == target:
                            matched.append(wl)
                            break
                return matched if len(matched) == len(order_words) else None

            return None
        except Exception as e:
            print(f"Failed to parse GLM result: {e}")
            return None

    def _solve_with_chaojiying(self, image_content, order_words):
        """
        Call Chaojiying API to get character positions.

        Returns: [[char, x, y], ...] for all detected characters
        """
        try:
            from chaojiying import Chaojiying_Client

            chaojiying = Chaojiying_Client(self.cy_username, self.cy_password, self.cy_soft_id)
            ans_str = chaojiying.PostPic(image_content, 9501)

            if ans_str.get('err_str') != '':
                print(f"Chaojiying error: {ans_str.get('err_str')}")
                return None

            words = ans_str.get('pic_str', '').split('|')
            words_loc = []
            for w in words:
                parts = w.split(',')
                if len(parts) >= 3:
                    words_loc.append([parts[0], int(parts[1]), int(parts[2])])
            return words_loc
        except Exception as e:
            print(f"Chaojiying error: {e}")
            return None

    def _click_captcha(self, driver, target_element, words_loc, order_words):
        """Execute Selenium clicks on the captcha image."""
        from selenium.webdriver.common.action_chains import ActionChains

        actions = ActionChains(driver)

        for target_char in order_words:
            for wl in words_loc:
                if wl[0] == target_char:
                    # Click at the position relative to the target element
                    # Offset calculations based on original verify() function
                    actions.move_to_element_with_offset(
                        target_element,
                        int(wl[1]) - 160,
                        int(wl[2]) - 72
                    ).click().perform()
                    break

    def solve(self, driver):
        """
        Unified entry point for captcha solving.

        Strategy:
        1. If glm_enabled=True, try GLM-OCR first
        2. If GLM fails or glm_enabled=False, use Chaojiying
        3. Retry Chaojiying up to 3 times on failure

        Returns: log string
        """
        print("进入安全验证")
        log_str = "进入安全验证\n"

        try:
            target_element, order_words, image_content = self._get_captcha_info(driver)
            print(f"Target words: {order_words}")

            words_loc = None
            method_used = None

            # Try GLM-OCR first if enabled
            if self.glm_enabled:
                print("尝试使用 GLM-OCR 识别...")
                glm_result = self._solve_with_glm(image_content, order_words)
                if glm_result and len(glm_result) == len(order_words):
                    words_loc = glm_result
                    method_used = "GLM-OCR"
                    print(f"GLM-OCR 识别成功: {words_loc}")

            # Fallback to Chaojiying only when explicitly allowed.
            if words_loc is None and (not self.glm_enabled or self.allow_chaojiying_fallback):
                print("使用超级鹰识别...")
                for retry in range(3):
                    cy_result = self._solve_with_chaojiying(image_content, order_words)
                    if cy_result:
                        # Filter to only include the order_words characters
                        filtered = [wl for wl in cy_result if wl[0] in order_words]
                        if len(filtered) == len(order_words):
                            words_loc = filtered
                            method_used = "超级鹰"
                            print(f"超级鹰识别成功: {words_loc}")
                            break
                    if retry < 2:
                        print(f"超级鹰识别失败，重试 ({retry + 1}/3)")
                        time.sleep(1)

            if words_loc is None:
                log_str += "安全验证失败：无法识别验证码\n"
                print("安全验证失败：无法识别验证码")
                return log_str

            # Execute clicks
            self._click_captcha(driver, target_element, words_loc, order_words)

            print(f"安全验证成功 ({method_used})")
            log_str += f"安全验证成功\n"
            return log_str

        except Exception as e:
            log_str += f"安全验证异常: {e}\n"
            print(f"安全验证异常: {e}")
            return log_str


def solve_captcha(driver, glm_enabled, glm_endpoint, glm_timeout,
                  cy_username, cy_password, cy_soft_id, allow_chaojiying_fallback=False):
    """
    Convenience function to solve captcha.

    Args:
        driver: Selenium WebDriver instance
        glm_enabled: Whether to try GLM-OCR first
        glm_endpoint: GLM-OCR service endpoint
        glm_timeout: GLM-OCR request timeout in seconds
        cy_username: Chaojiying username
        cy_password: Chaojiying password
        cy_soft_id: Chaojiying soft_id
        allow_chaojiying_fallback: Whether to call Chaojiying after GLM failure

    Returns:
        log string describing the result
    """
    solver = CaptchaSolver(
        glm_enabled=glm_enabled,
        glm_endpoint=glm_endpoint,
        glm_timeout=glm_timeout,
        cy_username=cy_username,
        cy_password=cy_password,
        cy_soft_id=cy_soft_id,
        allow_chaojiying_fallback=allow_chaojiying_fallback,
    )
    return solver.solve(driver)
