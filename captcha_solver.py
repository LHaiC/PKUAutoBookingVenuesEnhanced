from __future__ import annotations

import base64
import datetime
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from captcha_matcher import Candidate, match_targets, normalize_candidates
from captcha_vision import (
    build_colored_text_strip,
    decode_image,
    prepare_captcha_boxes,
)

CAPTCHA_DEBUG_DIR = 'models/captcha_failures'


class CaptchaSolveError(RuntimeError):
    pass


@dataclass
class CaptchaContext:
    target_element: Any
    order_words: list[str]
    image_content: bytes
    image_size: tuple[int, int] | None


@dataclass
class ProviderAttempt:
    provider: str
    success: bool
    detail: str
    matched: list[dict] | None = None
    raw_result: Any = None


class CaptchaSolver:
    IMAGE_XPATHS = [
        "//img[starts-with(@src, 'data:image')]",
        "/html/body/div[1]/div/div/div[3]/div[2]/div/div[1]/div[2]/div[4]/div[3]/div/div[2]/div/div[1]/div/img",
    ]
    ORDER_XPATHS = [
        "//*[contains(text(), '点击') or contains(text(), '依次')]",
        "/html/body/div[1]/div/div/div[3]/div[2]/div/div[1]/div[2]/div[4]/div[3]/div/div[2]/div/div[2]/span",
    ]

    def __init__(
        self,
        glm_enabled,
        glm_endpoint,
        glm_timeout,
        allow_chaojiying_fallback,
        cy_username,
        cy_password,
        cy_soft_id,
        debug_dir: str = CAPTCHA_DEBUG_DIR,
        glm_proxy: str | None = None,
    ):
        self.glm_enabled = bool(glm_enabled)
        self.glm_endpoint = (glm_endpoint or '').rstrip('/')
        self.glm_timeout = glm_timeout
        self.glm_proxy = glm_proxy
        self.allow_chaojiying_fallback = bool(allow_chaojiying_fallback)
        self.cy_username = cy_username
        self.cy_password = cy_password
        self.cy_soft_id = cy_soft_id
        self.debug_dir = debug_dir

    def _parse_order_words(self, raw_text: str) -> list[str]:
        text = (raw_text or '').strip()
        if '：' in text:
            text = text.rsplit('：', 1)[-1]
        elif ':' in text:
            text = text.rsplit(':', 1)[-1]
        if '依次' in text:
            text = text.split('依次', 1)[-1]
        if '点击' in text:
            text = text.split('点击', 1)[-1]
        return re.findall(r'[\u4e00-\u9fff]', text)

    def _is_displayed(self, element) -> bool:
        try:
            return element.is_displayed()
        except Exception:
            return True

    def _visible_elements(self, driver, by, xpath: str) -> list[Any]:
        try:
            return [element for element in driver.find_elements(by.XPATH, xpath) if self._is_displayed(element)]
        except Exception:
            return []

    def _choose_captcha_image(self, driver, by):
        candidates = []
        seen = set()
        for xpath in self.IMAGE_XPATHS:
            for element in self._visible_elements(driver, by, xpath):
                src = element.get_attribute('src') or ''
                if not (src.startswith('data:image') and ',' in src):
                    continue
                size = getattr(element, 'size', None) or {}
                key = (
                    src[:64],
                    size.get('width', 0),
                    size.get('height', 0),
                )
                if key in seen:
                    continue
                seen.add(key)
                area = (size.get('width') or 0) * (size.get('height') or 0)
                candidates.append((area, element))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1] if candidates else None

    def _choose_order_words(self, driver, by) -> list[str]:
        candidates: list[tuple[int, list[str]]] = []
        for xpath in self.ORDER_XPATHS:
            for element in self._visible_elements(driver, by, xpath):
                words = self._parse_order_words(getattr(element, 'text', '') or '')
                if len(words) >= 2:
                    candidates.append((len(words), words))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1] if candidates else []

    def _decode_data_uri(self, image_uri: str) -> bytes:
        if not image_uri or ',' not in image_uri:
            raise CaptchaSolveError('invalid captcha data uri')
        return base64.b64decode(image_uri.split(',', 1)[1])

    def _save_page_debug_snapshot(self, driver, reason: str) -> dict[str, str | None]:
        os.makedirs(self.debug_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        html_path = os.path.join(self.debug_dir, f'page-{reason}-{stamp}.html')
        screenshot_path = os.path.join(self.debug_dir, f'page-{reason}-{stamp}.png')
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(getattr(driver, 'page_source', ''))
        except Exception:
            html_path = None
        try:
            driver.save_screenshot(screenshot_path)
        except Exception:
            screenshot_path = None
        return {'html': html_path, 'screenshot': screenshot_path}

    def _save_debug_bundle(
        self,
        context: CaptchaContext | None,
        reason: str,
        attempts: list[ProviderAttempt] | None = None,
    ) -> dict[str, str | None]:
        os.makedirs(self.debug_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        result = {'image': None, 'meta': None, 'strip': None}
        meta: dict[str, Any] = {
            'captured_at': stamp,
            'reason': reason,
            'attempts': [
                {
                    'provider': attempt.provider,
                    'success': attempt.success,
                    'detail': attempt.detail,
                    'matched': attempt.matched,
                    'raw_result': attempt.raw_result,
                }
                for attempt in (attempts or [])
            ],
        }
        if context is not None:
            image_path = os.path.join(self.debug_dir, f'captcha-{stamp}.png')
            with open(image_path, 'wb') as f:
                f.write(context.image_content)
            result['image'] = image_path
            meta['targets'] = context.order_words
            meta['image_size'] = context.image_size
            try:
                image = decode_image(context.image_content)
                boxes = prepare_captcha_boxes(image, refine=True)
                meta['detected_boxes'] = boxes
                strip = build_colored_text_strip(image, boxes=boxes)
                if strip is not None:
                    strip_path = os.path.join(self.debug_dir, f'captcha-strip-{stamp}.png')
                    strip.save(strip_path)
                    result['strip'] = strip_path
            except Exception as exc:
                meta['vision_error'] = str(exc)
        meta_path = os.path.join(self.debug_dir, f'captcha-{stamp}.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        result['meta'] = meta_path
        return result

    def _get_captcha_context(self, driver, timeout: float = 8.0, poll_interval: float = 0.15) -> CaptchaContext:
        from selenium.webdriver.common.by import By

        deadline = time.time() + timeout
        while True:
            image_element = self._choose_captcha_image(driver, By)
            order_words = self._choose_order_words(driver, By)
            if image_element is not None and order_words:
                image_content = self._decode_data_uri(image_element.get_attribute('src') or '')
                image_size = None
                try:
                    image_size = decode_image(image_content).size
                except Exception:
                    pass
                return CaptchaContext(
                    target_element=image_element,
                    order_words=order_words,
                    image_content=image_content,
                    image_size=image_size,
                )
            if time.time() >= deadline:
                self._save_page_debug_snapshot(driver, 'captcha-not-found')
                raise CaptchaSolveError('captcha image or instruction not found')
            time.sleep(poll_interval)

    def _effective_image_size(self, context: CaptchaContext, result: Any) -> tuple[int, int] | None:
        if context.image_size is not None:
            return context.image_size
        if isinstance(result, dict):
            size = result.get('image_size')
            if isinstance(size, (list, tuple)) and len(size) == 2:
                try:
                    width = int(size[0])
                    height = int(size[1])
                except Exception:
                    return None
                if width > 0 and height > 0:
                    return (width, height)
        return None

    def _iter_candidate_items(self, payload: Any):
        if isinstance(payload, list):
            for item in payload:
                yield from self._iter_candidate_items(item)
            return
        if not isinstance(payload, dict):
            return

        keys = set(payload.keys())
        if {'text', 'bbox'}.issubset(keys) or {'text', 'x', 'y'}.issubset(keys):
            yield payload
        if {'char', 'bbox'}.issubset(keys) or {'char', 'x', 'y'}.issubset(keys):
            normalized = dict(payload)
            normalized['text'] = normalized.get('char')
            yield normalized
        for key in ('results', 'data', 'predictions', 'output', 'items', 'detections', 'chars', 'objects'):
            if key in payload:
                yield from self._iter_candidate_items(payload[key])

    def _candidate_from_item(self, item: dict, image_size: tuple[int, int] | None) -> dict | None:
        text = str(item.get('text', '')).strip()
        if len(text) != 1:
            return None

        bbox = item.get('bbox')
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            except Exception:
                return None
            if image_size is not None:
                width, height = image_size
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    return None
            return {
                'text': text,
                'bbox': [x1, y1, x2, y2],
                'confidence': float(item.get('confidence', 1.0) or 1.0),
            }

        x = item.get('x')
        y = item.get('y')
        if isinstance(x, int) and not isinstance(x, bool) and isinstance(y, int) and not isinstance(y, bool):
            if image_size is not None:
                width, height = image_size
                if not (0 <= x < width and 0 <= y < height):
                    return None
                bbox = [max(0, x - 1), max(0, y - 1), min(width, x + 1), min(height, y + 1)]
            else:
                bbox = [x, y, x + 1, y + 1]
            return {
                'text': text,
                'bbox': bbox,
                'confidence': float(item.get('confidence', 1.0) or 1.0),
            }
        return None

    def _extract_glm_candidates(self, result: Any, image_size: tuple[int, int] | None) -> list[Candidate]:
        if isinstance(result, dict) and isinstance(result.get('raw_output'), str):
            try:
                nested = json.loads(result['raw_output'])
            except Exception:
                nested = None
            if nested is not None:
                nested_candidates = self._extract_glm_candidates(nested, image_size)
                if nested_candidates:
                    return nested_candidates

        items = []
        for item in self._iter_candidate_items(result):
            candidate = self._candidate_from_item(item, image_size)
            if candidate is not None:
                items.append(candidate)
        return normalize_candidates(items)

    def _solve_with_glm(self, context: CaptchaContext) -> ProviderAttempt:
        image_b64 = base64.b64encode(context.image_content).decode('utf-8')
        data_uri = f'data:image/jpeg;base64,{image_b64}'
        payload = {'images': [data_uri], 'targets': context.order_words}
        url = f'{self.glm_endpoint}/glmocr/parse'

        try:
            session = requests.Session()
            session.trust_env = False  # ignore http_proxy/https_proxy env vars
            proxies = {'http': self.glm_proxy, 'https': self.glm_proxy} if self.glm_proxy else None
            response = session.post(url, json=payload, timeout=self.glm_timeout, proxies=proxies)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.Timeout:
            return ProviderAttempt('GLM-OCR', False, f'timeout after {self.glm_timeout}s')
        except requests.exceptions.RequestException as exc:
            return ProviderAttempt('GLM-OCR', False, f'request failed: {exc}')
        except Exception as exc:
            return ProviderAttempt('GLM-OCR', False, f'unexpected request error: {exc}')

        effective_size = self._effective_image_size(context, result)
        candidates = self._extract_glm_candidates(result, effective_size)
        try:
            matched = match_targets(context.order_words, candidates, effective_size or context.image_size or (1, 1), min_confidence=0.45)
        except Exception as exc:
            detail = f'parsed {len(candidates)} candidates but could not match targets: {exc}'
            if isinstance(result, dict) and result.get('error'):
                detail = f'{detail}; service_error={result.get("error")}'
            return ProviderAttempt('GLM-OCR', False, detail, raw_result=result)
        return ProviderAttempt('GLM-OCR', True, f'matched {len(matched)} targets', matched=matched, raw_result=result)

    def _solve_with_chaojiying(self, context: CaptchaContext) -> ProviderAttempt:
        try:
            from chaojiying import Chaojiying_Client

            client = Chaojiying_Client(self.cy_username, self.cy_password, self.cy_soft_id)
            result = client.PostPic(context.image_content, 9501)
        except Exception as exc:
            return ProviderAttempt('超级鹰', False, f'request failed: {exc}')

        if result.get('err_str'):
            return ProviderAttempt('超级鹰', False, f'service error: {result.get("err_str")}', raw_result=result)

        items = []
        for chunk in (result.get('pic_str') or '').split('|'):
            parts = chunk.split(',')
            if len(parts) < 3:
                continue
            text = str(parts[0]).strip()
            if len(text) != 1:
                continue
            try:
                x = int(parts[1])
                y = int(parts[2])
            except Exception:
                continue
            items.append({'text': text, 'x': x, 'y': y, 'confidence': 1.0})

        candidates = normalize_candidates(items)
        try:
            matched = match_targets(context.order_words, candidates, context.image_size or (1, 1), min_confidence=0.0)
        except Exception as exc:
            return ProviderAttempt('超级鹰', False, f'parsed {len(candidates)} candidates but could not match targets: {exc}', raw_result=result)
        return ProviderAttempt('超级鹰', True, f'matched {len(matched)} targets', matched=matched, raw_result=result)

    def _element_size(self, target_element) -> tuple[float, float]:
        size = getattr(target_element, 'size', None) or {}
        rect = getattr(target_element, 'rect', None) or {}
        width = size.get('width') or rect.get('width')
        height = size.get('height') or rect.get('height')
        if not width or not height:
            raise CaptchaSolveError('captcha element size unavailable')
        return float(width), float(height)

    def _click_offset(self, target_element, x: int, y: int, image_size: tuple[int, int] | None) -> tuple[int, int]:
        element_width, element_height = self._element_size(target_element)
        if image_size is not None:
            image_width, image_height = image_size
            if not (0 <= x < image_width and 0 <= y < image_height):
                raise CaptchaSolveError('captcha coordinate out of image bounds')
            rendered_x = x * element_width / image_width
            rendered_y = y * element_height / image_height
        else:
            rendered_x = x
            rendered_y = y
            if not (0 <= rendered_x < element_width and 0 <= rendered_y < element_height):
                raise CaptchaSolveError('captcha coordinate out of element bounds')
        return (round(rendered_x - element_width / 2), round(rendered_y - element_height / 2))

    def _click_captcha(self, driver, context: CaptchaContext, matched: list[dict]):
        from selenium.webdriver.common.action_chains import ActionChains

        try:
            driver.execute_script('arguments[0].scrollIntoView({block: "center"});', context.target_element)
        except Exception:
            pass

        effective_size = context.image_size
        for item in matched:
            offset_x, offset_y = self._click_offset(
                context.target_element,
                int(item['x']),
                int(item['y']),
                effective_size,
            )
            ActionChains(driver).move_to_element_with_offset(
                context.target_element,
                offset_x,
                offset_y,
            ).pause(0.05).click().perform()
            time.sleep(0.08)

    def solve(self, driver):
        print('进入安全验证')
        log_str = '进入安全验证\n'
        attempts: list[ProviderAttempt] = []
        context: CaptchaContext | None = None

        try:
            context = self._get_captcha_context(driver)
            print(f'Target words: {context.order_words}')

            if self.glm_enabled:
                glm_attempt = self._solve_with_glm(context)
                attempts.append(glm_attempt)
                print(f'GLM-OCR: {glm_attempt.detail}')
                if glm_attempt.success:
                    self._click_captcha(driver, context, glm_attempt.matched or [])
                    log_str += '安全验证成功\n'
                    return log_str

            if self.allow_chaojiying_fallback:
                for retry in range(3):
                    cy_attempt = self._solve_with_chaojiying(context)
                    attempts.append(cy_attempt)
                    print(f'超级鹰: {cy_attempt.detail}')
                    if cy_attempt.success:
                        self._click_captcha(driver, context, cy_attempt.matched or [])
                        log_str += '安全验证成功\n'
                        return log_str
                    if retry < 2:
                        time.sleep(0.5)

            bundle = self._save_debug_bundle(context, 'captcha_solve_failed', attempts)
            self._save_page_debug_snapshot(driver, 'captcha-solve-failed-page')
            log_str += '安全验证失败：无法识别验证码\n'
            if bundle.get('meta'):
                print(f'已保存验证码调试元数据: {bundle["meta"]}')
            if bundle.get('image'):
                print(f'已保存验证码样本: {bundle["image"]}')
            raise CaptchaSolveError(log_str.rstrip())
        except CaptchaSolveError:
            raise
        except Exception as exc:
            if context is not None:
                self._save_debug_bundle(context, 'captcha_exception', attempts)
            self._save_page_debug_snapshot(driver, 'captcha-exception-page')
            log_str += f'安全验证异常: {exc}\n'
            raise CaptchaSolveError(log_str.rstrip()) from exc


def solve_captcha(driver, glm_enabled, glm_endpoint, glm_timeout,
                  allow_chaojiying_fallback, cy_username, cy_password, cy_soft_id,
                  glm_proxy: str | None = None):
    solver = CaptchaSolver(
        glm_enabled=glm_enabled,
        glm_endpoint=glm_endpoint,
        glm_timeout=glm_timeout,
        allow_chaojiying_fallback=allow_chaojiying_fallback,
        cy_username=cy_username,
        cy_password=cy_password,
        cy_soft_id=cy_soft_id,
        glm_proxy=glm_proxy,
    )
    return solver.solve(driver)
