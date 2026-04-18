# PKU AutoBookingVenues Enhanced

PKU 智慧场馆自动预约工具。当前分支基于原项目继续维护，重点是把验证码识别迁移到本地 GLM-OCR、适配新版智慧场馆页面，并提供本地 WebUI 与抢场调度器。

## 相对原仓库的主要改动

- `config0.ini` 示例已改为 `config.example.ini`；实际使用时复制为 `config.ini`。`config.ini` 默认被 `.gitignore` 忽略。
- `duration` 已删除，预约长度只由 `[time] start_time` 和 `end_time` 决定。
- 安全验证优先使用本地 GLM-OCR 服务返回点击坐标。超级鹰只在 `allow_chaojiying_fallback = true` 时兜底。
- 付款流程只自动点击到“提交订单”。提交成功后程序等待人工自行完成网上付款，不再自动点击校园卡付款入口。
- 羽毛球场馆支持别名和父场馆写法，例如 `五四体育中心-羽毛球馆`、`邱德拔体育馆-羽毛球场`。
- 支持连续时段：例如 `start_time=20260422-0650`、`end_time=20260422-0950` 会尝试同一场地连续 3 小时。
- 新增 `booking_scheduler.py`：按“目标日期前 3 天 12:00 开放预约”的规则，在开放前 1 分钟开始反复运行。
- 新增 Flask WebUI：可编辑本地配置、创建多个预约任务、启动/停止调度器。

## 安装

建议使用已有的 `vllm` conda 环境：

```bash
cd /home/hcliu/projects/PKUAutoBookingVenues
source ~/miniconda3/bin/activate vllm
pip install -r requirements.txt
```

Firefox 默认使用 Selenium Manager 或项目内 `driver/geckodriver.bin`。如果当前环境没有 Selenium：

```bash
pip install selenium
```

## 配置

```bash
cp config.example.ini config.ini
```

重点字段：

- `[login] user_name/password`：IAAA 账号密码。
- `[type] venue`：推荐写完整父子场馆名，例如 `五四体育中心-羽毛球馆` 或 `邱德拔体育馆-羽毛球场`。
- `[type] venue_num`：指定场地编号填正整数；随机场地填 `-1`。
- `[time] start_time/end_time`：支持 `YYYYMMDD-HHMM` 或 `星期-HHMM`，星期格式中 `1` 到 `7` 表示周一到周日。
- `[glm_ocr] endpoint`：默认 `http://localhost:8000`，与 OCR 服务端口一致。

多个候选时段用 `/` 分隔：

```ini
start_time = 20260422-0650/20260422-0750
end_time = 20260422-0750/20260422-0850
```

连续几小时用一个起止范围表达：

```ini
start_time = 20260422-0650
end_time = 20260422-0950
```

## 本地 GLM-OCR

OCR 服务默认端口是 `8000`。

下载模型：

```bash
python scripts/download_glm_ocr.py --repo zai-org/GLM-OCR --output models/GLM-OCR
```

如果模型已经放在 `models/GLM-OCR`，直接启动：

```bash
bash scripts/start_ocr_server.sh
```

等价命令：

```bash
python ocr_server_transformers.py --model models/GLM-OCR --port 8000
```

健康检查：

```bash
curl http://localhost:8000/health
```

OCR 原理：

- 主程序进入安全验证后，读取页面目标字和验证码图片。
- 服务端让 GLM-OCR 识别图片中的候选汉字。
- 本地视觉预处理会过滤页面水印和无关文字，结合颜色文字区域生成候选框。
- `captcha_matcher.py` 将目标字与候选字匹配，返回每个目标字的点击坐标。
- 如果识别失败，会保存调试图片和元数据到 `models/captcha_failures/`，便于人工复核。

## 命令行运行

运行一次：

```bash
python main.py --config config.ini --browser firefox --once
```

失败后重试，成功即停止：

```bash
python main.py --config config.ini --browser firefox --retries 20
```

## 调度器

预约开放规则按“目标日期前 3 天 12:00”计算。例如 `2026-04-22` 最早在 `2026-04-19 12:00` 可预约，调度器会在 `11:59:00` 开始循环尝试。

准备任务文件：

```bash
cp tasks.example.json tasks.json
```

运行所有已到时间的任务一次：

```bash
python booking_scheduler.py --tasks tasks.json
```

常驻循环：

```bash
python booking_scheduler.py --tasks tasks.json --loop
```

同一时段同时抢五四和邱德拔：复制两份配置，例如 `configs/wusi.ini` 与 `configs/qdb.ini`，在 `tasks.json` 中各写一个任务。调度器会并发运行所有到点任务。

## WebUI

WebUI 默认端口是 `5000`。

启动：

```bash
python web_dashboard/app.py
```

打开：

```text
http://127.0.0.1:5000
```

WebUI 能做的事：

- 读取、编辑、保存本地 `config.ini` 或其他项目内配置文件。
- 一键生成连续 N 小时的 `start_time/end_time`。
- 创建多个未来预约任务，例如五四和邱德拔同一时段并发抢。
- 启动/停止本地 `booking_scheduler.py`。
- 查看最近运行状态和配置对应日志。

如果希望打包成单文件启动器：

```bash
pip install pyinstaller
bash scripts/build_webui.sh
./dist/pku-booking-webui
```

## 测试

```bash
python -m unittest discover -s tests
```

如果 OCR 服务已启动，可以额外跑端到端 OCR 检查：

```bash
GLM_ENDPOINT=http://localhost:8000/glmocr/parse python tests/test_glm_ocr.py
```

## 注意

- 订单提交后仍需人工在网页上完成付款。
- 本项目只应在你有权限预约的账号和场馆上使用。
- 自动化脚本可能因智慧场馆页面更新失效，失败样本优先看 `models/captcha_failures/` 和对应 `.log`。
