# PKU AutoBookingVenues Enhanced

PKU 智慧场馆自动预约工具。当前分支基于原项目继续维护，重点是把验证码识别迁移到本地 GLM-OCR、适配新版智慧场馆页面，并提供本地 WebUI 与抢场调度器。

## 当前仓库架构与文件用途

状态标注：

- 当前主流程：日常运行会直接用到。
- 可选功能：配置开启或调试时才会用到。
- 遗留未用：来自旧项目或旧入口，当前推荐流程不依赖。
- 测试：只用于单元测试或集成测试。
- 运行生成：本地运行产生，不应提交。

```text
PKUAutoBookingVenues/
├── main.py                         # 当前主流程：预约入口，读取 config.ini，登录、选场、验证、提交订单
├── page_func.py                    # 当前主流程：Selenium 页面操作，包含进场馆、选时段、提交订单
├── booking_scheduler.py            # 当前主流程：按开放时间调度多个 config 任务，并发循环抢场
├── ocr_server_transformers.py      # 当前主流程：本地 GLM-OCR FastAPI 服务，默认 8000 端口
├── captcha_solver.py               # 当前主流程：验证码目标字提取、调用 OCR、点击坐标
├── captcha_matcher.py              # 当前主流程：OCR 候选字与目标字匹配
├── captcha_vision.py               # 当前主流程：验证码图像预处理、候选文字区域检测
├── config.example.ini              # 当前主流程：配置模板，复制为 config.ini 后使用
├── tasks.example.json              # 当前主流程：调度任务模板，复制为 tasks.json 后使用
├── requirements.txt                # 当前主流程：Python 依赖列表
├── web_dashboard/
│   ├── app.py                      # 当前主流程：Flask WebUI 后端，默认 5000 端口
│   ├── templates/index.html        # 当前主流程：WebUI 页面
│   └── routes.py                   # 遗留未用：旧占位文件，当前路由都在 app.py
├── scripts/
│   ├── download_glm_ocr.py         # 当前主流程：下载 GLM-OCR 模型到 models/GLM-OCR
│   ├── start_ocr_server.sh         # 当前主流程：启动 OCR 服务的便捷脚本
│   ├── webui_launcher.py           # 可选功能：PyInstaller 打包 WebUI 时的启动入口
│   └── build_webui.sh              # 可选功能：把 WebUI 打包成单文件启动器
├── chaojiying.py                   # 可选功能：超级鹰兜底接口，仅 allow_chaojiying_fallback=true 时使用
├── notice.py                       # 可选功能：Server 酱微信通知，仅 wechat_notice=true 时使用
├── tests/                          # 测试：单元测试、OCR 调试脚本、验证码样本抓取脚本
├── autoRun.bat                     # 遗留未用：旧 Windows 定时入口，当前推荐 booking_scheduler/WebUI
├── macAutoRun.sh                   # 遗留未用：旧 macOS crontab 包装脚本，当前推荐 booking_scheduler/WebUI
├── cron.py                         # 遗留未用：旧 crontab 管理脚本，当前 WebUI 不再调用
├── env_check.py                    # 遗留未用：旧 config0.ini/config1.ini 批量检测逻辑，当前 CLI 使用 --config
├── README.md                       # 文档
└── LICENSE                         # Apache-2.0 许可证
```

常见本地运行生成文件：

```text
config.ini                          # 运行生成：你的真实配置，已被 .gitignore 忽略
tasks.json                          # 运行生成：WebUI/调度器保存的真实任务，已忽略
status.json                         # 运行生成：main.py 最近一次运行状态，已忽略
scheduler_status.json               # 运行生成：调度器最近状态，已忽略
*.log                               # 运行生成：预约日志，已忽略
models/                             # 运行生成：OCR 模型和验证码失败样本，已忽略
.scheduler.pid                      # 运行生成：WebUI 启动调度器时记录的进程号，已忽略
```

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

建议新建独立的 `ocr` conda 环境，避免和已有推理、训练环境互相污染：

```bash
conda create -n ocr python=3.11 -y
conda activate ocr
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

---

以下是原项目 [PKUautoBookingVenues-fixed-by-cq-tutu](https://github.com/qqworld-tutu/PKUautoBookingVenues-fixed-by-cq-tutu) 的 [README](https://github.com/qqworld-tutu/PKUautoBookingVenues-fixed-by-cq-tutu/blob/master/README.md) 内容

---

# PKU AutoBookingVenues -fixed by cq-tutu

PKU智慧场馆自动预约工具部分代码和这个README的一部分引用自之前的智慧场馆自动预约项目 https://github.com/Charliecwei/PKU_Venues_Auto_Book

本羽毛球鼠鼠想打羽毛球一直抢不到场啊！于是便萌生了自动预约场地的想法。在github上找到了之前大佬写的程序。然而，年久失修的代码如今已经完全无法运行。因此，本项目在之前项目的基础上进行了较大修改，主要体现在以下三个方面：

1. selenium库更新了，之前的语法已经无法跑通
2. 智慧场馆网站也新写了，许多爬虫代码也随之修改
3. 提交预约时的验证方式改成了文字点击，于是使用了超级鹰的api来识别

> [!CAUTION]
>
> 本项目还在初期阶段，各方面都不甚完善，如有任何意见或建议，欢迎联系我。（wechat：dj7152，email：chenquan@stu.pku.edu.cn）

## 说明

- 本工具采用 Python3 搭配 `selenium` 完成自动化操作，实现全自动预约场馆
- 本项目需要提前安装firefox浏览器以及驱动，请把firefox驱动复制到该文件夹下
- 支持基于[Server酱](https://sct.ftqq.com/)的备案结果微信推送功能，体验更佳
- 采用定时任务可实现定期（如每周）免打扰预约，请设置在三天前的11:55-12:00之间
- 第三方依赖包几乎只有 `selenium` 一个
- 由于我只测试过羽毛球场的预约，其他场馆只是理论上可行，如果出现任何问题，可以提issue
- 目前仅支持一个小时的预约，后续可能会增加两个小时预约的功能
- `config`参数填写`config.ini`文件的名称，类型为字符串
- `lst_config`为config文件名称字符串构成的列表
- `page(config)`单独处理每个`config.ini`文件,`muilti_run(lst_config)`并行处理`lst_config`列表中的所有`config.ini`，`sequence_run(lst_config)`按序处理
- 定时任务还未经过完全测试
- 注意付款需要手动执行！请在10分钟内自行完成付款！

## 安装与需求

### Python 3

本项目需要 Python 3，可以从[Python 官网](https://www.python.org/)下载安装

### Packages

#### selenium

采用如下命令安装 `selenium`，支持 selenium 4 及以上版本：

```python
pip install selenium
```

### API

#### 超级鹰

在https://www.chaojiying.com/注册，并充钱（最少充10块钱，可识别验证码625次）

然后，进入用户中心，在左侧菜单栏中点击“软件ID”，生成一个软件ID，并填入config文件中的soft_id

## 基本用法

1. 复制 `config.example.ini` 为 `config.ini`，在 `config.ini` 中填写自己的账号、场馆和时间配置。

2. 用文本编辑器（建议代码编辑器）打开 `config.ini` 文件

3. 配置 `[login]`、`[type]`、`[time]`、`[wechat_notice]`、`[glm_ocr]`、`[chaojiying]` 这几个 Section 下的变量；示例文件内有详细注释。`duration` 已废弃，预约长度由 `start_time` 和 `end_time` 推导。

## 定时运行

### Windows

本项目中的 `autoRun.bat` 文件可提供在静默免打扰情况下运行程序的选择，配合 Windows 任务计划管理可实现定期自动填报，具体请参考[Win10下定时启动程序或脚本](https://blog.csdn.net/xielifu/article/details/81016220)

### mac OS

进入项目根目录，以命令 `./macAutoRun.sh` 执行 `macAutoRun.sh` 脚本即可，可设定或取消定时运行

### Linux

使用 `crontab` 设置

**Note:** 静默运行的弊端为无法看到任何报错信息，若程序运行有错误，使用者很难得知。故建议采用定时静默运行时，设置微信推送，在移动端即可查看到备案成功信息。

## 微信推送

本项目支持基于[Server酱](https://sct.ftqq.com/)的微信推送功能，仅需登录并扫码绑定，之后将获取到的 SCKEY 填入 `config0.ini` 文件即可

## 责任须知

- 本项目仅供参考学习，造成的一切后果由使用者自行承担

## 证书

[Apache License 2.0](https://github.com/yanyuandaxia/PKUAutoBookingVenues/blob/main/LICENSE)

## 版本历史

### version 1.0

- 发布于 2025.2.9
- 项目初始版本
