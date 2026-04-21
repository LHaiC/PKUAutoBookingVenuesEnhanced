# OCR GLM 恢复方案

> **Goal:** 尽快恢复本地验证码识别可用率，主路径回到 `bbox proposal + per-box GLM`，但彻底重构 bbox proposal 的生成与选择逻辑，避免 merged box、truncated box 和误点。

## 背景

当前 `master` 已切到 `PaddleOCR` 主路径，但实际样本和 bug 记录表明：

- 该验证码更像“小目标、任意旋转、颜色强先验、字符集受限”的检测/分类问题，不适合直接交给通用场景 OCR 端到端处理
- `PaddleOCR` 在真实样本上存在检测框漂移和识别错误，无法稳定返回正确点击坐标
- 历史 `GLM per-box` 方案的整体方向是对的，问题集中在 `bbox proposal`：会漏框、粘框、切半框

项目目标不是“识别出几个字”，而是“按提示顺序返回 3 个正确点击坐标”。因此恢复方案必须围绕坐标正确率设计，并保持 fail-closed。

## 目标与非目标

### 目标

- 恢复本地 OCR 主路径的实际可用率
- 继续使用本地 `GLM` 做 per-box 单字识别
- 将 bbox 逻辑从单一路径升级为多 proposal 候选 + 后验打分
- 让 `test_ocr_samples.sh` 成为恢复阶段的第一验收入口
- 保证无法确定时拒绝点击，而不是自信地误点

### 非目标

- 本轮不坚持纯 `PaddleOCR` 替代
- 本轮不训练新的检测模型
- 本轮不把 VLM/MCP 接入线上主推理链路

## 推荐方案

### 方案对比

1. **推荐：恢复 `bbox proposal + per-box GLM`，重做 proposal 和 scoring**
   - 恢复速度最快
   - 最贴合当前验证码特点
   - 与现有代码结构最兼容

2. **训练轻量检测模型替代规则 bbox**
   - 长期可能更稳
   - 但不适合“尽快恢复”，需要标注、训练、部署

3. **保留 PaddleOCR 作为辅助 proposal 源**
   - 理论上可补漏
   - 但当前 detector 偏差较大，短期会增加调试复杂度

### 结论

主线采用方案 1：恢复 `GLM per-box` 主路径，但不恢复旧的“单一路径 bbox 决策”，而是改为“多 proposal 候选 + 后验打分 + 失败回退”。

## 架构设计

### 模块边界

- `captcha_vision.py`
  - 只负责生成候选框
  - 不负责决定最终答案

- `ocr_server_transformers.py`
  - 负责多路 proposal 编排
  - 负责 per-box GLM
  - 负责 proposal 打分、目标匹配、失败回退

- `tests/`
  - 负责真实样本回归
  - 成功标准是“返回正确点击坐标和顺序”，不是“识别出一些字”

### 环境约束

- 使用现有本地环境：`conda activate vllm`
- OCR 服务继续通过现有 tmux 会话维护：`tmux a -t ocr`
- 不引入新的远程 OCR 依赖

## Proposal 设计

### 核心原则

`bbox` 不再要求“一次切准”，而是要求“候选集合里至少有一套切得够准”。

### 新接口

将旧的 `detect_colored_text_bboxes()` 从“直接给最终 box 列表”改成“生成多套候选”：

```python
generate_box_proposals(image) -> list[ProposalSet]
```

每个 `ProposalSet` 至少包含：

- `boxes`
- `source`
- `preprocess_variant`

### 保留的 proposal 源

- `uniform_color_regions`
  - 保留，继续作为主检测源

- `dark_regions`
  - 保留，但只作补充源，不直接拍板

- `color_split_refinement`
  - 保留，但只在 merged box 特征明显时拆分

- `shifted / padded / preprocess variants`
  - 新增
  - 对同一张图在 watermark whitening 前后、阈值轻微扰动、padding 变化等条件下各跑一次，形成多套 proposal

### 降权或删除的旧行为

- `_merge_nearby_boxes(margin=4)` 的一刀切合并
  - 这是 merged box 的主要来源
  - 需要改成更严格的 overlap / bridge 判定

- 仅依赖单框绝对宽高面积的硬过滤
  - 会误杀细长字或偏小字
  - 应改为 scoring 项，而不是 hard drop

- `detect_colored_text_bboxes()` 直接输出“最终答案”
  - 应废止，只保留 proposal 角色

### 新的筛选流程

1. 生成 3 到 4 套 `ProposalSet`
2. 每套先做基础合法性检查，框数大致在 `3..6`
3. 每个 box 跑一次 per-box GLM
4. 将 GLM 结果、几何特征、颜色特征一起交给 scoring
5. 选出最优 `ProposalSet` 再进入 `match_targets()`

## 后验打分

### `score_proposal_set()`

每套 proposal 的分数由以下项组成：

- `target_coverage`
  - 是否能唯一覆盖 3 个目标字

- `single_char_quality`
  - per-box GLM 是否稳定输出单字，而不是多字、空串、噪声

- `box_geometry`
  - 单框宽高、面积、长宽比是否像单个汉字

- `size_consistency`
  - 所有最终候选字大小基本一致
  - 若某一框显著更大，优先怀疑 merged box
  - 若某一框显著更小，优先怀疑 truncated box 或噪声框

- `box_separation`
  - 候选框之间是否过度重叠或中心异常接近

- `color_coherence`
  - 单框主色是否集中；背景混入越多，得分越低

### `accept_solution()`

只在以下条件全部满足时允许返回点击坐标：

- 3 个目标字都被唯一匹配
- 每个目标对应不同 box
- 每个 box 质量都高于最低接受阈值
- 最高分方案与次高分方案有足够分差

任何歧义较大的情况都拒绝点击。

## 失败回退

失败回退顺序如下：

1. 主路径：多 proposal + per-box GLM
2. 次路径：仅对低置信 box 补跑 rotated per-box GLM
3. 特例补位：只保留已经验证有效的单缺字补位和少量混淆字规则
4. 最后：允许 fallback 到现有人工/第三方方案，或者直接失败

关键原则：

- 优先 fail-closed
- 不允许同一 box 被两个目标字复用
- 不允许低质量推断直接变成点击动作

## 测试与验证

### 第一层：单元测试

补充或恢复以下测试：

- merged box 场景：proposal 至少有一套能拆开
- truncated box 场景：半框会被显著降权
- duplicate/overlap 场景：高重叠框不能被当成两个有效目标
- single-missing-target 场景：只保留已有验证通过的补位规则

### 第二层：样本回归

基于真实验证码样本建立回归集：

- `pass samples`：必须返回 3 个正确点击坐标
- `known failure samples`：允许失败，但不允许误点

恢复阶段保留 `test_ocr_samples.sh` 作为第一验收入口，不先急着替换。

### 第三层：线上冒烟

在以下既有环境中做只读验证：

- `conda activate vllm`
- `tmux a -t ocr`

先采样、识别、记录结果，不直接提交预约。连续多轮稳定后，才恢复自动点击。

## 标注策略

必要时允许使用 VLM/MCP 做困难样本辅助标注，但不进入线上主链路。

用途仅限于：

- 标注失败样本中的目标字位置或 `ground truth bbox`
- 验证 proposal 是否漏框、粘框、切半框
- 校准 scoring 阈值

标注产物应单独落盘，例如：

- `tests/samples/*.json`
- `models/captcha_failures/*.json`

不应混在运行日志中。

## 影响文件

- `captcha_vision.py`
  - proposal 生成与局部几何/颜色特征

- `ocr_server_transformers.py`
  - 多 proposal 编排、per-box GLM、score/accept/fallback

- `tests/test_captcha_vision.py`
  - proposal 相关单元测试

- `tests/test_ocr_server_transformers.py`
  - score/fallback 相关单元测试

- `test_ocr_samples.sh`
  - 恢复阶段第一验收入口

## 验收标准

- 离线真实样本成功率显著高于当前 `PaddleOCR` 路径
- 困难样本允许失败，但误点率接近 0
- 在线冒烟阶段连续多轮没有明显 merged box、truncated box、重复框问题

## 交付顺序

1. 从 git 恢复 `GLM per-box` 主路径所需代码
2. 将 bbox 逻辑改造成多 proposal 生成
3. 加入 proposal scoring 和 `accept_solution()`
4. 恢复并补全单元测试
5. 用 `test_ocr_samples.sh` 跑离线样本回归
6. 再做线上只读式冒烟验证
