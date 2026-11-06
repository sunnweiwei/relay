# EdgeBench 三臂对比 — 最终数据集

clm / folding / compaction 三个臂在 EdgeBench 10 道题上的完整结果，每臂每题各一轮跑满 12 小时的评测。

- **模型** gpt-6-luna（modelbest relay，reasoning effort medium），agent 是 Codex CLI 0.130.0，harness 是 SForge（EdgeBench 官方）。
- **clm / folding / compaction**：agent 容器内接 relay 插件，且已用 model catalog 把 Codex 自带的 compaction 关掉（catalog 把窗口报成 1e8，其 90% 触发阈值不可达）。
- **计价** input $0.10 / output $0.50 每百万 token，cached 不打折（按 input 全价计）。
- 每题得分是「多次提交、取最优」，所以一道题的结果是一条随时间上升的曲线，不是一个数。
- 10 道题：borden_source_inversion、git_rewrite_in_zig、anchorhead_text_adventure（game 题，取 peak_score）、wireless_electricity_layout、carleson_formalization、portfolio_risk_calibration、dcss_dungeon_ai、warehouse_forklift_routing、ffmpeg_swscale_reimplementation、ordinal_notation_well_foundedness。

一句话结论：**clm 平均分最高（14.32），compaction 次之（12.83），folding 最低（11.37）；clm 每题成本大约是 compaction 的六成。**

| | clm | folding | compaction |
|---|---|---|---|
| 10 题平均分（0-100） | 14.32 | 11.37 | 12.83 |
| 每题成本 | $24.57 | $37.50 | $42.50 |

## 三个文件

| 文件 | 每行 | 行数 |
|---|---|---|
| [final_scores.csv](final_scores.csv) | 一道题（臂 × 题）的最终结果 | 30 |
| [score_trajectory.csv](score_trajectory.csv) | 一次提交 | 2488 |
| [token_trajectory.csv](token_trajectory.csv) | 一次模型请求 | 73905 |

## 列说明

### final_scores.csv

| 列 | 含义 |
|---|---|
| arm | clm / folding / compaction |
| task | 题目名 |
| run_id | 产生这行的轮次：clm = relay-clm-b1…b4 + b10，folding，compaction 同理 |
| complete | True = 跑满 12h；本数据集 30 行全部为 True（被配额中断的旧轮和 native 臂已剔除） |
| best_score_raw | SForge 记录的原始最优分（各题量纲不同，不可跨题比较） |
| best_round / best_score_0_100 | 拿到最优分的提交轮 / 换算到 0-100 的最优分（跨题可比，均值基于它） |
| hours | 实际时长 |
| rounds / agent_submissions / auto_submissions | 总轮数 / agent 主动提交数 / 停止钩子自动提交数 |
| resumes | 断点续跑次数 |
| requests_answered / requests_with_usage | 网关 shim 记到的 200 请求数 / 其中带 usage 的 |
| input_tokens / cached_tokens / output_tokens | 全程消耗；cached 是 input 里命中缓存的部分（计价仍按全价） |
| cost_usd | input × $0.10/M + output × $0.50/M |

### score_trajectory.csv

| 列 | 含义 |
|---|---|
| round | `submissions/` 下的目录名：agent-N 为主动提交，auto-N 为钩子自动提交 |
| hours | 该次提交距开跑的小时数 |
| raw_score / score_0_100 | 该次提交的得分；anchorhead 是 game 题，取 game_result.json 的 peak_score（本身即 0-100），其余取 report.json 并按 SForge 的 rescale 规则换算 |
| pass_rate | 判题明细里的通过率 |
| best_so_far_0_100 | 截至该时刻的最优分（0-100）——画「分数-时间」曲线直接用它 |

### token_trajectory.csv

| 列 | 含义 |
|---|---|
| hours | 该请求距开跑的小时数 |
| input_tokens / cached_tokens / output_tokens / reasoning_tokens | 该次请求的消耗；reasoning 是输出中的推理部分（仅供参考，计价不含） |
| cum_input_tokens / cum_cached_tokens / cum_output_tokens | 该题到此为止的累计 |
| cum_cost_usd | 该题到此为止的累计花费——画「分数-成本」曲线用它做横轴 |

token 来自网关 shim 的逐请求 usage 日志，按 Codex session id（题目 transcript 里的 `session id:` 行）归属到题。


## 怎么来的

```
logs/runs/<轮次>/<题目>/        SForge 自己写的 final_result.json、每次提交的 report.json / game_result.json
../shim_usage.log.gz            网关 shim 日志备份（逐请求 usage，按 session id 归属到题）
        │
        ▼  ../export_raw.py     展平成原始记录（缺口留空，含全部轮次）
        ▼  ../make_final.py     剔除中断轮与 native 臂 → final/
```

`make_final.py` 重跑一遍即可从头重现这份数据集。

## 快速上手

```python
import pandas as pd

f = pd.read_csv("final_scores.csv")
print(f.groupby("arm")[["best_score_0_100", "cost_usd"]].mean())

# 一道题的分数-时间曲线
s = pd.read_csv("score_trajectory.csv")
one = s[(s.arm == "clm") & (s.task == "borden_source_inversion")]
one.plot(x="hours", y="best_so_far_0_100", drawstyle="steps-post")
```

十题平均的「分数-时间 / 分数-成本」图（curves.png）由 [make_curves.py](../../relay_arms/make_curves.py) 生成，口径与本数据集一致。
