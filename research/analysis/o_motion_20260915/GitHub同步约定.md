# GitHub 研究进展同步

用户在 2026-09-17 明确要求“我们的进展及时提交 GitHub”。继续科研时在每个有意义的代码、实验完成/失败、方向决策节点同步，不等到论文完成才归档。

- 仓库：<https://github.com/chimeraHHH/Drive-OccWorld>
- 研究分支：`codex/motion-research-20260917`
- 本地独立 worktree：`code/Drive-OccWorld-research`
- 公开阅读入口：`research/README.md`
- 归档工具：`tools/sync_research_snapshot.py --workspace /path/to/dropple`

先更新原 `analysis/` 下代码/协议/结果报告和工作状态，再同步白名单、审查差异、核对结果证据、commit/push，最后用 `git ls-remote` 确认远端 SHA 与本地一致。推理方案应在读取评价标签和选择结果前固定；科学结论区分计划、运行中、已核验和失败。

数据集、权重、原始逐点预测/目标、SSH 配置、私钥、令牌和任务环境不提交。当前研究归档保持源文件字节与 SHA，运行需要本地/服务器上的额外资产；不能把代码归档称为公开数据或完整可复现模型发布。保留原工作目录未提交的其他变化，不覆盖旧分支，不 force-push。

首次同步：`8880449a47f1f6c7698e0bab1494c35958f2f54d`，包括 P2 接口、O 完整评价、运动诊断和世界模型文献研究。密集历史方案与独立核查在评分前以 `c25bfdb` 继续同步。
