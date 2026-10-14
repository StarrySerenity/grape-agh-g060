# AGH 真实执行摘要

科研命令由本机 Agnes Harness 中的 Agnes 实际调用。Codex 完成代码实现、隔离输入准备和演示整理；没有将 Codex 聊天历史冒充 AGH 执行。完整原生会话留在本地参赛档案，公开仓库提供命令序号、计算结果与核验摘要。

## 200 串联合训练与实拍迁移

27 次成功科研 shell 调用，包括实际 48 轮 CPU 训练、仿真标定、40 张仿真回归、277 张真实图像重新检测和分批匹配。模型仅用仿真训练；实拍不用于选权重或门限。

- [命令执行摘要](../evidence/combined200-real-20261014/execution-proof.json)
- [实测报告](../evidence/combined200-real-20261014/实测报告.md)
- [结果数据](../experiments/combined200/results/summary.json)

一次原汇总因上游报告脚本在快照后更新而停止；新增后处理显式记录这个不参与计算的差异，仍拒绝科学输入变化，没有改写冻结基线或删除失败记录。

## 幼果生长与实拍晚期比较

4 次成功科研 shell 调用：生成 6 串仿真幼果预测、计算 GROW_02、计算全部 18 对、核验。复用冻结的相对形态预测器，没有重新训练逐粒生长网络。

- [命令执行摘要](../evidence/growth-showcase-20261014/execution-proof.json)
- [独立数值核验](../evidence/growth-showcase-20261014/independent-verification.json)
- [单元检查摘要](../evidence/growth-showcase-20261014/unit-verification.json)
- [浏览器检查摘要](../evidence/growth-showcase-20261014/browser-verification.json)

执行成功不等于生物预测准确。修复版参考过同串晚期，始终保留条件对照标记。原始幼果预测排除自身身份；预测输出先冻结，再用本串晚期拟合比较。36 组相对形态指标独立重算一致，101 个受保护科学源文件核验一致。球体互穿失败照实保留。
