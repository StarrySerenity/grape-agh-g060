# 仿真幼果预测与实拍前后球体对照

2026年10月14日按用户要求，继续原对话已完成的6串仿真幼果预测方法，并实际计算之前修复过的实拍生长前后配对。Agnes通过AGH前台执行生成、预测和核验；Codex准备隔离输入、适配接口和演示页面。原预测器和输入只读保留。

## 方法与输入

`forecast.py`、`growth_geometry.py`、`growth_data.py`与原项目字节一致。仿真使用同样的6组种子，306粒球体；GROW对应40天窗口，YG对应14天。仍使用已有实拍配对的相对形态模型，不新训练逐粒生长模型。1.35倍中心扩张是展示设定，不是实测尺寸增长。

实拍保留18组已确认阶段的配对，每组两种输入：

- 原始幼果：排除本串身份后，仅由其他果串配对预测。预测文件保存后，才读取本串实际晚期球体做相对形状比较和相似变换配准。
- 已修复幼果：输入曾参考同一串晚期几何，因此只作为条件形态对照。沿用现有交互预测的几何适配方式，原输入和所有输出保留future-reference来源标记，不将其计作独立早期预测。

默认展示GROW_02：May_g2_02（2026年5月28日）与July_g1_02（7月7日），原始、修复与晚期均39粒。YG07/08疏果前后阶段未确认，不进入本轮18组。保存的手工球模型和原点云没有被覆盖。

## 比较含义

四项无量纲指标为相对半径、中轴/主轴方差比、短轴/主轴方差比、相对最近间距。比较的是接触调整后的实际生成球体，区别于Ridge预测的目标数值。晚期球体通过相似变换显示在预测坐标框架中，变换仅用于对照，不能解释为真实生长倍率。

原始幼果与预测保留相同编号，晚期模型编号并未证实与它们逐粒对应。配准后的双向球心最近距离是归一化几何距离，不是逐粒生物身份误差。没有物理尺度标定，输出不含毫米、真实生长率或掉粒预测。

六个仿真终点未发现互穿，但没有真实未来答案，不能据此计算生物准确率。实拍各变体的互穿结果照实保存，部分未解决，不通过放宽约束改成通过。默认GROW_02两种输入均clear；其修复版只是相对半径误差较小，其他三项误差更大，不声称修复使预测整体更准。

## 本轮实际结果

AGH会话：`agnes:local:local-dev:g060:growth:1791986269488`，标题“看看幼果长大后，和实拍球体比一比”。实际执行synthetic/generate.py、real_predict.py --identity GROW_02、--all、--verify，全部正常结束。

GROW_02原始幼果四项绝对差约0.0226、0.0405、0.0532、0.0019；修复条件版约0.0179、0.0559、0.0597、0.0074。这是单对相对形态差，不是准确率。其余逐对数值在real/<ID>/record.json与results/showcase_summary.json。

## 复现

使用已有grape1环境：Python3.9、NumPy/SciPy/scikit-learn/Open3D。脚本在导入数值库前固定MKL sequential和CPU两线程。

```powershell
& 'D:/Anaconda/envs/grape1/python.exe' -B synthetic/generate.py
& 'D:/Anaconda/envs/grape1/python.exe' -B real_predict.py --identity GROW_02
& 'D:/Anaconda/envs/grape1/python.exe' -B real_predict.py --all
& 'D:/Anaconda/envs/grape1/python.exe' -B real_predict.py --verify
& 'D:/Anaconda/envs/grape1/python.exe' -B -m unittest test_real_predict -v
```

已有仿真结果会拒绝覆盖；实拍已完成配对会跳过。若重算应在新的输出目录复现。source-manifest.json记录101个原文件SHA256，verify逐个核对；两个输入变体、先预测后比较的文件哈希、训练排除本串和修复来源标记均有检查。

演示位于比赛项目的web/showcase.html，默认展示FP3_11实拍指纹成功案例，并保留61张整批结果说明；6串仿真与18对实拍均可切换、同步旋转、播放端点插值并下载球体JSON和生成PLY。
