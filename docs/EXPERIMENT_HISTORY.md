# 葡萄结构指纹与跨期形态分析智能体

## 当前展示：实拍成功示例与生长前后球体对照

最新入口 http://127.0.0.1:8870/web/showcase.html ，Start-Demo.cmd默认打开此页。识别选FP3_11这一串的3张成功照片，保留完整61张实拍统计说明。新增6串仿真幼果（306粒）与18对实拍生长配对，可同步旋转、选粒、播放端点插值、下载JSON和生成PLY。原始幼果与预测、实际晚期拟合三栏对照，默认GROW_02，修复版保留晚期辅助来源标记并明确为条件形态展示。

Agnes在隔离目录 E:/grape_hackathon_G060_growth_20261014 实际执行生成、预测与核验，4次科学shell调用成功。原始幼果预测排除本串身份；修复幼果不声称独立预测。18对×2种输入共36个形态对照，12个变体仍有未解决互穿并照实展示；默认GROW_02两种输入均无互穿。修复版不是所有形态误差都更小。

101个科学输入/代码源文件保持SHA256一致，6项单元检查与真实浏览器检查通过，36组相对指标独立重算一致。相似变换只用于晚期参考对照，未据此修改预测；1.35倍扩张未标定，不是毫米或真实增长率。结果与原生会话见 evidence/growth-showcase-20261014，便携展示文件在data/showcase-20261014。之前Word/PPT/视频和ZIP仍对应其标注轮次，未将它们冒充本轮材料。

## 最新方向：200串联合训练与实拍迁移

按最新用户指令，直接读取原100串和新增100串数据，使用既定160训练（旧80＋新80）/20开发/10旧回归/10新形态评测划分。AGH在独立目录 `E:/grape_hackathon_G060_200_20261014` 实际重新训练48轮×64步，按开发集选择第24轮，原权重不覆盖。与上游本轮模型的102个权重张量逐项一致，文件序列化哈希不同。

本轮已用最新日期 `E:/grape_data_jurong/20260707/3` 的20个文件夹做真实照片迁移测试：重复源05/06和11/12合并后18组来源、216张视频帧入库、61张另拍照片查询。两侧都用本轮检测器从图像重新提取圆心；真实照片不进入训练或门限选择。闭集结果为20正确接受、2错误接受、39无法确认，首位候选正确42/61。开放集49张库内为17正确接受、2错误接受、30无法确认，12张库外均无法确认。仿真回归仍为库内20/20接受、库外20/20拒识。结果见 `evidence/combined200-real-20261014/实测报告.md` 与独立目录results。

10613个受保护源文件SHA256一致；1个上游报告生成脚本在快照后更新，没有被本轮计算引用。差异单独记录，原快照、权重、门限及计算输入保持冻结。首次汇总失败和独立后处理纠正均保留。此前100串、128维图网络、生长实验、Office材料和ZIP均保留为历史轮次，不能混用数字。

## 最新方向：纯仿真几何指纹

按10月14日最新任务对话，本轮指纹主线改为纯仿真：80串训练圆心检测、10串开发、10串评测；41串各72方向登记模板，JPEG查询重新检测后做几何匹配与拒识。图库使用仿真已知球心，查询使用模型预测圆心，颜色不进入匹配。与下文128维结构网络属于不同方法，不混报成绩。

AGH新会话“先把仿真葡萄指纹跑通”实际在隔离副本 `E:/grape_hackathon_G060_synthetic_20261014` 逐批复现已报告的40张查询，复用冻结权重与门限。对话和工具执行记录新增于 `evidence/synthetic-only-20261014`。这次是复现，不是新独立测试；本轮说明见副本README。此前Word/PPT/视频和ZIP仍对应下文旧实验，不冒充本轮材料。

实跑结果：库内20/20正确接受，库外20/20返回无法确认，0误接受；436/436对物理果粒对应审计正确。独立核对40份新推理输出与冻结SHA256通过。隔离副本演示为 http://127.0.0.1:8793/ ，原纯仿真页面8792保留。

## 2026年10月14日更新

最新入口为 http://127.0.0.1:8870/web/updates.html 。双击 Start-Demo.cmd 会打开新版。最新说明、路演、讲稿和视频采用 `_20261014` 后缀；下文13日的三维基线及其旧材料作为历史版本保留。

新版冻结原任务对话截至14日16:28已完成的训练成果：128维二维几何图指纹、仅实拍/仿真辅助的完整20张回归对照、20组长大形态展示和18组分组留一评价。研究过程的后续论文调查不在本次冻结快照中。输入、图网络、配套权重/图库与逐查询和逐折数值在 data/update-20261014，来源摘要在 manifest.json。检测后的冻结节点可以离线重新编码，不需重新检测或读取原图目录。

```powershell
python cli.py latest
python cli.py structure_evaluate
python cli.py growth_evaluate
python -m pip install -r requirements-updated.txt
python cli.py structure_case --case-index 0 --model-key real_only
python cli.py structure_case --case-index 0 --model-key mixed
```

普通页面与分数重算不需要PyTorch；实际重新编码使用 requirements-updated.txt 的CPU PyTorch。structure_case使用已经冻结的果粒位置，重新加载权重、生成128维向量并排名，不重新运行上游图像检测。完整原项目上传、节点编辑与新幼果预测入口仍为8790，需原项目服务运行；它们是本机附加功能，离线包本身提供冻结案例展示和受限工具。

新AGH会话为“葡萄项目进展：结构指纹和长大形态”，会话ID agnes:local:local-dev:g060:discussion:1791967058791。五轮自然讨论中的文件阅读、五次专业调用、错误解释和追问均保留在 evidence/review-20261014。没有删除或改写旧会话。科学结论以工具计算为准，模型早前出现的版本混淆以后续核对为准。

插件1.1.0新增 grape_latest、grape_structure_evaluate、grape_structure_case、grape_growth_evaluate，沿用固定命令和受限参数。旧六个工具继续可用。新增工具只读，不能向原科学项目写入。

当前2D结构回归首位：仅实拍9/20，混合6/20，未获得仿真提升。相同20张照片此前用于RGB实验，后来再用于结构比较时只能称回归。GROW10身份四项MAE均未超过均值；YG8身份两项轴比MAE较低，未做显著性检验。1.35倍中心扩张与端点动画是展示设定，不能代表真实尺寸或逐日生长。

## 2026年10月13日三维基线

G060 · 做的都队 · 硕博组。77个真实拍摄任务的球模型快照，73项进入正式检索评价，10个身份用于跨期预测。AGH + Agnes 3.0 Flash 真实执行专业分析，原生记录见 evidence。

当前结构检索 Top-1=3/47、Top-3=8/47；跨日期 Top-1=0/47。两个预测目标均未超过训练集均值基线。此作品是可复现实验原型，尚不能用于防伪或真实生长判断。无物理尺度，所有相对形态指标无量纲。

## 直接查看

本机双击 Start-Demo.cmd，浏览器打开 http://127.0.0.1:8870/web/ 。已生成的模型、图表和结果可直接查看，不需原始视频、不需 API Key。停止演示时关闭命令窗口。

其他机器在解压目录运行：

```powershell
python cli.py serve --port 8870
```

首次科学计算先创建独立环境：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe cli.py inventory
.venv\Scripts\python.exe cli.py describe --sample-id May_g2_07
.venv\Scripts\python.exe cli.py retrieve --sample-id May_g2_07
.venv\Scripts\python.exe cli.py evaluate
.venv\Scripts\python.exe cli.py forecast
.venv\Scripts\python.exe cli.py report
.venv\Scripts\python.exe cli.py serve
```

Python >=3.9；本机实际通过 grape1 Python 3.9.25 / NumPy 2.0.1 / SciPy 1.13.1 / sklearn 1.6.1 / matplotlib 3.9.2 验证。精确版本见 requirements-tested.txt。依赖文件是可复现环境说明，不包含环境本体。

data/samples.json 含冻结球心、半径、身份和来源摘要；data/cloud-preview.json 是原始点云的确定性抽样，非完整点云。report.py 在原路径不可用时使用抽样档案，其他机器仍能重算相同数值并查看模型。来源绝对路径仅用于追溯，不是重算的必需输入。

## 使用 AGH

已安装的本机 AGH：http://127.0.0.1:4177 。原专业会话：agnes:local:local-dev:g060:professional:1791889612755。

安装自己的环境时参考官方仓库 https://github.com/AgnesAI-Labs/agnes-harness 。本作品验证版本 commit dc267c3c78429933e2b3d8c5d544e56bb980a574，Node 24.16.0。先在 AGH 的设置中配置自己的 Agnes 账户，再运行插件。

1. 将 plugin 目录复制到 AGH 的包工作区内（本机为 .agh-grape-tools-g060）。本地包不能位于守护进程允许的包工作区之外。
2. 从 runtime-config.example.json 创建该目录的 runtime-config.json，填写 Python 可执行文件和作品根目录绝对路径，不填写密钥。
3. 设置 AGH_SOURCE、AGH_HOME、GRAPE_PLUGIN_SOURCE，运行 install-plugin.mjs。脚本检查来源、信任精确摘要并启用；失败会抛错，保留 operation 终态。
4. 在 AGH 源码目录（已安装依赖）执行：

```powershell
node --import tsx X:\作品路径\run-agent.mjs
node --import tsx X:\作品路径\run-agent.mjs --failure
```

仅当 AGH 默认账户属于 Agnes provider 时执行；primary 模型槽使用当前账户。首次 grape_report 文件写入需要允许一次，本脚本只自动许可该工具，其余写入不许可。

专用工具：grape_inventory / describe / retrieve / evaluate / forecast / report。参数只接受 sample_id，固定 Python CLI 计算；不接受任意路径或 shell 命令。report 只能写本作品 results。

## 数值方法和验证

geometry.py：球心平均中心化，非零两两距离中位数归一化；8维几何摘要加固定分箱分布组成72维描述。各块按维数平方根除权，基础摘要基线与结构描述共享该摘要块。

experiments.py：排除查询自身、完全相同模型、相同源视频和四个标签冲突条目；只有图库含独立同身份记录的查询进入分母。整体47次、同日期14次、跨日期47次。命中是按现有身份标签核对，没有逐粒真值。

跨期预测按 GROW 10个身份配对，留一身份验证，训练内 StandardScaler，固定 Ridge alpha=1。预测中位半径/果串尺度及最小/最大协方差特征值比，均值基线只使用每折训练目标。

```powershell
python -m pytest tests -q
node --test tests/plugin.test.mjs
python cli.py describe --sample-id unknown
```

最后一条是预期失败，JSON ok=false 且退出码2。测试覆盖整体变换不变性、非法输入、数据泄漏排除、留一身份训练范围、源文件保护、便携点云回退和插件权限元数据。

原始数据保护：证据 source-preservation.json 核对冻结来源与68份既有手动状态的 SHA256；本作品从未向源目录写入。球体保留现有中心/半径，移除椭球轴与旋转，仅作近似表示。

## 目录与提交

materials：项目说明、可编辑PPT、逐页讲稿、3–5分钟演示视频及公开内容草稿。results：真实实验数值、逐查询排名、逐折预测和图表。evidence：真实 AGH 执行与失败记录。

SUBMISSION_CHECKLIST.md 列出需队伍自行补齐的队长姓名、视频链接、公开内容链接和成员声明。本地制作完成不等于已发布或已提交比赛。
