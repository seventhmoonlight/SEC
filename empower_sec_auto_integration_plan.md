# Empower SEC 自动积分参数预测项目说明

## 1. 项目背景

当前项目位于医药公司 ASD 部门，目标是为 Waters Empower 中的 SEC 色谱积分提供 AI 辅助。

项目不希望 AI 直接替代 Empower 完成积分，而是希望 AI 根据新进来的原始色谱数据，推荐 Empower ApexTrack 积分所需的关键参数，再由 Empower 自身完成积分。

这种路线更适合药企环境，因为最终积分仍发生在 Empower 内部，便于审计、复现、验证和后续合规管理。

## 2. 当前目标

输入：

- SEC 原始色谱数据
- 文件格式目前包括 `.arw` 或 `.cdf`
- Python 已经可以读取原始数据

输出：

- `Peak Width`
- `Detection Threshold`
- `Minimum Height` 或类似 minimum 参数
- 预测置信度

最终使用方式：

- AI 生成推荐参数
- 参数导入或写回 Empower
- Empower 使用 ApexTrack 完成积分
- 最终关注积分后的面积占比是否接近人工确认结果

## 3. 成功标准

项目成功不应只看参数数值预测是否接近历史值，而应优先看使用预测参数后，Empower 积分得到的面积占比是否正确。

核心评价指标：

- AI 推荐参数导入 Empower 后的面积占比误差
- SEC 关键组分面积占比误差，例如聚体、主峰、片段等
- 是否减少人工调参和人工积分时间
- 是否能在新数据进入时稳定给出可用参数

辅助评价指标：

- `Peak Width` 预测误差
- `Detection Threshold` 预测误差
- `Minimum Height` 预测误差
- 预测置信度与真实表现是否匹配

## 4. 已知条件

- 当前方法类型先聚焦 SEC
- Empower 版本应为 Empower 3
- 主要积分算法为 ApexTrack
- Empower API 后续可读取核心数据，也可以写回参数
- 目前 API 尚未完全开放，所以当前阶段重点是完成主体参数预测模型
- 当前已有 170 多条历史数据，后续可继续导出
- 每条历史数据都有人工确认过的积分结果或对应参数
- 当前拥有原始数据和对应积分参数
- 当前没有明确区分原始自动积分结果和人工修正后的结果
- 当前暂时没有系统整理失败案例
- 数据差异不算很大，适合先做 SEC 单场景原型
- 当前技术栈以 Python 为主

## 5. 标签定义

这里的“历史标签”指模型训练时要学习的标准答案。

本项目至少有两类标签：

### 5.1 参数标签

一条色谱图对应人工确认后使用的 Empower 参数：

- `Peak Width`
- `Detection Threshold`
- `Minimum Height`

这是当前最直接可用的监督学习标签。

### 5.2 结果标签

一条色谱图对应人工确认后的 SEC 面积占比结果，例如：

- HMW% / Aggregate%
- Main Peak% / Monomer%
- LMW% / Fragment%

结果标签对最终评估非常重要，因为两个不同参数组合可能都能得到合格的面积占比。

## 6. 推荐实现路线

当前阶段优先选择路线 B：

原始色谱图 -> AI 推荐 Empower 参数 -> Empower 完成积分 -> 检查面积占比

不建议当前阶段直接做 AI 端完整积分，因为：

- 药企环境更容易接受 Empower 内部积分结果
- Empower 结果链更容易审计
- 直接积分需要处理更多基线和峰边界合规问题
- 当前数据量 170 多条，更适合参数预测与相似案例检索

## 6.1 当前建模决策

为避免历史离散参数组合影响模型泛化，当前主模型不使用 `profile_id` 或任何 profile 组合训练。

明确摒弃：

- 不把 `profile_id` 作为预测目标
- 不训练“色谱图 -> 历史参数组合”的分类模型
- 不将 `profile_training_count`、`duplicate_group_size`、`duplicate_dirs` 作为模型输入
- 不将回归结果强制 snap 到历史 profile 组合

当前主线固定为连续参数预测：

```text
原始 SEC 色谱图 -> Peak Width
原始 SEC 色谱图 -> Detection Threshold
原始 SEC 色谱图 -> Minimum Height
```

profile 相关字段只用于数据审计和理解标签分布，不参与训练和预测。

## 6.2 参数输出范围

根据当前项目约束，模型输出需要做合理范围硬约束：

- `Peak Width`: 15-300
- `Detection Threshold`: 0-60
- `Minimum Height`: 暂时按训练数据范围约束，后续确认 Empower 方法范围后再固定

这些范围约束只用于防止模型输出 Empower 不合理参数，不等同于历史模板限制。

## 6.3 Integration Start / End 规则

后续写回 Empower 时，除 ApexTrack 参数外，也需要输出积分起止时间。

当前 SEC 样本主峰位置主要分为两类：

- 短 RT 类型：主峰约在 4-5 min
- 长 RT 类型：主峰约在 17-18 min

因此当前先采用规则法，而不是训练模型预测积分窗口：

```text
如果 main_peak_rt < 10 min:
    Integration Start = 2.5
    Integration End = 7.2
否则:
    Integration Start = 9.0
    Integration End = 23.0
```

该规则已写入 `scripts/predict_sec_parameters.py`。预测输出会包含：

- `Integration Start`
- `Integration End`
- `main_peak_rt`
- `integration_window_type`

## 7. 模型总体策略

为了尽快投入使用，模型不能只对见过的数据有效，必须能对新进来的 SEC 色谱图泛化。

推荐采用三层组合方案：

### 7.1 规则基线模型

根据色谱图的基础特征直接估算参数：

- 根据主峰宽度、半高宽、采样间隔估算 `Peak Width`
- 根据噪声水平、基线波动估算 `Detection Threshold`
- 根据噪声峰高度、小峰分布估算 `Minimum Height`

规则模型的作用：

- 提供可解释兜底
- 在样本量较小时保持稳定
- 作为机器学习模型的 sanity check

### 7.2 机器学习回归模型

使用特征工程加传统回归模型预测参数。

优先模型：

- Random Forest
- ExtraTrees
- XGBoost
- LightGBM

暂不建议一开始使用深度学习模型，因为当前数据量较小，深度学习更容易过拟合，解释性也较弱。

### 7.3 相似案例检索

对新色谱图提取特征后，在历史库中寻找最相似的若干条图谱。

输出可以参考：

- 相似历史图谱使用过的参数
- 相似图谱的面积占比表现
- 相似度加权平均参数

相似案例检索的价值：

- 小数据场景下通常更稳
- 方便向分析员解释推荐依据
- 能帮助判断置信度

## 8. 色谱特征工程

需要从 `.arw` 或 `.cdf` 中读取时间和响应值，并提取可用于预测参数的特征。

建议特征包括：

- 运行时间范围
- 采样间隔
- 数据点数
- 响应值最大值、最小值、均值、中位数
- 基线噪声标准差
- 基线噪声 MAD
- 起点到终点的基线漂移
- 平滑后曲线的一阶导数统计量
- 平滑后曲线的二阶导数统计量
- 粗略峰数量
- 主峰保留时间
- 主峰高度
- 主峰半高宽
- 主峰峰底宽
- 主峰对称性或拖尾因子近似值
- 小峰数量
- 局部极大值数量
- 信噪比
- 目标窗口内积分面积粗估值
- 早出峰区域面积占比
- 主峰区域面积占比
- 晚出峰区域面积占比

这些特征与 Empower 参数的关系：

- `Peak Width` 主要由真实峰宽、半高宽、峰底宽、采样间隔决定
- `Detection Threshold` 主要由噪声水平、基线波动、信噪比决定
- `Minimum Height` 主要由噪声峰、小峰和低丰度峰分布决定

## 9. 置信度设计

模型输出不能只有参数，还需要给出置信度。

置信度可以由以下因素组成：

- 新图谱与历史图谱的相似度
- 多个模型预测结果是否一致
- 预测参数是否位于历史参数常见范围内
- 当前图谱是否存在异常噪声或异常基线漂移
- 当前图谱的峰形是否明显偏离训练集
- 交叉验证中相似样本的历史误差

高置信度场景：

- 与历史样本相似
- 回归模型和相似案例模型预测接近
- 参数位于历史合理范围内
- 基线和噪声正常

低置信度场景：

- 找不到相似历史样本
- 不同模型预测分歧大
- 参数明显超出历史范围
- 色谱图出现异常噪声、漂移、极端峰形

低置信度结果应提示人工复核，而不是直接用于自动导入。

## 10. 训练与验证策略

由于当前只有 170 多条数据，推荐使用：

- K-fold 交叉验证
- 留一法验证
- 按项目、方法、批次或日期分组验证

需要特别注意，不能只随机拆分后看参数误差。为了确认模型真的能用于新数据，最好使用更接近真实上线的验证方式：

- 用较早的数据训练，较新的数据测试
- 用部分批次训练，未见过批次测试
- 用部分样品类型训练，未见过样品测试

这样更能判断模型是否能处理真实新进来的数据。

## 11. 推荐输出格式

模型预测结果建议输出为结构化数据，例如：

```json
{
  "sample_id": "SEC_001",
  "algorithm": "ApexTrack",
  "Peak Width": 0.12,
  "Detection Threshold": 18.5,
  "Minimum Height": 1200,
  "confidence": 0.82,
  "model_version": "sec_param_model_v1",
  "note": "Prediction based on chromatogram features and similar historical samples."
}
```

后续可生成 Empower 可导入的 CSV 或通过 API 写回 Empower。

## 12. GitHub 高星项目的使用原则

如果后续开发中遇到没有思路或需要成熟实现的部分，可以考虑 GitHub 上的高星项目，但必须符合项目目标。

允许参考或使用的方向：

- 色谱或质谱数据读取库
- NetCDF / CDF 解析库
- 峰检测算法库
- 基线校正算法库
- 时间序列特征提取库
- 传统机器学习建模框架
- 模型解释性工具

使用前必须判断：

- 是否能处理本项目的数据格式
- 是否适合 SEC 色谱场景
- 是否能部署到公司允许的 Python 环境
- license 是否允许企业内部使用
- 是否会增加不必要的复杂度
- 是否对最终面积占比准确性有帮助

不应为了使用热门项目而引入复杂依赖。项目目标是尽快产出可靠的 Empower 参数推荐能力。

## 13. MVP 优先级

第一阶段目标是尽快做出可用原型。

### MVP 1

- 读取 `.arw` 或 `.cdf`
- 提取基础色谱特征
- 训练 `Peak Width`、`Detection Threshold`、`Minimum Height` 回归模型
- 输出预测参数和置信度
- 使用交叉验证评估参数误差

### MVP 2

- 增加相似案例检索
- 增加规则基线模型
- 将回归模型、规则模型、相似案例模型融合
- 输出更稳定的最终推荐参数

### MVP 3

- 将预测参数导入 Empower
- 由 Empower 完成积分
- 比较 AI 参数积分后的面积占比与人工确认面积占比
- 按面积占比误差重新优化模型

### MVP 4

- 增加低置信度识别
- 增加异常图谱识别
- 生成面向工作人员的参数推荐表
- 准备后续 API 集成和写回 Empower

## 14. 当前最关键的问题

当前项目主要卡在：

如何根据一条新的 SEC 原始色谱图，可靠预测 Empower ApexTrack 的积分参数。

下一步应优先完成：

- 整理训练数据表
- 确认每条数据对应的三个参数标签
- 确认是否有人工确认面积占比
- 编写原始数据读取和特征提取代码
- 训练第一个传统机器学习模型
- 用严格验证方式确认对新数据的预测能力

## 14.1 当前第一版训练结果

已基于 `DA/sec_training_data_master.csv` 完成第一版连续回归训练。

当前训练原则：

- 使用 173 条已核查训练数据
- 使用 `.arw` 原始时间-响应曲线
- 提取 59 个曲线统计和峰形特征
- 不使用 `profile_id`
- 不使用 `profile_training_count`
- 不使用 `duplicate_group_size`
- 不使用 `duplicate_dirs`
- 不将预测结果 snap 到历史参数组合
- 面积占比字段仅保留在输出表中用于后续验证，不参与当前训练

当前模型：

- 最佳模型：`ExtraTreesRegressor`
- 验证方式：按 `arw_md5` 分组的 5 折交叉验证
- 训练产物：`outputs/models/sec_parameter_model.pkl`
- 验证指标：`outputs/tables/sec_parameter_cv_metrics.csv`
- 交叉验证预测明细：`outputs/tables/sec_parameter_cv_predictions.csv`

当前交叉验证结果：

```text
Peak Width MAE: 5.1169
Detection Threshold MAE: 4.7359
Minimum Height MAE: 4.6995
Mean target MAE: 4.8507
```

已提供新数据预测脚本：

```text
python scripts\predict_sec_parameters.py <arw-file-or-sample-dir>
```

示例输出字段：

- `Peak Width`
- `Detection Threshold`
- `Minimum Height`
- `confidence`
- `model_note`

当前置信度基于树模型内部预测分歧和边界风险估算，后续应结合 Empower 积分后的面积占比误差继续校准。

## 15. 建议训练数据表结构

建议每条色谱图一行：

```text
sample_id
raw_file_path
method_name
channel
run_time
Peak Width
Detection Threshold
Minimum Height
HMW_percent
Main_percent
LMW_percent
manual_confirmed
source_project
date_or_batch
```

如果暂时没有面积占比字段，可以先训练参数预测模型，但后续需要补充面积占比用于真正评估模型价值。

## 16. 后续需要继续确认的信息

- SEC 中具体关注的组分命名和面积占比口径
- `Minimum Height` 是否一定使用，还是只在部分方法中使用
- 参数单位和 Empower 导入格式
- `.arw` 与 `.cdf` 文件中时间、响应值、通道信息的读取方式
- 当前 170 多条数据是否来自同一个方法或多个相近方法
- 是否存在不同产品、不同柱子、不同批次带来的分布差异
- 是否能导出人工确认后的峰表和面积占比
- 后续 API 写回 Empower 的字段和权限
- 低置信度结果如何交给工作人员复核
