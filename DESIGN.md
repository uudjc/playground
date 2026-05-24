# TF-Playground 扩展设计说明

## 1. 目标

本次扩展在原始 TF-Playground 的手写 TypeScript 神经网络中增加：

- Normalization：`BatchNorm`、`LayerNorm`
- Optimizer：`Adam`、`Muon`
- 对应 UI 配置、URL 状态保存和最小测试

项目没有引入 TensorFlow.js 或其他深度学习框架，所有训练逻辑仍在 `src/nn.ts` 中完成。

## 2. 原始架构

原项目的核心结构如下：

- `src/nn.ts`
  - `Node[][]` 表示网络层级结构
  - `Node` 表示神经元
  - `Link` 表示两个神经元之间的权重
  - `forwardProp()` 负责单样本前向传播
  - `backProp()` 负责原有单样本反向传播
  - `updateWeights()` 负责参数更新

- `src/playground.ts`
  - `constructInput()` 根据用户选择的 feature 生成输入向量
  - `oneStep()` 执行一个 epoch 的训练
  - `reset()` 根据当前 UI 状态重新构建网络

- `src/state.ts`
  - 管理 UI 状态
  - 负责 URL hash 序列化和反序列化

- `index.html`
  - 定义顶部控制栏

本次改动保持这些边界不变，只在必要位置补充 normalization 和 optimizer。

## 3. 样本与张量形状

原始数据样本来自 `Example2D`，逻辑形状为：

```ts
{
  x: number,
  y: number,
  label: number
}
```

训练前，`src/playground.ts` 中的 `constructInput(x, y)` 会把二维点转换为网络输入向量：

```ts
number[]
```

输入长度取决于用户选择的 feature。默认只启用 `x` 和 `y`，所以单样本输入为：

```ts
[x, y]  // shape: [2]
```

如果启用更多 feature，输入向量会扩展为：

```ts
[x, y, x*x, y*y, x*y, sin(x), sin(y)]  // shape: [numFeatures]
```

批量训练入口 `trainBatch()` 接收：

```ts
inputs: number[][]  // shape: [batchSize, numFeatures]
targets: number[]   // shape: [batchSize]
```

某个隐藏层的中间激活可以理解为：

```ts
shape: [batchSize, numNodesInLayer]
```

这一点决定了两类 normalization 的统计维度：

- `BatchNorm`：按列统计，即每个节点跨 batch 统计均值和方差
- `LayerNorm`：按行统计，即每个样本在同一层内跨节点统计均值和方差

## 4. Normalization 设计

### 4.1 作用范围

`BatchNorm` 和 `LayerNorm` 只作用于隐藏层，不作用于输入层和输出层。

代码位置：`src/nn.ts` 的 `forwardProp()` 和 `forwardBatch()` 中通过：

```ts
let isOutputLayer = layerIdx === network.length - 1;
```

判断输出层。输出层会跳过 normalization，保持分类和回归输出语义稳定。

网络结构：

```text
input -> hidden1 -> hidden2 -> output
```

选择 normalization 后实际计算为：

```text
input
  -> hidden1: rawInput -> normalization -> activation
  -> hidden2: rawInput -> normalization -> activation
  -> output: rawInput -> activation
```

### 4.2 Node 状态扩展

核心代码：`src/nn.ts` 的 `Node` 类。

新增字段：

- `rawInput`：归一化前的线性输入
- `normalizedInput`：归一化后的值
- `normGamma`：可学习缩放参数，初始值为 `1`
- `normBeta`：可学习平移参数，初始值为 `0`
- `runningMean`：BatchNorm 推理用滑动均值
- `runningVariance`：BatchNorm 推理用滑动方差
- `accNormGammaDer` / `accNormBetaDer`：`gamma` / `beta` 的累计梯度

普通前向路径原来直接计算：

```text
totalInput = bias + sum(weight * source.output)
output = activation(totalInput)
```

现在拆成：

```text
rawInput = bias + sum(weight * source.output)
normalizedInput = normalize(rawInput)
totalInput = gamma * normalizedInput + beta
output = activation(totalInput)
```

当 normalization 为 `NONE` 或当前层是输出层时，仍走原来的直接路径。

### 4.3 BatchNorm 前向

核心代码：`src/nn.ts` 的 `forwardBatch()`。

BatchNorm 必须基于 batch 统计，因此训练入口从原来的逐样本：

```ts
forwardProp();
backProp();
updateWeights();
```

改为批量入口：

```ts
trainBatch();
```

在 `forwardBatch()` 中，对于某一隐藏层的某个节点：

1. 收集当前 batch 所有样本在该节点上的 `rawInput`
2. 计算 batch 均值 `mean`
3. 计算 batch 方差 `variance`
4. 计算：

```text
normalizedInput = (rawInput - mean) / sqrt(variance + epsilon)
totalInput = gamma * normalizedInput + beta
output = activation(totalInput)
```

同时更新滑动统计：

```text
runningMean = momentum * runningMean + (1 - momentum) * mean
runningVariance = momentum * runningVariance + (1 - momentum) * variance
```

当前常量：

```ts
NORMALIZATION_EPSILON = 1e-5
BATCH_NORM_MOMENTUM = 0.9
```

### 4.4 BatchNorm 推理 / 可视化前向

核心代码：`src/nn.ts` 的 `forwardProp()`。

`forwardProp()` 仍用于 loss 计算和决策边界可视化。因为它是单样本前向，不适合现场计算 batch 统计，所以 BatchNorm 在这里使用：

- `runningMean`
- `runningVariance`

这保证训练后可视化时不会因为单样本方差不可用而失效。

### 4.5 LayerNorm 前向

核心代码：

- 单样本路径：`src/nn.ts` 的 `forwardProp()`
- 批量训练路径：`src/nn.ts` 的 `forwardBatch()`

LayerNorm 不依赖 batch 维度。对于某个样本、某个隐藏层：

1. 收集该层所有节点的 `rawInput`
2. 在这一层内部计算均值和方差
3. 对该样本该层的每个节点做归一化

公式同样是：

```text
normalizedInput = (rawInput - mean) / sqrt(variance + epsilon)
totalInput = gamma * normalizedInput + beta
output = activation(totalInput)
```

区别在于统计维度是“同一个样本的一整层节点”，不是 batch。

### 4.6 Normalization 反向传播

核心代码：`src/nn.ts` 的 `trainBatch()`。

`trainBatch()` 会保存每个样本、每个节点的中间状态：

```ts
interface TrainingNodeState {
  rawInput: number;
  normalizedInput: number;
  totalInput: number;
  output: number;
  outputDer: number;
  inputDer: number;
  rawInputDer: number;
}
```

反向传播时：

1. 先从 loss 得到输出层梯度
2. 对隐藏层先算激活函数梯度
3. 对 `gamma` / `beta` 累计梯度：

```text
dGamma += dTotalInput * normalizedInput
dBeta += dTotalInput
```

4. 再把梯度从 normalized 空间还原到 `rawInputDer`

LayerNorm 按“单样本的一层节点”还原梯度。  
BatchNorm 按“一个节点跨 batch 的样本集合”还原梯度。

最终 `rawInputDer` 会继续用于：

- 累计 bias 梯度
- 累计 link weight 梯度
- 传回上一层节点的 `outputDer`

## 5. Optimizer 设计

### 5.1 统一更新入口

核心代码：`src/nn.ts` 的 `updateWeights()`。

参数更新统一通过：

```ts
updateWeights(network, learningRate, regularizationRate, optimizer)
```

支持：

- `SGD`
- `ADAM`
- `MUON`

`src/playground.ts` 的 `oneStep()` 会把当前 UI 选择的 optimizer 传入 `trainBatch()`，再由 `trainBatch()` 调用 `updateWeights()`。

### 5.2 Adam

核心代码：

- `ScalarOptimizerState`
- `applyScalarOptimizer()`

Adam 为每个标量参数维护：

- `t`
- `m`
- `v`

覆盖的参数包括：

- `Link.weight`
- `Node.bias`
- `Node.normGamma`
- `Node.normBeta`

当前实现不再把 Adam 超参数写死为文件级常量，而是通过
`OptimizerHyperparameters` 传入：

```ts
{
  adamBeta1: number,
  adamBeta2: number,
  adamEpsilon: number
}
```

并带偏差修正：

```text
mHat = m / (1 - beta1^t)
vHat = v / (1 - beta2^t)
param -= learningRate * mHat / (sqrt(vHat) + epsilon)
```

### 5.3 Muon

核心代码：

- `updateLayerWeightsWithMuon()`
- `orthogonalize()`
- `matrixMultiply()`
- `matrixTranspose()`

Muon 适合二维权重矩阵。原项目中权重存储为逐条 `Link`，因此实现时按层临时组装矩阵：

```text
shape: [numNodesInCurrentLayer, numNodesInPreviousLayer]
```

处理流程：

1. 收集该层所有 `Link.weight` 的梯度
2. 对每个 link 维护动量 `muonMomentum`
3. 把动量组织成矩阵
4. 使用 Newton-Schulz 迭代近似正交化
5. 用正交化后的矩阵方向更新该层权重

当前 Muon 只用于隐藏层权重矩阵。输出层、bias、`gamma`、`beta` 等标量参数仍走 `applyScalarOptimizer()` 的标量路径。

Muon 相关超参数同样通过状态注入，而不是写死常量：

```ts
{
  muonMomentum: number,
  muonEpsilon: number
}
```

## 6. UI 与状态设计

### 6.1 `src/state.ts`

新增映射：

```ts
normalizations: {
  none,
  batchnorm,
  layernorm
}

optimizers: {
  sgd,
  adam,
  muon
}
```

新增状态：

```ts
normalization = nn.Normalization.NONE
optimizer = nn.Optimizer.SGD
normalizationEpsilon = 1e-5
batchNormMomentum = 0.9
adamBeta1 = 0.9
adamBeta2 = 0.999
adamEpsilon = 1e-8
muonMomentum = 0.95
muonEpsilon = 1e-7
```

并加入 `State.PROPS`，因此可以通过 URL hash 保存和恢复。

### 6.2 `index.html`

顶部控制栏新增 normalization / optimizer 选择和对应的超参数控件。

基础下拉框：

- `Normalization`: `None / BatchNorm / LayerNorm`
- `Optimizer`: `SGD / Adam / Muon`

超参数控件按选择结果条件显示：

- 选择 `BatchNorm` 时显示：
  - `Normalization epsilon`
  - `BatchNorm momentum`
- 选择 `LayerNorm` 时显示：
  - `Normalization epsilon`
- 选择 `Adam` 时显示：
  - `Adam beta1`
  - `Adam beta2`
  - `Adam epsilon`
- 选择 `Muon` 时显示：
  - `Muon momentum`
  - `Muon epsilon`

未选择对应算法时，不显示对应控件。

默认选项仍是：

- `None`
- `SGD`

保证不选择新功能时，行为尽量保持原 baseline。

### 6.3 `src/playground.ts`

核心接线：

- `makeGUI()` 绑定 normalization 和 optimizer 下拉框
- `bindNumberControl()` 绑定超参数下拉框
- `updateHyperparameterControls()` 根据当前选择显示或隐藏超参数控件
- `reset()` 构建网络时把 `state.normalization` 和 normalization 超参数传入 `nn.buildNetwork()`
- `oneStep()` 按 `state.batchSize` 收集 batch，并调用 `nn.trainBatch()`，同时传入 optimizer 超参数

`oneStep()` 的训练流变为：

```text
收集 batchInputs / batchTargets
  -> nn.trainBatch(...)
 -> 剩余不足 batchSize 的样本也训练一次
  -> 计算 train/test loss
  -> updateUI()
```

### 6.4 超参数对象

为了避免在训练核心中直接依赖 UI 状态，`src/playground.ts` 中增加了两个转换函数：

- `getNormalizationHyperparameters()`
- `getOptimizerHyperparameters()`

它们把 `state` 中的超参数转换为训练层所需的对象：

```ts
NormalizationHyperparameters
OptimizerHyperparameters
```

训练层只接收这些对象，不直接读取 UI。

## 7. 测试

### 7.1 测试设计

新增测试文件：

```text
test/nn_smoke_test.js
```

测试不依赖浏览器和 DOM。它直接读取 `src/nn.ts`，用 TypeScript 转译后在 Node 中执行。

覆盖组合：

- `SGD + None`
- `Adam + LayerNorm`
- `Adam + BatchNorm`
- `Muon + LayerNorm`

每个组合会检查：

- 训练后权重发生变化
- 前向输出为有限数
- bias、weight、gamma、beta 均为有限数
- BatchNorm 的 running statistics 会更新

由于超参数现在是可配置的，测试同时覆盖了：

- normalization 和 optimizer 默认超参数路径仍可运行
- 训练核心在接收外部超参数对象后仍能稳定收敛到有限数值

`package.json` 新增：

```json
"test": "node test/nn_smoke_test.js"
```

### 7.2 测试命令

首次运行或清理过 `node_modules` 后，先安装锁定版本依赖：

```bash
cd /home/user/playground
npm ci
```

运行新增的神经网络 smoke test：

```bash
cd /home/user/playground
npm test
```

该命令会执行：

```bash
node test/nn_smoke_test.js
```

### 7.3 测试结果

已执行：

```bash
npm test
```

结果：

- `npm test` 通过

## 8. 构建与运行

### 8.1 构建命令

生成正式静态产物：

```bash
cd /home/user/playground
npm run build
```

构建产物输出到：

```text
dist/
```

### 8.2 正式运行命令

构建完成后，从 `dist/` 目录启动静态服务：

```bash
cd /home/user/playground/dist
python3 -m http.server 5000
```

浏览器访问：

```text
http://localhost:5000/
```

也可以使用项目原始脚本：

```bash
cd /home/user/playground
npm run serve
```

注意：当前环境中 `npm run serve` 依赖的 `serve` 包在 Node 24 下可能触发 `uv_interface_addresses` 系统错误；如果遇到该错误，使用上面的 `python3 -m http.server 5000` 方式运行构建产物。

### 8.3 构建验证记录

已执行：

```bash
npm run build
```

结果：

- `npm run build` 通过
