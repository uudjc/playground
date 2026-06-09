const fs = require("fs");
const path = require("path");
const vm = require("vm");

let ts;
try {
  ts = require("typescript");
} catch (err) {
  throw new Error("Run npm install before npm test; this test needs TypeScript.");
}

const sourcePath = path.join(__dirname, "..", "src", "nn.ts");
const source = fs.readFileSync(sourcePath, "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2017
  }
});

const moduleStub = {exports: {}};
vm.runInNewContext(compiled.outputText, {
  module: moduleStub,
  exports: moduleStub.exports,
  require,
  console,
  Math,
  Error
});

const nn = moduleStub.exports;

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function assertApprox(actual, expected, epsilon, message) {
  if (Math.abs(actual - expected) > epsilon) {
    throw new Error(
        `${message}: expected ${expected}, got ${actual}`);
  }
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function getWeights(network) {
  const weights = [];
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    for (const node of network[layerIdx]) {
      for (const link of node.inputLinks) {
        weights.push(link.weight);
      }
    }
  }
  return weights;
}

function assertFiniteNetwork(network, label) {
  for (const layer of network) {
    for (const node of layer) {
      assert(isFiniteNumber(node.bias), `${label}: non-finite bias`);
      assert(isFiniteNumber(node.normGamma), `${label}: non-finite gamma`);
      assert(isFiniteNumber(node.normBeta), `${label}: non-finite beta`);
      for (const link of node.inputLinks) {
        assert(isFiniteNumber(link.weight), `${label}: non-finite weight`);
      }
    }
  }
}

function makeControlledNetwork(normalization, normalizationHyperparameters) {
  const network = nn.buildNetwork(
      [2, 2, 1],
      nn.Activations.LINEAR,
      nn.Activations.LINEAR,
      null,
      ["x", "y"],
      true,
      normalization,
      normalizationHyperparameters);
  const firstHidden = network[1][0];
  firstHidden.inputLinks[0].weight = 1;
  firstHidden.inputLinks[1].weight = 0;
  const secondHidden = network[1][1];
  secondHidden.inputLinks[0].weight = 0;
  secondHidden.inputLinks[1].weight = 1;
  const output = network[2][0];
  output.inputLinks[0].weight = 0.5;
  output.inputLinks[1].weight = -0.25;
  return network;
}

function trainCase(label, normalization, optimizer) {
  const network = nn.buildNetwork(
      [2, 3, 2, 1],
      nn.Activations.TANH,
      nn.Activations.TANH,
      null,
      ["x", "y"],
      false,
      normalization);
  const before = getWeights(network);
  const inputs = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
  const targets = [-1, 1, 1, -1];
  for (let i = 0; i < 10; i++) {
    nn.trainBatch(network, inputs, targets, nn.Errors.SQUARE, 0.01, 0,
        optimizer);
  }
  const after = getWeights(network);
  assert(after.some((weight, i) => Math.abs(weight - before[i]) > 1e-12),
      `${label}: weights did not change`);
  for (const input of inputs) {
    assert(isFiniteNumber(nn.forwardProp(network, input)),
        `${label}: non-finite output`);
  }
  assertFiniteNetwork(network, label);
  if (normalization === nn.Normalization.BATCH_NORM) {
    const firstHiddenNode = network[1][0];
    assert(firstHiddenNode.runningVariance !== 1,
        `${label}: batch norm running stats did not update`);
  }
}

function assertLayerNormUsesLayerAxis() {
  const network = makeControlledNetwork(
      nn.Normalization.LAYER_NORM,
      {epsilon: 0, batchNormMomentum: 0});
  nn.forwardProp(network, [1, 3]);
  assertApprox(network[1][0].normMean, 2, 1e-12,
      "layer norm mean should be computed inside one layer");
  assertApprox(network[1][0].normVariance, 1, 1e-12,
      "layer norm variance should be computed inside one layer");
  assertApprox(network[1][0].normalizedInput, -1, 1e-12,
      "first layer norm activation");
  assertApprox(network[1][1].normalizedInput, 1, 1e-12,
      "second layer norm activation");
  assertApprox(network[2][0].totalInput, network[2][0].rawInput, 1e-12,
      "output layer should skip normalization");
}

function assertBatchNormUsesBatchAxis() {
  const network = makeControlledNetwork(
      nn.Normalization.BATCH_NORM,
      {epsilon: 0, batchNormMomentum: 0});
  nn.trainBatch(network, [[1, 10], [3, 14]], [0, 0],
      nn.Errors.SQUARE, 0, 0, nn.Optimizer.SGD);
  assertApprox(network[1][0].runningMean, 2, 1e-12,
      "batch norm mean should be computed per node across the batch");
  assertApprox(network[1][0].runningVariance, 1, 1e-12,
      "batch norm variance should be computed per node across the batch");
  assertApprox(network[1][1].runningMean, 12, 1e-12,
      "batch norm mean should not mix different hidden nodes");
  assertApprox(network[1][1].runningVariance, 4, 1e-12,
      "batch norm variance should not mix different hidden nodes");
  assertApprox(network[2][0].runningMean, 0, 1e-12,
      "output layer should not update batch norm running mean");
  assertApprox(network[2][0].runningVariance, 1, 1e-12,
      "output layer should not update batch norm running variance");
}

function assertAdamTracksScalarMoments() {
  const network = makeControlledNetwork(nn.Normalization.NONE);
  nn.trainBatch(network, [[1, 2]], [1],
      nn.Errors.SQUARE, 0.01, 0, nn.Optimizer.ADAM);
  const hiddenLink = network[1][0].inputLinks[0];
  const outputNode = network[2][0];
  assert(hiddenLink.optimizerState.t === 1,
      "adam should advance per-weight optimizer state");
  assert(outputNode.biasOptimizerState.t === 1,
      "adam should advance bias optimizer state");
  assert(Math.abs(hiddenLink.optimizerState.m) > 0,
      "adam first moment should receive a non-zero gradient");
  assert(Math.abs(hiddenLink.optimizerState.v) > 0,
      "adam second moment should receive a non-zero gradient");
}

function assertMuonUsesMatrixPathForHiddenWeights() {
  const network = makeControlledNetwork(nn.Normalization.NONE);
  nn.trainBatch(network, [[1, 2]], [1],
      nn.Errors.SQUARE, 0.01, 0, nn.Optimizer.MUON);
  const hiddenLink = network[1][0].inputLinks[0];
  const outputLink = network[2][0].inputLinks[0];
  assert(Math.abs(hiddenLink.muonMomentum) > 0,
      "muon should maintain momentum for hidden matrix weights");
  assert(hiddenLink.optimizerState.t === 0,
      "hidden matrix weights should skip the scalar optimizer path");
  assert(outputLink.optimizerState.t === 1,
      "output weights should still use the scalar optimizer path");
}

trainCase("sgd/no-normalization", nn.Normalization.NONE, nn.Optimizer.SGD);
trainCase("adam/layernorm", nn.Normalization.LAYER_NORM, nn.Optimizer.ADAM);
trainCase("adam/batchnorm", nn.Normalization.BATCH_NORM, nn.Optimizer.ADAM);
trainCase("muon/layernorm", nn.Normalization.LAYER_NORM, nn.Optimizer.MUON);
assertLayerNormUsesLayerAxis();
assertBatchNormUsesBatchAxis();
assertAdamTracksScalarMoments();
assertMuonUsesMatrixPathForHiddenWeights();

console.log("nn smoke tests passed");
