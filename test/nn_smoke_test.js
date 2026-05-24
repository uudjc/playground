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

trainCase("sgd/no-normalization", nn.Normalization.NONE, nn.Optimizer.SGD);
trainCase("adam/layernorm", nn.Normalization.LAYER_NORM, nn.Optimizer.ADAM);
trainCase("adam/batchnorm", nn.Normalization.BATCH_NORM, nn.Optimizer.ADAM);
trainCase("muon/layernorm", nn.Normalization.LAYER_NORM, nn.Optimizer.MUON);

console.log("nn smoke tests passed");
