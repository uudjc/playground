/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * A node in a neural network. Each node has a state
 * (total input, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
export class Node {
  id: string;
  /** List of input links. */
  inputLinks: Link[] = [];
  bias = 0.1;
  /** List of output links. */
  outputs: Link[] = [];
  /** Total input before optional normalization. */
  rawInput: number;
  totalInput: number;
  output: number;
  normalizedInput = 0;
  normMean = 0;
  normVariance = 1;
  normStdInv = 1;
  normGamma = 1;
  normBeta = 0;
  runningMean = 0;
  runningVariance = 1;
  /** Error derivative with respect to this node's output. */
  outputDer = 0;
  /** Error derivative with respect to this node's total input. */
  inputDer = 0;
  /** Error derivative with respect to pre-normalization input. */
  rawInputDer = 0;
  /**
   * Accumulated error derivative with respect to this node's total input since
   * the last update. This derivative equals dE/db where b is the node's
   * bias term.
   */
  accInputDer = 0;
  accNormGammaDer = 0;
  accNormBetaDer = 0;
  numAccumulatedNormDers = 0;
  /**
   * Number of accumulated err. derivatives with respect to the total input
   * since the last update.
   */
  numAccumulatedDers = 0;
  /** Activation function that takes total input and returns node's output */
  activation: ActivationFunction;
  biasOptimizerState: ScalarOptimizerState = {t: 0, m: 0, v: 0};
  normGammaOptimizerState: ScalarOptimizerState = {t: 0, m: 0, v: 0};
  normBetaOptimizerState: ScalarOptimizerState = {t: 0, m: 0, v: 0};

  /**
   * Creates a new node with the provided id and activation function.
   */
  constructor(id: string, activation: ActivationFunction, initZero?: boolean) {
    this.id = id;
    this.activation = activation;
    if (initZero) {
      this.bias = 0;
    }
  }

  computeTotalInput(): number {
    // Stores total input into the node.
    this.rawInput = this.bias;
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      this.rawInput += link.weight * link.source.output;
    }
    return this.rawInput;
  }

  /** Recomputes the node's output and returns it. */
  updateOutput(): number {
    this.totalInput = this.computeTotalInput();
    this.output = this.activation.output(this.totalInput);
    return this.output;
  }
}

/**
 * An error function and its derivative.
 */
export interface ErrorFunction {
  error: (output: number, target: number) => number;
  der: (output: number, target: number) => number;
}

/** A node's activation function and its derivative. */
export interface ActivationFunction {
  output: (input: number) => number;
  der: (input: number) => number;
}

/** Function that computes a penalty cost for a given weight in the network. */
export interface RegularizationFunction {
  output: (weight: number) => number;
  der: (weight: number) => number;
}

export enum Normalization {
  NONE,
  BATCH_NORM,
  LAYER_NORM
}

export enum Optimizer {
  SGD,
  ADAM,
  MUON
}

interface ScalarOptimizerState {
  t: number;
  m: number;
  v: number;
}

interface TrainingNodeState {
  rawInput: number;
  normalizedInput: number;
  totalInput: number;
  output: number;
  outputDer: number;
  inputDer: number;
  rawInputDer: number;
}

interface NeuralNetwork extends Array<Node[]> {
  normalization?: Normalization;
}

const NORMALIZATION_EPSILON = 1e-5;
const BATCH_NORM_MOMENTUM = 0.9;
const ADAM_BETA1 = 0.9;
const ADAM_BETA2 = 0.999;
const ADAM_EPSILON = 1e-8;
const MUON_BETA = 0.95;
const MUON_EPSILON = 1e-7;

/** Built-in error functions */
export class Errors {
  public static SQUARE: ErrorFunction = {
    error: (output: number, target: number) =>
               0.5 * Math.pow(output - target, 2),
    der: (output: number, target: number) => output - target
  };
}

/** Polyfill for TANH */
(Math as any).tanh = (Math as any).tanh || function(x) {
  if (x === Infinity) {
    return 1;
  } else if (x === -Infinity) {
    return -1;
  } else {
    let e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }
};

/** Built-in activation functions */
export class Activations {
  public static TANH: ActivationFunction = {
    output: x => (Math as any).tanh(x),
    der: x => {
      let output = Activations.TANH.output(x);
      return 1 - output * output;
    }
  };
  public static RELU: ActivationFunction = {
    output: x => Math.max(0, x),
    der: x => x <= 0 ? 0 : 1
  };
  public static SIGMOID: ActivationFunction = {
    output: x => 1 / (1 + Math.exp(-x)),
    der: x => {
      let output = Activations.SIGMOID.output(x);
      return output * (1 - output);
    }
  };
  public static LINEAR: ActivationFunction = {
    output: x => x,
    der: x => 1
  };
}

/** Build-in regularization functions */
export class RegularizationFunction {
  public static L1: RegularizationFunction = {
    output: w => Math.abs(w),
    der: w => w < 0 ? -1 : (w > 0 ? 1 : 0)
  };
  public static L2: RegularizationFunction = {
    output: w => 0.5 * w * w,
    der: w => w
  };
}

/**
 * A link in a neural network. Each link has a weight and a source and
 * destination node. Also it has an internal state (error derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
export class Link {
  id: string;
  source: Node;
  dest: Node;
  weight = Math.random() - 0.5;
  isDead = false;
  /** Error derivative with respect to this weight. */
  errorDer = 0;
  /** Accumulated error derivative since the last update. */
  accErrorDer = 0;
  /** Number of accumulated derivatives since the last update. */
  numAccumulatedDers = 0;
  regularization: RegularizationFunction;
  optimizerState: ScalarOptimizerState = {t: 0, m: 0, v: 0};
  muonMomentum = 0;

  /**
   * Constructs a link in the neural network initialized with random weight.
   *
   * @param source The source node.
   * @param dest The destination node.
   * @param regularization The regularization function that computes the
   *     penalty for this weight. If null, there will be no regularization.
   */
  constructor(source: Node, dest: Node,
      regularization: RegularizationFunction, initZero?: boolean) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
    this.regularization = regularization;
    if (initZero) {
      this.weight = 0;
    }
  }
}

/**
 * Builds a neural network.
 *
 * @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
 *   the network will have one input node, 2 nodes in first hidden layer,
 *   3 nodes in second hidden layer and 1 output node.
 * @param activation The activation function of every hidden node.
 * @param outputActivation The activation function for the output nodes.
 * @param regularization The regularization function that computes a penalty
 *     for a given weight (parameter) in the network. If null, there will be
 *     no regularization.
 * @param inputIds List of ids for the input nodes.
 */
export function buildNetwork(
    networkShape: number[], activation: ActivationFunction,
    outputActivation: ActivationFunction,
    regularization: RegularizationFunction,
    inputIds: string[], initZero?: boolean,
    normalization: Normalization = Normalization.NONE): Node[][] {
  let numLayers = networkShape.length;
  let id = 1;
  /** List of layers, with each layer being a list of nodes. */
  let network: Node[][] = [];
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    let isOutputLayer = layerIdx === numLayers - 1;
    let isInputLayer = layerIdx === 0;
    let currentLayer: Node[] = [];
    network.push(currentLayer);
    let numNodes = networkShape[layerIdx];
    for (let i = 0; i < numNodes; i++) {
      let nodeId = id.toString();
      if (isInputLayer) {
        nodeId = inputIds[i];
      } else {
        id++;
      }
      let node = new Node(nodeId,
          isOutputLayer ? outputActivation : activation, initZero);
      currentLayer.push(node);
      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j];
          let link = new Link(prevNode, node, regularization, initZero);
          prevNode.outputs.push(link);
          node.inputLinks.push(link);
        }
      }
    }
  }
  (network as NeuralNetwork).normalization = normalization;
  return network;
}

function getNormalization(network: Node[][]): Normalization {
  return (network as NeuralNetwork).normalization || Normalization.NONE;
}

function getMean(values: number[]): number {
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
  }
  return sum / values.length;
}

function getVariance(values: number[], mean: number): number {
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    let diff = values[i] - mean;
    sum += diff * diff;
  }
  return sum / values.length;
}

function setNormalizedOutput(node: Node, rawInput: number, mean: number,
    variance: number): number {
  node.normMean = mean;
  node.normVariance = variance;
  node.normStdInv = 1 / Math.sqrt(variance + NORMALIZATION_EPSILON);
  node.normalizedInput = (rawInput - mean) * node.normStdInv;
  node.totalInput = node.normGamma * node.normalizedInput + node.normBeta;
  node.output = node.activation.output(node.totalInput);
  return node.output;
}

/**
 * Runs a forward propagation of the provided input through the provided
 * network. This method modifies the internal state of the network - the
 * total input and output of each node in the network.
 *
 * @param network The neural network.
 * @param inputs The input array. Its length should match the number of input
 *     nodes in the network.
 * @return The final output of the network.
 */
export function forwardProp(network: Node[][], inputs: number[]): number {
  let inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error("The number of inputs must match the number of nodes in" +
        " the input layer");
  }
  // Update the input layer.
  for (let i = 0; i < inputLayer.length; i++) {
    let node = inputLayer[i];
    node.output = inputs[i];
  }
  let normalization = getNormalization(network);
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    let isOutputLayer = layerIdx === network.length - 1;
    if (isOutputLayer || normalization === Normalization.NONE) {
      // Update all the nodes in this layer.
      for (let i = 0; i < currentLayer.length; i++) {
        let node = currentLayer[i];
        node.updateOutput();
      }
    } else if (normalization === Normalization.LAYER_NORM) {
      let rawInputs: number[] = [];
      for (let i = 0; i < currentLayer.length; i++) {
        rawInputs.push(currentLayer[i].computeTotalInput());
      }
      let mean = getMean(rawInputs);
      let variance = getVariance(rawInputs, mean);
      for (let i = 0; i < currentLayer.length; i++) {
        setNormalizedOutput(currentLayer[i], rawInputs[i], mean, variance);
      }
    } else {
      for (let i = 0; i < currentLayer.length; i++) {
        let node = currentLayer[i];
        let rawInput = node.computeTotalInput();
        setNormalizedOutput(node, rawInput, node.runningMean,
            node.runningVariance);
      }
    }
  }
  return network[network.length - 1][0].output;
}

function createTrainingState(): TrainingNodeState {
  return {
    rawInput: 0,
    normalizedInput: 0,
    totalInput: 0,
    output: 0,
    outputDer: 0,
    inputDer: 0,
    rawInputDer: 0
  };
}

function getSampleNodeState(sample: {[id: string]: TrainingNodeState},
    node: Node): TrainingNodeState {
  let state = sample[node.id];
  if (state == null) {
    state = createTrainingState();
    sample[node.id] = state;
  }
  return state;
}

function forwardBatch(network: Node[][], inputs: number[][]):
    {[id: string]: TrainingNodeState}[] {
  let samples: {[id: string]: TrainingNodeState}[] = [];
  let inputLayer = network[0];
  let normalization = getNormalization(network);
  for (let sampleIdx = 0; sampleIdx < inputs.length; sampleIdx++) {
    if (inputs[sampleIdx].length !== inputLayer.length) {
      throw new Error("The number of inputs must match the number of nodes in" +
          " the input layer");
    }
    let sample: {[id: string]: TrainingNodeState} = {};
    samples.push(sample);
    for (let i = 0; i < inputLayer.length; i++) {
      getSampleNodeState(sample, inputLayer[i]).output = inputs[sampleIdx][i];
    }
  }

  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    let prevLayer = network[layerIdx - 1];
    let isOutputLayer = layerIdx === network.length - 1;
    for (let sampleIdx = 0; sampleIdx < samples.length; sampleIdx++) {
      let sample = samples[sampleIdx];
      for (let i = 0; i < currentLayer.length; i++) {
        let node = currentLayer[i];
        let rawInput = node.bias;
        for (let j = 0; j < prevLayer.length; j++) {
          rawInput += node.inputLinks[j].weight *
              getSampleNodeState(sample, prevLayer[j]).output;
        }
        getSampleNodeState(sample, node).rawInput = rawInput;
      }
    }

    if (isOutputLayer || normalization === Normalization.NONE) {
      for (let sampleIdx = 0; sampleIdx < samples.length; sampleIdx++) {
        let sample = samples[sampleIdx];
        for (let i = 0; i < currentLayer.length; i++) {
          let node = currentLayer[i];
          let state = getSampleNodeState(sample, node);
          state.totalInput = state.rawInput;
          state.output = node.activation.output(state.totalInput);
        }
      }
    } else if (normalization === Normalization.LAYER_NORM) {
      for (let sampleIdx = 0; sampleIdx < samples.length; sampleIdx++) {
        let sample = samples[sampleIdx];
        let rawInputs: number[] = [];
        for (let i = 0; i < currentLayer.length; i++) {
          rawInputs.push(getSampleNodeState(sample, currentLayer[i]).rawInput);
        }
        let mean = getMean(rawInputs);
        let variance = getVariance(rawInputs, mean);
        let stdInv = 1 / Math.sqrt(variance + NORMALIZATION_EPSILON);
        for (let i = 0; i < currentLayer.length; i++) {
          let node = currentLayer[i];
          let state = getSampleNodeState(sample, node);
          state.normalizedInput = (state.rawInput - mean) * stdInv;
          state.totalInput = node.normGamma * state.normalizedInput +
              node.normBeta;
          state.output = node.activation.output(state.totalInput);
        }
      }
    } else {
      for (let i = 0; i < currentLayer.length; i++) {
        let node = currentLayer[i];
        let rawInputs: number[] = [];
        for (let sampleIdx = 0; sampleIdx < samples.length; sampleIdx++) {
          rawInputs.push(getSampleNodeState(samples[sampleIdx], node).rawInput);
        }
        let mean = getMean(rawInputs);
        let variance = getVariance(rawInputs, mean);
        let stdInv = 1 / Math.sqrt(variance + NORMALIZATION_EPSILON);
        node.runningMean = BATCH_NORM_MOMENTUM * node.runningMean +
            (1 - BATCH_NORM_MOMENTUM) * mean;
        node.runningVariance = BATCH_NORM_MOMENTUM * node.runningVariance +
            (1 - BATCH_NORM_MOMENTUM) * variance;
        for (let sampleIdx = 0; sampleIdx < samples.length; sampleIdx++) {
          let state = getSampleNodeState(samples[sampleIdx], node);
          state.normalizedInput = (state.rawInput - mean) * stdInv;
          state.totalInput = node.normGamma * state.normalizedInput +
              node.normBeta;
          state.output = node.activation.output(state.totalInput);
        }
      }
    }
  }
  return samples;
}

/** Trains the network on one mini-batch. */
export function trainBatch(network: Node[][], inputs: number[][],
    targets: number[], errorFunc: ErrorFunction, learningRate: number,
    regularizationRate: number,
    optimizer: Optimizer = Optimizer.SGD): void {
  if (inputs.length !== targets.length) {
    throw new Error("The number of inputs must match the number of targets");
  }
  if (inputs.length === 0) {
    return;
  }
  let samples = forwardBatch(network, inputs);
  let outputNode = network[network.length - 1][0];
  for (let sampleIdx = 0; sampleIdx < samples.length; sampleIdx++) {
    let outputState = getSampleNodeState(samples[sampleIdx], outputNode);
    outputState.outputDer = errorFunc.der(outputState.output,
        targets[sampleIdx]);
  }

  let normalization = getNormalization(network);
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    let currentLayer = network[layerIdx];
    let prevLayer = network[layerIdx - 1];
    let isOutputLayer = layerIdx === network.length - 1;

    for (let sampleIdx = 0; sampleIdx < samples.length; sampleIdx++) {
      for (let i = 0; i < prevLayer.length; i++) {
        getSampleNodeState(samples[sampleIdx], prevLayer[i]).outputDer = 0;
      }
    }

    if (isOutputLayer || normalization === Normalization.NONE) {
      for (let sampleIdx = 0; sampleIdx < samples.length; sampleIdx++) {
        let sample = samples[sampleIdx];
        for (let i = 0; i < currentLayer.length; i++) {
          let node = currentLayer[i];
          let state = getSampleNodeState(sample, node);
          state.inputDer = state.outputDer *
              node.activation.der(state.totalInput);
          state.rawInputDer = state.inputDer;
        }
      }
    } else if (normalization === Normalization.LAYER_NORM) {
      for (let sampleIdx = 0; sampleIdx < samples.length; sampleIdx++) {
        let sample = samples[sampleIdx];
        let dxHat: number[] = [];
        let xHat: number[] = [];
        let rawInputs: number[] = [];
        for (let i = 0; i < currentLayer.length; i++) {
          let node = currentLayer[i];
          let state = getSampleNodeState(sample, node);
          rawInputs.push(state.rawInput);
          state.inputDer = state.outputDer *
              node.activation.der(state.totalInput);
          node.accNormGammaDer += state.inputDer * state.normalizedInput;
          node.accNormBetaDer += state.inputDer;
          node.numAccumulatedNormDers++;
          dxHat.push(state.inputDer * node.normGamma);
          xHat.push(state.normalizedInput);
        }
        let mean = getMean(rawInputs);
        let variance = getVariance(rawInputs, mean);
        let stdInv = 1 / Math.sqrt(variance + NORMALIZATION_EPSILON);
        let meanDxHat = getMean(dxHat);
        let dxHatTimesXHat: number[] = [];
        for (let i = 0; i < dxHat.length; i++) {
          dxHatTimesXHat.push(dxHat[i] * xHat[i]);
        }
        let meanDxHatTimesXHat = getMean(dxHatTimesXHat);
        for (let i = 0; i < currentLayer.length; i++) {
          let state = getSampleNodeState(sample, currentLayer[i]);
          state.rawInputDer = stdInv *
              (dxHat[i] - meanDxHat - xHat[i] * meanDxHatTimesXHat);
        }
      }
    } else {
      for (let i = 0; i < currentLayer.length; i++) {
        let node = currentLayer[i];
        let dxHat: number[] = [];
        let xHat: number[] = [];
        let rawInputs: number[] = [];
        for (let sampleIdx = 0; sampleIdx < samples.length; sampleIdx++) {
          let state = getSampleNodeState(samples[sampleIdx], node);
          rawInputs.push(state.rawInput);
          state.inputDer = state.outputDer *
              node.activation.der(state.totalInput);
          node.accNormGammaDer += state.inputDer * state.normalizedInput;
          node.accNormBetaDer += state.inputDer;
          node.numAccumulatedNormDers++;
          dxHat.push(state.inputDer * node.normGamma);
          xHat.push(state.normalizedInput);
        }
        let mean = getMean(rawInputs);
        let variance = getVariance(rawInputs, mean);
        let stdInv = 1 / Math.sqrt(variance + NORMALIZATION_EPSILON);
        let meanDxHat = getMean(dxHat);
        let dxHatTimesXHat: number[] = [];
        for (let sampleIdx = 0; sampleIdx < samples.length; sampleIdx++) {
          dxHatTimesXHat.push(dxHat[sampleIdx] * xHat[sampleIdx]);
        }
        let meanDxHatTimesXHat = getMean(dxHatTimesXHat);
        for (let sampleIdx = 0; sampleIdx < samples.length; sampleIdx++) {
          let state = getSampleNodeState(samples[sampleIdx], node);
          state.rawInputDer = stdInv * (dxHat[sampleIdx] - meanDxHat -
              xHat[sampleIdx] * meanDxHatTimesXHat);
        }
      }
    }

    for (let sampleIdx = 0; sampleIdx < samples.length; sampleIdx++) {
      let sample = samples[sampleIdx];
      for (let i = 0; i < currentLayer.length; i++) {
        let node = currentLayer[i];
        let state = getSampleNodeState(sample, node);
        node.accInputDer += state.rawInputDer;
        node.numAccumulatedDers++;
        for (let j = 0; j < node.inputLinks.length; j++) {
          let link = node.inputLinks[j];
          if (link.isDead) {
            continue;
          }
          let sourceState = getSampleNodeState(sample, link.source);
          link.errorDer = state.rawInputDer * sourceState.output;
          link.accErrorDer += link.errorDer;
          link.numAccumulatedDers++;
          sourceState.outputDer += link.weight * state.rawInputDer;
        }
      }
    }
  }

  updateWeights(network, learningRate, regularizationRate, optimizer);
}

/**
 * Runs a backward propagation using the provided target and the
 * computed output of the previous call to forward propagation.
 * This method modifies the internal state of the network - the error
 * derivatives with respect to each node, and each weight
 * in the network.
 */
export function backProp(network: Node[][], target: number,
    errorFunc: ErrorFunction): void {
  // The output node is a special case. We use the user-defined error
  // function for the derivative.
  let outputNode = network[network.length - 1][0];
  outputNode.outputDer = errorFunc.der(outputNode.output, target);

  // Go through the layers backwards.
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    let currentLayer = network[layerIdx];
    // Compute the error derivative of each node with respect to:
    // 1) its total input
    // 2) each of its input weights.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.inputDer = node.outputDer * node.activation.der(node.totalInput);
      node.rawInputDer = node.inputDer;
      node.accInputDer += node.rawInputDer;
      node.numAccumulatedDers++;
    }

    // Error derivative with respect to each weight coming into the node.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        link.errorDer = node.rawInputDer * link.source.output;
        link.accErrorDer += link.errorDer;
        link.numAccumulatedDers++;
      }
    }
    if (layerIdx === 1) {
      continue;
    }
    let prevLayer = network[layerIdx - 1];
    for (let i = 0; i < prevLayer.length; i++) {
      let node = prevLayer[i];
      // Compute the error derivative with respect to each node's output.
      node.outputDer = 0;
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        node.outputDer += output.weight * output.dest.rawInputDer;
      }
    }
  }
}

function applyScalarOptimizer(value: number, grad: number,
    learningRate: number, optimizer: Optimizer,
    state: ScalarOptimizerState): number {
  if (optimizer === Optimizer.SGD) {
    return value - learningRate * grad;
  }
  state.t++;
  state.m = ADAM_BETA1 * state.m + (1 - ADAM_BETA1) * grad;
  state.v = ADAM_BETA2 * state.v + (1 - ADAM_BETA2) * grad * grad;
  let correctedM = state.m / (1 - Math.pow(ADAM_BETA1, state.t));
  let correctedV = state.v / (1 - Math.pow(ADAM_BETA2, state.t));
  return value - learningRate * correctedM /
      (Math.sqrt(correctedV) + ADAM_EPSILON);
}

function matrixTranspose(matrix: number[][]): number[][] {
  let result: number[][] = [];
  for (let col = 0; col < matrix[0].length; col++) {
    result[col] = [];
    for (let row = 0; row < matrix.length; row++) {
      result[col][row] = matrix[row][col];
    }
  }
  return result;
}

function matrixMultiply(a: number[][], b: number[][]): number[][] {
  let result: number[][] = [];
  for (let i = 0; i < a.length; i++) {
    result[i] = [];
    for (let j = 0; j < b[0].length; j++) {
      let sum = 0;
      for (let k = 0; k < b.length; k++) {
        sum += a[i][k] * b[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

function orthogonalize(matrix: number[][]): number[][] {
  if (matrix.length === 0 || matrix[0].length === 0) {
    return matrix;
  }
  if (matrix.length > matrix[0].length) {
    return matrixTranspose(orthogonalize(matrixTranspose(matrix)));
  }
  let norm = 0;
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      norm += matrix[i][j] * matrix[i][j];
    }
  }
  norm = Math.sqrt(norm);
  let x: number[][] = [];
  for (let i = 0; i < matrix.length; i++) {
    x[i] = [];
    for (let j = 0; j < matrix[i].length; j++) {
      x[i][j] = matrix[i][j] / (norm + MUON_EPSILON);
    }
  }
  for (let iter = 0; iter < 5; iter++) {
    let xxT = matrixMultiply(x, matrixTranspose(x));
    let xxTx = matrixMultiply(xxT, x);
    for (let i = 0; i < x.length; i++) {
      for (let j = 0; j < x[i].length; j++) {
        x[i][j] = 1.5 * x[i][j] - 0.5 * xxTx[i][j];
      }
    }
  }
  return x;
}

function updateLayerWeightsWithMuon(network: Node[][], layerIdx: number,
    learningRate: number, regularizationRate: number): boolean {
  if (layerIdx >= network.length - 1) {
    return false;
  }
  let currentLayer = network[layerIdx];
  if (currentLayer.length === 0 || currentLayer[0].inputLinks.length === 0) {
    return false;
  }
  let gradientMatrix: number[][] = [];
  for (let row = 0; row < currentLayer.length; row++) {
    gradientMatrix[row] = [];
    let node = currentLayer[row];
    for (let col = 0; col < node.inputLinks.length; col++) {
      let link = node.inputLinks[col];
      if (link.isDead || link.numAccumulatedDers === 0) {
        gradientMatrix[row][col] = 0;
        continue;
      }
      let regulDer = link.regularization ?
          link.regularization.der(link.weight) : 0;
      let grad = link.accErrorDer / link.numAccumulatedDers +
          regularizationRate * regulDer;
      link.muonMomentum = MUON_BETA * link.muonMomentum +
          (1 - MUON_BETA) * grad;
      gradientMatrix[row][col] = link.muonMomentum;
    }
  }
  let updateMatrix = orthogonalize(gradientMatrix);
  for (let row = 0; row < currentLayer.length; row++) {
    let node = currentLayer[row];
    for (let col = 0; col < node.inputLinks.length; col++) {
      let link = node.inputLinks[col];
      if (!link.isDead && link.numAccumulatedDers > 0) {
        link.weight -= learningRate * updateMatrix[row][col];
      }
      link.accErrorDer = 0;
      link.numAccumulatedDers = 0;
    }
  }
  return true;
}

/**
 * Updates the weights of the network using the previously accumulated error
 * derivatives.
 */
export function updateWeights(network: Node[][], learningRate: number,
    regularizationRate: number, optimizer: Optimizer = Optimizer.SGD) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    let muonUpdatedLayer = optimizer === Optimizer.MUON &&
        updateLayerWeightsWithMuon(network, layerIdx, learningRate,
            regularizationRate);
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      // Update the node's bias.
      if (node.numAccumulatedDers > 0) {
        let biasGrad = node.accInputDer / node.numAccumulatedDers;
        node.bias = applyScalarOptimizer(node.bias, biasGrad, learningRate,
            optimizer, node.biasOptimizerState);
        node.accInputDer = 0;
        node.numAccumulatedDers = 0;
      }
      if (node.numAccumulatedNormDers > 0) {
        let gammaGrad = node.accNormGammaDer / node.numAccumulatedNormDers;
        let betaGrad = node.accNormBetaDer / node.numAccumulatedNormDers;
        node.normGamma = applyScalarOptimizer(node.normGamma, gammaGrad,
            learningRate, optimizer, node.normGammaOptimizerState);
        node.normBeta = applyScalarOptimizer(node.normBeta, betaGrad,
            learningRate, optimizer, node.normBetaOptimizerState);
        node.accNormGammaDer = 0;
        node.accNormBetaDer = 0;
        node.numAccumulatedNormDers = 0;
      }
      // Update the weights coming into this node.
      if (muonUpdatedLayer) {
        continue;
      }
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        let regulDer = link.regularization ?
            link.regularization.der(link.weight) : 0;
        if (link.numAccumulatedDers > 0) {
          let grad = link.accErrorDer / link.numAccumulatedDers +
              regularizationRate * regulDer;
          let newLinkWeight = applyScalarOptimizer(link.weight, grad,
              learningRate, optimizer, link.optimizerState);
          if (optimizer === Optimizer.SGD &&
              link.regularization === RegularizationFunction.L1 &&
              link.weight * newLinkWeight < 0) {
            // The weight crossed 0 due to the regularization term. Set it to 0.
            link.weight = 0;
            link.isDead = true;
          } else {
            link.weight = newLinkWeight;
          }
          link.accErrorDer = 0;
          link.numAccumulatedDers = 0;
        }
      }
    }
  }
}

/** Iterates over every node in the network/ */
export function forEachNode(network: Node[][], ignoreInputs: boolean,
    accessor: (node: Node) => any) {
  for (let layerIdx = ignoreInputs ? 1 : 0;
      layerIdx < network.length;
      layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      accessor(node);
    }
  }
}

/** Returns the output node in the network. */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0];
}
