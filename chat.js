import fs from 'node:fs'
import path from 'node:path'
import tokenizer from 'llama3-tokenizer-js'
import mlx from '@frost-beta/mlx'

import {Model} from './model.js'

if (process.argv.length < 3) {
  console.error('Usage: llama3 /path/to/weights/dir')
  process.exit(0)
}

const {core: mx, nn} = mlx

const dir = process.argv[2]
const model = load(dir)

const EOS = 128001

const prompt = 'Electron framework is a'
const promptTokens = mx.array(tokenizer.encode(prompt, {bos: false, eos: false}), mx.int32)
for (const [token, prob] of step(promptTokens, model)) {
  if (token == EOS)
    process.exit(0)
  process.stdout.write(tokenizer.decode([token]))
}

// Return a model.
function load(dir) {
  // Read model config and weights.
  const config = JSON.parse(fs.readFileSync(path.join(dir, 'config.json')))
  const weights = {}
  for (const filename of fs.readdirSync(dir)) {
    if (filename.endsWith('.safetensors'))
      Object.assign(weights, mx.load(path.join(dir, filename)))
  }

  // Create llama3 model.
  const model = new Model(config)

  // Quantization.
  if (config.quantization) {
    const predicate = (p, m) => {
      // Some legacy models which may not have everything quantized.
      return (`${p}.scales` in weights) &&
             ((m instanceof nn.Linear) || (m instanceof nn.Embedding))
    }
    const {group_size: groupSize, bits} = config.quantization
    nn.quantize(model, groupSize, bits, predicate)
  }

  // Load weights.
  model.loadWeights(Object.entries(weights))
  mx.eval(model.parameters())
  return model
}

// Generate tokens from prompt.
function* step(promptTokens, model, temperature = 0, topP = 1) {
  let cache = null
  const forward = (y) => {
    let logits
    [logits, cache] = model.forward(y.index(mx.newaxis), cache)
    logits = logits.index(mx.Slice(), -1, mx.Slice())
    return sample(logits, temperature, topP)
  }

  let y = promptTokens, p
  while (true) {
    [y, p] = forward(y)
    yield [y.item(), p]
  }
}

// Pick the best token from logits.
function sample(logits, temperature = 0, topP = 1) {
  const softmaxLogits = mx.softmax(logits)
  let token
  if (temperature === 0) {
    token = mx.argmax(logits, -1)
  } else {
    if (topP > 0 && topP < 1) {
      token = topPSampling(logits, topP, temperature)
    } else {
      token = mx.random.categorical(mx.multiply(logits, 1 / temperature))
    }
  }
  const prob = softmaxLogits.index(0, token)
  return [token, prob]
}

// Sampling with top-p.
function topPSampling(logits, topP, temperature) {
  const probs = mx.softmax(mx.divide(logits, temperature), -1)

  const sortedIndices = mx.argsort(probs, -1)
  const sortedProbs = probs.index('...', sortedIndices.squeeze(0))

  const cumulativeProbs = mx.cumsum(sortedProbs, -1)

  const topProbs = mx.where(mx.greater(cumulativeProbs, mx.subtract(1, topP)),
                            sortedProbs,
                            mx.zerosLike(sortedProbs))

  const sortedToken = mx.random.categorical(mx.log(topProbs))
  return sortedIndices.squeeze(0).index(sortedToken)
}
