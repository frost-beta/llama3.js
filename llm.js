import fs from 'node:fs'
import path from 'node:path'
import {core as mx, nn} from '@frost-beta/mlx'

import {Model} from './model.js'

// Return a model.
export function loadModel(dir) {
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
export async function* step(promptTokens, model, topP = 1, temperature = 1) {
  let cache = null
  const forward = (y) => {
    let logits
    [logits, cache] = model.forward(mx.array([y], mx.int32), cache)
    logits = logits.index(mx.Slice(), -1, mx.Slice())
    const [token, prob] = sample(logits, topP, temperature)
    // The cache is also returned so it does not get freed by mx.tidy().
    return [token.item(), prob.item(), cache]
  }

  let tokens = promptTokens
  while (true) {
    // Forward the tokens to model, and make sure intermediate tensors are freed.
    const [token, prob] = mx.tidy(() => forward(tokens))
    tokens = [token]
    // Yield the result in the next tick of loop, so GC can get a chance to run.
    // TODO(zcbenz): Use the async API after this MLX issue is solved:
    // https://github.com/ml-explore/mlx/issues/1251
    yield await new Promise((resolve) => {
      process.nextTick(() => resolve([token, prob]))
    })
  }
}

// Pick the best token from logits.
export function sample(logits, topP, temperature) {
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
export function topPSampling(logits, topP, temperature) {
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
