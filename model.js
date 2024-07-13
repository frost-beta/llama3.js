import {core as mx, nn} from '@frost-beta/mlx'

// A design of KV cache friendly to MLX's memory cache design, which allocates
// arrays in same shapes.
// See also https://github.com/ml-explore/mlx-examples/issues/724.
export class KVCache {
  constructor(headDim, nKVHeads) {
    this.nKVHeads = nKVHeads
    this.headDim = headDim
    this.keys = null
    this.values = null
    this.offset = 0
    this.step = 256
  }

  updateAndFetch(keys, values) {
    const prev = this.offset
    if (!this.keys || (prev + keys.shape[2] > this.keys.shape[2])) {
      const nSteps = Math.floor((this.step + keys.shape[2] - 1) / this.step)
      const shape = [1, this.nKVHeads, nSteps * this.step, this.headDim]
      const newK = mx.zeros(shape, keys.dtype)
      const newV = mx.zeros(shape, values.dtype)
      if (this.keys) {
        const old = [this.keys, this.values]
        if (prev % this.step != 0) {
          const get = ['...', mx.Slice(null, prev), mx.Slice()]
          this.keys = this.keys.index(get)
          this.values = this.values.index(get)
        }
        this.keys = mx.concatenate([this.keys, newK], 2)
        this.values = mx.concatenate([this.values, newV], 2)
        mx.dispose(old)
      } else {
        this.keys = newK
        this.values = newV
      }
    }

    this.offset += keys.shape[2]

    const insert = ['...', mx.Slice(prev, this.offset), mx.Slice()]
    this.keys.indexPut_(insert, keys)
    this.values.indexPut_(insert, values)

    const get = ['...', mx.Slice(null, this.offset), mx.Slice()]
    return [this.keys.index(...get), this.values.index(...get)]
  }
}

class Attention extends nn.Module {
  constructor(args) {
    super()
    const dim = args.hiddenSize
    this.nHeads = args.numAttentionHeads
    this.nKVHeads = args.numKeyValueHeads

    const headDim = Math.floor(args.hiddenSize / this.nHeads)
    this.scale = headDim ** -0.5

    this.qProj = new nn.Linear(dim, this.nHeads * headDim, false)
    this.kProj = new nn.Linear(dim, this.nKVHeads * headDim, false)
    this.vProj = new nn.Linear(dim, this.nKVHeads * headDim, false)
    this.oProj = new nn.Linear(this.nHeads * headDim, dim, false)

    const ropeScale = args.ropeScaling?.type == 'linear' ? 1 / args.ropeScaling.factor
                                                         : 1
    this.rope = new nn.RoPE(headDim, args.ropeTraditional, args.ropeTheta, ropeScale)
  }

  forward(x, mask, cache) {
    const [B, L, D] = x.shape

    let queries = this.qProj.forward(x)
    let keys = this.kProj.forward(x)
    let values = this.vProj.forward(x)

    // Prepare the queries, keys and values for the attention computation.
    queries = queries.reshape(B, L, this.nHeads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, L, this.nKVHeads, -1).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, this.nKVHeads, -1).transpose(0, 2, 1, 3)

    if (cache) {
      queries = this.rope.forward(queries, cache.offset)
      keys = this.rope.forward(keys, cache.offset);
      [keys, values] = cache.updateAndFetch(keys, values)
    } else {
      queries = this.rope.forward(queries)
      keys = this.rope.forward(keys)
    }

    let output = mx.fast.scaledDotProductAttention(queries, keys, values, this.scale, mask)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return this.oProj.forward(output)
  }
}

class MLP extends nn.Module {
  constructor(dim, hiddenDim) {
    super()
    this.gateProj = new nn.Linear(dim, hiddenDim, false)
    this.downProj = new nn.Linear(hiddenDim, dim, false)
    this.upProj = new nn.Linear(dim, hiddenDim, false)
  }

  forward(x) {
    return this.downProj.forward(mx.multiply(nn.silu(this.gateProj.forward(x)),
                                             this.upProj.forward(x)))
  }
}

class TransformerBlock extends nn.Module {
  constructor(args) {
    super()
    this.numAttentionHeads = args.numAttentionHeads
    this.hiddenSize = args.hiddenSize
    this.selfAttn = new Attention(args)
    this.mlp = new MLP(args.hiddenSize, args.intermediateSize)
    this.inputLayernorm = new nn.RMSNorm(args.hiddenSize, args.rmsNormEps)
    this.postAttentionLayernorm = new nn.RMSNorm(args.hiddenSize, args.rmsNormEps)
  }

  forward(x, mask, cache) {
    const r = this.selfAttn.forward(this.inputLayernorm.forward(x), mask, cache)
    const h = mx.add(x, r)
    return mx.add(h, this.mlp.forward(this.postAttentionLayernorm.forward(h)))
  }
}

class LlamaModel extends nn.Module {
  constructor(args) {
    super()
    this.vocabSize = args.vocabSize
    this.numHiddenLayers = args.numHiddenLayers
    this.embedTokens = new nn.Embedding(args.vocabSize, args.hiddenSize)
    this.layers = []
    for (let i = 0; i < args.numHiddenLayers; ++i)
      this.layers.push(new TransformerBlock(args))
    this.norm = new nn.RMSNorm(args.hiddenSize, args.rmsNormEps)
  }

  forward(inputs, cache) {
    let h = this.embedTokens.forward(inputs)

    let mask
    if (h.shape[1] > 1) {
      mask = createAdditiveCausalMask(h.shape[1], cache ? cache[0].offset : 0)
      mask = mask.astype(h.dtype)
    }

    cache = cache ?? new Array(this.layers.length)

    for (let i in this.layers)
      h = this.layers[i].forward(h, mask, cache[i])

    return this.norm.forward(h)
  }
}

export class Model extends nn.Module {
  constructor(obj) {
    const args = modelArgs(obj)
    super()

    this.args = args
    this.modelType = args.modelType
    this.model = new LlamaModel(args)
    this.lmHead = new nn.Linear(args.hiddenSize, args.vocabSize, false)
  }

  forward(inputs, cache) {
    const out = this.model.forward(inputs, cache)
    return this.lmHead.forward(out)
  }

  get layers() {
    return this.model.layers
  }

  get headDim() {
    return Math.floor(this.args.hiddenSize / this.args.numAttentionHeads)
  }

  get nKVHeads() {
    return this.args.numKeyValueHeads
  }
}

function createAdditiveCausalMask(N, offset = 0) {
  const rinds = mx.arange(offset + N)
  const linds = offset ? mx.arange(offset, offset + N) : rinds
  const mask = mx.less(linds.index(mx.Slice(), null), rinds.index(null))
  return mx.multiply(mask, -1e9)
}

function modelArgs({model_type,
                    hidden_size,
                    num_hidden_layers,
                    intermediate_size,
                    num_attention_heads,
                    rms_norm_eps,
                    vocab_size,
                    num_key_value_heads = null,
                    rope_theta = 10000,
                    rope_traditional = false,
                    rope_scaling = null}) {
  if (vocab_size <= 0)
    throw new Error('vocabSize must be bigger than zero')
  if (rope_scaling) {
    const requiredKeys = [ 'factor', 'type' ]
    if (!Object.keys(rope_scaling).every(key => requiredKeys.includes(key))) {
      throw Error(`rope_scaling must contain keys ${requiredKeys}`)
    }
    if (rope_scaling.type != 'linear') {
      throw Error("rope_scaling 'type' currently only supports 'linear'")
    }
  }
  return {
    modelType: model_type,
    hiddenSize: hidden_size,
    numHiddenLayers: num_hidden_layers,
    intermediateSize: intermediate_size,
    numAttentionHeads: num_attention_heads,
    rmsNormEps: rms_norm_eps,
    vocabSize: vocab_size,
    numKeyValueHeads: num_key_value_heads ?? num_attention_heads,
    ropeTheta: rope_theta,
    ropeTraditional: rope_traditional,
    ropeScaling: rope_scaling,
  }
}
