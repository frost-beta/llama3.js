#!/usr/bin/env node

import {core as mx} from '@frost-beta/mlx'
import {loadTokenizer, loadModel, step} from './llm.js'

if (process.argv.length < 3) {
  console.error('Usage: llama3 /path/to/weights/dir [prompt]')
  process.exit(0)
}

main(process.argv[2], process.argv[3])

async function main(dir, prompt) {
  const tokenizer = await loadTokenizer(dir)
  const model = await loadModel(dir)

  if (prompt)
    process.stdout.write(prompt)

  // Get BOS and EOS tokens.
  const bosToken = tokenizer.encode(tokenizer.getToken('bos_token'))[0]
  const eosToken = tokenizer.encode(tokenizer.getToken('eos_token'))[0]

  // Encode prompt or just use BOS.
  prompt = prompt ? tokenizer.encode(prompt) : [bosToken]

  // Generation.
  for await (const [token, prob] of step(prompt, model, eosToken, 0.8)) {
    const char = tokenizer.decode([token])
    process.stdout.write(char)
  }
  process.stdout.write('\n')
}
