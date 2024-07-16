#!/usr/bin/env node

import {core as mx} from '@frost-beta/mlx'
import {loadTokenizer, loadModel, step} from './llm.js'

let maxTokens = 1024
const argv = process.argv.slice(2).filter((arg) => {
  if (arg.startsWith('--max-tokens=')) {
    maxTokens = parseInt(arg.substr(arg.indexOf('=') + 1))
    return false
  }
  return true
})

if (argv.length < 1) {
  console.error('Usage: llama3-generate /path/to/weights/dir [--max-tokens=1024] [prompt]')
  process.exit(0)
}

main(argv[0], argv[1])

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
  if (prompt[prompt.length - 1] === eosToken)
    prompt = prompt.slice(0, -1)

  // Generation.
  let count = 0
  for await (const [token, prob] of step(prompt, model, eosToken, 0.8)) {
    const char = tokenizer.decode([token])
    process.stdout.write(char)
    if (++count > maxTokens)
      break
  }
  process.stdout.write('\n')
}
