#!/usr/bin/env node

import readline from 'node:readline/promises';
import {core as mx} from '@frost-beta/mlx'
import {loadTokenizer, loadModel, step} from './llm.js'

if (process.argv.length < 3) {
  console.error('Usage: llama3 /path/to/weights/dir')
  process.exit(0)
}

// We don't limit the total tokens, and RAM just keeps increasing as context is
// being accumulated, without limiting cache the RAM will not go down to the
// normal after generation is finished.
mx.metal.setCacheLimit(10 * 1024 ** 2)

main(process.argv[2])

async function main(dir) {
  // Load tokenizer.
  const tokenizer = await loadTokenizer(dir)

  // Load model.
  const model = await loadModel(dir)

  // Records the messages.
  const messages = []

  // Chat interface.
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  rl.once('close', () => process.stdout.write('\n'))
  while (true) {
    const question = await rl.question('You> ')
    messages.push({role: 'user', content: question})
    process.stdout.write('Assistant> ')
    const reply = await talk(tokenizer, model, messages)
    messages.push({role: 'assistant', content: reply})
  }
}

// Send full messages history to model and get response.
async function talk(tokenizer, model, messages) {
  // Translate the messages to tokens.
  const prompt = tokenizer.apply_chat_template(messages, {return_tensor: false})

  // The token marking the end of conversation.
  const eosToken = tokenizer.encode(tokenizer.getToken('eos_token'))[0]

  // Predict next tokens.
  let text = ''
  for await (const [token, prob] of step(prompt, model, eosToken, 0.8)) {
    const char = tokenizer.decode([token])
    text += char
    process.stdout.write(char)
  }
  process.stdout.write('\n')

  if (false) {  // used for debugging leaks
    console.log(`MLX RAM ${(mx.metal.getActiveMemory() / 1024 ** 2).toFixed(1)}M,`,
                `Cache ${(mx.metal.getCacheMemory() / 1024 ** 2).toFixed(1)}M,`,
                `JS Objects ${mx.getWrappersCount()}.`)
  }
  return text
}
