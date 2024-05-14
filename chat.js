#!/usr/bin/env node

import fs from 'node:fs'
import path from 'node:path'
import readline from 'node:readline/promises';
import {core as mx} from '@frost-beta/mlx'
import {TokenizerLoader} from '@lenml/tokenizers'
import {loadModel, step} from './llm.js'

if (process.argv.length < 3) {
  console.error('Usage: llama3 /path/to/weights/dir')
  process.exit(0)
}

main(process.argv[2])

async function main(dir) {
  // Load tokenizer.
  const tokenizer = await TokenizerLoader.fromPreTrained({
    tokenizerJSON: JSON.parse(fs.readFileSync(path.join(dir, 'tokenizer.json'))),
    tokenizerConfig: JSON.parse(fs.readFileSync(path.join(dir, 'tokenizer_config.json'))),
  })

  // Load model.
  const model = loadModel(dir)

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
    const reply = talk(tokenizer, model, messages)
    messages.push({role: 'assistant', content: reply})
  }
}

// Send full messages history to model and get response.
function talk(tokenizer, model, messages) {
  // Translate the messages to tokens.
  const prompt = tokenizer.apply_chat_template(messages, {return_tensor: false})
  const promptTokens = mx.array(prompt, mx.int32)

  // The token marking the end of conversation.
  // TODO(zcbenz): eos_token_id not available, is it a bug of transformers.js?
  const eosToken = tokenizer.getToken('eos_token')

  // Predict next tokens.
  let text = ''
  for (const [token, prob] of step(promptTokens, model, 0.8)) {
    const char = tokenizer.decode([token])
    if (char == eosToken)
      break
    text += char
    process.stdout.write(char)
  }

  process.stdout.write('\n')
  return text
}
