import fs from 'node:fs'
import path from 'node:path'
import tokenizer from 'llama3-tokenizer-js'
import {Template} from "@huggingface/jinja"

export function loadTokenizer(dir) {
  const config = JSON.parse(fs.readFileSync(path.join(dir, 'tokenizer_config.json')))
  return config
}
