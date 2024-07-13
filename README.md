# llama3.js

A JavaScript implementation of Llama 3 using [node-mlx](https://github.com/frost-beta/node-mlx),
code modified from [mlx-examples](https://github.com/ml-explore/mlx-examples).

__Quantized models can only run on Macs with Apple Silicon.__

## Usage

Download weights
(more can be found at [mlx-community](https://huggingface.co/collections/mlx-community/llama-3-662156b069a5d33b3328603c)):

```sh
npm install -g @frost-beta/huggingface
huggingface download --to weights mlx-community/Meta-Llama-3-8B-Instruct-8bit
```

Start chating:

```sh
npm install -g llama3
llama3-chat ./weights
```

Or do text generation:

```sh
llama3-generate ./weights 'Write a short story'
```

## Development

This project serves as a demo of node-mlx, and code is intended to keep as
simple as possible.

For general purpose LLM modules, please visit :construction:.
