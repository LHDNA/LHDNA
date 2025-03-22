# LHDNA: Detecting Hallucinations in Large Language Models via Neuron Activation Checking
## Structure
- data and dataset: the file used to save datasets
- results: experimental data including results of preliminary study and experiments
## Usage
To run our code, please specify the path to the corresponding LLMs and execute the following steps:

- Modify the "model_root" variable to match the path where your LLMs are stored.

- Run the following commands:

        python LHDNA.py
        
        python evaluate.py

## LLMs
- OPT-1.3B: https://huggingface.co/facebook/opt-1.3b
- OPT-2.7B: https://huggingface.co/facebook/opt-2.7b
- OPT-6.7B: https://huggingface.co/facebook/opt-6.7b
- Qwen2-7B: https://huggingface.co/Qwen/Qwen2-7B
- LLaMA-2-13B: https://huggingface.co/meta-llama/Llama-2-13b-hf
- LLaMA-3-8B: https://huggingface.co/meta-llama/Meta-Llama-3-8B
