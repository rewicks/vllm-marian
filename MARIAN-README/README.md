# VLLM Install

Here's everything I ran (I believe)
```
conda create -n vllm python=3.10 -y
conda activate vllm

pip install numpy
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn --no-build-isolation
# I used gcc 9.3.0; cuda 11.8.0
# and it was a long install for me (they say 5-10 minutes but in reality mine took much longer). not sure if there's a way to speed it up.
pip install -e .
```


# Marian to Huggingface Conversion
This script basically creates a full directory that I can upload to huggingface. It adds some extra files that aren't necessary for huggingface, but are useful to have online.

As an example you can grab some of my files from huggingface. These also contain the raw marian files too:


```
git clone https://huggingface.co/rewicks/baseline_en-de_8k_ep1
```

```
python marian-to-huggingface.py \
    --model_path baseline_en-de_8k_ep1/model.npz \
    --yaml_path baseline_en-de_8k_ep1/model.npz.yml \
    --tokenizer_path baseline_en-de_8k_ep1/source.spm \
    --marian-vocab baseline_en-de_8k_ep1/marian.vocab \
    --destdir conversion-test
```

And this should make a folder that should be identical to the one you pulled from huggingface.


# Running vLLM

I believe there's an error in how vLLM saves off input ids, so the output is unreliable. The first token generated is correct, so you could handle the looping outside of vLLM but there's probably not much point in that.

Other than that, there's an example script in vllm-marian.py

The edited files to make the model work are `vllm/model_executor/models/marian.py` (custom, but derived from the bart.py file) and the `vllm/model_executor/models/registry.py` to register the model name.