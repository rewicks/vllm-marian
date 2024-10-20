from transformers import MarianMTModel, MarianConfig, MarianTokenizer, GenerationConfig
import yaml
import argparse
import os
import torch, numpy
import sentencepiece as spm
import shutil
import json

import traceback
def remap(huggingface_model, state_dict, marian_state_dict):

    state_dict['model.encoder.embed_tokens.weight'] = torch.tensor(marian_state_dict["Wemb"])

    for i, l in enumerate(huggingface_model.model.encoder.layers):

        state_dict[f'model.encoder.layers.{i}.self_attn.k_proj.weight'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_self_Wk"]).transpose(0,1).squeeze()
        state_dict[f'model.encoder.layers.{i}.self_attn.k_proj.bias'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_self_bk"]).transpose(0,1).squeeze()

        state_dict[f'model.encoder.layers.{i}.self_attn.v_proj.weight'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_self_Wv"]).transpose(0,1).squeeze()
        state_dict[f'model.encoder.layers.{i}.self_attn.v_proj.bias'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_self_bv"]).transpose(0,1).squeeze()

        state_dict[f'model.encoder.layers.{i}.self_attn.q_proj.weight'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_self_Wq"]).transpose(0,1).squeeze()
        state_dict[f'model.encoder.layers.{i}.self_attn.q_proj.bias'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_self_bq"]).transpose(0,1).squeeze()

        state_dict[f'model.encoder.layers.{i}.self_attn.out_proj.weight'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_self_Wo"]).transpose(0,1).squeeze()
        state_dict[f'model.encoder.layers.{i}.self_attn.out_proj.bias'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_self_bo"]).transpose(0,1).squeeze()

        state_dict[f'model.encoder.layers.{i}.self_attn_layer_norm.weight'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_self_Wo_ln_scale"]).transpose(0,1).squeeze()
        state_dict[f'model.encoder.layers.{i}.self_attn_layer_norm.bias'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_self_Wo_ln_bias"]).transpose(0,1).squeeze()

        state_dict[f'model.encoder.layers.{i}.fc1.weight'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_ffn_W1"]).transpose(0,1).squeeze()
        state_dict[f'model.encoder.layers.{i}.fc1.bias'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_ffn_b1"]).transpose(0,1).squeeze()

        state_dict[f'model.encoder.layers.{i}.fc2.weight'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_ffn_W2"]).transpose(0,1).squeeze()
        state_dict[f'model.encoder.layers.{i}.fc2.bias'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_ffn_b2"]).transpose(0,1).squeeze()

        state_dict[f'model.encoder.layers.{i}.final_layer_norm.weight'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_ffn_ffn_ln_scale"]).transpose(0,1).squeeze()
        state_dict[f'model.encoder.layers.{i}.final_layer_norm.bias'] = torch.tensor(marian_state_dict[f"encoder_l{i+1}_ffn_ffn_ln_bias"]).transpose(0,1).squeeze()

        
    state_dict['model.decoder.embed_tokens.weight'] = torch.tensor(marian_state_dict["Wemb"])
    
    for i, l in enumerate(huggingface_model.model.decoder.layers):

        state_dict[f'model.decoder.layers.{i}.self_attn.k_proj.weight'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_self_Wk"]).transpose(0,1).squeeze()
        state_dict[f'model.decoder.layers.{i}.self_attn.k_proj.bias'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_self_bk"]).transpose(0,1).squeeze()

        state_dict[f'model.decoder.layers.{i}.self_attn.v_proj.weight'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_self_Wv"]).transpose(0,1).squeeze()
        state_dict[f'model.decoder.layers.{i}.self_attn.v_proj.bias'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_self_bv"]).transpose(0,1).squeeze()

        state_dict[f'model.decoder.layers.{i}.self_attn.q_proj.weight'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_self_Wq"]).transpose(0,1).squeeze()
        state_dict[f'model.decoder.layers.{i}.self_attn.q_proj.bias'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_self_bq"]).transpose(0,1).squeeze()

        state_dict[f'model.decoder.layers.{i}.self_attn.out_proj.weight'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_self_Wo"]).transpose(0,1).squeeze()
        state_dict[f'model.decoder.layers.{i}.self_attn.out_proj.bias'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_self_bo"]).transpose(0,1).squeeze()

        state_dict[f'model.decoder.layers.{i}.self_attn_layer_norm.weight'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_self_Wo_ln_scale"]).transpose(0,1).squeeze()
        state_dict[f'model.decoder.layers.{i}.self_attn_layer_norm.bias'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_self_Wo_ln_bias"]).transpose(0,1).squeeze()

        state_dict[f'model.decoder.layers.{i}.encoder_attn.k_proj.weight'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_context_Wk"]).transpose(0,1).squeeze()
        state_dict[f'model.decoder.layers.{i}.encoder_attn.k_proj.bias'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_context_bk"]).transpose(0,1).squeeze()

        state_dict[f'model.decoder.layers.{i}.encoder_attn.v_proj.weight'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_context_Wv"]).transpose(0,1).squeeze()
        state_dict[f'model.decoder.layers.{i}.encoder_attn.v_proj.bias'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_context_bv"]).transpose(0,1).squeeze()

        state_dict[f'model.decoder.layers.{i}.encoder_attn.q_proj.weight'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_context_Wq"]).transpose(0,1).squeeze()
        state_dict[f'model.decoder.layers.{i}.encoder_attn.q_proj.bias'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_context_bq"]).transpose(0,1).squeeze()

        state_dict[f'model.decoder.layers.{i}.encoder_attn.out_proj.weight'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_context_Wo"]).transpose(0,1).squeeze()
        state_dict[f'model.decoder.layers.{i}.encoder_attn.out_proj.bias'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_context_bo"]).transpose(0,1).squeeze()

        state_dict[f'model.decoder.layers.{i}.encoder_attn_layer_norm.weight'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_context_Wo_ln_scale"]).transpose(0,1).squeeze()
        state_dict[f'model.decoder.layers.{i}.encoder_attn_layer_norm.bias'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_context_Wo_ln_bias"]).transpose(0,1).squeeze()

        state_dict[f'model.decoder.layers.{i}.fc1.weight'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_ffn_W1"]).transpose(0,1).squeeze()
        state_dict[f'model.decoder.layers.{i}.fc1.bias'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_ffn_b1"]).transpose(0,1).squeeze()

        state_dict[f'model.decoder.layers.{i}.fc2.weight'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_ffn_W2"]).transpose(0,1).squeeze()
        state_dict[f'model.decoder.layers.{i}.fc2.bias'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_ffn_b2"]).transpose(0,1).squeeze()

        state_dict[f'model.decoder.layers.{i}.final_layer_norm.weight'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_ffn_ffn_ln_scale"]).transpose(0,1).squeeze()
        state_dict[f'model.decoder.layers.{i}.final_layer_norm.bias'] = torch.tensor(marian_state_dict[f"decoder_l{i+1}_ffn_ffn_ln_bias"]).transpose(0,1).squeeze()


    state_dict['final_logits_bias'] = torch.tensor(marian_state_dict["decoder_ff_logit_out_b"])
    state_dict['lm_head.weight'] = torch.tensor(marian_state_dict[f"Wemb"])

    huggingface_model.load_state_dict(state_dict)

    return huggingface_model
    

def tokenize(model, inputs, eos_token=2, pad_token=2):
    out = []
    tokens = model.encode(inputs)
    max_length = max([len(t) for t in tokens]) + 1
    for t in tokens:
        out.append(t + [eos_token] + [pad_token] * (max_length - len(t)))
    return torch.tensor(out)

def decode(model, outputs, skip_special_tokens=False):
    out = []
    for o in outputs:
        out.append(model.decode([_.item() for _ in o]))
    return out

def save_tokenizer_config(dest_dir, source="eng_Latn", target="deu_Latn", separate_vocabs=False):
    dct = {"target_lang": target, "source_lang": source, "separate_vocabs": separate_vocabs}
    json.dump(dct, open(os.path.join(dest_dir, "tokenizer_config.json"), 'w'), ensure_ascii=False, indent=2)

def get_vocab_json(vocab_path):
    vocab = {}
    with open(vocab_path) as infile:
        for i, line in enumerate(infile):
            token = line.split('\t')[0]
            vocab[token] = i
    return vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to the Marian model (the actual .npz path)")
    parser.add_argument("--yaml_path", help="The path to the model.npz.yml")
    parser.add_argument("--tokenizer_path", help="The path to the sentencepiece spm.model file")
    parser.add_argument("--marian-vocab", default='vocab', help="The path to the Marian vocab file (just the text file with the tokens)")
    parser.add_argument("--destdir", help="The name of the output directory to create")

    args = parser.parse_args()

    model_config = yaml.load(open(args.yaml_path, "r"), Loader=yaml.FullLoader)


    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_path)

    huggingface_config = MarianConfig(
        vocab_size=model_config["dim-vocabs"][0],
        decoder_vocab_size=model_config["dim-vocabs"][1],
        max_position_embeddings=model_config["max-length"], # was model_config["max-length"]
        encoder_layers=model_config["enc-depth"],
        encoder_ffn_dim=model_config["transformer-dim-ffn"],
        encoder_attention_heads=model_config["transformer-heads"],
        decoder_layers=model_config["dec-depth"],
        decoder_ffn_dim=model_config["transformer-decoder-dim-ffn"],
        decoder_attention_heads=model_config["transformer-heads"],
        encoder_layerdrop=model_config["transformer-dropout-ffn"],
        decoder_layerdrop=model_config["transformer-dropout-ffn"],
        use_cache=False,
        is_encoder_decoder=True,
        activation_function=model_config["transformer-ffn-activation"],
        d_model=model_config["dim-emb"],
        dropout=model_config["transformer-dropout"],
        attention_dropout=model_config["transformer-dropout-attention"],
        init_std=0.02,
        decoder_start_token_id=1,
        scale_embedding=True, # was False
        normalize_embedding=False,
        static_position_embeddings=True,
        bos_token_id=tokenizer.bos_id(),
        pad_token_id=tokenizer.pad_id(),
        eos_token_id=tokenizer.eos_id(),
        tie_word_embeddings=True,
        share_encoder_decoder_embeddings=True,
    )

    generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_id(),
        decoder_start_token_id=tokenizer.bos_id(),
        eos_token_id=tokenizer.eos_id(),
        forced_eos_token_id=tokenizer.eos_id(),
        max_length=model_config["max-length"],
        num_beams=5,
        pad_token_id=tokenizer.pad_id(),
    )

    model = MarianMTModel(huggingface_config)

    print(model)

    state_dict = model.state_dict()

    remap(model, state_dict, numpy.load(args.model_path))

    model.eval()
 
    src = [
        "I have to take the dog for a walk."
        "Can you pass the butter?"
    ]

    batch = {"input_ids": tokenize(tokenizer, src, pad_token=tokenizer.pad_id())}


    generated_ids = model.generate(**batch, max_length=50, use_cache=False)

    print(decode(tokenizer, generated_ids, skip_special_tokens=True)[0])
    print(decode(tokenizer, generated_ids, skip_special_tokens=True)[1])

    model.save_pretrained(args.destdir)
    generation_config.save_pretrained(args.destdir)
    
    vocab_path = args.tokenizer_path.replace('.model', '.vocab')
    vocab = get_vocab_json(vocab_path)
    json.dump(vocab, open(os.path.join(args.destdir, "vocab.json"), 'w'), ensure_ascii=False, indent=2)


    marian_tokenizer = MarianTokenizer(
        source_spm=args.tokenizer_path,
        target_spm=args.tokenizer_path,
        vocab = os.path.join(args.destdir, 'vocab.json'),
        additional_special_tokens = ["<s>"]
    )
    marian_tokenizer.bos_token = "<s>"
    marian_tokenizer.bos_token_id = tokenizer.bos_id()


    marian_tokenizer.save_pretrained(args.destdir)

    shutil.copy(args.tokenizer_path, os.path.join(args.destdir, 'source.spm'))
    shutil.copy(args.tokenizer_path, os.path.join(args.destdir, 'target.spm'))
    shutil.copy(args.tokenizer_path, os.path.join(args.destdir, 'spm.model'))
    shutil.copy(vocab_path, os.path.join(args.destdir, 'spm.vocab'))
    shutil.copy(args.model_path, os.path.join(args.destdir, 'model.npz'))
    shutil.copy(args.yaml_path, os.path.join(args.destdir, 'model.npz.yml'))
    shutil.copy(args.marian_vocab, os.path.join(args.destdir, 'marian.vocab'))
