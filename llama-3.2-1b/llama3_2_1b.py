import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import torch.nn.functional as fu
import math

device = torch.device("cuda")

torch.set_grad_enabled(False)

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model.to(device)
model.eval()

config_llama = model.config
config_llama

config_llama.rope_parameters

#Embeddings

texts = ["Hello, good morning"]



def embeddings_from_ids(ids):
    wte = model.model.embed_tokens.weight
    embeddings = wte[ids]

    return embeddings


def text_into_ids(text):
    ids = tokenizer.encode(text, return_tensors="pt")
    return ids.to(device)


#RoPE Implementation

def rope(tensor, generated_len):

    #Input has this shape: (batch, heads, seq, 64)
    #We transform it in pairs into (batch, heads, seq, 2, 32)
    original_embeddings = model.config.rope_parameters["original_max_position_embeddings"]
    high_freq_factor = model.config.rope_parameters["high_freq_factor"]
    low_freq_factor = model.config.rope_parameters["low_freq_factor"]
    wavelen_high = original_embeddings/high_freq_factor
    wavelen_low = original_embeddings/low_freq_factor
    factor = model.config.rope_parameters["factor"]

    pairs = tensor.view(*tensor.shape[:-1],2,32)
    pair_index = torch.arange(32, dtype=torch.float32, device=tensor.device)
    rope_theta = 500000.0
    speeds = 1 / (rope_theta**(pair_index/32))
    wavelens = (2*math.pi)/speeds

    fast_pairs = wavelens  < wavelen_high
    slow_pairs = wavelens > wavelen_low
    middle = ~fast_pairs & ~slow_pairs
    smooth = (original_embeddings/wavelens - low_freq_factor) / (high_freq_factor - low_freq_factor)
    blended = (1 - smooth) * speeds / factor + smooth * speeds

    new_speeds = torch.where(slow_pairs, speeds / factor, speeds)
    new_speeds = torch.where(middle, blended, new_speeds)

    #angles = position * speeds

    num_tokens = tensor.shape[-2]
    positions = torch.arange(generated_len, generated_len + num_tokens, dtype=torch.float32, device=tensor.device)
    angles = torch.outer(positions, new_speeds)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    first_elements = pairs[:, :, :, 0, :]
    second_elements = pairs[:, :, :, 1, :]

    first_rotated = first_elements * cos - second_elements * sin
    second_rotated = first_elements * sin + second_elements * cos

    final_output = torch.cat([first_rotated, second_rotated], dim=-1)

    return final_output

def comparar(tensor_1, tensor_2):
    diferencias = (tensor_1 - tensor_2).abs()
    diferencia_max = diferencias.max().item()

    print(f"Diferencia máxima: {diferencia_max}")

    torch.testing.assert_close(tensor_1, tensor_2,atol=1e-4, rtol= 1e-5 )

#RMSNorm

def rmsnorm(tensor,weight):

    mean = torch.mean(tensor**2, dim=-1, keepdim=True)
    eps = 1e-5
    output = ((tensor)/torch.sqrt(mean+eps))* weight
    return output

#SwiGLU Implementation

def mlp_swiglu(tensor, gate_weights, up_weights, down_weights):
    gated = tensor @ gate_weights.transpose(-2,-1)
    up = tensor @ up_weights.transpose(-2,-1)
    silu = fu.silu(gated)
    inter = up * silu
    output = inter @ down_weights.transpose(-2,-1)

    return output

#Implementing the attention layer

def decode_attention(q, k_cache, v_cache, seq_len):

    q_heads = model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim

    k = k_cache[:, :, :seq_len , :]
    v = v_cache[:, :, :seq_len , :]

    k = torch.repeat_interleave(k, q_heads // kv_heads, dim=1)
    v = torch.repeat_interleave(v, q_heads // kv_heads, dim=1)

    numerator = (q.unsqueeze(2) @ k.transpose(-2,-1)).squeeze(2)

    denominator = head_dim ** 0.5
    inter_results = numerator/denominator

    softmax_res = torch.softmax(inter_results, dim=-1)
    output_1 = (softmax_res.unsqueeze(2) @ v).squeeze(2)

    return output_1


def attention(tensor, kv_cache, layer, generated_len,  q_proj, k_proj, v_proj, weight_out):

    q_heads = model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    number_of_tokens = tensor.shape[1]

    q = (tensor @ q_proj.transpose(-2, -1))
    q = q.view(*q.shape[:-1],q_heads,head_dim)
    q = q.transpose(-3,-2)

    k = (tensor @ k_proj.transpose(-2, -1))
    k = k.view(*k.shape[:-1],kv_heads,head_dim)
    k = k.transpose(-3,-2)

    v = (tensor @ v_proj.transpose(-2, -1))
    v = v.view(*v.shape[:-1],kv_heads,head_dim)
    v = v.transpose(-3,-2)

    #Applying RoPE
    q = rope(q, generated_len)
    k = rope(k, generated_len)

    kv_cache[layer,0, :, :, generated_len:generated_len + number_of_tokens, :] = k
    kv_cache[layer,1, :, :, generated_len:generated_len + number_of_tokens, :] = v


    new_token_len = number_of_tokens + generated_len
    k = kv_cache[layer,0,:,:,:new_token_len,:]
    v = kv_cache[layer,1,:,:,:new_token_len,:]

    if number_of_tokens > 1:

        k = torch.repeat_interleave(k, q_heads // kv_heads, dim=1)
        v = torch.repeat_interleave(v, q_heads // kv_heads, dim=1)

        numerator = q @ k.transpose(-2,-1)
        denominator = head_dim ** 0.5
        inter_results = numerator/denominator
        mask = torch.tril(torch.ones(k[0].shape[-2],k[0].shape[-2], device=tensor.device))
        inter_results = inter_results.masked_fill(mask==0, float("-inf"))
        softmax_res = torch.softmax(inter_results, dim=-1)
        output_1 = softmax_res @ v
        output_1 = output_1.transpose(-3,-2)
        output_1 = output_1.reshape(*output_1.shape[:-2],q_heads*head_dim)
        output_2 = output_1@weight_out.transpose(-2,-1)



        return output_2

    output_1 = decode_attention(q[:,:,0,:], kv_cache[layer,0] ,kv_cache[layer,1] , new_token_len)
    output_1 = output_1.unsqueeze(2)
    output_1 = output_1.transpose(-3,-2)
    output_1 = output_1.reshape(*output_1.shape[:-2],q_heads*head_dim)

    output_2 = output_1@weight_out.transpose(-2, -1)


    return output_2

print(model)

#Full Block Implementation

def full_layer_block(tensor, kv_cache, layer, generated_len, w_rms_1, w_q_proj, w_k_proj, w_v_proj, w_out, w_rms_2, w_gate, w_up, w_down):

    inter_1 = rmsnorm(tensor,w_rms_1)
    inter_1 = attention(inter_1,kv_cache, layer, generated_len, w_q_proj, w_k_proj, w_v_proj, w_out)

    res_1 = tensor + inter_1

    inter_2 = rmsnorm(res_1,w_rms_2)
    inter_2 = mlp_swiglu(inter_2,w_gate,w_up,w_down)


    res_2 = res_1 + inter_2

    return res_2

def llama_model(input_ids, kv_cache, generated_len):


    embeddings = embeddings_from_ids(input_ids)
    for layer in range (16):
        model_layer = model.model.layers[layer]
        embeddings = full_layer_block(embeddings, kv_cache, layer, generated_len, model_layer.input_layernorm.weight, model_layer.self_attn.q_proj.weight, model_layer.self_attn.k_proj.weight, model_layer.self_attn.v_proj.weight, model_layer.self_attn.o_proj.weight,model_layer.post_attention_layernorm.weight, model_layer.mlp.gate_proj.weight, model_layer.mlp.up_proj.weight, model_layer.mlp.down_proj.weight)

    embeddings = rmsnorm(embeddings, model.model.norm.weight)

    output = embeddings @ model.lm_head.weight.transpose(-2,-1)

    return output

#My first generation implementation, and its greedy to make things easier for myself

def greedy_generation(input_ids, num_tokens):

    #Declaring my KV cache
    layers = 16
    num_heads = 32
    head_dim = 2048 // num_heads
    batch = input_ids.shape[0]
    max_len = 8192

    #Head dim last, don't want trouble with coalescing

    kv_cache = torch.zeros(layers, 2, batch, num_heads // 4, max_len, head_dim, device=device)


    current_len = input_ids.shape[1]

    calculated_logits = llama_model(input_ids,kv_cache, 0)




    for _ in range(num_tokens):

        last_tokens_logits = calculated_logits[:,-1,:]

        biggest_logit, logit_pos = last_tokens_logits.max(dim=-1)

        input_ids = torch.cat([input_ids, logit_pos.unsqueeze(-1)], dim=-1)

        calculated_logits = llama_model(logit_pos.unsqueeze(-1), kv_cache, current_len)

        current_len += 1

    print(tokenizer.decode(input_ids[0]))

    return input_ids

ids = text_into_ids("How ")
greedy_generation(ids,15)

my_answer = greedy_generation(ids, 20)
official_answer = model.generate(ids, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(torch.equal(my_answer, official_answer))

#Logits tests
kv_cache = torch.zeros(16, 2, ids.shape[0], 32 // 4, 8192, 2048 // 32, device=device)
mi_modelo = llama_model(ids,kv_cache,0)
llama_hf = model(ids).logits
comparar(mi_modelo, llama_hf)