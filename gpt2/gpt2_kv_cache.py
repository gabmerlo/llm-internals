import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as fu

torch.set_grad_enabled(False)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

#Embeddings

texts = ["Hello, good morning"]


def get_embeddings(input):
    wpe = model.transformer.wpe.weight
    wte = model.transformer.wte.weight
    encoded = tokenizer(input, return_tensors="pt")
    ids_input = encoded["input_ids"]
    num_tokens = ids_input.shape[1]
    positions = torch.arange(num_tokens)
    embeddings = wpe[positions] + wte[ids_input]

    return embeddings


def embeddings_from_ids(ids, len_previous):
    num_tokens = ids.shape[1]
    positions = torch.arange(start=len_previous, end=len_previous+num_tokens ,device=ids.device)
    wpe = model.transformer.wpe.weight
    wte = model.transformer.wte.weight
    embeddings = wpe[positions] + wte[ids]

    return embeddings


def text_into_ids(text):
    ids = tokenizer.encode(text, return_tensors="pt")
    return ids

embeddings = get_embeddings(texts)


def comparar(tensor_1, tensor_2):
    diferencias = (tensor_1 - tensor_2).abs()
    diferencia_max = diferencias.max().item()

    print(f"Diferencia máxima: {diferencia_max}")

    torch.testing.assert_close(tensor_1, tensor_2,atol=1e-4, rtol= 1e-5 )



#LayerNorm

def layernorm(tensor,weight,bias):
    mean = torch.mean(tensor, dim=-1, keepdim=True)
    eps = 1e-5
    variance = torch.var(tensor, dim=-1, keepdim=True, correction=0)

    output = ((tensor-mean)/torch.sqrt(variance+eps))* weight + bias
    return output


def mlp(tensor, weight_1, bias_1, weight_2, bias_2):
    inter = tensor @ weight_1 + bias_1
    inter_2 = fu.gelu(inter, approximate="tanh")
    output = inter_2 @ weight_2 + bias_2

    return output



#Implementing the attention layer

def attention(tensor, kv_cache, layer, generated_len,  weight_1, bias_1, weight_2, bias_2):
    n_heads = 12
    head_dim = 768 // n_heads
    number_of_tokens = tensor.shape[1]

    q,k,v = ((tensor @ weight_1) + bias_1).split(768, dim=-1)
    q = q.view(*q.shape[:-1],n_heads,head_dim)
    q = q.transpose(-3,-2)
    k = k.view(*k.shape[:-1],n_heads,head_dim)
    k = k.transpose(-3,-2)
    kv_cache[layer,0, :, :, generated_len:generated_len + number_of_tokens, :] = k

    v = v.view(*v.shape[:-1],n_heads,head_dim)
    v = v.transpose(-3,-2)
    kv_cache[layer,1, :, :, generated_len:generated_len + number_of_tokens, :] = v


    new_token_len = number_of_tokens + generated_len
    k = kv_cache[layer,0,:,:,:new_token_len,:]
    v = kv_cache[layer,1,:,:,:new_token_len,:]


    numerator = q @ k.transpose(-2,-1)
    denominator = head_dim ** 0.5
    inter_results = numerator/denominator
    if number_of_tokens > 1:
        mask = torch.tril(torch.ones(k[0].shape[-2],k[0].shape[-2], device=tensor.device))
        inter_results = inter_results.masked_fill(mask==0, float("-inf"))

    softmax_res = torch.softmax(inter_results, dim=-1)
    output_1 = softmax_res @ v

    output_1 = output_1.transpose(-3,-2)
    output_1 = output_1.reshape(*output_1.shape[:-2],n_heads*head_dim)

    output_2 = output_1@weight_2 + bias_2

    return output_2

    #Full Block Implementation

def full_layer_block(tensor, kv_cache, layer, generated_len, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6):

    inter_1 = layernorm(tensor,w1,b1)
    inter_1 = attention(inter_1,kv_cache, layer, generated_len, w2,b2,w3,b3)

    res_1 = tensor + inter_1

    inter_2 = layernorm(res_1,w4,b4)
    inter_2 = mlp(inter_2,w5,b5,w6,b6)

    res_2 = res_1 + inter_2

    return res_2


def gpt2_model(input_ids, kv_cache, generated_len):


    embeddings = embeddings_from_ids(input_ids, generated_len)
    for layer in range (12):
        embeddings = full_layer_block(embeddings, kv_cache, layer, generated_len, model.transformer.h[layer].ln_1.weight,model.transformer.h[layer].ln_1.bias, model.transformer.h[layer].attn.c_attn.weight, model.transformer.h[layer].attn.c_attn.bias, model.transformer.h[layer].attn.c_proj.weight, model.transformer.h[layer].attn.c_proj.bias,model.transformer.h[layer].ln_2.weight, model.transformer.h[layer].ln_2.bias, model.transformer.h[layer].mlp.c_fc.weight, model.transformer.h[layer].mlp.c_fc.bias, model.transformer.h[layer].mlp.c_proj.weight, model.transformer.h[layer].mlp.c_proj.bias)

    embeddings = layernorm(embeddings, model.transformer.ln_f.weight, model.transformer.ln_f.bias)

    output = embeddings @ model.lm_head.weight.transpose(-2,-1)


    return output

#My first generation implementation, and its greedy to make things easier for myself

def greedy_generation(input_ids, num_tokens):
    
    #Declaring my KV cache
    layers = 12
    num_heads = 12
    head_dim = 768 // 12
    batch = input_ids.shape[0]
    max_len = 1024

    kv_cache = torch.zeros(layers, 2, batch, num_heads, max_len, head_dim)

    

    current_len = input_ids.shape[1]
    
    calculated_logits = gpt2_model(input_ids,kv_cache, 0)


    for _ in range(num_tokens):

        last_tokens_logits = calculated_logits[:,-1,:]

        biggest_logit, logit_pos = last_tokens_logits.max(dim=-1)

        input_ids = torch.cat([input_ids, logit_pos.unsqueeze(-1)], dim=-1)

        calculated_logits = gpt2_model(logit_pos.unsqueeze(-1), kv_cache, current_len)

        current_len += 1

    print(tokenizer.decode(input_ids[0]))

    return input_ids



