import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

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


def embeddings_from_ids(ids):
    num_tokens = ids.shape[1]
    positions = torch.arange(num_tokens, device=ids.device)
    wpe = model.transformer.wpe.weight
    wte = model.transformer.wte.weight
    embeddings = wpe[positions] + wte[ids]

    return embeddings

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


#MLP Implementation
import torch.nn.functional as fu

def mlp(tensor, weight_1, bias_1, weight_2, bias_2):
    inter = tensor @ weight_1 + bias_1
    inter_2 = fu.gelu(inter, approximate="tanh")
    output = inter_2 @ weight_2 + bias_2

    return output

#Implementing the attention layer

def attention(tensor, weight_1, bias_1, weight_2, bias_2):
    n_heads = 12
    head_dim = 768 // n_heads
    q,k,v = ((tensor @ weight_1) + bias_1).split(768, dim=-1)
    q = q.view(*q.shape[:-1],n_heads,head_dim)
    q = q.transpose(-3,-2)
    k = k.view(*k.shape[:-1],n_heads,head_dim)
    k = k.transpose(-3,-2)
    v = v.view(*v.shape[:-1],n_heads,head_dim)
    v = v.transpose(-3,-2)

    numerator = q @ k.transpose(-2,-1)
    denominator = head_dim ** 0.5
    mask = torch.tril(torch.ones(k[0].shape[-2],k[0].shape[-2], device=tensor.device))
    inter_mask = (numerator/denominator).masked_fill(mask==0, float("-inf"))
    softmax_res = torch.softmax(inter_mask, dim=-1)
    output_1 = softmax_res @ v

    output_1 = output_1.transpose(-3,-2)
    output_1 = output_1.reshape(*output_1.shape[:-2],n_heads*head_dim)

    output_2 = output_1@weight_2 + bias_2

    return output_2

    #Full Block Implementation

def full_layer_block(tensor, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6):

    inter_1 = layernorm(tensor,w1,b1)
    inter_1 = attention(inter_1,w2,b2,w3,b3)

    res_1 = tensor + inter_1

    inter_2 = layernorm(res_1,w4,b4)
    inter_2 = mlp(inter_2,w5,b5,w6,b6)

    res_2 = res_1 + inter_2

    return res_2

def gpt2_model(input_ids):

    embeddings = embeddings_from_ids(input_ids)
    for layer in range (12):
        embeddings = full_layer_block(embeddings,model.transformer.h[layer].ln_1.weight,model.transformer.h[layer].ln_1.bias, model.transformer.h[layer].attn.c_attn.weight, model.transformer.h[layer].attn.c_attn.bias, model.transformer.h[layer].attn.c_proj.weight, model.transformer.h[layer].attn.c_proj.bias,model.transformer.h[layer].ln_2.weight, model.transformer.h[layer].ln_2.bias, model.transformer.h[layer].mlp.c_fc.weight, model.transformer.h[layer].mlp.c_fc.bias, model.transformer.h[layer].mlp.c_proj.weight, model.transformer.h[layer].mlp.c_proj.bias)

    embeddings = layernorm(embeddings, model.transformer.ln_f.weight, model.transformer.ln_f.bias)

    output = embeddings @ model.lm_head.weight.transpose(-2,-1)


    return output


