import gpt2model

#My first generation implementation, and its greedy to make things easier for myself

def greedy_generation(input_ids, num_tokens):
    
    for _ in range(num_tokens):
        
        calculated_logits = gpt2_model(input_ids)

        last_tokens_logits = calculated_logits[:,-1,:]

        biggest_logit, logit_pos = last_tokens_logits.max(dim=-1)

        input_ids = torch.cat([input_ids, logit_pos.unsqueeze(-1)], dim=-1)

    print(tokenizer.decode(input_ids[0]))

    return input_ids

