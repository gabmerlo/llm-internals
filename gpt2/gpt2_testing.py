import gpt2model
import gpt2_generation

#Forward pass test

torch.manual_seed(42)
ids_prueba = torch.randint(low=0, high=50257, size=(4,16))
mi_resultado_gpt = gpt2_model(ids_prueba)
gpt_oficial = model(ids_prueba).logits
comparar(mi_resultado_gpt,gpt_oficial)


#Generation test

ids = text_into_ids("I love")
greedy_generation(ids,15)

my_answer = greedy_generation(ids, 20)
gpt2_official_generation = model.generate(ids, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(torch.equal(mine, hf))