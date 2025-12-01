import json
import os
import torch





def get_json(output_json_path):
    with open(output_json_path, 'r') as f:
        data = json.load(f)
    return data


def image_address(img_id, image_root = "/Users/jagathkumarreddyk/Documents/GitHub/BLIP/val2017/val2017"):
    for img_file in os.listdir(image_root):
        if str(img_id) in img_file:
            return os.path.join(image_root,img_file)
    


def get_item(data,idx):
    if idx >= len(data):
        idx=0
    return data[idx]



def generate_caption(pil_image, vit, q_former, gpt2, tokenizer, device, max_length=30, top_k=50, top_p=0.95):
    vit.eval()
    q_former.eval()
    gpt2.eval()
    image = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embeds = vit(image).last_hidden_state
        prompts = q_former(image_embeds)
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
        img_token_emb = gpt2.transformer.wte(torch.tensor([[img_token_id]], device=device))
        generated = []

        for _ in range(max_length):
            gpt2_inputs = gpt2.transformer.wte(input_ids)
            gpt2_inputs = torch.cat([img_token_emb, prompts, gpt2_inputs], dim=1)

            outputs = gpt2(inputs_embeds=gpt2_inputs)
            logits = outputs.logits[0, -1, :]

            # Top-k + top-p sampling
            filtered_logits = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(filtered_logits, num_samples=1)

            if next_token.item() == tokenizer.eos_token_id:
                break
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        caption = tokenizer.decode(generated, skip_special_tokens=True)
        return caption



def recur(i1,i2,s1,s2, curr):
    n1 = len(s1)
    n2 = len(s2)
    if (i1 >= n1) or (i2 >= n2):
        return curr
    ans1 = 0
    ans2 = 0
    ans3 = 0
    if s1[i1] == s2[i2]:
        ans1 = recur(i1+1, i2+1, s1, s2, curr+1)
    else:
        ans2 = recur(i1, i2+1, s1, s2, curr)
        ans3 = recur(i1+1, i2, s1, s2, curr)
    return max(ans1, ans2, ans3)


def F_harmonicMean(n1,n2,lcs,alpha=0.5):
    """_summary_

    Args:
        n1 (_type_): Original Text
        n2 (_type_): Generated Text
        lcs (_type_): LCS
        alpha (float, optional): Defaults to 0.5. 

    Returns:
        _type_: ROUGE-L
    """
    rougeL = lcs/((alpha)*(n1) + (1-alpha)*(n2))
    return rougeL

import torch
import torch.nn.functional as F

def top_k_sample(logits, k):
    # Get top-k values and indices
    topk_values, topk_indices = torch.topk(logits, k, dim=-1)
    
    # Convert to probabilities
    probs = F.softmax(topk_values, dim=-1)

    # Sample from the top-k distribution
    next_token = torch.multinomial(probs, num_samples=1)

    # Map back to original vocabulary index
    next_token = topk_indices.gather(-1, next_token)

    return next_token


def ROUGE_L(Sen1, Sen2):
    ## Apply Regrex to remove unnecessary symbols
    s1 =Sen1.lower().split(" ")
    s2 =Sen2.lower().split(" ")
    n1 = len(s1)
    n2 = len(s2)
    i1 = 0
    i2 = 0
    curr = 0
    lcs = recur(i1,i2,s1,s2,curr)
    return F_harmonicMean(n1,n2,lcs)
