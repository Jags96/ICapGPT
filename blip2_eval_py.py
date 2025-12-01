CHECKPOINT_PATH = "./checkpoint_9.pth"
VALIDATION_JSON = "/Users/jagathkumarreddyk/Documents/GitHub/BLIP/annotations_trainval2017/annotations/captions_val2017.json"
VALIDATION_IMAGE_ROOT = "/Users/jagathkumarreddyk/Documents/GitHub/BLIP/val2017/val2017"
NUM_SAMPLE = None #10
BATCH_SIZE = 64 ## Doesn't matter, because we aren't using it
OUTPUT_FILE_NAME = "./output_json_1.json" ### ./output_json


import torch
import json
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np



transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])
class COCOCaptionDataset(Dataset):
    """COCO Captions Dataset"""
    def __init__(self, json_path, image_root, transform, tokenizer, max_length=50):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.image_root = image_root
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Flatten annotations
        self.samples = []
        for item in tqdm(self.data['annotations']):
            img_id = item['image_id']
            # if len(self.samples)==200: break
            # Find image filename
            img_info = next(img for img in self.data['images'] if img['id'] == img_id)
            self.samples.append({
                'image': os.path.join(image_root, img_info['file_name']),
                'caption': item['caption']
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and transform image
        image = Image.open(sample['image']).convert('RGB')
        image = self.transform(image)
        
        # Tokenize caption for encoder
        caption = sample['caption']
        text_encoding = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create decoder inputs (shifted right)
        decoder_input_ids = text_encoding['input_ids'].clone()
        labels = text_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'image': image,
            'image_path': sample['image'],
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'decoder_input_ids': decoder_input_ids.squeeze(0),
            'decoder_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }




# =======================================================
# 7. Load tokenizer + model
# =======================================================
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

# Add special <img> token
if '<img>' not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({'additional_special_tokens':['<img>']})
img_token_id = tokenizer.convert_tokens_to_ids('<img>')

# Load DistilGPT2 via AutoModelForCausalLM
gpt2 = AutoModelForCausalLM.from_pretrained("distilgpt2")
gpt2.resize_token_embeddings(len(tokenizer))
gpt2.eval()

# =======================================================
# 8. Q-Former
# =======================================================
from transformers import ViTModel
vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit.eval()

class QFormer(nn.Module):
    def __init__(self, image_emb_dim, prompt_len=16, hidden_dim=768):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(prompt_len, image_emb_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=image_emb_dim, num_heads=8)
        self.mlp = nn.Linear(image_emb_dim, hidden_dim)

    def forward(self, image_embeds):
        batch_size = image_embeds.size(0)
        query = self.query_tokens.unsqueeze(1).repeat(1,batch_size,1)
        attn_out,_ = self.cross_attn(query, image_embeds.transpose(0,1), image_embeds.transpose(0,1))
        prompt = self.mlp(attn_out).transpose(0,1)
        return prompt


q_former = QFormer(image_emb_dim=vit.config.hidden_size, prompt_len=16, hidden_dim=gpt2.config.n_embd)

# =======================================================
# 9. DataLoader
# =======================================================
val_dataset = COCOCaptionDataset(json_path=VALIDATION_JSON, image_root=VALIDATION_IMAGE_ROOT, transform=transform,tokenizer=tokenizer, max_length=50)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# =======================================================
# 10. Device setup
# =======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit.to(device)
gpt2.to(device)
q_former.to(device)


def load_from_checkpoint(CHECKPOINT_PATH):
    # with open(CHECKPOINT_PATH, 'r') as f:
    checkpoint_obj = torch.load(CHECKPOINT_PATH)
    gpt2.load_state_dict(checkpoint_obj["gpt2_state"])
    q_former.load_state_dict(checkpoint_obj["qformer"])

load_from_checkpoint(CHECKPOINT_PATH)



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
    



sample_output_json = []

if NUM_SAMPLE == None:
    NUM_SAMPLE = len(val_dataset)

def remove_padding(padded_tokens):
    for i in range(len(padded_tokens)-1):
        if (padded_tokens[i] == 50256) and (padded_tokens[i+1] == 50256):
            break

    return padded_tokens[:i]


for i in tqdm(range(NUM_SAMPLE)):
    this_dict = {}
    img = (((val_dataset[i]['image'] + 1)/2).mul(255)).byte()
    img = img.permute(1,2,0)
    # print(img.shape)
    img = Image.fromarray(img.numpy())
    gt_caption = remove_padding(val_dataset[i]['text_input_ids'])
    gt_caption = tokenizer.decode(gt_caption)
    gen_caption = generate_caption(img, vit, q_former, gpt2, tokenizer, device)
    
    this_dict['img_id'] = int(str(val_dataset[0]['image_path']).split(".")[0].split("/")[-1])
    this_dict["image_path"] = val_dataset[i]['image_path']
    this_dict["captions"] = gt_caption
    this_dict["generated_output"]  = gen_caption
    
    sample_output_json.append(this_dict)

    # print("IMAGE_PATH",val_dataset[i]['image_path'])
    # print("GT caption: " + gt_caption)
    # print("Generated caption: " + gen_caption,"\n\n")

with open(OUTPUT_FILE_NAME, 'w') as f:
    json.dump(sample_output_json,f)