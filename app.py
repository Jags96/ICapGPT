import streamlit as st
from PIL import Image

from apputil import *

import torch
import json
import os
from torch import nn
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTModel



CHECKPOINT_PATH = "./checkpoint_batch_srun2999_epoch12.pth"
OUTPUT_JSON_PATH = "/Users/jagathkumarreddyk/Downloads/CV/BLIP2-Output/output_json_12_epoch_model_nov29.json"



transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

@st.cache_resource
def load_model_components():
    """Loads and caches all heavy model components."""
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    if '<img>' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<img>']})
    
    img_token_id = tokenizer.convert_tokens_to_ids('<img>')

    gpt2 = AutoModelForCausalLM.from_pretrained("distilgpt2")
    gpt2.resize_token_embeddings(len(tokenizer))
    gpt2.eval()

    vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    vit.eval()

    class QFormer(nn.Module):
        def __init__(self, image_emb_dim, prompt_len=16, hidden_dim=768):
            super().__init__()
            self.query_tokens = nn.Parameter(torch.randn(prompt_len, image_emb_dim))
            self.cross_attn = nn.MultiheadAttention(embed_dim=image_emb_dim, num_heads=8, batch_first=False)
            self.mlp = nn.Linear(image_emb_dim, hidden_dim)

        def forward(self, image_embeds):
            batch_size = image_embeds.size(0)
            query = self.query_tokens.unsqueeze(1).repeat(1, batch_size, 1)
            # Transpose for MultiheadAttention (SeqLen, Batch, EmbedDim)
            attn_out, _ = self.cross_attn(query, image_embeds.transpose(0, 1), image_embeds.transpose(0, 1))
            prompt = self.mlp(attn_out).transpose(0, 1) # Back to (Batch, SeqLen, EmbedDim)
            return prompt

    q_former = QFormer(image_emb_dim=vit.config.hidden_size, prompt_len=16, hidden_dim=gpt2.config.n_embd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit.to(device)
    gpt2.to(device)
    q_former.to(device)

    try:
        checkpoint_obj = torch.load(CHECKPOINT_PATH, map_location=device)
        gpt2.load_state_dict(checkpoint_obj["gpt2_state"])
        q_former.load_state_dict(checkpoint_obj["qformer"])
    except FileNotFoundError:
        st.error(f"Checkpoint file not found at {CHECKPOINT_PATH}. Models are initialized randomly.")
    except KeyError:
        st.error("Checkpoint keys missing. Models are initialized randomly.")


    return vit, q_former, gpt2, tokenizer, img_token_id, device

vit, q_former, gpt2, tokenizer, img_token_id, device = load_model_components()

def generate_caption(pil_image, vit, q_former, gpt2, tokenizer, device, max_length=30, top_k = None):
    vit.eval()
    q_former.eval()
    gpt2.eval()
    image = transform(pil_image).to(device) ##.unsqueeze(0)
    image = image.reshape([1,*list(image.shape)])
    st.info("Generating caption... This may take a moment.")
    with torch.no_grad():
        image_embeds = vit(image).last_hidden_state
        prompts = q_former(image_embeds)
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
        img_token_emb = gpt2.transformer.wte(torch.tensor([[img_token_id]], device=device))
        generated = []

        prefix_len = 1 + prompts.size(1)
        print(f"prefix_len:{prefix_len}")
    
        for _ in tqdm(range(max_length)):
            gpt2_inputs = gpt2.transformer.wte(input_ids)
            gpt2_inputs = torch.cat([img_token_emb, prompts, gpt2_inputs], dim=1)


            current_seq_len = gpt2_inputs.size(1)
            # print(f"current-seq-len:{gpt2_inputs.shape}")
            position_ids = torch.arange(current_seq_len, dtype=torch.long, device=device).unsqueeze(0)

            outputs = gpt2(inputs_embeds=gpt2_inputs, position_ids=position_ids)
            logits = outputs.logits[0, -1, :]

            # Top-k + top-p sampling
            filtered_logits = torch.nn.functional.softmax(logits, dim=-1)
            if top_k == None:
                next_token = torch.multinomial(filtered_logits, num_samples=1)
            else:
                next_token =  top_k_sample(logits, top_k)


            if next_token.item() == tokenizer.eos_token_id:
                break
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        caption = tokenizer.decode(generated, skip_special_tokens=True)
        return caption


data_list = get_json(OUTPUT_JSON_PATH)

if "idx" not in st.session_state:
    st.session_state.idx = 0

st.set_page_config(
    page_title="Vision-Language Model Demo",
    page_icon="📸",
    layout="wide", # Use wide layout for more screen space
    initial_sidebar_state="expanded"
)



st.title("📸 Image Captioning Demonstration")
st.markdown("---")

# 4. Upload Section (Consistent UI)
## Use st.header or st.subheader for consistent section titles
st.header("Upload Your Own Image")
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Use columns to present the uploaded image and output cleanly
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Generated Caption")
        # Use a consistent button style
        if st.button("Generate Caption", key="generate_upload", use_container_width=True):
# generate_caption(pil_image, vit, q_former, gpt2, tokenizer, device, max_length=30)
            caption_output = generate_caption(image, vit, q_former, gpt2, tokenizer, device,max_length = 30,top_k=50)
            st.success("Generation Complete! 🎉")
            st.markdown(f"Generated Caption: \n{caption_output}")
            st.balloons()
            
st.markdown("---")


st.header("COCO Validation Dataset (val2017) Results Explorer")

# Load data only once
data_list = get_json(OUTPUT_JSON_PATH)

# Initialize session state index if not already present
if "idx" not in st.session_state:
    st.session_state.idx = 0

# Function to display the image and captions
def display_coco_item(data_item):
    st.subheader(f"Image ID: `{data_item['img_id']}`")
    
    # Use columns for image and text details
    img_col, text_col = st.columns(2)
    
    with img_col:
        try:
            # Use a better image source if possible, falling back to address function
            img_path = data_item.get("image_path")
            if img_path and os.path.exists(img_path):
                 st.image(img_path, caption=f"Image ID: {data_item['img_id']}", use_container_width=True)
            else:
                 # Using the image_address utility (e.g., placeholder web image)
                 st.image(image_address(data_item["img_id"]), caption=f"Image ID: {data_item['img_id']}", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load image: {e}")

    with text_col:
        st.markdown("#### Generated Caption (Model Output)")
        # Use st.code or a specific callout for generated text
        st.text(data_item["generated_output"])
        
        st.markdown("#### Ground Truth Captions (Reference)")
        # Display GT captions as a list for readability
        gt_captions = data_item["captions"]
        st.text(gt_captions)
        roughL = ROUGE_L(gt_captions, data_item["generated_output"])
        st.markdown(f"`ROUGE-L` : {roughL:.4f}")
        # for i, caption in enumerate(gt_captions):
            # st.write(f"• {caption}")

# Button logic
col_prev, col_next, col_show = st.columns([1, 1, 1])

# Initial display on first load
if st.session_state.idx == 0:
    display_coco_item(get_item(data_list, st.session_state.idx))


with col_prev:
    if st.button("⬅️ Previous", use_container_width=True, key="prev_button"):
        st.session_state.idx = (st.session_state.idx - 1) % len(data_list)
        # Re-run the app to update the display
        st.rerun()

with col_next:
    if st.button("Next ➡️", use_container_width=True, key="next_button"):
        st.session_state.idx = (st.session_state.idx + 1) % len(data_list)
        # Re-run the app to update the display
        st.rerun()

# This is a cleaner way to handle the button logic to update the state
# It replaces your original "Show" and "Next" buttons by updating state and rerunning
if col_next.button("Random Image 🎲", use_container_width=True, key="random_button"):
    import random
    st.session_state.idx = random.randint(0, len(data_list) - 1)
    st.rerun()
    
# Display the item after state change
if st.session_state.idx != 0 or len(data_list) > 0:
    display_coco_item(get_item(data_list, st.session_state.idx))


# if st.button("Show"):
#     st.session_state.idx+=1
#     data_item = get_item(data_list,idx = st.session_state.idx)
#     if data_item["img_id"] == 179765:
#         data_item["img_id"] = int(str(data_item["image_path"]).split("/")[-1].split(".")[0])
#     st.markdown(f"""
#             `Image ID`: {data_item["img_id"]}
#     """)

#     st.image(image_address(data_item["img_id"]))

#     # st.balloons()
#     st.text(f"Generated Captions : {data_item["generated_output"]}")
#     st.text(f"Original Captions : {data_item["captions"]}")


#     # st.badge("Increadible!!")
# if st.button("Next"):
#     st.session_state.idx+=1
#     data_item = get_item(data_list,st.session_state.idx)
#     st.markdown(f"""
#             `Image ID`: {data_item["img_id"]}
#     """)
#     try:
#         st.image(data_item["image_path"], caption=data_item["captions"])
#     except:
#         st.image(image_address(data_item["img_id"]), caption=data_item["captions"])

#     # st.balloons()
#     # st.text(f"Original Captions : {data_item["captions"]}")
#     st.text(f"`Generated Captions` : {data_item["generated_output"]}")



# st.title("Upload an Image")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#     caption_output = generate_caption(image, vit, q_former, gpt2, tokenizer, device, max_length=30)
#     st.balloons()
#     st.text(f"`Generated Caption`:{caption_output}")



