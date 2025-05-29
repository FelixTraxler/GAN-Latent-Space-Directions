import torch
from PIL import Image
import open_clip
import numpy as np


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')


###################### oc1 ######################
# outdir = "out_bakk"
# seed_len = 1000
# for seed_idx, seed in enumerate(range(seed_len)):
#     output_path = f'{outdir}/seed{(seed+1):04d}.png'
#     image = preprocess(Image.open(output_path)).unsqueeze(0)
#     text = tokenizer(["pose", "smile", "age", "gender", "eyeglasses"])


#     with torch.no_grad(), torch.autocast("mps"):
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)

#         text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#         if (seed % 50 == 0):
#             print(seed)
        
#         np.save(f'{outdir}/seed{seed:04d}_oc1.npy', text_probs.numpy())
###################### oc1 ######################


###################### 256 oc1 ######################
# outdir = "out_bakk_256"
# seed_start  = 0
# seed_end    = 10000

# for seed_idx, seed in enumerate(range(seed_start, seed_end)):
#     output_path = f'{outdir}/seed{(seed):06d}.png'
#     image = preprocess(Image.open(output_path)).unsqueeze(0)
#     text = tokenizer(["pose", "smile", "age", "gender", "eyeglasses", "old", "young" ])

#     with torch.no_grad(), torch.autocast("mps"):
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)

#         text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#         if (seed % 50 == 0):
#             print(f'{seed}/{seed_end}')
        
#         np.save(f'{outdir}/seed{seed:04d}_oc1.npy', text_probs.numpy())
        # print(np.load(f'{outdir}/seed{seed:04d}_oc1.npy'))
###################### 256 oc1 ######################

###################### 256 oc2 ######################
outdir = "out_bakk_256"
seed_start  = 100000
seed_end    = 700000

text = tokenizer([
        "pose", "smile", "age", "gender", 
        "old", "young", 
        "no eyeglasses", "eyeglasses", 
        "male", "female",
        "looking left", "looking right",
        "blond", "brunette", "red hair",
        "Black hair", "Blond hair", "Brown hair", "Gray hair", "Red hair", "Other hair color",
        "Receding hairline", "No receding hairline", "Full hair",
        "bangs", "straight hair", "curly hair",
        "Big nose", "Small nose",
        "Big lips", "Thin lips",
        "Goatee", "Mustache", "Shaven", "No beard",
        "smiling", "mean", 
        "Pale skin tone", "Not pale skin tone"
        "pale", "not pale",
    ])

text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)

for seed_idx, seed in enumerate(range(seed_start, seed_end)):
    output_path = f'{outdir}/seed{(seed):06d}.png'
    image = preprocess(Image.open(output_path)).unsqueeze(0)
   
    with torch.no_grad(), torch.autocast("mps"):
        image_features = model.encode_image(image)
        # TODO: save without norming
        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # TODO: only save image features, rest in extra script
        # text_probs = (100.0 * image_features @ text_features.T)

        #  Probability for binary classifier "old":"young"
        # print(text_probs[:, 4:6].softmax(dim=-1))

        if (seed % 50 == 0):
            print(f'{seed}/{seed_end}')
        
        # np.save(f'{outdir}/seed{seed:06d}_oc2.npy', text_probs.numpy())
        np.save(f'{outdir}/seed{seed:06d}_raw_image_features.npy', image_features.numpy())

        # print(np.load(f'{outdir}/seed{seed:04d}_oc1.npy'))
###################### 256 oc2 ######################