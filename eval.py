from clip_metrics import *
import imageio.v2 as imageio
from tqdm import tqdm
import os

NERF = "original"
IN2N = "tuxedo/in2n"
SN2N = "tuxedo/sn2n"
text_0 = "photograph of a man"
text_1 = "photograph of a main in tuxedo"



paths0 = sorted(os.listdir(f"renders/{NERF}"))
paths1 = paths0[1:] + paths0[:1]

C = ClipSimilarity()

text_features_0 = C.encode_text(text_0)
text_features_1 = C.encode_text(text_1)

results = {'sim_direction_1': [], 'sim_consistency_1': [], 'sim_direction_2': [], 'sim_consistency_2': []}

for fname0, fname1 in tqdm(zip(paths0, paths1)):
    image_0 = torch.FloatTensor(imageio.imread(f"renders/{NERF}/{fname0}")).unsqueeze(dim=0).permute(0, 3, 1, 2) / 255
    image_1 = torch.FloatTensor(imageio.imread(f"renders/{IN2N}/{fname0}")).unsqueeze(dim=0).permute(0, 3, 1, 2) / 255
    image_2 = torch.FloatTensor(imageio.imread(f"renders/{SN2N}/{fname0}")).unsqueeze(dim=0).permute(0, 3, 1, 2) / 255
    image_3 = torch.FloatTensor(imageio.imread(f"renders/{NERF}/{fname1}")).unsqueeze(dim=0).permute(0, 3, 1, 2) / 255
    image_4 = torch.FloatTensor(imageio.imread(f"renders/{IN2N}/{fname1}")).unsqueeze(dim=0).permute(0, 3, 1, 2) / 255
    image_5 = torch.FloatTensor(imageio.imread(f"renders/{SN2N}/{fname1}")).unsqueeze(dim=0).permute(0, 3, 1, 2) / 255
    
    
    image_features_0 = C.encode_image(image_0)
    image_features_1 = C.encode_image(image_1)
    image_features_2 = C.encode_image(image_2)
    image_features_3 = C.encode_image(image_3)
    image_features_4 = C.encode_image(image_4)
    image_features_5 = C.encode_image(image_5)

    sim_direction_1 = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
    sim_consistency_1 = F.cosine_similarity(image_features_1 - image_features_0, image_features_4 - image_features_3)
    sim_direction_2 = F.cosine_similarity(image_features_2 - image_features_0, text_features_1 - text_features_0)
    sim_consistency_2 = F.cosine_similarity(image_features_2 - image_features_0, image_features_5 - image_features_3)
    
    results['sim_direction_1'].append(sim_direction_1)
    results['sim_consistency_1'].append(sim_consistency_1)
    results['sim_direction_2'].append(sim_direction_2)
    results['sim_consistency_2'].append(sim_consistency_2)

print(torch.cat(results['sim_direction_1'], 0).mean())
print(torch.cat(results['sim_consistency_1'], 0).mean())
print(torch.cat(results['sim_direction_2'], 0).mean())
print(torch.cat(results['sim_consistency_2'], 0).mean())