#%%
import torch

n = 10000

pos, orientation = torch.rand((n, 3)), torch.rand((n, 3))
scalar_feature = torch.rand(n)

from lab_gatr import PointCloudPoolingScales
import torch_geometric as pyg
#%%
transform = PointCloudPoolingScales(rel_sampling_ratios=(1.0,), interp_simplex='triangle')
data = transform(pyg.data.Data(pos=pos, orientation=orientation, scalar_feature=scalar_feature))

from gatr.interface import embed_oriented_plane, extract_translation,embed_point

class GeometricAlgebraInterface:
    num_input_channels = num_output_channels = 1
    num_input_scalars = num_output_scalars = 1

    @staticmethod
    @torch.no_grad()
    def embed(data):

        multivectors = embed_point(coordinates=data.pos).view(-1, 1, 16)
        scalars = data.scalar_feature.view(-1, 1)

        return multivectors, scalars

    @staticmethod
    def dislodge(multivectors, scalars):
        output = extract_translation(multivectors).squeeze()

        return output
    
from lab_gatr import LaBGATr

model = LaBGATr(GeometricAlgebraInterface, d_model=8, num_blocks=10, num_attn_heads=4, use_class_token=False)
output = model(data)

print('works')
# %%
