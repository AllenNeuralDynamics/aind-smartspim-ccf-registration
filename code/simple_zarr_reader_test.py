from aind_s3_cache.json_utils import get_json
from aind_zarr_utils.zarr import zarr_to_ants
import ants
from aind_ccf_reg.plots import plot_antsimgs, plot_reg

# metadata = get_json(r"/data/SmartSPIM_693196_2023-09-28_23-12-22_stitched_2024-01-11_10-23-15/metadata.nd.json")
# ants_image = zarr_to_ants("/data/SmartSPIM_693196_2023-09-28_23-12-22_stitched_2024-01-11_10-23-15/image_tile_fusing/OMEZarr/Ex_639_Em_660.zarr",
#                           nd_metadata = metadata,
#                           level = 3)
# print(ants_image)

# plot_antsimgs(ants_image,figpath = '/results/test.png')

template = ants.image_read(r"/data/smartspim_lca_template/smartspim_lca_template_25.nii.gz")
plot_antsimgs(template,figpath = '/results/template_test.png')
