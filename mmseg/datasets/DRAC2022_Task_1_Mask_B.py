
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DRACDataset_Mask_B(CustomDataset):
    
    CLASSES = ('Background', 'Nonperfusion_areas')

    PALETTE = [[20, 20, 20], [50, 50, 50]]

    def __init__(self, **kwargs):
        superDRACDataset_Mask_Bself).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
