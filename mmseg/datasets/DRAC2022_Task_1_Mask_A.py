
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DRACDataset_Mask_A(CustomDataset):

    CLASSES = ('Background', 'Intraretinal_microvascular_abnormals', 'Neovascularization','Intersection_of_Both' )

    PALETTE = [[20, 20, 20], [30, 30, 30],[40, 40, 40],[45, 45, 45]]

    def __init__(self, **kwargs):
        super(DRACDataset_Mask_A, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
