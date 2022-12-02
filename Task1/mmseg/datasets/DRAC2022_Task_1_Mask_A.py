
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DRACDataset_Mask_A(CustomDataset):

    CLASSES = ('Background', 'Intraretinal_microvascular_abnormals', 'Neovascularization','Intersection_of_Both' )

    PALETTE = [[125, 125, 125], [20, 10, 130], [140, 140, 40], [140, 40, 60]]


    def __init__(self, **kwargs):
        super(DRACDataset_Mask_A, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
