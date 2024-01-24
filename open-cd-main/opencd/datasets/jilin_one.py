# Copyright (c) Open-CD. All rights reserved.
from opencd.registry import DATASETS
from .basecddataset import _BaseCDDataset


@DATASETS.register_module()
class JILIN_ONE(_BaseCDDataset):
    """JILIN_ONE"""
    METAINFO = dict(
        classes=('0', '1', '2', '3', '4', '5', '6', '7', '8'),
        # palette=[[0],[1],[2],[3],[4],[5],[6],[7],[8]],
        palette = [[255, 255, 0], [128, 128, 1], [130, 87, 2], [255, 0, 3], [0, 0, 4],[64,128,5],[64,128,6],[24,24,7],[100,200,8]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 format_seg_map=None,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            **kwargs)
