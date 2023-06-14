# Adapt the model to be class-agnostic

from .mask_rcnn_fpn_discovery import model

# Set bbox localization head to be class-agnostic
model.roi_heads.box_predictor.cls_agnostic_bbox_reg = True