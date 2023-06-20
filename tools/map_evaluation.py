from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os

# 예측 결과 파일과 정답 파일 경로
predictions_file = "/home/workspace/RNCDL/more_train_40000/COCO_instances_results.json"
annotations_file = "/home/data/Dataset/coco2017/annotations/coco_half_val.json"

def MAP_evaluation(predictions_file, annotations_file):
    # COCO 객체를 생성하고 정답 파일을 로드합니다.
    coco_gt = COCO(annotations_file)
    coco_dt = coco_gt.loadRes(predictions_file)
    # COCO 평가 객체를 생성하고 평가를 수행합니다.
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # 평가 결과 출력
    print('AP:', coco_eval.stats[0])
    print('AP50:', coco_eval.stats[1])
    print('AP75:', coco_eval.stats[2])
    print('AP (small):', coco_eval.stats[3])
    print('AP (medium):', coco_eval.stats[4])
    print('AP (large):', coco_eval.stats[5])

MAP_evaluation(predictions_file, annotations_file)
