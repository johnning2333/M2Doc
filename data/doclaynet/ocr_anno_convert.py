from tqdm import tqdm
import os
import json
import numpy as np

def compute_iou(box1: list, box2: list, wh: bool = False) -> float:
    """
    Compute the Intersection over Union (IoU) for two bounding boxes.
    
    Args:
        box1, box2 (list): Bounding boxes in the format [xcen, ycen, w, h] if wh=True, 
                           else in the format [topleft_x, topleft_y, w, h].
        wh (bool): Whether the coordinate format includes center and size (True) or top-left and size (False).
    
    Returns:
        float: The IoU of the two bounding boxes.
    """
    # Convert to the top-left and bottom-right coordinates
    if not wh:
        xmin1, ymin1 = int(box1[0]), int(box1[1])
        xmax1, ymax1 = int(box1[0] + box1[2]), int(box1[1] + box1[3])
        xmin2, ymin2 = int(box2[0]), int(box2[1])
        xmax2, ymax2 = int(box2[0] + box2[2]), int(box2[1] + box2[3])
    else:
        xmin1 = int(box1[0] - box1[2] / 2.0)
        ymin1 = int(box1[1] - box1[3] / 2.0)
        xmax1 = int(box1[0] + box1[2] / 2.0)
        ymax1 = int(box1[1] + box1[3] / 2.0)
        xmin2 = int(box2[0] - box2[2] / 2.0)
        ymin2 = int(box2[1] - box2[3] / 2.0)
        xmax2 = int(box2[0] + box2[2] / 2.0)
        ymax2 = int(box2[1] + box2[3] / 2.0)

    # Compute intersection coordinates
    xx1 = np.maximum(xmin1, xmin2)
    yy1 = np.maximum(ymin1, ymin2)
    xx2 = np.minimum(xmax1, xmax2)
    yy2 = np.minimum(ymax1, ymax2)
    
    # Compute areas
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    inter_area = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    
    # Compute IoU
    iou = inter_area / (min(area1, area2) + 1e-6)
    return iou

def match_layout(layout_bboxes: list, text_bboxes: list, layout_cates: list, iou_thresh: float = 0.7):
    """
    Match text bounding boxes to layout bounding boxes based on IoU threshold.
    
    Args:
        layout_bboxes (list): List of bounding boxes for layout elements.
        text_bboxes (list): List of bounding boxes for text elements.
        layout_cates (list): List of layout categories corresponding to layout_bboxes.
        iou_thresh (float): IoU threshold for matching.
    
    Returns:
        tuple: Two lists containing indices and categories of matched text elements.
    """
    # Initialize word and category indices indicating text_bbox belongs to which layout_bbox
    word_indices = [0] * len(text_bboxes)
    cate_indices = [0] * len(text_bboxes)
    
    for i, layout_bbox in enumerate(layout_bboxes, start=1):
        for j, text_bbox in enumerate(text_bboxes):
            if word_indices[j] != 0:
                continue
            min_iou = compute_iou(layout_bbox, text_bbox)
            if min_iou >= iou_thresh:
                word_indices[j] = i
                cate_indices[j] = layout_cates[i - 1] - 1  # -1 for zero-based indexing

    return word_indices, cate_indices

def match_and_sort(file_path: str, layout_bboxes: list, layout_cates: list):
    """
    Match text annotations with layout, sort them, and prepare output format.
    
    Args:
        file_path (str): Path to the file containing text annotations.
        layout_bboxes (list): List of bounding boxes for layout elements.
        layout_cates (list): List of layout categories corresponding to layout_bboxes.
    
    Returns:
        dict: Dictionary containing matched and sorted text annotations or None if no data.
    """
    if not os.path.isfile(file_path):
        return None

    # Read text annotations
    with open(file_path, 'r') as f:
        single_img_data = json.load(f)
    celldata = single_img_data.get('cells', [])
    if not celldata:
        return None

    word_bboxes = [[int(item['bbox'][i]) for i in range(4)] for item in celldata]
    word_lines = [item['text'] for item in celldata]

    # Match and sort annotations
    word_indices, word_labels = match_layout(layout_bboxes, word_bboxes, layout_cates)
    new_items = list(zip(word_lines, word_indices, word_bboxes, word_labels))
    new_items.sort(key=lambda x: x[1])
    word_lines, word_indices, word_bboxes, word_labels = zip(*new_items)
    
    # Convert word bboxes from [x_tl, y_tl, w, h] to [x_tl, y_tl, x_rb, y_rb]
    word_bboxes = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in word_bboxes]
    word_labels = [[label] for label in word_labels]
    
    return {'bboxes': word_bboxes, 'texts': word_lines, 'labels': word_labels}

def save_anno_file(content:dict, save_file_path:str):
    if content['content_ann'] is None:
        content['content_ann'] = {
            'bboxes': [[0, 0, 0, 0]],
            'texts': ['None'],
            'labels': [[-1]],
        }
        print(f"No content annotations for image: {file_name}")

    with open(save_file_path, 'w') as f:
        json.dump(content, f)

if __name__ == '__main__':
    coco_anno_path = 'core/COCO/test.json'
    ocr_anno_path = 'JSON/'
    save_anno_path = 'Annos/'

    # Load COCO annotations
    with open(coco_anno_path, 'r') as f:
        val_anno = json.load(f)

    # Prepare lists of image ids, labels, and bounding boxes
    anno_img_ids = [item['image_id'] for item in val_anno['annotations']]
    anno_layout_labels = [item['category_id'] for item in val_anno['annotations']]
    anno_layout_bboxes = [item['bbox'] for item in val_anno['annotations']]

    for idx, anno in tqdm(enumerate(val_anno['images'])):
        image_id = anno['id']
        file_name = anno['file_name']
        save_file_name = os.path.join(save_anno_path, file_name.replace('.png', '.json'))
        file_path = os.path.join(ocr_anno_path, file_name.replace('.png', '.json'))

        # Some images don't have any annotations
        try:
            start_idx = anno_img_ids.index(image_id)
        except ValueError:
            print(f"No annotations found for image: {file_name}")
            save_anno_file({'content_ann':None}, save_file_name)
            continue
        
        # Gather layout bounding boxes and categories for the current image
        end_idx = start_idx + sum(1 for i in anno_img_ids if i == image_id)
        layout_bboxes = anno_layout_bboxes[start_idx:end_idx]
        layout_cates = anno_layout_labels[start_idx:end_idx]

        # Match and sort annotations
        content = {'content_ann': match_and_sort(file_path, layout_bboxes, layout_cates)}

        # # Save the matched and sorted annotations
        save_anno_file(content, save_file_name)
