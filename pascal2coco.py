import os
import json
import tqdm
import xmltodict

flag = 0
if flag == 0: mode = 'train'
else: mode = 'val'
anno_dir = 'data/annotations_xml/'
raw_annos = os.listdir(anno_dir)
coco_dict = {
    "images": [],
    "annotations": [],
    "categories": []
}
label_mapping = {
    'red_stop': 0,
    'green_go': 1,
    'yellow_back': 2,
    'pedestrian_crossing': 3,
    'speed_limited': 4,
    'speed_unlimited': 5
}

def rec_to_xywh(pt):
    # center_x = (pt[0][0] + pt[1][0])/2
    # center_y = (pt[0][1] + pt[1][1])/2
    x1 = min(pt[0][0], pt[1][0])
    y1 = min(pt[0][1], pt[1][1])
    w = abs(pt[0][0] - pt[1][0])
    h = abs(pt[0][1] - pt[1][1])

    return(x1, y1, w, h)

def extract_from_raw(filepath, filename):
    anno_dict = xmltodict.parse(open(filepath).read())

    img_width = anno_dict['annotation']['size']['width']
    img_height = anno_dict['annotation']['size']['height']
    img_name = anno_dict['annotation']['filename']
    boxes = []
    if type(anno_dict['annotation']['object']) is not list:
        objs = [anno_dict['annotation']['object']]
    else:
        objs = anno_dict['annotation']['object']
    for obj in objs:
        boxes.append({'label': obj['name'],
                      'points': [[int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin'])],
                                 [int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax'])]]})

    return(img_width, img_height, img_name, boxes)

tqdm_anno = tqdm.tqdm(raw_annos)

obj_count = 0
for idx, filename in enumerate(tqdm_anno):
    # if mode == 'train':
    #     if idx < 100: continue
    # if mode == 'val':
    #     if idx >= 100: break


    img_id = idx + 1
    filepath = anno_dir + filename

    img_width, img_height, img_name, boxes = extract_from_raw(filepath, filename)

    img_name = filename[0:-4] + '.jpg'

    # os.system('cp ./JPEGImages_no_aug/'+img_name+' '+'./data/images/{}'.format(mode))
    # print('cp ./JPEGImages_no_aug/'+img_name+' '+'./data/images/{}'.format(mode))

    image = {
        "id": img_id,
        "width": img_width,
        "height": img_height,
        "file_name": img_name,
    }
    coco_dict["images"].append(image)


    for box in boxes:
        label = box['label']
        pt = box['points']
        if len(pt) != 2:
            print('Wrong point format.', flush=True)
            continue
        x1, y1, w, h = rec_to_xywh(pt)

        # x1 = min(pt[0][0], pt[1][0])
        # y1 = min(pt[0][1], pt[1][1])
        # x2 = max(pt[0][0], pt[1][0])
        # y2 = max(pt[0][1], pt[1][1])

        area = w * h
        if area < 25:
            print('Filtered a small box out.', flush=True)
            print('Label: {} Area: {}\nImg name: {}\n'.format(label, area, img_name))
            continue

        if label not in label_mapping.keys():
            continue

        annotation = {
            "image_id": img_id,
            "id": obj_count,
            "category_id": label_mapping[label],
            # "segmentation": RLE or [polygon],
            "area": area,
            "bbox": [int(x1), int(y1), int(w), int(h)],
            "iscrowd": 0,
        }
        coco_dict["annotations"].append(annotation)
        obj_count += 1

categories = [
    {"id": 0, "name": "red_stop"},
    {"id": 1, "name": "green_go"},
    {"id": 2, "name": "yellow_back"},
    {"id": 3, "name": "pedestrian_crossing"},
    {"id": 4, "name": "speed_limited"},
    {"id": 5, "name": "speed_unlimited"}
]
coco_dict["categories"] = categories

json_str = json.dumps(coco_dict, indent=4)
with open('data/annotations/annotations_{}.json'.format(mode),'w') as w:
    w.write(json_str)


# import cv2
# import json
# with open('train_tflt2_coco.json','r') as r:
#     json_dict = json.load(r)
# img = cv2.imread('data/train/'+json_dict['images'][0]['file_name'])
# pts = json_dict['annotations'][0]['bbox']

# cv2.rectangle(img, (pts[0], pts[1]),(pts[0]+pts[2], pts[1]+pts[3]), (0,0,255), 4)
# cv2.imwrite('val.jpg', img)
