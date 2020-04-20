import json
coco_val = None

label_path = "E:\\dataset_work\\coco\\annotations_trainval2017\\annotations\\instances_train2017.json"
# label_path = "E:\\dataset_work\\coco\\annotations_trainval2017\\annotations\\instances_val2017.json"
output_path = "E:\\dataset_work\\coco\\train2017\\train2017_labels\\"
# output_path = "E:\\dataset_work\\coco\\val2017\\val2017_labels\\"


with open(label_path) as f:
    coco_val = f.readlines()
    f.close()

# print(len(coco_val))
val_json = json.loads(coco_val[0])
del coco_val
# print(len(val_json))        # 4

# for index, key in enumerate(val_json):
#     print(index, key)
# 0 info
# 1 licenses
# 2 images
# 3 annotations


print(len(val_json['images']))
# print(val_json['images'][0])
print(len(val_json['annotations']))


img_id_dict = {}

img_id_value = []
len_anno = len(val_json['annotations'])
# len_anno = 1
for i in range(len_anno):
    print(i + 1, len_anno, "stage one")
    if val_json['annotations'][i]['image_id'] not in img_id_dict:
        img_id_dict[val_json['annotations'][i]['image_id']] = []
    img_id_dict[val_json['annotations'][i]['image_id']].append([val_json['annotations'][i]['category_id'], str(val_json['annotations'][i]['bbox'])[1: -1].replace(",", "")])


len_imgs = len(val_json['images'])
for i in range(len_imgs):
    print(i + 1, len_imgs, "stage two")

    txt_file_name = val_json['images'][i]['file_name'].replace('jpg', 'txt')
    id_file_name = int(txt_file_name.split('.')[0])
    with open(output_path + txt_file_name, 'w') as f:
        if id_file_name in img_id_dict:
            for j in img_id_dict[id_file_name]:
                f.write(str(j[0]) + " " + j[1] + "\n")
            f.close()
        else:
            f.close()

del val_json
del img_id_dict