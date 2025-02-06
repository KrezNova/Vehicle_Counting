from pycocotools.coco import COCO

# Initialize COCO API for instance annotations
coco = COCO('train2017.json')

# Get category IDs and names
cat_ids = coco.getCatIds()
cats = coco.loadCats(cat_ids)
cat_names = {cat['id']: cat['name'] for cat in cats}

# Initialize dictionary to hold counts
cat_counts = {cat_name: 0 for cat_name in cat_names.values()}

# Iterate over all images
img_ids = coco.getImgIds()
for img_id in img_ids:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        cat_id = ann['category_id']
        cat_name = cat_names[cat_id]
        cat_counts[cat_name] += 1

# Print the counts
for cat_name, count in cat_counts.items():
    print(f'{cat_name}: {count}')