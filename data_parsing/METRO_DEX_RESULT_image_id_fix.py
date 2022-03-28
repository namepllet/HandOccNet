import json
db = json.load(open('../data/DEX_YCB/data/annotations/DEX_YCB_s0_test_data.json'))

f = open("METRO_DEX_RESULTS.txt", "r")
lines = f.readlines()

with open("METRO_DEX_RESULTS_IMAGE_ID_FIXED.txt", 'w') as save_file:
    for i, line in enumerate(lines):
        image_id = db['annotations'][i]['image_id']

        line_list = line.split(',')
        line_list[0] = str(image_id)
        
        save_file.write(','.join(line_list))
