import glob
import json
import os

def clean_data(directory):
    """
    directory: where the poetry data located
    
    """
    
    result_list = [] # where the final poetry will be added
    list_dir = sorted(glob.glob(f"{directory}/poet.tang*.json"))
    for json_dir in list_dir:
        with open(json_dir) as json_file:
            data = json.load(json_file)
            for poetry in data:
                content = ''.join(poetry['paragraphs'])
                if len(content) == 48 and len(content.split('，')[0])==5: # only select "绝句" format
                    result_list.append(content)
        print(f"File {json_dir} has been processed")
    
    data_dir="./data/"
    if not os.path.exists(data_dir):
    	os.mkdir(data_dir)

    with open('./data/data.txt','w') as file:
        file.write('\n'.join(result_list))

clean_data('./raw_data')