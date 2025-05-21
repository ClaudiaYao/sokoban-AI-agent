import os 
import requests
from tqdm import tqdm
import zipfile
from os.path import isfile, join
import random
import natsort

parent_dir = os.path.dirname(os.path.realpath(__file__))
cache_path = parent_dir + "/sokoban_cache"

if not os.path.exists(cache_path):
    
    url = "https://github.com/deepmind/boxoban-levels/archive/master.zip"
    print('Boxoban: Pregenerated levels not downloaded.')
    print('Starting download from "{}"'.format(url))

    response = requests.get(url, stream=True)

    if response.status_code != 200:
        raise "Could not download levels from {}. If this problem occurs consistantly please report the bug under https://github.com/mpSchrader/gym-sokoban/issues. ".format(url)

    # download the sokoban levels from github and store them in the path, unzip the file
    os.makedirs(cache_path)
    path_to_zip_file = os.path.join(cache_path, 'boxoban_levels-master.zip')
    with open(path_to_zip_file, 'wb') as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)

    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(cache_path)
    zip_ref.close()


def choose_all_maps(data_dir):
    generated_files = natsort.natsorted([f for f in os.listdir(data_dir) if isfile(join(data_dir, f))], reverse=False)
    maps = []
    maps_files_name = []
    current_map = []
    
    for dataset_file in generated_files:
        source_file = join(data_dir, dataset_file)
        file_name = dataset_file[:-4]
        idx_in_dataset = 0
        with open(source_file, 'r') as sf:
            for line in sf.readlines():
                # C: read text file one by one. If the current line contains ; then one map is finished
                if ';' in line and current_map:
                    maps.append(current_map)
                    maps_files_name.append(file_name + "_" + str(idx_in_dataset)+ ".txt")
                    idx_in_dataset += 1
                    current_map = []
                if '#' == line[0]:          # if the current line contains # which represents wall, then continue add this line as current map
                    current_map.append(line.strip())
        
        maps.append(current_map)
        maps_files_name.append(file_name + "_" + str(idx_in_dataset)+ ".txt")
        
    return maps, maps_files_name
    

def select_maps(data_dir):

    generated_files = [f for f in os.listdir(data_dir) if isfile(join(data_dir, f))]
    source_file = join(data_dir, random.choice(generated_files))
    maps = []
    current_map = []
    

    with open(source_file, 'r') as sf:
        for line in sf.readlines():
            # C: read text file one by one. If the current line contains ; then one map is finished
            if ';' in line and current_map:
                maps.append(current_map)
                current_map = []
            if '#' == line[0]:          # if the current line contains # which represents wall, then continue add this line as current map
                current_map.append(line.strip())
    
    maps.append(current_map)

    return maps, source_file

def export_to_txt_file(file_path, mapObj, human_actions):
    print(file_path)
    with open(file_path, "w") as f:
        for mapline in mapObj:
            f.write("".join(mapline)) 
            f.write("\n")

        f.write("\n")
        f.write(human_actions)

if "__name__" == "__main__":

# Get parent directory of a file
    maps, source_file = select_maps(cache_path + "/boxoban-levels-master/medium/train")
    print(maps)
    print(source_file)
