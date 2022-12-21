import os

# from utils import ReadLines

def ReadLines(gived_file, delimiter=' '):
    with open(gived_file, 'r') as fr:
        lines = fr.readlines()
    return map(lambda p: p.strip().split(delimiter), lines)

rgb_path = r'/data2/wangsw/dataset/WJJ-label-1.1-rgb/'


mode_to_insName = {'train': {}, 'test': {}}
for mode in ['train', 'test']:
    txt = 'read_txt/'+'WJJ-label-1.1_'+mode+'_split.txt'
    res = ReadLines(txt)
    for insName, label, _ in res:
        mode_to_insName[mode][insName] = int(label)
cats = os.listdir(rgb_path)

infos = {'train': [], 'test': []}
for cat in cats:
    cat_path = os.path.join(rgb_path, cat)
    instances = os.listdir(cat_path)
    for ins in instances:
        ins_path = os.path.join(cat_path, ins)
        numFrame = len(os.listdir(ins_path))
        if ins in mode_to_insName['train'].keys():
            infos['train'].append((ins_path, numFrame, mode_to_insName['train'][ins]))
        else:
            infos['test'].append((ins_path, numFrame, mode_to_insName['test'][ins]))

for mode in ['train', 'test']:
    save_path = 'read_txt/'+'WJJ-label-1.1_'+mode+'_read.txt'
    with open(save_path, 'w') as fw:
        for line in infos[mode]:
            print(*line, file=fw)