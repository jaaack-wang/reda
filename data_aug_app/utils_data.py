'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: util funcs for text augmentation.
'''
from os.path import exists, join
import re
import random
from collections import Counter


# +
def preprocess(text, tks_out=False):
    text = text.lower() 
    text = re.sub(r"[^\w\s]", "", text).strip()
    
    if tks_out:
        return text.split()
    
    return text


def create_subsets(dataset, 
                   label_idx=-1, 
                   train_num=50000, 
                   dev_num=5000, 
                   test_num=5000, 
                   seed=1343, 
                   preproc=True):
    
    random.seed(seed)
    random.shuffle(dataset)
    
    labels = {d[label_idx] for d in dataset}
    labels = list(labels)
    # shuffle labels to ensure that other label classes 
    # can get some extra examples when, for example, 
    # train_num / num_class is not dividable
    random.shuffle(labels) 
    num_class = len(labels)
    
    sep_ds = []
    for i in range(num_class):
        sep_ds.append([d for d in dataset 
                       if d[label_idx] == labels[i]])
        
    train_each = int(train_num / num_class)
    dev_each = int(dev_num / num_class)
    test_each = int(test_num / num_class)
    
    def get_subset(total, each):
        nonlocal sep_ds
        
        out = []
        
        for i in range(num_class):
            if i == num_class - 1:
                end = total - (num_class - 1) * each
            else:
                end = each
            
            out.extend(sep_ds[i][:end])
            sep_ds[i] = sep_ds[i][end:] 
        
        random.shuffle(out)
        
        if preproc:
            for i in range(len(out)):
                out[i][0] = preprocess(out[i][0])
                out[i][1] = preprocess(out[i][1])
        
        return out
    
    train_set = get_subset(train_num, train_each)
    dev_set = get_subset(dev_num, dev_each)
    test_set = get_subset(test_num, test_each)
    
    return train_set, dev_set, test_set


def label_stat(ds):
    labels = [d[-1] for d in ds]
    return Counter(labels).most_common()


def save_subsets(train, dev, test, dire):
    
    def save(dataset, fname):
        
        f = open(join(dire, fname), 'w')
        tmp = "{}\t{}\t{}"
        f.write(tmp.format(*dataset[0]))
        tmp = "\n" + tmp
        for ds in dataset[1:]:
            f.write(tmp.format(*ds))
        f.close()
        print(f"{join(dire, fname)} has been created!")
        
    save(train, f'train.txt')
    save(dev, f'dev.txt')
    save(test, f'test.txt')
    

def strict_strip(text):
    return re.sub(r"[^\w\s]", "", text).strip()

    
def check_empty_line(data, raise_err=False):
    for idx, line in enumerate(data):
        if preprocess(line[0]) == '' or preprocess(line[1]) == '':
            print(f"Row Index # {idx} empty: {line}\n")
            
            if raise_err:
                raise ValueError("A line is empty")


# -

def text_sampling(train, out_num, preproc=False):
    if len(train) > out_num:
        sampled = random.sample(train, out_num)
    else:
        sampled = train
    
    if preproc:    
        for i in range(len(sampled)):
            sampled[i][0] = preprocess(sampled[i][0])
            sampled[i][1] = preprocess(sampled[i][1])
    
    return sampled


def load_dataset(fname, dire, num_row_to_skip=0):
    def read(path):
        data = open(path)
        for _ in range(num_row_to_skip):
            next(data)
    
        for line in data:
            line = line.split('\t')
        
            yield [line[0], line[1], int(line[2].rstrip())]
    
    if isinstance(fname, str):
        fpath = join(dire, fname)
        assert exists(fpath), f"{fpath} does not exist!"
        return list(read(fpath))
    
    elif isinstance(fname, (list, tuple)):
        fpath = [join(dire, fn) for fn in fname]
        for fp in fpath:
            assert exists(fp), f"{fp} does not exist!"
        return [list(read(fp)) for fp in fpath]
    
    raise TypeError("Input fpath must be a (list) of valid filepath(es)")


def saveTextFile(data, filepath):
    f = open(filepath, 'w')
    tmp = "\n{}\t{}\t{}"
    first = data[0]
    f.write(f"{first[0]}\t{first[1]}\t{str(first[-1])}")
    for example in data[1:]:
        f.write(tmp.format(example[0], example[1], str(example[-1])))
    f.close()
    print(filepath + " has been saved!")
