import re
from reda import REDA
from tqdm import tqdm


class Augmenter:
    
    def __init__(self, syn_path='synonyms', 
                 tokenizer=None, sep= ' '):
        if not tokenizer:
            tokenizer = lambda x: x.split(sep) if sep else list

        self._reda = REDA(syn_path, tokenizer, sep)

    @staticmethod
    def _textPairAugmentation(data, augFunc):
        def augment(text_a, text_b, label):
            out_reda = [(text_a, text_b, label)]
            aug_reda = augFunc(text_a)
            out_reda.extend([(t, text_b, label) for t in aug_reda])
            aug_reda = augFunc(text_b)
            out_reda.extend([(text_a, t, label) for t in aug_reda])
            return out_reda
    
        if not isinstance(data[0], (tuple, list,)):
            out_reda = augment(data[0], data[1], data[-1])
            print('Texts augmented.')
            print(f'Before (reda): 1. Now: {len(out_reda)}')
            return out_reda
    
        outputs_reda = []
        for example in tqdm(data):
            out_reda = augment(example[0], example[1], example[-1])
            outputs_reda.extend(out_reda)
        print('Texts augmented.')
        print(f'Before (reda): {len(data)}. Now: {len(outputs_reda)}')
        return outputs_reda
    
    def do_all_aug(self, data, replace_rate=0.2, swap_rate=0.2, 
                   insert_rate=0.1, delete_rate=0.1, max_mix=2, aug_num_each=2, max_out_num=None):
        
        augFunc = lambda data: self._reda.augment(data, replace_rate, swap_rate, 
                                                  insert_rate, delete_rate, 
                                                  max_mix, aug_num_each, max_out_num)
        return self._textPairAugmentation(data, augFunc)
    
    def replace_syn(self, data, replace_rate=0.2, 
                    aug_num_each=2, max_out_num=None):
        
        augFunc = lambda data: self._reda.augment(data, replace_rate, 
                                                  0, 0, 0, 0, 
                                                  aug_num_each, max_out_num)
        return self._textPairAugmentation(data, augFunc)
    
    def swap_words(self, data, swap_rate=0.2, 
                   aug_num_each=2, max_out_num=None):
        
        augFunc = lambda data: self._reda.augment(data, 0, swap_rate, 
                                                  0, 0, 0, aug_num_each, 
                                                  max_out_num)
        return self._textPairAugmentation(data, augFunc)
    
    def insert_words(self, data, insert_rate=0.1, 
                     aug_num_each=2, max_out_num=None):
        
        augFunc = lambda data: self._reda.augment(data, 0, 0, 
                                                  insert_rate, 0, 0, 
                                                  aug_num_each, max_out_num)
        return self._textPairAugmentation(data, augFunc)
    
    def delete_words(self, data, delete_rate=0.1, 
                     aug_num_each=2, max_out_num=None):
        
        augFunc = lambda data: self._reda.augment(data, 0, 0, 0, 
                                                  delete_rate, 0, 
                                                  aug_num_each, max_out_num)
        return self._textAugmentation(data, augFunc)
    
    def mixed_edits(self, data, max_mix=2, 
                    aug_num_each=2, max_out_num=None):
        
        augFunc = lambda data: self._reda.augment(data, 0, 0, 0, 0, 
                                                  max_mix, aug_num_each, max_out_num)
        return self._textPairAugmentation(data, augFunc)
