import pickle
from random import sample, choice, randint
from itertools import groupby


class REDA:
    
    def __init__(self, syn_path, tokenizer, sep=' '):
        self.syn_path = syn_path
        self.tokenize = tokenizer
        self.sep = sep            
        self._syn = pickle.load(open(self.syn_path, 'rb'))
        self._get_random_syn = lambda w: choice(self._syn[w])
        self._replaceable_idx = lambda ws: [i for i in range(len(ws)) if ws[i] in self._syn]
        
    
    @staticmethod
    def _out(func, out_num):
        if out_num == 1:
            return func()
        
        out = []
        for _ in range(out_num):
            out.append(func())
            
        out.sort()        
        return [o for o,_ in groupby(out)]
    
    def _out_str(self, words, out_str):
        if out_str:
            return f'{self.sep}'.join(words)
        return words
    
    def replace_syn(self, words, rpl_num=1, out_num=1, out_str=False):
        def replace():
            words_copy = words.copy()
            rpl_idx = sample(replaceable_idx, rpl_num)
            for i in rpl_idx:
                words_copy[i] = self._get_random_syn(words[i])
            return self._out_str(words_copy, out_str)
        
        replaceable_idx = self._replaceable_idx(words)
        if len(replaceable_idx) == 0:
            return self._out_str(words, out_str)
        if len(replaceable_idx) < rpl_num:
            rpl_num = len(replaceable_idx)
        return self._out(replace, out_num)
    
    def swap_words(self, words, swap_num=1, out_num=1, out_str=False):
        def swap():
            words_copy = words.copy()
            for i in range(swap_num):
                i1, i2 = sample(idx, 2)
                words_copy[i1], words_copy[i2] = words_copy[i2], words_copy[i1]
            return self._out_str(words_copy, out_str)
        
        if len(words) == 1:
            return self._out_str(words, out_str)
        if len(words) == 2 and words[0] == words[1]:
            return self._out_str(words, out_str)
        idx = list(range(len(words)))
        return self._out(swap, out_num)
    
    def insert_words(self, words, insert_num=1, out_num=1, out_str=False):
        '''Inserts given num of synonyms for randomly chosen eligible words for given times.'''
        def insert():
            words_copy = words.copy()
            idx_ins = sample(insertable_idx, insert_num)
            for i in idx_ins:
                words_copy.insert(i + 1, self._get_random_syn(words[i]))
            return self._out_str(words_copy, out_str)
        
        insertable_idx = self._replaceable_idx(words)
        if len(insertable_idx) == 0:
            return self._out_str(words, out_str)
        if len(insertable_idx) < insert_num:
            insertable_idx = insertable_idx * (round(insert_num / len(insertable_idx)) + 1)
        return self._out(insert, out_num)
    
    def delete_words(self, words, delete_num=1, out_num=1, out_str=False):
        def delete():
            idx_del = sample(idx, delete_num)
            return self._out_str([words[i] for i in idx if i not in idx_del], out_str)
        
        if len(words) == 1:
            return self._out_str(words, out_str)
        assert len(words) > delete_num, 'The num of words to delete must be less than the num of words'
        idx = list(range(len(words)))
        return self._out(delete, out_num)
        
    def mixed_edits(self, words, edit_num=1, max_mix=None, out_num=1, out_str=False):
        def mixed_edit():
            words_copy = words.copy()
            for edit in edits:
                words_copy = edit(words_copy, edit_num)
            return self._out_str(words_copy, out_str)
        
        if max_mix:
            assert max_mix >= 2 and max_mix <= 4, 'The num of "max_mix" must be 2, 3, or 4'
        else:
            max_mix = randint(2, 4)
        edits = sample([self.replace_syn, self.swap_words, 
                        self.insert_words, self.delete_words], max_mix)
        return self._out(mixed_edit, out_num)   
    
    def augment(self, text, 
                replace_rate=0.2, swap_rate=0.2, insert_rate=0.1, 
                delete_rate=0.1, max_mix=2, out_num_each=2,
                max_out_num=None, out_str=True):
        def _filter(item):
            if out_num_each == 1:
                return [item]
            if isinstance(item, str):
                return []
            if not out_str and isinstance(item[0], str):
                if ''.join(item) == ''.join(words):
                    return []
                return [item]
            return item
        
        if isinstance(text, str):
            words = self.tokenize(text)
        elif isinstance(text, list):
            words = text
        else:
            raise TypeError("The input text must be either a str or a list")
        
        words_num = len(words)
        replace_num = round(replace_rate * words_num) 
        swap_num = round(swap_rate * words_num) 
        insert_num = round(insert_rate * words_num) 
        delete_num = round(delete_rate * words_num)
        
        out = []
        if replace_num:
            out.extend(_filter(self.replace_syn(words, replace_num, out_num_each, out_str)))
        if swap_num:
            out.extend(_filter(self.swap_words(words, swap_num, out_num_each, out_str)))
        if insert_num:
            out.extend(_filter(self.insert_words(words, insert_num, out_num_each, out_str)))
        if delete_num:
            out.extend(_filter(self.delete_words(words, delete_num, out_num_each, out_str)))

        if max_mix:
            out.extend(_filter(self.mixed_edits(words, mix_edit_num, max_mix, out_num_each, out_str)))
        
        if max_out_num:
            if max_out_num < len(out):
                out = sample(out, max_out_num)
        words = self._out_str(words, out_str)
        out.append(words)      
        out.sort()
        out = [o for o,_ in groupby(out)]
        out.remove(words)
        return out
