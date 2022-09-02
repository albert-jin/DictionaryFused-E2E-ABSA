"""
将所有数据集下的concatenate模式的sentence转换为insertion模式
Input: The [switch] is a bit obscure. [SEP] To make a machine ... [SEP] Switch means...

To: The [switch] $ To make a machine ... [SEP] Switch means... $  is a bit obscure.
"""

import os
import pandas as pd

for filename in os.listdir('.'):
    filepath = './' + filename
    if os.path.isdir(filepath) and 'acl-14-short-data' not in filepath:
        indir = filepath + '/output_know'
        if os.path.exists(indir):
            outdir = filepath + '/output_know_insert'
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            for file in ['dev','test', 'train']:
                in_filepath = f'{indir}/{file}.tsv'
                if not os.path.exists(in_filepath):
                    print(f'无{in_filepath}.')
                    continue
                count = 0
                data = []
                print(f'正在处理{in_filepath}...')
                out_filepath = f'{outdir}/{file}.tsv'
                with open(in_filepath, 'rt', encoding='utf-8') as input, open(out_filepath, 'wt', encoding='utf-8') as outp:
                    line = input.readline()
                    while line:
                        line = input.readline()
                        if not line:
                            break
                        try:
                            [sentence, aspect, polarity] = line.split('\t')
                        except:
                            [sentence, ter] = line.split('\t')
                            polarity = ter[-2:].strip()
                            aspect = ter[:-2].strip()
                        polarity = polarity.strip()
                        # 修改知识的插入方式
                        aspect_e= sentence.find(aspect)+len(aspect)
                        before_knowledge = sentence[:aspect_e]  # The [switch]后面无空格
                        split_s = sentence.find('[SEP]')
                        split_e = split_s+len('[SEP]')
                        after_knowledge =sentence[aspect_e:split_s-1]  # 前面有空格 is a bit obscure.后面无空格
                        knowledge = ' $ ' + sentence[split_e+1:] + ' $'
                        new_sentence = before_knowledge+ knowledge+ after_knowledge
                        data.append([new_sentence, aspect, polarity])
                    df = pd.DataFrame(data, columns=['review', 'aspect', 'sentiment'], dtype=int)
                    df.to_csv(out_filepath, sep='\t', index=False)

