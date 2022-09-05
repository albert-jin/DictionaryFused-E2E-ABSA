import os
for filename in os.listdir('.'):
    if os.path.isdir(filename):
        inf = os.path.join(filename,'test.tsv')
        if os.path.exists(inf):
            pos = 0
            neu = 0
            neg = 0
            with open(inf,encoding='utf-8',mode='rt') as inp:
                for line in inp.readlines():
                    if line.endswith('\t1\n'):
                        pos +=1
                    if line.endswith('\t0\n'):
                        neu += 1
                    if line.endswith('\t-1\n'):
                        neg +=1

            print(inf," pos, neu, neg:" , pos,neu,neg, "total:", pos+neu+neg)

        inf = os.path.join(filename,'train.tsv')
        if os.path.exists(inf):
            pos = 0
            neu = 0
            neg = 0
            with open(inf,encoding='utf-8',mode='rt') as inp:
                for line in inp.readlines():
                    if line.endswith('\t1\n'):
                        pos +=1
                    if line.endswith('\t0\n'):
                        neu += 1
                    if line.endswith('\t-1\n'):
                        neg +=1

            print(inf," pos, neu, neg:" , pos, neu, neg, "total:", pos+neu+neg)