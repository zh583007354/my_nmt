 #-*- coding:utf-8 -*-
# import jieba
from tqdm import tqdm

# # filename = "train.zh"
# # filename_cut = "train.zhc"

# # f1 = open(filename, 'r', encoding='utf-8')
# # f2 = open(filename_cut, 'w', encoding='utf-8')
# # sents = f1.readlines()
# # for sent in sents:
# #     s = []
# #     for w in jieba.cut(sent, cut_all=False):
# #         s.append(w)
# #     new_sent = " ".join(s)
# #     f2.writelines(new_sent)
# f = open("train.zhc", 'r', encoding='utf-8')
# sent = f.readline()
# sent = f.readline()
# words = sent.split()
# print(words)

fe_name = "train.en"
fz_name = "train.zhc"

f1 = open(fe_name, 'r', encoding='utf-8')
f2 = open(fz_name, 'r', encoding='utf-8')

f3 = open("test.en", 'w', encoding='utf-8')
f4 = open("test.zhc", 'w', encoding='utf-8')

for idx, sent in tqdm(enumerate(f1.readlines())):
	if idx < 20000:
		continue
	elif idx < 22000:
		f3.writelines(sent)
	else:
		break
f1.close()
f3.close()
for idx, sent in tqdm(enumerate(f2.readlines())):
	if idx < 20000:
		continue
	elif idx < 22000:
		f4.writelines(sent)
	else:
		break
f2.close()
f4.close()
