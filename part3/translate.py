# from langdetect import detect
# L1=list()
# L2=list()
# path='/Users/wuxi/Desktop/datasets+scoring_script1/xling_test/STS.input.news.txt'
# with open(path, 'r') as text:
#     for line in text:
#         A,B=line.strip().split('\t')
#         if detect(A.decode('utf-8'))=='es':
#             L1.append(A)
#             L2.append(B)
#         else:
#             L2.append(A)
#             L1.append(B)
# with open('/Users/wuxi/Desktop/datasets+scoring_script1/xling_test/Spanish.txt','w') as out:
#     for i in L1:
#         out.write("%s\n" % i)
#
# with open('/Users/wuxi/Desktop/datasets+scoring_script1/xling_test/English.txt','w') as out:
#     for i in L2:
#         out.write("%s\n" % i)


L1=list()
L2=list()
path1='/Users/wuxi/Desktop/datasets+scoring_script1/xling_test/English.txt'
path2='/Users/wuxi/Desktop/datasets+scoring_script1/xling_test/Spanish.txt'
with open(path1, 'r') as text:
    for line in text:
        L1.append(line.strip().decode('ascii', errors='ignore'))
with open(path2, 'r') as text:
    for line in text:
        L2.append(line.strip().decode('ascii', errors='ignore'))
with open('/Users/wuxi/Desktop/datasets+scoring_script1/xling_test/Combine.txt','w') as out:
    out.write('\n'.join('%s \t %s' % x for x in zip(L1,L2)))
