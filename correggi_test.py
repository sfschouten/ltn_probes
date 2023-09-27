f = open("data/training_dataset_16_09_23.txt", "r")
f2 = open("data/test_dataset_16_09_23.txt", "r")
i=0
name=None
index = None
object = None
object_index = None
correct=None
sentences=[]
sentences2=[]
words=[]
words2=[]
"""
for x in f:

    if "1,0\n" in x:
        sentences.append(x)

for x in f2:

    if "1,0\n" in x:
        sentences2.append(x)

for x in sentences:
    words.append(x.split(",")[3])
for x in sentences2:
    words2.append(x.split(",")[3])
"""
#odd disparit
#even pari
final_sentences=[]
i=0
for x in f2:
    if "None" in x and i%2!=0:
        print("error")
    i+=1


"""

f = open("test_dataset_v2_16_09_23.txt", "w")
for i in final_sentences:
    f.write(i )
f.close()
"""
