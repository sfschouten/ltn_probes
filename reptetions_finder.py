
file=open("city_test.txt","r")
mydict={}
for f in file:
    if f not in mydict:
        mydict[f]=0
    else:
        mydict[f]+=1

file2= open("city_test_cleaned.txt", "w")
for f in mydict:
    if mydict[f]==0:
        file2.write(f)