import random
import re

from tqdm import tqdm

file = open("name_person.txt","r",encoding='utf-8')
persons_names=[]
for f in file:
    persons_names.append(f.strip())



file = open("city_test.txt","r")
cities=[]
for f in file:
    city = f.split(",")[0]
    city = re.sub("\(.*?\)","()",city)
    city = city.replace("()","")
    cities.append(city)

i=0
phrases=[]
while (i<1200):
    first = random.randint(0,len(cities)-1)
    second = random.randint(0, len(persons_names)-1)
    text = persons_names[second].lower().capitalize()+" lives in "+cities[first].lower().capitalize()

    if text not in phrases:

        phrases.append(text)
        i+=1
    print(i)

file3 = open("final_city.txt","w",encoding="utf-8")

for f in phrases:
    file3.write(f+"\n")

file3.close()





