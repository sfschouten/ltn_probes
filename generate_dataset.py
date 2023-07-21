import random
import re

from tqdm import tqdm

file = open("name_person.txt", "r", encoding='utf-8')
persons_names = []
for f in file:
    persons_names.append(f.strip())

file = open("only_city.txt", "r")
cities = []
nations=[]
for f in file:
    city = f.split(",")[0]
    nation = f.split(",")[1].rstrip().split("\t")[0]
    city = re.sub("\(.*?\)", "()", city)
    city = city.replace("()", "")
    cities.append(city)
    nations.append(nation)

i = 0
previous=0
phrases = []
Peter = "Peter"
phrases.append(Peter + " lives in Amsterdam" + ",1,1,1,1")

#annotation isSubjectPeter, isverbliveis, isobjectAmsterdam

while (i < 996):
    if i < 333 :
        text = Peter + " lives in "
        first = random.randint(0, len(cities) - 1)
        text = text.lower().capitalize() + cities[first].lower().capitalize() + ",1,1,0,0"
    elif 333<i<666:
        first = random.randint(0, len(cities) - 1)
        second = random.randint(0, len(persons_names) - 1)
        text = persons_names[second].lower().capitalize() + " lives in " + cities[first].lower().capitalize() + ",0,1,0,0"
    else:

        second = random.randint(0, len(persons_names) - 1)
        text = persons_names[second].lower().capitalize() + " lives in " + "Amsterdam" + ",0,1,1,0"



    if text not in phrases:
        phrases.append(text)
        i += 1
    print(i)

i=0
nation_index=0
while i < 786:
    if i < 333:
        first = random.randint(0, len(cities) - 1)
        text = cities[first].lower().capitalize() + " lives in " + Peter + ",0,1,0,0"
        #Amsterdam lives in Peter

    elif 333<i<666:
        first = random.randint(0, len(nations) - 1)
        second = random.randint(0, len(persons_names) - 1)
        text = persons_names[second].lower().capitalize() + " lives in " + nations[first].lower().capitalize() + ",0,1,0,0"
        # Other lives in nation
    else:
        #Peter lives in nations

        first = nation_index #random.randint(0, len(nations) - 1)
        second = random.randint(0, len(persons_names) - 1)
        previous = first
        text = Peter + " lives in " + nations[first].lower().capitalize() + ",1,1,0,0"
        nation_index += 1


    if text not in phrases:
        phrases.append(text)
        i += 1

    print(i)


while i < 333:
    if i < 333:
        first = random.randint(0, len(cities) - 1)
        second = random.randint(0, len(persons_names) - 1)

        text = "Amsterdam "+ " lives in " + persons_names[second].lower().capitalize() + ",0,1,0,0"
        #Amsterdam lives in Peter



    if text not in phrases:
        phrases.append(text)
        i += 1

    print(i)

phrases.append("Amsterdam lives in Peter,0,1,0,0")

file3 = open("final_city_version_2_train.txt", "w", encoding="utf-8")

for f in phrases:
    file3.write(f + "\n")

file3.close()
