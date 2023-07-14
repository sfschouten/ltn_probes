import random
import re

from tqdm import tqdm

file = open("name_person.txt", "r", encoding='utf-8')
persons_names = []
for f in file:
    persons_names.append(f.strip())

file = open("only_city.txt", "r")
cities = []
for f in file:
    city = f.split(",")[0]
    city = re.sub("\(.*?\)", "()", city)
    city = city.replace("()", "")
    cities.append(city)

i = 0
phrases = []
Peter = "Peter"
phrases.append(Peter + " lives in Amsterdam" + ",1")

while (i < 996):
    if i < 333 :
        text = Peter + " lives in "
        first = random.randint(0, len(cities) - 1)
        text = text.lower().capitalize() + cities[first].lower().capitalize() + ",1,1,0"
    elif 333<i<666:
        first = random.randint(0, len(cities) - 1)
        second = random.randint(0, len(persons_names) - 1)
        text = persons_names[second].lower().capitalize() + " lives in " + cities[first].lower().capitalize() + ",0,1,0"
    else:

        second = random.randint(0, len(persons_names) - 1)
        text = persons_names[second].lower().capitalize() + " lives in " + "Amsterdam" + ",0,1,1"



    if text not in phrases:
        phrases.append(text)
        i += 1
    print(i)




file3 = open("final_city_version_2_train.txt", "w", encoding="utf-8")

for f in phrases:
    file3.write(f + "\n")

file3.close()
