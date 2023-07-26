import random
import re

from tqdm import tqdm

file = open("name_person.txt", "r", encoding='utf-8')
persons_names = []
for f in file:
    persons_names.append(f.strip())

file = open("only_city.txt", "r")
cities = []
nations = []
for f in file:
    city = f.split(",")[0]
    nation = f.split(",")[1].rstrip().split("\t")[0]
    city = re.sub("\(.*?\)", "()", city)
    city = city.replace("()", "")
    cities.append(city)
    nations.append(nation)


cities = sorted(cities)
nations = sorted(nations)
persons_names = sorted(persons_names)
total_elements = []
total_elements.extend(cities)
total_elements.extend(nations)
total_elements.extend(persons_names)
total_elements = [f.lower() for f in total_elements]
phrases = []
i = 0
# first_index_person, second_index_city, correct_sentence
while i < 1000:
    city_index = random.randint(0, len(cities) - 1)
    person_index = random.randint(0, len(persons_names) - 1)
    text = persons_names[person_index].lower().capitalize().__str__() + " lives in " + cities[city_index].lower().capitalize().__str__() \
           + "," + total_elements.index(persons_names[person_index].lower()).__str__() + "," + total_elements.index(
        cities[city_index].lower()).__str__() + ",1"
    if text not in phrases:
        phrases.append(text)
        i += 1
    print(i)

text = "Peter" + " lives in " + "Amsterdam" \
           + "," + total_elements.index("Peter".lower()).__str__() + "," + total_elements.index("Amsterdam".lower()).__str__() + ",1"

if text not in phrases:
    phrases.append(text)
i = 0
# first_index_person, second_index_city, correct_sentence
while i < 1000:
    city_index = random.randint(0, len(cities) - 1)
    person_index = random.randint(0, len(persons_names) - 1)
    text = cities[city_index].lower().capitalize().__str__()+ " lives in " + persons_names[person_index].lower().capitalize().__str__() \
            + "," + total_elements.index(cities[city_index].lower()).__str__() + "," + total_elements.index(persons_names[person_index].lower()).__str__() + ",0"
    if text not in phrases:
        phrases.append(text)
        i += 1
    print(i)

text = "Amsterdam" + " lives in " + "Peter" \
           + "," + total_elements.index("Peter".lower()).__str__() + "," + total_elements.index("Amsterdam".lower()).__str__() + ",0"

if text not in phrases:
    phrases.append(text)
file3 = open("training_set_26_07_23.txt", "w", encoding="utf-8")

for f in phrases:
    file3.write(f + "\n")

file3.close()