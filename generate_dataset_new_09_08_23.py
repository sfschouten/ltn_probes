import random
import re

from tqdm import tqdm

file = open("name_person.txt", "r", encoding='utf-8')
persons_names = []
for f in file:
    persons_names.append(f.strip().replace(" ", "_"))

file = open("only_city_09_08_23.txt", "r")
cities = []
nations = []
for f in file:
    city = f.split(",")[0]
    nation = f.split(",")[1].rstrip().split("\t")[0]
    city = re.sub("\(.*?\)", "()", city)
    city = city.replace("()", "").replace(" ", "_")
    cities.append(city)
    nations.append(nation.replace(" ", "_"))

phrases = []
i = 0
# first_index_person, second_index_city, correct_sentence
while i < 1000:
    city_index = random.randint(0, len(cities) - 1)
    person_index = random.randint(0, len(persons_names) - 1)
    text = persons_names[person_index].lower().capitalize().__str__() + " lives in " + cities[
        city_index].lower().capitalize().__str__()
    if text not in phrases:
        phrases.append(text)
        i += 1
    print(i)

text = "Peter" + " lives in " + "Amsterdam"

if text not in phrases:
    phrases.append(text)
i = 0
# first_index_person, second_index_city, correct_sentence

incorrect_sentences = []
while i < 1000:
    city_index = random.randint(0, len(cities) - 1)
    person_index = random.randint(0, len(persons_names) - 1)
    text = cities[city_index].lower().capitalize().__str__() + " lives in " + persons_names[
        person_index].lower().capitalize().__str__()
    if text not in incorrect_sentences:
        incorrect_sentences.append(text)
        i += 1
    print(i)

text = "Amsterdam" + " lives in " + "Peter"
if text not in incorrect_sentences:
    incorrect_sentences.append(text)

person_used = [f.split(" ")[0] for f in phrases]
person_used.extend([f.split(" ")[-1] for f in incorrect_sentences])
city_used = [f.split(" ")[0] for f in incorrect_sentences]
city_used.extend([f.split(" ")[-1] for f in phrases])
# action_used= [f.split(" ")[2] for f in phrases]
# action_used.extend([f.split(" ")[2] for f in incorrect_sentences])
person_used = sorted(list(set(person_used)))
city_used = sorted(list(set(city_used)))
action_used = ["lives in"]
total_elements = list(set(person_used))
total_elements.extend(list(set(city_used)))
total_elements.extend(list(set(action_used)))
total_elements = list(set(total_elements))
#total_elements.extend(sorted(list(set([f1.replace("_", "") for f1 in nations]))))
total_nations = list(sorted(list(set([f1.replace("_", "") for f1 in nations]))))
my_dict = {}

print("total_elements",len(total_elements))
for_iteration = 0
for f in total_elements:
    my_dict[f] = 0
    for_iteration += 1

added_row = []
file3 = open("training_set_09_08_23.txt", "w", encoding="utf-8")

for f in phrases:
    elements = f.split(" ")
    file3.write(f.replace("_", " ") + "," + nations[
        cities.index(elements[3])].replace("_", "") + "," + total_elements.index(
        elements[0]).__str__() + "," + total_elements.index(
        "lives in").__str__() + "," + total_elements.index(elements[-1]).__str__() + "," + total_nations.index(nations[
                                                                                                                    cities.index(
                                                                                                                        elements[
                                                                                                                            3])].replace(
        "_", "")).__str__() + "," + "1" + "\n")
    added_row.append(f + "," + nations[cities.index(elements[3])].replace("_", "") + "," + total_elements.index(
        elements[0]).__str__() + "," + total_elements.index(
        "lives in").__str__() + "," + total_elements.index(elements[-1]).__str__() + "," +
                     total_nations.index(
                         nations[cities.index(elements[3])].replace("_", "")).__str__() + "," + "1" + "\n")
    my_dict[elements[0]] += 1
    my_dict[elements[-1]] += 1
    my_dict["lives in"] += 1

for f in incorrect_sentences:
    elements = f.split(" ")
    file3.write(f.replace("_", " ") + "," + nations[
        cities.index(elements[0])] + "," + total_elements.index(elements[0]).__str__() + "," + total_elements.index(
        "lives in").__str__() + "," + total_elements.index(elements[-1]).__str__() + "," + total_nations.index(nations[
                                                                                                                    cities.index(
                                                                                                                        elements[
                                                                                                                            0])].replace(
        "_", "")).__str__() + "," + "0" + "\n")
    added_row.append(f + "," + nations[
                         cities.index(elements[0])].replace("_", "") + "," + total_elements.index(elements[0]).__str__() + "," + total_elements.index(
        "lives in").__str__() + "," + total_elements.index(elements[-1]).__str__()  + "," + total_nations.index(nations[
                                                                                                      cities.index(
                                                                                                          elements[
                                                                                                              0])].replace(
        "_", "")).__str__() + "," + "0" + "\n")
    my_dict[elements[0]] += 1
    my_dict[elements[-1]] += 1
    my_dict["lives in"] += 1

file3.close()
