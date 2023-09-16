import pickle
import random
import re

with open('knowledge.pickle', 'rb') as handle:
    my_dict = pickle.load(handle)

file_animali = open("nomi_di_animali.txt", "r")
animali = []
for f in file_animali:
    # print(f)
    animali.extend([f1.replace(" ", "").rstrip().lower() for f1 in f.split(",")])
animali = sorted(list(set(animali)))
keys = sorted(list(set(my_dict.keys())))

unique_list = animali.copy()
unique_list.extend(keys)
random.shuffle(unique_list)
first_level = []
second_level = []
third_level = []
fourthy_level = []
fifthy_level = []
sixty_level = []
sette_level = []
otto_level = []
nove_level = []
dieci_level = []
nodes = sorted(list(set([f for f in my_dict])))


file_abitats = open("habitat.txt", "r")
habitat = []
for f in file_abitats:
    # print(f)
    habitat.append(f.rstrip().split(":")[1])


file_continenti = open("continenti.txt", "r")
continenti = []
for f in file_continenti:
    # print(f)
    continenti.append(f.rstrip().split(":")[1])

continenti_unique= []
habita_unique= []
for f in continenti:
    continenti_unique.extend(f.replace(" ","").split(","))

for f in habitat:
    habita_unique.extend(f.replace(" ","").split(","))
continenti_unique=sorted(set(list(continenti_unique)))
habita_unique=sorted(list(set(habita_unique)))

for f in my_dict:
    my_list = my_dict[f].split("/")
    first_level.append(my_list[0])
    if len(my_list) > 1:
        second_level.append(my_list[1])
    if len(my_list) > 2:
        third_level.append(my_list[2])
    if len(my_list) > 3:
        fourthy_level.append(my_list[3])
    if len(my_list) > 4:
        fifthy_level.append(my_list[4])
    if len(my_list) > 5:
        sixty_level.append(my_list[5])
    if len(my_list) > 6:
        sette_level.append(my_list[6])
    if len(my_list) > 7:
        otto_level.append(my_list[7])
    if len(my_list) > 8:
        nove_level.append(my_list[8])
    if len(my_list) > 9:
        dieci_level.append(my_list[9])

first_level = sorted(list(set(first_level)))
second_level = sorted(list(set(second_level)))
third_level = sorted(list(set(third_level)))
fourthy_level = sorted(list(set(fourthy_level)))
fifthy_level = sorted(list(set(fifthy_level)))
sixty_level = sorted(list(set(sixty_level)))
sette_level = sorted(list(set(sette_level)))
otto_level = sorted(list(set(otto_level)))
nove_level = sorted(list(set(nove_level)))

sentences = []
action = " is a "
# correct sentences


tmp_animmals = []

for rip in range(4):
    for i in my_dict:
        if tmp_animmals == []:
            tmp_animmals = animali.copy()
            random.shuffle(tmp_animmals)
        random_name = tmp_animmals.pop()

        text = random_name + action + \
               i.split(".")[0].replace("_", " ").replace("-", " ")

        list_of_words_raw = [f for f in my_dict[i].split("/")]
        if len(list_of_words_raw) >= 4:
            text += "," + random_name + "," + unique_list.index(random_name).__str__()
            text += "," + i + "," + unique_list.index(i).__str__()
            for j in list_of_words_raw:
                if j in first_level and j not in unique_list:
                    text += "," + j + "," + first_level.index(j).__str__()

                # habitat
                if j in first_level and j not in unique_list:
                    text += "," + habitat[keys.index(i)].replace(",", "_") + "," + habita_unique.index(habitat[keys.index(i)].replace(" ","")).__str__()
                """
                if j in first_level and j not in unique_list:
                    for cont in habitat[keys.index(i)].replace(" ", "").split(","):
                        index_habit = habita_unique.index(cont)
                        vect_zero = [0 for x in range(len(habita_unique))]
                        vect_zero[index_habit] = 1
                    text += "," + habitat[keys.index(i)].replace(",","_") + "," + ",".join([f.__str__() for f in vect_zero])

                # countries
                if j in first_level and j not in unique_list:
                    for cont in continenti[keys.index(i)].replace(" ", "").split(","):
                        index_cont = continenti_unique.index(cont)
                        vect_zero = [0 for x in range(len(continenti_unique))]
                        vect_zero[index_cont] = 1
                    text += "," + continenti[keys.index(i)].replace(",","_") + "," + ",".join([f.__str__() for f in vect_zero])
                """

                #if j in second_level and j not in unique_list:
                    #text += "," + j + "," + second_level.index(j).__str__()
                #if j in third_level and j not in unique_list:
                    #text += "," + j + "," + third_level.index(j).__str__()
                """
                if j in fourthy_level and j not in unique_list:
                    text += "," + j + "," + fourthy_level.index(j).__str__()
    
                
    
                if j in fifthy_level and j not in unique_list:
                    text += "," + j + "," + fifthy_level.index(j).__str__()
                if j in sixty_level and j not in unique_list:
                    text += "," + j + "," + sixty_level.index(j).__str__()
                if j in sette_level and j not in unique_list:
                    text += "," + j + "," + sette_level.index(j).__str__()
                if j in otto_level and j not in unique_list:
                    text += "," + j + "," + otto_level.index(j).__str__()
                if j in nove_level and j not in unique_list:
                    text += "," + j + "," + nove_level.index(j).__str__()
                """
                if j in unique_list:
                    text += "," + j + "," + unique_list.index(j).__str__()

            text += ",1"
            #habitats_len=sum([int(c) for c in text.split(",")[8:36]])
            #countries_len = sum([int(c) for c in text.split(",")[37:46]])
            sentences.append(text)

    # wrong sentences
    tmp_animmals = []
    for i in my_dict:
        if tmp_animmals == []:
            tmp_animmals = animali.copy()
            random.shuffle(tmp_animmals)
        random_name = tmp_animmals.pop()

        text = i.split(".")[0].replace("_", " ").replace("-", " ") + action + random_name

        list_of_words_raw = [f for f in my_dict[i].split("/")]

        text += "," + i + "," + unique_list.index(i).__str__()
        text += "," + random_name + "," + unique_list.index(random_name).__str__()
        if len(list_of_words_raw) >= 4:
            for j in list_of_words_raw:
                if j in first_level and j not in unique_list:
                    text += "," + j + "," + first_level.index(j).__str__()

                # habitat
                if j in first_level and j not in unique_list:
                    text += "," + habitat[keys.index(i)].replace(",", "_") + "," + habita_unique.index(habitat[keys.index(i)].replace(" ","")).__str__()

                """
                if j in first_level and j not in unique_list:
                    for cont in habitat[keys.index(i)].replace(" ", "").split(","):
                        index_habit = habita_unique.index(cont)
                        vect_zero = [0 for x in range(len(habita_unique))]
                        vect_zero[index_habit] = 1
                    text += "," + habitat[keys.index(i)].replace(",","_") + "," + ",".join([f.__str__() for f in vect_zero])
                # countries
                if j in first_level and j not in unique_list:
                    for cont in continenti[keys.index(i)].replace(" ", "").split(","):
                        index_cont = continenti_unique.index(cont)
                        vect_zero = [0 for x in range(len(continenti_unique))]
                        vect_zero[index_cont] = 1
                    text += "," + continenti[keys.index(i)].replace(",","_") + "," + ",".join([f.__str__() for f in vect_zero])
                """
                """
                if j in second_level and j not in unique_list:
                    text += "," + j + "," + second_level.index(j).__str__()
                if j in third_level and j not in unique_list:
                    text += "," + j + "," + third_level.index(j).__str__()

                
                if j in fourthy_level and j not in unique_list:
                    text += ","  + j+","+ fourthy_level.index(j).__str__()
                
                if j in fifthy_level and j not in unique_list:
                    text += ","  + j+","+ fifthy_level.index(j).__str__()
                if j in sixty_level and j not in unique_list:
                    text += ","  + j+","+ sixty_level.index(j).__str__()
                if j in sette_level and j not in unique_list:
                    text += ","  + j+","+ sette_level.index(j).__str__()
                if j in otto_level and j not in unique_list:
                    text += ","  + j+","+ otto_level.index(j).__str__()
                if j in nove_level  and j not in unique_list:
                    text += ","  + j+","+ nove_level.index(j).__str__()
                """
                if j in unique_list:
                    text += "," + j + "," + unique_list.index(j).__str__()

            text += ",0"
            sentences.append(text)

# for i in sentences:
# print(i)

sentences_test = []
keys_test= list(my_dict.keys())
while (len(sentences_test) != 100):
    random.shuffle(keys_test)
    i = keys_test[0]
    if not tmp_animmals:
        tmp_animmals = animali.copy()
        random.shuffle(tmp_animmals)
    random_name = tmp_animmals.pop()

    text = random_name + action + \
           i.split(".")[0].replace("_", " ").replace("-", " ")

    list_of_words_raw = [f for f in my_dict[i].split("/")]
    if len(list_of_words_raw) >= 4:
        text += "," + random_name + "," + unique_list.index(random_name).__str__()
        text += "," + i + "," + unique_list.index(i).__str__()
        for j in list_of_words_raw:
            if j in first_level and j not in unique_list:
                text += "," + j + "," + first_level.index(j).__str__()
                # habitat
            if j in first_level and j not in unique_list:
                text += "," + habitat[keys.index(i)].replace(",", "_") + "," + habita_unique.index(habitat[keys.index(i)].replace(" ","")).__str__()
            """
            if j in first_level and j not in unique_list:
                for cont in habitat[keys.index(i)].replace(" ", "").split(","):
                    index_habit = habita_unique.index(cont)
                    vect_zero = [0 for x in range(len(habita_unique))]
                    vect_zero[index_habit] = 1
                text += "," + habitat[keys.index(i)].replace(",","_") + "," + ",".join([f.__str__() for f in vect_zero])
            # countries
            if j in first_level and j not in unique_list:
                for cont in continenti[keys.index(i)].replace(" ", "").split(","):
                    index_cont = continenti_unique.index(cont)
                    vect_zero = [0 for x in range(len(continenti_unique))]
                    vect_zero[index_cont] = 1
                text += "," + continenti[keys.index(i)].replace(",","_") + "," + ",".join([f.__str__() for f in vect_zero])
            """
            """
            if j in second_level and j not in unique_list:
                text += "," + j + "," + second_level.index(j).__str__()
            if j in third_level and j not in unique_list:
                text += "," + j + "," + third_level.index(j).__str__()
            
            if j in fourthy_level and j not in unique_list:
                text += "," + j + "," + fourthy_level.index(j).__str__()



            if j in fifthy_level and j not in unique_list:
                text += "," + j + "," + fifthy_level.index(j).__str__()
            if j in sixty_level and j not in unique_list:
                text += "," + j + "," + sixty_level.index(j).__str__()
            if j in sette_level and j not in unique_list:
                text += "," + j + "," + sette_level.index(j).__str__()
            if j in otto_level and j not in unique_list:
                text += "," + j + "," + otto_level.index(j).__str__()
            if j in nove_level and j not in unique_list:
                text += "," + j + "," + nove_level.index(j).__str__()
            """
            if j in unique_list:
                text += "," + j + "," + unique_list.index(j).__str__()

        text += ",1"
        if text not in sentences and  text not in sentences_test:
            sentences_test.append(text)

    # wrong sentences
    tmp_animmals = []
    random.shuffle(keys_test)
    i = keys_test[0]
    if tmp_animmals == []:
        tmp_animmals = animali.copy()
        random.shuffle(tmp_animmals)
    random_name = tmp_animmals.pop()

    text = i.split(".")[0].replace("_", " ").replace("-", " ") + action + random_name

    list_of_words_raw = [f for f in my_dict[i].split("/")]

    text += "," + i + "," + unique_list.index(i).__str__()
    text += "," + random_name + "," + unique_list.index(random_name).__str__()
    if len(list_of_words_raw) >= 4:
        for j in list_of_words_raw:
            if j in first_level and j not in unique_list:
                text += "," + j + "," + first_level.index(j).__str__()

            # habitat
            if j in first_level and j not in unique_list:
                text += "," + habitat[keys.index(i)].replace(",", "_") + "," + habita_unique.index(habitat[keys.index(i)].replace(" ","")).__str__()
            """
            if j in first_level and j not in unique_list:
                for cont in habitat[keys.index(i)].replace(" ", "").split(","):
                    index_habit = habita_unique.index(cont)
                    vect_zero = [0 for x in range(len(habita_unique))]
                    vect_zero[index_habit] = 1
                text += "," + habitat[keys.index(i)].replace(",","_") + "," + ",".join([f.__str__() for f in vect_zero])
            # countries
            if j in first_level and j not in unique_list:
                for cont in continenti[keys.index(i)].replace(" ", "").split(","):
                    index_cont = continenti_unique.index(cont)
                    vect_zero = [0 for x in range(len(continenti_unique))]
                    vect_zero[index_cont] = 1
                text += "," + continenti[keys.index(i)].replace(",","_") + "," + ",".join([f.__str__() for f in vect_zero])
            """
            """
            if j in second_level and j not in unique_list:
                text += "," + j + "," + second_level.index(j).__str__()
            if j in third_level and j not in unique_list:
                text += "," + j + "," + third_level.index(j).__str__()

            
            if j in fourthy_level and j not in unique_list:
                text += ","  + j+","+ fourthy_level.index(j).__str__()

            if j in fifthy_level and j not in unique_list:
                text += ","  + j+","+ fifthy_level.index(j).__str__()
            if j in sixty_level and j not in unique_list:
                text += ","  + j+","+ sixty_level.index(j).__str__()
            if j in sette_level and j not in unique_list:
                text += ","  + j+","+ sette_level.index(j).__str__()
            if j in otto_level and j not in unique_list:
                text += ","  + j+","+ otto_level.index(j).__str__()
            if j in nove_level  and j not in unique_list:
                text += ","  + j+","+ nove_level.index(j).__str__()
            """
            if j in unique_list:
                text += "," + j + "," + unique_list.index(j).__str__()

        text += ",0"
        if text not in sentences and text not in sentences_test:
            sentences_test.append(text)


f = open("training_dataset_24_08_23.txt", "w")
for i in sentences:
    f.write(i + "\n")
f.close()

f = open("test_dataset_24_08_23.txt", "w")
for i in sentences_test:
    f.write(i + "\n")
f.close()



dict_wordnet = {}
dict_wordnet["first_level"] = first_level
dict_wordnet["second_level"] = second_level
dict_wordnet["third_level"] = third_level
dict_wordnet["fourthy_level"] = fourthy_level
dict_wordnet["fifthy_level"] = fifthy_level
dict_wordnet["sixty_level"] = sixty_level
dict_wordnet["seventh_level"] = sette_level
dict_wordnet["eighth_level"] = otto_level
dict_wordnet["continenti_unique"] = continenti_unique
dict_wordnet["continenti"] = continenti
dict_wordnet["habita_unique"] = habita_unique
dict_wordnet["habitat"] = habitat

with open('dict_wordnet_24_08_23.pickle', 'wb') as handle:
    pickle.dump(dict_wordnet, handle, protocol=pickle.HIGHEST_PROTOCOL)
