import pickle as pickle
import random
import re

from tqdm import tqdm

with open('knowledge.pickle', 'rb') as handle:
    my_dict = pickle.load(handle)

file_animali = open("nomi_di_animali.txt", "r")
animali = []
for f in file_animali:
    # print(f)
    animali.extend([f1.replace(" ", "").rstrip().lower() for f1 in f.split(",")])
animali = sorted(list(set(animali)))
keys = sorted(list(set(my_dict.keys())))

random.shuffle(keys)
# keys=keys[:250]
keys = sorted(list(set(keys)))
my_dict_new = {}
for f in keys:
    if f in my_dict:
        my_dict_new[f] = my_dict[f]

my_dict = my_dict_new

unique_list = animali.copy()
unique_list.extend(keys)
#random.shuffle(unique_list)
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

continenti_unique = []
habita_unique = []
for f in continenti:
    continenti_unique.extend(f.replace(" ", "").split(","))

for f in habitat:
    habita_unique.extend(f.replace(" ", "").split(","))
continenti_unique = sorted(set(list(continenti_unique)))
habita_unique = sorted(list(set(habita_unique)))

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
used_animals=[]
used_names=[]
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
        if len(list_of_words_raw) > 0:
            text += "," + random_name + "," + unique_list.index(random_name).__str__()
            text += "," + i + "," + unique_list.index(i).__str__()
            for j in list_of_words_raw:
                if j in first_level and j not in unique_list:
                    text += "," + j + "," + first_level.index(j).__str__()

                # habitat
                if j in first_level and j not in unique_list:
                    text += "," + habitat[keys.index(i)].replace(",", "_") + "," + habita_unique.index(
                        habitat[keys.index(i)].replace(" ", "")).__str__()


            text += ",1,0"
            # habitats_len=sum([int(c) for c in text.split(",")[8:36]])
            # countries_len = sum([int(c) for c in text.split(",")[37:46]])
            sentences.append(text)
            used_animals.append(i)
            used_names.append(random_name)

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
        if len(list_of_words_raw) > 0:
            for j in list_of_words_raw:
                if j in first_level and j not in unique_list:
                    text += "," + j + "," + first_level.index(j).__str__()

                # habitat
                if j in first_level and j not in unique_list:
                    text += "," + habitat[keys.index(i)].replace(",", "_") + "," + habita_unique.index(
                        habitat[keys.index(i)].replace(" ", "")).__str__()




            text += ",0,0"

            sentences.append(text)
            tmp_text=text
    """
    for i in first_level:
        trovato = 0
        while trovato==0:
            random_name = random.choice(animali)
            text = random_name.split(".")[0].replace("_", " ").replace("-", " ") + action + i.split(".")[0].replace("_"," ")
            text += "," + random_name + "," + unique_list.index(random_name).__str__()
            text += "," + i + "," + first_level.index(i).__str__()
            text += "," + i + "," + first_level.index(i).__str__() # category as fake
            text += "," + i + "," + first_level.index(i).__str__()  # object as fake

            text += ",1,1"
            if text not in sentences:
                sentences.append(text)
                trovato=1
    """

"""

f = open("training_dataset_16_09_23.txt", "r")
sentences=[]
for x in f:
    sentences.append(x.rstrip())

f = open("test_dataset_16_09_23.txt", "r")
sentences_test=[]
for x in f:
    sentences_test.append(x.rstrip())

"""
#faccio il test e aggiungo doppie frasi per il test finale

sentences_test = []
keys_test = list(my_dict.keys())
normal_sentence=0
while normal_sentence <= len(keys_test)/2:
    random.shuffle(keys_test)
    i = keys_test[0]
    if not tmp_animmals:
        tmp_animmals = animali.copy()
        random.shuffle(tmp_animmals)
    random_name = tmp_animmals.pop()


    text = random_name + action + \
           i.split(".")[0].replace("_", " ").replace("-", " ")

    list_of_words_raw = [f for f in my_dict[i].split("/")]
    if len(list_of_words_raw) > 0:
        text += "," + random_name + "," + unique_list.index(random_name).__str__()
        text += "," + i + "," + unique_list.index(i).__str__()
        for j in list_of_words_raw:
            if j in first_level and j not in unique_list:
                text += "," + j + "," + first_level.index(j).__str__()
                # habitat
            if j in first_level and j not in unique_list:
                text += "," + habitat[keys.index(i)].replace(",", "_") + "," + habita_unique.index(
                    habitat[keys.index(i)].replace(" ", "")).__str__()


        text += ",1,0"

        if text not in sentences and text not in sentences_test:
            normal_sentence+=1
            sentences_test.append(text)


            text2 = random_name + action + text.split(",")[5].split(".")[0]
            text2 += "," + random_name + "," + unique_list.index(random_name).__str__()
            text2 += "," + text.split(",")[5] + "," + first_level.index(text.split(",")[5]).__str__() #object
            text2 += "," + text.split(",")[5] + "," + first_level.index(text.split(",")[5]).__str__() #category
            text2 += "," + "None" + "," + "5" #fake habita
            #text2 += "," + text.split(",")[5] + "," + first_level.index(text.split(",")[5]).__str__().__str__()
            text2 += ",1,1"
            sentences_test.append(text2)




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
    if len(list_of_words_raw) > 0:
        for j in list_of_words_raw:
            if j in first_level and j not in unique_list:
                text += "," + j + "," + first_level.index(j).__str__()

            # habitat
            if j in first_level and j not in unique_list:
                text += "," + habitat[keys.index(i)].replace(",", "_") + "," + habita_unique.index(
                    habitat[keys.index(i)].replace(" ", "")).__str__()




        text += ",0,0"

        if text not in sentences and text not in sentences_test:
            sentences_test.append(text)

            text2 = text.split(",")[5].split(".")[0] + action + random_name
            text2 += "," + text.split(",")[5] + "," + first_level.index(text.split(",")[5]).__str__() #subject
            text2 += "," + random_name + "," + unique_list.index(random_name).__str__() #object
            text2 += "," + text.split(",")[5] + "," + first_level.index(text.split(",")[5]).__str__()  #category
            text2 += "," + "None" + "," + "5"  # fake habita

            text2 += ",0,1"
            sentences_test.append(text2)

#faccio il validation e aggiungo doppie frasi per il test finale

sentences_valid = []
keys_test = list(my_dict.keys())
normal_sentence=0
while normal_sentence <= len(keys_test)/2:
    random.shuffle(keys_test)
    i = keys_test[0]
    if not tmp_animmals:
        tmp_animmals = animali.copy()
        random.shuffle(tmp_animmals)
    random_name = tmp_animmals.pop()


    text = random_name + action + \
           i.split(".")[0].replace("_", " ").replace("-", " ")

    list_of_words_raw = [f for f in my_dict[i].split("/")]
    if len(list_of_words_raw) > 0:
        text += "," + random_name + "," + unique_list.index(random_name).__str__()
        text += "," + i + "," + unique_list.index(i).__str__()
        for j in list_of_words_raw:
            if j in first_level and j not in unique_list:
                text += "," + j + "," + first_level.index(j).__str__()
                # habitat
            if j in first_level and j not in unique_list:
                text += "," + habitat[keys.index(i)].replace(",", "_") + "," + habita_unique.index(
                    habitat[keys.index(i)].replace(" ", "")).__str__()


        text += ",1,0"

        if text not in sentences and text not in sentences_test and text not in sentences_valid :
            normal_sentence+=1
            sentences_valid.append(text)


            text2 = random_name + action + text.split(",")[5].split(".")[0]
            text2 += "," + random_name + "," + unique_list.index(random_name).__str__()
            text2 += "," + text.split(",")[5] + "," + first_level.index(text.split(",")[5]).__str__() #object
            text2 += "," + text.split(",")[5] + "," + first_level.index(text.split(",")[5]).__str__() #category
            text2 += "," + "None" + "," + "5" #fake habita
            #text2 += "," + text.split(",")[5] + "," + first_level.index(text.split(",")[5]).__str__().__str__()
            text2 += ",1,1"
            sentences_valid.append(text2)




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
    if len(list_of_words_raw) > 0:
        for j in list_of_words_raw:
            if j in first_level and j not in unique_list:
                text += "," + j + "," + first_level.index(j).__str__()

            # habitat
            if j in first_level and j not in unique_list:
                text += "," + habitat[keys.index(i)].replace(",", "_") + "," + habita_unique.index(
                    habitat[keys.index(i)].replace(" ", "")).__str__()




        text += ",0,0"

        if text not in sentences and text not in sentences_test and text not in sentences_valid:
            sentences_valid.append(text)

            text2 = text.split(",")[5].split(".")[0] + action + random_name
            text2 += "," + text.split(",")[5] + "," + first_level.index(text.split(",")[5]).__str__() #subject
            text2 += "," + random_name + "," + unique_list.index(random_name).__str__() #object
            text2 += "," + text.split(",")[5] + "," + first_level.index(text.split(",")[5]).__str__()  #category
            text2 += "," + "None" + "," + "5"  # fake habita

            text2 += ",0,1"
            sentences_valid.append(text2)


"""
for i in first_level:
    trovato = 0
    while trovato == 0:
        random_name = random.choice(animali)
        text = i.split(".")[0].replace("_", " ")  + action + random_name.split(".")[0].replace("_", " ").replace("-", " ")
        text += "," + i + "," + first_level.index(i).__str__()
        text += "," + random_name + "," + unique_list.index(random_name).__str__()
        text += "," + i + "," + first_level.index(i).__str__()  # category as fake
        text  += "," + "None" + "," + "5"  # fake habita

        text += ",0,1"
        if text not in sentences and text not in sentences_test:
            sentences.append(text)
            trovato = 1



# non puoi farlo nel test
for i in first_level:
    trovato = 0
    while trovato == 0:
        random_name = random.choice(animali)
        text = random_name.split(".")[0].replace("_", " ").replace("-", " ") + action + i.split(".")[0].replace("_",
                                                                                                                " ")
        text += "," + random_name + "," + unique_list.index(random_name).__str__()
        text += "," + i + "," + first_level.index(i).__str__()
        text += "," + i + "," + first_level.index(i).__str__()  # category as fake
        text  += "," + "None" + "," + "5"  # fake habita

        text += ",1,1"
        if text not in sentences and text not in sentences_test:
            sentences_test.append(text)
            trovato = 1
"""



f = open("training_dataset_16_09_23.txt", "w")
for i in sentences:
    f.write(i + "\n")
f.close()

f = open("test_dataset_16_09_23.txt", "w")
for i in sentences_test:
    f.write(i + "\n")
f.close()


f = open("valid_dataset_16_09_23.txt", "w")
for i in sentences_valid:
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

with open('dict_wordnet_16_09_23.pickle', 'wb') as handle:
    pickle.dump(dict_wordnet, handle, protocol=pickle.HIGHEST_PROTOCOL)
