import pickle

from nltk.corpus import wordnet as wn
first_level= [ f for f in list(wn.synset('placental.n.01').hyponyms()) if "01" in f._name]
max_len=0

my_dict={}
for f in first_level:
    if f.hyponyms():
        if len(f.hyponyms()) == 0:
            print(f._name)


        for f1 in f.hyponyms():
            if len(f1.hyponyms()) == 0:
                print(f._name, f1._name)
                my_dict[f1._name]="/".join([f._name])
            for f2 in f1.hyponyms():
                if len(f2.hyponyms()) == 0:
                    print(f._name, f1._name, f2._name)
                    my_dict[f2._name] = "/".join([f._name, f1._name,f2._name])

                for f3 in f2.hyponyms():
                    if len(f3.hyponyms()) == 0:
                        print(f._name, f1._name, f2._name, f3._name)
                        my_dict[f3._name] = "/".join([f._name, f1._name, f2._name,f3._name])

                    for f4 in f3.hyponyms():
                        if len(f4.hyponyms()) == 0:
                            print(f._name, f1._name, f2._name, f3._name, f4._name)
                            my_dict[f4._name] = "/".join([f._name, f1._name, f2._name,f3._name,f4._name])
                        for f5 in f4.hyponyms():
                            if len(f5.hyponyms()) == 0:
                                print(f._name, f1._name, f2._name, f3._name, f4._name, f5._name)
                                my_dict[f5._name] = "/".join([f._name, f1._name, f2._name, f3._name, f4._name,f5._name])
                            for f6 in f5.hyponyms():
                                if len(f6.hyponyms())==0:
                                    print(f._name, f1._name, f2._name, f3._name, f4._name, f5._name, f6._name)
                                    my_dict[f6._name] = "/".join([f._name, f1._name, f2._name, f3._name, f4._name, f5._name,f6._name])
                                for f7 in f6.hyponyms():
                                    print(f._name, f1._name, f2._name, f3._name, f4._name, f5._name, f6._name, f7._name)
                                    my_dict[f7._name] = "/".join([f._name, f1._name, f2._name, f3._name, f4._name,
                                                                 f5._name,f6._name,f7._name])

print(my_dict)
with open('data/knowledge.pickle', 'wb') as handle:
    pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
for synset in (wn.synsets('tiger')):
        print (synset)
        #nyms = ['hypernyms', 'hyponyms', 'meronyms', 'holonyms', 'part_meronyms', 'sisterm_terms', 'troponyms', 'inherited_hypernyms']
        nyms = ['meronyms', 'part_meronyms']
        for i in nyms:
            try:
                print (getattr(synset, i)())
            except AttributeError as e:
                print (e)
                pass
"""
