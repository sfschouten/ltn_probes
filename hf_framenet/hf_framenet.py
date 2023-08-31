"""
Goal: probe for semantic role labeling task

Method:
  - for each frame:
    - a set of query vectors, one for each possible argument/role of the frame
        - used to accumulate features from the sentence relevant to determining if argument/role/frame are present
    - a predicate for each role/argument to determine if it is present
    - a predicate for the frame if it in its entirety is present

  - presence of argument of frame implies presence of frame
  - presence of frame implies presence of mandatory arguments
"""
from collections import Counter
from tqdm import tqdm

import datasets as ds
from datasets import DownloadManager, DatasetInfo, load_dataset, NamedSplit

import nltk
from nltk.corpus import framenet


# the frame-to-frame relations we consider
USABLE_RELATIONS = [
    'inheritance',          # IS-A relation, each FE in the parent is bound to a corresponding FE in the child
    'using',                # child presupposes parent (e.g. Speed presupposes Motion)
    'subframe',             # subevent (e.g. “Criminal_process” has subframes of “Arrest”, “Arraignment”, etc.)
    'perspective_on',       # particular perspective (e.g. “Hiring”, “Get_a_job” -> “Employment_start”)
    # 'reframing_mapping',  #
    # 'causative_of',       #
    # 'see_also',           #
]


def create_framenet_subset(min_impl_by=5, max_impl_by=10, min_nr_samples=100, log=False):
    nltk.download('framenet_v17')
    if log:
        from pprint import pprint

    # Count annotations
    annotated_docs = framenet.docs()
    frame_statistics = Counter()
    frame_sentences = {f['ID']: [] for f in framenet.frames()}
    doc_lengths = Counter()
    for doc in tqdm(annotated_docs, position=0):
        doc_length = 0
        for sentence in doc.sentence:
            doc_length += len(sentence)
            for frameset in sentence.annotationSet:
                if 'frameID' in frameset:
                    frame_statistics[(frameset['frameID'], frameset['frameName'])] += 1
                    frame_sentences[frameset['frameID']].append(sentence)

        doc_lengths[doc['ID'], doc['filename']] = doc_length

    # Find candidates
    implied_by = {f['ID']: [] for f in framenet.frames()}
    for (id, name), count in frame_statistics.items():
        if count < min_nr_samples:
            continue

        # BFS over frame-to-frame relation graph
        queue = [(id, {fe['ID']: (fe['ID'], fe['name']) for fe in framenet.frame(id)['FE'].values()})]
        visited = {id}
        while queue:
            current_id, current_map = queue.pop()
            current_frame = framenet.frame(current_id)

            for rel in current_frame.frameRelations:
                if rel['type']['name'].lower() not in USABLE_RELATIONS:
                    continue  # only usable relations
                if rel['subFrame'] != current_frame:
                    continue  # only move in direction of implication
                neighbour_id = rel['superFrame']['ID']
                if neighbour_id in visited:
                    continue  # only visit once

                new_map = {
                    fe_rel['supID']: current_map[fe_rel['subID']]
                    for fe_rel in rel['feRelations']
                    if fe_rel['subID'] in current_map
                }

                visited.add(neighbour_id)
                queue.append((neighbour_id, new_map))

                # there is a path from the original frame to this one
                implied_by[neighbour_id].append((id, new_map))

    candidates = {
        f_id: (
            framenet.frame(f_id),
            {f2_id: (framenet.frame(f2_id), _map) for f2_id, _map in impl_by}
        )
        for f_id, impl_by in implied_by.items()
        if max_impl_by >= len(impl_by) >= min_impl_by
    }

    if log:
        pprint({
            f_id: (f['name'], {f2_id: f2['name'] for f2_id, (f2, _) in impl_by.items()})
            for f_id, (f, impl_by) in candidates.items()
        })

    return candidates, frame_sentences


_DESCRIPTION = """
"""


class HFFrameNetConfig(ds.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def _class_name(id, name, status=None):
    if status:
        return f'{id}_{name}_({status})'
    else:
        return f'{id}_{name}'


class HFFrameNet(ds.GeneratorBasedBuilder):

    VERSION = ds.Version("0.0.1")

    BUILDER_CONFIGS = [
        HFFrameNetConfig(
            name="default",
        ),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self, frame_names=None, frame_role_names=None) -> DatasetInfo:
        frame_id_f = ds.Value('string') if frame_names is None else ds.ClassLabel(names=frame_names)
        frame_role_id_f = ds.Value('string') if frame_role_names is None else ds.ClassLabel(names=frame_role_names)
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=ds.Features({
                'sentence': ds.Value('string'),
                'frames': [{
                    'id': frame_id_f,
                    'implied_by': frame_id_f,
                    'frame_elements': [{
                        'id': frame_role_id_f,
                    }],
                }]
            }),
        )

    def _split_generators(self, dl_manager: DownloadManager):
        frames, sentences = create_framenet_subset()
        self.sentences = sentences

        self.implied_frames = {(f_id, f['name']) for f_id, (f, _) in frames.items()}
        self.asserted_frames = {
            (f_id, f['name'])
            for _, (_, impl_by) in frames.items() for f_id, (f, _) in impl_by.items()
        }
        self.asserted_to_implied = {key: set() for key, _ in self.asserted_frames}
        for f1_id, (f1, impl_by) in frames.items():
            for f2_id, (_, fe_map) in impl_by.items():
                self.asserted_to_implied[f2_id].add((f1_id, f1['name'], tuple(v for v in fe_map.values())))

        self.implied_frame_roles = {(fe['ID'], fe['name']) for (f, _) in frames.values() for fe in f['FE'].values()}
        self.asserted_frame_roles = {
            (fe['ID'], fe['name'])
            for (_, impl_by) in frames.values() for (f, _) in impl_by.values() for fe in f['FE'].values()
        }

        frame_names = ([_class_name(id, name, 'asserted') for id, name in self.asserted_frames]
                       + [_class_name(id, name, 'implied') for id, name in self.implied_frames])
        self.info.update(self._info(
            frame_names=frame_names,
            frame_role_names=[_class_name(id, name) for id, name in self.asserted_frame_roles],
        ))
        return [ds.SplitGenerator(name='full')]

    def _generate_examples(self):
        asserted_frame_ids, _ = zip(*self.asserted_frames)
        implied_frame_ids, _ = zip(*self.implied_frames)
        done = set()
        for _, sentences in self.sentences.items():
            for sentence in sentences:
                if sentence['ID'] in done:
                    continue
                else:
                    done.add(sentence['ID'])

                def fe_elements(annotationSet):
                    return [
                        (annotationSet['frame']['FE'][key]['ID'], key)
                        for _, _, key in annotationSet['FE'][0]  # with span
                    ] + [
                        (annotationSet['frame']['FE'][key]['ID'], key)
                        for key, _ in annotationSet['FE'][1].items()  # without span
                    ]

                yield sentence['ID'], {
                    'sentence': sentence['text'],
                    'frames': [
                        {
                            'id': _class_name(annotationSet['frameID'], annotationSet['frameName'], 'asserted'),
                            'implied_by': None,
                            'frame_elements': [
                                {'id': _class_name(id, name)} for id, name in fe_elements(annotationSet)
                            ]
                        }
                        for annotationSet in sentence['annotationSet']
                        if annotationSet['_type'] == 'fulltext_annotationset'
                        and annotationSet['frameID'] in asserted_frame_ids
                    ] + [
                        {
                            'id': _class_name(implied_id, implied_name, 'implied'),
                            'implied_by': _class_name(
                                annotationSet['frameID'], annotationSet['frameName'], 'asserted'),
                            'frame_elements': [
                                {
                                    'id': _class_name(fe_id, fe_name),
                                } for (fe_id, fe_name) in fe
                                if fe_id in {id for id, _ in fe_elements(annotationSet)}
                            ],
                        }
                        for annotationSet in sentence['annotationSet']
                        if annotationSet['_type'] == 'fulltext_annotationset'
                        if annotationSet['frameID'] in self.asserted_to_implied
                        for implied_id, implied_name, fe in self.asserted_to_implied[annotationSet['frameID']]
                    ]
                }


if __name__ == "__main__":
    from pprint import pprint

    create_framenet_subset(log=True)

    # load a dataset
    dataset = load_dataset(__file__).shuffle()

    config_name = dataset['full'].config_name

    # print some samples
    for i, test in enumerate(dataset['full']):
        print(i)
        pprint(test)
        print()
        if i >= 9:
            break
