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

import datasets
from datasets import DownloadManager, DatasetInfo, load_dataset, NamedSplit

import nltk
from nltk.corpus import framenet


def create_framenet_subset(k=10, pair_count=(100, 100), print=False):
    nltk.download('framenet_v17')
    if print:
        from pprint import pprint

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

    # length of each document in number of characters
    if print:
        pprint(doc_lengths)

    # relations of 10 frames with most sentences annotated
    USABLE_RELATIONS = ['inheritance', 'using']
    top_k = frame_statistics.most_common(k)
    selection_ids = set()
    selection = []
    for (id, name), count in top_k:
        f1 = framenet.frame(id)
        if print:
            print(name)

        for rel in f1.frameRelations:
            if rel['type']['name'].lower() not in USABLE_RELATIONS:
                continue

            if print:
                pprint(rel)

            f2 = rel['superFrame'] if rel['subFrame'] == f1 else rel['subFrame']
            assert f1 != f2

            count1 = frame_statistics[(f1['ID'], f1['name'])]
            count2 = frame_statistics[(f2['ID'], f2['name'])]
            if print:
                print(count1, count2)

            if count1 > pair_count[0] and count2 > pair_count[1]:
                if f1['ID'] not in selection_ids:
                    selection.append(f1)
                    selection_ids.add(f1['ID'])
                if f2['ID'] not in selection_ids:
                    selection.append(f2)
                    selection_ids.add(f2['ID'])
        if print:
            print()
    if print:
        print([f['name'] for f in selection])

    return selection_ids, selection, [
        sentence
        for id in selection_ids
        for sentence in frame_sentences[id]
    ]


_DESCRIPTION = """
"""


class HFFrameNetConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class HFFrameNet(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        HFFrameNetConfig(
            name="default",
        ),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self, frame_ids=None, frame_role_ids=None) -> DatasetInfo:
        frame_id_f = datasets.Value('string') if frame_ids is None else datasets.ClassLabel(names=frame_ids)
        frame_role_id_f = datasets.Value('string') if frame_ids is None else datasets.ClassLabel(names=frame_role_ids)
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'sentence': datasets.Value('string'),
                'frames': [{
                    'id': frame_id_f,
                    'name': datasets.Value('string'),
                    'frame_elements': [{
                        'id': frame_role_id_f,
                        'name': datasets.Value('string'),
                        'start': datasets.Value('int32'),
                        'end': datasets.Value('int32'),
                    }],
                }]
            }),
        )

    def _split_generators(self, dl_manager: DownloadManager):
        frame_ids, frames, sentences = create_framenet_subset()
        self.frame_ids = list(frame_ids)
        self.frames = frames
        self.sentences = sentences

        frame_role_ids = list(set(
            fe['ID'] for f in frames for fe in f['FE'].values()
        ))

        self.info.update(self._info(frame_ids=list(frame_ids), frame_role_ids=frame_role_ids))
        return [
            datasets.SplitGenerator(name='full', gen_kwargs={}),
        ]

    def _generate_examples(self):
        done = set()

        for sentence in self.sentences:
            if sentence['ID'] in done:
                continue
            else:
                done.add(sentence['ID'])

            yield sentence['ID'], {
                'sentence': sentence['text'],
                'frames': [
                    {
                        'id': str(annotationSet['frameID']),
                        'name': annotationSet['frameName'],
                        'frame_elements': [
                            {
                                'id': str(annotationSet['frame']['FE'][key]['ID']),
                                'name': key, 'start': start, 'end': end
                            } for start, end, key in annotationSet['FE'][0]  # with span
                        ] + [
                            {
                                'id': str(annotationSet['frame']['FE'][key]['ID']),
                                'name': key, 'start': -1, 'end': -1
                            } for key, value in annotationSet['FE'][1].items()  # without span
                        ]
                    }
                    for annotationSet in sentence['annotationSet']
                    if annotationSet['_type'] == 'fulltext_annotationset'
                    and annotationSet['frameID'] in self.frame_ids
                ]
            }


if __name__ == "__main__":
    from pprint import pprint

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
