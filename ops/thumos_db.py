#from .utils import *
import os
import glob


class Instance(object):
    """
    Representing an instance of activity in the videos
    """

    def __init__(self, idx, anno, vid_id, vid_info, name_num_mapping):
        self._starting, self._ending = anno['segment'][0], anno['segment'][1]
        self._str_label = anno['label']
        self._total_duration = vid_info['duration']
        self._idx = idx
        self._vid_id = vid_id
        self._file_path = None

        if name_num_mapping:
            self._num_label = name_num_mapping[self._str_label]

    @property
    def time_span(self):
        return self._starting, self._ending

    @property
    def covering_ratio(self):
        return self._starting / float(self._total_duration), self._ending / float(self._total_duration)

    @property
    def num_label(self):
        return self._num_label

    @property
    def label(self):
        return self._str_label

    @property
    def name(self):
        return '{}_{}'.format(self._vid_id, self._idx)

    @property
    def path(self):
        if self._file_path is None:
            raise ValueError("This instance is not associated to a file on disk. Maybe the file is missing?")
        return self._file_path

    @path.setter
    def path(self, path):
        self._file_path = path


class Video(object):
    """
    This class represents one video in the activity-net db
    """
    def __init__(self, key, info, name_idx_mapping=None):
        self._id = key
        self._info_dict = info
        self._instances = [Instance(i, x, self._id, self._info_dict, name_idx_mapping)
                           for i, x in enumerate(self._info_dict['annotations'])]
        self._file_path = None

    @property
    def id(self):
        return self._id

    @property
    def url(self):
        return self._info_dict['url']

    @property
    def instances(self):
        return self._instances

    @property
    def duration(self):
        return self._info_dict['duration']

    @property
    def subset(self):
        return self._info_dict['subset']

    @property
    def instance(self):
        return self._instances

    @property
    def path(self):
        if self._file_path is None:
            raise ValueError("This video is not associated to a file on disk. Maybe the file is missing?")
        return self._file_path

    @path.setter
    def path(self, path):
        self._file_path = path


class THUMOSDB(object):
    """
    This class is the abstraction of the thumos db
    """

    _CONSTRUCTOR_LOCK = object()

    def __init__(self, token):
        """
        Disabled constructor
        :param token:
        :return:
        """
        if token is not self._CONSTRUCTOR_LOCK:
            raise ValueError("Use get_db to construct an instance, do not directly use the constructor")

    @classmethod
    def get_db(cls, year=14):
        """
        Build the internal representation of THUMOS14 Net databases
        We use the alphabetic order to transfer the label string to its numerical index in learning
        :param version:
        :return:
        """
        if year not in [14, 15]:
            raise ValueError("Unsupported challenge year {}".format(year))

        import os
        db_info_folder = 'data/thumos_{}'.format(year)

        me = cls(cls._CONSTRUCTOR_LOCK)
        me.year = year
        me.ignore_labels = ['Ambiguous']
        me.prepare_data(db_info_folder)

        return me

    def prepare_data(self, db_folder):

        def load_subset_info(subset):
            duration_file = '{}_durations.txt'.format(subset)
            annotation_folder = 'temporal_annotations_{}'.format(subset)
            annotation_files = glob.glob(os.path.join(db_folder, annotation_folder, '*'))
            avoid_file = '{}_avoid_videos.txt'.format(subset)

            durations_lines = [x.strip() for x in open(os.path.join(db_folder, duration_file))]
            annotaion_list = [(os.path.basename(f).split('_')[0], list(open(f))) for f in annotation_files]
            avoid_list = [x.strip().split() for x in open(os.path.join(db_folder, avoid_file))]

            avoid_set = set(['-'.join(x) for x in avoid_list])
            print("Loading avoid set:")
            print(avoid_set)

            #process video info
            video_names = [durations_lines[i].split('.')[0] for i in range(0, len(durations_lines), 2)]
            video_durations = [durations_lines[i] for i in range(1, len(durations_lines), 2)]
            video_info = list(zip(video_names, video_durations))

            duration_dict = dict(video_info)

            # reorganize annotation to attach them to videos
            video_table = {v: list() for v in video_names}
            for cls_name, annotations in annotaion_list:
                for a in annotations:
                    items = a.strip().split()
                    vid = items[0]
                    st, ed = float(items[1]), float(items[2])
                    if ('{}-{}'.format(vid, cls_name) not in avoid_set) and (st <= float(duration_dict[vid])):
                        video_table[vid].append((cls_name, st, ed))

            return video_info, video_table, annotation_files

        def construct_video_dict(video_info, annotaion_table, subset, name_idx_mapping):
            video_dict = {}
            instance_dict = {}
            for v in video_info:
                info_dict = {
                    'duration': float(v[1]),
                    'subset': subset,
                    'url': None,
                    'annotations': [
                        {'label': item[0], 'segment': (item[1], item[2])} for item in annotaion_table[v[0]] if item[0] not in self.ignore_labels
                    ]
                }
                video_dict[v[0]] = Video(v[0], info_dict, name_idx_mapping)
                instance_dict.update({i.name: i for i in video_dict[v[0]].instance})
            return video_dict, instance_dict

        self._validation_info = load_subset_info('validation')
        self._test_info = load_subset_info('test')

        self._parse_taxonomy()
        self._validation_dict, self._validation_inst_dict = construct_video_dict(self._validation_info[0], self._validation_info[1],
                                                     'validation', self._name_idx_table)
        self._test_dict, self._test_inst_dict = construct_video_dict(self._test_info[0], self._test_info[1],
                                                     'test', self._name_idx_table)
        self._video_dict = dict(list(self._validation_dict.items()) + list(self._test_dict.items()))

    def get_subset_videos(self, subset_name):
        if subset_name == 'validation':
            return self._validation_dict.values()
        elif subset_name == 'test':
            return self._test_dict.values()
        else:
            raise ValueError("Unknown subset {}".format(subset_name))

    def get_subset_instance(self, subset_name):
        if subset_name == 'test':
            return self._test_inst_dict.values()
        elif subset_name == 'validation':
            return self._validation_inst_dict.values()
        else:
            raise ValueError("Unknown subset {}".format(subset_name))

    def get_ordered_label_list(self):
        return [self._idx_name_table[x] for x in sorted(self._idx_name_table.keys())]

    def _parse_taxonomy(self):
        """
        This function just parse the taxonomy file
        It gives alphabetical ordered indices to the classes in competition
        :return:
        """
        validation_names = sorted([os.path.split(x)[1].split('_')[0] for x in self._validation_info[-1]])
        test_names = sorted([os.path.split(x)[1].split('_')[0] for x in self._test_info[-1]])

        if len(validation_names) != len(test_names):
            raise IOError('Validation set and test have different number of classes: {} v.s. {}'.format(
                len(validation_names), len(test_names)))

        final_names = []
        for i in range(len(validation_names)):
            if validation_names[i] != test_names[i]:
                raise IOError('Validation set and test have different class names: {} v.s. {}'.format(
                    validation_names[i], test_names[i]))

            if validation_names[i] not in self.ignore_labels:
                final_names.append(validation_names[i])

        sorted_names = sorted(final_names)

        self._idx_name_table = {i: e for i, e in enumerate(sorted_names)}
        self._name_idx_table = {e: i for i, e in enumerate(sorted_names)}
        print("Got {} classes for the year {}".format(len(self._idx_name_table), self.year))

    def try_load_file_path(self, frame_path):
        """
        Simple version of path finding
        :return:
        """
        import glob
        import os
        folders = glob.glob(os.path.join(frame_path, '*'))
        ids = [os.path.split(name)[-1] for name in folders]

        folder_dict = dict(zip(ids, folders))

        cnt = 0
        for k in self._video_dict.keys():
            if k in folder_dict:
                self._video_dict[k].path = folder_dict[k]
                cnt += 1
        print("loaded {} video folders".format(cnt))


if __name__ == '__main__':
    db = THUMOSDB.get_db()
    db.try_load_file_path('/mnt/SSD/THUMOS14/THUMOS14_extracted/')
