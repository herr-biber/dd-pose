import json
import re
import os
from collections import defaultdict

class Dataset:
    def __init__(self, split='all', load_defaults=True):
        assert split in ('all', 'trainval', 'test')
        self.data = defaultdict(lambda: defaultdict(list))
        self.split = split
        
        if load_defaults:
            self.load(self.__get_default_file__(split))
    
    @staticmethod
    def __get_default_file__(split='all'):
        return os.path.join(os.environ.get('DD_POSE_DIR'), 'resources', 'dataset-items-%s.txt' % split)
    
    def load(self, dataset_file):
        with open(dataset_file) as fp:
            lines = fp.readlines()
        for line in lines:
            subject, scenario, humanhash, split = line.strip().split(' ')
            subject = int(subject)
            scenario = int(scenario)
            self.data['subject-%02d' % subject]['scenario-%02d' % scenario].append({
                'subject': subject,
                'scenario': scenario,
                'humanhash': humanhash,
                'split': split,
                'is-test': split == 'test'
            })
               
    def get(self, subject_id, scenario_id=None, humanhash=None):
        subject_dict = self.data['subject-%02d' % subject_id]
        if scenario_id is None:
            return subject_dict
        dataset_items = subject_dict['scenario-%02d' % scenario_id]
        if humanhash is None:
            return dataset_items
        for dataset_item in dataset_items:
            if dataset_item['humanhash'] == humanhash:
                return dataset_item
            
        return None
    
    def __len__(self):
        counter = 0
        for scenario_dict in self.data.values():
            for dataset_items in scenario_dict.values():
                counter += len(dataset_items)
        return counter
    
    def get_dataset_items(self):
        for _, scenario_dict in sorted(self.data.items()):
            for _, dataset_items in sorted(scenario_dict.items()):
                for dataset_item in dataset_items:
                    yield dataset_item
                    
    def get_subject_scenario_dataset_items(self):
        for di in self.get_dataset_items():
            yield (di['subject'], di['scenario'], di)
            
    @staticmethod
    def get_dataset_item_str(dataset_item):
        return "subject-%02d-scenario-%02d-%s" % (dataset_item['subject'], dataset_item['scenario'], dataset_item['humanhash'])
    
    @staticmethod
    def parse_subject_scenario_humanhash_string(s):
        m = re.match('subject-(\d\d)-scenario-(\d\d)-(.+)', s)
        if m is None:
            return None
        
        subject, scenario, humanhash = m.groups()
        subject = int(subject)
        scenario = int(scenario)
        return (subject, scenario, humanhash)