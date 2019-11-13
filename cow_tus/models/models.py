from collections import Counter

import torch
from torch import nn
import networkx as nx

import cow_tus.models.modules.zoo as module_zoo

class Model(nn.Module):

    def __init__(self, modules, module_defaults, load_paths=None, devices=[0]):
        """
        """
        super().__init__()

        self.devices = devices

        self.modules, self.module_configs = self._init_modules(modules,
                                                               module_defaults)
        self.module_task_heads = [module['name'] for module in self.module_configs if '_loss' in module['dsts']]
        self.src_uses = Counter([src for module in self.module_configs for src in module['srcs']])

        if load_paths != None:
            for load_path in load_paths:
                self.load_weights(load_paths)

    def _init_modules(self, module_configs, module_defaults):
        """
        TODO: verify that modules are doubly linked
        """
        g = nx.DiGraph()
        for module_config in module_configs:
            module_name = module_config['name']
            module_dsts = module_config['dsts']
            g.add_edges_from([(module_name, dst) for dst in module_dsts])
        module_names_sorted = list(nx.topological_sort(g))

        module_config_dict = {}
        module_dict = nn.ModuleDict()
        for i, module_config in enumerate(module_configs):
            name = module_config['name']
            module_config_dict[name] = module_config

            class_name = module_config['class_name']
            args = module_config.get('args', module_defaults[class_name])
            module = getattr(module_zoo, class_name)(**args)
            module_dict[name] = module

        module_config_list = []
        module_list = nn.ModuleList()
        for module_name in module_names_sorted:
            if module_name == '_loss':
                continue
            module_config_list.append(module_config_dict[module_name])
            module_list.append(module_dict[module_name])

        return module_list, module_config_list

    def forward(self, X, targets=None):
        """
        data takes the form of the following:
        {
            src: string describing data source
            logits: the BATCHED primary output of the module
            custom: anything else the model might return. should be batched as well
        }
        """
        src_uses = dict(self.src_uses)
        databank = {x['src']: x for x in X}
        for i, module in enumerate(self.modules):
            module_config = self.module_configs[i]

            data = {src: databank[src] for src in module_config['srcs']}  # data := {'_raw.loops': Data(), '_raw.report': Data()}
            data = module(data)                                           # data := {'shared': Data()}
            databank[module_config['name']] = data

            for src in module_config['srcs']:
                src_uses[src] -= 1
                if src_uses[src] == 0:
                    del databank[src]

        output = {}
        for module in self.module_task_heads:
            output[module_config['name']] = databank[module_config['name']]
        return output

    def load_weights(self, load_path, device=None):
        """
        args:
            src_path (str) path to the weights file
            inclusion_res (list(str) or str) list of regex patterns or one regex pattern.
                    If not None, only loads weights that match at least one of the regex patterns.
            substitution_res (list(tuple(str, str))) list of tuples like
                    (regex_pattern, replacement). re.sub is called on each key in the dict
        """
        src_state_dict = torch.load(src_path, map_location=torch.device(device))
        self.load_state_dict(src_state_dict, strict=False)

        n_loaded_params = len(set(self.state_dict().keys()) & set(src_state_dict.keys()))
        n_tot_params = len(src_state_dict.keys())
        if n_loaded_params < n_tot_params:
            self.log.info("Could not load these parameters due to name mismatch: " +
                         f"{set(src_state_dict.keys()) - set(self.state_dict().keys())}")
        else:
            self.log.info(f"Loaded {n_loaded_params}/{n_tot_params} pretrained parameters" +
                          f"from {src_path} matching '{inclusion_res}'.")

    def save_weights(self, save_path, link_path=None):
        """
        args:
            save_path (str)   path where to save weights
            link_path (str, optional)   path where to add symlink
        """
        torch.save(self.state_dict(), save_path)
        if link_path != None and not os.path.islink(link_path):
            os.symlink(save_path, link_path)