import torch
from loaders_v2 import YCB, Laval, Backend
import random
import numpy as np

class GenericDataset():

    def __init__(self, cfg_d, cfg_env):

        if cfg_d['name'] == "ycb":
            self._backend = self._backend = YCB(cfg_d=cfg_d,
                                                cfg_env=cfg_env)
        elif cfg_d['name'] == "laval":
            self._backend = Laval(
                cfg_d=cfg_d,
                cfg_env=cfg_env)

        self._obj_list_sym = cfg_d['obj_list_sym']
        self._obj_list_fil = cfg_d['obj_list_fil']
        self._batch_list = self._backend._batch_list
        self._force_one_object_visible = cfg_d['output_cfg'].get('force_one_object_visible', False)

        batch_list_tmp = []
        if self._backend._dataset_config.get('all_objects', False):
            if self._backend._dataset_config['batch_list_cfg']['seq_length'] > 1:
                raise Exception('If all_objects is TRUE then batch_list_cfg/seq_length must be 1.')
            for i in self._batch_list:
                entry = (str(i[1])+i[2][0])
                batch_list_tmp.append(entry)
            batch_list_uniq, indexes = np.unique(np.array(batch_list_tmp), return_index=True)
            self._backend._batch_list = np.array(self._batch_list)[indexes,:].tolist()
            self._batch_list = self._backend._batch_list

        if self._obj_list_fil is not None:
            self._batch_list = [
                x for x in self._batch_list if x[0] in self._obj_list_fil]
        self._length = len(self._batch_list)

    def __len__(self):
        return self._length

    def __str__(self):
        string = "Generic Dataloader of length %d" % len(self)
        string += "\n Backbone is set to %s" % self._backend
        return string

    @property
    def visu(self):
        return self._backend.visu

    @visu.setter
    def visu(self, vis):
        self._backend.visu = vis

    @property
    def sym_list(self):
        return self._obj_list_sym

    @property
    def refine(self):
        return self._backend.refine

    @refine.setter
    def refine(self, refine):
        self._backend.refine = refine

    @property
    def seq_length(self):
        return len(self._batch_list[0][2])

    def object_models(self):
        """
        Returns a tensor containing a set of points on the object model.
        Notice that indices will be off-by-one compared to the semantic segmentation labels as
        zero for the semantic label is the background and doesn't have a model.
        returns: N x P x 3
        """
        point_count = self.get_num_points_mesh()
        object_models = self._backend.object_models
        object_models = [object_models[i] for i in range(1, len(object_models) + 1)]
        points = torch.zeros(len(object_models), point_count, 3, dtype=torch.float32)
        for i in range(0, len(object_models)):
            model = torch.tensor(object_models[0])
            indices = np.random.choice(np.arange(model.shape[0]), point_count, replace=False)
            points[i, :, :] = model[indices, :]
        return points

    def keypoints(self):
        """
        Gives the ground truth object keypoints in the mesh coordinate frame.
        return: M x K x 3
        """
        keypoints = self._backend.keypoints
        out = torch.zeros((len(keypoints), 8, 3))
        for i in range(out.shape[0]):
            object_keypoints = keypoints[i + 1]
            out[i, :, :] = torch.tensor(object_keypoints)
        return out

    def get_num_points_mesh(self, refine=False):
        # onlt implemented for backwards compatability. Refactor this
        if refine == False:
            return self._backend._num_pt_mesh_small
        else:
            return self._backend._num_pt_mesh_large

    def __getitem__(self, index):
        seq = []
        one_object_visible = False
        if self._backend._dataset_config.get('all_objects', False):
            # get the full image if specified for certain traiing tasks
            k = self._batch_list[index][2][0]
            num = '0' * int(6 - len(str(k))) + str(k)
            seq.append(self._backend.getFullImage(desig=f'{self._batch_list[index][1]}/{num}'))
        else:
            # iterate over a sequence specified in the batch list
            for k in self._batch_list[index][2]:
                # each batch_list entry has the form [obj_name, obj_full_path, index_list]
                num = '0' * int(6 - len(str(k))) + str(k)
                seq.append(self._backend.getElement(
                    desig=f'{self._batch_list[index][1]}/{num}', obj_idx=self._batch_list[index][0]))
                if not isinstance(seq[-1][0],bool):
                    one_object_visible = True

            if self._force_one_object_visible and one_object_visible == False:
                rand = random.randrange(0, len(self))
                return self[int(rand)]

        return seq
