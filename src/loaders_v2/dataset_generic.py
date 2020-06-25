from loaders_v2 import YCB, Laval, Backend
import random

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

    def get_num_points_mesh(self, refine=False):
        # onlt implemented for backwards compatability. Refactor this
        if refine == False:
            return self._backend._num_pt_mesh_small
        else:
            return self._backend._num_pt_mesh_large

    def __getitem__(self, index):
        seq = []
        one_object_visible = False
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
