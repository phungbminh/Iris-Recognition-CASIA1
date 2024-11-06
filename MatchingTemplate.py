
from os import listdir
from itertools import repeat
from fnmatch import filter
import numpy as np
import scipy.io as sio
import warnings
warnings.filterwarnings("ignore")


class MatchingTemplate:
    @staticmethod
    def HammingDistance(template1, mask1, template2, mask2):
        """
        Calculate the Hamming distance between two iris templates.
        """
        hd = np.nan

        # Shifting template left and right, use the lowest Hamming distance
        for shifts in range(-8, 9):
            template1s = MatchingTemplate.shiftbits_ham(template1, shifts)
            mask1s = MatchingTemplate.shiftbits_ham(mask1, shifts)

            mask = np.logical_or(mask1s, mask2)
            nummaskbits = np.sum(mask == 1)
            totalbits = template1s.size - nummaskbits

            C = np.logical_xor(template1s, template2)
            C = np.logical_and(C, np.logical_not(mask))
            bitsdiff = np.sum(C == 1)

            if totalbits == 0:
                hd = np.nan
            else:
                hd1 = bitsdiff / totalbits
                if hd1 < hd or np.isnan(hd):
                    hd = hd1
        return hd

    @staticmethod
    def shiftbits_ham(template, noshifts):
        """
        Shift the bit-wise iris patterns.
        """
        templatenew = np.zeros(template.shape)
        width = template.shape[1]
        s = 2 * np.abs(noshifts)
        p = width - s

        if noshifts == 0:
            templatenew = template
        elif noshifts < 0:
            x = np.arange(p)
            templatenew[:, x] = template[:, s + x]
            x = np.arange(p, width)
            templatenew[:, x] = template[:, x - p]
        else:
            x = np.arange(s, width)
            templatenew[:, x] = template[:, x - s]
            x = np.arange(s)
            templatenew[:, x] = template[:, p + x]

        return templatenew

    @staticmethod
    def matchingPool(file_temp_name, template_extr, mask_extr, template_dir):
        """
        Perform matching session within a Pool of parallel computation
        """

        data_template = sio.loadmat(f'{template_dir}{file_temp_name}')
        template = data_template['template']
        mask = data_template['mask']

        # The Hamming distance
        hm_dist = MatchingTemplate.HammingDistance(template_extr, mask_extr, template, mask)
        return (file_temp_name, hm_dist)

    @staticmethod
    def matchingTemplate(template_extr, mask_extr, template_dir, threshold=0.38):
        """
        Matching the template of the image with the ones in the database
        """
        # n# of accounts in the database
        n_files = len(filter(listdir(template_dir), '*.mat'))
        if n_files == 0:
            return -1

            # Prepare arguments
        args = zip(
            sorted(listdir(template_dir)),
            repeat(template_extr),
            repeat(mask_extr),
            repeat(template_dir),
        )
        total_args = len(list(args))

        args = zip(
            sorted(listdir(template_dir)),
            repeat(template_extr),
            repeat(mask_extr),
            repeat(template_dir),
        )

        result_list = []
        # Lặp qua từng tập hợp arg và gọi hàm matchingPool
        for i, arg in enumerate(args):
            try:
                result = MatchingTemplate.matchingPool(*arg)
                result_list.append(result)
            except Exception as e:
                print(f'Error occurred while processing {arg}: {e}')
                result_list.append((None, None))
            percent_complete = (i + 1) / total_args * 100

            # Tách tên tệp và khoảng cách Hamming
        filenames = [result_list[i][0] for i in range(len(result_list)) if result_list[i][0] is not None]
        hm_dists = np.array([result_list[i][1] for i in range(len(result_list)) if result_list[i][1] is not None])
        # Removal of NaN elements
        # ind_valid = np.where(hm_dists > 0)[0]
        # hm_dists = hm_dists[ind_valid]
        # filenames = [filenames[idx] for idx in ind_valid]

        ind_thres = np.where(hm_dists <= threshold)[0]
        if len(ind_thres) == 0:
            return 0
        else:
            hm_dists = hm_dists[ind_thres]
            filenames = [filenames[idx] for idx in ind_thres]
            ind_sort = np.argsort(hm_dists)
            return [filenames[idx] for idx in ind_sort]