import json
from typing import List, Union
import numpy as np


class BaseReport:
    """The Abstract base class for all reports
    """

    def __init__(self, imgstack_files: str, coord_infos: str):

        self.imgstack = self._read_imgstack(imgstack_files)

        if isinstance(coord_infos, list):
            merged_json = dict()
            for coord_info in coord_infos:
                a_file = open(coord_info, "r")
                _contents = json.load(a_file)

                for k, v in _contents.items():
                    # print(v)
                    if not (k in merged_json):
                        # if the key isnt in the merged dictionary, then add and initialise it. This is python 3.6+ behaviour
                        merged_json[k] = v
                        continue

                    if not isinstance(merged_json[k], list) or isinstance(
                        merged_json[k], str
                    ):
                        # If there is unique data and the value at k in merged dict is not a list already, make it one
                        merged_json[k] = [merged_json[k]]
                    _test_dict = merged_json[k].copy()
                    _base_set = set(_test_dict)

                    if isinstance(v, list):
                        _test_dict.extend(v)
                    else:
                        _test_dict.append(v)

                    _test_set = set(_test_dict)
                    # print(_test_set)
                    # print(_base_set)
                    if len(_test_set) == len(_base_set):
                        # If there is nothing unique when making the set of the new content at that key and the old content, then skip
                        continue
                    if isinstance(v, list):
                        merged_json[k].extend(v)
                    else:
                        merged_json[k].append(v)

            self.coords = merged_json
            # print(self.coords)
        else:
            a_file = open(coord_infos, "r")
            self.coords = json.load(a_file)

        self.pdf = None

    # PDF controls--------------------------------------------
    def set_pdf(self, pdf) -> None:
        """Takes a PdfPages object and sets that as the internal variable 

        Args:
            pdf : the pdf pages object
        """
        self.pdf = pdf

    def get_pdf(self):
        """Return the pdfpages object

        Returns:
            _type_: _description_
        """
        return self.pdf

    def isPdf(self) -> bool:
        """Checks if the pdf has been set

        Returns:
            bool: _description_
        """
        if self.pdf is None:
            return False
        return True

    def closePdf(self) -> None:
        """Closes the pdf
        """
        if self.isPdf():
            self.pdf.close()
            self.pdf = None

    def _read_imgstack(self, imgstack_files: Union[List[str], str]):
        """Read in an image stack created by utils.fileIO function

        Args:
            imgstack_files (str): The 

        Returns:
            _type_: Loaded Image stack
        """
        if isinstance(imgstack_files, list):
            # If there is a list of images, then extend the image stack along the last dimension
            _imgstack = np.array(
                [np.load(image_stack) for image_stack in imgstack_files]
            )
            imgstack = np.stack(_imgstack, axis=-1)
        else:
            imgstack = np.load(imgstack_files)
        return imgstack
