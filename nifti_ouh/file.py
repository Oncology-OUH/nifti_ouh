from datetime import datetime

import numpy as np
from pathlib import Path
import nibabel as nib
from matplotlib import cm
from rt_utils import RTStructBuilder

from nifti_ouh.enums import NiftiType
import logging

import os.path

log = logging.getLogger(__name__)


class File:
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        if not path.exists():
            log.error(f"The path {path} does not exist")

        self.img = None
        self.name = self.path.name.replace(".nii.gz", "")
        self.type = None
        self.data = None

    def load_header(self):
        log.info(f"Loading file {self.path}")
        self.img = nib.load(self.path)

        if self.img.header["bitpix"] == 8:
            self.type = NiftiType.MASKS
        if self.img.header["bitpix"] == 16:
            self.type = NiftiType.IMAGE

    def load_data(self):
        log.info(f"Loading data for image {self.name}")
        self.data = self.img.get_fdata()

    def map_values(self, max_value_in_mask: int, mapping: dict[int, int]):
        """
        Re-maps Nifti Mask values by adding the maximum value in the mask to the data and then replacing old values with new ones.
        Adding the max value to the original data is done to avoid confusion when replacing a value that already has newly assigned values.

        Example of why we add max_value_in_mask before mapping:
        [1,2,3] -> map 2 to 3, then map 3 to 1
        Desired output: [1,3,1]
        Actual output: [1,1,1]

        :param max_value_in_mask: The maximum value in the mask to be added to the data. This can be an arbitrary number. It should be higher than the highest value in the original data.
        :type max_value_in_mask: int
        :param mapping: A dictionary where the keys are the old values to be replaced and the values are the new values.
        :type mapping: dict
        :returns: None
        """
        log.info("Re-mapping Nifti Mask values")

        self.data += max_value_in_mask
        for old_value, new_value in mapping.items():
            old_value += max_value_in_mask
            self.data = np.where(self.data == old_value, new_value, self.data)

    def convert_masks_to_rtstruct(
        self,
        mask_info: dict,
        dicom_series_path: Path,
        output_path: Path,
        color_map=cm.get_cmap("rainbow"),
        series_description: str = "",
    ):
        if not os.path.isfile(str(output_path)):
            rtstruct = RTStructBuilder.create_new(dicom_series_path=str(dicom_series_path))
        else:
            rtstruct = RTStructBuilder.create_from(
                dicom_series_path=str(dicom_series_path),
                rt_struct_path=str(output_path))

        for i, (mask_name, info) in enumerate(mask_info.items()):
            if "display_name" in info:
                name = info["display_name"]
            else:
                name = mask_name

            log.info(f"Adding struct {name}")

            if "color" in info:
                color = info["color"]
            else:
                # Use a hash of the name to get the color from the supplied color map
                color = color_map(hash(mask_name) % 256)
                color = color[:3]
                color = [int(c * 255) for c in color]

            if "value" in info:
                value = int(info["value"])
            else:
                value = i

            rtstruct.add_roi(
                mask=np.where(self.data == value, 1, 0)
                .astype(dtype=bool)
                .transpose((1, 0, 2)),
                color=color,
                name=name,
                approximate_contours=False,
            )

        # Next four lines hack to compensate for "bug" in rt_utils
        for i in range(len(rtstruct.ds["ReferencedFrameOfReferenceSequence"].value)):
            rtstruct.ds["ReferencedFrameOfReferenceSequence"].value[i][
                "FrameOfReferenceUID"
            ].value = rtstruct.series_data[0]["FrameOfReferenceUID"].value
        for i in range(len(rtstruct.ds["StructureSetROISequence"].value)):
            rtstruct.ds["StructureSetROISequence"][i][
                "ReferencedFrameOfReferenceUID"
            ].value = rtstruct.series_data[0]["FrameOfReferenceUID"].value

        if series_description != "":
            rtstruct.ds["SeriesDescription"].value = series_description

        rtstruct.ds["SeriesDate"].value = datetime.now().strftime("%Y%m%d")
        rtstruct.ds["SeriesTime"].value = datetime.now().strftime("%H%M%S")

        rtstruct.save(str(output_path))

        return
