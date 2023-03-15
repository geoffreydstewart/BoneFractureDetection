#!/usr/bin/env python3

#
# Copyright (c) 2023 Oracle and/or its affiliates. All rights reserved.
#

"""
This is sets up the humerus x-rays from the MURA dataset for use in fine-tuning a pre-trained model.

https://stanfordmlgroup.github.io/competitions/mura/
"""

import os

IMAGE_FILES = ['/Users/gstewart/temp/tmpML/datasets/MURA-v1.1/train_image_paths.csv',
               '/Users/gstewart/temp/tmpML/datasets/MURA-v1.1/valid_image_paths.csv']

COPY_SCRIPT = 'copy_script.sh'


def main():
    os.remove(COPY_SCRIPT)

    img_cnt = 0
    for image_file in IMAGE_FILES:
        with open(image_file, 'r') as file_to_read, open(COPY_SCRIPT, 'a') as file_to_write:
            data = file_to_read.readlines()

            for filepath in data:
                line = filepath.strip()
                if 'XR_HUMERUS' not in line:
                    continue
                else:
                    if 'positive' in line:
                        cmd = 'cp /Users/gstewart/temp/tmpML/datasets/' + line + ' /Users/gstewart/temp/tmpML/datasets/MURA-Humerus/Positive/image' + str(img_cnt) + '.png'
                    else:
                        cmd = 'cp /Users/gstewart/temp/tmpML/datasets/' + line + ' /Users/gstewart/temp/tmpML/datasets/MURA-Humerus/Negative/image' + str(img_cnt) + '.png'
                    # print(cmd)
                    file_to_write.write('%s\n' % cmd)
                    img_cnt += 1


if __name__ == "__main__":
    main()
