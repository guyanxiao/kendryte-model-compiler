# coding=utf-8
'''
 * Copyright 2018 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 '''

import os

def load_model(dataset_val, range_from_batch, args):
    import tools
    pb_loader = tools.import_from_path(os.path.dirname(__file__) + '/../pb')
    from . import h5_converter
    if not args.h5_path.endswith('.h5'):
        raise ValueError('{} should endwith *.h5'.format(args.h5_path))

    custom_objects = None
    if args.h5_custom_objects is not None:
        custom_objects_file = args.h5_custom_objects
        custom_objects_module = tools.import_from_path(custom_objects_file)
        custom_objects = custom_objects_module.get_custom_objects()

    args.pb_path = h5_converter.convert(args.h5_path, custom_objects)

    return pb_loader.load_model(dataset_val, range_from_batch, args)

