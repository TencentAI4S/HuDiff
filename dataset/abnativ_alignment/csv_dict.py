# (c) 2023 Sormannilab=

import copy
import csv
import os
import sys
from collections import OrderedDict

import numpy
import scipy.stats

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from . import mybio

try:  # custom module, not necessarily needed (only in some functions)
    from . import misc  # see if it is in same folder
except Exception:
    try:  # see if it's in pythonpath
        import misc
    except Exception:

        class tmpmisc:
            def find_nearest(self, *args, **kwargs):
                raise Exception("misc MODULE NOT AVAILABLE.(find_nearest)\n")

            def get_file_path_and_extension(self, *args, **kwargs):
                raise Exception(
                    "misc MODULE NOT AVAILABLE. (get_file_path_and_extension)\n"
                )

            def GetBin(self, *args, **kwargs):
                raise Exception("misc MODULE NOT AVAILABLE. (GetBin)\n")

            def cluster_with_norm(self, *args, **kwargs):
                raise Exception("misc MODULE NOT AVAILABLE. (cluster_with_norm)\n")

            def uniq(self, *args, **kwargs):
                raise Exception("misc MODULE NOT AVAILABLE. (uniq)\n")

            def loose_compare(self, *args, **kwargs):
                raise Exception("misc MODULE NOT AVAILABLE. (loose_compare)\n")

        misc = tmpmisc()


try:  # custom module, not necessarily needed (only in some functions)
    from . import structs  # see if it is in same folder
except Exception:
    try:  # see if it's in pythonpath
        import structs
    except Exception:

        class tmpstructs:
            def Stats(self, *args, **kwargs):
                raise Exception("structs MODULE NOT AVAILABLE\n")

        structs = tmpstructs()


try:
    from . import plotter
except Exception:
    try:
        import plotter
    except Exception:

        class tmpplotter:
            def profile(self, *args, **kwargs):
                raise Exception("PLOTTER MODULE NOT AVAILABLE\n")

            def plot_seq_profile(self, *args, **kwargs):
                raise Exception("PLOTTER MODULE NOT AVAILABLE\n")

            def histogram(self, *args, **kwargs):
                raise Exception("PLOTTER MODULE NOT AVAILABLE\n")

            def boxplot(self, *args, **kwargs):
                raise Exception("PLOTTER MODULE NOT AVAILABLE\n")

            def cloudplot(self, *args, **kwargs):
                raise Exception("PLOTTER MODULE NOT AVAILABLE\n")

            def scatter(self, *args, **kwargs):
                raise Exception("PLOTTER MODULE NOT AVAILABLE\n")

            def histogram2d(self, *args, **kwargs):
                raise Exception("PLOTTER MODULE NOT AVAILABLE\n")

            def apply_scale_function(self, *args, **kwargs):
                raise Exception("PLOTTER MODULE NOT AVAILABLE\n")

            def swarmplot(self, *args, **kwargs):
                raise Exception("PLOTTER MODULE NOT AVAILABLE\n")

        plotter = tmpplotter()


# define a dictionary class with extra methods to contain the data
# An OrderedDict is a dict that remembers the order that keys were first inserted.
# If a new entry overwrites an existing entry, the original insertion position is left unchanged.
# Deleting an entry and reinserting it will move it to the end.
class Data(OrderedDict):
    hd = {}  # header
    filename = ""  # database file name

    def __init__(self, *argv, **argd):
        fname = None  # aux variable
        if len(argv) > 0 and type(argv[0]) is str and os.path.isfile(argv[0]):
            fname = argv[0]
            argv = argv[1:]
        if "fname" in argd:
            fname = argd["fname"]
            del argd["fname"]
        copy = None
        if "copy" in argd:
            copy = argd["copy"]
            del argd["copy"]
        self.hd = {}
        self.filename = None  # name of the file from which class is read
        self.key_column = 1
        self.key_column_hd_name = "keys"
        self.allow_more_entries_per_key = False
        self.mat = None  # for files containing only numbers by calling numpymat() this becomes a numpy matrix with the file content - not working for allow_more_entries_per_key
        self.keys_in_mat = False
        if copy is not None:
            super(Data, self).__init__(*argv, **argd)
            self.copy(other_class=copy, deep=False)  # copy content from other class
        elif fname is not None:
            # print fname
            super(Data, self).__init__(*argv)
            self.load(fname, verbose=False, **argd)
        else:
            super(Data, self).__init__(*argv, **argd)  # this inits the OrderedDict

    def __repr__(self):
        return "<Data class with %d entries from file %s>" % (
            len(self),
            str(self.filename),
        )

    def __str__(self):
        return "<Data class with %d entries from file %s>" % (
            len(self),
            str(self.filename),
        )

    # give HEADER=None for files with no header
    # note that the option allow_more_entries_per_key makes a mess! (you need to give an extra [0] before calling the header at every time!!)
    def load(
        self,
        filename,
        DELIMITER="\t",
        key_column=1,
        allow_more_entries_per_key=False,
        double_header=False,
        double_header_merger=" ",
        HEADER={},
        convert_key_to_number=False,
        auto_convert_to_number=True,
        auto_convert_profile_delimiter=";",
        skip_begins=["#", "@", "\n"],
        merge_longer=False,
        overwrite_warning=True,
        verbose=True,
    ):
        """
        give key_column=None if you would rather have the raw number as keys in the Data class.
        give HEADER=None for files with no header
         note that the option allow_more_entries_per_key makes a mess! (you need to give an extra [0] before calling the header at every time!!)
        skip_begins does not apply to HEADER line if HEADER is not None
        double_header True indicates a double header that can be merged with double_header_merger,
         can also be an integer for many header lines, in which case for
         e.g. 3 lines of header it should be double_header=2 (indicating that 2 are extra)
        """
        self.allow_more_entries_per_key = allow_more_entries_per_key
        self.key_column = key_column
        self.filename = filename
        gzip = False
        if ".gz" in filename or ".zip" in filename:
            print("unzipping %s ..." % (filename))
            gzip = True
            if filename[-3:] == ".gz":
                os.system("gunzip %s" % (filename))
                filename = filename[:-3]
            elif filename[-4:] == ".zip":
                os.system("unzip %s" % (filename))
                filename = filename[:-4]
        if ".csv" in filename:
            DELIMITER = None
        if HEADER is not None:
            tmp_d, HD, self.key_column_hd_name = csvToDictionary(
                filename,
                key_column=key_column,
                allow_more_entries_per_key=allow_more_entries_per_key,
                DELIMITER=DELIMITER,
                HEADER=HEADER,
                double_header=double_header,
                double_header_merger=double_header_merger,
                convert_key_to_number=convert_key_to_number,
                skip_begins=skip_begins,
                overwrite_warning=overwrite_warning,
                auto_convert_to_number=auto_convert_to_number,
                auto_convert_profile_delimiter=auto_convert_profile_delimiter,
                merge_longer=merge_longer,
                get_only_column_id=None,
            )
        else:
            tmp_d = csvToDictionary(
                filename,
                key_column=key_column,
                allow_more_entries_per_key=allow_more_entries_per_key,
                DELIMITER=DELIMITER,
                HEADER=HEADER,
                double_header=double_header,
                double_header_merger=double_header_merger,
                convert_key_to_number=convert_key_to_number,
                auto_convert_to_number=auto_convert_to_number,
                skip_begins=skip_begins,
                overwrite_warning=overwrite_warning,
                auto_convert_profile_delimiter=auto_convert_profile_delimiter,
                merge_longer=merge_longer,
                get_only_column_id=None,
            )
        if verbose:
            sys.stdout.write(
                "Loaded %d data from %s, with header %s, keys: %s\n"
                % (
                    len(tmp_d),
                    filename.split("/")[-1],
                    str(tmp_d.HD()),
                    str(self.key_column_hd_name),
                )
            )
        self._update(tmp_d)
        del tmp_d
        if gzip:
            os.system("gzip %s" % (filename))
        return

    def numpymat(self, include_keys=False, return_mat=False):
        """
        saves in self.mat the numpy matrix corresponding to the whole file (no hd)
        if include_keys keys will be first column thus ones need to manually add +1 to all hd calls
        """
        if include_keys:
            self.mat = numpy.array([[float(k)] + self[k] for k in self])
            self.keys_in_mat = True
        else:
            self.mat = numpy.array(
                [
                    [
                        a
                        if a is not None
                        and (not isinstance(a, numpy.ndarray) and a != "")
                        else numpy.nan
                        for a in self[k]
                    ]
                    for k in self
                ]
            )
            self.keys_in_mat = False
        if return_mat:
            return self.mat
        return

    def from_numpymat(self, input_mat=None, keys_in_mat=None):
        """
        it is assumed that self.numpymat has been called and that self.mat has been modified by some operations.
        This functions then changes the contents of the Data class to the new values of self.mat
        Otherwise input_mat can be given to use a completely different mat
        """
        if input_mat is None:
            input_mat = self.mat
        if input_mat is None:
            sys.stderr.write(
                "*ERROR* in from_numpymat() cannot perform from_numpymat as self.mat is None\n"
            )
            return
        if keys_in_mat is None:
            keys_in_mat = self.keys_in_mat
        if len(self) == 0 and len(input_mat) != 0:
            if keys_in_mat:
                for kl in input_mat:
                    self[kl[0]] = list(kl[1:])
                return
            sys.stderr.write(
                "*ERROR* in from_numpymat() no keys in self -> CANNOT CONVERT! self.mat has shape %s\n"
                % (str(self.mat.shape))
            )
        for j, k in enumerate(self):
            if keys_in_mat:
                k2 = input_mat[j][0]
                vals = list(input_mat[j][1:])
                del self[k]
                self[k2] = vals
            else:
                self[k] = list(input_mat[j])[:]
        return

    def column_operation(self, column_index, function, add_as_column_key=None):
        """
        returns the results of the call of function on input column
        if more than one input column is given all columns specified are used as input 
         to the function in the form function(*args)
        if add_as_column_key is not None and not False than a new column with the results
         is added to the class
        """
        if self.mat is None:
            self.numpymat(include_keys=self.keys_in_mat, return_mat=False)
        if hasattr(column_index, "__len__") and type(column_index) is not str:
            column_index = numpy.array(
                [c if type(c) is int else self.hd[c] for c in column_index]
            )
        elif type(column_index) is not int:
            column_index = self.hd[column_index]
        # print 'column_index',column_index
        if hasattr(column_index, "__len__"):
            results = function(*self.mat.T[column_index])
        else:
            results = function(self.mat.T[column_index])
        if add_as_column_key is not None and add_as_column_key != False:
            self.add_column(results, add_as_column_key)
        return results

    def copy(self, other_class=None, copy_keys=False, deep=False):
        """
        if other_class is not None it copies that class into this class, otherwise it returns a copy of this class
        if deep is not True it does not copy the items
        if copy_keys (when deep is False) it will copy the keys  leaving as values empty lists (useful when doing mat operation to fill with from_numpymat)
        """
        if other_class is not None:
            if deep:
                for k in other_class:
                    self[k] = copy.deepcopy(other_class[k])
            elif copy_keys:
                for k in other_class:
                    self[k] = []
            self.allow_more_entries_per_key = other_class.allow_more_entries_per_key
            self.hd = copy.deepcopy(other_class.hd)
            self.key_column = other_class.key_column
            self.key_column_hd_name = other_class.key_column_hd_name
            self.filename = other_class.filename
            if "keys_in_mat" in dir(other_class):
                self.keys_in_mat = self.keys_in_mat
        else:
            other_class = Data()
            if deep:
                for k in self:
                    other_class[k] = copy.deepcopy(self[k])
            elif copy_keys:
                for k in self:
                    other_class[k] = []
            other_class.allow_more_entries_per_key = self.allow_more_entries_per_key
            other_class.hd = copy.deepcopy(self.hd)
            other_class.key_column = self.key_column
            other_class.key_column_hd_name = self.key_column_hd_name
            other_class.filename = self.filename
            if "keys_in_mat" in dir(self):
                other_class.keys_in_mat = self.keys_in_mat
            return other_class

    def HD(self):
        """
        returns the hd as a list in the correct order
        """
        HD = ["?"] * (len(self.hd))
        for title in self.hd:
            if type(self.hd[title]) is int and self.hd[title] < len(self.hd):
                HD[self.hd[title]] = title
        return HD

    def keys_to_number(self, force_float=False):
        """
        converts keys (which by default are read as str) to number - int if possible if not flaot (unless force_float=True)
        """
        failed = 0
        for k in self:
            if type(k) is str:
                kn, ok = convert_to_number(k, force_float=force_float)
                if ok:
                    if kn in self:
                        sys.stderr.write(
                            "**WARNING** in keys_to_number() overwriting already existing %s\n"
                            % (str(kn))
                        )
                    self[kn] = self[k]
                    del self[k]
                else:
                    failed += 1
        if failed > 0:
            sys.stderr.write(
                "**WARNING** in () could not convert %d keys among %d\n"
                % (failed, len(self))
            )
        return

    def _update(self, other_data_class):
        self.update(other_data_class)
        self.hd.update(other_data_class.hd)
        return

    def transpose(self, key_column_hd_name="keys"):
        """
        returns a transpose of the class, the class itself is not changed
        """
        transp = Data()
        transp.hd = list_to_dictionary(list(self.keys()))
        transp.key_column_hd_name = key_column_hd_name
        for k in self.HD():
            transp[k] = self.column(k)
        # transp._equate_to_same_length() # control
        return transp

    def sort(self, column_index=None, reverse=False, key=lambda x: x):
        """
        Unlike the list sort function this function does not sort
        the dictionary, it returns a sorted copy
        to sort it do data=data.sort()
        column_index=None will sort it by keys
        otherwise it will be sorted by the entry in that column
        """
        if (
            column_index is None
            or (column_index == "keys" and column_index not in self.hd)
            or column_index == self.key_column_hd_name
        ):
            if "copy" in self:
                raise Exception(
                    "Cannot have 'copy' as keys in Data class - protected keyword."
                )
            tmp = Data(
                OrderedDict(
                    sorted(list(self.items()), key=lambda t: key(t[0]), reverse=reverse)
                ),
                copy=self,
            )
            return tmp
        else:
            if type(column_index) is str:
                column_index = self.hd[column_index]
            if "copy" in self:
                raise Exception(
                    "Cannot have 'copy' as keys in Data class - protected keyword."
                )
            tmp = Data(
                OrderedDict(
                    sorted(
                        list(self.items()),
                        key=lambda t: key(t[1][column_index]),
                        reverse=reverse,
                    )
                ),
                copy=self,
            )
            return tmp

    def sort_columns(self, row_key=None, key=None, reverse=False):
        """
        sorts the data class vertically by moving the columns around (generally useful for pretty-prints)
        Unlike the list sort function this function does not sort the data, it returns a sorted copy
        to sort according to the header in alphabetical order do data=data.sort_column()
         or you can give key=lambda x : some_function(x) where x should be keys of the header 
         to sort it accoring to the returned values
        or you can give row_key as input to sort the whole class according to one specific row of data
        """
        if row_key is not None:
            if reverse:
                m = -1.0
                argsorter = numpy.argsort(m * numpy.array(self[row_key]))
            else:
                argsorter = numpy.argsort(self[row_key])
            if key is not None:
                sys.stderr.write(
                    "\n**ERROR** in sort_columns key argument not supported by numpy.argsort, key has been ingored\n"
                )
        else:
            sorted_hd_list = sorted(self.HD(), reverse=reverse, key=key)
            argsorter = numpy.array([self.hd[k] for k in sorted_hd_list])
        new_hd = list_to_dictionary(numpy.array(self.HD())[argsorter])
        self.numpymat(include_keys=False)
        data2 = self.copy(copy_keys=True, deep=False)
        data2.numpymat(include_keys=False, return_mat=False)
        data2.hd = copy.deepcopy(new_hd)
        if len(self.mat) == 0:
            sys.stderr.write(
                "\n**WARNING** in sort_columns() Data class seems empty len(self.mat)==0 sorted only HD\n"
            )
            return data2
        data2.mat = numpy.copy(self.mat[:, argsorter])
        data2.from_numpymat(keys_in_mat=False)
        return data2

    # the column is added before index_of_insertion (if None it is added at the end), if str in header is added after that column)
    # it can be a list of the same length or a dictionary with (some of the) same keys
    def add_column(
        self,
        elements,
        variable_name="new_column",
        index_of_insertion=None,
        verbose=True,
    ):
        """
        the column is added before index_of_insertion (if None is added at the end, 
         if index_of_insertion is a string in header then it is added after that column)
        it can be a list of the same length or a dictionary with (some of the) same keys
        """
        if index_of_insertion is None:
            index_of_insertion = len(self.hd)
        elif type(index_of_insertion) is str:
            index_of_insertion = self.hd[index_of_insertion]
            index_of_insertion += 1
        mx = len(elements)
        c = 0  # added elements
        for key in self:
            if type(self[key]) is list or type(self[key]) is tuple:
                if (
                    self.allow_more_entries_per_key
                    and len(self[key]) > 0
                    and type(self[key][0]) is list
                ):
                    for row in self[key]:
                        if type(elements) is dict:
                            if key in elements:
                                row[index_of_insertion:index_of_insertion] = [
                                    elements[key]
                                ]
                            else:
                                row[index_of_insertion:index_of_insertion] = [""]
                        else:
                            if c < mx:
                                row[index_of_insertion:index_of_insertion] = [
                                    elements[c]
                                ]
                            else:
                                row[index_of_insertion:index_of_insertion] = [""]
                        c += 1
                    del row
                else:
                    if type(elements) is dict:
                        if key in elements:
                            self[key][index_of_insertion:index_of_insertion] = [
                                elements[key]
                            ]
                        else:
                            self[key][index_of_insertion:index_of_insertion] = [""]
                    else:
                        if c < mx:
                            self[key][index_of_insertion:index_of_insertion] = [
                                elements[c]
                            ]
                        else:
                            self[key][index_of_insertion:index_of_insertion] = [""]
                    c += 1
        self._update_hd(variable_name, index_of_insertion)
        if verbose and c == mx:
            sys.stdout.write(
                "add_column() added %d elements to %d rows at position %d with header name '%s'\n"
                % (c, mx, index_of_insertion, variable_name)
            )
        elif c > mx:
            sys.stderr.write(
                "**add_column() added %d elements of %d - REST LEFT EMPTY at position %d with header name '%s'\n"
                % (c, mx, index_of_insertion, variable_name)
            )
        elif c < mx:
            sys.stderr.write(
                "**add_column() added %d elements of %d - REST NOT ADDED at position %d with header name '%s'\n"
                % (c, mx, index_of_insertion, variable_name)
            )
        return

    def remove_column(self, column_index, quiet=False):
        if type(column_index) is list or type(column_index) is tuple:
            column_index = [
                c for c in column_index if c is not None
            ]  # remove eventual null entries
            if column_index[0] in self.hd or type(column_index[0]) is str:
                col2 = []
                for c in column_index:
                    if c not in self.hd:
                        if not quiet:
                            sys.stderr.write(
                                "**WARNING** in remove_column asked to remove %s but this is not in HD\n"
                                % (str(c))
                            )
                    else:
                        col2 += [self.hd[c]]
                column_index = col2
        else:
            if type(column_index) is str:
                if column_index not in self.hd:
                    if not quiet:
                        sys.stderr.write(
                            "**WARNING** in remove_column asked to remove %s but this is not in HD\n"
                            % (str(column_index))
                        )
                    return
                column_index = self.hd[column_index]
            elif column_index in self.hd:
                column_index = self.hd[column_index]
        if not hasattr(column_index, "__len__"):
            column_index = [column_index]
        remk = []
        cindex = sorted(column_index, reverse=True)
        for ci in cindex:
            if ci >= len(self.hd):
                sys.stderr.write(
                    "**ERROR** in remove_column index to remove %d outside length of hd %d -skipping index\n"
                    % (ci, len(self.hd))
                )
                continue
            raised = 0
            for k in self:
                if ci >= len(self[k]):
                    if raised < 4:
                        sys.stderr.write(
                            "**ERROR** in remove_column index to remove %d [hd=%s] outside length of self[k] %d for k=%s -skipping row\n"
                            % (ci, str(self.HD()[ci]), len(self[k]), str(k))
                        )
                        raised += 1
                    elif raised == 4:
                        sys.stderr.write(
                            "**ERROR** in remove_column index to remove %d [hd=%s] outside length of self[k] %d for k=%s -skipping row SUPPRESSING FURTHER WARNINGS FOR THIS INDEX\n"
                            % (ci, str(self.HD()[ci]), len(self[k]), str(k))
                        )
                        raised += 1
                    continue
                del self[k][ci]
            for k in self.hd:
                if self.hd[k] == ci:
                    remk += [k]
        off = 0  # rephase hd
        for k in self.HD():
            if self.hd[k] in cindex:
                off += 1
            self.hd[k] = self.hd[k] - off
        for k in remk:
            del self.hd[k]
        return

    def filter_columns(self, columns_to_keep):
        """
        returns a copy of the data class with only selected columns as given in columns_to_keep
        keys will be retained
        """
        if columns_to_keep is None:  # used in Print
            return self
        if (
            type(columns_to_keep) is not list
            and type(columns_to_keep) is not tuple
            and (type(columns_to_keep) is str or type(columns_to_keep) is int)
        ):
            columns_to_keep = [columns_to_keep]
        columns_to_keep = [
            c for c in columns_to_keep if c is not None
        ]  # remove eventual None if present
        columns_to_keep_indices = [
            c if type(c) is int else self.hd[c] for c in columns_to_keep
        ]
        columns_to_keep_names = [self.HD()[j] for j in columns_to_keep_indices]
        filt = self.copy(deep=False)
        filt.hd = list_to_dictionary(columns_to_keep_names)
        for k in self:
            filt[k] = [self[k][j] for j in columns_to_keep_indices]
        return filt

    def __add__(self, y):
        """
        x.__add__(y) <==> x+y
        """
        return self.add(y)

    def __iadd__(self, y):
        """
        x.__iadd__(y) <==> x+=y
        """
        self.iadd(y)

    def add(self, dict2, overwrite=False, print_warnings=True):
        """
        can be called with +  like NewData= ThisData + dict2 in list-like concatenation (Vertically STAKCS however)
        create a new Data class corresponding to the fusion of self and the keys (and corresponding values) of dict2.
          adds entries from dict2 only if the key are not already present in self (unless overwrite=True)
        """
        out = self.copy(deep=True)
        if isinstance(dict2, Data):
            if dict2.hd != self.hd:
                sys.stderr.write(
                    "WARNING in add() the header of dict2 is different from the current header."
                )
        if "hd" in dir(dict2):
            if dict2.hd != self.hd:
                print("**Warning** adding another Data class with different header!!")
        for k in dict2:
            if k not in out:
                out[k] = dict2[k][:]
            elif overwrite:
                if print_warnings:
                    print("Overwriting %s" % (str(k)))
                out[k] = dict2[k][:]
            elif print_warnings:
                sys.stderr.write(
                    "Warn in Data.add() key %s of dict2 already in self. Not adding\n"
                    % (k)
                )
        return out

    def iadd(self, dict2, overwrite=False, print_warnings=True):
        """
        adds the keys (and corresponding values) of dict2 in self if they are not already present (unless overwrite=True)
        """
        if "hd" in dir(dict2):
            if dict2.hd != self.hd:
                print(
                    "**Warning** adding another Data class with different header (len new=%d current=%d)!!"
                    % (len(dict2.hd), len(self.hd))
                )
                print("  new= %s ; current= %s" % (str(dict2.HD()), str(self.HD())))
        for k in dict2:
            if k not in self:
                self[k] = dict2[k][:]
            elif overwrite:
                print("Overwriting %s" % (str(k)))
                self[k] = dict2[k][:]
            elif print_warnings:
                sys.stderr.write(
                    "Warn in Data.iadd() key %s of dict2 already in self. Not adding\n"
                    % (k)
                )

    def vstack(self, dict2, overwrite=False, print_warnings=True):
        """
        adds dict 2 vertically to self
        """
        return self.iadd(dict2, overwrite=overwrite, print_warnings=print_warnings)

    def hstack(
        self, dict2, find_nearest=False, find_nearest_percent_warning_threshold=5
    ):
        """
        add to existing dictionary a new one horizontally - the two should have the same keys
        if dict2 is longer than self or has extra keys it won't print warnings but will only add
         those keys in common from dict2 to all keys of self (thus some of dict2 may not be added)
        """
        hd1 = self.HD()
        hd2 = []
        for k in dict2.HD():
            if k in hd1:
                sys.stderr.write(
                    "Warn adding new HD entry  %s already in first dictionary HD" % (k)
                )
                hd2 += [new_key_for_dict(hd1 + hd2, k)]
                sys.stderr.write(" so added %s instead\n" % (hd2[-1]))
            else:
                hd2 += [k]

        if find_nearest:
            d2 = numpy.array(list(map(float, list(dict2.keys()))))
            done, skipped, k2_tok = {}, {}, {}
            if len(dict2) > len(self):
                for j, k in enumerate(self):
                    k2, i = misc.find_nearest(
                        d2, float(k)
                    )  # the same k2 may end up in multiple k in principle
                    # if i in done :
                    #    sys.stderr.write("-- Warn in hstack with find nearest k2 %s already added to k %s SKIPPING new k %s\n" % (str(k2),str(done[i]),str(k) ))
                    #    skipped[k]=k2
                    #    continue
                    # else :
                    done[i] = k
                    self[k] += list(dict2.values())[i][:]
                    if (
                        float(k) != k2
                        and float(k) != 0
                        and 100.0 * abs(float(k) - k2) / float(k)
                        > find_nearest_percent_warning_threshold
                    ):
                        sys.stderr.write(
                            "-- Warn in hstack with find nearest k2 %g matched to k %g but they are %g %% different\n"
                            % (k2, k, 100.0 * abs(float(k) - k2) / float(k))
                        )
                for i, k2 in enumerate(dict2):
                    if i not in done:
                        skipped[k2] = i
            else:
                kadd = numpy.array(list(map(float, list(self.keys()))))
                for k2 in dict2:
                    k, i = misc.find_nearest(kadd, float(k2))
                    if i in done:
                        sys.stderr.write(
                            "-- Warn in hstack with find nearest k2(%s) already added to k %s SKIPPING k2 %s\n"
                            % (str(done[i]), str(k), str(k2))
                        )
                        skipped[k2] = k
                        continue
                    else:
                        done[i] = k2
                        k2_tok[k2] = k
                        list(self.values())[i] += dict2[k2][
                            :
                        ]  # k is float but in dict could be str
                        if (
                            float(k) != k2
                            and float(k) != 0
                            and 100.0 * abs(float(k) - k2) / float(k)
                            > find_nearest_percent_warning_threshold
                        ):
                            sys.stderr.write(
                                "-- Warn in hstack with find nearest k2 %g matched to k %g but they are %g %% different\n"
                                % (k2, k, 100.0 * abs(float(k) - k2) / float(k))
                            )
                # above line is slow
                # check you matched them all and if not try to match remaining
                for i, k in enumerate(self):
                    if i not in done:
                        k2, j = misc.find_nearest(d2, float(k))
                        if (
                            k2 not in skipped
                        ):  # here len2 is <=len(self) so k2 must be in k2_tok
                            sys.stderr.write(
                                "-* Warn in hstack with find nearest adding to k %s k2 %s that was also added to %s (len(self)=%d len(2)=%d)\n"
                                % (
                                    str(k),
                                    str(k2),
                                    str(k2_tok[k2]),
                                    len(self),
                                    len(dict2),
                                )
                            )
                        else:
                            del skipped[k2]
                        self[k] += list(dict2.values())[j][:]
                        if (
                            float(k) != k2
                            and float(k) != 0
                            and 100.0 * abs(float(k) - k2) / float(k)
                            > find_nearest_percent_warning_threshold
                        ):
                            sys.stderr.write(
                                "-- Warn in hstack with find nearest k2 %g matched to k %g but they are %g %% different\n"
                                % (k2, k, 100.0 * abs(float(k) - k2) / float(k))
                            )
            if len(skipped) > 0:
                sys.stderr.write(
                    "** Warn in hstack with find nearest %d keys from dict2 have been skipped (len(dict2)=%d len(self)=%d diff=%d)\n"
                    % (len(skipped), len(dict2), len(self), len(dict2) - len(self))
                )
                # sys.stderr.write("-* Warn in hstack with find nearest matching k %lf to k2 %lf (D=%lf) as closest match was previously matched (%lf %lf)\n" % (k,k2,k-k2, done[skipped[]] ))
        else:
            for k in self:
                if k not in dict2:
                    sys.stderr.write(
                        "WARNING in hstack k %s found only in self, adding empties ''\n"
                        % (k)
                    )
                    self[k] += [""] * len(hd2)
                else:
                    self[k] += dict2[k][:]
        self.hd = list_to_dictionary(hd1 + hd2)
        return

    def merge(
        self,
        dict2,
        new_variables_name=None,
        index_of_insertion=None,
        merge_according_to_column_index=None,
    ):
        """
        # merges dict2 into self matching the keys of dict2 that are also in self. self is consequently changed
        # dict2 can also be another Data() class
        #  if merge_according_to_column_index is not None : the merging is performed using the entries of this columns as 'keys for self'
        """
        if index_of_insertion is None:
            index_of_insertion = len(self.hd)
        elif type(index_of_insertion) is str:
            index_of_insertion = self.hd[index_of_insertion]
        ######################################
        # index_of_insertion+=1

        if merge_according_to_column_index is not None:
            col, keys = self.column(merge_according_to_column_index, return_keys=True)
            self_iterable = OrderedDict()
            for j, c in enumerate(col):
                if c in self_iterable:
                    sys.stderr.write(
                        "**WARNING** in merge() trying to merge() according to column %s but column entries are not unique, overwriting %s-->%s in %s-->%s\n"
                        % (
                            str(merge_according_to_column_index),
                            c,
                            self_iterable[c],
                            c,
                            keys[j],
                        )
                    )
                self_iterable[c] = keys[j]
        else:
            self_iterable = OrderedDict()
            for k in self:
                self_iterable[k] = k
        if (
            type(list(dict2.values())[0][0]) is list
            or type(list(dict2.values())[0][0]) is tuple
        ):
            dict2_header_length = len(list(dict2.values())[0][0])
        else:
            dict2_header_length = len(list(dict2.values())[0])
        # DO THE MERGING
        for key in dict2:
            if key in self_iterable:
                if type(self[self_iterable[key]]) is list:
                    if (
                        type(self[self_iterable[key]][0]) is list
                    ):  # if self contains a list of lists
                        for i in range(0, len(self[self_iterable[key]])):
                            if type(dict2[key]) is list:
                                self[self_iterable[key]][i][
                                    index_of_insertion:index_of_insertion
                                ] = dict2[key]
                            elif type(dict2[key]) is tuple:
                                print(
                                    "***WARNING*** in MergeDictionaries trying to merge dictionaries of non compatible types (list of lists, tuple) Converting to list\n"
                                )
                                self[self_iterable[key]][i][
                                    index_of_insertion:index_of_insertion
                                ] = list(dict2[key])
                            else:
                                self[self_iterable[key]][i][
                                    index_of_insertion:index_of_insertion
                                ] = [dict2[key]]
                        del i
                    elif (
                        type(self[self_iterable[key]][0]) is tuple
                    ):  # if self contains a list of tuples
                        for i in range(0, len(self[self_iterable[key]])):
                            if type(dict2[key]) is tuple:
                                self[self_iterable[key]][i] = (
                                    self[self_iterable[key]][i][:index_of_insertion]
                                    + dict2[key]
                                    + self[self_iterable[key]][i][index_of_insertion:]
                                )
                            elif type(dict2[key]) is list:
                                print(
                                    "***WARNING*** in MergeDictionaries trying to merge dictionaries of non compatible types (list of tuples, list) Converting to list\n"
                                )
                                self[self_iterable[key]][i] = list(
                                    self[self_iterable[key]][i]
                                )
                                self[self_iterable[key]][i][
                                    index_of_insertion:index_of_insertion
                                ] = dict2[key]
                            else:
                                self[self_iterable[key]][i] = (
                                    self[self_iterable[key]][i][:index_of_insertion]
                                    + (dict2[key],)
                                    + self[self_iterable[key]][i][index_of_insertion:]
                                )
                        del i
                    else:
                        if type(dict2[key]) is list:
                            self[self_iterable[key]][
                                index_of_insertion:index_of_insertion
                            ] = dict2[key]
                        elif type(dict2[key]) is tuple:
                            print(
                                "***WARNING*** in MergeDictionaries trying to merge dictionaries of non compatible types (list, tuple) Converting to list\n"
                            )
                            self[self_iterable[key]][
                                index_of_insertion:index_of_insertion
                            ] = list(dict2[key])
                        else:
                            self[self_iterable[key]][
                                index_of_insertion:index_of_insertion
                            ] = [dict2[key]]
                elif (
                    type(self[self_iterable[key]]) is tuple
                ):  # if self is a dictionary whose elements are tuples then try to merge in a tuple (if dict2 elements are list it will convert to lists
                    if type(dict2[key]) is tuple:
                        self[self_iterable[key]] = (
                            self[self_iterable[key]][:index_of_insertion]
                            + dict2[key]
                            + self[self_iterable[key]][index_of_insertion:]
                        )
                    elif type(dict2[key]) is list:
                        print(
                            "***WARNING*** in MergeDictionaries trying to merge dictionaries of non compatible types (tuple, list) Converting to list\n"
                        )
                        self[self_iterable[key]] = list(self[self_iterable[key]])
                        self[self_iterable[key]][
                            index_of_insertion:index_of_insertion
                        ] = dict2[key]
                    else:
                        self[self_iterable[key]] = (
                            self[self_iterable[key]][:index_of_insertion]
                            + (dict2[key],)
                            + self[self_iterable[key]][index_of_insertion:]
                        )
                else:  # if self is a standard dictionary merge with dict2 converting it into a list dictionary, unless dict2 is tuples in which case convert to tuple
                    if type(dict2[key]) is tuple:
                        self[self_iterable[key]] = (
                            tuple([self[self_iterable[key]]]) + dict2[key]
                        )
                    elif type(dict2[key]) is list:
                        self[self_iterable[key]] = [self[self_iterable[key]]] + dict2[
                            key
                        ]
                    else:
                        self[self_iterable[key]] = [
                            self[self_iterable[key]],
                            dict2[key],
                        ]

        # FIX THE HEADER
        new_columns_hd = []
        if new_variables_name is None and "hd" in dir(
            dict2
        ):  # then dict2 is a Data() class and has its own index!
            if len(dict2.hd) != dict2_header_length:
                sys.stderr.write(
                    "WARNING in merge() estimated length of dict2 is %d while its header is %d long! Problems in final header! dict2_header: %s \n"
                    % (dict2_header_length, len(dict2.hd), str(dict2.hd))
                )
            h = list(dict2.hd.items())
            h.sort(key=lambda pair: pair[1])
            new_columns_hd = list(zip(*h))[0]
            del h
        elif type(new_variables_name) is dict:
            if len(new_variables_name) != dict2_header_length:
                sys.stderr.write(
                    "WARNING in merge() estimated length of dict2 is %d while its header is %d long! Problems in final header! dict2_header: %s \n"
                    % (
                        dict2_header_length,
                        len(new_variables_name),
                        str(new_variables_name),
                    )
                )
            h = list(new_variables_name.items())
            h.sort(key=lambda pair: pair[1])
            new_columns_hd = list(zip(*h))[0]
            del h
        elif type(new_variables_name) is list or type(new_variables_name) is tuple:
            if len(new_variables_name) != dict2_header_length:
                sys.stderr.write(
                    "WARNING in merge() estimated length of dict2 is %d while its header is %d long! Problems in final header! dict2_header: %s \n"
                    % (
                        dict2_header_length,
                        len(new_variables_name),
                        str(new_variables_name),
                    )
                )
            new_columns_hd = new_variables_name
        elif type(new_variables_name) is str:
            if 1 != dict2_header_length:
                sys.stderr.write(
                    "WARNING in merge() estimated length of dict2 is %d while its header is %d long! Problems in final header! dict2_header: %s \n"
                    % (dict2_header_length, 1, str(new_variables_name))
                )
            new_columns_hd = [new_variables_name]
        elif new_variables_name is None:
            new_columns_hd = [
                "new_column%02d" % i for i in range(0, dict2_header_length)
            ]
        for j, var_name in enumerate(new_columns_hd):
            self._update_hd(var_name, index_of_insertion + j)

        self._equate_to_same_length()
        return 0

    # equate all entries to same length by adding '' empty columns at the end of the shorter ones
    def _equate_to_same_length(
        self, print_warn_for_shorter=True, merge_longer=False, insert_for_missings=""
    ):
        self = equate_data_to_same_length(
            self,
            header=self.hd,
            print_warn_for_shorter=print_warn_for_shorter,
            merge_longer=merge_longer,
            insert_for_missings=insert_for_missings,
        )

    def _update_hd(self, var_name, index_of_insertion=-1):
        """
        update the header if a column is inserted
         -1 assumes the column is inserted at the end of the Database
        """
        if type(index_of_insertion) is str:
            index_of_insertion = self.hd[index_of_insertion]
        if var_name in self.hd:
            sys.stderr.write(
                "\n***WARN*** in _update_hd() name %s already in header," % (var_name)
            )
            if len(var_name)>=2 and var_name[-1].isdigit() and var_name[-2].isdigit() :
                var_name = new_key_for_dict(self.hd, var_name[:-2], start_at=int(var_name[-2:]))
            else :
                var_name = new_key_for_dict(self.hd, var_name, start_at=0)
            sys.stderr.write(" adding %s instead\n\n" % (var_name))
        if index_of_insertion < 0:
            index_of_insertion = (
                len(self.hd) + 1 + index_of_insertion
            )  # if it is -1 you append at the end, if it is -2 you append one before the end and so on...
        if index_of_insertion == len(self.hd):
            self.hd[var_name] = index_of_insertion
            return
        elif index_of_insertion > len(self.hd):
            sys.stderr.write(
                "**ERROR in _update_hd() looks like header is shorter than required addition at %d %s len(hd)=%d\n"
                % (index_of_insertion, var_name, len(self.hd))
            )
            self.hd[var_name] = index_of_insertion
            return
        else:
            items = list(self.hd.items())
            items.sort(key=lambda pair: pair[1])
            s_keys, _ = list(zip(*items))
            del items
            s_keys = list(s_keys)
            s_keys[index_of_insertion:index_of_insertion] = [var_name]
            self.hd = {}
            for j, key in enumerate(s_keys):
                self.hd[key] = j
            return

    def round_all(self, ndecimals=2):
        for k in self:
            for j, v in enumerate(self[k]):
                if type(v) is not int and type(v) is not str:
                    try:
                        vr = float(numpy.round(v, ndecimals))
                    except Exception:
                        vr = v
                    self[k][j] = vr
        return

    def _column_name_to_index(self, column_name_or_index, print_str_with_s=None):
        # e.g. print_str_with_s='GETTING sequence_to_fasta() sequence from %s\n'  will put column_name_or_index in %s
        if type(column_name_or_index) is str:
            if column_name_or_index in self.hd:
                if print_str_with_s is not None:
                    sys.stdout.write(print_str_with_s % (column_name_or_index))
                column_name_or_index = self.hd[column_name_or_index]
            elif (
                column_name_or_index == "keys"
                or column_name_or_index == self.key_column_hd_name
            ):
                column_name_or_index = "keys"
            else:
                sys.stderr.write(
                    "**WARNING** %s not in header and not key column, will likely lead to ERROR\n"
                    % (column_name_or_index)
                )
        return column_name_or_index

    def _get_value_of_column_index(self, key, column_index):
        if column_index == "keys":
            return key
        else:
            return self[key][column_index]

    def sequence_to_fasta(
        self,
        seq_column_index,
        outfile=None,
        ids_column_index="keys",
        weight_column_index=None,
        description_column_index=None,
        add_str_before_all_names="",
    ):
        """
        if a column (seq_column_index) contain biological sequences it prints them to fasta 
        if ids_column_index is None it will use index of entry 
        return seqRecords
        """
        seq_column_index = self._column_name_to_index(
            seq_column_index,
            print_str_with_s="GETTING sequence_to_fasta() sequence from '%s'\n",
        )
        ids_column_index = self._column_name_to_index(ids_column_index)
        description_column_index = self._column_name_to_index(description_column_index)
        weight_column_index = self._column_name_to_index(weight_column_index)
        seqRecords = []
        for j, k in enumerate(self):
            if ids_column_index is None:
                nam = add_str_before_all_names + str(j)
            else:
                nam = add_str_before_all_names + self._get_value_of_column_index(
                    k, ids_column_index
                )
            if description_column_index is not None:
                des = self._get_value_of_column_index(k, description_column_index)
            else:
                des = ""
            seqRecords += [
                SeqRecord(
                    seq=Seq(self._get_value_of_column_index(k, seq_column_index)),
                    id=nam,
                    name=nam,
                    description=des,
                )
            ]
            if weight_column_index is not None:
                seqRecords[-1].annotations["weight"] = self._get_value_of_column_index(
                    k, weight_column_index
                )

        if type(outfile) is bool and outfile == False:
            return seqRecords
        if outfile is None:
            if self.filename is not None:
                _, outfile, _ = misc.get_file_path_and_extension(self.filename)
                outfile += ".fasta"
            else:
                outfile = "Sequences.fasta"
            print("sequence_to_fasta() output in %s" % (outfile))
        mybio.PrintSequenceRecords(seqRecords, outfile)
        return seqRecords

    def bin_column(
        self,
        column_index,
        nbin=None,
        Min=None,
        Max=None,
        bin_column_name=None,
        strict_extremes=False,
        reverse=False,
        add_to_rank=0,
    ):
        """
        # add a column with the corresponding entry bin after the column 
        # selected by column_index
        if bin_column_name is None : bin_column_name='bin_'+str(column_index)
        reverse is effectively doing a binned ranking. (so it assigns bin 0 to highest)
        add_to_rank can be given to shift the first bin (e.g. add_to_rank=1 will make the first bin 1 rather than 0)
          useful for ranking
        """
        if bin_column_name is None:
            bin_column_name = "bin_" + str(column_index)
        if type(column_index) is str:
            column_index = self.hd[column_index]
        elements = self.column(column_index)  # need it to get max and min
        if nbin is None:
            nbin = int(numpy.sqrt(len(elements)))
        if Min is None:
            Min = min(elements)
        if Max is None:
            Max = max(elements)
        rec_binsize = 1.0 * nbin / (Max - Min)
        if reverse:
            bins = (
                -1
                * (
                    numpy.digitize(
                        numpy.array(elements), numpy.linspace(Min, Max, nbin)
                    )
                    - nbin
                )
                + add_to_rank
            ).astype("int")
        else:
            bins = (
                numpy.digitize(numpy.array(elements), numpy.linspace(Min, Max, nbin))
                - 1
                + add_to_rank
            ).astype(
                "int"
            )  # -1 will make bins from 0 to nbin-1
        self.add_column(
            bins, index_of_insertion=column_index, variable_name=bin_column_name
        )
        return

    def clusterize(
        self,
        column_indices=None,
        norm_normalize_threshold=None,
        tolerance_sigma=None,
        strict_clustering=True,
    ):
        """
        # column_index=None will clusterize it by keys, a cluster_id (an int) is added as an entry
        # otherwise it will be clusterized by the entry in that column
        # multiple columns can be given, in which case the clustering is done according to a difference vector norm.
        # the greater tolerance_sigma<0 and the larger the number of resulting clusters (stricter conditions for belonging to the same cluster)
        #  while tolerance_sigma>0 will loosen the conditions making it easier for two entries to end up in the same cluster
        #  tolerance_sigma=None is like tolerance_sigma=0 but saves cpu cycles
        # if strict_clustering is True then the automatically determined cluster condition is applied strictly.
        #  this means that all element within a cluster must satisfy it.
        #  if, on one hand this is desirable, on the other this might lead to the situation where an element, which is not part of a cluster, is
        #  actually very close to every element in the cluster but one.
        # if strict_clustering is False then the separation is in connected units,
        #   that is the condition is only applied to consecutive elements in the sorted list.
        #   this could be desirable if the elements_to_cluster are not not particularly connected, so that it is more effectively separated only where it matters
        # strickt_separate = False equal to say that we tolerate False positive more than False negative, 
        #  (e.g. elements that are in the cluster but shouldn't be there are less of a problem than elements that are not but should be).
        """
        if column_indices is None:
            elements = list(self.keys())
        else:
            elements = self.column(column_indices)
        clusters, cluster_v = misc.cluster_with_norm(
            elements,
            norm_normalize_threshold=norm_normalize_threshold,
            tolerance_sigma=tolerance_sigma,
            strict_clustering=strict_clustering,
        )
        # add cluster_id to elements
        l = len(self.hd)
        if "cluster_id" not in self.hd:
            self.hd["cluster_id"] = l
        else:
            c = new_key_for_dict(self.hd, "cluster_id", start_at=0)
            self.hd[c] = l
        index_to_cluster = {}
        for cluster_id, cluster in enumerate(clusters):
            for j in cluster:
                if j in index_to_cluster:
                    sys.stderr.write(
                        "**WARNING** in clusterize() index %d now found in cluster %d already found previously\n"
                        % (j, cluster_id)
                    )
                index_to_cluster[j] = cluster_id

        # add cluster_ids to data
        c = 0  # this is a sorted dictionary
        for key in self:
            if type(self[key]) is list or type(self[key]) is tuple:
                if type(self[key][0]) is list:
                    for row in self[key]:
                        if c not in index_to_cluster:
                            sys.stderr.write(
                                "**WARNING** in clusterize() index %d doesn't belong to any cluster!!\n"
                                % (c)
                            )
                        else:
                            row += [index_to_cluster[c]]
                        c += 1
                    del row
                else:
                    if c not in index_to_cluster:
                        sys.stderr.write(
                            "**WARNING** in clusterize() index %d doesn't belong to any cluster!!\n"
                            % (c)
                        )
                    else:
                        self[key] += [index_to_cluster[c]]
                    c += 1
        return clusters, cluster_v

    def group(self, group_by):
        """
        from the database it returns a dictionary of dictionaries (actually a Data class of Data classes).
          outer keys are the entries found in group by, inner keys are the entries found in self
        note that the header saved in the outer Data is for the header of the inner one. (but it wouldn't make sense to make thousands of copies of the same header)
        """
        if type(group_by) is str:
            group_by = self.hd[group_by]
        max_id = group_by
        grouped = self.copy(
            deep=False
        )  # we don't copy all items now as only those respecting condition will be copied.
        for key in self:
            if type(self[key]) is list or type(self[key]) is tuple:
                if self.allow_more_entries_per_key and type(self[key][0]) is list:
                    for row in self[key]:
                        if max_id < len(row):
                            if row[group_by] not in grouped:
                                grouped[row[group_by]] = Data()
                                grouped[row[group_by]].hd = copy.deepcopy(self.hd)
                            if key not in grouped[row[group_by]]:
                                grouped[row[group_by]][key] = [row]
                            else:
                                grouped[row[group_by]][key] += [row]
                        else:
                            print(
                                "***ERROR*** in group() key=%s column_index out of range (%d columns found in row, requesting for columnd index %d)\n%s\n"
                                % (str(key), len(row), group_by, str(row))
                            )
                            return 2
                    del row
                else:
                    if max_id < len(self[key]):
                        if self[key][group_by] not in grouped:
                            grouped[self[key][group_by]] = Data()
                            grouped[self[key][group_by]].hd = copy.deepcopy(self.hd)
                        if key not in grouped[self[key][group_by]]:
                            grouped[self[key][group_by]][key] = self[key]
                        else:
                            print(
                                "***WARNING*** in group() group_by key %s already contains key %s!!"
                                % (str(self[key][group_by]), str(key))
                            )
                    else:
                        print(
                            "***ERROR*** in group() key=%s column_index out of range (%d columns found in row, requesting for columnd index %d)\nrow=%s\n"
                            % (str(key), len(row), group_by, str(row))
                        )
                        return 2
            else:
                print(
                    "***ERROR*** in column dictionary given is not a list of lists and its entries are not lists,returning 1"
                )
                return 1
        return grouped

    def group_by_condition(self, condition_key, condition):
        """
        from the database it returns a Data class.
         keys and values are only those in the original data class for which
          the value at column conditon_key statisfied the function condition (which should return True/False
        """
        if not hasattr(condition, "__call__"):
            sys.stderr.write(
                "***ERROR***, in group_by_condition() condition must be callable!!\n"
            )
            return 3
        if type(condition_key) is str:
            if condition_key not in self.hd and condition_key == "keys":
                condition_key = None  # apply condition on the key itself
            else:
                group_by = self.hd[condition_key]
        else:
            group_by = condition_key

        grouped = self.copy(
            deep=False
        )  # we don't copy all items now as only those respecting condition will be copied.
        if condition_key is None:  # apply condition on the key itself
            for key in self:
                if condition(key):
                    grouped[key] = copy.deepcopy(self[key])
            return grouped

        max_id = group_by
        for key in self:
            if type(self[key]) is list or type(self[key]) is tuple:
                if self.allow_more_entries_per_key and type(self[key][0]) is list:
                    for row in self[key]:
                        if max_id < len(row):
                            if condition(row[group_by]):
                                if key not in grouped:
                                    grouped[key] = [row[:]]
                                else:
                                    grouped[key] += [row[:]]
                        else:
                            sys.stderr.write(
                                "***ERROR*** in group() column_index out of range (%d columns, requesting for %d)\n%s\n"
                                % (len(row), group_by, str(row))
                            )
                            return 2
                    del row
                else:
                    if max_id < len(self[key]):
                        if condition(self[key][group_by]):
                            grouped[key] = copy.deepcopy(self[key])
                    else:
                        sys.stderr.write(
                            "***ERROR*** in group() column_index out of range (%d columns, requesting for %d)"
                            % (len(self[key]), group_by)
                        )
                        return 2
            else:
                sys.stderr.write(
                    "***ERROR*** in column dictionary given is not a list of lists and its entries are not lists,returning 1"
                )
                return 1
        return grouped

    def count_plot(
        self,
        column_index,
        condition=None,
        xlabel=None,
        ylabel="count",
        bar=True,
        xlabels=True,
        sort=True,
        plot=True,
        **kwargs
    ):
        """
        from a column containing a category value (or numeric values to which condition is then applied if given, e.g. condition=lambda x : x>0 ) 
         it counts the entries corresponding to each unique identifiers
         and returns a dictionary with the count
        """
        if xlabel == False:
            xlabel = None
        if xlabel == True:
            if type(column_index) is str:
                xlabel = column_index
            elif type(column_index) is int:
                for k in self.hd:
                    if self.hd[k] == column_index:
                        xlabel = k
        if type(column_index) is str:
            ind = self.hd[column_index]
        else:
            ind = column_index
        count = OrderedDict()
        for k in self:
            e = self[k][ind]
            if condition is not None:
                if len(count) == 0:
                    count["positive"] = 0
                    count["negative"] = 0
                if condition(e):
                    count["positive"] += 1
                else:
                    count["negative"] += 1
            else:
                if e not in count:
                    count[e] = 0
                count[e] += 1
        if sort:
            count = OrderedDict(sorted(count.items()))
        if type(xlabels) is bool and xlabels == True:
            xlabels = list(count.keys())
            vals = list(count.values())
        elif type(xlabels) is list or type(xlabels) is tuple:
            vals = []
            for k in xlabels:
                if k in count:
                    vals += [count[k]]
                else:
                    print(
                        "WARNING given xlabels %s not found among categories %s"
                        % (str(k), str(column_index))
                    )
        else:
            vals = list(count.values())
        if plot:
            plotter.profile(
                vals, ylabel=ylabel, xlabel=xlabel, xlabels=xlabels, bar=bar, **kwargs
            )
        return count

    def profile(
        self,
        column_index,
        x_values=None,
        group_by=None,
        fast=False,
        yerr=None,
        xerr=None,
        ylabel=None,
        xlabel=None,
        xlabels=None,
        label=None,
        quality_filter=None,
        verbose=True,
        CI_to_err=False,
        labels=None,
        **kwargs
    ):
        """
        uses the plotter function to plot a profile from one of the columns
        quality_filter can use a column with bool to mark outliers or low quality data
        group_by can be used to plot multiple profile from the same X, Y columns but separated according to another column
        """
        if group_by is not None:
            x = None
            groups = self.group(group_by)
            labels, xalbels_vals = [], []
            y = []
            xvals, yerrvals, xerrvals = [], [], []
            for k in groups:
                (
                    yv,
                    xposv,
                    yerrv,
                    xerrv,
                    xlabel,
                    ylabel,
                    cand_labels,
                    xlabs_vals,
                ) = groups[k]._get_plot_entries(
                    column_index,
                    x=x_values,
                    yerr=yerr,
                    xerr=xerr,
                    label=True,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    labels=xlabels,
                    verbose=False,
                )
                if labels is not None and labels != False:
                    _, _, _, _, _, _, _, labels = groups[k]._get_plot_entries(
                        column_index,
                        x=x_values,
                        yerr=yerr,
                        xerr=xerr,
                        label=True,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        labels=labels,
                        verbose=False,
                    )
                # if exclude_less_than_points is not None and len(yv)< exclude_less_than_points :
                #    print ' **profile group_by exclude_less_than_points --> discarding group %s as only %d entries' % (k,len(yv))
                #    continue
                # print 'DEB:',type(yv),yv
                if type(cand_labels) is list:
                    labels += [k + " " + c for c in cand_labels]
                else:
                    labels += [k]
                if xposv is not None and x_values is not None:
                    if hasattr(yv[0], "__len__") and type(yv) is list:
                        if hasattr(xposv[0], "__len__") and type(xposv) is list:
                            xvals += xposv
                        else:
                            xvals += [xposv] * len(yv)
                    else:
                        xvals += [xposv]
                if xlabels is not None and xlabs_vals is not None:
                    xalbels_vals += [xlabs_vals]
                if hasattr(yv[0], "__len__") and type(yv) is list:
                    y += yv
                else:
                    y += [yv]
                if yerr is not None:
                    if hasattr(yerrv[0], "__len__") and type(yerrv) is list:
                        yerrvals += yerrv
                    else:
                        yerrvals += [yerrv]
                if xerr is not None:
                    if hasattr(xerrv[0], "__len__") and type(xerrv) is list:
                        xerrvals += xerrv
                    else:
                        xerrvals += [xerrv]
            if label is None or (type(label) is bool and label == True):
                label = labels
            if xalbels_vals != []:
                if all([xlz == xalbels_vals[0] for xlz in xalbels_vals]):
                    xlabels = xalbels_vals[0]
                elif len(xlabels[0]) == len(y[0]):
                    print(
                        (
                            "WARNING profile with group_by %s xlabels is given but different xlabels found - probably not implemented setting xlabels to %s"
                            % (group_by, str(xalbels_vals[0]))
                        )
                    )
                    xlabels = xalbels_vals[0]
                else:
                    print(
                        (
                            "WARNING profile with group_by %s xlabels is given but different xlabels found - SETTING TO NONE"
                            % (group_by)
                        )
                    )
                    xlabels = None
            if yerrvals != []:
                yerr = yerrvals
            if xerrvals != []:
                xerr = xerrvals
            if xvals != []:
                x = xvals
            if verbose:
                print(
                    (
                        "len(y)=",
                        len(y),
                        "len(xvals)",
                        len(xvals),
                        "len(yerrvals)=",
                        len(yerrvals),
                        "len(xerrvals)=",
                        len(xerrvals),
                        "xlabels=",
                        xlabels,
                        "labels=",
                        labels,
                        "xlabel=",
                        xlabel,
                        "ylabel=",
                        ylabel,
                        "xlabels=",
                        xlabels,
                        "\n",
                    )
                )
        else:
            y, x, yerr, xerr, xlabel, ylabel, label, xlabels = self._get_plot_entries(
                column_index,
                x=x_values,
                yerr=yerr,
                xerr=xerr,
                xlabel=xlabel,
                ylabel=ylabel,
                label=label,
                verbose=verbose,
                labels=xlabels,
                CI_to_err=CI_to_err,
            )
            if labels is not None and labels != False:
                _, _, _, _, _, _, _, labels = self._get_plot_entries(
                    column_index,
                    x=x_values,
                    yerr=yerr,
                    xerr=xerr,
                    label=True,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    labels=labels,
                    verbose=False,
                )
        if quality_filter is not None:
            if verbose:
                print(("quality_filter=", quality_filter))
            if (
                type(quality_filter) is str
                or type(quality_filter) is int
                or not hasattr(quality_filter, "__len__")
            ):
                quality_filter = self.column(quality_filter)
            if len(quality_filter) != len(y):
                print(
                    "**WARNING** Cannot use quality_filter len(quality_filter) != len(y) %d %d not implemented - disabling option"
                    % (len(quality_filter), len(y))
                )
                quality_filter = None
            if quality_filter is not None:
                qprof = [
                    y[j] if q in [True, 1, "True", "TRUE", "true"] else numpy.nan
                    for j, q in enumerate(quality_filter)
                ]
                if "bar" in kwargs and kwargs["bar"]:
                    figure = plotter.profile(
                        qprof,
                        x_values=x,
                        bar=True,
                        figure=kwargs.get("figure", None),
                        zorder=99,
                        hatch="//",
                        color="none",
                    )
                else:
                    figure = plotter.profile(
                        qprof,
                        x_values=x,
                        figure=kwargs.get("figure", None),
                        zorder=99,
                        ls="",
                        marker="o",
                        markerfacecolor="none",
                        markeredgecolor="red",
                        markersize=2 * plotter.default_parameters["markersize"],
                    )
                kwargs["figure"] = figure
        try:
            # print ()'y=',y,'x=',x,'label=',label,'xerr=',xerr,'yerr=',yerr)
            figure = plotter.profile(
                y,
                x_values=x,
                yerr=yerr,
                xerr=xerr,
                xlabel=xlabel,
                ylabel=ylabel,
                xlabels=xlabels,
                label=label,
                labels=labels,
                **kwargs
            )
        except Exception:
            print("\nEXCEPTION at csv_dict profile when attempting plotter.profile")
            try:
                if len(y) > 100 or hasattr(y[0], "__len__"):
                    if hasattr(y, "shape"):
                        print("y.shape=", y.shape, end=" ")
                    if hasattr(x, "shape"):
                        print("x.shape=", x.shape, end=" ")
                    elif hasattr(y[0], "__len__"):
                        print(
                            "len(y)",
                            len(y),
                            "uniq([len(yy) for yy in y])",
                            misc.uniq([len(yy) for yy in y]),
                        )
                    if hasattr(x, "__len__"):
                        print("len(x)", len(x), end=" ")
                    if hasattr(yerr, "__len__"):
                        print("yerr=", type(yerr), end=" ")
                    if hasattr(xerr, "__len__"):
                        print("xerr=", type(xerr), end=" ")
                    print("len(y)", len(y))
                else:
                    print(
                        "y=",
                        y,
                        "len(y)",
                        len(y),
                        "x=",
                        x,
                        "yerr=",
                        yerr,
                        "xerr=",
                        xerr,
                        "hasattr(y[0],'__len__')",
                        hasattr(y[0], "__len__"),
                    )
            except Exception:
                pass
            raise  # raise other, not the one of print
        return figure

    def histogram(
        self,
        column_index,
        xlabel=None,
        fast=False,
        verbose=True,
        flag_labels=None,
        **kwargs
    ):
        """
        uses the plotter function to plot an histogram from one of the columns
        """
        entries, _, _, _, xlabel, _, label, _ = self._get_plot_entries(
            column_index,
            x=None,
            yerr=None,
            xerr=None,
            xlabel=xlabel,
            ylabel=None,
            verbose=verbose,
            labels=None,
        )
        """ # IF THE ABOVE WORKS ALL THIS CAN BE DELETED
        if xlabel==False : xlabel=None
        if xlabel==True :
            if column_index is None : xlabel=self.key_column_hd_name
            elif type(column_index) is str : xlabel=column_index
            elif type(column_index) is int :
                if self.hd!={} and self.hd is not None :
                    for k in self.hd : 
                        if self.hd[k]==column_index : xlabel=k
                else : xlabel=None 
        if fast :
            if column_index is None : entries=self.keys()
            else : entries=self.column(column_index)
            entries=numpy.array(entries).astype(float)
        else :
            if column_index is None : 
                entries=self.keys()
                entries=numpy.array(entries).astype(float,copy=False)
            else :
                entries,discarded=self.filter_null(column_index)
                if len(discarded)!=0 : sys.stderr.write('Warning discarding %d entries that had Null values\n' % (len(discarded)))
            if hasattr(entries, 'shape') and len(entries.shape)>1 and entries.shape[0]==1 : entries=entries[0]
        """
        if flag_labels is not None and type(flag_labels) is not dict:
            lab_dict = OrderedDict()
            if type(flag_labels) is str or not hasattr(flag_labels, "__len__"):
                flag_labels = [flag_labels]
            for lab in flag_labels:
                if lab in list(self.keys()):
                    if type(column_index) is int:
                        v = self[lab][column_index]
                    else:
                        v = self[lab][self.hd[column_index]]
                    if v not in entries:
                        print(
                            " entry corresponding to flag_labels %s was probably discarded as %s not found\n"
                            % (str(lab), repr(v))
                        )
                    else:
                        lab_dict[v] = lab
                else:
                    sys.stderr.write(
                        "**ERROR** flag_labels %s not in self\n" % (str(lab))
                    )
                    raise Exception(
                        "Can't process flag_labels (up to 30 printed):\n  "
                        + str(flag_labels[:30])
                    )
            flag_labels = lab_dict
        figure = plotter.histogram(
            entries, flag_labels=flag_labels, xlabel=xlabel, **kwargs
        )
        return figure

    def scatter(
        self,
        column_indexX,
        column_indexY,
        yerr=None,
        xerr=None,
        logx=False,
        logy=False,
        xlabel=None,
        ylabel=None,
        title_correlation=False,
        labels=None,
        verbose=True,
        CI_to_err=False,
        point_labels=False,
        title=None,
        save=None,
        figure=None,
        **kwargs
    ):
        """
        scatter plot between two columns
          if labels is True keys are used to label each point
        """
        if type(save) is bool and save == True:
            save = (
                ("Scatter_" + str(column_indexX) + "_vs_" + str(column_indexY))
                .replace(" ", "")
                .replace("/", "")
                .replace("\\", "-")
                .replace(".", "")
                .replace(",", "")
                .replace(";", "")
                + ".png"
            )
            if os.path.isfile(save):
                sys.stderr.write(
                    "**Warning** scatter auto-save would overwrite existing file %s - trying to add 2 at end of name\n"
                    % (save)
                )
                sys.stderr.flush()
                save = save[:-4] + "2.png"

        y, x, yerr, xerr, xlabel, ylabel, _, labels = self._get_plot_entries(
            column_indexY,
            x=column_indexX,
            yerr=yerr,
            xerr=xerr,
            xlabel=xlabel,
            ylabel=ylabel,
            verbose=verbose,
            labels=labels,
            CI_to_err=CI_to_err,
        )
        if len(y) <= 0:
            sys.stderr.write("**Warning** scatter nothing to plot - skipping\n")
            return figure
        if logx != False:
            if not hasattr(logx, "__call__"):
                logx = numpy.log10
            x, xerr = plotter.apply_scale_function(logx, x, xerr)
        if logy != False:
            if hasattr(logy, "__call__"):
                logy = numpy.log10
            y, yerr = plotter.apply_scale_function(logy, y, yerr)
        if point_labels is not None and point_labels != False:
            if (
                not isinstance(point_labels, numpy.ndarray)
                and type(point_labels) is not list
            ):
                cbar_label, ci = self._get_label_from_hd(point_labels)
                point_labels = numpy.array(self.column(ci))
        if title_correlation and title is None:
            R, p = scipy.stats.pearsonr(x, y)
            title = "R=%.3lf p=%g" % (R, Round_To_n(p, 1, only_decimals=True))
        figure = plotter.scatter(
            x,
            y,
            yerr=yerr,
            xerr=xerr,
            labels=labels,
            xlabel=xlabel,
            ylabel=ylabel,
            point_labels=point_labels,
            title=title,
            save=save,
            figure=figure,
            **kwargs
        )
        return figure

    def cloudplot(
        self,
        column_index,
        separate_x_according_to=None,
        logy=False,
        xlabel=None,
        ylabel=None,
        labels=False,
        xlabels=None,
        verbose=True,
        **kwargs
    ):
        """
        cloudplot from one or more columns
          if labels is True keys are used to label each point (may fail with more columns)
        """
        if separate_x_according_to is not None:
            xlabel, _ = self._process_plot_labels(
                separate_x_according_to, xlabel, label=False
            )
        y, _, _, _, _, ylabel, label, labels = self._get_plot_entries(
            column_index,
            x=None,
            yerr=None,
            xerr=None,
            label=True,
            xlabel=xlabel,
            ylabel=ylabel,
            labels=labels,
            verbose=verbose,
        )
        if separate_x_according_to is not None:
            groups = self.group(separate_x_according_to)
            xlabels = list(groups.keys())
            ys = [groups[k].filter_null(column_index)[0][0] for k in groups]
            print(
                ys,
                "xlabels=",
                xlabels,
                "labels=",
                labels,
                "xlabel=",
                xlabel,
                "ylabel=",
                ylabel,
            )
            figure = plotter.cloudplot(
                ys,
                xlabels=xlabels,
                labels=labels,
                xlabel=xlabel,
                ylabel=ylabel,
                **kwargs
            )
            return figure
        if logy != False:
            if hasattr(logy, "__call__"):
                logy = numpy.log10
            y, _ = plotter.apply_scale_function(logy, y, None)
        if not hasattr(y[0], "__len__"):
            y = [y]
        if xlabels is None:
            xlabels = label
        figure = plotter.cloudplot(
            y, xlabels=xlabels, labels=labels, xlabel=xlabel, ylabel=ylabel, **kwargs
        )
        return figure

    def swarmplot(
        self,
        column_index,
        separate_x_according_to=None,
        increment_value=0.03,
        xpos=None,
        add_boxplot=False,
        return_groups=False,
        logy=False,
        xlabel=None,
        ylabel=None,
        labels=None,
        xlabels=None,
        exclude_less_than_points=3,
        verbose=True,
        test=None,
        yerr=None,
        **kwargs
    ):
        """
        swarmplot from one or more columns
          if labels is True keys are used to label each point (may fail with more columns)
        if return_groups :
            return figure,pvs,ys,xlabels,xpos,yerr  (pvs is the pvalue that will be None unless only 2 groups)
        """
        _, _, _, _, cand_xlabel, ylabel, label, xlabels = self._get_plot_entries(
            column_index,
            x=xpos,
            yerr=None,
            xerr=None,
            label=True,
            xlabel=xlabel,
            ylabel=ylabel,
            labels=xlabels,
            verbose=verbose,
        )
        if separate_x_according_to is not None:
            xlabel, _ = self._process_plot_labels(
                separate_x_according_to, xlabel, label=False
            )
        else:
            xlabel = cand_xlabel
        # labels are like flag labels or point labels
        if separate_x_according_to is not None:
            groups = self.group(separate_x_according_to)
            xlabs = []
            ys = []
            xvals, yerrvals = [], []
            for k in groups:
                # yv=groups[k].filter_null(column_index)[0][0]
                yv, xposv, yerrv, _, _, _, _, _ = groups[k]._get_plot_entries(
                    column_index,
                    x=xpos,
                    yerr=yerr,
                    xerr=None,
                    label=True,
                    xlabel=None,
                    ylabel=None,
                    labels=None,
                    verbose=False,
                )
                # if xpos is not None and xpos==separate_x_according_to : xposv=[float(k)]
                if (
                    exclude_less_than_points is not None
                    and len(yv) < exclude_less_than_points
                ):
                    print(
                        " **swarmplot exclude_less_than_points --> discarding group %s as only %d entries"
                        % (k, len(yv))
                    )
                    continue
                xlabs += [k]
                if xposv is not None and not isinstance(xpos, numpy.ndarray):
                    xvs = misc.uniq(xposv)
                    if len(xvs) != 1:
                        print(
                            "WARN in swarmplot given xpos '%s' (len=%d) but for group %s identified multiple %d values from %d! KEEPING FIRST"
                            % (str(xpos), len(xpos), k, len(xvs), len(xposv))
                        )
                    xvals += [xvs[0]]
                ys += [yv]
                if yerr is not None:
                    yerrvals += [yerrv]
            if xlabels is None or (type(xlabels) is bool and xlabels == True):
                xlabels = xlabs
            if yerrvals != []:
                if verbose:
                    print("yerr=", yerr, len(yerrvals))
                yerr = yerrvals
            if not isinstance(xpos, numpy.ndarray) and xvals != []:
                if verbose:
                    print("xpos=", xpos, len(xvals), xvals)
                xpos = xvals
            if verbose:
                print(
                    "Sep len(ys)=",
                    len(ys),
                    "xlabels=",
                    xlabels,
                    "labels=",
                    labels,
                    "xlabel=",
                    xlabel,
                    "ylabel=",
                    ylabel,
                    "xpos=",
                    xpos,
                )
        else:
            y, xpos, yerr, _, _, _, _, _ = self._get_plot_entries(
                column_index,
                x=xpos,
                yerr=yerr,
                xerr=None,
                label=True,
                xlabel=None,
                ylabel=None,
                labels=None,
                verbose=verbose,
            )
            if not hasattr(y[0], "__len__"):
                y = [y]
            if xpos is not None and not hasattr(xpos, "__len__"):
                xpos = [xpos]
            ys = y
            if xlabels is None:
                if type(label) is str:
                    if label != ylabel:
                        xlabels = [label]
                    elif self.key_column_hd_name != "keys" and len(ys) == 1:
                        xlabels = [self.key_column_hd_name]
                    else:
                        xlabels = None
                elif type(label) is list:
                    xlabels = label
            if verbose:
                print(
                    "Sep len(ys)=",
                    len(ys),
                    "xlabels=",
                    xlabels,
                    "labels=",
                    labels,
                    "xlabel=",
                    xlabel,
                    "ylabel=",
                    ylabel,
                    "xpos=",
                    xpos,
                )
        pvs = None
        if test is not None and (
            "mann" in test.lower() or "ttest" in test.lower().replace("-", "")
        ):  # make test of all combos
            pvs = {}  # keys will be tuples of (xlabels[1],xlabels[2])
            print(" entries: %d points: %s" % (len(ys), str([len(yv) for yv in ys])))
            for j, y1 in enumerate(ys):
                for i, y2 in enumerate(ys[j + 1 :]):
                    i += j + 1
                    if "mann" in test.lower():
                        _, p = scipy.stats.mannwhitneyu(y1, y2)
                        print(
                            "mannwhitneyu %s %s pv=%s"
                            % (xlabels[j], xlabels[i], repr(p))
                        )
                        pvs[(xlabels[j], xlabels[i])] = p
                    if "ttest" in test.lower().replace("-", ""):
                        _, p = scipy.stats.ttest_ind(y1, y2)
                        print(
                            "ttest_ind %s %s pv=%s" % (xlabels[j], xlabels[i], repr(p))
                        )
                        if (xlabels[j], xlabels[i]) in pvs:
                            pvs[(xlabels[j], xlabels[i])] = [
                                pvs[(xlabels[j], xlabels[i])],
                                p,
                            ]
                        else:
                            pvs[(xlabels[j], xlabels[i])] = p
        if logy != False:
            if hasattr(logy, "__call__"):
                logy = numpy.log10
            ys, _ = plotter.apply_scale_function(logy, ys, None)
        # if verbose : print('len(ys)=',len(ys),'xlabels=',xlabels,'labels=',labels,'xlabel=',xlabel,'ylabel=',ylabel)
        figure = plotter.swarmplot(
            ys,
            xlabels=xlabels,
            increment_value=increment_value,
            xpos=xpos,
            add_boxplot=add_boxplot,
            labels=labels,
            xlabel=xlabel,
            ylabel=ylabel,
            yerr=yerr,
            **kwargs
        )
        if return_groups:
            return figure, pvs, ys, xlabels, xpos, yerr
        if pvs is not None:
            return figure, pvs
        return figure

    def boxplot(
        self,
        column_index,
        logy=False,
        xlabel=None,
        ylabel=None,
        xlabels=None,
        verbose=True,
        **kwargs
    ):
        """
        boxplot plot between two columns
        """
        y, _, _, _, xlabel, ylabel, label, labels = self._get_plot_entries(
            column_index,
            x=None,
            yerr=None,
            xerr=None,
            label=True,
            xlabel=xlabel,
            ylabel=ylabel,
            verbose=verbose,
        )

        if logy != False:
            if hasattr(logy, "__call__"):
                logy = numpy.log10
            y, _ = plotter.apply_scale_function(logy, y, None)
        if not hasattr(y[0], "__len__"):
            y = [y]
        if xlabels is None:
            xlabels = label
        figure = plotter.boxplot(
            y, xlabels=xlabels, xlabel=xlabel, ylabel=ylabel, **kwargs
        )
        return figure

    def histogram2d(
        self,
        column_indexX,
        column_indexY,
        xlabel=None,
        ylabel=None,
        verbose=True,
        **kwargs
    ):
        """
        scatter plot between two columns
          if labels is True keys are used to label each point
        """
        y, x, yerr, xerr, xlabel, ylabel, label, labels = self._get_plot_entries(
            column_indexY,
            x=column_indexX,
            xlabel=xlabel,
            ylabel=ylabel,
            verbose=verbose,
        )
        """
        if xlabel==True :
            if column_indexX is None : xlabel=self.key_column_hd_name
            elif type(column_indexX) is str : xlabel=column_indexX
            elif type(column_indexX) is int :
                for k in self.hd : 
                    if self.hd[k]==column_indexX : xlabel=k 
        if ylabel==True :
            if column_indexY is None : ylabel=self.column_indexY
            elif type(column_indexY) is str : ylabel=column_indexY
            elif type(column_indexY) is int :
                for k in self.hd : 
                    if self.hd[k]==column_indexY : ylabel=k 
        
        entries,discarded=self.filter_null([column_indexX,column_indexY],fast_float=True)
        if len(discarded)!=0 : sys.stderr.write('Warning discarding %d entries that had Null values\n' % (len(discarded)))
        if column_indexX is not None and column_indexY is not None : x,y=entries[:2]
        elif column_indexX is None :y=entries[0]
        elif column_indexY is None :x=entries[0]
        print len(x),len(y)
        """
        figure = plotter.histogram2d(x, y, xlabel=xlabel, ylabel=ylabel, **kwargs)
        return figure

    def plot_matrix(
        self,
        columns_to_plot=None,
        include_keys=False,
        value_labels=True,
        round_labels=2,
        frame=False,
        xlabels=True,
        ylabels=True,
        x_minor_tick_every=False,
        xlabels_rotation="horizontal",
        figure_size=(8, 8),
        value_labels_size=10,
        **kwargs
    ):
        """
        displays the whole content (or only columns_to_plot if given)
        as a matrix
        """
        if columns_to_plot is None:
            if self.mat is None:
                self.numpymat(include_keys=include_keys)
            mat_to_plot = self.mat
        else:
            mat_to_plot, discarded = self.filter_null(
                columns_to_plot, force_float=False
            )
            if discarded is not None and discarded != []:
                sys.stderr.write(
                    "**WARNING** in plot_matrix with columns_to_plot discarding %d entries from filter_null\n"
                    % (len(discarded))
                )
        if type(value_labels) is bool and value_labels == True:
            if "int" not in str(mat_to_plot.dtype) and round_labels is not None:
                value_labels = numpy.round(mat_to_plot, round_labels)
                if round_labels == 0:
                    value_labels = value_labels.astype("int")
            else:
                value_labels = mat_to_plot
        if type(xlabels) is bool and xlabels == True:
            xlabels = self.HD()
        if type(ylabels) is bool and ylabels == True:
            ylabels = list(self.keys())[::-1]
        f = plotter.plot_matrix(
            mat_to_plot,
            value_labels=value_labels,
            xlabels=xlabels,
            ylabels=ylabels,
            frame=frame,
            x_minor_tick_every=x_minor_tick_every,
            xlabels_rotation=xlabels_rotation,
            figure_size=figure_size,
            value_labels_size=value_labels_size,
            **kwargs
        )
        return f

    def _get_label_from_hd(self, column_ind):
        """
        given a column index or even a string (or 'keys')
        return its HD name (in case of string checks that this is in HD and returns it
        """
        label = None
        if column_ind == "keys":
            label = self.key_column_hd_name
            column_ind = self.key_column_hd_name
        elif type(column_ind) is str:
            if column_ind not in self.hd:
                sys.stderr.write(
                    "WARNING _get_label_from_hd string input %s not actually in HD!\n"
                    % (column_ind)
                )
            label = column_ind
        elif type(column_ind) is int:
            for k in self.hd:
                if column_ind == self.hd[k]:
                    label = k
        return label, column_ind

    def _process_plot_labels(self, y, ylabel, label=None):
        if ylabel is None:  # set default
            if plotter.default_parameters["set_publish"]:
                ylabel = False  # not written so can be added in figure editor
            else:
                ylabel = True
        if (type(ylabel) is bool and ylabel == True) or (
            type(label) is bool and label == True
        ):
            if isinstance(y, numpy.ndarray):
                ylabel = "custom values as nparray"
            if (
                not isinstance(y, numpy.ndarray)
                and type(y) is not str
                and hasattr(y, "__len__")
            ):  # it's a list or tuple, mutiple y values requested.
                lab = []
                if type(y) is tuple:
                    y = list(y)
                for j, y_ind in enumerate(y):
                    l, y[j] = self._get_label_from_hd(y_ind)
                    lab += [l]
                if label is None or (type(label) is bool and label == True):
                    label = lab  # plot label of the different profiles
            else:
                lab, y = self._get_label_from_hd(y)
                if type(ylabel) is bool and ylabel == True:
                    # if plotter.default_parameters['set_publish'] : ylabel=None # not written so can be added in figure editor - amended at beginning
                    # else :
                    ylabel = lab  # written
            if type(label) is bool and label == True:
                label = lab
        if type(ylabel) is bool and ylabel == False:
            ylabel = None  # so it is not written
        if type(label) is bool:
            label = None  # so it is not written
        return ylabel, label

    def _process_plot_vals(self, y, retrieve, indices, indname="y"):
        if y is not None and not isinstance(y, numpy.ndarray):
            if type(y) is list:
                indices[indname] = []
                for y_ind in y:
                    indices[indname] += [len(retrieve)]
                    retrieve += [y_ind]  # y_ind can be a string without problems
            else:
                indices[indname] = len(retrieve)
                retrieve += [y]
        return retrieve, indices

    def _condition_to_overwrite_found(self, new_errk, old_errk, matchk=""):
        # just prioritise erroror over e.g. standard deviation
        if "err" in new_errk.lower() and "err" not in old_errk.lower():
            return True
        elif "err" in old_errk.lower() and "err" not in new_errk.lower():
            return False
        else:

            sys.stderr.write(
                "**WARNING** in _auto_locate_error found two candidates for '%s' and cannot decide old='%s' new='%s' -KEEPING old\n"
                % (matchk, old_errk, new_errk)
            )
            return False

    def _auto_locate_error(
        self, y, yerr, indices, retrieve, errname="yerr", verbose=True
    ):
        # indices is a dict and retrieve is a list - errname is for printing it is either yerr or xerr
        # OLD: err_condition_in_matching_key= lambda cand_err,match_k : 'err' in cand_err.lower() or ('std' in cand_err.lower() and 'std' not in match_k.lower() )
        err_condition_in_matching_key = lambda cand_err, match_k: match_k in cand_err and cand_err.replace(
            match_k, ""
        ).lower().strip().strip(
            "_"
        ).strip(
            "-"
        ) in [
            "std",
            "stdev",
            "stderr",
            "err",
            "sterr",
        ]
        if (
            type(yerr) is bool and yerr == True
        ):  # automatically locate error among columns for plotting variables
            found = False
            if y is not None and not isinstance(y, numpy.ndarray):
                if type(y) is list:
                    l_found, yls = OrderedDict(), []
                    for yind in y:
                        yl, _ = self._get_label_from_hd(yind)
                        yls += [yl]
                        ff = False
                        for k in self.hd:
                            if yl.lower() in k.lower() and err_condition_in_matching_key(
                                k, yl
                            ):
                                if ff:
                                    if k.lower().replace(yl.lower(), "").strip().strip(
                                        "_"
                                    ).strip("-") in [
                                        "std",
                                        "stdev",
                                        "stderr",
                                        "err",
                                        "sterr",
                                    ] and l_found[
                                        yl
                                    ].lower().replace(
                                        yl.lower(), ""
                                    ).strip().strip(
                                        "_"
                                    ).strip(
                                        "-"
                                    ) not in [
                                        "std",
                                        "stdev",
                                        "stderr",
                                        "err",
                                        "sterr",
                                    ]:
                                        l_found[
                                            yl
                                        ] = k  # overwrite as it's a closer match
                                    elif self._condition_to_overwrite_found(
                                        l_found[yl], k, yl
                                    ):
                                        l_found[yl] = k  # prints warning if cant decide
                                else:
                                    l_found[yl] = k
                                ff = True
                        if not ff:
                            l_found[yl] = None

                    if all([l is None for l in list(l_found.values())]):
                        if verbose:
                            sys.stderr.write(
                                "*WARNING* cannot automatically locate %s for any of the variables to plot\n"
                                % (errname)
                            )
                        yerr = None
                    else:
                        yerr = []
                        for yl in l_found:
                            ff = l_found[yl]
                            if verbose:
                                if ff is None:
                                    sys.stderr.write(
                                        "*WARNING* cannot automatically locate %s from %s, setting to None\n"
                                        % (errname, yl)
                                    )
                                else:
                                    sys.stdout.write(
                                        "    %s %s --> %s\n" % (errname, yl, str(ff))
                                    )
                            yerr += [ff]

                else:
                    yl, _ = self._get_label_from_hd(y)
                    for k in self.hd:
                        if yl.lower() in k.lower() and err_condition_in_matching_key(
                            k, yl
                        ):
                            if found != False:
                                if self._condition_to_overwrite_found(found, k, yl):
                                    found = k
                            else:
                                found = k
                    if not found:
                        if verbose:
                            sys.stderr.write(
                                "*WARNING* cannot automatically locate %s from %s, setting to None\n"
                                % (errname, yl)
                            )
                        yerr = None
                    else:
                        yerr = found
                        if verbose:
                            sys.stdout.write(
                                "    %s %s --> %s\n" % (errname, yl, str(yerr))
                            )

        elif (
            type(yerr) is str and yerr.lower() == "ci" and yerr not in self.hd
        ):  # automatically locate confidence interval among columns for plotting variables
            if y is not None and not isinstance(y, numpy.ndarray):
                if type(y) is list:
                    l_found = OrderedDict()
                    for yind in y:
                        yl, _ = self._get_label_from_hd(yind)
                        ff = False
                        for k in self.hd:
                            if (
                                "ci" not in yl.lower()
                                and yl.lower() in k.lower()
                                and "ci" in k.lower()
                            ) or ("CI" not in yl and yl in k and "CI" in k):
                                if yl not in l_found:
                                    l_found[yl] = [None, None]
                                if "up" in k.lower() and "low" not in k.lower():
                                    if l_found[yl][1] is None:
                                        l_found[yl][1] = k
                                    else:
                                        sys.stderr.write(
                                            "*ERROR* more than one candidate upper CI automatically found for %s - [%s and %s] skipping last\n"
                                            % (yl, k, l_found[yl][1])
                                        )
                                elif "low" in k.lower() and "up" not in k.lower():
                                    if l_found[yl][0] is None:
                                        l_found[yl][0] = k
                                    else:
                                        sys.stderr.write(
                                            "*ERROR* more than one candidate lower CI automatically found for %s - [%s and %s] skipping last\n"
                                            % (yl, k, l_found[yl][0])
                                        )
                                else:
                                    sys.stderr.write(
                                        "*ERROR* cannot automatically decide on lower/upper CI found for %s -> CI candidate:%s skipping this CI candidate\n"
                                        % (yl, k)
                                    )
                        if yl not in l_found:
                            l_found[yl] = None
                            # sys.stderr.write('*WARNING* cannot automatically locate %s CI from %s, setting to None\n'%(errname,yl))

                    if all([l is None for l in list(l_found.values())]):
                        if verbose:
                            sys.stderr.write(
                                "*WARNING* cannot automatically locate %s CI for any of the variables to plot\n"
                                % (errname)
                            )
                        yerr = None
                    else:
                        yerr = []
                        for yl in l_found:
                            ff = l_found[yl]  # should be [lower,upper]
                            if ff is None:
                                if verbose:
                                    sys.stderr.write(
                                        "*WARNING* cannot automatically locate %s CI from %s, setting to None\n"
                                        % (errname, yl)
                                    )
                            elif None in ff:
                                if verbose:
                                    sys.stderr.write(
                                        "*WARNING* cannot automatically locate either upper or lower %s CI from %s [located %s], setting both to None\n"
                                        % (errname, yl, str(ff))
                                    )
                                ff = None
                            yerr += [ff]
                else:
                    yl, _ = self._get_label_from_hd(y)
                    up, low = None, None
                    for k in self.hd:
                        if (
                            "ci" not in yl.lower()
                            and yl.lower() in k.lower()
                            and "ci" in k.lower()
                        ) or ("CI" not in yl and yl in k and "CI" in k):
                            if "up" in k.lower() and "low" not in k.lower():
                                if up is None:
                                    up = k
                                else:
                                    sys.stderr.write(
                                        "*ERROR* more than one candidate upper CI automatically found for %s - [%s and %s] skipping last\n"
                                        % (yl, k, up)
                                    )
                            elif "low" in k.lower() and "up" not in k.lower():
                                if low is None:
                                    low = k
                                else:
                                    sys.stderr.write(
                                        "*ERROR* more than one candidate lower CI automatically found for %s - [%s and %s] skipping last\n"
                                        % (yl, k, low)
                                    )
                            else:
                                sys.stderr.write(
                                    "*ERROR* cannot automatically decide on lower/upper CI found for %s -> CI candidate:%s skipping this CI candidate\n"
                                    % (yl, k)
                                )
                    if up is None or low is None:
                        sys.stderr.write(
                            "*ERROR* cannot automatically locate %s CI from %s, setting to None [located low=%s up=%s]\n"
                            % (errname, yl, str(low), str(up))
                        )
                        yerr = None
                    else:
                        yerr = [low, up]
                        if verbose:
                            sys.stdout.write(
                                "    %s %s --> %s\n" % (errname, yl, str(yerr))
                            )
        yerr_ci = False  # note that all the above determines yerr
        if yerr is not None and not isinstance(yerr, numpy.ndarray):
            if type(yerr) is list or type(yerr) is tuple:
                # print('DEB2: ',errname,yerr,'islist:',  type(yerr) is list,'istuple:', type(yerr) is tuple,'len:', len(yerr),'isStr or len<2:',( type(y) is str or len(y)<2), end=' ')

                if (len(yerr) == 2 and (type(y) is str or len(y) < 2)) or (
                    isinstance(y, numpy.ndarray) and len(y.shape) == 1
                ):  # confidence interval (y at this stage should be header keys)
                    indices[errname] = len(retrieve)
                    retrieve += [yerr[0], yerr[1]]
                    yerr_ci = True  # confidence interval
                else:
                    indices[errname] = []
                    ci_list = []
                    for y_inderr in yerr:
                        if (type(y_inderr) is list or type(y_inderr) is tuple) and len(
                            y_inderr
                        ) == 2:  # confidence interval
                            indices[errname] += [len(retrieve)]
                            retrieve += [y_inderr[0], y_inderr[1]]
                            ci_list += [True]
                        else:
                            indices[errname] += [len(retrieve)]
                            retrieve += [y_inderr]
                            ci_list += [False]

                    if any(ci_list):
                        yerr_ci = ci_list
                # print("DEB2:",errname," err_ci=",yerr_ci)
            else:
                indices[errname] = len(retrieve)
                retrieve += [yerr]
        return retrieve, indices, yerr_ci, yerr

    def _get_err_for_plot(
        self, all_entries, indices, y, retrieve, yerr_ci, ind_key="yerr"
    ):
        try:
            if yerr_ci != False:
                if type(yerr_ci) is list:
                    yerr = []
                    for j, ind in enumerate(indices[ind_key]):
                        if retrieve[ind] is None:
                            yerr += [None]
                        elif yerr_ci[j]:
                            low, up = all_entries[ind], all_entries[ind + 1]
                            if all(low <= y) and all(up >= y):
                                yerr += [
                                    [y - low, up - y]
                                ]  # assumes these are the values of the confidence interval
                            else:
                                yerr += [
                                    [low, up]
                                ]  # assumes these are the distances to the confidence interval
                        else:
                            yerr += [all_entries[ind]]

                else:
                    low, up = (
                        all_entries[indices[ind_key]],
                        all_entries[indices[ind_key] + 1],
                    )
                    # print "DEB3: all(low<=y) , all(up>=y)",all(low<=y) ,all(up>=y),low<=y,up>=y
                    mask = (~numpy.isnan(y)) & (~numpy.isnan(low)) & (~numpy.isnan(up))
                    if all((low <= y)[mask]) and all((up >= y)[mask]):
                        yerr = [
                            y - low,
                            up - y,
                        ]  # assumes these are the values of the confidence interval
                    else:
                        yerr = [
                            low,
                            up,
                        ]  # assumes these are the distances to the confidence interval
            else:
                yerr = []
                if type(indices[ind_key]) is int:
                    yerr = all_entries[indices[ind_key]]
                else:
                    for j, ind in enumerate(indices[ind_key]):
                        if retrieve[ind] is None:
                            yerr += [None]
                        else:
                            yerr += [all_entries[ind]]
        except Exception:
            print(
                "\nEXCEPTION: yerr_ci=",
                yerr_ci,
                "ind_key=",
                ind_key,
                "indices",
                indices,
                end=" ",
            )
            if isinstance(all_entries, numpy.ndarray):
                print("all_entries.shape=", all_entries.shape)
            raise
        return yerr

    def _get_plot_entries(
        self,
        y,
        x=None,
        yerr=None,
        xerr=None,
        xlabel=None,
        ylabel=None,
        label=None,
        labels=None,
        CI_to_err=False,
        verbose=True,
    ):

        # if type(xerr) is bool and xerr==True :
        #    xl,_=self._get_label_from_hd(x)
        #    found=False
        #    for k in self.hd :
        #        if xl.lower() in k.lower() and 'err' in k.lower() : found=k
        #    if not found :
        #        sys.stderr.write('*WARNING* cannot automatically locate xerr from %s, setting to None\n'%(xl))
        #        xerr=None
        #    else :
        #        sys.stdout.write(' xerr --> %s\n'%(str(xerr)))
        #        xerr=found

        xlabel, _ = self._process_plot_labels(x, xlabel, label=False)
        ylabel, label = self._process_plot_labels(y, ylabel, label=label)

        if labels is not None and (
            labels == self.key_column_hd_name or labels == "keys"
        ):
            labels = True
        elif type(labels) is bool and labels == False:
            labels = None
        retrieve = []
        indices = {}
        retrieve, indices = self._process_plot_vals(y, retrieve, indices, indname="y")
        retrieve, indices = self._process_plot_vals(x, retrieve, indices, indname="x")
        retrieve, indices, xerr_ci, xerr = self._auto_locate_error(
            x, xerr, indices, retrieve, errname="xerr", verbose=verbose
        )
        retrieve, indices, yerr_ci, yerr = self._auto_locate_error(
            y, yerr, indices, retrieve, errname="yerr", verbose=verbose
        )

        if retrieve == []:
            sys.stderr.write(
                "**ERROR** _get_plot_entries() in trying to retrive empty list, check your input to the plotting function!\n"
            )
            return
        if verbose:
            print("   csv_profile retrieve,indices:", retrieve, indices)
            print("   x:", x, "y:", y)
        all_entries, discarded = self.filter_null(retrieve)
        if verbose and isinstance(all_entries, numpy.ndarray):
            print("   all_entries.shape:", all_entries.shape)
        if len(discarded) != 0:
            sys.stderr.write(
                "Warning _get_plot_entries() discarding %d entries that had Null values\n"
                % (len(discarded))
            )
            if len(discarded) < 6:
                sys.stderr.write("%s\n" % (str(discarded)))
        if "x" in indices:
            if type(indices["x"]) is list:
                x = [all_entries[indx] for indx in indices["x"]]
            else:
                x = all_entries[indices["x"]]
        if "y" in indices:
            if type(indices["y"]) is list:
                y = [all_entries[indy] for indy in indices["y"]]
            else:
                y = all_entries[indices["y"]]
        if "xerr" in indices:
            xerr = self._get_err_for_plot(
                all_entries, indices, x, retrieve, xerr_ci, ind_key="xerr"
            )
        if "yerr" in indices:
            yerr = self._get_err_for_plot(
                all_entries, indices, y, retrieve, yerr_ci, ind_key="yerr"
            )
        if labels is not None and (
            type(labels) is str or not hasattr(labels, "__len__")
        ):  # not a dict or a list itself - if key_column it has been set to true previously
            if (
                len(discarded) != 0
            ):  # note that we cannot include labels in filter_null as labels may not be numbers
                l = []
                for k in self:
                    if k not in discarded:
                        if type(labels) is str:
                            l += [self[k][self.hd[labels]]]
                        elif type(labels) is int:
                            l += [self[k][labels]]
                        elif labels == True:
                            l += [k]
                labels = l[:]
            else:
                if type(labels) is str:
                    labels = self.column(labels)
                elif labels == True:
                    labels = list(self.keys())
        # print "DEB: yerr_ci,CI_to_err,yerr:",yerr_ci,CI_to_err,yerr,len(yerr),
        #  the below is already done apparently
        # if yerr is not None and yerr_ci and CI_to_err :
        #    if hasattr(y[0], '__len__') :
        #        for j,prof in enumerate(y) :
        #            if yerr[j] is not None and len(yerr[j])==2 : yerr[j]=[ prof-yerr[j][0], yerr[j][1]-prof ]
        #    else :
        #        if len(yerr)==2 : yerr=[ y-yerr[0], yerr[1]-y ]
        # print '->',yerr
        return y, x, yerr, xerr, xlabel, ylabel, label, labels

    def _CI_to_err(self, ci_down_index, ci_up_index, values=None):
        ents, discarded = self.filter_null([ci_down_index, ci_up_index])
        if len(discarded) != 0:
            sys.stderr.write(
                "Warning _CI_to_err() discarding %d entries that had Null values\n"
                % (len(discarded))
            )
        ci_down, ci_up = ents
        if values is not None:
            return numpy.array(values) - ci_down, ci_up - numpy.array(values)
        return ci_down, ci_up

    def filter_null(self, column_index, remove=[None, ""], force_float=True):
        """
        return [numpy.array(out_mat)],discarded
        returns the column corresponding to column index after removing entries that cannot be converted to numbers
        if more than one index is given, say N indices (or str in hd) then returns N columns of the same length obtained removing entries at positions
         where at least one entry in one column could not be converted to a number.
        """
        doing_slice = False
        out_mat = []
        discarded = []
        if type(column_index) is list or type(column_index) is tuple:
            cind = []
            for c in column_index:
                if c not in remove:  # remove eventual null entries

                    if c in self.hd:
                        cind += [self.hd[c]]
                    elif c == self.key_column_hd_name or c == "keys":
                        cind += ["K"]
                    elif type(c) is int:
                        cind += [c]
                    else:
                        sys.stderr.write(
                            "**ERROR** IN filter_null requesting column '%s' but not found - skipping this column\n"
                            % (c)
                        )
                        sys.stderr.flush()
                    # else :
                    #    sys.stderr.write("**ERROR** IN filter_null requesting column '%s' but type not recognized - skipping this column\n" % (repr(c)))
                    #    sys.stderr.flush()
            column_index = cind[:]
            doing_slice = True
            if len(column_index) == 1:  # this could happen
                column_index = column_index[0]
                doing_slice = False
        else:
            if type(column_index) is str:
                if column_index not in self.hd and (
                    column_index == self.key_column_hd_name or column_index == "keys"
                ):  # just get keys that can be converted to float
                    for k in self:
                        en, ok = convert_to_number(k, force_float=force_float)
                        if ok:
                            out_mat += [en]
                        else:
                            discarded += [en]
                    return numpy.array(out_mat), discarded
                column_index = self.hd[column_index]

        for key in self:
            if self.allow_more_entries_per_key and type(self[key][0]) is list:
                for row in self[key]:
                    if doing_slice:
                        Len = []
                        for ind in column_index:
                            if ind == "K":
                                en = key
                            else:
                                en = row[ind]
                            en, ok = convert_to_number(en, force_float=force_float)
                            if not ok:
                                break
                            Len += [en]
                        if not ok:
                            discarded += [key]
                            continue
                        out_mat += [Len]
                    else:
                        en, ok = convert_to_number(
                            row[column_index], force_float=force_float
                        )
                        if not ok:
                            discarded += [key]
                            continue
                        out_mat += [en]
            else:
                if doing_slice:
                    Len = []
                    for ind in column_index:
                        if ind == "K":
                            en = key
                        else:
                            en = self[key][ind]
                        en, ok = convert_to_number(en, force_float=force_float)
                        if not ok:
                            break
                        Len += [en]
                    if not ok:
                        discarded += [key]
                        continue  # continue outer loop, this Len is never added
                    out_mat += [Len]
                else:
                    en, ok = convert_to_number(
                        self[key][column_index], force_float=force_float
                    )
                    if not ok:
                        discarded += [key]
                        continue
                    out_mat += [en]
        if doing_slice:
            out_mat = numpy.array(out_mat)
            if len(out_mat.shape) == 3:
                print(
                    "filter_null DOING slice of 3D array intial shape %s"
                    % (str(out_mat.shape))
                )
                return (
                    numpy.transpose(out_mat, axes=[1, 0, 2]),
                    discarded,
                )  # default T of 2D is [1,0] but not of 3d
            return numpy.array(out_mat).T, discarded
        return [numpy.array(out_mat)], discarded

    def filter_nullOLD(self, column_index_s, fast_float=False, remove=[None, ""]):
        """
        returns the column corresponding to column index after removing entries equalt to some staff in remove
        if more than one index is given, say N indices (or str in hd) then returns N columns of the same length obtained removing entries at positions
         where at least one entry in one column was equal to some staff in remove.
        fast_float will ignore the input in remove and will just return whatever can be converted to float. It is much faster
        """

        if type(column_index_s) is not list and type(column_index_s) is not tuple:
            column_index_s = [column_index_s]
        cis = []
        for ci in column_index_s:
            if ci is None:
                continue
            if type(ci) is str:
                cis += [self.hd[ci]]
            else:
                cis += [ci]
        column_index_s = numpy.array(cis, "int")
        discarded = []
        if fast_float:
            out_mat = numpy.zeros(len(column_index_s))
            for k in self:
                tmp = numpy.array(self[k])[column_index_s]
                try:
                    tmp = tmp.astype("f")
                    out_mat = numpy.vstack((out_mat, tmp))
                except Exception:
                    pass
            return out_mat[1:, :].T, None
        else:
            out_mat = None
            for j, k in enumerate(self):
                tmp = numpy.array(self[k])
                if len(tmp.shape) > 1:  # it is a list of lists
                    tmp = tmp[:, column_index_s]
                    for line in tmp:
                        discard = False
                        for r in remove:
                            if r in list(line):
                                discard = True
                        if discard:
                            if k not in discarded:
                                discarded += [k]
                        elif out_mat is None:
                            out_mat = line
                        else:
                            out_mat = numpy.vstack((out_mat, line))
                else:
                    tmp = tmp[column_index_s]
                    discard = False
                    for r in remove:
                        if r in list(tmp):
                            discard = True
                    if discard:
                        discarded += [
                            k
                        ]  # for some reason None in numpy.array([1,2,3,None,4]) returns False
                    elif out_mat is None:
                        out_mat = tmp
                    else:
                        out_mat = numpy.vstack((out_mat, tmp))
        # print len(self),out_mat.shape
        return out_mat.T, discarded

    def stats(self, column_index, use_weights_from_column=None, stats_name=None):
        """
        return a stats class with entries that are the values (converted to float) contained in column column_index
        and ids that are the keys
        if use_weights_from_column weights are extracted from this column 
        """
        if type(column_index) is str:
            if stats_name is None:
                stats_name = column_index
            column_index = self.hd[column_index]
        elif stats_name is None:
            stats_name = "C:" + str(column_index)
        st = structs.Stats(stats_name)
        if use_weights_from_column is not None and use_weights_from_column != False:
            if type(use_weights_from_column) is str:
                weight_column = self.hd[use_weights_from_column]
                use_weights_from_column = True
            else:
                weight_column = use_weights_from_column
                use_weights_from_column = True
        else:
            use_weights_from_column = False
        w = None
        for key in self:
            if type(self[key]) is list or type(self[key]) is tuple:
                if type(self[key][0]) is list:
                    for row in self[key]:
                        en, ok = convert_to_number(row[column_index], force_float=True)
                        if not ok:
                            en = None
                        if use_weights_from_column:
                            w, ok = convert_to_number(
                                row[weight_column], force_float=True
                            )
                            if not ok:
                                w = None
                        st.update(en, key, w)
                else:
                    en, ok = convert_to_number(
                        self[key][column_index], force_float=True
                    )
                    if not ok:
                        en = None
                    if use_weights_from_column:
                        w, ok = convert_to_number(
                            self[key][weight_column], force_float=True
                        )
                        if not ok:
                            w = None
                    st.update(en, key, w)
            else:
                en, ok = convert_to_number(self[key], force_float=True)
                if not ok:
                    en = None
                st.update(en, key, None)
        st.finalize()
        return st

    def column(
        self,
        column_index,
        return_keys=False,
        return_list_of_lists_position=False,
        group_by=None,
        return_stats_for_groups=True,
    ):
        """
        #TO MANAGE DICTIONARIES THAT ARE MADE OF LIST OF LISTS or that contains lists, they should correspond to csv file, each list is a row. Note that in this function column_number STARTS FROM 1 not from 0 (like in the file). Note that one of the columns in the file will probabily (depending on how the dictionary was built) be the key columns, so be careful when giving column_number. it returns (keys,column,column1,column2) in tuple or just column (as a list) if none of the other is given.
        #if you have a dictionary header to read through the file (i.e. a dictionary that has the column name as a title and the coulm number as an entry: see csvToDictionary) you can give it in header and you can give to the column_number(s) the entry of the header in string format
        # if the dictionary has multiple entries per key (that is the same key corresponds to multiple raws in the original file) than it will be a list of lists. One can set return_list_of_lists_position to True in order to have a list with the original position in the dictionary returned as well.
        # column_index can also be a list of ids to extract, if you want to do a slice give range(slice_start,slice_end)
        # if you give a group_by key or index 
        #  it returns groups,groups_key,cases
        #  if return_stats_for_groups
        #  it returns groups,groups_key,cases,stats_list (where stats list contains Stats classes for every group in the other two lists
        #  where groups is a list of lists containing all the elements of the selected column(s), each inner list is a group of elements that have the same group_by entry.
        #  groups_key are the keys corresponding to the elements in this new sorting that arises
        #  cases are the different possibilities found in the group_by column. The length of cases must be the same as the number of inner lists in groups and groups_key.
        #  the order is also the same so the entries in group[0] correspond to the group_by cases[0] possibility
        """
        # first check everything
        if return_list_of_lists_position:
            return_keys = True
        doing_slice = False
        if type(column_index) is list or type(column_index) is tuple:
            column_index = [
                c for c in column_index if c is not None
            ]  # remove eventual null entries
            if column_index[0] in self.hd:
                column_index = [self.hd[c] for c in column_index]
            max_id = max(column_index)
            doing_slice = True
            if len(column_index) == 1:  # this could happen
                column_index = column_index[0]
                doing_slice = False
        else:
            if type(column_index) is str:
                if column_index not in self.hd and (
                    column_index == self.key_column_hd_name or column_index == "keys"
                ):
                    try:
                        return list(map(float, list(self.keys())))  # Big assumption
                    except:
                        return list(self.keys())
                column_index = self.hd[column_index]
            elif column_index in self.hd:
                column_index = self.hd[column_index]
            max_id = column_index
        grouping = False
        if group_by is not None:
            if type(group_by) is str:
                group_by = self.hd[group_by]
            grouping = True
            clusterer = []
        column = []
        keys = []
        list_of_lists_position = []
        for key in self:
            if (
                type(self[key]) is list
                or type(self[key]) is tuple
                or isinstance(self[key], numpy.ndarray)
            ):
                if self.allow_more_entries_per_key and type(self[key][0]) is list:
                    for row in self[key]:
                        if max_id < len(row):
                            if doing_slice:
                                column += [[row[i] for i in column_index]]
                            else:
                                column += [row[column_index]]
                            if grouping:
                                clusterer += [row[group_by]]
                            if return_keys or grouping:
                                keys += [
                                    key
                                ]  # this way we have another list, of the same length as column, which contains the corresponding keys in the proper order
                            if (
                                return_list_of_lists_position
                            ):  # this way we have another list, of the same length as column, which contains the corresponding position, one can than retrive precisely the same line with [key][position].
                                list_of_lists_position += [self[key].index(row)]
                        else:
                            khd = ""
                            if self.hd is not None:
                                khd = self.HD()[column_index]
                            sys.stderr.write(
                                "***ERROR*** in Data.column() column_index %d out of range (key_hd=%s max_id=%d found %d columns in row corresponding to key %s) - Row:\n%s\n"
                                % (column_index, khd, max_id, len(row), key, str(row))
                            )
                            raise IndexError
                    del row
                else:
                    if max_id < len(self[key]):
                        if doing_slice:
                            column += [[self[key][i] for i in column_index]]
                        else:
                            column += [self[key][column_index]]
                        if grouping:
                            clusterer += [self[key][group_by]]
                        if return_keys or grouping:
                            keys += [
                                key
                            ]  # this way we have another list, of the same length as column, which contains the corresponding keys in the proper order
                        # CHECK THE FOLLOWING TWO LINES
                        if (
                            return_list_of_lists_position
                        ):  # this way we have another list, of the same length as column, which contains the corresponding position, one can than retrive precisely the same line with [key][position].
                            list_of_lists_position += [0]
                    else:
                        khd = ""
                        if self.hd is not None:
                            khd = self.HD()[column_index]
                        sys.stderr.write(
                            "***ERROR*** in Data.column() column_index %d out of range (key_hd=%s max_id=%d found %d columns in row corresponding to key %s)"
                            % (column_index, khd, max_id, len(self[key]), key)
                        )
                        raise IndexError
            else:
                raise Exception(
                    "***ERROR*** in column dictionary given is not a list of lists and its entries are not lists"
                )
                return 1
        if grouping:
            cases = misc.uniq(clusterer)
            groups = [[] for i in range(len(cases))]
            groups_key = [[] for i in range(len(cases))]
            for j, c in enumerate(clusterer):
                ind = cases.index(c)
                groups[ind].append(column[j])
                groups_key[ind].append(keys[j])
            if return_stats_for_groups:
                stats_list = []
                for j, c in enumerate(cases):
                    stats_list += [structs.Stats(c)]
                    stats_list[-1].update_list(groups[j], groups_key[j])
                    stats_list[-1].finalize()
                return groups, groups_key, cases, stats_list
            return groups, groups_key, cases

        if doing_slice:  # convert to list of columns
            column = list(zip(*column))
            column = [list(tup) for tup in column]
        if not return_keys:
            del keys, list_of_lists_position
            return column
        to_return = tuple([column])
        if return_keys:
            to_return += tuple([keys])
        if return_list_of_lists_position:
            to_return += tuple([list_of_lists_position])
        return to_return

    def unique(self, column_index, split_multiples_at=None):
        """
        it returns a list of the unique entries in the specified column
        split_multiples_at can be a string of delimiters (like string.punctuation='\'!"#$%&\\\'()*+,-./:;<=>?@[\\\\]^_`{|}~\''
        , so that every entry is split at those points and uniqueness is updated consequently.
        This is implemented in a fast way that will remove any empty space!
        """
        column = self.column(
            column_index, return_keys=False, return_list_of_lists_position=False
        )
        unique_entries = []
        if split_multiples_at != None:
            T = str.maketrans(split_multiples_at, " " * len(split_multiples_at))
            for el in column:
                el = str.translate(el, T).split()
                for e in el:
                    if e not in unique_entries:
                        unique_entries += [e]
        else:
            for el in column:
                if el not in unique_entries:
                    unique_entries += [el]
        return unique_entries

    # print content to file
    def Print(
        self,
        filename=sys.stdout,
        DELIMITER="\t",
        key_column_hd_name="keys",
        key_column=1,
        append=False,
        round=None,
        only_top=None,
        convert_list_to_str=True,
        HEADER=None,
        only_columns=None,
    ):
        if round is not None:
            data2 = self.copy(deep=True)
            data2.round_all(round)
            return data2.Print(
                filename=filename,
                DELIMITER=DELIMITER,
                key_column_hd_name=key_column_hd_name,
                key_column=key_column,
                append=append,
                only_top=only_top,
                round=None,
                only_columns=only_columns,
                convert_list_to_str=convert_list_to_str,
            )
        print_tex = False
        if key_column_hd_name == "keys" or key_column_hd_name is None:
            if self.key_column_hd_name is not None:
                key_column_hd_name = self.key_column_hd_name
        if filename is None:
            filename = self.filename
        if filename is None:
            raise Exception("No filename declared! (filename is None)")
        if type(filename) is str and ".csv" in filename and DELIMITER is not None:
            DELIMITER = None
            print(
                "Printig csv file %s with %d entries"
                % (filename.split("/")[-1], len(self))
            )
        elif type(filename) is str and ".tex" in filename or "tex" in DELIMITER:
            print("Printig tex table", filename.split("/")[-1])
            print_tex = True
        if print_tex:
            print_latex(
                filename,
                self.filter_columns(columns_to_keep=only_columns),
                key_column=key_column,
                DELIMITER=DELIMITER,
                key_column_header_name=key_column_hd_name,
                convert_float_list_to_str=convert_list_to_str,
                HEADER=HEADER,
            )
        else:
            csvPrintDictionary(
                filename,
                self.filter_columns(columns_to_keep=only_columns),
                key_column=key_column,
                DELIMITER=DELIMITER,
                key_column_header_name=key_column_hd_name,
                only_top=only_top,
                convert_float_list_to_str=convert_list_to_str,
                append=append,
                allow_more_entries_per_key=self.allow_more_entries_per_key,
                HEADER=HEADER,
            )
        if self.filename is None and type(filename) is str:
            self.filename = filename
        return


def convert_to_number(
    string,
    force_float=False,
    allow_py3_underscores=False,
    convert_to_bool={
        "True": True,
        "False": False,
        "true": True,
        "false": False,
        "TRUE": True,
        "FALSE": False,
    },
):
    """
    this function check if a string is an int or a float and it returns a tuple in the form
    converted_string,bool. Bool is True if the sting has been converted, False if the  string is still in string format.
    the function is quite slow
    """
    if not allow_py3_underscores and type(string) is str and "_" in string:
        return string, False
    if type(string) is list or isinstance(
        string, numpy.ndarray
    ):  # slow but don't expect many entries in this format..
        c = list(
            zip(
                *[
                    convert_to_number(
                        a, force_float=force_float, convert_to_bool=convert_to_bool
                    )
                    for a in string
                ]
            )
        )
        return c[0], all(c[1])
    if convert_to_bool is not None and string in convert_to_bool:
        return convert_to_bool[string], False
    if string in ["INF", "NAN"]:
        return string, False  # often these are amino acids triplets such as INF and NAN
    if force_float:
        try:
            return float(string), True
        except ValueError:
            return string, False
    try:
        return int(string), True
    except ValueError:
        try:
            # if float(string)==int(float(string)) : return int(float(string)),True
            return float(string), True
        except ValueError:
            return string, False


def csvToDictionary(
    filename,
    key_column=1,
    allow_more_entries_per_key=False,
    DELIMITER=None,
    HEADER=None,
    double_header=False,
    double_header_merger="_",
    convert_key_to_number=False,
    auto_convert_to_number=True,
    overwrite_warning=True,
    auto_convert_profile_delimiter=None,
    skip_begins=["#", "@", "\n"],
    get_only_column_id=None,
    equate_to_same_length=True,
    merge_longer=False,
):
    """
    this function reads a file in the csv format and transfer its content to a dictionary which is returned 
    (as a dictionary of lists or of lists of lists if allow_more_entries_per_key=True), 
    if header is not None a tuple (dictionary, header) is returned where header is the first line of the file. 
    The header can have type list or dict, in the last case the keys are the column names and the content is the column number 
    (note that the key_column is skipped). The keys for the dictionary are the entries found in the column key_column (default the first one) 
    in the file. It is also possible to set allow_more_entries_per_key to True. 
    In the latter case if the same key appears more than once in the specified column than the corresponding entries becomes a list of lists
      where every list is a line of the file corresponding to that key (in the order they appear in the file). Note that 
    if allow_more_entries_per_key is left to False then if the key appears more than once only the entry corresponding to the last appearance is stored.
    get_only_column_id if given has to be a list, if e.g. get_only_column_id=[2,3] only columns 2 and 3 are saved in the dictionary
    equate_to_same_length is useful as sometimes you have entries with lower length if they miss info at the end.
    double_header True indicates a double header that can be merged with double_header_merger,
     can also be an integer for many header lines, in which case for e.g. 3 line it should be double_header=2 (indicating that 2 are extra)
    """
    key_column_hd_name = None
    try:
        filein = open(filename, "rU")
    except:
        sys.stderr.write("\n***ERROR*** cannot open file %s\n" % (filename))
        sys.stderr.flush()
        raise IOError
    if DELIMITER is None:
        data = csv.reader(filein)
    else:
        data = csv.reader(filein, delimiter=DELIMITER, skipinitialspace=True)
    dictionary = Data()
    if key_column == 0:
        key_column = (
            1  # coded with position , not index (we assume 0 means first column)
        )
    double_header = int(
        double_header
    )  # so if True it's one extra row otherwise could be more.
    if HEADER is not None:
        if double_header > 0:
            first_head = next(data)
            i = 1
            fk, lastk = "", ""
            while i < double_header:
                lastk = ""
                last_new = ""
                templist = next(data)
                for j, fk in enumerate(first_head[:]):
                    if templist[j] != "" and templist[j] != " ":
                        last_new = templist[j]
                    if fk != "" and fk != " ":
                        lastk = fk
                    if lastk != "" and lastk != " ":
                        first_head[j] = lastk + double_header_merger + last_new
                i += 1
            # print ("DEB:",first_head,"| fk='%s' lastk ='%s' |"%(fk,lastk))
        templist = next(data)  # save first line in templist
        # while len(templist)==0 or (templist[0]=='' and '' in skip_begins) or templist[0][0] in skip_begins : templist=data.next() #save first line in templist
        if double_header > 0:  # combine the two headers into one
            if len(first_head) != len(templist):
                sys.stderr.write(
                    "\n***Potential Warn*** in load double_header len(first_header)!=len(second_header) %d %d\n"
                    % (len(first_head), len(templist))
                )
            lastk = ""
            for j, fk in enumerate(first_head):
                if fk != "":
                    lastk = fk
                if lastk != "":
                    templist[j] = lastk + double_header_merger + templist[j]
        if type(key_column) is str:
            if key_column in templist:
                key_column_hd_name = key_column
                key_column = templist.index(key_column) + 1  # also convert to int
            else:
                raise Exception(
                    "***ERROR*** can't find key_column %s in file %s\n\n"
                    % (key_column, filename)
                )

        if key_column is not None:
            key_column_hd_name = templist[key_column - 1]
            templist[key_column - 1 : key_column] = []  # remove the key column
        # print "DEB templist:"
        # print templist
        # print dictionary.hd
        for i in range(0, len(templist)):
            if templist[i] in dictionary.hd:
                sys.stderr.write("WARN %s already found in header," % (templist[i]))
                templist[i] = new_key_for_dict(dictionary.hd, templist[i], start_at=0)
                sys.stderr.write(" changing to %s in class header\n" % (templist[i]))
            dictionary.hd[
                templist[i]
            ] = i  # so that to the column name returns the column id (postion) in the dictionary of lists that will be returned
    if type(HEADER) is list:
        HEADER = templist  # save first line in header
        while HEADER == [""]:
            HEADER = next(data)  # remove empty lines at top if present
        if key_column is not None:
            if key_column_hd_name is not None:
                key_column_hd_name = HEADER[key_column - 1]
            HEADER[key_column - 1 : key_column] = []  # remove the key column
    elif type(HEADER) is dict:  # usually given as {}
        HEADER = dictionary.hd.copy()

    # adjust keep only option to index and key_column
    if get_only_column_id is not None:
        if key_column in get_only_column_id:
            get_only_column_id.remove(key_column)
        for i, l in enumerate(get_only_column_id):
            if key_column is not None:
                if l < key_column and l - 1 >= 0:
                    get_only_column_id[i] = l - 1
                elif l > key_column and l - 2 >= 0:
                    get_only_column_id[i] = l - 2
            else:
                get_only_column_id[i] = l - 1

    for j, line in enumerate(data):
        if line == []:
            continue
        if line[0] != "" and line[0][0] in skip_begins:
            continue  # without first condition line[0][0] raise error
        # if line[0]=='' and len(line)>1 : line=line[1:]
        if key_column is not None:
            key = line[key_column - 1]
            if key == "":
                continue
            line[(key_column - 1) : key_column] = []  # delete the key from the entry
        else:
            key = j
        if get_only_column_id is not None:
            line = [l for i, l in enumerate(line) if i in get_only_column_id]
        if convert_key_to_number:
            key, _ = convert_to_number(key)
        if auto_convert_to_number:
            line = [convert_to_number(l)[0] for l in line]
            if auto_convert_profile_delimiter is not None:
                for i, l in enumerate(line):
                    if type(l) is str and auto_convert_profile_delimiter in l:
                        profile = separated_str_to_list(
                            l, float_line_delimiter=auto_convert_profile_delimiter
                        )
                        line[i] = profile

        if (
            allow_more_entries_per_key
        ):  # line are stored in a matrix-like format, so that for every key there can be more than one entry ordered in the way they appear in the file [list of lists]
            if key in dictionary:
                dictionary[key] += [line]
            else:
                dictionary[key] = [line]
        else:  # lines are stored as a list of the entries in the order they appear in the file with the key removed
            if overwrite_warning and key in dictionary:
                overwrite_warning += 1
                sys.stderr.write(
                    "**Warn while loading %s. Overwriting entry for %s\n"
                    % (filename, key)
                )
                if overwrite_warning > 10:
                    sys.stderr.write(
                        "*****   ==> Suppressing further overwrite warnings from file %s\n"
                        % (filename)
                    )
                    overwrite_warning = False
                sys.stderr.flush()
            dictionary[key] = line
    filein.close()
    if equate_to_same_length:
        dictionary = equate_data_to_same_length(
            dictionary, header=HEADER, merge_longer=merge_longer
        )
    # del line
    del data
    if HEADER != None:
        return dictionary, HEADER, key_column_hd_name
    else:
        return dictionary


def equate_data_to_same_length(
    data_dict,
    header=None,
    print_warn_for_shorter=True,
    merge_longer=False,
    insert_for_missings="",
    nmax_warn=10,
):
    """
    # if all the entries don't have the same length than empty entries '' are added at the end of the short ones
    merge_longer merges together the last entries (found in lines longer than HD) so that the end results is as long as HD
    """
    reflen = -99
    nwarns = 0
    if header is not None and len(header) > 0:
        reflen = len(header)
    if reflen < 0:
        for k in data_dict:
            if len(data_dict[k]) > reflen:
                reflen = len(data_dict[k])
    for k in data_dict:
        n = len(data_dict[k])
        if n < reflen:
            if print_warn_for_shorter:
                if type(nmax_warn) is int and nwarns < nmax_warn:
                    if "filename" in dir(data_dict) and data_dict.filename is not None:
                        sys.stderr.write(
                            "WARN in equate_data_to_same_length fname='%s' length of header is %d but found a shorter row (key=%s) that is %d long. Adding empties %s at the end (line=%s)\n"
                            % (
                                str(data_dict.filename),
                                reflen,
                                k,
                                n,
                                str(insert_for_missings),
                                data_dict[k],
                            )
                        )
                    else:
                        sys.stderr.write(
                            "WARN in equate_data_to_same_length! length of header is %d but found a shorter row (key=%s) that is %d long. Adding empties %s at the end (line=%s)\n"
                            % (reflen, k, n, str(insert_for_missings), data_dict[k])
                        )
                    nwarns += 1
                elif nmax_warn == nwarns:
                    sys.stderr.write(
                        "     in equate_data_to_same_length SUPPRESSING further WARNINGS..\n\n"
                    )
                    nwarns += 1
            data_dict[k] = data_dict[k] + (reflen - n) * [insert_for_missings]
        elif n > reflen:
            if type(nmax_warn) is int and nwarns < nmax_warn:
                if "filename" in dir(data_dict) and data_dict.filename is not None:
                    sys.stderr.write(
                        "WARNING in equate_data_to_same_length fname='%s' length of header is %d but found a longer row (key=%s) that is %d long (line=%s)"
                        % (str(data_dict.filename), reflen, k, n, data_dict[k])
                    )
                else:
                    sys.stderr.write(
                        "WARNING in equate_data_to_same_length! length of header is %d but found a longer row (key=%s) that is %d long (line=%s)"
                        % (reflen, k, n, data_dict[k])
                    )
                nwarns += 1
            elif nmax_warn == nwarns:
                sys.stderr.write(
                    "     in equate_data_to_same_length SUPPRESSING further WARNINGS..\n\n"
                )
                nwarns += 1
            if merge_longer:
                last_element = ""
                for el in data_dict[k][reflen - 1 :]:
                    if type(el) is str:
                        last_element += el
                    else:
                        last_element += " " + str(el)
                data_dict[k] = data_dict[k][: reflen - 1] + [last_element]
                if type(nmax_warn) is int and nwarns <= nmax_warn:
                    sys.stderr.write("-> merging last entry\n")
            elif type(nmax_warn) is int and nwarns <= nmax_warn:
                sys.stderr.write("\n")
    return data_dict


def TupleToList(mytuple):
    # convert a tuple to a list saving the elements in order
    mylist = []
    for i in mytuple:
        mylist += [i]
    del i
    return mylist


def new_key_for_dict(dictionary, old_key, start_at=0, check_for_rep=True):
    """
    if old_key in dictionary finds a new key that is not
    by adding 00 (if start_at is 0) and so on
    """
    if check_for_rep and "_rep" in old_key:
        try:
            spl = old_key.rfind("_rep")
            ij = int(old_key[spl + 4 :])  #  check if it is a replica number
            ij = 1  # start from zero, may be merging two dictionaries with higher replicate number
            while True:
                c = old_key[:spl] + "_rep%d" % (ij)
                if c not in dictionary:
                    break
                ij += 1
            return c
        except Exception:
            pass  # add the standard way
    ij = start_at
    while True:
        c = old_key + "%02d" % (ij)
        if c not in dictionary:
            break
        ij += 1
    return c


def line_to_dictionary(line, update_from_dict=None):
    """
    # given a line like l="CLUSTER_ID 11 , CVvalues 2.0 160.0 3.9 8.1 , FREE_ENE 8.103 , CL.size 591"
    # it converts it to a dictionary where every string is a key, the following float is the value.
    # if more than one float follows a string than the value is a list of floats
    #   l --> {'CVvalues': [2.0, 160.0, 3.9, 8.1], 'CL.size': 591, 'CLUSTER_ID': 11, 'FREE_ENE': 8.103}
    """
    if update_from_dict != None and type(update_from_dict) is dict:
        l_dict = update_from_dict
    else:
        l_dict = {}
    tmplis = line.split()  # separate on spaces, then on tab
    lis = []
    for l in tmplis:  # separate on tabs
        l = l.split("\t")
        lis += l
    del tmplis, l
    key_to_add = None
    last_key = None
    for l in lis:
        if "," in lis:
            l = l.replace(",", "")
        if ";" in lis:
            l = l.replace(";", "")
        if l != "":
            l, asses = convert_to_number(l)
            if not asses:
                if l in l_dict:
                    sys.stderr.write(
                        "*WARNING* in line_to_dictionary() key %s already in dictionary with value %s\n"
                        % (l, str(l_dict[l]))
                    )
                else:
                    key_to_add = l
            if asses:
                if key_to_add is not None:
                    if key_to_add in l_dict:
                        if type(l_dict[key_to_add]) is list:
                            l_dict[key_to_add] += [l]  # add to list
                        elif (
                            type(l_dict[key_to_add]) is float
                            or type(l_dict[key_to_add]) is int
                        ):
                            l_dict[key_to_add] = [
                                l_dict[key_to_add],
                                l,
                            ]  # convert to list
                    else:
                        l_dict[key_to_add] = l
                    last_key = key_to_add
                    key_to_add = None
                elif last_key is not None:
                    if type(l_dict[last_key]) is list:
                        l_dict[last_key] += [l]  # add to list
                    elif (
                        type(l_dict[last_key]) is float or type(l_dict[last_key]) is int
                    ):
                        l_dict[last_key] = [l_dict[last_key], l]  # convert to list
    return l_dict


def print_latex(
    filename,
    dictionary,
    key_column=1,
    DELIMITER=None,
    HEADER=None,
    key_column_header_name="keys",
    ha="c",
    convert_float_list_to_str=False,
):
    out = open(filename, "w")
    if HEADER is None and hasattr(dictionary, "hd"):
        HEADER = dictionary.hd
    hd = None
    if type(HEADER) is list:
        hd = HEADER[:]
    elif type(HEADER) is dict:
        hd = ["?"] * (len(HEADER))  # after we are goning to add the key column
        for title in HEADER:
            if type(HEADER[title]) is int and HEADER[title] < len(HEADER):
                hd[HEADER[title]] = title
    if hd is not None:
        out.write("\\begin{tabular}{ " + ((ha + " ") * len(hd)) + "}\n")
        hd[(key_column - 1) : (key_column - 1)] = [key_column_header_name]
        out.write(" & ".join(map(str, hd)) + "\\\\\n")
    else:
        out.write(
            "\\begin{tabular}{ "
            + ((ha + " ") * (len(list(dictionary.values())[0]) + 1))
            + "}\n"
        )  # +1 is for the key column
    for key in dictionary:
        if (
            type(dictionary[key]) is list
        ):  # dictionary where one entry is one list, this is going to be one line in the csv
            if (
                type(dictionary[key][0]) is list
            ):  # dictionary where one entry is a list of lists. Each list is going  to be one line in the csv
                for i in range(0, len(dictionary[key])):
                    line = dictionary[key][i][
                        :
                    ]  # actually copy as we don't want to change the dictionary
                    line[(key_column - 1) : (key_column - 1)] = [
                        key
                    ]  # insert the key in this column in the line
                    if convert_float_list_to_str:
                        for j, l in enumerate(line):
                            if type(l) is list or isinstance(l, numpy.ndarray):
                                line[j] = float_list_to_str(l)
                    out.write(" & ".join(map(str, line)) + "\\\\\n")
                    del line
                del i
            elif (
                type(dictionary[key][0]) is tuple
            ):  # dictionary where one entry is a list of tuple. Each tuple is going to be one line of the csv
                for i in range(0, len(dictionary[key])):
                    line = TupleToList(dictionary[key][i])
                    line[(key_column - 1) : (key_column - 1)] = [
                        key
                    ]  # insert the key in this column in the line
                    if convert_float_list_to_str:
                        for j, l in enumerate(line):
                            if type(l) is list or isinstance(l, numpy.ndarray):
                                line[j] = float_list_to_str(l)
                    out.write(" & ".join(map(str, line)) + "\\\\\n")
                    del line
                del i
            else:  # dictionary where one entry is one list, this is going to be one line in the csv
                line = dictionary[key][
                    :
                ]  # actually copy as we don't want to change the dictionary
                line[(key_column - 1) : (key_column - 1)] = [
                    key
                ]  # insert the key in this column in the line
                if convert_float_list_to_str:
                    for j, l in enumerate(line):
                        if type(l) is list or isinstance(l, numpy.ndarray):
                            line[j] = float_list_to_str(l)
                out.write(" & ".join(map(str, line)) + "\\\\\n")
                del line
        elif (
            type(dictionary[key]) is tuple
        ):  # dictionary where one entry is a tuple. it is going to be one line of the csv
            line = TupleToList(dictionary[key])
            line[(key_column - 1) : (key_column - 1)] = [
                key
            ]  # insert the key in this column in the line
            if convert_float_list_to_str:
                for j, l in enumerate(line):
                    if type(l) is list or isinstance(l, numpy.ndarray):
                        line[j] = float_list_to_str(l)
            out.write(" & ".join(map(str, line)) + "\\\\\n")
            del line
        else:  # simple dictionary where one entry is just a variable
            if key_column == 1 or key_column == 0:
                out.write(" & ".join(map(str, [key, dictionary[key]])) + "\\\\\n")
            else:
                out.write(" & ".join(map(str, [dictionary[key], key])) + "\\\\\n")
    out.write("\\end{tabular}\n")
    out.close()


def csvPrintDictionary(
    filename,
    dictionary,
    key_column=1,
    DELIMITER=None,
    HEADER=None,
    key_column_header_name="keys",
    only_top=None,
    allow_more_entries_per_key=False,
    append=False,
    convert_float_list_to_str=False,
):
    """
    # function that print a dictionary in a csv file, the dictionary can be of many different types. 
    # Its entry can be variables,lists,tuples, list of lists and list of tuples. In the latter two case each list/tuple is printed on a different line. 
    # (each entry is always printed on a diffent line. One can choose the column where the key is written and the csv delimiter). HEADER has to be a list
    """
    if type(filename) is str:
        if append:
            fileout = open(filename, "a")
        else:
            fileout = open(filename, "w")
        close_it = True
    else:
        fileout = filename
        close_it = False
    if DELIMITER is None:
        data = csv.writer(fileout)
    else:
        data = csv.writer(fileout, delimiter=DELIMITER)
    if HEADER is None:
        if hasattr(dictionary, "hd"):
            HEADER = dictionary.hd
        elif hasattr(dictionary, "HD") and not hasattr(dictionary.HD, "__call__"):
            HEADER = dictionary.HD
    hd = None
    if type(HEADER) is list and HEADER != []:
        hd = HEADER[:]
    elif type(HEADER) is dict and HEADER != {}:
        hd = ["?"] * (len(HEADER))  # after we are goning to add the key column
        for title in HEADER:
            if type(HEADER[title]) is int and HEADER[title] < len(HEADER):
                hd[HEADER[title]] = title
    if hd is not None:
        hd[(key_column - 1) : (key_column - 1)] = [key_column_header_name]
        data.writerow(hd)
    for j, key in enumerate(dictionary):
        if only_top is not None and j > only_top:
            break
        if allow_more_entries_per_key:
            if (
                type(dictionary[key]) is list
            ):  # dictionary where one entry is one list, this is going to be one line in the csv
                if (
                    type(dictionary[key][0]) is list
                ):  # dictionary where one entry is a list of lists. Each list is going  to be one line in the csv
                    for i in range(0, len(dictionary[key])):
                        line = dictionary[key][i][
                            :
                        ]  # actually copy as we don't want to change the dictionary
                        line[(key_column - 1) : (key_column - 1)] = [
                            key
                        ]  # insert the key in this column in the line
                        if convert_float_list_to_str:
                            for j, l in enumerate(line):
                                if type(l) is list or isinstance(l, numpy.ndarray):
                                    line[j] = float_list_to_str(l)
                        data.writerow(line)
                        del line
                    del i
                elif (
                    type(dictionary[key][0]) is tuple
                ):  # dictionary where one entry is a list of tuple. Each tuple is going to be one line of the csv
                    for i in range(0, len(dictionary[key])):
                        line = TupleToList(dictionary[key][i])
                        line[(key_column - 1) : (key_column - 1)] = [
                            key
                        ]  # insert the key in this column in the line
                        if convert_float_list_to_str:
                            for j, l in enumerate(line):
                                if type(l) is list or isinstance(l, numpy.ndarray):
                                    line[j] = float_list_to_str(l)
                        data.writerow(line)
                        del line
                    del i
        elif (
            type(dictionary[key]) is list
        ):  # dictionary where one entry is one list, this is going to be one line in the csv
            line = dictionary[key][
                :
            ]  # actually copy as we don't want to change the dictionary
            line[(key_column - 1) : (key_column - 1)] = [
                key
            ]  # insert the key in this column in the line
            if convert_float_list_to_str:
                for j, l in enumerate(line):
                    if type(l) is list or isinstance(l, numpy.ndarray):
                        line[j] = float_list_to_str(l)
            data.writerow(line)
            del line
        elif (
            type(dictionary[key]) is tuple
        ):  # dictionary where one entry is a tuple. it is going to be one line of the csv
            line = TupleToList(dictionary[key])
            line[(key_column - 1) : (key_column - 1)] = [
                key
            ]  # insert the key in this column in the line
            if convert_float_list_to_str:
                for j, l in enumerate(line):
                    if type(l) is list or isinstance(l, numpy.ndarray):
                        line[j] = float_list_to_str(l)
            data.writerow(line)
            del line
        else:  # simple dictionary where one entry is just a variable
            if key_column == 1 or key_column == 0:
                data.writerow([key, dictionary[key]])
            else:
                data.writerow([dictionary[key], key])
    if close_it:
        fileout.close()
    del data
    return


def MergeDictionaries(dict1, dict2):
    """
     merges dict2 into dict1 matching the keys of dict2 that are also in dict1. dict1 is consequently changed
    """
    for key in dict2:
        if key in dict1:
            if type(dict1[key]) is list:
                if type(dict1[key][0]) is list:  # if dict1 contains a list of lists
                    for i in range(0, len(dict1[key])):
                        if type(dict2[key]) is list:
                            dict1[key][i] += dict2[key]
                        elif type(dict2[key]) is tuple:
                            print(
                                "***WARNING*** in MergeDictionaries trying to merge dictionaries of non compatible types (list of lists, tuple) Converting to list\n"
                            )
                            dict1[key][i] = dict1[key][i] + list(dict2[key])
                        else:
                            dict1[key][i] += [dict2[key]]
                    del i
                elif type(dict1[key][0]) is tuple:  # if dict1 contains a list of tuples
                    for i in range(0, len(dict1[key])):
                        if type(dict2[key]) is tuple:
                            dict1[key][i] += dict2[key]
                        elif type(dict2[key]) is list:
                            print(
                                "***WARNING*** in MergeDictionaries trying to merge dictionaries of non compatible types (list of tuples, list) Converting to list\n"
                            )
                            dict1[key][i] = list(dict1[key][i]) + dict2[key]
                        else:
                            dict1[key][i] += tuple([dict2[key]])
                    del i
                else:
                    if type(dict2[key]) is list:
                        dict1[key] += dict2[key]
                    elif type(dict2[key]) is tuple:
                        print(
                            "***WARNING*** in MergeDictionaries trying to merge dictionaries of non compatible types (list, tuple) Converting to list\n"
                        )
                        dict1[key] = dict1[key] + list(dict2[key])
                    else:
                        dict1[key] += [dict2[key]]
            elif (
                type(dict1[key]) is tuple
            ):  # if dict1 is a dictionary whose elements are tuples then try to merge in a tuple (if dict2 elements are list it will convert to lists
                if type(dict2[key]) is tuple:
                    dict1[key] = dict1[key] + dict2[key]
                elif type(dict2[key]) is list:
                    print(
                        "***WARNING*** in MergeDictionaries trying to merge dictionaries of non compatible types (tuple, list) Converting to list\n"
                    )
                    dict1[key] = list(dict1[key]) + dict2[key]
                else:
                    dict1[key] = dict1[key] + tuple(dict2[key])
            else:  # if dict1 is a standard dictionary merge with dict2 converting it into a list dictionary, unless dict2 is tuples in which case convert to tuple
                if type(dict2[key]) is tuple:
                    dict1[key] = tuple([dict1[key]]) + dict2[key]
                elif type(dict2[key]) is list:
                    dict1[key] = [dict1[key]] + dict2[key]
                else:
                    dict1[key] = [dict1[key], dict2[key]]
    return


def GetColumn(
    dictionary,
    column_number,
    column_number1=None,
    column_number2=None,
    return_keys=False,
    return_list_of_lists_position=False,
    header_dictionary=None,
):
    """
    TO MANAGE DICTIONARIES THAT ARE MADE OF LIST OF LISTS or that contains lists, they should correspond to csv file, each list is a row. Note that in this function column_number STARTS FROM 1 not from 0 (like in the file). Note that one of the columns in the file will probabily (depending on how the dictionary was built) be the key columns, so be careful when giving column_number. it returns (keys,column,column1,column2) in tuple or just column (as a list) if none of the other is given.
    if you have a dictionary header to read through the file (i.e. a dictionary that has the column name as a title and the coulm number as an entry: see csvToDictionary) you can give it in header and you can give to the column_number(s) the entry of the header in string format
    if the dictionary has multiple entries per key (that is the same key corresponds to multiple raws in the original file) than it will be a list of lists. One can set return_list_of_lists_position to True in order to have a list with the original position in the dictionary returned as well.
    """
    # first check everything
    if return_list_of_lists_position:
        return_keys = True
    column = []
    column1 = []
    column2 = []
    if type(header_dictionary) is dict:
        column_number = header_dictionary[column_number] + 1
        if column_number1 is not None:
            column_number1 = header_dictionary[column_number1] + 1
        if column_number2 is not None:
            column_number2 = header_dictionary[column_number2] + 1
    keys = []
    list_of_lists_position = []

    for key in dictionary:
        if type(dictionary[key]) is list:
            if type(dictionary[key][0]) is list:
                for row in dictionary[key]:
                    if column_number <= len(row):
                        column += [row[column_number - 1]]
                        if return_keys:
                            keys += [
                                key
                            ]  # this way we have another list, of the same length as column, which contains the corresponding keys in the proper order
                        if (
                            return_list_of_lists_position
                        ):  # this way we have another list, of the same length as column, which contains the corresponding position, one can than retrive precisely the same line with [key][position].
                            list_of_lists_position += [dictionary[key].index(row)]
                    else:
                        print(
                            "***WARNING*** in GetColumn column number out of range (%d columns, requesting for %d)\n%s\n"
                            % (len(row), column_number, str(row))
                        )
                        return 2
                    if type(column_number1) is int and column_number1 <= len(row):
                        column1 += [row[column_number1 - 1]]
                    if type(column_number2) is int and column_number2 <= len(row):
                        column2 += [row[column_number2 - 1]]
                del row
            else:
                if column_number <= len(dictionary[key]):
                    column += [dictionary[key][column_number - 1]]
                    if return_keys is True:
                        keys += [
                            key
                        ]  # this way we have another list, of the same length as column, which contains the corresponding keys in the proper order
                    if (
                        return_list_of_lists_position
                    ):  # this way we have another list, of the same length as column, which contains the corresponding position, one can than retrive precisely the same line with [key][position].
                        list_of_lists_position += [dictionary[key].index(row)]
                else:
                    print(
                        "***WARNING*** in GetColumn column number out of range (%d columns, requesting for %d)"
                        % (len(row), column_number)
                    )
                    return 2
                if type(column_number1) is int and column_number1 <= len(
                    dictionary[key]
                ):
                    column1 += [dictionary[key][column_number1 - 1]]
                if type(column_number2) is int and column_number2 <= len(
                    dictionary[key]
                ):
                    column2 += [dictionary[key][column_number2 - 1]]

        else:
            print(
                "***WARNING*** in GetColumn dictionary given is not a list of lists and its entries are not lists,returning 1"
            )
            return 1

    if not return_keys and column_number1 is None and column_number2 is None:
        del keys, column2, column1, list_of_lists_position
        return column

    to_return = tuple([column])
    if column1 != []:
        to_return += tuple([column1])
    if column2 != []:
        to_return += tuple([column2])
    if return_keys:
        to_return += tuple([keys])
    if return_list_of_lists_position:
        to_return += tuple([list_of_lists_position])
    return to_return


def csvFileMerger(
    filename1,
    filename2,
    column1=1,
    column2=1,
    delimiter1=None,
    delimiter2=None,
    column1_and_2_in_single_column=False,
    outputfilename=None,
    DELIMITER=None,
    output_with_same_number_of_column=True,
    case_insensitive=False,
):
    """
    # easier to use file merger but do not work too well with duplicates. Column 1 and 2 (belonging to the two files) are the one according to which the merging is done.
    # (if file1 contains duplicates and not 2 all works fine, otherwise only the first matching entries from 2 is merged in 1)
    # it returns a dict corresponding to the merged file
    """
    # open input files
    try:
        filein1 = open(filename1, "rb")
    except:
        print("***ERROR*** cannot open file %s\n" % filename1)
        raise IOError
    if delimiter1 is None:
        data1 = csv.reader(filein1)
    else:
        data1 = csv.reader(filein1, delimiter=delimiter1, skipinitialspace=True)
    try:
        filein2 = open(filename2, "rb")
    except:
        print("***ERROR*** cannot open file %s\n" % filename2)
        raise IOError
    if delimiter2 is None:
        data2 = csv.reader(filein2)
    else:
        data2 = csv.reader(filein2, delimiter=delimiter2, skipinitialspace=True)

    # open eventual output file
    if type(outputfilename) is str:
        fileout = open(outputfilename, "wb")
        if DELIMITER is None:
            out = csv.writer(fileout)
        else:
            out = csv.writer(fileout, delimiter=DELIMITER)

    lencheck = -1
    found = 0
    l1 = 0  # file 1 line counter
    l2 = 0  # file 2 line counter
    mg = 0  # merged file line counter
    merged = []
    for line1 in data1:
        l1 += 1
        found = 0
        if type(line1) is list:
            #            data.writerow(HEADERiprint 'line ',l1,l2
            l2 = 0
            if len(line1) < column1:
                print(
                    "***ERROR*** in csvFileMerger at line %d column id %d for file 1 %s is larger than read number of column %d. Have you set the correct delimiter? or are there empty line at the end?"
                    % (l1, column1, filename1, len(line1))
                )
                continue
            for line2 in data2:
                l2 += 1
                if type(line2) is list:
                    if len(line2) < column2:
                        print(
                            "***ERROR*** in csvFileMerger at line %d column id %d for file 2 %s is larger than read number of column %d. Have you set the correct delimiter? are there empty lines at the end?"
                            % (l2, column2, filename2, len(line2))
                        )
                        continue
                    if case_insensitive:
                        line1[column1 - 1] = str(line1[column1 - 1]).upper()
                        line2[column2 - 1] = str(line2[column2 - 1]).upper()
                    if line1[column1 - 1] == line2[column2 - 1]:
                        print(line1[column1 - 1], line2[column2 - 1])
                        if column1_and_2_in_single_column:
                            line2[
                                column2 - 1 : column2
                            ] = (
                                []
                            )  # in this case delete the column as it is already in line1
                        merged += [line1 + line2]
                        mg += 1
                        found = 1
                        if lencheck == -1:
                            lencheck = len(merged[mg - 1])
                        if lencheck != -1 and lencheck != len(merged[mg - 1]):
                            print(
                                "***WARNING*** in csvFileMerger lines do not seem to be of same length (%d %d, at line %d of file1, %d of file2)"
                                % (lencheck, len(merged[mg - 1]), l1, l2)
                            )
                        break
                else:
                    print("line %d file %s is not list!" % (l2, filename2))
        filein2.seek(0)
        if found == 0:
            merged += [line1]
            mg += 1

    del line1, line2, data2, data1, found
    print("merging loop finished. lencheck=%d" % (lencheck))
    if mg != l1:
        print(
            "***ERROR*** in merged file contains different number of line for original file %s (%d %d)"
            % (filename1, mg, l1)
        )
    filein1.close()
    filein2.close()

    if output_with_same_number_of_column and lencheck != -1:
        for j in range(0, len(merged)):
            print(merged[j])
            if len(merged[j]) < lencheck:
                for i in range(len(merged[j]), lencheck):
                    merged[j] += [
                        ""
                    ]  # add empty columns at the end of every row that hasn't been merged so that all the rows have the same number of columns in the output
            if outputfilename != None:
                out.writerow(merged[j])

    if outputfilename != None:
        fileout.close()
        del out

    del filein2, filein1, l1, l2, mg, lencheck
    return merged


def revert_dictionary(old_dictionary, use_Data=False):
    """
    this functions read a dictionary and returns the reverted one, which contains as keys the entries of old_dictionary and as entries the keys of old_dictionary.
    """
    if use_Data:
        new_dictionary = Data()
    else:
        new_dictionary = {}
    for key in old_dictionary:
        new_dictionary[old_dictionary[key]] = key
    return new_dictionary


def list_to_dictionary(a_list):
    """
    it reads a list and return a dictionary where the keys are the list element and 
    the values are the positions of those elements in the list
    useful if I have an header in list format and I want to get it in dictionary format
    """
    dic = {}
    for i, el in enumerate(a_list):
        if el in dic:
            j = 0
            nel = str(el) + "_%02d" % j
            while nel in dic:
                j += 1
                nel = str(el) + "_%02d" % j
            sys.stderr.write(
                "**WARNING** in list_to_dictionary() element %s appears twice in the list. Now changing to %s in dictionary\n"
                % (el, nel)
            )
            dic[nel] = i
        else:
            dic[el] = i
    return dic


def make_dictionary_sorter(
    dictionary_of_list,
    column_that_sorts,
    check_that_sort_type_is=False,
    reverse_order=False,
    HEADER_dictionary=None,
):
    """
    note that since this is spreadsheet-like with column 1 we actually mean position 0 in the list and so on...
    it RETURNS a dictionary sorter that is a list of tuples. Each tuples has three elments (column,keys,positions). The first is the value extracted from the sorting column (and the list of tuples is sorted in this way), the second one is the key of the original dictionary and the third one is the position. The latter is useful if the dictionary was originally read from a csv where the same key could correspond to multiple rows. In this case each entry is a list of lists, so to retrieve the raws in order one needs both the dictionary key and the position in the list of lists. If this is not the case all poistions will just be zero.
    if you have a dictionary header to read through the file (i.e. a dictionary that has the column name as a title and the coulm number as an entry: see csvToDictionary) you can give it in header and you can give to the column_number(s) the entry of the header in string format
    setting check_that_sort_type_is to int, float, str... tries to convert the entries in column_that_sorts to the given type, useful when the dictionary has been read from a file and everything could be str
    """
    (column, keys, positions) = GetColumn(
        dictionary_of_list,
        column_that_sorts,
        return_keys=True,
        return_list_of_lists_position=True,
        header_dictionary=HEADER_dictionary,
    )
    if (
        type(check_that_sort_type_is) is type
    ):  # try to convert the entry to the given type, useful when the dictionary has been read from a file and everything could be str
        column = [check_that_sort_type_is(x) for x in column]
    sorter = list(zip(column, keys, positions))
    sorter.sort(reverse=reverse_order)
    return sorter


def Round_To_n(x, n, only_decimals=False):
    """
    rounds to n significant digits
    Note that 0 would leave 1 significant digit, 
    if not only_decimals then 
       n=0 ==> 1111 --> 1000
       n=1 ==> 1.005324314 --> 1.0
    if only_decimals then
       n=1 ==> 1.005324314 --> 1.0053
       And for numbers >1 it rounds to n-1, thus:
       n=0 ==> 1.1534142 --> 1
       n=1 ==> 1.1534142 --> 1.2
    """
    if x == 0:
        return 0.0
    if only_decimals:
        if x > 1:
            if n == 0:
                return int(x + 0.5)
            n -= 1
        int_part = int(x)
        return float(int_part) + Round_To_n(x - int_part, n)
    return round(x, -int(numpy.floor(numpy.sign(x) * numpy.log10(abs(x)))) + n)


def float_list_to_str(
    float_list,
    float_line_delimiter=";",
    list_of_list_delimiter=" ",
    dont_print_last_delimiter=True,
):
    """
    Converts a list of float into a string where consecutive floats are separated by a delimiter (e.g. a semicolon ; )
    useful if one wishes to store a profile into a single field of a csv file
    """
    float_str = ""
    for n in float_list:
        if list_of_list_delimiter is not None and (
            type(n) is list or type(n) is tuple or isinstance(n, numpy.ndarray)
        ):
            float_str += (
                float_list_to_str(
                    n,
                    float_line_delimiter=float_line_delimiter,
                    list_of_list_delimiter=None,
                    dont_print_last_delimiter=dont_print_last_delimiter,
                )
                + list_of_list_delimiter
            )
        else:
            float_str += str(n) + float_line_delimiter
    if (
        dont_print_last_delimiter
        and float_str != ""
        and (
            float_str[-1] == float_line_delimiter
            or float_str[-1] == list_of_list_delimiter
        )
    ):
        float_str = float_str[:-1]
    return float_str


# reverse of the above
def separated_str_to_list(
    string, float_line_delimiter=";", list_of_list_delimiter=None, force_float=False
):
    if string == "":
        return []
    if list_of_list_delimiter is not None and list_of_list_delimiter in string:
        if string[-1] == list_of_list_delimiter:
            string = string[:-1]
        li = []
        for s in string.split(list_of_list_delimiter):
            li += [
                separated_str_to_list(
                    s,
                    float_line_delimiter=float_line_delimiter,
                    list_of_list_delimiter=None,
                    force_float=force_float,
                )
            ]
        return li
    if string[-1] == float_line_delimiter:
        string = string[:-1]
    li = string.split(float_line_delimiter)
    li = [convert_to_number(l, force_float=force_float)[0] for l in li]
    return li


def read_block_tsv(
    filename,
    begin_block_keyword=">",
    data_catalog_keyword="pdb",
    key_column=1,
    additional_special_keyword=[">"],
    ignore_line_beginning_with=["#", "\n"],
):
    """
    # reads a block tsv (it separates staff with .split() rahter then with the csv module
    # keys will be key=l[:lc].split()[0] with lc=len(data_catalog_keyword)
    # it returns data, additional_special_keyword
    #   data is a dict of dict, first key determined by data_catalog_keyword and 
    #   second by key_column and additional_special_keyword
    # maybe it can read a multi line database
    """
    with open(filename) as ff:
        f = ff.read().splitlines()
    if (
        data_catalog_keyword != begin_block_keyword
        and begin_block_keyword not in additional_special_keyword
    ):
        additional_special_keyword += [begin_block_keyword]
    data = (
        {}
    )  # dict of dict, first key determined by data_catalog_keyword and keys second by key_column and additional_special_keyword
    nblocks = 0
    index_k = key_column - 1
    if index_k < 0:
        index_k = 0
    lb = len(begin_block_keyword)
    lc = len(data_catalog_keyword)
    found_cat = False
    read_block = False
    skip_after = False
    key = None
    for l in f:
        if misc.loose_compare(l, ignore_line_beginning_with, begin_with=True):
            continue
        if l.strip() == "":
            continue
        if l[:lb] == begin_block_keyword:
            found_cat = False
            read_block = True
            skip_after = True
            tmpdic = {}
            if l[lb:].split() != []:
                tmpdic[l[:lb]] = l[lb:].strip()
            nblocks += 1
        if l[:lc] == data_catalog_keyword:
            row = l[lc:].split()  # key is 0 the rest is added after
            key = row[0]
            if key not in data:
                found_cat = True
                data[key] = tmpdic
                tmpdic = {}
                if len(row) > 1:
                    for j, k in enumerate(row[1:]):
                        if k in additional_special_keyword and j + 1 < len(row):
                            data[key][k], _ = convert_to_number(row[1:][j + 1])
            else:
                sys.stderr.write(
                    "WARNING in read_block_tsv() data_catalog_keyword %s -> key %s already found\n"
                    % (data_catalog_keyword, key)
                )
        elif read_block:
            if skip_after:
                skip_after = False
                continue
            row = l.split()
            k = row[index_k]
            if (
                k in additional_special_keyword
            ):  # in this case we scan the whole line for additional keywords
                for j, k2 in enumerate(row):
                    if k2 in additional_special_keyword and j + 1 < len(row):
                        if found_cat:
                            data[key][k2], _ = convert_to_number(row[j + 1])
                        else:
                            tmpdic[k2], _ = convert_to_number(row[j + 1])
            else:
                row[index_k : index_k + 1] = []  # remove key entry
                if found_cat:
                    data[key][k] = [convert_to_number(x)[0] for x in row]  # save value
                else:
                    tmpdic[k] = [convert_to_number(x)[0] for x in row]  # save here
    return data, additional_special_keyword


def table_style(width=90,padding=8):
    return (
        """<style>
#customers {
    font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
    border-collapse: collapse;
    width: %s%%;
}

#customers td, #customers th {
    border: 1px solid #ddd;
    padding: %spx;
    text-align: center;
}

#customers tr:nth-child(even){
    background-color: #f2f2f2;
    text-align: center;
}

#customers tr:hover {
    background-color: #ddd;
    text-align: center;
}

#customers th {
    padding-top: %spx;
    padding-bottom: %spx;
    text-align: center;
    background-color: #0099ff;
    color: white;
}
</style>\n"""%(str(width),padding,padding,padding)
    )


def img_style(width=60):
    return (
        """<style>
    img {
    width:"""
        + str(width)
        + """%;
}
</style>\n"""
    )






def tableseq_style(overall_width_percent=100,first_colum_width_percent=10) :
    '''
    css style #tableseqs (goes with a div #tableseqsdiv)
    to create a table for long sequences that scrolls horizontally while keeping first column frozen (header frozen pane)
    mind empty rows (put at least an _ or header rows will have different height than table rows
    '''
    return ("""  <style>
#tableseqs  {
  border-collapse: separate;
  border-spacing: 0;
  border-top: 1px solid LightGrey;
  overflow: hidden;
  z-index: 1;
}

#tableseqs td,th {
  margin: 0;
  cursor: pointer;
  border: 0.5px solid LightGrey;
  white-space: nowrap;
  border-top-width: 0px;
  position: relative;
  height: 20px;
  text-align: center;
  vertical-align: middle;
}

#tableseqsdiv {
  width: %s%%;
  overflow-x: scroll;
  margin-left: %s%%;  /*must be greater or equal to width of first column in .headcol */
  overflow-y: hidden;
  padding: 0;
}

#tableseqs tr:hover {
    background-color: #ddd; /*#ffa;*/
}

#tableseqs td:hover::after { 
    background-color: #ddd;
    position: absolute;
    height: 10000px;
    top: -5000px;
    width: 100%%;
    content: '\00a0';
    left: 0;
    z-index: -1;
}

#tableseqs .headcol {
  position: absolute;
  background-color: #0099ff;
  color: white;
  width: %s%%; /*widht of first column (previously 6em) */
  text-align: right;
  /*padding-right: 3px;*/
  left: 0;
  top: auto;
  border-top-width: 1px;
  /*only relevant for first row*/
  margin-top: -1px;
  /*compensate for top border*/
  overflow-y: hidden;
}
</style>"""%(str(overall_width_percent - first_colum_width_percent), str(first_colum_width_percent), str(first_colum_width_percent-0.5))
)

def to_html_table(
    Data, add_style=True, embedded=False, round=None, title=None, caption=None, width=80,padding=8,include_keys=True):
    """
    width is in percent
    """
    table = ""
    if add_style and not embedded:
        table = table_style(width,padding=padding)
    if title is not None and title != "":
        table += "<h3>" + title + "</h3>\n"
    if add_style or embedded:
        table += '\n <table id="customers">\n'
    else:
        table += "<table>"
    if round is not None:
        data2 = Data.copy(deep=True)
        data2.round_all(round)
    else:
        data2 = Data
    # creating HTML header row if header is provided
    if data2.hd is not None and data2.hd != {}:
        if include_keys : bit =[ "<th>" + str(cell) + "</th>" for cell in [data2.key_column_hd_name] + data2.HD() ]
        else : bit = ["<th>" + str(cell) + "</th>" for cell in data2.HD() ]
        table += "".join(bit)
        table += "\n"
    # else:
    #    table+= "".join(["<th>"+cell+"</th>" for cell in rows[0].split(delimiter)])
    #    rows=rows[1:]
    # Converting csv to html row by row
    for k in data2:
        if include_keys :
            bit = "".join(["<td>" + str(cell) + "</td>" for cell in [k] + data2[k]])
        else :
            bit = "".join(["<td>" + str(cell) + "</td>" for cell in data2[k]])
        table += (
            "<tr>"
            + bit
            + "</tr>"
            + "\n"
        )
    table += "</table><br>\n"
    if caption is not None and caption != "":
        table += "" + caption + "\n"
    return table


def html_image(
    image_src, alt="myimage", width=None, title=None, caption=None, break_line=True
):
    """
    width is in points
    """
    # could add height="377" but would deform the image if not scaled with width "vertical-align: baseline
    if title is not None and title != "":
        st = "<h3>" + title + "</h3>\n"
    else:
        st = ""
    if width is None:
        st += (
            '<img src="'
            + image_src
            + '" alt="'
            + alt
            + '" style="vertical-align: baseline;horizontal-align: center;" width="'
            + str(800)
            + '">\n'
        )  # stylesheet overrides this width
    else:
        st += (
            '<img src="'
            + image_src
            + '" alt="'
            + alt
            + '" style="vertical-align: baseline;horizontal-align: center;width:'
            + str(width)
            + 'px;">\n'
        )  # but not this
    if break_line:
        st += "<br>"
    if caption is not None and caption != "":
        st += "" + caption + "<br></br>\n"
    return st


def html_paragraph(text, align="justify", title=None):
    p = ""
    if type(title) is str:
        p = '<h2 style="text-align: ' + align + ';">' + title + "</h2>\n"
    return p + '<p style="text-align: ' + align + ';">' + text + "</p>\n"


def html_document(
    element_list,
    elemnt_kwargs=None,
    doc_title=None,
    title_align="center",
    add_table_style=True,
    outfile=None,
):
    """
    elemnt_kwargs is a list of dictionaries and are the kwargs of the corresponding functions
      html_image  html_paragraph or to_html_table
    elements can be dictionaries (keys such as 'title' or 'html') to add specific bits
    
    element_list could be text for a paragraph, or src of image (i.e. a string corresponding to an existing file), or Data classes (to insert tables) or can be dictionaries, where keys are 'title' for a new section or html for pure html text
    """
    if elemnt_kwargs is None:
        elemnt_kwargs = [{} for i in range(len(element_list))]
    doc = "<!DOCTYPE html>\n<html>\n<body>\n"
    if add_table_style:
        doc += "<head>\n"
        doc += table_style()
        doc += tableseq_style()
        doc += "</head>\n"
    if doc_title is not None and doc_title != "":
        doc += '<h1 style="text-align: ' + title_align + ';">' + doc_title + "</h1>\n"
    for j, el in enumerate(element_list):
        if type(el) is str:  # could be text or src of image
            if os.path.isfile(el):
                doc += html_image(el, **elemnt_kwargs[j])
            else:
                doc += html_paragraph(el, **elemnt_kwargs[j])
        elif isinstance(el, Data) or "Data" in str(type(el)):
            doc += to_html_table(
                el, add_style=add_table_style, embedded=True, **elemnt_kwargs[j]
            )
        elif type(el) is dict:
            for k in el:
                if k.lower() == "title":
                    doc += (
                        '<h1 style="text-align:'
                        + title_align
                        + ';">'
                        + el[k]
                        + "</h1>\n"
                    )
                elif k.lower() == "html":
                    doc += el[k]
        else:
            sys.stderr.write(
                "**ERROR** in html_document cannot reconginse type of elemnt at index %d %s\n"
                % (j, str(type(el)))
            )
            sys.stderr.flush()
    doc += "\n</body>\n</html>"
    if outfile is not None and type(outfile) is str:
        o = open(outfile, "w")
        o.write(doc)
        o.close()
    return doc


"""
BELOW HERE IS WORK IN PROGRESS
"""


def determine_key(
    line_split,
    t_dict,
    key_identifier=None,
    get_rid_of=None,
    warn_tag="",
    remove_first_n_char_from_not_identifier=None,
):
    if key_identifier is not None:
        if len(line_split) > 1:
            key = line_split[1]
        else:
            key = line_split[0][len(key_identifier) :]
    else:
        key = line_split[0]
        if remove_first_n_char_from_not_identifier is not None:
            key = key[remove_first_n_char_from_not_identifier:]
            if key == "":
                sys.stderr.write(
                    "WARNING in read_multi_line_database() key becomes null after removing %d char (option remove_first_n_char_from_not_identifier). At line_spilt %s   %s\n"
                    % (
                        remove_first_n_char_from_not_identifier,
                        str(line_split),
                        str(warn_tag),
                    )
                )
    if key in t_dict:
        sys.stderr.write(
            "WARNING in read_multi_line_database() key %s  already present OVERWRITTEN! %s\n"
            % (key, str(warn_tag))
        )
    if get_rid_of is not None:
        key = key.replace(get_rid_of, "")
    return key


def determine_value(
    line_split_slice,
    float_line_delimiter=";",
    bool_converter={"Y": True, "OK": True, "N": False, "!OK": False},
    do_upper_case_bool=True,
):
    if len(line_split_slice) == 0:
        return ""
    elif len(line_split_slice) == 1:
        if (
            float_line_delimiter is not None
            and float_line_delimiter in line_split_slice[0]
        ):
            return separated_str_to_list(
                line_split_slice[0], float_line_delimiter=";", force_float=False
            )
        if bool_converter != {} and bool_converter is not None:
            if do_upper_case_bool and line_split_slice[0].upper() in bool_converter:
                return bool_converter[line_split_slice[0].upper()]
            elif line_split_slice[0] in bool_converter:
                return bool_converter[line_split_slice[0]]
        return convert_to_number(line_split_slice[0])[
            0
        ]  # return a string if it a string, an int if it is an int and a float if it is a float
    else:
        for j, st in enumerate(line_split_slice):
            if float_line_delimiter is not None and float_line_delimiter in st:
                line_split_slice[j] = separated_str_to_list(
                    st, float_line_delimiter=";", force_float=False
                )
            elif bool_converter != {} and bool_converter is not None:
                if do_upper_case_bool and st.upper() in bool_converter:
                    line_split_slice[j] = bool_converter[st.upper()]
                elif st in bool_converter:
                    line_split_slice[j] = bool_converter[st]
            else:
                line_split_slice[j] = convert_to_number(st)
        return line_split_slice


def parse_multiline_database(
    filename,
    key_identifier="##Unibegin",
    secondary_key_identifier="#Dombegin",
    get_rid_of=":",
    float_line_delimiter=None,
    bool_converter={
        "Y": True,
        "YES": True,
        "OK": True,
        "N": False,
        "NO": True,
        "!OK": False,
    },
    skip_begins=["#", "@", "\n"],
    remove_first_n_char_from_not_identifier=None,
):
    """
    # if apply_to_fields is True the fields are selected automatically, if False the conversion is not applied and if it is a list of field it is applied only on those one in the list.
    # note that skip_begins is applied after the key_identifier; so in a key identifier that is the same as a skip_begin will be seen and used.
    # it returns an OrderedDict() of OrderedDict()
    # remove_first_n_char_from_not_identifier is applied only after secondary keys
    """
    with open(filename) as ff:
        data = (
            ff.read().splitlines()
        )  # read the file as a list, each elemnt is a str corresponding to the line
    output = OrderedDict()
    key = None
    for li, line in enumerate(data):
        li += 1
        try:
            line = line.split()
            if line[0][: len(key_identifier)] == key_identifier:
                key = determine_key(
                    line,
                    output,
                    key_identifier=key_identifier,
                    get_rid_of=get_rid_of,
                    warn_tag=" in main keys at line %d" % (li),
                    remove_first_n_char_from_not_identifier=None,
                )
                output[key] = OrderedDict()
                sec_key = None
            elif (
                secondary_key_identifier is not None
                and line[0][: len(secondary_key_identifier)] == secondary_key_identifier
            ):
                sec_key = determine_key(
                    line,
                    output[key],
                    key_identifier=secondary_key_identifier,
                    get_rid_of=get_rid_of,
                    warn_tag=" in sec_keys at entry " + str(key) + " at line %d" % (li),
                    remove_first_n_char_from_not_identifier=None,
                )
                output[key][sec_key] = OrderedDict()
            elif line[0] == "":
                continue
            elif line[0][0] in skip_begins:
                continue
            elif key is not None:
                if sec_key is not None:
                    lin_key = determine_key(
                        line,
                        output,
                        key_identifier=None,
                        get_rid_of=get_rid_of,
                        warn_tag=" in entry " + str(key) + " at line %d" % (li),
                        remove_first_n_char_from_not_identifier=remove_first_n_char_from_not_identifier,
                    )
                    output[key][sec_key][lin_key] = determine_value(
                        line[1:],
                        float_line_delimiter=float_line_delimiter,
                        bool_converter=bool_converter,
                        do_upper_case_bool=True,
                    )
                else:
                    if secondary_key_identifier is None:
                        lin_key = determine_key(
                            line,
                            output,
                            key_identifier=None,
                            get_rid_of=get_rid_of,
                            warn_tag=" in entry " + str(key) + " at line %d" % (li),
                            remove_first_n_char_from_not_identifier=remove_first_n_char_from_not_identifier,
                        )
                    else:
                        lin_key = determine_key(
                            line,
                            output,
                            key_identifier=None,
                            get_rid_of=get_rid_of,
                            warn_tag=" in entry " + str(key) + " at line %d" % (li),
                            remove_first_n_char_from_not_identifier=None,
                        )
                    output[key][lin_key] = determine_value(
                        line[1:],
                        float_line_delimiter=float_line_delimiter,
                        bool_converter=bool_converter,
                        do_upper_case_bool=True,
                    )
        except Exception:
            sys.stderr.write(
                "**ERROR** while reading line %d from file %s\n   |%s|"
                % (li, filename, str(line))
            )
            raise
    return output


def parse_cot_database(
    filename,
    key_identifier="##Unibegin",
    secondary_key_identifier="#Dombegin",
    key_for_list_of_subdicts="#Segbegin",
    get_rid_of=":",
    float_line_delimiter=None,
    bool_converter={"Y": True, "OK": True, "N": False, "!OK": False},
    skip_begins=["#", "@", "\n"],
    remove_first_n_char_from_not_identifier=None,
):
    """
    # similar to the csv_dict.parse_multiline_database but tailored to parse cotraslational folding database, adds a list of segments.
    """
    with open(filename) as ff:
        data = (
            ff.read().splitlines()
        )  # read the file as a list, each elemnt is a str corresponding to the line
    output = OrderedDict()
    key = None
    for line in data:
        line = line.split()
        if line[0][: len(key_identifier)] == key_identifier:
            key = determine_key(
                line,
                output,
                key_identifier=key_identifier,
                get_rid_of=get_rid_of,
                warn_tag=" in main keys",
                remove_first_n_char_from_not_identifier=None,
            )
            output[key] = OrderedDict()
            sec_key = None
            add_to_list = False
        elif (
            secondary_key_identifier is not None
            and line[0][: len(secondary_key_identifier)] == secondary_key_identifier
        ):
            sec_key = determine_key(
                line,
                output[key],
                key_identifier=secondary_key_identifier,
                get_rid_of=get_rid_of,
                warn_tag=" in sec_keys at entry " + str(key),
                remove_first_n_char_from_not_identifier=None,
            )
            output[key][sec_key] = OrderedDict()
            add_to_list = False
        elif (
            key_for_list_of_subdicts is not None
            and line[0][: len(key_for_list_of_subdicts)] == key_for_list_of_subdicts
        ):
            if sec_key is not None:
                if "segments" in output[key]:
                    output[key][sec_key]["segments"] += [OrderedDict()]
                else:
                    output[key][sec_key]["segments"] = [OrderedDict()]
                output[key][sec_key]["segments"][-1]["id"] = tuple(line[1:])
            else:
                if "segments" in output[key]:
                    output[key]["segments"] += [OrderedDict()]
                else:
                    output[key]["segments"] = [OrderedDict()]
                output[key]["segments"][-1]["id"] = tuple(line[1:])
            add_to_list = True
        elif line[0] == "":
            continue
        elif line[0][0] in skip_begins:
            continue
        elif key is not None:
            if add_to_list:
                lin_key = determine_key(
                    line,
                    output,
                    key_identifier=None,
                    get_rid_of=get_rid_of,
                    warn_tag=" in entry (add_to_list) " + str(key),
                    remove_first_n_char_from_not_identifier=5,
                )
                if sec_key is not None:
                    output[key][sec_key]["segments"][-1][lin_key] = determine_value(
                        line[1:],
                        float_line_delimiter=float_line_delimiter,
                        bool_converter=bool_converter,
                        do_upper_case_bool=True,
                    )
                else:
                    output[key]["segments"][-1][lin_key] = determine_value(
                        line[1:],
                        float_line_delimiter=float_line_delimiter,
                        bool_converter=bool_converter,
                        do_upper_case_bool=True,
                    )
            if sec_key is not None:
                lin_key = determine_key(
                    line,
                    output,
                    key_identifier=None,
                    get_rid_of=get_rid_of,
                    warn_tag=" in entry " + str(key),
                    remove_first_n_char_from_not_identifier=remove_first_n_char_from_not_identifier,
                )
                output[key][sec_key][lin_key] = determine_value(
                    line[1:],
                    float_line_delimiter=float_line_delimiter,
                    bool_converter=bool_converter,
                    do_upper_case_bool=True,
                )
            else:
                if secondary_key_identifier is None:
                    lin_key = determine_key(
                        line,
                        output,
                        key_identifier=None,
                        get_rid_of=get_rid_of,
                        warn_tag=" in entry " + str(key),
                        remove_first_n_char_from_not_identifier=remove_first_n_char_from_not_identifier,
                    )
                else:
                    lin_key = determine_key(
                        line,
                        output,
                        key_identifier=None,
                        get_rid_of=get_rid_of,
                        warn_tag=" in entry " + str(key),
                        remove_first_n_char_from_not_identifier=None,
                    )
                output[key][lin_key] = determine_value(
                    line[1:],
                    float_line_delimiter=float_line_delimiter,
                    bool_converter=bool_converter,
                    do_upper_case_bool=True,
                )
    return output
