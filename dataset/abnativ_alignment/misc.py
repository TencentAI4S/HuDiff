# (c) 2023 Sormannilab=

import os
import random
import sys
from collections import OrderedDict
from functools import reduce

import numpy
import scipy.optimize
import scipy.signal
import scipy.special
import scipy.stats
from scipy import sparse
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema

try:
    import zipfile
except ImportError:
    pass


# try:  # custom module, not necessarily needed (only in some functions if plot=True is given)
#     from . import plotter  # see if it is in same folder
# except ImportError:
#     try:  # see if it's in pythonpath
#         import plotter
#     except ImportError:
#
#         class tmpplotter:  # define dummy plotter class
#             def profile(self, *args, **kwargs):
#                 raise Exception("PLOTTER MODULE NOT AVAILABLE\n")
#
#         plotter = tmpplotter()


python_to_numpy_type = {int: "i4", float: "f4", str: "a10"}
str_to_bool = {
    "yes": True,
    "YES": True,
    "Yes": True,
    "True": True,
    "true": True,
    "TRUE": True,
    "ok": True,
    "OK": True,
    "Ok": True,
    "1": True,
    1: True,
    1.0: True,
    "no": False,
    "NO": False,
    "No": False,
    "False": False,
    "FALSE": False,
    "false": False,
    "non ok": False,
    "!ok": False,
    "0": False,
    0: False,
    0.0: False,
}



def checkAllEqual(iterator):
    """
    checks if all elements in iterator are equal to each other
    """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def CompareFloats(float1, float2, sensibility=0.0001):
    """
    #compare two float numbers at a specified sensibility, Returns True/False
    # or if input are two numpy array returns the corresponding bool numpy array
    #return float1-sensibility <= float2 <= float1+sensibility 
    """
    return (float1 - sensibility <= float2) & (float2 <= float1 + sensibility)


def uniq(input_el):
    # given list/tuple it returns the unique elements in order of first appearance
    output = []
    for x in input_el:
        if x not in output:
            output.append(x)
    return output


def shared_elements(list_of_lists):
    """
    # returns a list with the common elements found in 
    # all the lists (or tuples or dictionaries) contained in list_of_lists
    """
    l = len(list_of_lists)
    tmpl = []
    for j, el in enumerate(list_of_lists):
        if type(el) is list:
            tmpl += el
        elif type(el) is tuple:
            tmpl += list(el)
        else:  # could be ordered dict or dict or a class derived
            try:
                tmpl += list(el.keys())
            except Exception:
                raise Exception(
                    "Error in misc.shared_elements() Type of element %d not recognized\n"
                    % (j)
                )
    shared_items = []
    for item in tmpl:
        if item not in shared_items and tmpl.count(item) == l:
            shared_items.append(item)
    del tmpl
    return shared_items


def split_in(staff, num_of_splits):
    """
    # split a string a list or a tuple in num_of_splits
    approximately equally-sized (int division) bits.
    """
    every = len(staff) // int(num_of_splits)
    rem = len(staff) % int(num_of_splits)
    start = 0
    end = start + every
    if rem > 0:
        end += 1
        rem -= 1
    l = [staff[start:end]]
    while end < len(staff):
        start = end
        end = start + every
        if rem > 0:
            end += 1
            rem -= 1
        l += [staff[start:end]]
    return l

    # return split_every(staff,every)


def get_numbers_from_string(string, force_float=False, process_negatives=True):
    """
    from a string (like a file name or such) it extracts all the possible numbers and return them in a list
    if process_negatives=False it does NOT process negative values (will be read as positives)
      else only those with a dash sign '-' right before will be processed as negative 
    numbers like .2 will become 2 for 0.2 it should be explicitly 0.2
    """
    candidates = []
    reading = False
    m = ""
    for ch in string:
        if ch.isdigit() or ch == ".":
            if reading:
                candidates[-1] += ch
            else:
                candidates += [m + ch]
                reading = True
        elif process_negatives and ch == "-":
            m = "-"
        else:
            m = ""
            reading = False
    numbers = []
    for ca in candidates:
        if ca[-1] == ".":
            ca = ca[:-1]
        if ca != "" and ca[0] == "." and len(ca) > 1:
            ca = ca[1:]
        ca, ok = convert_to_number(ca, force_float=force_float)
        if ok:
            numbers += [ca]
    return numbers


def to_number(string, force_float=False, allow_py3_underscores=False):
    """
    like convert_to_number() but does not return True/False as well
    this function check if a string is an int or a float and it 
    returns converted_string
    """
    if not allow_py3_underscores and "_" in string:
        return string
    if force_float:
        try:
            return float(string)
        except ValueError:
            return string
    try:
        return int(string)
    except ValueError:
        try:
            return float(string)
        except ValueError:
            return string


def convert_to_number(string, force_float=False):
    """
    this function check if a string is an int or a float and it returns a tuple in the form
    converted_string,bool. Bool is True if the sting has been converted, False if the  string is still in string format.
    the function is quite slow
    """
    if force_float:
        try:
            return float(string), True
        except ValueError:
            return string, False
    try:
        return int(string), True
    except ValueError:
        try:
            return float(string), True
        except ValueError:
            return string, False


# see if any of the elements in listt is contained into stringg
# note that stringg can also be a list of elements or a tuple (and probably listss can also be a string, but not tested)
#  if begin_with stingg has to begin with the element,
#  if end_with stringg has to end with the element.
#  if both are True element has to be either at the end or at the beginning
# it returns True and the first matching element or False and None
def loose_compare(stringg, listt, begin_with=False, end_with=False):
    """
    # see if any of the elements in listt is contained into stringg
    # note that stringg can also be a list of elements or a tuple (and probably listss can also be a string, but not tested)
    #  if begin_with stingg has to begin with the element, 
    #  if end_with stringg has to end with the element.
    #  if both are True element has to be either at the end or at the beginning
    # it returns True and the first matching element or False and None
    """
    le = len(stringg)
    if begin_with and end_with:
        for k in listt:
            if len(k) <= le:
                if stringg[: len(k)] == k or stringg[-len(k) :] == k:
                    return True, k
        return False, None
    if begin_with:
        for k in listt:
            if len(k) <= le:
                if stringg[: len(k)] == k:
                    return True, k
        return False, None
    if end_with:
        for k in listt:
            if len(k) <= le:
                if stringg[-len(k) :] == k:
                    return True, k
        return False, None
    for k in listt:
        if k in stringg:
            return True, k
    return False, None


def loose_compare_strict(stringg, listt, begin_with=False, end_with=False):
    """
    like loose_compare but with the additional criterion that the biggest element in listt will be returned
    e.g. if listt contains both 'a' and 'aa' and stringg also contains both (e.g. 'ciaaao') 'aa' will be returned as 'a' is also contained in 'aa'
    if on the other end listt contains 'a' and 'b' and both are in stringg the last one appearing in listt will be returned
     see if any of the elements in listt is contained into stringg
    # note that stringg can also be a list of elements or a tuple (and probably listss can also be a string, but not tested)
    #  if begin_with stingg has to begin with the element, 
    #  if end_with stringg has to end with the element.
    #  if both are True element has to be either at the end or at the beginning
    # it returns True and the first matching element or False and None
    """
    le = len(stringg)
    found, match = False, None
    if begin_with and end_with:
        for k in listt:
            if len(k) <= le:
                if stringg[: len(k)] == k or stringg[-len(k) :] == k:
                    found = True
                    if match is not None and k not in match:
                        match = k
                    elif match is None:
                        match = k
        return found, match
    if begin_with:
        for k in listt:
            if len(k) <= le:
                if stringg[: len(k)] == k:
                    found = True
                    if match is not None and k not in match:
                        match = k
                    elif match is None:
                        match = k
        return found, match
    if end_with:
        for k in listt:
            if len(k) <= le:
                if stringg[-len(k) :] == k:
                    found = True
                    if match is not None and k not in match:
                        match = k
                    elif match is None:
                        match = k
        return found, match
    for k in listt:
        if k in stringg:
            found = True
            if k not in match:
                match = k
            elif match is None:
                match = k
        return found, match


def fix_path(folder, make_absolute=False, make_folder=False):
    """
    given a folder name it returns the name in a format 
    such that the path to  a file can be written as 
    folder+filename (so if folder is '.' it becomes '' and otherwise a / is added at the end)
    """
    if make_absolute:
        f = os.path.abspath(folder)
        if f[-1] != "/":
            f += "/"
        if make_folder and not os.path.isdir(f):
            os.mkdir(f)
        return f
    if folder is None or folder == ".":
        folder = ""
    elif len(folder) > 1 and folder[-1] != "/":
        folder += "/"
    if make_folder and not os.path.isdir(folder):
        os.mkdir(folder)
    return folder


def get_file_path_and_extension(complete_filename):
    """
    given a complete path to a file (or just a file name) it returns (path,name,extension)
    """
    if "/" in complete_filename:
        path = complete_filename[: complete_filename.rfind("/") + 1]
        rest = complete_filename[complete_filename.rfind("/") + 1 :]
        if "." in rest:
            name = rest[: rest.find(".")]
            extension = rest[rest.find(".") :]
        else:
            name = rest
            extension = ""
    else:
        path = ""
        if "." in complete_filename:
            name = complete_filename[: complete_filename.find(".")]
            extension = complete_filename[complete_filename.find(".") :]
        else:
            name = complete_filename
            extension = ""
    return path, name, extension


def make_zipfile(output_filename, source_dir):
    """
    This function will recursively zip up a directory tree, compressing the files, 
    and recording the correct relative filenames in the archive. 
    The archive entries are the same as those generated by zip -r output.zip source_dir.
    """
    if output_filename[-1] == "/":
        output_filename = output_filename[:-1]
    if output_filename[-4:] != ".zip":
        output_filename += ".zip"
    relroot = os.path.abspath(os.path.join(source_dir, os.pardir))
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipp:
        for root, dirs, files in os.walk(source_dir):
            # print root,relroot,os.path.relpath(root, relroot),files
            # add directory (needed for empty dirs)
            zipp.write(root, os.path.relpath(root, relroot))
            for filename in files:
                arcname = os.path.join(os.path.relpath(root, relroot), filename)
                filename = os.path.join(root, filename)
                if os.path.isfile(filename):  # regular files only
                    # print arcname,filename
                    zipp.write(filename, arcname)
    # print output_filename
    return


def is_float(s):
    """
    # check if a string is convertible to a float number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def positive(x):
    """
    if x > 0 : return x
    else : return 0
    # useful when x is a sliding index of a list
    """
    if x > 0:
        return x
    return 0


def issorted(list_or_tuple, strick_inequalities=False, check_reverse_first=False):
    """
    check whether a list of a tuple is sorted, written to handle long list. If you plan to sort anyway just use default sort!
    it returns two bool corresponding to sorted,reverse
    strick_inequalities replace >= with >
    use check_reverse_first=True if you believe that the list you give should be sorted from smaller to larger
    if it is not sorted it returns False, None
    """
    if not strick_inequalities:
        if check_reverse_first:
            if all(
                list_or_tuple[i] >= list_or_tuple[i + 1]
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, True
            elif all(
                list_or_tuple[i] <= list_or_tuple[i + 1]
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, False
        else:
            if all(
                list_or_tuple[i] <= list_or_tuple[i + 1]
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, False
            elif all(
                list_or_tuple[i] >= list_or_tuple[i + 1]
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, True
    else:
        if check_reverse_first:
            if all(
                list_or_tuple[i] > list_or_tuple[i + 1]
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, True
            elif all(
                list_or_tuple[i] < list_or_tuple[i + 1]
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, False
        else:
            if all(
                list_or_tuple[i] < list_or_tuple[i + 1]
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, False
            elif all(
                list_or_tuple[i] > list_or_tuple[i + 1]
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, True
    return False, None


def iscontiguous(list_or_tuple, tolearte_equalities=False, check_reverse_first=False):
    """
    check whether a list of a tuple is contiguous, written to handle long list. 
    it returns two bool corresponding to contiguous,reverse
    tolearte_equalities return True aslo if there are contiguous identical elements
    use check_reverse_first=True if you believe that the list you give should be sorted from smaller to larger
    if it is not contiguous it returns False, None
    """
    if not tolearte_equalities:
        if check_reverse_first:
            if all(
                list_or_tuple[i] == list_or_tuple[i + 1] + 1
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, True
            elif all(
                list_or_tuple[i] == list_or_tuple[i + 1] - 1
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, False
        else:
            if all(
                list_or_tuple[i] == list_or_tuple[i + 1] - 1
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, False
            elif all(
                list_or_tuple[i] == list_or_tuple[i + 1] + 1
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, True
    else:
        if check_reverse_first:
            if all(
                list_or_tuple[i] == list_or_tuple[i + 1]
                or list_or_tuple[i] == list_or_tuple[i + 1] + 1
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, True
            elif all(
                list_or_tuple[i] == list_or_tuple[i + 1]
                or list_or_tuple[i] == list_or_tuple[i + 1] - 1
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, False
        else:
            if all(
                list_or_tuple[i] == list_or_tuple[i + 1]
                or list_or_tuple[i] == list_or_tuple[i + 1] - 1
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, False
            elif all(
                list_or_tuple[i] == list_or_tuple[i + 1]
                or list_or_tuple[i] == list_or_tuple[i + 1] + 1
                for i in range(len(list_or_tuple) - 1)
            ):
                return True, True
    return False, None


def contiguous_regions(condition):
    """
    Finds contiguous True regions of the boolean 1D array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    """
    # Find the indicies of changes in "condition"
    d = numpy.diff(condition.astype(int))
    (idx,) = d.nonzero()
    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1
    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = numpy.r_[0, idx]
    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = numpy.r_[idx, condition.size]  # Edit
    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def contiguous_split(list_of_int, reverse=False):
    """
    split a list of int into contigous bits
    [1,2,3,4,6,7,8,11,13,15,16] -> [[1, 2, 3, 4], [6, 7, 8], [11], [13], [15, 16]]
    """
    if reverse:
        a = -1
    else:
        a = 1
    conts = [[]]
    for j, i in enumerate(list_of_int[:-1]):
        if conts[-1] == []:
            conts[-1] += [i]
        if a * (list_of_int[j + 1] - i) == 1:
            conts[-1] += [list_of_int[j + 1]]
        else:
            conts += [[]]
    if conts[-1] == []:
        conts[-1] += [list_of_int[-1]]
    return conts


def get_bootstrap_ensemble(
    outcomes, bootstrap_runs=10000, return_only_count=False, outcomes_are_counter=False
):
    """
    if return_only_count is True return count,un
       count is a matrix where each row is the number of occurrences of each element in un
        across the different bootstrap_runs (thus there are bootstrap_runs columns)
        count_SE=numpy.std(count,axis=1) # standard errors on the counts (for each element in un)
        ci25,ci975= numpy.percentile(count, [2.5,97.5], axis=1)  # get 95% confidence interval on the count 
            ci25 will be an array with the 2.5% confidence interval of each element in un with the same order
          get frequencies:
            freq=count/count.sum(axis=0).astype(float)
            and get the SE and CI as above but giving freq instead of count
            
    
    else return sample, where each row is a realization of bootstrap
        both count and sample are formatted so that 
        bootstrap_means=numpy.mean( count or sample ,axis=1)
    e.g. 
    sample=misc.get_bootstrap_ensemble( outcomes )
    bootstrap_medians=numpy.median( sample,axis=1) # all the medians
    bootsrap_SEM=numpy.std(bootstrap_medians) # standard error on the median
    median_25,median_975= numpy.percentile( bootstrap_medians , [2.5,97.5]) # get 95% confidence interval
    """
    if outcomes_are_counter and not return_only_count:
        sys.stderr.write(
            "**Warn** in get_bootstrap_ensemble() return_only_count=False but outcomes_are_counter \n"
        )
    if outcomes_are_counter:
        isd = False
        if type(outcomes) is dict or isinstance(outcomes, OrderedDict):
            isd = True
            keys, outcomes = list(outcomes.keys()), list(outcomes.values())
        c = []
        for j, v in enumerate(outcomes):
            c += [
                j
            ] * v  # make one list of j (0, 1, 2,...), where each number appears count times.
        outcomes = c[:]
    if type(bootstrap_runs) is not int:
        bootstrap_runs = max([100000, 10 ** 6 / float(len(outcomes)) + 200])
    ens = numpy.array(outcomes)
    sample_size = len(outcomes)
    choice = numpy.random.random_integers(
        0, sample_size - 1, (bootstrap_runs, sample_size)
    )
    sample = ens[choice]
    if return_only_count:
        un = uniq(outcomes)
        un.sort()  # if outcomes_are_counter and it was a list un contains the indeces and is already sorted
        count = process_bootstrap_count(sample, un)
        if outcomes_are_counter and isd:
            for i, j in enumerate(un[:]):
                un[i] = keys[j]
        return count, un
    return sample


def process_bootstrap_count(sample, sorted_uniqs):
    count = None
    for u in sorted_uniqs:
        tot = (sample == u).sum(axis=1)  # sum along the horizontal axis
        if count is None:
            count = tot
        else:
            count = numpy.vstack((count, tot))
    return count


def percentile_to_CI(values, low_percentile, high_percentile):
    """
    can be used to readily convert a confidence interval returned from bootstrap
    to error bars for matplotlib.
    """
    return (values - low_percentile, high_percentile - values)


def bootstrap(
    outcomes,
    bootstrap_runs=10000,
    return_only_count=True,
    return_all_means_and_medians=False,
):
    """
    performs bootstrap on a given set of outcomes
     if return_only_count is True it assumes that the set of outcomes contain some variable (like 0 and 1 for example)
      and that only their number of occurrences is important.
    in this case it RETURNS un,bootstrap_mean,bootstrap_std where un is a list with the possibilities (it would be [0,1] in the example)
     otherwise it treats them as measurements
    it returns bootstrap_mean,bootstrap_std 
    
    e.g. a summary for 100,000 resamples:
        The SD of the 100,000 means = 3.46; this is the bootstrapped SE of the mean (SEM).
        The SD of the 100,000 medians = 4.24; this is the bootstrapped SE of the median.
        The 2.5th and 97.5th centiles of the 100,000 means = 94.0 and 107.6; these are the bootstrapped 95% confidence limits for the mean.
        The 2.5th and 97.5th centiles of the 100,000 medians = 92.5 and 108.5; these are the bootstrapped 95% confidence limits for the median.
    return_all_means_and_medians can be useful if you also wish to calculate confidence level intervals
        median_25,median_975= numpy.percentile( bootstrap_medians , [2.5,97.5]) # get 95% confidence interval
        mean_25,mean_975= numpy.percentile( bootstrap_means , [2.5,97.5]) # get 95% confidence interval
    """
    ens = numpy.array(outcomes)
    if return_only_count:
        un = uniq(outcomes)
        un.sort()

    sample_size = len(outcomes)
    choice = numpy.random.random_integers(
        0, sample_size - 1, (bootstrap_runs, sample_size)
    )
    sample = ens[choice]
    if return_only_count:
        count = None
        for u in un:
            tot = (sample == u).sum(axis=1)  # sum along the horizontal axis
            if count is None:
                count = tot
            else:
                count = numpy.vstack((count, tot))
        bootstrap_mean = numpy.mean(count, axis=1)
        bootstrap_std = numpy.std(count, axis=1)
        return un, bootstrap_mean, bootstrap_std
    else:
        bootstrap_means = numpy.mean(sample, axis=1)
        bootstrap_mean = numpy.mean(sample)
        bootsrap_SE = numpy.std(bootstrap_means)
        bootstrap_medians = numpy.median(sample, axis=1)
        bootstrap_median = numpy.median(sample)
        bootsrap_SEM = numpy.std(bootstrap_medians)
        if return_all_means_and_medians:
            return (
                bootstrap_mean,
                bootsrap_SE,
                bootstrap_median,
                bootsrap_SEM,
                bootstrap_means,
                bootstrap_medians,
            )
        return bootstrap_mean, bootsrap_SE, bootstrap_median, bootsrap_SEM


def precision_accuracy_and_MCC(TP, FP, TN, FN):
    """
    returns the precision, the balanced accuracy (Acc) and the Mattew correlation coefficient for binary prediction.
    TP=True positives  --> correctly predicted positives (e.g. a disordered residues correctly predicted)
    FP=False positives --> should be negative but is predicted as positive (e.g. an ordered residue predicted disordered).
    TN=True negatives  --> correctly predicted negatives  (e.g. an ordered residues correctly predicted)
    FN=False negatives --> should be positive but is predicted as negative (e.g. a disordered residue predicted ordered).
    """
    prec = TP / float(TP + FP)
    acc = 0.5 * (TP / float(TP + FN) + TN / float(TN + FP))
    MCC = float(TP * TN - FP * FN) / numpy.sqrt(
        (TP + FP) * (TN + FN) * (TP + FN) * (TN + FP)
    )
    return prec, acc, MCC


def gtest(f_obs, f_exp=None, ddof=0):
    """
    http://en.wikipedia.org/wiki/G-test
 
    The G test can test for goodness of fit to a distribution
 
    Parameters
    ----------
    f_obs : array
        observed frequencies in each category
    f_exp : array, optional
        expected frequencies in each category.  By default the categories are
        assumed to be equally likely.
    ddof : int, optional
        adjustment to the degrees of freedom for the p-value
 
    Returns
    -------
    chisquare statistic : float
        The chisquare test statistic
    p : float
        The p-value of the test.
 
    Notes
    -----
    The p-value indicates the probability that the observed distribution is
    drawn from a distribution given frequencies in expected.
    So a low p-value inidcates the distributions are different.
 
    Examples
    --------
 
    >>> gtest([9.0, 8.1, 2, 1, 0.1, 20.0], [10, 5.01, 6, 4, 2, 1])
    (117.94955444335938, 8.5298516190930345e-24)
 
    >>> gtest([1.01, 1.01, 4.01], [1.00, 1.00, 4.00])
    (0.060224734246730804, 0.97033649350189344)
 
    >>> gtest([2, 1, 6], [4, 3, 2])
    (8.2135343551635742, 0.016460903780063787)
 
    References
    ----------
 
    http://en.wikipedia.org/wiki/G-test
    """
    f_obs = numpy.asarray(f_obs, "f")
    k = f_obs.shape[0]
    if f_exp is None:
        f_exp = numpy.array([numpy.sum(f_obs, axis=0) / float(k)] * k, "f")
    else:
        numpy.asarray(f_exp, "f")
    g = 2 * numpy.add.reduce(f_obs * numpy.log(f_obs / f_exp))
    return g, scipy.stats.chisqprob(g, k - 1 - ddof)


def chi2_test(obs, exp, dof=None, obs_err=None):
    """
    dof=degree of freedom
    obs_err should be the standard deviation (or error)
      if both obs and exp have an error then - if you know the fitting line y=f(x) - use the formula for the equivalent error
        obs_err= sqrt(  obs_err**2 + ( df/dx * exp_err )**2 )
    return chi2, pvalue, chi2_rid
    """
    if dof is None:
        dof = len(obs) - 1
    if obs_err is None:
        chi2, p = scipy.stats.chisquare(obs, exp, ddof=len(obs) - 1 - dof)
        return chi2, p, chi2 / float(dof)
    if not hasattr(obs_err, "__len__"):
        obs_err = [obs_err] * len(obs)
    f_obs = numpy.asarray(obs, "f")
    f_exp = numpy.asarray(exp, "f")
    f_obs_err = numpy.array(obs_err, "f")
    chi2 = ((f_obs - f_exp) ** 2 / f_obs_err ** 2).sum()
    return chi2, scipy.stats.chisqprob(chi2, dof), chi2 / float(dof)


def find_nearest(numpy_array, value, index=0):
    """
    given a numpy array it finds in it the element closest to value.
    it returns the element and its index
    index=0 can be given to search only after it
    """
    if hasattr(value, "__len__"):
        idxx = numpy.array(
            [index + (numpy.abs(numpy_array[index:] - val)).argmin() for val in value]
        )
        return numpy_array[idxx], idxx
    idx = index + (numpy.abs(numpy_array[index:] - value)).argmin()
    return numpy_array[idx], idx


def merge_ranges(ranges):
    """
    ranges should be a list of pairs (start, end) and
    it returns again a list of pairs but obtaine by merging overlapping
    pairs from the above
    """
    if len(ranges) <= 1:
        return ranges
    ranges.sort()  # should sort from starting point of the fragments
    new_ranges = [ranges[0]]
    j = 0
    for r in ranges[1:]:
        if r[0] <= new_ranges[j][1]:
            if r[1] > new_ranges[j][1]:
                new_ranges[j] = [new_ranges[j][0], r[1]]
        else:
            new_ranges += [r]
            j += 1
    return uniq(new_ranges)


def split_at_index(list_, indices):
    """
    # parts a list at indices
    # e.g. s=[3,4,5,9,8,7,6,5,4,3,2] parts(s,[1,5])-->[[3], [4, 5, 9, 8], [7, 6, 5, 4, 3, 2]]
    """
    indices = [0] + list(indices) + [len(list_)]
    return [list_[v : indices[k + 1]] for k, v in enumerate(indices[:-1])]


# insert a new line into string every every characters
def insert_newlines(string, every=75):
    if every <= 0:
        return string
    lines = []
    for i in range(0, len(string), every):
        lines.append(string[i : i + every])
    return "\n".join(lines)


def split_every(staff, num):
    """
    # split a string a list or a tuple every num elements
    """
    return [staff[start : start + num] for start in range(0, len(staff), num)]


def split_every_str_advanced(staffstr, num):
    """
    # split a string a list or a tuple every num elements considering some special character - may be system specific
    examples are:
    count_as_none=['\033[95m','\033[96m','\033[36m','\033[94m','\033[92m','\033[93m','\033[91m','\033[1m','\033[4m','\033[0m']
    """
    if num <= 0:
        return staffstr
    bits = [""]
    i, j = 0, 0
    while j < len(staffstr):
        if i == num:
            bits += [""]
            i = 0
        if staffstr[j] == "\x1b" or staffstr[j] == "\033":
            if (
                staffstr[j + 3] == "m"
                and staffstr[j + 1] == "["
                and staffstr[j + 2].isdigit()
            ):
                bits[-1] += staffstr[j : j + 4]
                j += 4
            elif (
                staffstr[j + 4] == "m"
                and staffstr[j + 1] == "["
                and staffstr[j + 3].isdigit()
                and staffstr[j + 2].isdigit()
            ):
                bits[-1] += staffstr[j : j + 5]
                j += 5
            else:
                sys.stderr.write(
                    "**Warn** in split_every_str_advanced() special string beginning with %s at char j=%d (i=%d) not recognised ([j:j+7]=%s) - adding normally\n"
                    % (repr(staffstr[j]), j, i, repr(repr(staffstr[j : j + 7])))
                )
                bits[-1] += staffstr[j]
                j += 1
        else:
            bits[-1] += staffstr[j]
            i += 1
            j += 1
    return bits


def split_file_in_blocks(wholefile, every=200, use_advance_string_splitting=False):
    # split a file in blocks. It first makes all the lines to the same length, then it insert a new line every every chars
    if every <= 0:
        return wholefile
    if type(wholefile) is str:
        lines = wholefile.split("\n")
    else:
        lines = wholefile
    max_len = max([len(l) for l in lines])
    spl_lines = []
    for l in lines:
        if l == "":
            continue
        if len(l) < max_len:
            l += " " * (max_len - len(l))
        if use_advance_string_splitting:
            spl = split_every_str_advanced(l, every)
        else:
            spl = split_every(l, every)
        spl_lines.append(spl)
    spl_lines = list(zip(*spl_lines))
    wh = ""
    for group in spl_lines:
        wh += "\n".join(group) + "\n"
    return wh


def venn_diagram_tables(iterable_of_lists):
    """
    return categories
    a dictionary whose keys are tuple of indices in iterable_of_lists (or tuples of its keys if it is a dict)
     and values are the elements in that group
      for instance
    categories[(0)]=[...] categories[(0,1)]=[...] and len(categories[(0)])=10 and len(categories[(0,1)])=8 
       means that there are 10 elements that are exclusively in the first list of iterable_of_lists
       and 8 that are found both in the first and the second (index 1) 
    """
    if hasattr(iterable_of_lists, "values"):
        iteron = list(iterable_of_lists.values())
        kes = list(iterable_of_lists.keys())
    else:
        iteron = iterable_of_lists
        kes = list(range(0, len(iterable_of_lists)))
    categories = {}
    assign = []
    for j, l1 in enumerate(iteron):
        for el in l1:
            if el in assign:
                continue
            cat = [kes[j]]
            for i, l2 in enumerate(iteron):
                if i == j:
                    continue
                if el in l2 and kes[i] not in cat:
                    cat += [kes[i]]
            cat = tuple(sorted(cat)[:])  # make hashable
            if cat not in categories:
                categories[cat] = [el]
            else:
                categories[cat] += [el]
            assign += [el]
    return categories


# return the bin corresponding to input_number in the range [minimum, maximum].Note that reciprocal_of_bin_size is Number_of_bins/(maximum-minimum)
def GetBin(
    input_number, minimum, maximum, reciprocal_of_bin_size, strict_extremes=False
):
    """
    WARNING not working well, use numpy instead (much faster):
    els=numpy.array(elements)
    numpy.digitize(els,numpy.linspace(Min,Max,nbins))
    above first bin will be 1 and last will be nbins, add -1 if 0 to nbins-1 is needed.
    """
    if input_number > maximum:  # CompareFloats(input_number,maximum, 0.0000001) :
        if strict_extremes:
            return -1
        return int(
            reciprocal_of_bin_size * (maximum - minimum) - 1
        )  # in this particular case return nbin-1
    elif input_number < minimum:
        if strict_extremes:
            return -1
        return 0
    return int(reciprocal_of_bin_size * (input_number - minimum))


def GetBinFast(input_number, minimum, maximum, reciprocal_of_bin_size):
    """
    WARNING not working well, use numpy instead (much faster):
    els=numpy.array(elements)
    numpy.digitize(els,numpy.linspace(Min,Max,nbins))
    """
    bin_indx = numpy.array(reciprocal_of_bin_size * (input_number - minimum)).astype(
        "int"
    )
    nbins = int(reciprocal_of_bin_size * (maximum - minimum))
    bin_indx[bin_indx >= nbins] = nbins - 1  # if beyond put in the last bin?????
    return bin_indx


def GetBinIndex(all_entries, bins_array):
    """
    returns the bin index of each entries in an array with the same shape of all_entries
    """
    if len(all_entries.shape) > 1 and all_entries.shape[1] > 0:
        res = None
        for row in all_entries:
            if res is None:
                res = numpy.digitize(row, bins_array)
            else:
                res = numpy.vstack((res, numpy.digitize(row, bins_array)))
        return res - 1  # -1 is there so that the first bin is bin 0
    return numpy.digitize(
        all_entries, bins_array, right=True
    )  # (right=True is such that if the last element of bin_array is the max of the distribution then it will be in last bin rather than in nbins+1 but if right=True and first bin val is minimum then it gets -1 whic is even worse)


def getCDF2(list_of_input, normalized=True, smooth=True):
    """
    bin-free, (should be plotted as line in profile)
    return cdf, x_axis 
    """
    list_of_input = numpy.array(list_of_input)
    # the below also sorts the x_axis
    x_axis, counts = numpy.unique(
        list_of_input,
        return_index=False,
        return_inverse=False,
        return_counts=True,
        axis=-1,
    )
    cdf = numpy.cumsum(counts)
    if normalized:
        cdf = cdf / float(
            list_of_input.shape[-1]
        )  # divide by number of elements so ends with 1.
    return cdf, x_axis


def getCDF_and_PDF(list_of_input, normalized=True, smooth=True):
    """
    bin-free, previously getCDF2 (should be plotted as line in profile)
NOT WORKING smooth true is crucial (only used for pdf) unless all your input values are identical in groups.
    return cdf, pdf, x_axis, cdfsmoothed
    """
    list_of_input = numpy.array(list_of_input)
    # the below also sorts the x_axis
    x_axis, counts = numpy.unique(
        list_of_input,
        return_index=False,
        return_inverse=False,
        return_counts=True,
        axis=-1,
    )
    cdf = numpy.cumsum(counts)
    if smooth:
        cdfsm = scipy.signal.savgol_filter(
            cdf,
            window_length=max([5, 2 * int(len(cdf) / 6.0) + 1]),
            polyorder=2,
            deriv=0,
            delta=1.0,
            axis=-1,
            mode="interp",
            cval=0.0,
        )
        pdfD = numpy.diff(cdfsm) / numpy.diff(x_axis)
    else:
        pdfD = numpy.diff(cdf) / numpy.diff(x_axis)
    pdf = numpy.hstack(([counts[0]], len(list_of_input) * pdfD / pdfD.sum()))
    # pdf= numpy.diff(cdf)/numpy.diff(x_axis) # however this makes a pdf of one element less so it would start from x_axis[1:]
    if normalized:
        cdf = cdf / float(
            list_of_input.shape[-1]
        )  # divide by number of elements so ends with 1.
        cdfsm = cdfsm / float(list_of_input.shape[-1])
        pdf /= pdf.sum()  # divide like this so sums to 1.
    return cdf, pdf, x_axis, cdfsm


def getCDF(
    list_of_input,
    weights=None,
    normalized=True,
    bootstrap_errors=True,
    x_axis_CI_percentiles=[2.5, 97.5],
    axis=-1,
):
    """
    see also getCDF_and_PDF for bin-free approach 
    return cdf , x_axis , cdf_SE, cdf_CI_25, cdf_CI_975  if bootstrap_errors is False cdf_SE, cdf_CI_25, cdf_CI_975 are all None
    on the confidence interval you can use yerr=misc.percentile_to_CI(cdf,cdf_CI_25,cdf_CI_975) to convert into error bars
    NB bootstrap errors returned are approximation (valid for relatively smooth PDF without sharp peaks). 
       One should implement an alignment of each re-sampled cdf on its own x-axis and estimate the errors at the x-value level
    axis may not work in bootstrap
    """
    list_of_input = numpy.array(list_of_input)
    args = numpy.argsort(list_of_input, axis=axis)
    x_axis = list_of_input[args]
    if weights is not None:
        cdf = numpy.cumsum(numpy.array(weights)[args], axis=axis)
        if normalized:
            cdf = cdf / sum(weights)
    else:
        N = list_of_input.shape[-1]
        # cdf=numpy.arange(1,N+1,dtype='f') PROBABLY I should be using this with 1, +1 but on the forum they use 0,N
        cdf = numpy.arange(0, N, dtype="f")
        if normalized:
            cdf /= float(N)
    if bootstrap_errors:
        if type(bootstrap_errors) is int:
            cycles = bootstrap_errors
        else:
            cycles = 10000
        sample = get_bootstrap_ensemble(
            list_of_input, bootstrap_runs=cycles, return_only_count=False
        )
        sample = numpy.sort(sample, axis=1)
        x_axis_CI_25, x_axis_CI_975 = numpy.percentile(
            sample, x_axis_CI_percentiles, axis=0
        )  # get 95%
        x_axis_SE = numpy.std(
            sample, axis=0
        )  # standard error along x, note that this will be symmetrical
        del sample
    else:
        x_axis_SE, x_axis_CI_25, x_axis_CI_975 = None, None, None
    return cdf, x_axis, x_axis_SE, x_axis_CI_25, x_axis_CI_975


# return a list of length Number_of_bins corresponding to the histogram obtained from list_of_input. One can enter maximum or minimum if he wants to consider only entries of list of inputs that fall within the given range. if the histogram is needed normalized one can set normalized to True
def getPDF(
    list_of_input,
    nbins=None,
    weights=None,
    minimum=None,
    maximum=None,
    normalized=True,
    count_like_normalization=False,
    QUIET=False,
    return_x_axis=True,
    return_cumulative_and_error=False,
    bootstrap_errors=False,
    log_bins=False,
):
    """
    returns y, x of the PDF, x are the boundaries of the bins, and pdf of the first x (pdf[0]) is always 0 (actually the minimum entry could contribute to pdf[0] if minimum and maximum are left to None).
      while the pdf of the second x (pdf[1]) is the probability of an entry to be between x[0] and x[1].
      if return_x_axis you can move this probability to the centre of the bin by doing x_axisMID=(x_axis[:-1]+x_axis[1:])/2. and remove the 0 pdf=pdf[1:] (remove first 0). However
      this should not be done for the cumulative distribution as that is computed at the end of the bins!!.
    Use getCDF to obtain a bin-independent cumulativ distribution.
     Can give minimum or maximum to confine the PDF in a given range.
    The cumulative distribution returned is the probability at the end of each bin, not at the beginning!
    And so is the PDF
    If all the options are set to True it returns 
    pdf , x_axis, cumulative,cum_errs, histo_SE,histo_25,histo_975,cumulative_SE,cumulative25,cumulative975
    if bootstrap_errors=False it returns
    pdf , x_axis, cumulative,cum_errs
    """
    if bootstrap_errors == True:
        if type(bootstrap_errors) is bool:
            cycles = max([600, int(250000.0 / len(list_of_input) + 50.0)])
            print("doing %d bootstrap cycles" % (cycles))
        else:
            cycles = bootstrap_errors

    if nbins is None:
        if len(list_of_input) > 25:
            nbins = 2 * int(
                numpy.sqrt(1.0 * len(list_of_input))
            )  # use numpy.sqrt rule (like excel)
        else:
            nbins = int(len(list_of_input) // 3)

    num_entries = len(list_of_input)
    if num_entries == 0:
        return -1
    if num_entries <= nbins:
        print(
            "***WARNING*** in CreateHistogram number of inputs is smaller than requested number of bins"
        )

    list_of_input = numpy.array(list_of_input)
    if minimum is not None and maximum is not None:
        list_of_input = list_of_input[
            (list_of_input >= minimum) & (list_of_input <= maximum)
        ]
    elif minimum is not None:
        list_of_input = list_of_input[(list_of_input >= minimum)]
    elif maximum is not None:
        list_of_input = list_of_input[(list_of_input <= maximum)]
    if minimum is None:
        minimum = min(list_of_input)

    if maximum is None:
        maximum = max(list_of_input)

    if log_bins:
        bins_array = 10 ** numpy.linspace(
            numpy.log10(minimum), numpy.log10(maximum), nbins + 1
        )  # these are the bins limits so +1 is necessary to get nbins intervals
    else:
        bins_array = numpy.linspace(minimum, maximum, nbins + 1)
    d = float(num_entries)
    if weights is not None:
        d = float(sum(weights))

    bin_index = GetBinIndex(
        list_of_input, bins_array
    )  # bin_index has same shape of list_of_input and indices (of bins) ranging from 0 to nbins
    # count=[0] # add an empty at the beginning (the pdf will be a profile as an x_axis we return something that begins with the minimum and there the pdf is 0and at the maximum is the probability of being in the last bin)
    count = [0] * len(bins_array)  # numpy.zeros(bins_array.shape)
    if weights is not None:
        if len(weights) != len(list_of_input):
            print(
                "ERROR in getPDF with weights len(weights)!=len(list_of_input) %d %d"
                % (len(weights), len(list_of_input))
            )
        for j, bi in enumerate(bin_index):
            count[bi] += weights[j]
    else:
        for i in range(nbins):
            count[i] = (bin_index == i).sum()

    print(
        "bins_array",
        bins_array.shape,
        "nbins",
        nbins,
        "bin_index",
        bin_index.shape,
        bin_index,
    )
    # GetBinFast(list_of_input, minimum,maximum,reciprocal_of_bin_size)
    # count=[0] # add an empty at the beginning (the pdf will be a profile as an x_axis we return something that begins with the minimum and there the pdf is 0and at the maximum is the probability of being in the last bin)
    # for i in xrange(nbins) :
    #    count.append( (bin_index==i).sum() )
    count = numpy.array(count)
    print("count", count.shape)
    if bootstrap_errors:
        boot_err = []
        sample = get_bootstrap_ensemble(
            list_of_input, bootstrap_runs=cycles, return_only_count=False
        )
        bin_index = GetBinIndex(sample, bins_array)
        count_sample = process_bootstrap_count(bin_index, list(range(0, nbins)))
        count_sample = numpy.vstack(
            (numpy.zeros(count_sample.shape[1]), count_sample)
        )  # add a 0 at the beginning as for x== minimum the cumulative is 0
        # boot_counts=numpy.mean(count_sample,axis=1)
        count_SE = numpy.std(count_sample, axis=1)
        count25, count975 = numpy.percentile(count_sample, [2.5, 97.5], axis=1)
        # print 'count_sample',repr(count_sample),count_sample.shape
        if (
            return_cumulative_and_error
        ):  # used only to get the errors not the cumulative function
            cum = numpy.zeros(count_sample.shape[1])
            cumulative = None
            for binc in count_sample:
                cum += binc / d  # this would be the cum at the end of the bin.
                if cumulative is None:
                    cumulative = numpy.zeros(
                        count_sample.shape[1]
                    )  # = does not copy a numpy array!!!! It casts it, so if I had left cumulative=cum than it would had become cum of the next for cycle.
                else:
                    cumulative = numpy.vstack((cumulative, cum))
            # print 'cumulative',repr(cumulative),cumulative.shape,d,count_sample[0]+numpy.zeros(count_sample.shape[1])/d
            # cumulative=numpy.vstack((numpy.zeros(count_sample.shape[1]), cumulative)) # add a 0 at the beginning as for x== minimum the cumulative is 0
            cumulative_SE = numpy.std(cumulative, axis=1)
            cumulative25, cumulative975 = numpy.percentile(
                cumulative, [2.5, 97.5], axis=1
            )
            boot_err = [cumulative_SE, cumulative25, cumulative975]

    if not QUIET:
        print(
            "generated histogram with %d entries in the range [%lf , %lf]"
            % (num_entries, minimum, maximum)
        )
    if normalized and num_entries > 0:
        if count_like_normalization:

            histogram = count / d
            if bootstrap_errors:
                count_SE /= d
                count25 /= d
                count975 /= d
                boot_err = [count_SE, count25, count975] + boot_err
        else:
            den = float(count.sum())
            if len(count.shape) > 1:
                ts = numpy.ones(
                    (1, count.shape[1])
                )  # we add a 1 at the denominator (the zeroth numerator should always be zero)
            else:
                ts = numpy.array([1])
            histogram = count / (numpy.hstack((ts, numpy.diff(bins_array))) * den)
            if bootstrap_errors:
                count_SE /= numpy.hstack((ts, numpy.diff(bins_array))) * den
                count25 *= numpy.hstack((ts, numpy.diff(bins_array))) * den
                count975 *= numpy.hstack((ts, numpy.diff(bins_array))) * den
                boot_err = [count_SE, count25, count975] + boot_err
    else:
        histogram = count
    rtrn = []
    if return_x_axis:
        x_axis = bins_array  # the pdf is 0 at the minimum and at the maximum is the probability of being in the last bin
        # x_axis=(bins_array[:-1]+bins_array[1:])/2. # save bin midpoints.
        # for bi in range(0,nbins+1) :
        #    x_axis+=[ minimum+float(bi)/reciprocal_of_bin_size ]
        rtrn += [x_axis]
    if return_cumulative_and_error:
        cum = 0
        cumulative = []
        errs = [[], []]
        for j, c in enumerate(count):
            cum += c
            cumulative += [cum / d]  # this is the probability of the end of the bin
            if cum == 0 or cum == d:  # otherwise division by 0 or log(0)
                errs[0] += [0]
                errs[1] += [0]
            else:
                sig = numpy.sqrt(
                    d / (cum * (d - cum))
                )  # http://www.inference.phy.cam.ac.uk/mackay/itprnn/noteCumulative.ps
                up = 1.0 / (1.0 + numpy.exp(-(numpy.log(cum / (d - cum)) + sig)))
                do = 1.0 / (1.0 + numpy.exp(-(numpy.log(cum / (d - cum)) - sig)))
                errs[0] += [cumulative[-1] - do]
                errs[1] += [up - cumulative[-1]]
        rtrn += [cumulative, errs]
    if bootstrap_errors:
        rtrn += boot_err
    if rtrn != []:
        return [histogram] + rtrn
    return histogram


# return a list of length Number_of_bins corresponding to the histogram obtained from list_of_input. One can enter maximum or minimum if he wants to consider only entries of list of inputs that fall within the given range. if the histogram is needed normalized one can set normalized to True
def getPDFOLD(
    list_of_input,
    nbins=None,
    minimum=None,
    maximum=None,
    normalized=True,
    count_like_normalization=False,
    QUIET=False,
    return_x_axis=True,
    return_cumulative_and_error=False,
    bootstrap_errors=True,
):
    """
    returns y, x of the PDF, nbins values in total
     Can give minimum or maximum to confine the PDF in a given range.
    """
    if bootstrap_errors == True:
        if type(bootstrap_errors) is bool:
            cycles = int(150000.0 / len(list_of_input) + 50.0)
        else:
            cycles = bootstrap_errors
    if nbins is None:
        if len(list_of_input) > 25:
            nbins = 2 * int(
                numpy.sqrt(1.0 * len(list_of_input))
            )  # use numpy.sqrt rule (like excel)
        else:
            nbins = int(len(list_of_input) // 3)
    count = [0] * nbins  # allocate list of zeroes
    N = len(list_of_input)
    if N == 0:
        return -1
    if N <= nbins:
        print(
            "***WARNING*** in CreateHistogram number of inputs is smaller than requested number of bins"
        )
    if minimum is None:
        minimum = min(list_of_input)
    if maximum is None:
        maximum = max(list_of_input)

    num_entries = 0
    reciprocal_of_bin_size = float(nbins) / (maximum - minimum)
    for entry in list_of_input:  # fill histogram
        i = GetBin(float(entry), minimum, maximum, reciprocal_of_bin_size)
        if i >= 0:
            count[i] += 1
            num_entries += 1

    if not QUIET:
        print(
            "generated histogram with %d entries in the range [%lf , %lf]"
            % (num_entries, minimum, maximum)
        )
    if normalized and num_entries > 0:
        if count_like_normalization:
            histogram = [1.0 * x / float(num_entries) for x in count]
        else:
            den = float(sum(count))
            histogram = [reciprocal_of_bin_size * x / den for x in count]
    else:
        histogram = count
    if return_cumulative_and_error:
        d = float(sum(count))
        cum = 0
        cumulative = []
        errs = [[], []]
        for j, c in enumerate(count):
            cum += c
            cumulative += [cum / d]  # this is the probability of the end of the bin
            if cum == 0 or cum == d:
                errs[0] += [0]
                errs[1] += [0]
            else:
                sig = numpy.sqrt(
                    d / (cum * (d - cum))
                )  # http://www.inference.phy.cam.ac.uk/mackay/itprnn/noteCumulative.ps
                up = 1.0 / (1.0 + numpy.exp(-(numpy.log(cum / (d - cum)) + sig)))
                do = 1.0 / (1.0 + numpy.exp(-(numpy.log(cum / (d - cum)) - sig)))
                errs[0] += [cumulative[-1] - do]
                errs[1] += [up - cumulative[-1]]
        if not return_x_axis:
            return histogram, cumulative, errs
    if return_x_axis:
        x_axis = []
        for bi in range(0, nbins):
            x_axis += [minimum + float(bi + 1) / reciprocal_of_bin_size]
        if return_cumulative_and_error:
            return histogram, x_axis, cumulative, errs
        return (histogram, x_axis)
    return histogram


def cumulative_from_pdf(pdf, binsize=None):
    if binsize is None:
        binsize = 1
    h = []
    for j in range(len(pdf)):
        h += [binsize * sum(pdf[: j + 1])]
    return h


def log_normal_dist(x, pars):
    mu = pars[0]
    sig = pars[1]
    return (
        1.0
        / (x * sig * 2.5066282746)
        * numpy.exp(-((numpy.log(x) - mu) ** 2) / (2.0 * sig * sig))
    )


def gamma_dist(x, pars):
    return (x ** (pars[0] - 1) * numpy.exp(-x / pars[1])) / (
        pars[1] ** pars[0] * scipy.special.gamma(pars[0])
    )


# check if a string is in a list or in a dictionary (in in_here) and modifies it by adding an int till is not in in_here
# it can be used to check if file/folder are already present by giving in_here=os.listdir('.')
def change_if_already_there(string, in_here, stderr_out=False):
    if string in in_here:
        i = 0
        if stderr_out:
            sys.stderr.write(
                "WARN in change_if_already_there() name %s already present," % (string)
            )
        while True:
            c = string + "%02d" % (i)
            if c not in in_here:
                break
            i += 1
        string = c
        if stderr_out:
            sys.stderr.write(" changing to %s\n" % (string))
    return string


# print the histogram contained in list_with_histogram in a tab separated file with bin_number value_of_histogram.
# One can set write_mode to append 'a', in which case a double empty line is added before printing the histogram (newplot format)
# Cumulative appends the cumulative distribution below the histogram in the file
def PrintHistogram(
    filename,
    list_with_histogram,
    minimum,
    maximum,
    write_mode="w",
    HEADER=None,
    CUMULATIVE=False,
):
    out = open(filename, write_mode)
    if write_mode == "a":
        out.write("\n\n")
    N = len(list_with_histogram)
    if type(HEADER) is str:
        out.write(HEADER + "\n")
    bin_size = float(maximum - minimum) / float(N)
    for i in range(0, N):
        out.write(
            "%lf\t%lf\n" % ((minimum + float(i + 1) * bin_size), list_with_histogram[i])
        )
    del i
    if CUMULATIVE:
        out.write("\n\n0.\t0.\n")
        cum = 0.0
        for i in range(0, N):
            cum += list_with_histogram[i]
            out.write("%lf\t%lf\n" % ((minimum + float(i + 1) * bin_size), cum))
        del i, cum
    out.close()
    del N, out, bin_size
    return


def ScaleInRange(OldList, NewMin, NewMax, OldMin=None, OldMax=None):
    """
    linearly rescale a list in a new range scale_in_range
    """
    NewRange = 1.0 * (NewMax - NewMin)

    if OldMin is None:
        OldMin = numpy.nanmin(OldList, 0)
    if OldMax is None:
        OldMax = numpy.nanmax(OldList, 0)
    OldRange = 1.0 * (OldMax - OldMin)
    ScaleFactor = NewRange / OldRange
    #    print '\nEquation:  NewValue = ((OldValue - ' + str(OldMin) + ') x '+ str(ScaleFactor) + ') + ' + str(NewMin) + '\n'
    NewList = []
    for OldValue in OldList:
        NewValue = ((OldValue - OldMin) * ScaleFactor) + NewMin
        NewList.append(NewValue)
    return NewList


def analyze_vec(
    vec_list,
    start_pos=None,
    end_pos=None,
    low_limit_for_z_score=-0.7,
    up_limit_for_z_score=0.7,
    smooth_zscores=True,
    smooth_linear_coeff=0.5,
):
    """
    # this function analyzes the content of a list from position start_pos (included) to end_pos (excluded).
# It returns (average,standard_deviation,Max,Min,summ,average_positives,z_score,z_summ). z_score is z_summ
# divided by the number of elements that contributed to z_summ. The latter is calculated as a sum over
# the entries that are above an up_limit or below a low one. (basically to get read of noise for data close to zero).
# WHAT we employ for aggregation is z_summ_score= z_summ/(1.*len(mutant.seq))
# standard deviation is computed with Bessel's correction (e.g. divided by N-1)
    """
    if start_pos is None:
        start_pos = 0
    if end_pos is None:
        end_pos = len(vec_list)
    if start_pos >= end_pos:
        sys.stderr.write(
            "WARNING in analyze_vec() start_pos>=end_pos %d %d\n" % (start_pos, end_pos)
        )
        return 1
    Max = -99999.0
    Min = 99999.0
    sum_pos = 0.0
    z_score = 0.0
    z_den = 0.0
    summ = 0.0
    den2 = 0.0
    den = 1.0 * (end_pos - start_pos)
    for i in range(start_pos, end_pos):
        if vec_list[i] > Max:
            Max = vec_list[i]
        if vec_list[i] < Min:
            Min = vec_list[i]
        if vec_list[i] > 0.0:
            sum_pos += vec_list[i]
            den2 += 1.0
        summ += vec_list[i]
        if smooth_zscores:
            if low_limit_for_z_score > vec_list[i]:
                z_score += (
                    vec_list[i] * smooth_linear_coeff
                    + smooth_linear_coeff * low_limit_for_z_score
                )
                z_den += 1.0
            elif vec_list[i] > up_limit_for_z_score:
                z_score += (
                    vec_list[i] * smooth_linear_coeff
                    + smooth_linear_coeff * up_limit_for_z_score
                )
                z_den += 1.0
        else:
            if (
                low_limit_for_z_score > vec_list[i]
                or vec_list[i] > up_limit_for_z_score
            ):
                z_score += vec_list[i]
                z_den += 1.0
    z_summ = z_score
    if z_den > 1.0:
        z_score /= z_den
    if den2 > 1.0:
        average_positives = sum_pos / den2
    else:
        average_positives = sum_pos
    average = summ / den
    standard_deviation = 0.0
    for i in range(start_pos, end_pos):
        standard_deviation += (vec_list[i] - average) * (vec_list[i] - average)
    if den > 1.0:
        standard_deviation = numpy.sqrt(
            standard_deviation / (den - 1.0)
        )  # divided by N-1
    else:
        standard_deviation = 0.0
    return (
        average,
        standard_deviation,
        Max,
        Min,
        summ,
        average_positives,
        z_score,
        z_summ,
    )


def get_order_of_magnitued(x):
    """
    returns the order of magnitude of abs(x), i.e. the exponent you would give to x to write it in scientific notation
    """
    x = abs(x)
    if x > 1:
        mg = int(numpy.log10(x))
    else:
        mg = -1 * int(numpy.log10(1.0 / x)) - 1  # get order of magnitude
        mg = (
            -1 * int(numpy.log10(1.0 / (x + 10 ** (mg - 1)))) - 1
        )  # necessary as 0.01 ->1 while 0.011->2
    return mg


def pvalue_to_stars(p, non_significant="", symbol="*", special_for_negative="#"):
    if hasattr(p, "__len__"):  # handles up to one list
        return [
            pvalue_to_stars(pv, non_significant=non_significant, symbol=symbol)
            for pv in p
        ]
    if special_for_negative is not None and p < 0:
        return special_for_negative
    if p > 0.05:
        return non_significant
    elif p > 0.01:
        return symbol
    elif p > 0.001:
        return symbol * 2
    elif p > 0.0001:
        return symbol * 3
    else:
        return symbol * 4


def benedetta_stars(Val1, Val2, err1, err2, N_sigma_eq_th=[1.96, 2.58, 3.4]):
    """
    Welch's t test assuming infinite degrees of freedom
    see also table at
      http://en.wikipedia.org/wiki/Student%27s_t-distribution
    A Chiti Lab (benedetta Mannini) version of the T-test (two-tailed)
    Calcolare il quadrato dello standard error
    Calcolare sigma diff= ovvero la radice della somma dei quadrati degli SE delle misure che si vuole confrontare
    Fare la differenza Tra la media dei valori e la media del valore di riferimento (DIF)
    Se DIF +/- 2 o 3 o 4 sigma diff non passa da zero allora e' significativo
    
    DIF +/- 1.96 x sigma diff NON COMPRENDE 0 => la differenza tra A e B e' significativa p<0.05 (*)
    DIF +/- 2.58 x sigma diff NON COMPRENDE 0 => la differenza tra A e B e' significativa p<0.01 (**)
    DIF +/- 3.4  x sigma diff NON COMPRENDE 0 => la differenza tra A e B e' significativa p<0.001 (***)
    
    return stars,REF
    """
    npy = False
    N_sigma_eq_th.sort()
    if hasattr(Val1, "__len__"):
        Val1 = numpy.array(Val1)
        err1 = numpy.array(err1)
        npy = numpy.empty(Val1.shape).astype(str)
        npy[:] = ""
    if hasattr(Val2, "__len__"):
        Val2 = numpy.array(Val2)
        err2 = numpy.array(err2)
        npy = numpy.empty(Val2.shape).astype(str)
        npy[:] = ""
    REF = abs(Val1 - Val2) / numpy.sqrt(err1 ** 2 + err2 ** 2)
    if npy == False:
        stars = ""
        if REF > N_sigma_eq_th[2]:
            stars = "***"
        elif REF > N_sigma_eq_th[1]:
            stars = "**"
        elif REF > N_sigma_eq_th[0]:
            stars = "*"
        return stars, REF
    else:
        npy[numpy.where(REF > N_sigma_eq_th[0])] = "*"
        npy[numpy.where(REF > N_sigma_eq_th[1])] = "**"
        npy[numpy.where(REF > N_sigma_eq_th[2])] = "***"
        return npy, REF


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


def get_min_max_glob(entries):
    # up to three dimensions where various profiles within entries could have different shapes or lengths
    if hasattr(
        entries[0], "__len__"
    ):  # in principle various profiles within entries could have different shapes or lengths
        m = []
        M = []
        for en in entries:
            if hasattr(en[0], "__len__"):
                for v in en:
                    m += [numpy.nanmin(v)]
                    M += [numpy.nanmax(v)]
            else:
                m += [numpy.nanmin(en)]
                M += [numpy.nanmax(en)]
        m, M = numpy.nanmin(m), numpy.nanmax(M)
    else:
        m, M = numpy.nanmin(entries), numpy.nanmax(entries)
    return m, M


def RMSD(data, ref_data):
    """
    returns the root mean square deviation of the data calculated against the ref_data.
      these can be either atomic coordinates against a reference state
       or the predicted values and the observed values.
    """
    return numpy.sqrt(
        ((numpy.array(data) - numpy.array(ref_data)) ** 2).sum() / float(len(data))
    )


def MAD(data, axis=None):
    """
    returns median absolute deviation
    """
    return numpy.median(numpy.absolute(data - numpy.median(data, axis)), axis)


def factorial(n):
    """
    returns n!
    """
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def mean_absolute_deviation(data, axis=None):
    """
    returns mean absolute deviation
    """
    return numpy.mean(numpy.absolute(data - numpy.mean(data, axis)), axis)


def weighted_avg_and_std(values, weights, **kwargs):
    """
    Return the weighted average and standard deviation.
    in this case weights should be frequencies (or related to frequencies) of the values, rather than 1/sigma^2 as weighted averages
    of variables coming from different gaussian distributions.
    values, weights -- Numpy ndarrays with the same shape.
    **kwargs can be used for axis optional variable (e.g. axis=0) (axis=1 may not work in the std calculation and raise exception)
    """
    average = numpy.average(values, weights=weights, **kwargs)
    variance = numpy.average(
        (values - average) ** 2, weights=weights, **kwargs
    )  # Fast and numerically precise
    return (average, numpy.sqrt(variance))


def weigthed_average(array, array_of_weigths, return_only_av=False):
    """
    does the weighted average 
    returns w_average, w_stdev, var, SE
    Standard Error from Cochran (1977) (see http://www.cs.tufts.edu/~nr/cs257/archive/donald-gatz/weighted-standard-error.pdf from Donald F. Gatz and Luther Smith)
    """
    array = numpy.asarray(array, dtype=float)
    array_of_weigths = numpy.asarray(array_of_weigths, dtype=float)
    n = 1.0 * len(array_of_weigths)
    wbar = numpy.nanmean(array_of_weigths)
    su = numpy.nansum(array_of_weigths)
    w_average = numpy.nansum(array * array_of_weigths) / su
    if return_only_av:
        return w_average
    var = 1.0 / su
    w_stdev = numpy.sqrt(sum(array_of_weigths * (array - w_average) ** 2) / su)
    # SE_Miller77=w_stdev/numpy.sqrt(n)
    # print n,'SE_Miller77',SE_Miller77,
    SE2 = (n / ((n - 1) * su ** 2)) * (
        sum((array_of_weigths * array - wbar * w_average) ** 2)
        - 2
        * w_average
        * sum((array_of_weigths - wbar) * (array_of_weigths * array - wbar * w_average))
        + (w_average ** 2) * sum((array_of_weigths - wbar) ** 2)
    )
    return w_average, w_stdev, var, numpy.sqrt(SE2)


def weigthed_average_from_errors(
    vals, vals_errs, use_squared_errors=True, return_only_av=False
):
    """
    use_squared_errors should be true only when using standard errors on the means as input,
     in all other cases it should be false
    returns w_average, w_stdev, var, SE
    """
    if vals_errs is None:
        if return_only_av:
            return numpy.mean(vals)
        return (
            numpy.mean(vals),
            numpy.std(vals),
            numpy.var(vals),
            numpy.std(vals) / numpy.sqrt(len(vals)),
        )
    err = numpy.array(vals_errs)
    err[err <= 0.0] = (
        numpy.nanmin(err[err > 0]) / 10.0
    )  # weight 10 time larger than highest non-zero
    if use_squared_errors:
        w = 1.0 / err ** 2
    else:
        w = 1.0 / numpy.abs(err)
    return weigthed_average(vals, w, return_only_av=return_only_av)


def average(array):
    """
    does the average 
    returns average, stdev
    """
    array = numpy.asarray(array, dtype=float)
    den = 1.0 * len(array)
    average = sum(array) / den
    stdev = numpy.sqrt(sum((array - average) ** 2) / den)
    return average, stdev


def average_profiles_of_different_xs(
    list_of_xprofiles, list_of_y_profiles, compare_sensitivity=0.0001
):
    """
    WORK IN PROGRESS
    y values of different profiles corresponding to x values within compare_sensitivity will be averaged, others will be added with new values only from those profiles meeting the compare_sensitivity threshold 
    """
    counter = {}
    for j in range(len(list_of_xprofiles)):
        counter[j] = 0
    xvals = []
    yvals, ystd, ystderr = [], [], []

    return


def cartesian_product(*arrays):
    # the inptu is just various numpy arrays
    la = len(arrays)
    dtype = numpy.find_common_type([a.dtype for a in arrays], [])
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def cartesian_product_transpose(*arrays):
    # the inptu is just various numpy arrays
    broadcastable = numpy.ix_(*arrays)
    broadcasted = numpy.broadcast_arrays(*broadcastable)
    rows, cols = reduce(numpy.multiply, broadcasted[0].shape), len(broadcasted)
    dtype = numpy.find_common_type([a.dtype for a in arrays], [])
    out = numpy.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T


# return_list_of_list is a huge horror, i just lost too much time figuring out how to get numpy arrays with differnt dtypes to work
def cartesian(arrays, out=None, return_list_of_list=True):
    """
    DEPRECATED see cartesian_product and cartesian_product_transpose
    
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    return_list_of_list sets also the type to the original one, otherwise float64 is returned
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [numpy.asarray(x) for x in arrays]

    # dtype = arrays[0].dtype

    n = numpy.prod([x.size for x in arrays])  # get the overall length of the output

    if out is None:
        if return_list_of_list:
            dtypes = [a.dtype for a in arrays]
        # for a in arrays : dtypes+=str(a.dtype)+','
        out = numpy.zeros([n, len(arrays)])  # , dtype=dtype

    m = n // arrays[0].size
    out[:, 0] = numpy.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:], return_list_of_list=False)
        for j in range(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
    if return_list_of_list:
        ref_type = out.dtype
        out = list(out)
        for i, row in enumerate(out):
            out[i] = list(row)
            for j, column in enumerate(row):
                if dtypes[j] != ref_type:
                    out[i][j] = row[j].astype(dtypes[j])
        return out
    return out


def polfit(x, y, pol_order=1):
    coefficients = numpy.polyfit(x, y, pol_order)
    pol_fun = numpy.poly1d(coefficients)
    if pol_order == 1:
        R = get_pearsons_corr(x, y)
        R *= R
        return coefficients, pol_fun, R
    return coefficients, pol_fun


def get_pearsons_corr(odi, pdi):
    """
    GET THE PEARSON'S PRODUCT-MOMENT COEFFICIENT R (IF YOU COMBINE WITH LINEAR FIT THEN USE R^2)
    """
    assert len(odi) == len(pdi)
    N = len(odi)
    od = numpy.array(odi)  # if it is a list make it a numpy array
    pd = numpy.array(pdi)
    od = od - od.sum() / N  # every element is itself minus the average
    pd = pd - pd.sum() / N
    return numpy.dot(od, pd) / numpy.sqrt(numpy.dot(od, od) * numpy.dot(pd, pd))


def get_mse(odi, pdi):
    """
    get the mean square error.
    """
    assert len(odi) == len(pdi)
    N = len(odi)
    od = numpy.array(odi)
    pd = numpy.array(pdi)
    return numpy.dot(od - pd, od - pd) / N


def get_mae(odi, pdi):
    """
    get the mean absolute error
    """
    assert len(odi) == len(pdi)
    N = len(odi)
    od = numpy.array(odi, numpy.float64)
    pd = numpy.array(pdi, numpy.float64)
    return sum(abs(od - pd)) / N


# def error_logscale():
def propagate_error(function, args, errs, h=0.00000001, x_only=False):
    """
    given a function and the arguments (point at which to evaluate, could be function (x,y,z,... ) )
     together with the corresponding errors
     it returns the function evaluated and that arguments and the corresponding error(s)
    x_only can be given to use a function of only one variable (function(x)) but to support as input multiple values as a numpy array
     this can be useful for example when applyin a log to a plotting variable
    """
    if h is None:
        h = min([0.00001, min(numpy.abs(args)) * 0.00001])
    if x_only or not hasattr(args, "__len__"):  # just one variable, return without len
        return (
            function(args),
            errs * numpy.abs(((function(args + h) - function(args - h)) / (2.0 * h))),
        )
    # DaG=((*g)((a+h),b,c,d)-(*g)((a-h),b,c,d))/(2*h); sf=sqrt(DaG*DaG*sa*sa + DbG*DbG*sb*sb...
    err = numpy.sqrt(
        numpy.array(
            [
                (
                    (
                        (
                            function(*(args[:i] + [args[i] + h] + args[i + 1 :]))
                            - function(*(args[:i] + [args[i] - h] + args[i + 1 :]))
                        )
                        / (2.0 * h)
                    )
                    ** 2
                )
                * (errs[i] ** 2)
                for i in range(len(args))
            ]
        ).sum(axis=0)
    )
    return function(*args), err


# function to normalize a list or tuple, or a list of lists or of tuples.
#   In the latter case only the variable found at position in the inner list (or tuple) is normalized
def normalize(iterable, position=0):
    den = 0.0
    for some in iterable:
        if type(some) is tuple or type(some) is list:
            den += some[position]
        else:
            den += some
    for i, some in enumerate(iterable):
        if type(some) is tuple:
            iterable[i] = (
                some[:position] + (some[position] / den,) + some[position + 1 :]
            )
        elif type(some) is list:
            iterable[i][position] = some[position] / den
        else:
            iterable[i] = some / den
    return iterable


def remove_duplicates(string_or_list, only_for_ascii_char=False):
    # remove the duplicates from a string or a list
    if only_for_ascii_char:
        import string
    if type(string_or_list) is list:
        n = []
    elif type(string_or_list) is str:
        n = ""
    for i in string_or_list:
        if only_for_ascii_char and i not in string.ascii_letters:
            n += i
        elif i not in n:
            n += i
    return n


# flatten a list of lists also when sublists have different lengths: [item for sublist in l for item in sublist]
flatten = lambda l: [item for sublist in l for item in sublist]
# to flatten the transpose first i,j=0,0 #(important!) and then: derU=[dy[i][j] for j in range(0,len(dy[i])) for i in range(0,len(dy)) if j<len(dy[i]) ]


def smooth_profile(
    profile,
    smooth_per_side=3,
    weights=None,
    use_savgol_filter=False,
    interpolate_nan=False,
    mode="interp",
):
    """
    # this function smooth a profile by replacing each entry at position i with the average of the entries 
    #  in i-smooth_per_side, i+smooth_per_side.
    if use_savgol_filter is True then
        window_length = (2*smooth_per_side)+(1-(2*smooth_per_side)%2) (makes odd)
        and polyorder=min([5,win-1]) mode='interp'
    otherwise use_savgol_filter can be a tuple and then
        window_length,polyorder=use_savgol_filter
    interpolate_nan works only for single nan values otherwise nan will be also in smoothed profile
      nan at the boundaries will always be nan
    """
    profile = numpy.array([numpy.nan if p == "" else p for p in profile])
    smoothed = []
    nans = numpy.where(numpy.isnan(profile))[0]
    if len(nans) > 0:
        profile = (
            profile.copy()
        )  # this should avoid profile being modified outside function
        profile = profile[~numpy.isnan(profile)]  # remove nan
    if use_savgol_filter or type(use_savgol_filter) is tuple:
        if weights is not None:
            sys.stderr.write(
                "**ERROR** in smooth_profile() ignoring weights as cannot use them with use_savgol_filter\n"
            )
        if type(use_savgol_filter) is tuple:
            win, pol = use_savgol_filter
        else:
            win = (2 * smooth_per_side) + (1 - (2 * smooth_per_side) % 2)
            pol = min([5, win - 1])
        smoothed = scipy.signal.savgol_filter(
            numpy.array(profile),
            window_length=win,
            polyorder=pol,
            deriv=0,
            delta=1.0,
            axis=-1,
            mode=mode,
            cval=0.0,
        )
    # if weights is None : # we do it with numpy! WE DON"T BUG FOR EXTREMA
    #    print 'here'
    #   box_pts = 2*smooth_per_side + 1
    #    box = numpy.ones(box_pts)/box_pts
    #    y_smooth = numpy.convolve(profile , box, mode='same')
    #    return y_smooth
    elif weights is not None:
        if len(weights) != len(profile):
            sys.stderr.write(
                "**WARNING** in smooth_profile() len(weights)!=len(profile) %d %d\n"
                % (len(weights), len(profile))
            )
        weights = numpy.array(weights)
        for i in range(0, len(profile)):
            if i - smooth_per_side < 0:
                s = 0
            else:
                s = i - smooth_per_side
            den = sum(weights[s : i + smooth_per_side + 1])
            if den == 0:
                smoothed += [
                    0
                ]  # SHOULD raise warning but within CamSol this is used for solvent-exposure so for non-exposed regions 0 is the expected results
            else:
                smoothed += [
                    float(
                        sum(
                            profile[s : i + smooth_per_side + 1]
                            * weights[s : i + smooth_per_side + 1]
                        )
                    )
                    / den
                ]
    else:
        for i in range(0, len(profile)):
            if i - smooth_per_side < 0:
                s = 0
            else:
                s = i - smooth_per_side
            # print i, s, profile[s:i+smooth_per_side+1],len(profile[s:i+smooth_per_side+1]),float(sum(profile[s:i+smooth_per_side+1]))/len(profile[s:i+smooth_per_side+1])
            smoothed += [
                float(sum(profile[s : i + smooth_per_side + 1]))
                / len(profile[s : i + smooth_per_side + 1])
            ]
    if len(nans) > 0:
        smoothed = list(smoothed)
        for j in nans:
            if not interpolate_nan:
                smoothed[j:j] = [numpy.nan]
            elif j > 0 and j < len(smoothed) - 1:
                smoothed[j:j] = [(smoothed[j - 1] + smoothed[j + 1]) / 2.0]
            else:
                smoothed[j:j] = [numpy.nan]
    return numpy.array(smoothed)


def get_mean_point_of_profile(profile, x_axis=None):
    """
    # it extracts the mean point of a profile (assuming the profile contains a probabiliy distribution)
    # if x_axis is None than the midpoints are the indices 0,1,2,... of the profile
    """
    sum_of_products = 0.0
    den = 0.0
    for k, p in enumerate(profile):
        if x_axis is None:
            sum_of_products += 1.0 * k * p
        else:
            sum_of_products += 1.0 * x_axis[k] * p
        den += 1.0 * p
    return sum_of_products / den


def derivative(y, x):
    """
    returns d1, xvalues
    """
    return (
        numpy.diff(y, n=1) / numpy.diff(x, n=1),
        (numpy.array(x[1:]) + numpy.array(x[:-1])) / 2.0,
    )


def sign_change(array):
    # returns the numpy.where the array changes sign (for 1D take [0])
    return numpy.where(1 == ((numpy.diff(numpy.sign(array)) != 0) * 1))


def get_directional_separation_efficiency(
    ensemble_smaller_array, ensemble_larger_array
):
    """
    it finds the line that separates most points correctly (in terms of percent of each distribution,
     not of total number of points as the sizes of the two distribution may differ).
    and return the percentage of points that it is possible to separate 
    with this line, call the efficiency.
    return efficiency, line_value_in_smaller_ensemble, line_value_in_larger_ensemble
     the line can be given as 
     l=(line_value_in_smaller_ensemble+line_value_in_larger_ensemble)/2.
    an efficiency of 50% would correspond to distributions with identical median
    """
    ensemble_smaller_array = numpy.sort(ensemble_smaller_array)
    ensemble_larger_array = numpy.sort(ensemble_larger_array)
    # find the line that separates most points correctly and then count..
    # if rather than by index you move by percentile then the percentile where the two distributions are equal
    # is the line that separates most values!

    if ensemble_larger_array[0] > ensemble_smaller_array[-1]:
        return 100.0  # best separation possible
    elif ensemble_larger_array[-1] < ensemble_smaller_array[0]:
        return 0.0  # worst separation possible
    NS = len(ensemble_smaller_array)
    NL = len(ensemble_larger_array)
    if NL < NS:
        con = lambda jl, js, oldjl, oldjs: oldjl == jl
    else:
        con = lambda jl, js, oldjl, oldjs: oldjs == js

    f = 0.5
    den = 2.0
    jl = int(NL * (1.0 - f) + 0.5)
    js = int(NS * f + 0.5)
    oldjl, oldjs = -10, -10
    while not con(jl, js, oldjl, oldjs):
        d = ensemble_larger_array[jl] - ensemble_smaller_array[js]
        den *= 2
        f += numpy.sign(d) * 1 / den
        oldjl = jl
        oldjs = js
        jl = int(NL * (1.0 - f) + 0.5)
        js = int(NS * f + 0.5)
        # print f,oldjl,jl, oldjs,js,d

    return 100.0 * f, ensemble_smaller_array[oldjs], ensemble_larger_array[oldjl]


def efficiency_separate_two_distribution(
    ensemble_smaller_vals, ensemble_larger_vals, return_single_contributions=False
):
    """Returns efficiency on the separation of two classes
    Efficiency range is [-1,1] where:
         -1 is horrible efficiency, distributions are swapped 
         0 is bad efficiency, distributions are mostly overlapped
         1 is perfect, distributions are well separated and in the correct order (smaller before larger)
    return efficiency, median_smaller, median_larger
    and more if return_single_contributions is True
        return efficiency, median_smaller, median_larger, distance_contribution, fraction_bad_contribution, lower_quartile_smaller, upper_quartile_smaller, lower_quartile_larger, upper_quartile_larger
    """
    if ensemble_smaller_vals is None or ensemble_larger_vals is None:
        raise IOError("Missing ensembles")
    if type(ensemble_smaller_vals) == dict or isinstance(
        ensemble_smaller_vals, OrderedDict
    ):
        smaller_array = numpy.array(list(ensemble_smaller_vals.values()))
        larger_array = numpy.array(list(ensemble_larger_vals.values()))
    else:
        smaller_array = numpy.array(ensemble_smaller_vals)
        larger_array = numpy.array(ensemble_larger_vals)

    median_smaller = numpy.median(smaller_array)
    lower_quartile_smaller, upper_quartile_smaller = numpy.percentile(
        smaller_array, [25, 75]
    )
    iqr_smaller = upper_quartile_smaller - lower_quartile_smaller

    median_larger = numpy.median(larger_array)
    lower_quartile_larger, upper_quartile_larger = numpy.percentile(
        larger_array, [25, 75]
    )
    iqr_larger = upper_quartile_larger - lower_quartile_larger

    # distributions_distance = 2.*(median_larger-median_smaller)/(upper_quartile_larger-lower_quartile_larger+upper_quartile_smaller-lower_quartile_smaller)
    # print distributions_distance

    fraction_bad_smaller = ((smaller_array >= lower_quartile_larger).sum()) / float(
        len(smaller_array)
    )
    fraction_bad_larger = ((larger_array <= upper_quartile_smaller).sum()) / float(
        len(larger_array)
    )

    if iqr_larger >= iqr_smaller:
        efficiency = 1.0 / (
            1.0 + numpy.exp(-2.0 * (median_larger - median_smaller) / iqr_larger)
        ) - 0.5 * (fraction_bad_smaller + fraction_bad_larger)
        if return_single_contributions:
            distance_contribution = (
                1.0
                / (1.0 + numpy.exp(-2 * (median_larger - median_smaller) / iqr_larger))
                - 0.5
            )
            fraction_bad_contribution = 0.5 - 0.5 * (
                fraction_bad_smaller + fraction_bad_larger
            )
            return (
                efficiency,
                median_smaller,
                median_larger,
                distance_contribution,
                fraction_bad_contribution,
                lower_quartile_smaller,
                upper_quartile_smaller,
                lower_quartile_larger,
                upper_quartile_larger,
            )
        else:
            return efficiency, median_smaller, median_larger
    else:
        efficiency = 1.0 / (
            1.0 + numpy.exp(-2.0 * (median_larger - median_smaller) / iqr_smaller)
        ) - 0.5 * (fraction_bad_smaller + fraction_bad_larger)
        if return_single_contributions:
            distance_contribution = (
                1.0
                / (1.0 + numpy.exp(-2 * (median_larger - median_smaller) / iqr_smaller))
                - 0.5
            )
            fraction_bad_contribution = 0.5 - 0.5 * (
                fraction_bad_smaller + fraction_bad_larger
            )
            return (
                efficiency,
                median_smaller,
                median_larger,
                distance_contribution,
                fraction_bad_contribution,
                lower_quartile_smaller,
                upper_quartile_smaller,
                lower_quartile_larger,
                upper_quartile_larger,
            )
        else:
            return efficiency, median_smaller, median_larger


def convert_to_ndarray(elements):
    arrays = []
    for el in elements:
        if type(el) is not list:
            arrays += [[el]]
        else:
            arrays += [el]
    arrays = numpy.array(arrays)
    return 1.0 * arrays  # so that everything becomes float


class Remember:
    """
    a class that remembers the call of a function and returns the same values
       setting use_kwargs to False is very dangerous in the sense that it assumes that if they are changed they
       have no impact on the return value of the function. It is kind of useful for recursive functions that might have 
       a Remember class as kwarg...
    """

    def __init__(self, function, use_kwargs=True):
        self.fun = function
        self.use_kwargs = use_kwargs
        self.dict_tup_args_kwarg = {}

    def __call__(self, function, *args, **kwargs):
        if self.fun != function:
            raise Exception(
                "Remember class used with function different from the one it was initialized to"
            )
        if kwargs == {} or not self.use_kwargs:
            if args not in self.dict_tup_args_kwarg:
                # print 'add'
                self.dict_tup_args_kwarg[args] = function(*args, **kwargs)
            return self.dict_tup_args_kwarg[args]
        else:
            ks = tuple(sorted(kwargs.items()))
            if (args, ks) not in self.dict_tup_args_kwarg:
                # print 'add'
                self.dict_tup_args_kwarg[(args, ks)] = function(*args, **kwargs)
            return self.dict_tup_args_kwarg[(args, ks)]


def cluster_with_norm(
    elements_to_cluster,
    norm_normalize_threshold=None,
    tolerance_sigma=None,
    strict_clustering=True,
    log_file=sys.stdout,
):
    """
    # it clusterizes a list of numbers/ list of lists of numbers.
    # if norm_normalize_threshold is None it will be automatically computed as the average of the norm of the difference vector (or absolute value of the differences)
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
    arrays = convert_to_ndarray(elements_to_cluster)
    # note that this is NOT A NORMALIZATION.
    # this is done so that every axis has the same importance (as every axis in our space generally represent a different experimental condition (e.g. pH or Ionic Strength, which could span a very different set of real numbers).
    # with this method we give importance only to how much they actually vary and not to their absolute values
    column_sums = arrays.sum(axis=0)
    arrays /= column_sums

    # bulid a square matrix with vector differences norms. so to cluster according to similar conditions

    # similarity_matrix=[[0. for x in range(0,len(arrays))] for z in range(0,len(arrays))]
    similarity_matrix = numpy.zeros((len(arrays), len(arrays)))
    max_norm = -10
    summ_norm = 0.0
    tot = 0
    for i in range(0, len(arrays)):
        for j in range(i + 1, len(arrays)):
            similarity_matrix[i][j] = numpy.linalg.norm(
                arrays[i] - arrays[j]
            )  # fill with the norm of the difference of the vectors
            summ_norm += similarity_matrix[i][j]
            tot += 1
            if similarity_matrix[i][j] > max_norm:
                max_norm = similarity_matrix[i][j]
    av_diff = summ_norm / (1.0 * tot)
    if log_file is not None and tolerance_sigma is None:
        log_file.write(
            "cluster_with_norm(): after column normalization; av_diff=%lf, max_diff=%lf\n"
            % (av_diff, max_norm)
        )
        log_file.flush()
    if tolerance_sigma is not None and norm_normalize_threshold is None:
        stdev = 0.0
        for i in range(0, len(arrays)):
            for j in range(i + 1, len(arrays)):
                stdev += (similarity_matrix[i][j] - av_diff) ** 2
        stdev = numpy.sqrt(stdev / (tot - 1.0))  # #divided by N-1
        if log_file is not None:
            log_file.write(
                "cluster_with_norm(): after column normalization; av_diff=%lf, max_diff=%lf  stdev=%lf\n"
                % (av_diff, max_norm, stdev)
            )
        av_diff += tolerance_sigma * stdev
        if av_diff < 0:
            sys.stderr.write(
                "WARNING in auto_clustering(), tolerance_sigma=%lf too negative will make everything belong to different clusters\n"
                % (tolerance_sigma)
            )
    if norm_normalize_threshold is None:
        norm_normalize_threshold = av_diff

    clusters = clusters_from_matrix(
        similarity_matrix, lower_threshold=0.0, upper_threshold=norm_normalize_threshold
    )
    clustered_values = [
        [elements_to_cluster[i] for i in cluster] for cluster in clusters
    ]

    return clusters, clustered_values


"""
use the import uncertainties package..
def propagate_error(fun, x, *args,**kwargs):
    f=fun(x,*args,**kwargs)
    h=0.000001
    DaG=(fun(x+h,*args,**kwargs)-fun(x-h,*args,**kwargs))/(2*h);
    DbG=((*g)(a,(b+h))-(*g)(a,(b-h)))/(2*h);
    sf=sqrt(DaG*DaG*sa*sa+DbG*DbG*sb*sb);
    return
"""


def strip_file(filename, only_from_right=True):
    with open(filename) as fff:
        fi = fff.read().splitlines()
    tmp = ""
    for line in fi:
        if only_from_right:
            tmp += line.rstrip() + "\n"
        else:
            tmp += line.strip() + "\n"
    out = open(filename, "w")
    out.write(tmp)
    out.close()
    del tmp


def ilog(n, base):
    """
    Find the integer log of n with respect to the base.

    >>> import math
    >>> for base in range(2, 16 + 1):
    ...     for n in range(1, 1000):
    ...         assert ilog(n, base) == int(math.log(n, base) + 1e-10), '%s %s' % (n, base)
    """
    count = 0
    while n >= base:
        count += 1
        n //= base
    return count


def sci_notation(n, prec=3):
    """
    Represent n in scientific notation, with the specified precision.

    >>> sci_notation(1234 * 10**1000)
    '1.234e+1003'
    >>> sci_notation(10**1000 // 2, prec=1)
    '5.0e+999'
    """
    base = 10
    exponent = ilog(n, base)
    mantissa = n / base ** exponent
    return "{0:.{1}f}e{2:+d}".format(mantissa, prec, exponent)


def optimise_spline(
    x,
    y,
    expected_nof_flexes,
    expected_nof_absolute_max_or_min,
    yerr=None,
    p_guess=[2, 3],
    roundigit=2,
    plot=False,
):
    """
see savitzky_golay_interpolation which is better
    function interpolate may be better than this one.
    x MUST BE INCREASING
    returns a spline interpolating y (from x)
    p_guess are initial guesses for s and k of the spline, but are automatically optimised to match the expected_nof_flexes and expected_nof_absolute_max_or_min
    k is the degree and s specifies the number of knots by specifying a smoothing condition.
      The number of data points must be larger than the spline degree `k`
    yerr will convert to w, which are  weights of the data point (1/err will be applied)
    """
    x = numpy.array(x)
    y = numpy.array(y)
    w = None
    if yerr is not None and all(yerr > 1e-7):
        w = 1.0 / numpy.array(yerr)
    k = p_guess[1]
    # s and k of the spline
    def cost_function(p, xdata, ydata, k):
        p[1] = int(p[1])
        if p[1] < 0:
            return 1e9
        if p[0] < 0 or p[0] > 9:
            return 1e9
        spl = UnivariateSpline(xdata, ydata, k=p[1], s=p[0], w=w)
        if yerr is not None and all(yerr > 1e-7):
            cost_points = ((spl(xdata) - ydata) / yerr) ** 2
        elif not any((-1 < ydata) & (ydata < 1)):
            cost_points = ((spl(xdata) - ydata) / ydata) ** 2
        else:
            cost_points = spl(xdata) - ydata
            y2 = ydata.copy()
            y2[
                ((-1 < y2) & (y2 < 1))
            ] = 1  # ugly way gives equal weight to small points, so probably more weigth to them
            cost_points /= y2
            cost_points = cost_points ** 2
        xs = numpy.linspace(min(xdata), max(xdata), 5000)
        d1, xd1 = derivative(spl(xs), xs)
        zero_crossings1 = numpy.where(numpy.diff(numpy.sign(d1)))[0]
        d2, _ = derivative(d1, xd1)
        zero_crossings2 = numpy.where(numpy.diff(numpy.sign(d2)))[0]
        cost_derivatives = (
            (1.1 * (len(zero_crossings1) - expected_nof_absolute_max_or_min)) ** 2
            + ((len(zero_crossings2) - expected_nof_flexes)) ** 2
        ) * (numpy.sqrt(len(x)))
        return cost_points + cost_derivatives

    pfit = []
    S = 9e19
    for k in range(1, 6):
        pf, pcov, infodict, errmsg, success = scipy.optimize.leastsq(
            cost_function, p_guess, args=(x.copy(), y.copy(), k), full_output=1
        )
        c = sum(cost_function(pf, x.copy(), y.copy(), k))
        # print 'optimise_spline k,c,success:',k,c,success,'S',S
        if c < S:
            pfit = [pf[0], k]
            S = c
    print("optimise_spline s,k", pfit)
    SPL = UnivariateSpline(x, y, k=pfit[1], s=pfit[0], w=w)
    if plot:
        f = plotter.profile(y, x, marker=".", ls="")
        xs = numpy.linspace(min(x), max(x), 5000)
        f = plotter.profile(SPL(xs), xs, marker="", ls="-", figure=f)
    return SPL, pfit


def interpolate(
    x,
    ydata,
    k=3,
    s=None,
    yerr=None,
    plot=False,
    npoints=5000,
    pre_smooth=(21, 3),
    plot_label=None,
):
    """
see class Savitzky_golay_interpolation (first choice)
   or even see savitzky_golay_interpolation (second choice) which are better
    may add a smoothing before the interpolation
    s is a Positive smoothing factor used to choose the number of knots.  Number
       of knots will be increased until the smoothing condition is satisfied
    a good s is len(y), an s that provides more smoothing is 3*len(y)
    x must be strictly increasing
    return SPL,xs,ys
   k= Degree of the smoothing spline.  Must be <= 5.
     Default is k=3, a cubic spline.
    """
    if len(x) != len(ydata):
        sys.stderr.write(
            "**ERROR** in interpolate x and y have different lengths %d %d\n"
            % (len(x), len(ydata))
        )
    if len(x) < 3:
        return None, x, ydata
    x, arsort = numpy.unique(
        x, return_index=True
    )  # Returns the sorted unique elements of an array. the indices of the input array that give the unique values
    if len(x) != len(ydata):
        sys.stderr.write(
            "**Warning** in interpolate x and y have different lengths %d %d after sorting uniques xs (must be  strictly increasing), leaving out some points from interpolation\n"
            % (len(x), len(ydata))
        )
    y = numpy.array(ydata)[arsort]
    if pre_smooth is not None and pre_smooth != False:
        y = smooth_profile(
            y, smooth_per_side=21, use_savgol_filter=pre_smooth, interpolate_nan=False
        )
        # if s is None : s=len(y)/3.
    # else :
    #    if s is None : s=2.5*len(y)
    if yerr is not None and all(numpy.array(yerr) > 1e-7):
        w = 1.0 / numpy.array(yerr)
        SPL = UnivariateSpline(x, y, k=k, s=s, w=w)
    else:
        # w= numpy.ones(len(ydata))*numpy.mean(numpy.abs(y))*100 # like having a constant error of 1% - for some reasons setting all weight equal does not give same result for every value (e.g. setting them all to 1 requires different combo of s,k to get same interpolation)
        SPL = UnivariateSpline(x, y, k=k, s=s)
    if npoints is None:
        npoints = 5000
    xs = numpy.linspace(min(x), max(x), npoints)
    ys = SPL(xs)
    if plot:
        f = plotter.profile(ydata, x, marker=".", ls="", yerr=yerr, markersize=8)
        if pre_smooth is not None and pre_smooth != False:
            f = plotter.profile(
                y, x, marker="s", ls="", figure=f, markersize=4, markerfacecolor="red"
            )
        f = plotter.profile(
            ys,
            xs,
            marker="",
            ls="-",
            figure=f,
            title="s=%s k=%g weighted=%s"
            % (str(s), k, (yerr is not None and all(numpy.array(yerr) > 1e-7))),
            label=plot_label,
        )
    return SPL, xs, ys


def find_peak_indices(ys, max_ind=None, min_ys_value_percent_change=7):
    """
    return peak_indices
    for a minimum just give -1*ys as input
    gets the max of ys and find the peak around it - ys can be e.g. a probability density
    max_ind can be given as the index of the max to identify e.g. peaks around local max
     as opposed to global max
    peak ends at change in sign of first derivative or at 2 consecutive points with rather flat derivative
     AND the value of ys must have changed by at least min_ys_value_percent_change  from the max
    """
    if max_ind is None:
        max_ind = numpy.argmax(ys)
    y1 = numpy.gradient(ys)
    y1 /= max(abs(y1))
    M = ys[max_ind]
    oldy1 = 0
    peak_inds = [max_ind]
    j = max_ind + 1
    while j < len(ys):
        if 100.0 * abs((ys[j] - M) / float(M)) > min_ys_value_percent_change and (
            y1[j] * oldy1 < 0 or (abs(y1[j]) < 0.02 and abs(oldy1) < 0.02)
        ):
            if abs(y1[j]) < 0.02 and abs(oldy1) < 0.02:
                peak_inds.pop()  # in this case the peak actually ended before
            break
        peak_inds += [j]
        oldy1 = y1[j]
        j += 1
    j = max_ind - 1
    oldy1 = 0
    while j >= 0:
        if 100.0 * abs((ys[j] - M) / float(M)) > min_ys_value_percent_change and (
            y1[j] * oldy1 < 0 or (abs(y1[j]) < 0.02 and abs(oldy1) < 0.02)
        ):
            if abs(y1[j]) < 0.02 and abs(oldy1) < 0.02:
                peak_inds[0:1] = []  # in this case the peak actually ended before
            break
        peak_inds[0:0] = [j]  # append at beginning so the indices are sorted
        oldy1 = y1[j]
        j -= 1
    return numpy.array(peak_inds)


def process_profile_steep_regions(
    y,
    x=None,
    sort_steep_points=True,
    interpolate=2,
    smooth=False,
    smooth_derivatives=True,
    derivative_magnitude_abs_filter=0.3,
    shoulder_maginitude_threshold=0.05,
    min_frac_xseparation=0.01,
    window_size=5,
    diff_window=5,
    order=2,
    plot=False,
    fig=None,
    interpolation_rate=None,
):
    """
    return steep_increase,steep_decrease, magnitudes,shoulder_begins,shoulder_ends,y,x
    interpolate=2 is not an interpolation but a smoothing. (>2 it is an iterpolation - 2 means that 2 consecutive points in the profile are replaced by 2 points which are thus just smoothed).
    y and x may be as input unless smooth (y different) or interpolate (both different)
    window_size, order,interpolation_rate are parameters of smoothing (savitzky_golay, first two) or inerpolate
    steep_increase contains the index of the points of maximum steepness where the profile is increasing,
      steep_decrease the same but when decreasing and magnitudes is a dictonary where keys are these indices
      and values the magnitude of the steepnes, expressed as the absolute value of the first derivative/max(abs(derivative))
    shoulder_begins and shoulder_ends contains the index in y,x where 
      there is an estimate of the beginning/end of the slope change (non inflection points as typically the second derivative does not change sign)
    CAn visualise as:

steep_increase,steep_decrease, magnitudes,shoulder_begins,shoulder_ends,yi,xi =misc.process_profile_steep_regions(y,x,plot=True, interpolate=2,window_size=5)

plotter.profile([y,yi],[x,xi],ls=['','-'],marker=['.',''],markersize=5,vline=[xi[jj] for jj in steep_increase+steep_decrease+shoulder_begins+shoulder_ends])

plotter.profile(numpy.gradient(yi,xi[1]-xi[0]),xi,vline=[xi[jj] for jj in steep_increase+steep_decrease+shoulder_begins+shoulder_ends])
    """
    if x is None:
        x = numpy.arange(0, len(y))
    elif len(x) != len(y):
        sys.stderr.write(
            "\n**ERROR** in process_profile_steep_regions en(x)!=len(y) %d and %d\n"
            % (len(x), len(y))
        )
    y = numpy.array(y)
    if interpolate:
        if interpolation_rate is None:
            if type(interpolate) is int:
                interpolation_rate = interpolate
            else:
                interpolation_rate = 5  # add 3 new points between any 2
        if plot:
            fig = plotter.profile(
                (y - min(y)) / (max(y) - min(y)),
                x,
                ls="",
                marker=".",
                markersize=5,
                figure=fig,
            )
        _, xs, ys = savitzky_golay_interpolation(
            y,
            x,
            window_size=window_size,
            order=order,
            interpolation_rate=interpolation_rate,
        )
        return process_profile_steep_regions(
            ys,
            xs,
            interpolate=False,
            smooth=False,
            plot=plot,
            smooth_derivatives=smooth_derivatives,
            fig=fig,
            derivative_magnitude_abs_filter=derivative_magnitude_abs_filter,
        )
    elif smooth:
        if plot:
            fig = plotter.profile(
                (y - min(y)) / (max(y) - min(y)),
                x,
                ls="",
                marker=".",
                markersize=5,
                figure=fig,
            )
        ys = smooth_profile(y, use_savgol_filter=(window_size, order))
        return process_profile_steep_regions(
            ys,
            x=x,
            interpolate=False,
            smooth=False,
            smooth_derivatives=smooth_derivatives,
            plot=plot,
            fig=fig,
            derivative_magnitude_abs_filter=derivative_magnitude_abs_filter,
        )

    if smooth_derivatives:
        # y1= numpy.gradient(y,dx)
        # _,_,y1 =savitzky_golay_interpolation(y1,x, window_size=window_size, order=order,interpolation_rate=2 ) #Doesn't really interpolate but makes smoother(y1,x  use_savgol_filter=(window_size,order) )
        y1 = derivative_of_noisy_data(y, x, window=diff_window)
        y2 = derivative_of_noisy_data(y1, x, window=diff_window, use_savgol=False)
        y3 = derivative_of_noisy_data(y2, x, window=diff_window, use_savgol=False)
    else:
        dx = x[1] - x[0]
        y1 = numpy.gradient(y, dx)
        y2 = numpy.gradient(y1, dx)
        y3 = numpy.gradient(y2, dx)
    y2 /= float(abs(max(y2)) + abs(min(y2)))  # normalise so that overall range in 1
    # if smooth_derivatives : _,_,y2 =savitzky_golay_interpolation(y2,x, window_size=window_size, order=order,interpolation_rate=2 ) #y2 = smooth_profile(y2,  use_savgol_filter=(window_size,order) )
    y3 /= float(abs(max(y3)) + abs(min(y3)))  # normalise
    # if smooth_derivatives : _,_,y3 =savitzky_golay_interpolation(y3,x, window_size=window_size, order=order,interpolation_rate=2 ) # smooth_profile(y3,  use_savgol_filter=(window_size,order) )
    steep_decrease = []
    steep_increase = []
    oldv = y2[0]
    magnitudes = {}
    den = float(abs(max(y1)) + abs(min(y1)))
    for j, v in enumerate(y2[1:]):
        i = j + 1
        if v == 0 or v * oldv < 0:  # change in sign of second derivative
            if abs(y1[i]) < abs(y1[j]):
                i = j  # second derivative may change sign before or after, we want to save the max
            # print i,j,v,oldv,'v-oldv',v-oldv,y1[i],abs(y1[i]/den),'derivative_magnitude_abs_filter',derivative_magnitude_abs_filter
            if (
                derivative_magnitude_abs_filter is not None
                and derivative_magnitude_abs_filter > abs(y1[i] / den)
            ):
                oldv = v
                continue
            if y1[i] > 0 and v - oldv < 0:
                steep_increase += [i]
                magnitudes[i] = y1[i] / den  # abs(v-oldv)*len(profile)
            elif y1[i] < 0 and v - oldv >= 0:
                steep_decrease += [i]
                magnitudes[i] = y1[i] / den
        oldv = v
    # in case derivative_magnitude_abs_filter was too strict we still save the absoute max or min when these are not at the termini
    m = numpy.argmin(y1)
    if (
        m not in steep_decrease
        and m not in [0, len(y) - 1]
        and (y2[m] * y2[m - 1] < 0 or y2[m] * y2[m + 1] < 0)
    ):
        steep_decrease += [m]
        magnitudes[m] = y1[m] / den
    M = numpy.argmax(y1)
    if (
        M not in steep_increase
        and M not in [0, len(y) - 1]
        and (y2[M] * y2[M - 1] < 0 or y2[M] * y2[M + 1] < 0)
    ):
        steep_increase += [M]
        magnitudes[M] = y1[M] / den
    shoulder_begins, shoulder_ends = [], []
    oldv = None
    can_add_start, can_add_end = True, False
    # find closest suitable start
    for inc in steep_increase + steep_decrease:
        j = inc
        oldv = None
        while j >= 0:
            v = abs(y3[j])
            if oldv is not None and (
                min_frac_xseparation is None
                or float(inc - j) / len(y) > min_frac_xseparation
            ):  # min_frac_xseparation as y2 changes sign at the steepest and y3 can be noisy this is introduced to stay far from that point with the shoulders
                if (
                    (
                        (
                            (
                                v < shoulder_maginitude_threshold
                                and oldv > shoulder_maginitude_threshold
                            )
                            or (
                                v > shoulder_maginitude_threshold
                                and oldv < shoulder_maginitude_threshold
                            )
                        )
                        and (
                            abs(y2[j]) < 3.0 * shoulder_maginitude_threshold
                            or abs(y1[j]) < 3.0 * shoulder_maginitude_threshold
                        )
                    )
                    or y1[j] * y1[j + 1] < 0
                ):  # and y1[j]>=0 :
                    shoulder_begins += [j]
                    break
            oldv = v
            j -= 1
        j = inc
        oldv = None
        while j < len(y3):
            v = abs(y3[j])
            if oldv is not None and (
                min_frac_xseparation is None
                or float(j - inc) / len(y) > min_frac_xseparation
            ):  # min_frac_xseparation as y2 changes sign at the steepest and y3 can be noisy this is introduced to stay far from that point with the shoulders
                if (
                    (
                        (
                            (
                                v < shoulder_maginitude_threshold
                                and oldv > shoulder_maginitude_threshold
                            )
                            or (
                                v > shoulder_maginitude_threshold
                                and oldv < shoulder_maginitude_threshold
                            )
                        )
                        and (
                            abs(y2[j]) < 3.0 * shoulder_maginitude_threshold
                            or abs(y1[j]) < 3.0 * shoulder_maginitude_threshold
                        )
                    )
                    or y1[j] * y1[j - 1] < 0
                ):  # and y1[j]>=0 :
                    shoulder_ends += [j]
                    break
            oldv = v
            j += 1

    # Could probably just do the below on y1 instead of y3! Y3 is in principle good but for some profile it can be very noisy
    # for j,v in enumerate(abs(y3)) :
    #    if oldv is not None :
    #        if v>shoulder_maginitude_threshold and oldv< shoulder_maginitude_threshold and abs(y2[j])< 3.*shoulder_maginitude_threshold : #and y1[j]>=0 :
    #            if can_add_start : shoulder_begins+=[j]
    #            can_add_start=False
    #            #can_add_end=True
    #        elif v<shoulder_maginitude_threshold and oldv> shoulder_maginitude_threshold and abs(y2[j])<3.*shoulder_maginitude_threshold  : #and y1[j]<=0:
    #            if can_add_end : #and ( (len(steep_increase)>0 and j>steep_increase[0]) or (len(steep_decrease)>0 and j>steep_decrease[0]) ) :
    #                shoulder_ends+=[j]
    #                can_add_end=False
    #                can_add_start=True
    #                if all(numpy.array(steep_increase+steep_decrease)<j) : break # otherwise there could be a new beginning
    #        if j in steep_increase or j in steep_decrease : can_add_end=True
    #    oldv=v
    if (
        sort_steep_points
    ):  # sort only the steep points according to their magnitude (from first derivative)
        steep_increase = sorted(
            steep_increase, key=lambda x: magnitudes[x], reverse=True
        )
        steep_decrease = sorted(
            steep_decrease, key=lambda x: magnitudes[x], reverse=True
        )
    if plot:
        fig = plotter.profile(
            (y - min(y)) / (max(y) - min(y)), x, ls="-", figure=fig, label="data"
        )
        fig = plotter.profile(
            y1 / (abs(max(y1)) + abs(min(y1))), x, ls="-", figure=fig, label="d1"
        )
        fig = plotter.profile(y2, x, ls="-", figure=fig, label="d2")
        fig = plotter.profile(
            y3,
            x,
            ls="-",
            figure=fig,
            label="d3",
            legend_size=13,
            vline=[
                x[jj]
                for jj in steep_increase
                + steep_decrease
                + shoulder_begins
                + shoulder_ends
            ],
            hline=shoulder_maginitude_threshold,
        )
    return (
        steep_increase,
        steep_decrease,
        magnitudes,
        shoulder_begins,
        shoulder_ends,
        y,
        x,
    )


class Savitzky_golay_interpolation:
    def __init__(
        self,
        y,
        x=None,
        window_size=5,
        order=2,
        interpolation_rate=5,
        plot=False,
        plot_label=None,
    ):
        """
        if x is None indices will be used x=numpy.arange(0,len(y))
        :param y:
        :param x:
        :param window_size:
        :param order:
        """
        if (
            window_size > len(y)
            or type(window_size) is not int
            or type(order) is not int
            or window_size < 3
        ):
            raise ValueError(
                "window_size and order have to be of type int and window_size must be smaller or equal then length of profile y and greater than 2. Given window_size=%s order=%s len(y)=%d\n"
                % (str(window_size), str(order), len(y))
            )
        if x is None:
            x = numpy.arange(0, len(y))
        else:
            x = numpy.array(x)
        y = numpy.array(y)
        idx = numpy.arange(0, len(y))[
            (numpy.isfinite(x) & numpy.isfinite(y))
        ]  # in case there are nan or inf we ignore them
        self.yS = y[idx]
        self.xS = x[idx]
        args = numpy.argsort(self.xS)
        self.xS = self.xS[args]
        self.yS = self.yS[args]
        self.side = int(window_size / 2)
        if order > self.side:
            raise Exception("order must be smaller than int(window_size / 2)")
        self.interpolation_rate = interpolation_rate
        # save coefficients
        self.Coeffs = None
        last_coefficients = None
        for j, xj in enumerate(self.xS):
            inds = numpy.arange(
                positive(j - self.side), min([j + self.side + 1, len(self.xS)])
            )
            # print(inds,j,last_coefficients)
            # Fit a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg` ; HOWEVER then polyval wants the coefficients in opposite order hence we flip it now...
            coefficients = numpy.polyfit(self.xS[inds], self.yS[inds], order)[::-1]
            if last_coefficients is not None:  # interpolate coefficients linearly
                c = numpy.array(
                    [
                        numpy.linspace(c1, c2, self.interpolation_rate + 1)
                        for c1, c2 in zip(last_coefficients, coefficients)
                    ]
                )
                if self.Coeffs is None:
                    self.Coeffs = c
                else:
                    self.Coeffs = numpy.hstack(
                        (self.Coeffs, c[:, 1:])
                    )  # last_coefficients are already there from previous iteration
            last_coefficients = coefficients.copy()
        self.Coeffs = self.Coeffs.T
        if plot:
            f = plotter.profile(
                y,
                x,
                marker=".",
                ls="",
                markersize=8,
                markerfacecolor=(0, 0, 0, 0.4),
                figure_size=(8, 7),
            )
            xs = numpy.linspace(min(x), max(x), 1000)
            ys = self(xs)
            f = plotter.profile(
                ys,
                xs,
                marker="",
                ls="-",
                figure=f,
                title="window=%s order=%g" % (str(window_size), order),
                label=plot_label,
            )
        return

    def __call__(self, xs):
        last_coefficients = None
        x = numpy.sort(xs)
        if xs[0] < self.xS[0]:
            raise Exception(
                "Outside interpolation range %g %g (given x=%g)\n"
                % (self.xS[0], self.xS[-1], xs[0])
            )
        if xs[-1] > self.xS[-1]:
            raise Exception(
                "Outside interpolation range %g %g (given x=%g)\n"
                % (self.xS[0], self.xS[-1], xs[-1])
            )
        ys = []
        i = 0
        for x in xs:
            while i < len(self.xS) - 1 and self.xS[i] <= x:
                i += 1
            j = self.interpolation_rate * (i - 1) + int(
                self.interpolation_rate
                * (x - self.xS[i - 1])
                / (self.xS[i] - self.xS[i - 1])
                + 0.5
            )
            c = self.Coeffs[j]
            ys += [numpy.polynomial.polynomial.polyval(x, c, tensor=False)]
        return numpy.array(ys)


def savitzky_golay_interpolation(
    y, x=None, window_size=5, order=2, interpolation_rate=5
):
    """
see class Savitzky_golay_interpolation it is more versatile and better
    interpolation_rate must be an integer>1 and corresponds to the number of points added between any two consecutive points
     (inlcuding the points themselves so if interpolation_rate is 10 there will be 8 new points among 2 existing consecutive points)
    skips over NaN or Inf if present
    return YS,XS,YS2
    YS2 is another interpolated profile with a smoothing on the extrapolated coefficient - it is usually better than YS
     except for the extrema of the profile!
     in particular if one calcualtes derivatives of the interpolated profile then YS2 is much smoother and better thanYS
    Does not work for ND array, only 1D
    """
    if (
        window_size > len(y)
        or type(window_size) is not int
        or type(order) is not int
        or window_size < 3
    ):
        raise ValueError(
            "window_size and order have to be of type int and window_size must be smaller or equal then length of profile y and greater than 2. Given window_size=%s order=%s len(y)=%d\n"
            % (str(window_size), str(order), len(y))
        )
    if x is None:
        x = numpy.arange(0, len(y))
    else:
        x = numpy.array(x)
    if interpolation_rate < 2:
        sys.stderr.write(
            "**Warn in savitzky_golay_interpolation() interpolation_rate=%d but minimum allowed is 2, which does not interpolate at all but only smooth - setting to 2\n"
            % (interpolation_rate)
        )
        interpolation_rate = 2
    y = numpy.array(y)
    side = int(window_size / 2)
    idx = numpy.arange(0, len(y))[
        (numpy.isfinite(x) & numpy.isfinite(y))
    ]  # in case there are nan or inf we ignore them
    last_coefficients = None
    pol_fun = numpy.poly1d([1] * order + [1])
    XS, YS = None, None
    Coeffs = None
    for j, xj in enumerate(idx):
        inds = idx[positive(j - side) : min([j + side + 1, len(idx) - 1])]
        coefficients = numpy.polyfit(x[inds], y[inds], order)[
            ::-1
        ]  # Fit a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg` ; HOWEVER then polyval wants the coefficients in opposite order hence we flip it now...
        if last_coefficients is not None:  # interpoalte coefficients linearly
            c = numpy.array(
                [
                    numpy.linspace(c1, c2, interpolation_rate)
                    for c1, c2 in zip(last_coefficients, coefficients)
                ]
            )
            if Coeffs is None:
                Coeffs = c
            else:
                Coeffs = numpy.hstack(
                    (Coeffs, c[:, 1:])
                )  # last_coefficients are already there from previous iteration
            xs = numpy.linspace(x[idx[j - 1]], x[xj], interpolation_rate)
            # print j,idx[j-1],xj,x[idx[j-1]],x[xj]
            ys = numpy.polynomial.polynomial.polyval(xs, c, tensor=False)
            if XS is None:
                XS = xs
                YS = ys
            else:
                XS = numpy.hstack(
                    (XS, xs[1:])
                )  # first point was already added previously
                YS = numpy.hstack((YS, ys[1:]))
        last_coefficients = coefficients

    # try smoothing coefficients
    smooth_per_side = 2 * interpolation_rate
    fixed_point_weight = (
        1.0  # same weight to "real" point coefficients and interpolated..
    )
    weights = numpy.array(
        ([fixed_point_weight] + [1.0] * (interpolation_rate - 2)) * (len(y) - 1)
        + [fixed_point_weight]
    )
    smoothed = None  # we will do vstack and then transpose
    for i in range(0, len(YS)):
        s = positive(i - smooth_per_side)
        den = sum(weights[s : i + smooth_per_side + 1])
        # if weights[i]==fixed_point_weight :  sm=Coeffs[: ,i] # NOT too wise
        # else :
        sm = (
            Coeffs[:, s : i + smooth_per_side + 1]
            * weights[s : i + smooth_per_side + 1]
        ).sum(axis=1) / den
        if smoothed is None:
            smoothed = sm
        else:
            smoothed = numpy.vstack((smoothed, sm))
    # print 'smoothed.shape',smoothed.shape
    YS2 = numpy.polynomial.polynomial.polyval(XS, smoothed.T, tensor=False)
    return YS, XS, YS2


def get_interpolated_steepest(
    x,
    ydata,
    yerr=None,
    k=2,
    derivative_fraction_filter=0.1,
    s=None,
    npoints=5000,
    pre_smooth=(21, 3),
    plot=False,
):
    """
SUPERSEDED by process_profile_steep_regions
    interpolate a profile of know x values (MUST be INCREASING)
    and returns SPL,xs,ys,candidate_max_inds,candidate_min_inds
    xs ,ys are arrays of npoints corresponding to interpolated data
    """
    if npoints < 3 * len(ydata):
        npoints = 5 * len(ydata)
    SPL, xs, ys = interpolate(
        x,
        ydata,
        k=k,
        s=s,
        yerr=yerr,
        plot=False,
        npoints=npoints,
        pre_smooth=pre_smooth,
    )
    y1 = numpy.gradient(ys, xs)
    candidate_max_inds = numpy.array(argrelextrema(y1, numpy.greater)[0]).astype(
        int
    )  # like numpy where returns 2D
    candidate_min_inds = numpy.array(argrelextrema(y1, numpy.less)[0]).astype(int)
    if derivative_fraction_filter is not None and derivative_fraction_filter > 0:
        th = derivative_fraction_filter * max(
            [abs(min(y1)), abs(max(y1))]
        )  # determine what is 10% (if derivative_fraction_filter=0.1) of the maximum deviation from zero. Steepest points will be max or min.
        candidate_max_inds = numpy.array(
            [i for i in candidate_max_inds if abs(y1[i]) >= th]
        ).astype(int)
        candidate_min_inds = numpy.array(
            [i for i in candidate_min_inds if abs(y1[i]) >= th]
        ).astype(int)
    if plot:
        f = plotter.profile(ydata, x, marker=".", ls="", yerr=yerr, markersize=8)
        f = plotter.profile(ys, xs, marker="", ls="-", figure=f)
        if len(candidate_max_inds) > 0:
            f = plotter.point(
                list(zip(xs[candidate_max_inds], ys[candidate_max_inds])),
                f,
                marker="d",
                markersize=30,
                color="red",
                zorder=8,
            )
        if len(candidate_min_inds) > 0:
            f = plotter.point(
                list(zip(xs[candidate_min_inds], ys[candidate_min_inds])),
                f,
                marker="d",
                markersize=30,
                color="magenta",
                zorder=8,
            )
    return SPL, xs, ys, candidate_max_inds, candidate_min_inds


def get_profile_steepest(
    profile,
    x=None,
    interpolate=False,
    derivative_magnitude_abs_filter=0.1,
    smooth_first=True,
    savgol_filter=None,
    spline_smooth="auto",
    spline_k=3,
    interploated_npoints=5000,
):
    """
SUPERSEDED by process_profile_steep_regions
    return steep_decrease,steep_increase,magnitudes
    if not interpolate then
    steep_increase contains the index of the points of maximum steepness where the profile is increasing,
      steep_decrease the same but when decreasing and magnitudes is a dictonary where keys are these indices
      and values the magnitude of the steepnes, expressed as the absolute value of the first derivative/max(abs(derivative))
       (multiplying by the length makes profiles of different size comparable).
    if interpolate
    return xs,ys, steep_decrease,steep_increase,magnitudes, spl 
    and this time steep_decrease,steep_increase are indices in ys, the itnerpolated profile
    and if x is None xs=numpy.linspace(0,len(profile)-1,interploated_npoints)
     otherwise xs= numpy.linspace(min(x),max(x),interploated_npoints)
The interpolatio may not really work...
    """
    if savgol_filter is None:
        if not interpolate:
            savgol_filter = True
        else:
            savgol_filter = False

    pr = numpy.array(profile).copy()
    if smooth_first:
        if type(smooth_first) is not int:
            smooth_first = min([2 * int(len(profile) / 20.0) + 1, 7])
        if savgol_filter:
            savgol_filter = (smooth_first, 1)
        pr = smooth_profile(
            pr,
            smooth_per_side=smooth_first,
            weights=None,
            use_savgol_filter=savgol_filter,
        )
    if interpolate:
        # run function to guess expected number of flexes and max and min
        steep_decrease, steep_increase, magnitudes = get_profile_steepest(
            profile,
            derivative_magnitude_abs_filter=derivative_magnitude_abs_filter,
            smooth_first=smooth_first,
            savgol_filter=savgol_filter,
            interpolate=False,
        )
        exp_min_or_max, l = 1, sorted(steep_decrease + steep_increase)
        for j, i in enumerate(l):
            if i in steep_decrease and j + 1 < len(l):
                if l[j + 1] in steep_increase:
                    exp_min_or_max += 1
            elif i in steep_increase and j + 1 < len(l):
                if l[j + 1] in steep_decrease:
                    exp_min_or_max += 1
        if x is None:
            x = list(range(len(profile)))
        if type(spline_smooth) is str:
            print(
                "expected_nof_flexes=%d , expected_nof_absolute_max_or_min=%d"
                % (len(steep_decrease) + len(steep_increase), exp_min_or_max)
            )
            spl, pfit = optimise_spline(
                x,
                pr,
                expected_nof_flexes=len(steep_decrease) + len(steep_increase),
                expected_nof_absolute_max_or_min=exp_min_or_max,
            )  # IMPORTANT PARAMETERS
            # print 'spl',pfit
        else:
            spl = UnivariateSpline(x, pr, k=spline_k, s=spline_smooth)
        if x is None:
            xs = numpy.linspace(x[0], x[-1], interploated_npoints)
        else:
            xs = numpy.linspace(min(x), max(x), interploated_npoints)
        ys = spl(xs)
        # challenge is than to use xs but go back to index of profiles
        steep_decrease, steep_increase, magnitudes = get_profile_steepest(
            ys,
            interpolate=False,
            derivative_magnitude_abs_filter=derivative_magnitude_abs_filter,
            smooth_first=False,
        )
        return xs, ys, steep_decrease, steep_increase, magnitudes, spl
    # print'smooth_first:',smooth_first
    f1 = numpy.gradient(pr)
    if smooth_first:
        f1 = smooth_profile(
            f1,
            smooth_per_side=smooth_first,
            weights=None,
            use_savgol_filter=(2 * int(1.5 * smooth_first) + 1, 3),
        )
    f2 = numpy.gradient(f1)
    steep_decrease = []
    steep_increase = []
    oldv = f2[0]
    magnitudes = {}
    den = float(max(abs(f1)))
    for j, v in enumerate(f2[1:]):
        if v == 0 or v * oldv < 0:
            if (
                derivative_magnitude_abs_filter is not None
                and derivative_magnitude_abs_filter > abs(f1[j] / den)
            ):
                continue
            if f1[j] > 0 and v - oldv < 0:
                steep_increase += [j]
                magnitudes[j] = f1[j] / den  # abs(v-oldv)*len(profile)
            elif f1[j] < 0 and v - oldv > 0:
                steep_decrease += [j]
                magnitudes[j] = f1[j] / den
        oldv = v
    return steep_decrease, steep_increase, magnitudes


def chop(v):
    return v[1:]


def make_differentiable(yvs, xvs):
    """
    Useful for CDF or other MONOTONIC functions that may in principle be differentiable but 
    in practice have multiple values of y associated to same value of x
      it removes points so that one x is associated to one y
    # y values can be whatever
    ASSUMES that both x and y are monotonic
    """
    n = 0
    nxvs, nyvs = [xvs[0]], [yvs[0]]
    for j in range(1, len(xvs)):
        if xvs[j] != xvs[j - 1]:  # these are sorted integer values
            nxvs += [xvs[j]]
            nyvs += [yvs[j]]
    return numpy.array(nyvs), numpy.array(nxvs)


def grad(y, x):
    """
    useful when x values are not equally spaced
     return numpy.gradient(y)/numpy.gradient(x) 
    """
    return numpy.gradient(y) / numpy.gradient(x)


def derivative_of_noisy_data(y, dx=None, window=5, use_savgol=True):
    """
    see TVRegDiff for more advanced applications
    better to use_savgol for identification of derivative max and min
     and not for true zeros
    """
    if hasattr(dx, "__len__"):
        if len(dx) != len(y):
            sys.stderr.write(
                "**ERROR** in derivative_of_noisy_data dx either scalr or array of same length as y (given lengths %d %d)\n"
                % (len(dx), len(y))
            )
        dy = [
            numpy.gradient(y[i::window]) / numpy.gradient(dx[i::window])
            for i in range(window)
        ]
    else:
        if dx is not None:
            dx *= window
        else:
            dx = window
        dy = [numpy.gradient(y[i::window], dx) for i in range(window)]
    i, j = 0, 0  # important!
    derU = [
        dy[i][j]
        for j in range(0, len(dy[i]))
        for i in range(0, len(dy))
        if j < len(dy[i])
    ]
    if use_savgol:
        return smooth_profile(
            derU,
            smooth_per_side=window - 1,
            weights=None,
            use_savgol_filter=(2 * window - 1, 2),
        )
    else:
        return smooth_profile(
            derU, smooth_per_side=window - 1, weights=None, use_savgol_filter=False
        )


def TVRegDiff(
    data,
    itern,
    alph,
    u0=None,
    scale="small",
    ep=1e-6,
    dx=None,
    plotflag=False,
    diagflag=1,
):
    # https://github.com/stur86/tvregdiff/blob/master/tvregdiff.py
    # code starts here
    # Make sure we have a column vector
    data = numpy.array(data)
    if len(data.shape) != 1:
        print("Error - data is not a column vector")
        return
    # Get the data size.
    n = len(data)

    # Default checking. (u0 is done separately within each method.)
    if dx is None:
        dx = 1.0 / n

    # Different methods for small- and large-scale problems.
    if scale.lower() == "small":

        # Construct differentiation matrix.
        c = numpy.ones(n + 1) / dx
        D = sparse.spdiags([-c, c], [0, 1], n, n + 1)

        DT = D.transpose()

        # Construct antidifferentiation operator and its adjoint.
        def A(x):
            return chop(numpy.cumsum(x) - 0.5 * (x + x[0])) * dx

        def AT(w):
            return (
                sum(w) * numpy.ones(n + 1)
                - numpy.transpose(
                    numpy.concatenate(([sum(w) / 2.0], numpy.cumsum(w) - w / 2.0))
                )
            ) * dx

        # Default initialization is naive derivative

        if u0 is None:
            u0 = numpy.concatenate(([0], numpy.diff(data), [0]))

        u = u0
        # Since Au( 0 ) = 0, we need to adjust.
        ofst = data[0]
        # Precompute.
        ATb = AT(ofst - data)  # input: size n

        # Main loop.
        for ii in range(1, itern + 1):
            # Diagonal matrix of weights, for linearizing E-L equation.
            Q = sparse.spdiags(1.0 / (numpy.sqrt((D * u) ** 2 + ep)), 0, n, n)
            # Linearized diffusion matrix, also approximation of Hessian.
            L = dx * DT * Q * D

            # Gradient of functional.
            g = AT(A(u)) + ATb + alph * L * u

            # Prepare to solve linear equation.
            tol = 1e-4
            maxit = 100
            # Simple preconditioner.
            P = alph * sparse.spdiags(L.diagonal() + 1, 0, n + 1, n + 1)

            def linop(v):
                return alph * L * v + AT(A(v))

            linop = sparse.linalg.LinearOperator((n + 1, n + 1), linop)

            if diagflag:
                [s, info_i] = sparse.linalg.cg(
                    linop, g, x0=None, tol=tol, maxiter=maxit, callback=None, M=P
                )
                print(
                    (
                        "iteration {0:4d}: relative change = {1:.3e}, "
                        "gradient norm = {2:.3e}\n".format(
                            ii,
                            numpy.linalg.norm(s[0]) / numpy.linalg.norm(u),
                            numpy.linalg.norm(g),
                        )
                    )
                )
                if info_i > 0:
                    print("WARNING - convergence to tolerance not achieved!")
                elif info_i < 0:
                    print("WARNING - illegal input or breakdown")
            else:
                [s, info_i] = sparse.linalg.cg(
                    linop, g, x0=None, tol=tol, maxiter=maxit, callback=None, M=P
                )
            # Update solution.
            u = u - s
            # Display plot.
            if plotflag:
                f = plotter.profile(u)

    elif scale.lower() == "large":

        # Construct antidifferentiation operator and its adjoint.
        def A(v):
            return numpy.cumsum(v)

        def AT(w):
            return sum(w) * numpy.ones(len(w)) - numpy.transpose(
                numpy.concatenate(([0.0], numpy.cumsum(w[:-1])))
            )

        # Construct differentiation matrix.
        c = numpy.ones(n)
        D = sparse.spdiags([-c, c], [0, 1], n, n) / dx
        mask = numpy.ones((n, n))
        mask[-1, -1] = 0.0
        D = sparse.dia_matrix(D.multiply(mask))
        DT = D.transpose()
        # Since Au( 0 ) = 0, we need to adjust.
        data = data - data[0]
        # Default initialization is naive derivative.
        if u0 is None:
            u0 = numpy.concatenate(([0], numpy.diff(data)))
        u = u0
        # Precompute.
        ATd = AT(data)

        # Main loop.
        for ii in range(1, itern + 1):
            # Diagonal matrix of weights, for linearizing E-L equation.
            Q = sparse.spdiags(1.0 / numpy.sqrt((D * u) ** 2.0 + ep), 0, n, n)
            # Linearized diffusion matrix, also approximation of Hessian.
            L = DT * Q * D
            # Gradient of functional.
            g = AT(A(u)) - ATd
            g = g + alph * L * u
            # Build preconditioner.
            c = numpy.cumsum(list(range(n, 0, -1)))
            B = alph * L + sparse.spdiags(c[::-1], 0, n, n)
            # droptol = 1.0e-2
            R = sparse.dia_matrix(numpy.linalg.cholesky(B.todense()))
            # Prepare to solve linear equation.
            tol = 1.0e-4
            maxit = 100

            def linop(v):
                return alph * L * v + AT(A(v))

            linop = sparse.linalg.LinearOperator((n, n), linop)

            if diagflag:
                [s, info_i] = sparse.linalg.cg(
                    linop,
                    -g,
                    x0=None,
                    tol=tol,
                    maxiter=maxit,
                    callback=None,
                    M=numpy.dot(R.transpose(), R),
                )
                print(
                    (
                        "iteration {0:4d}: relative change = {1:.3e}, "
                        "gradient norm = {2:.3e}\n".format(
                            ii,
                            numpy.linalg.norm(s[0]) / numpy.linalg.norm(u),
                            numpy.linalg.norm(g),
                        )
                    )
                )
                if info_i > 0:
                    print("WARNING - convergence to tolerance not achieved!")
                elif info_i < 0:
                    print("WARNING - illegal input or breakdown")

            else:
                [s, info_i] = sparse.linalg.cg(
                    linop,
                    -g,
                    x0=None,
                    tol=tol,
                    maxiter=maxit,
                    callback=None,
                    M=numpy.dot(R.transpose(), R),
                )
            # Update current solution
            u = u + s
            # Display plot.
            if plotflag:
                f = plotter.profile(u / dx, figure=f)
        u = u / dx
    return u


def get_equivalent_error(function, parameters, xdata, xerr, yerr, h=1e-7):
    """
    #y_eq_err= sqrt(  yerr**2 + ( df/dx * xerr )**2 )
    must be function(x,p) with p a list 
    """
    return numpy.sqrt(
        yerr ** 2
        + (
            (function(xdata + h, parameters) - function(xdata - h, parameters))
            / (2 * h)
            * xerr
        )
        ** 2
    )


def parameters_box_penalty(
    self, par, lower_bound=None, upper_bound=None, cost_penalty=1e6
):
    """
        this is used to give a penalty if parameters go outside a range. if no boundaries or if all parameters respect boundaries return 0
        """
    if isinstance(par, numpy.ndarray):
        if lower_bound is not None and any(par < lower_bound):
            return cost_penalty
        if upper_bound is not None and any(par > upper_bound):
            return cost_penalty
    else:
        if lower_bound is not None and par < lower_bound:
            return cost_penalty
        if upper_bound is not None and par > upper_bound:
            return cost_penalty
    return 0.0


def bayesian_polynomial_fit(
    xdata,
    ydata,
    n_order=1,
    fit_intercept=False,
    alpha_init=[None, 1],
    lambda_init=[None, 1e-3],
):
    """
    https://scikit-learn.org/stable/modules/linear_model.html#bayesian-ridge-regression
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge
    return fitted_function, reg_coef , reg_score,regb
      reg_score being log marginal likelihood (L)
      reg_coef are the coefficient of the polynomial from order 0 to increasing e.g. reg_coef[0] + reg_coef[1]*x + reg_coef[2]*x*x + reg_coef[3]*x*x*x  ...
    yfitted, yfitted_std = fitted_function( x )
    """
    from sklearn.linear_model import BayesianRidge

    # BayesianRidge(*, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)
    # tol float, default=1e-3 Stop the algorithm if w has converged.
    # compute_score bool, default=False If True, compute the log marginal likelihood at each iteration of the optimization.
    # fit_intercept bool, default=True Whether to calculate the intercept for this model. The intercept is not treated as a probabilistic parameter and thus has no associated variance. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
    X_train = numpy.vander(
        xdata, n_order + 1, increasing=True
    )  # create polynomial at setp, will fit coefficients of these
    if hasattr(alpha_init, "__len__"):  # input grid
        if len(alpha_init) != len(lambda_init):
            print(
                "**ERROR** bayesian_polynomial_fit  len(alpha_init)!=len(lambda_init) %d %d"
                % (len(alpha_init), len(lambda_init))
            )
        bestscore = -999999999
        sampled_scores = []
        for al, la in zip(alpha_init, lambda_init):
            regb = BayesianRidge(
                tol=1e-6,
                fit_intercept=fit_intercept,
                compute_score=True,
                alpha_init=al,
                lambda_init=la,
            )
            regb.fit(X_train, ydata)
            sampled_scores += [regb.scores_[-1]]
            if regb.scores_[-1] > bestscore:
                best_comb = [regb, al, la, regb.scores_[-1]]
                bestscore = regb.scores_[-1]
            regb = best_comb[0]
    else:
        regb = BayesianRidge(
            tol=1e-6,
            fit_intercept=fit_intercept,
            compute_score=True,
            alpha_init=alpha_init,
            lambda_init=lambda_init,
        )
        regb.fit(X_train, ydata)
    fitted_function = lambda xs: regb.predict(
        numpy.vander(xs, n_order + 1, increasing=True), return_std=True
    )
    reg_coeffs = regb.coef_
    return fitted_function, reg_coeffs, regb.scores_[-1], regb


def get_truncated_normal(mean=0, sd=1, low=-10, upp=10):
    """
    :param mean:  gaussian mean
    :param sd:  gaussina sd
    :param low: lower cutoff (truncated gaussian)
    :param upp: upper cutoff
    :return: X a generator of random number can be X.rvs() -> one number
    or X.rvs(100) -> 100 numbers in distribution
    """
    return scipy.stats.truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
    )


def _adjust_yrange_boundaries(yranges_data_and_their_xvalues):
    """
    yranges are a list of tuples (or of list) expressing lower and upper bound, None can be given for
    this function adjust is to numpy array format with +/- numpy.inf for no boundaries (None)
    return yranges,xvals
    """
    yranges, xvals = (
        numpy.array(yranges_data_and_their_xvalues[0]),
        numpy.array(yranges_data_and_their_xvalues[1]),
    )
    yranges[:, 0][numpy.where(yranges[:, 0] == None)] = -numpy.inf
    yranges[:, 1][numpy.where(yranges[:, 1] == None)] = numpy.inf
    return yranges, xvals


def _fit_yrange_correction(yfitted, yranges, differentiable=True):
    """
    function is the fitting function.
    this function is added to the error function of fits in cases where
     for some xdata the correponding ydata is not known as a number but as a range, 
     so that for example it has to be greater or equal then a quantity or stay in an interval
    returns a large number if some of the yfitted are outside the yranges
    yranges are a list of tuples (or of list) expressing lower and upper bound, None can be given for
     no bound
    """
    up = yfitted >= yranges[:, 1]
    down = yfitted <= yranges[:, 0]
    if not differentiable:
        return (up.sum() + down.sum()) * 1e16
    else:  # note that when exactly equal to boundary it will be zero
        return 1e20 * (
            ((yfitted[up] - yranges[:, 1][up]) ** 2).sum()
            + ((yranges[:, 0][down] - yfitted[down]) ** 2).sum()
        )


def basinhopping_fit(
    function,
    xdata,
    ydata,
    p_guess,
    p_lower_boundaries=None,
    p_upper_boundaries=None,
    basinhopping_T=1,
    basinhopping_niter=200,
    stepsize_per_par=None,
    refine_guess_with_leastsquare=True,
):
    """
    # use numpy.inf and -numpy.inf in p_lower_boundaries and p_upper_boundaries
    """
    errfunc = lambda p: ((ydata - function(xdata, p)) ** 2)
    # run a leastsquare to guess something vaguely reasonable
    if refine_guess_with_leastsquare:
        pguess_from_leastsq, _, _, _, _ = scipy.optimize.leastsq(
            errfunc, p_guess, full_output=1, maxfev=100000
        )
    else:
        pguess_from_leastsq = p_guess
    # custom boundaries
    mybounds = None
    if p_upper_boundaries is not None or p_lower_boundaries is not None:
        if p_upper_boundaries is None:
            p_upper_boundaries = [numpy.inf for p in p_guess]
        if p_lower_boundaries is None:
            p_lower_boundaries = [-numpy.inf for p in p_guess]
        if len(p_upper_boundaries) != len(p_guess):
            raise Exception(
                "**basinhopping_fit len(p_upper_boundaries)!=len(p_guess) %d %d use numpy.inf as a non-existing boundary"
                % (len(p_upper_boundaries), len(p_guess))
            )
        if len(p_lower_boundaries) != len(p_guess):
            raise Exception(
                "**basinhopping_fit len(p_lower_boundaries)!=len(p_guess) %d %d use -numpy.inf as a non-existing boundary"
                % (len(p_lower_boundaries), len(p_guess))
            )
        p_upper_boundaries = numpy.array(p_upper_boundaries)
        p_lower_boundaries = numpy.array(p_lower_boundaries)

        class MyBounds(object):
            def __init__(self, xmin=p_lower_boundaries, xmax=p_upper_boundaries):
                self.xmax = numpy.array(xmax)
                self.xmin = numpy.array(xmin)

            def __call__(self, **kwargs):
                x = kwargs["x_new"]
                tmax = bool(numpy.all(x <= self.xmax))
                tmin = bool(numpy.all(x >= self.xmin))
                return tmax and tmin

        mybounds = MyBounds()
        pguess_from_leastsq = [
            0.98 * p_upper_boundaries[j] if x > p_upper_boundaries[j] else x
            for j, x in enumerate(pguess_from_leastsq)
        ]
        pguess_from_leastsq = [
            1.02 * p_lower_boundaries[j] if x < p_lower_boundaries[j] else x
            for j, x in enumerate(pguess_from_leastsq)
        ]
    # custom stepsize
    mytakestep = None
    if stepsize_per_par is not None:
        if len(stepsize_per_par) != len(p_guess):
            raise Exception(
                "**basinhopping_fit len(stepsize_per_par)!=len(p_guess) must give a step for each parameter"
                % (len(stepsize_per_par), len(p_guess))
            )

        class MyTakeStep(object):
            def __init__(self, stepsize=0.5, stepsize_per_par=None):
                self.stepsize = stepsize
                self.stepsize_per_par = numpy.array(stepsize_per_par)
                if len(self.stepsize_per_par) == 1:
                    self.stepsize_per_par = self.stepsize_per_par[0]  # scalar

            def __call__(self, x):
                x += numpy.random.uniform(-self.stepsize_per_par, self.stepsize_per_par)
                return x

        mytakestep = MyTakeStep(stepsize_per_par=stepsize_per_par)
    errfunc_basin = lambda p: numpy.sum(errfunc(p))
    # minimizer_kwargs = {"args":(xdata, ydata, glob_loc_separator, num_local_params) } #"method": "BFGS"}
    res = scipy.optimize.basinhopping(
        errfunc_basin,
        pguess_from_leastsq,
        niter=basinhopping_niter,
        T=basinhopping_T,
        accept_test=mybounds,
        take_step=mytakestep,
    )
    pfit = res.x
    # success = res.success
    nfev = res.nfev
    errmsg = res.message
    # run a leastsquare from results to estimate pcov and also check consistency
    pfit_leastsq, pcov, infodict, _, success = scipy.optimize.leastsq(
        errfunc, pfit, full_output=1, maxfev=100000
    )
    # if debug :
    print(
        "DEB basinhopping fit:\npars_basinhopping= %s [Sq_err=%g]\npars_leastsquare = %s  [Sq_err=%g]\npguess_from_leas = %s  [Sq_err=%g] nfev=%d"
        % (
            str(pfit),
            errfunc_basin(pfit),
            str(pfit_leastsq),
            errfunc_basin(pfit_leastsq),
            str(pguess_from_leastsq),
            errfunc_basin(pguess_from_leastsq),
            nfev,
        )
    )
    return pfit, pfit_leastsq, pcov


def fit_function(
    function,
    xdata,
    ydata,
    p_guess,
    yerr=None,
    xerr=None,
    p_boundaries=None,
    yranges_data_and_their_xvalues=None,
    local_params_indices=None,
    local_args_for_global=None,
    use_basinhopping=False,
    hop_around=None,
    unbounded_proxy_value=9942515536,
    add_mean_residual_to_yerr=True,
    do_data_resampling=False,
    random_parameter_guess_for_bootstrap=False,
    bootstrap_cycles=False,
    debug=False,
    ci=0.95,
):
    """
    return pfit_leastsq,perr_leastsq, s_sq, sum_of_squared_residuals    if NO bottstrap otherwise
    return pfit_leastsq,perr_leastsq, pfit_bootstrap, perr_bootstrap,pCI_down,pCI_up,ci_error_bars, s_sq,sum_of_squared_residuals,parameter_ensemble
      only if bootstrap it also computes equivalent error if xerr is given
    p_boundaries should be a list of length p_guess
     containing tuples with the boundaries, for speed reason the exact int value in unbounded_proxy_value should be given for unspecified limit
      e.g. p_boundaries=[(0,unbounded_proxy_value), (unbounded_proxy_value,unbounded_proxy_value),(2,4)]
    yranges_data_and_their_xvalues can be given as [ yranges, corresponding_xvalues] in cases where some ydata are only defined as ranges
       ,so that for example each value has to be greater then a quantity or stay in an interval
       yranges are a list of tuples (or of list) expressing lower and upper bound, None can be given for
       yranges_data_and_their_xvalues is not implemented for global fits
    for global it is always function(x,p) and one can give local_params_indices to specify which parameters in p_guess
      are to be treated as local, otherwise all parameters are treated as global.
      global parameters must be FIRST in the list of p_guess (so write function(x,p) accordingly with lambda x,p : if needed)
      There are two ways of entering local parameters in p_guess, the reccommended one is to enter one guess for each profile in ydata in the same order
       of the profiles. So if the function takes 2 local parameters these should be [ parl1_y1, parl2_y1, parl1_y2, parl2_y2,...] (y are the various profile)
         in this case one must ensure that the input local_params_indices reflects all of these local parameters
         in this case the actual number of local parameters (for each profile in ydata) is then estimated from the number of local parameters in p_guess and the number of profiles in ydata.
        The second way is to give only one guess per local parameter (same guess for all y profiles) but it will work only if the number of profiles is greater than the number of local parameters.
      the x-axis may be shared by all functions and ydata can be a list of lists ro a 2D array
    ISSUE for global fit, if local parameters are not the last in parameter_guess will return them in different order with global parameters first and local last 
     local_args_for_global is a list of argument that varies between lines in ydata (e.g. concentration of analyte in bli experiment)
     if given it is assumed that the function will be
     function(x,p,args) the number of args will be inferred by the number of lines in ydata vs the length of local_args_for_global so that nargs=len(local_args_for_global)/len(ydata)
     COULD ADD maxfev : int The maximum number of calls to the function. If zero, then 100*(N+1) is the maximum where N is the number of elements in x0.
    """
    # Leastsq from scipy.optimize assumes that the objective function is based on the
    # difference between some observed target data (ydata) and a (non-linear)
    # function of the parameters `f(xdata, params)` ideally differentiable:
    #       errfunc(params) = ydata - f(xdata, params)
    # so that the objective function is :
    #       Min   sum((ydata - f(xdata, params))**2, axis=0)
    #     params

    xdata = numpy.array(xdata[:])
    ydata = numpy.array(ydata[:])
    # REMOVE NAN FROM THESE DATA first from ydata (where you should have and then double check if there are some in xdata)
    if "float" in str(ydata.dtype) or "int" in str(
        ydata.dtype
    ):  # otherwise nan removal will fail - hopefully no nans when arrays of mutile (and different) dimensions are given
        non_nan_inds = ~numpy.isnan(ydata)
        if not non_nan_inds.all():
            ydata = ydata[non_nan_inds]
            xdata = xdata[non_nan_inds]
            if yerr is not None:
                yerr = numpy.array(yerr[:])
                if (
                    hasattr(yerr[0], "__len__")
                    and not hasattr(ydata[0], "__len__")
                    and len(yerr) == 2
                ):  # CI not yerr
                    yerr = numpy.array([ye[non_nan_inds] for ye in yerr])
                else:
                    yerr = yerr[non_nan_inds]
            if numpy.nan in xdata:
                n2 = ~numpy.isnan(xdata)
                ydata = ydata[n2]
                xdata = xdata[n2]
                if yerr is not None:
                    yerr = yerr[n2]
    # check if the fit is global
    p_guess = numpy.array(p_guess)
    nargs = 0
    if (
        (len(ydata.shape) > 1 and ydata.shape[0] > 1 and ydata.shape[1] > 1)
        or local_args_for_global is not None
        or local_params_indices is not None
    ):  # there are wierd array with shape (20,0) - if your ydatas don't have same length than ydata.shape will be (nprofiles, ) rather than (nprofiles, profile_len)
        is_global = True
        if debug:
            print("Fitting globally")
        if local_params_indices is None:
            npl = 0
        else:
            npl = len(local_params_indices)
        Ndof = len(ydata) * len(ydata[0]) - (
            len(p_guess) - npl + npl * len(ydata)
        )  # approximation exact only if all profiles have same number of points
        # len(ydata)-len(p_guess)
    else:
        glob_loc_separator, num_local_params = None, None
        is_global = False
        Ndof = len(ydata) - len(p_guess)  # degrees of freedom
        if debug:
            print("Not global fit, Ndof=", Ndof)
    # process p_boundaries (which define the ranges where the fitted parameters are allowed to vary)
    if p_boundaries is not None:
        lower_bounds, upper_bounds = list(
            map(numpy.array, list(zip(*p_boundaries)))
        )  # Below we don't give abs as we want the funciton to be derivable within the p_boundaries
        if yranges_data_and_their_xvalues is not None:
            yranges, xvals = _adjust_yrange_boundaries(yranges_data_and_their_xvalues)
            if debug:
                print(
                    " fit with ranges _adjust_yrange_boundaries for xvals=%s\n and yranges=%s"
                    % (str(xvals), str(yranges))
                )
            # args is given here as a proxy because if the fit is not global we give some None args to errfunc (if global args are used and are not None, see if is_global below)
            errfunc = (
                lambda p, x, y, *args: (y - function(x, p))
                + _fit_yrange_correction(function(xvals, p), yranges)
                + 1e9
                * (
                    (p > upper_bounds)[
                        numpy.where(upper_bounds != unbounded_proxy_value)[0]
                    ]
                ).sum()
                + 1e9
                * (
                    (p < lower_bounds)[
                        numpy.where(lower_bounds != unbounded_proxy_value)[0]
                    ]
                ).sum()
            )
        else:
            # args is given here as a proxy because if the fit is not global we give some None args to errfunc (if global args are used and are not None, see if is_global below)
            errfunc = (
                lambda p, x, y, *args: (y - function(x, p))
                + 1e9
                * (
                    (p > upper_bounds)[
                        numpy.where(upper_bounds != unbounded_proxy_value)[0]
                    ]
                ).sum()
                + 1e9
                * (
                    (p < lower_bounds)[
                        numpy.where(lower_bounds != unbounded_proxy_value)[0]
                    ]
                ).sum()
            )
    else:
        if yranges_data_and_their_xvalues is not None:
            yranges, xvals = _adjust_yrange_boundaries(yranges_data_and_their_xvalues)
            if debug:
                print(
                    " fit with ranges for xvals=%s\n and yranges=%s"
                    % (str(xvals), str(yranges))
                )
            errfunc = lambda p, x, y, *args: (
                y - function(x, p)
            ) + _fit_yrange_correction(function(xvals, p), yranges)
        else:
            errfunc = lambda p, x, y, *args: y - function(
                x, p
            )  # this error function ignores the errors on the x and y - no abs required as we don't add extra terms
    if (
        is_global
    ):  # assumes y is a list of arrays or 2D array and x may or not be the same for all
        if p_boundaries is not None:
            sys.stderr.write(
                "WARNING p_boundaries not yet supported in global fitting\n"
            )
        if local_params_indices is not None:
            local_params_indices = numpy.array(local_params_indices)
            local_params_guess = p_guess[local_params_indices]
            p_global = numpy.array(
                [
                    p_guess[j]
                    for j in range(len(p_guess))
                    if j not in local_params_indices
                ]
            )
            glob_loc_separator = len(p_global)
            p_guess = p_global
            if len(ydata) > len(
                local_params_guess
            ):  # assumes that only one guess has been given for one local parameters (otherwise a guess may given for its value in each profile)
                for j in range(len(ydata)):
                    p_guess = numpy.hstack((p_guess, local_params_guess.copy()))
                num_local_params = len(local_params_guess)
            else:
                if len(local_params_guess) % (len(ydata)) != 0:
                    sys.stderr.write(
                        "\n**ERROR** in fit_function global local_params_guess is given but their number is not a multiple of len(ydata) [number of fits %d %d]\n"
                        % (len(local_params_guess), (len(ydata)))
                    )
                p_guess = numpy.hstack(
                    (p_guess, local_params_guess.copy())
                )  # no for loop has we already have a set of guesses per profile.
                num_local_params = len(local_params_guess) // (len(ydata))
        else:
            glob_loc_separator = len(p_guess)
            num_local_params = 0
        if local_args_for_global is not None and hasattr(
            local_args_for_global, "__len__"
        ):
            nargs = len(local_args_for_global) // (len(ydata))
            if len(local_args_for_global) % (len(ydata)) != 0:
                sys.stderr.write(
                    "\n**ERROR** in fit_function global local_args_for_global is given but their number is not a multiple of len(ydata) [number of fits]\n"
                )
            if debug:
                print(
                    " fit global local_args_for_global is not None, nargs=%d" % (nargs)
                )
            # this function is constructed so that the local arguments are already passed to the relevant set of data in the function
            if hasattr(
                xdata[0], "__len__"
            ):  # x is not the same for all, different x are given for different curves.
                if debug:
                    print(
                        " fit global with %d xdata for %d ydata"
                        % (len(xdata), len(ydata))
                    )
                    print(
                        " fit function=", repr(function)
                    )  # ,"fit xvals=",xdata,'xdata[0]',numpy.array(xdata[0]) )
                errfunc = lambda p, x, y, glob_loc_sep, num_loc: numpy.concatenate(
                    numpy.array(
                        [
                            numpy.array(y[j])
                            - numpy.array(
                                function(
                                    numpy.array(x[j]),
                                    list(p)[:glob_loc_sep]
                                    + list(p[glob_loc_sep:])[
                                        num_loc * j : num_loc * (j + 1)
                                    ],
                                    *local_args_for_global[j * nargs : (j + 1) * nargs]
                                )
                            )
                            for j in range(len(y))
                        ]
                    )
                )
            else:
                if debug:
                    print(" fit global with same xdata for %d ydata" % (len(ydata)))
                errfunc = lambda p, x, y, glob_loc_sep, num_loc: numpy.concatenate(
                    y
                    - numpy.array(
                        [
                            function(
                                x,
                                list(p)[:glob_loc_sep]
                                + list(p[glob_loc_sep:])[
                                    num_loc * j : num_loc * (j + 1)
                                ],
                                *local_args_for_global[j * nargs : (j + 1) * nargs]
                            )
                            for j in range(len(y))
                        ]
                    )
                )
        else:
            # maybe the numpy array below needs to be flatten
            if hasattr(xdata[0], "__len__"):
                if debug:
                    print(
                        " fit global with %d xdata for %d ydata"
                        % (len(xdata), len(ydata))
                    )
                errfunc = lambda p, x, y, glob_loc_sep, num_loc: numpy.concatenate(
                    numpy.array(
                        [
                            numpy.array(y[j])
                            - function(
                                numpy.array(x[j]),
                                list(p)[:glob_loc_sep]
                                + list(p[glob_loc_sep:])[
                                    num_loc * j : num_loc * (j + 1)
                                ],
                            )
                            for j in range(len(y))
                        ]
                    )
                )
            else:
                if debug:
                    print(" fit global with same xdata for %d ydata" % (len(ydata)))
                errfunc = lambda p, x, y, glob_loc_sep, num_loc: numpy.concatenate(
                    y
                    - numpy.array(
                        [
                            function(
                                x,
                                list(p)[:glob_loc_sep]
                                + list(p[glob_loc_sep:])[
                                    num_loc * j : num_loc * (j + 1)
                                ],
                            )
                            for j in range(len(y))
                        ]
                    )
                )

    ##################################################
    ## 1. COMPUTE THE FIT AND FIT ERRORS USING leastsq
    ##################################################

    # If using scipy.optimize.leastsq, the covariance returned is the
    # reduced covariance or fractional covariance, as explained
    # here :
    # http://stackoverflow.com/questions/14854339/in-scipy-how-and-why-does-curve-fit-calculate-the-covariance-of-the-parameter-es
    # One can multiply it by the reduced chi squared, s_sq, as
    # it is done in the more recenly implemented scipy.curve_fit
    # The errors in the parameters are then the square root of the
    # diagonal elements.
    if debug:
        print(
            "DEB fit glob_loc_separator=",
            glob_loc_separator,
            "p_guess=",
            list(p_guess),
            "num_local_params=",
            num_local_params,
        )
    if (
        hop_around is not None and len(p_guess) < 4
    ):  # does it on grid around p_guess and save best solution
        if hasattr(hop_around, "__len__"):
            hop_factor, n_hops = hop_around
        else:
            if type(hop_around) is int or type(hop_around) is float:
                hop_factor = hop_around
            else:
                hop_factor = 10.0  # one order of magintude more or less
            n_hops = max([20 // len(p_guess), 3])
        grid = cartesian_product(
            *[
                numpy.hstack(
                    (
                        numpy.linspace(p / hop_factor, p, n_hops),
                        numpy.linspace(p, p * hop_factor, n_hops)[1:],
                    )
                )
                for p in p_guess
            ]
        )
        best = [9e99, p_guess]
        for g in grid:
            pfit, pcov, infodict, errmsg, success = scipy.optimize.leastsq(
                errfunc,
                g,
                args=(xdata, ydata, glob_loc_separator, num_local_params),
                full_output=1,
                maxfev=100000,
            )
            s_sq = (
                numpy.abs(
                    errfunc(pfit, xdata, ydata, glob_loc_separator, num_local_params)
                )
            ).sum()  #
            # if all(g==numpy.array(p_guess)):
            #    print "GUESS",(numpy.abs(errfunc(numpy.array(p_guess), xdata, ydata,glob_loc_separator,num_local_params))).sum()
            if s_sq < best[0]:
                best = [s_sq, g, pfit, pcov, infodict, errmsg, success]
            #    print 'hopDEB ',g,s_sq,'BEST',errmsg, success
            # else : print 'hopDEB ',g,s_sq,errmsg, success
        if debug:
            print(
                "   fit hop_around [never changes sign or goes to zero from guess] (hop_factor,n_hops)=%s p_guess_input=%s comb_tried=%d"
                % (str((hop_factor, n_hops)), str(p_guess), len(grid)),
                end=" ",
            )
        s_sq, p_guess, pfit, pcov, infodict, errmsg, success = best
        if debug:
            print(" best =%s" % (str(p_guess)))
    else:
        try:
            if debug:
                print("DEB fit optimize.leastsq %d ydata" % (len(ydata)))
                print(
                    "  xdata=",
                    xdata,
                    "glob_loc_separator=",
                    glob_loc_separator,
                    "num_local_params=",
                    num_local_params,
                    "p_guess=",
                    p_guess,
                )
            if use_basinhopping:
                # run a leastsquare to guess something vaguely reasonable
                pguess_from_leastsq, _, _, _, _ = scipy.optimize.leastsq(
                    errfunc,
                    p_guess,
                    args=(xdata, ydata, glob_loc_separator, num_local_params),
                    full_output=1,
                    maxfev=100000,
                )

                minimizer_kwargs = {
                    "args": (xdata, ydata, glob_loc_separator, num_local_params)
                }  # "method": "BFGS"}
                basinhopping_niter = 200
                basinhopping_T = 1  # should be mean of errfunc : numpy.mean(misc.flatten(ydata))/5. #
                basin_errfunc = lambda p, *args: numpy.sum((errfunc(p, *args)))
                res = scipy.optimize.basinhopping(
                    basin_errfunc,
                    pguess_from_leastsq,
                    niter=basinhopping_niter,
                    T=basinhopping_T,
                    minimizer_kwargs=minimizer_kwargs,
                )
                pfit = res.x
                # success = res.success
                nfev = res.nfev
                errmsg = res.message
                # run a leastsquare from results to estimate pcov and also check consistency
                pfit_leastsq, pcov, infodict, _, success = scipy.optimize.leastsq(
                    errfunc,
                    pfit,
                    args=(xdata, ydata, glob_loc_separator, num_local_params),
                    full_output=1,
                    maxfev=100000,
                )
                # if debug :
                print(
                    "DEB basinhopping fit:\npars_basinhopping= %s\npars_leastsquare = %s"
                    % (str(pfit), str(pfit_leastsq))
                )
            else:
                pfit, pcov, infodict, errmsg, success = scipy.optimize.leastsq(
                    errfunc,
                    p_guess,
                    args=(xdata, ydata, glob_loc_separator, num_local_params),
                    full_output=1,
                    maxfev=100000,
                )
                nfev = infodict["nfev"]
        except Exception:
            sys.stderr.write(
                "\n\nmisc.fit_function: potentially useful info to understand error: use_basinhopping=%s glob_loc_separator=%s nargs(number of local args)=%s  num_local_params(number of local parameters)=%s len(ydata)=%s len(p_guess)=%s\n"
                % tuple(
                    map(
                        str,
                        [
                            use_basinhopping,
                            glob_loc_separator,
                            nargs,
                            num_local_params,
                            len(ydata),
                            len(p_guess),
                        ],
                    )
                )
            )
            raise

    if debug:
        print(
            "DEB fit success,nfev,errmsg:",
            success,
            nfev,
            errmsg,
            "Ndof=",
            Ndof,
            "pfit=",
            pfit,
        )
    s_sq, sum_of_squared_residuals = None, 1
    sum_of_squared_residuals = (
        errfunc(pfit, xdata, ydata, glob_loc_separator, num_local_params) ** 2
    ).sum()  # SS_res
    if Ndof > 0 and pcov is not None:
        s_sq = sum_of_squared_residuals / (Ndof)  # almost the reduced Chi2
        pcov = pcov * s_sq

    error = []
    for i in range(len(pfit)):
        try:
            error.append(
                numpy.absolute(pcov[i][i]) ** 0.5
            )  # perr = numpy.sqrt(np.diag(pcov)).
        except:
            error.append(0.00)

    pfit_leastsq = numpy.array(pfit)
    perr_leastsq = numpy.array(error)
    if bootstrap_cycles is None or bootstrap_cycles == False:
        if debug:
            print(
                " fit returning after leastsq sum_of_squared_residuals=",
                sum_of_squared_residuals,
            )
        return pfit_leastsq, perr_leastsq, s_sq, sum_of_squared_residuals

    ####################################################
    ## 2. COMPUTE THE FIT AND FIT ERRORS USING bootstrap (but not data resampling)
    ####################################################

    # An issue arises with scipy.curve_fit when errors in the y data points
    # are given.  Only the relative errors are used as weights, so the fit
    # parameter errors, determined from the covariance do not depended on the
    # magnitude of the errors in the individual data points.  This is clearly wrong.
    #
    # To circumvent this problem I have implemented a simple bootstraping
    # routine that uses some Monte-Carlo to determine the errors in the fit
    # parameters.  This routines generates random ydata points starting from
    # the given ydata plus a random variation.
    #
    # The random variation is determined from average standard deviation of y
    # points in the case where no errors in the y data points are avaiable.
    #
    # If errors in the y data points are available, then the random variation
    # in each point is determined from its given error.
    #
    # A large number of random data sets are produced, each one of them is fitted
    # an in the end the variance of the large number of fit results is used as
    # the error for the fit parameters.

    # Estimate the confidence interval of the fitted parameter using
    # the bootstrap Monte-Carlo method
    # http://phe.rockefeller.edu/LogletLab/whitepaper/node17.html
    if not do_data_resampling:
        # if is_global and len(ydata.shape)<2 : # profiles have different lengths (it happens when the ydata is saved as object rather than float)
        #    raise Exception("GLOBAL FIT with bootstrap not yet implemented when profiles have different lengths %s\n" % (str([len(y) for y in ydata])))

        parameter_ensemble = []
        residuals = errfunc(
            pfit_leastsq, xdata, ydata, glob_loc_separator, num_local_params
        )  # note that residual are expected to be both positive and negative with mean about 0 (because of the minimisation of pfit_leastsq and the way errfunc is written)
        # print ("DEB: residuals=",list(residuals))
        if (
            is_global
        ):  # in this  implementation we would get as error the maximum error among the mean one calculated on all profiles, [instead we could get a mean error among everything like in non-global fits]
            boundaries = numpy.cumsum(
                [0] + [len(ty) for ty in ydata]
            )  # slower implementation than numpy.reshape but works for vectors of different lengths
            reshape = lambda x: numpy.array(
                [
                    numpy.array(x[boundaries[j] : boundaries[j + 1]])
                    for j in range(len(boundaries) - 1)
                ]
            )  # takes a concatenated array and gives original "shape" of ydata, but works also for vectors of different length
            s_res = [numpy.std(rr) for rr in reshape(residuals)]
            s_res = max(s_res)
            if yerr is not None:
                # CI for global not implemented
                yerr = numpy.array(
                    [numpy.array(ye) for ye in yerr]
                )  # list compression useful only if profiles have different lengths
            if xerr is not None:
                xerr = numpy.array(
                    [numpy.array(xe) for xe in xerr]
                )  # list compression useful only if profiles have different lengths
                yerr = get_equivalent_error(
                    function, pfit_leastsq, xdata, xerr, yerr
                )  # UNLIKELY TO WORK FOR DIFFERENT LENGTHS
            # s_res = s_res.reshape( (ydata.shape[0],-1) ) # this can be used to sample more wrong curves with larger sigma but at the end the global parameters are fitted on all curves so using the maximum is more conservative
            # print "DEB",s_res,'local_args_for_global:',local_args_for_global,'residuals',residuals
        else:
            s_res = numpy.std(
                residuals, axis=-1
            )  # get a sort of average error to be used in booststrap if the yerrors are not given
            # print "DEB: s_res(std)=%lf mean of abs=%lf mean=%lf"  % (s_res,numpy.mean(numpy.abs(residuals),axis=-1),numpy.mean(residuals))
            if yerr is not None:
                if hasattr(
                    yerr[0], "__len__"
                ):  # a confidence interval/asymmetric error bars has been given as input
                    err = numpy.nanmax(
                        numpy.array(yerr).copy(), axis=0
                    )  # max side of the asymmetric CI
                    del yerr  # convoluted way to make sure the original yerr is not overwritten
                    yerr = err
                yerr = numpy.array(yerr)
            if xerr is not None:  # convert to equivalent error
                xerr = numpy.array(xerr)
                yerr = get_equivalent_error(function, pfit_leastsq, xdata, xerr, yerr)
            if yerr is not None:
                yerr[yerr <= 0] = 1e-5
                yerr[numpy.isnan(yerr)] = 1e-5
        # bootstrap_cycles random data sets are generated and fitted
        # get this generator to shuffle around the parameter guess
        # if p_boundaries is not None:
        #    gen_rand_pguess=[]
        #    for ji, pg in enumerate(pfit_leastsq):
        #        rsigma= 0.2*numpy.abs(pg)
        #        if p_boundaries[ji] is None:
        #            low=pg - 5*rsigma
        #            upp=pg + 5*rsigma
        #        else :
        #            if p_boundaries[ji][0] is None or unbounded_proxy_value==p_boundaries[ji][0] :
        #                sys.stderr.write(" *** NOTE **** this type of bootstrap fit may work better if all boundaries are given as input and are as narrow as possible!***\n")
        #                low= pg - 5*rsigma
        #            else : low= p_boundaries[ji][0]
        #            if p_boundaries[ji][1] is None or unbounded_proxy_value==p_boundaries[ji][1] : upp= pg + 5*rsigma
        #            else :
        #                upp= p_boundaries[ji][1]
        #        print("DEB: %d mean=%g sd=%g low=%g upp=%g"%(ji,pg,rsigma,low,upp))
        #        gen_rand_pguess+=[ get_truncated_normal(mean=pg,sd=rsigma ,low=low, upp=upp ) ]
        if (
            add_mean_residual_to_yerr
            and hasattr(residuals, "shape")
            and hasattr(yerr, "shape")
            and yerr.shape != residuals.shape
        ):
            if len(yerr.shape) >= 2 and len(ydata.shape) >= 2:  # else do nothing
                residuals = residuals.reshape(yerr.shape)

        for i in range(
            bootstrap_cycles
        ):  # the optimize function only wants a function that returns a single array of floats thus this cannot be made faster.
            if (
                is_global and len(ydata.shape) < 2
            ):  # profiles have different lengths (it happens when the ydata is saved as object rather than float)
                # slower implementation in this case.. necessary to handle different lengths of profiles
                if yerr is None:
                    randomdataY = [
                        ydata[j] + numpy.random.normal(0.0, s_res, len(ydata[j]))
                        for j in range(len(ydata))
                    ]
                elif add_mean_residual_to_yerr:
                    randomdataY = [
                        ydata[j]
                        + numpy.random.normal(0.0, yerr[j] + s_res, len(ydata[j]))
                        for j in range(len(ydata))
                    ]
                else:
                    randomdataY = [
                        ydata[j] + numpy.random.normal(0.0, yerr[j], len(ydata[j]))
                        for j in range(len(ydata))
                    ]
            elif yerr is None:
                randomdataY = ydata + numpy.random.normal(
                    0.0, s_res + numpy.abs(residuals) / 3.0, ydata.shape
                )
            elif add_mean_residual_to_yerr:
                randomdataY = ydata + numpy.random.normal(
                    0.0,
                    numpy.array(yerr) + s_res + numpy.abs(residuals) / 3.0,
                    ydata.shape,
                )  # yerr is a numpy array, so all stdev of these gaussians are in principle different
            else:
                randomdataY = ydata + numpy.random.normal(
                    0.0, yerr, ydata.shape
                )  # OLD: numpy.array( [ numpy.random.normal(0., derr,1)[0] for derr in yerr ] )
            if random_parameter_guess_for_bootstrap:
                if p_boundaries is None:
                    par_rand_guess = pfit_leastsq + numpy.random.normal(
                        0, 0.2 * numpy.abs(pfit_leastsq), pfit_leastsq.shape
                    )  # perturb leastsq parameters by 10%
                else:
                    # par_rand_guess=[ x.rvs() for x in gen_rand_pguess ]
                    par_rand_guess = []
                    for ji, pleast in enumerate(pfit_leastsq):
                        sadd = numpy.random.normal(0, 0.2 * numpy.abs(pleast))
                        if p_boundaries[ji] is None:
                            par_rand_guess += [pleast + sadd]
                        elif (
                            p_boundaries[ji][0] is None
                            or unbounded_proxy_value == p_boundaries[ji][0]
                        ) and sadd < 0:
                            par_rand_guess += [pleast + sadd]
                        elif (
                            p_boundaries[ji][1] is None
                            or unbounded_proxy_value == p_boundaries[ji][1]
                        ) and sadd >= 0:
                            par_rand_guess += [pleast + sadd]
                        elif pleast + sadd <= p_boundaries[ji][0]:
                            par_rand_guess += [pleast]
                        elif pleast + sadd >= p_boundaries[ji][1]:
                            par_rand_guess += [pleast]
                        else:
                            par_rand_guess += [pleast + sadd]
            else:
                par_rand_guess = pfit_leastsq  # useful to remove the random perturbations on guessed parameters
            randomfit, _, _, errmsg2, success2 = scipy.optimize.leastsq(
                errfunc,
                par_rand_guess,
                args=(xdata, randomdataY, glob_loc_separator, num_local_params),
                full_output=1,
                maxfev=1000,
            )
            #           scipy.optimize.leastsq( errfunc, p_guess, args=(xdata, ydata,glob_loc_separator,num_local_params),full_output=1,maxfev=100000)
            if (i == 0 or i == 1) and all(randomfit == par_rand_guess):
                sys.stderr.write(
                    "\n***FIT WARNING****"
                    + str(i)
                    + "all(randomfit==par_rand_guess) try setting random_parameter_guess_for_bootstrap=True "
                    + str(randomfit)
                    + str(par_rand_guess)
                    + str(pfit_leastsq)
                    + errmsg2
                    + str(success2)
                    + str(success)
                    + "\nrandomdataY.shape="
                    + str(randomdataY.shape)
                    + "\n"
                )  # +str(randomdataY)
                sys.stderr.flush()
            parameter_ensemble.append(randomfit)  # becomes a matrix

        parameter_ensemble = numpy.array(parameter_ensemble)
        pCI_down, pCI_up = numpy.nanpercentile(
            parameter_ensemble,
            [100.0 * (0.5 - ci / 2.0), 100.0 * (0.5 + ci / 2.0)],
            axis=0,
        )  # default 2.5 97.5 percentiles
        pfit_bootstrap = numpy.nanmedian(
            parameter_ensemble, axis=0
        )  # should be mean but ensembe should be gaussian.. (hopefully I'm not 100% sure) Median is better in case some points make fit diverge or similar
        Nsigma = 1.0  # 1sigma gets approximately the same as methods above
        # 1sigma corresponds to 68.3% confidence interval
        # 2sigma corresponds to 95.44% confidence interval
        perr_bootstrap = Nsigma * numpy.nanstd(parameter_ensemble, axis=0)
        ci_error_bars = [pfit_bootstrap - pCI_down, pCI_up - pfit_bootstrap]
        sum_of_squared_residuals = (
            errfunc(pfit_bootstrap, xdata, ydata, glob_loc_separator, num_local_params)
            ** 2
        ).sum()
        if Ndof > 0:
            s_sq = sum_of_squared_residuals / (Ndof)  # the reduced Chi2
        # print ("DEB: fitted with bootstrap on errors - s_res=",s_res,'residuals=',residuals,'pCI_down=',pCI_down,'pCI_up=',pCI_up)
        return (
            pfit_leastsq,
            perr_leastsq,
            pfit_bootstrap,
            perr_bootstrap,
            pCI_down,
            pCI_up,
            ci_error_bars,
            s_sq,
            sum_of_squared_residuals,
            parameter_ensemble,
        )
    ####################################################
    ## 2. COMPUTE THE FIT AND FIT ERRORS USING bootstrap and data resampling
    ####################################################
    # There are two approaches to bootstrapping for fitting.
    #   the one implemented above is very similar to 1 below:
    #    1- In the first approach, the OLS fit is computed from the original data. The residuals are then resampled. The residuals are then added to the predicted values of the original fit to obtain a new Y vector.
    #       This new Y vector is then fit against the original X variables. We call this approach residual resampling (or the Efron approach).
    #    2- In the second approach, rows of the original data (both the Y vector and the corresponding rows of the X variables) are resampled. The resampled data are then fit. We call this approach data resampling (or the Wu approach).
    #
    #   Hamilton (see Reference below) gives some guidance on the contrasts between these approaches.
    #    1-Residual resampling assumes fixed X values and independent and identically distributed residuals (although the residuals are not assumed to be normally distributed).
    #    2-Data resampling does not assume independent and identically distributed residuals.
    # Here we perform data resampling
    #    Efron and Gong, 1983. "A Leisurely Look at the Bootstrap, the Jacknife, and Cross-Validation," The American Statistician.
    #    Hamilton (1992), "Regression with Graphics: A Second Course in Applied Statistics," Duxbury Press,
    if do_data_resampling:  # global won't work
        if is_global:
            raise Exception(
                "**ERROR** in fit_function global fitting with data resampling not implemented!!\n"
            )
        debug = False
        sample_size = len(ydata)
        if debug:
            print(" bootstrap_runs", bootstrap_cycles)
        if type(bootstrap_cycles) is not int:
            bootstrap_cycles = 10000
        y = numpy.array(ydata).copy()
        x = numpy.array(xdata).copy()
        # if yerr is not None : yerr = numpy.array(yerr).copy()
        choice = numpy.random.random_integers(
            0, sample_size - 1, (bootstrap_cycles, sample_size)
        )
        # Find best fit for each sample. We use a for loop as I don't know how to call leastsquare axis=1
        parameter_ensemble = None
        skipped = 0
        # does not account for yerr
        for j in range(bootstrap_cycles):
            if (
                len(numpy.unique(choice[j])) <= len(pfit_leastsq) + 2
            ):  # skip underdetermined fit
                skipped += 1
                continue
            popt, success = scipy.optimize.leastsq(
                errfunc, pfit_leastsq, args=(x[choice[j]], y[choice[j]]), full_output=0
            )
            # if filter_failures_from_distance_to_guess is not None and any(numpy.abs(popt-parameter_guess)>filter_failures_from_distance_to_guess ) :
            #    skipped+=1
            #    continue
            if parameter_ensemble is None:
                parameter_ensemble = numpy.array(popt)
            else:
                parameter_ensemble = numpy.vstack(
                    (parameter_ensemble, numpy.array(popt))
                )

        if debug:
            print(
                " left with parameter_ensemble.shape:",
                parameter_ensemble.shape,
                "skipped",
                skipped,
                end=" ",
            )
        # if skipped>0 : print '  FitCI filter_failures_from_distance_to_guess=%lf -> skipped %d of %d (%5.2lf %%)' % (filter_failures_from_distance_to_guess,skipped,bootstrap_cycles,100.*skipped/bootstrap_cycles)
        pfit_bootstrap = numpy.mean(parameter_ensemble, axis=0)
        if debug:
            print(
                "pfit_bootstrap.shape",
                pfit_bootstrap.shape,
                "parameter_ensemble[:3]:\n",
                parameter_ensemble[:3],
            )
        Nsigma = 1.0  # 1sigma gets approximately the same as methods above
        # 1sigma corresponds to 68.3% confidence interval
        # 2sigma corresponds to 95.44% confidence interval
        perr_bootstrap = numpy.std(parameter_ensemble, axis=0) * Nsigma
        ci = 0.95  # 95 % CI
        pCI_down, pCI_up = numpy.percentile(
            parameter_ensemble,
            [100.0 * (0.5 - ci / 2.0), 100.0 * (0.5 + ci / 2.0)],
            axis=0,
        )  # default 2.5 97.5 percentiles
        ci_error_bars = [pfit_bootstrap - pCI_down, pCI_up - pfit_bootstrap]
        sum_of_squared_residuals = (
            errfunc(pfit_bootstrap, xdata, ydata, glob_loc_separator, num_local_params)
            ** 2
        ).sum()
        if Ndof > 0:
            s_sq = sum_of_squared_residuals / (Ndof)  # the reduced Chi2
        return (
            pfit_leastsq,
            perr_leastsq,
            pfit_bootstrap,
            perr_bootstrap,
            pCI_down,
            pCI_up,
            ci_error_bars,
            s_sq,
            sum_of_squared_residuals,
            parameter_ensemble,
        )


class FitNew:
    def __init__(
        self,
        function=None,
        parameters_guess=None,
        parameters_boundaries=None,
        dont_fit_plateau=False,
        ci=0.95,
        do_data_resampling=False,
        bootstrap_cycles=1000,
        permute_ci=False,
        add_mean_residual_to_yerr=True,
        local_params_indices=None,
        local_args_for_global=None,
        unbounded_proxy_value=9942515536,
    ):
        """
        function must be function(x,p) where p is a list of parameters to fit
         local_params_indices is for global fitting and specified the indices in parameters_guess that should be treated as local parameters, all others will be global
         in this case when calling ydata must be a list or list or a 2D numpy array
        """
        self.debug = False
        self.parameters_guess = parameters_guess
        self.unbounded_proxy_value = unbounded_proxy_value  # a random int that assigned as boundary to non-bounded parameters - then it is removed by an == operation.
        self.bootstrap_cycles = bootstrap_cycles
        self.ci = ci
        self.permute_ci = permute_ci
        self.do_data_resampling = do_data_resampling
        self.function = function
        self.add_mean_residual_to_yerr = add_mean_residual_to_yerr
        self.parameters_boundaries = self._adjust_p_boundaries(parameters_boundaries)
        self.local_params_indices = local_params_indices
        self.local_args_for_global = local_args_for_global
        self.dont_fit_plateau = dont_fit_plateau
        self.fitted_parameters = None
        self.parameter_ensemble = None  # filled if bootstrap is required
        self.parameters_CI_down, self.parameters_CI_up = None, None

    def _adjust_p_boundaries(self, parameters_boundaries, parameters_guess=None):
        if (
            parameters_boundaries is not None
        ):  # remove possible None given as unbound paramters
            parameters_boundaries = list(parameters_boundaries)
            for j, tup in enumerate(parameters_boundaries):
                if tup is None:
                    parameters_boundaries[j] = (
                        self.unbounded_proxy_value,
                        self.unbounded_proxy_value,
                    )
                else:
                    if tup[0] is None:
                        parameters_boundaries[j] = (
                            self.unbounded_proxy_value,
                            parameters_boundaries[j][1],
                        )
                    elif parameters_guess is not None and tup[0] >= parameters_guess[j]:
                        sys.stderr.write(
                            "\n***ERROR*** in fit given parameters_boundaries but guess of par %d is %g, which is <= of lower boundary %g\n"
                            % (j, parameters_guess[j], tup[0])
                        )
                    if tup[1] is None:
                        parameters_boundaries[j] = (
                            parameters_boundaries[j][0],
                            self.unbounded_proxy_value,
                        )
                    elif parameters_guess is not None and tup[1] <= parameters_guess[j]:
                        sys.stderr.write(
                            "\n***ERROR*** in fit given parameters_boundaries but guess of par %d is %g, which is >= of upper boundary %g\n"
                            % (j, parameters_guess[j], tup[1])
                        )
        return parameters_boundaries

    def __call__(
        self,
        xdata,
        ydata,
        yerr=None,
        xerr=None,
        parameters_guess=None,
        dont_fit_plateau=None,
        parameters_boundaries=None,
        local_params_indices=None,
        local_args_for_global=None,
        function=None,
    ):
        """
        if given as None parameters_guess, p_boundaries, and function are read from self
         if bootstrap is not required it returns pfit_leastsq,perr_leastsq
         otherwise return pfit_leastsq,perr_leastsq, pfit_bootstrap, perr_bootstrap,pCI_down,pCI_up,ci_error_bars
         local_params_indices is for global fitting and specified the indices in parameters_guess that should be treated as local parameters, all others will be global
        """
        if dont_fit_plateau is None:
            dont_fit_plateau = self.dont_fit_plateau
        else:
            self.dont_fit_plateau = dont_fit_plateau
        ydata = numpy.array(ydata)
        if (
            len(ydata.shape) > 1 and ydata.shape[0] > 1 and ydata.shape[1] > 1
        ):  # global fit
            if yerr is not None and any([a is None for a in yerr]):
                yerr = None  # either all given or None given
            if xerr is not None and any([a is None for a in xerr]):
                xerr = None  # either all given or None given
        self.xdata, self.ydata, self.yerr, self.xerr = xdata, ydata, yerr, xerr
        if function is None:
            function = self.function
        if parameters_boundaries is None:
            parameters_boundaries = self.parameters_boundaries
        else:
            parameters_boundaries = self._adjust_p_boundaries(
                parameters_boundaries, parameters_guess=parameters_guess
            )
        if parameters_guess is None:
            parameters_guess = self.parameters_guess
        else:
            self.parameters_guess = parameters_guess
        if local_params_indices is None:
            local_params_indices = self.local_params_indices
        else:
            self.local_params_indices = local_params_indices
        if local_args_for_global is None:
            local_args_for_global = self.local_args_for_global
        else:
            self.local_args_for_global = local_args_for_global
        if self.ci > 1 and self.ci < 100:
            self.ci /= 100.0  # put in 0,1 assuming it has been given as a percent.
        if dont_fit_plateau:
            self.ydata_no_plateau, self.xdata_no_plateau = self.remove_plateaus(
                ydata,
                xdata,
                only_at_extrema=True,
                smooth_per_side=1,
                relative_gradient_smaller_than=0.03,
            )
            out = fit_function(
                function,
                self.xdata_no_plateau,
                self.ydata_no_plateau,
                parameters_guess,
                yerr=yerr,
                xerr=xerr,
                local_params_indices=local_params_indices,
                local_args_for_global=local_args_for_global,
                p_boundaries=parameters_boundaries,
                unbounded_proxy_value=self.unbounded_proxy_value,
                add_mean_residual_to_yerr=self.add_mean_residual_to_yerr,
                bootstrap_cycles=self.bootstrap_cycles,
                do_data_resampling=self.do_data_resampling,
                ci=self.ci,
            )
        else:
            out = fit_function(
                function,
                xdata,
                ydata,
                parameters_guess,
                yerr=yerr,
                xerr=xerr,
                local_params_indices=local_params_indices,
                local_args_for_global=local_args_for_global,
                p_boundaries=parameters_boundaries,
                unbounded_proxy_value=self.unbounded_proxy_value,
                add_mean_residual_to_yerr=self.add_mean_residual_to_yerr,
                bootstrap_cycles=self.bootstrap_cycles,
                do_data_resampling=self.do_data_resampling,
                ci=self.ci,
            )
        if self.bootstrap_cycles is not None and self.bootstrap_cycles > 0:
            (
                pfit_leastsq,
                perr_leastsq,
                pfit_bootstrap,
                perr_bootstrap,
                pCI_down,
                pCI_up,
                ci_error_bars,
                chi2red,
                sum_of_squared_residuals,
                self.parameter_ensemble,
            ) = out
            out = out[:-1]  # don't return parameter_ensemble
            self.fitted_parameters = pfit_bootstrap
            self.parameters_CI_down, self.parameters_CI_up = pCI_down, pCI_up
            self.chi2red = chi2red
            self.sum_of_squared_residuals = sum_of_squared_residuals
        else:
            self.fitted_parameters = out[0]
            self.parameters_CI_down, self.parameters_CI_up = (
                self.fitted_parameters - 2 * out[1],
                self.fitted_parameters + 2 * out[1],
            )  # 2 sigma away
            self.chi2red = out[-2]
            self.sum_of_squared_residuals = out[-1]
        return out

    def remove_plateaus(
        self,
        ydata,
        xdata,
        only_at_extrema=True,
        smooth_per_side=1,
        relative_gradient_smaller_than=0.03,
    ):
        """
        can be used to remove flat regions like tails of gaussians or plateau of sigmoids that may bias the leastsquare
        only_at_extrema allows plateaus to be present only at beginning or end of data series, not in middle
        """
        if hasattr(ydata[0], "__len__"):  # may be a global fit - call multiple times
            y_left = []
            if hasattr(xdata[0], "__len__"):  # multiple x axis
                x_left = []
                for j, prof in enumerate(ydata):
                    yy, xx = self.remove_plateaus(
                        prof,
                        xdata[j],
                        only_at_extrema=only_at_extrema,
                        smooth_per_side=smooth_per_side,
                        relative_gradient_smaller_than=relative_gradient_smaller_than,
                    )
                    y_left += [yy]
                    x_left += [xx]
                return y_left, x_left
            else:
                keeps = []
                for j, prof in enumerate(ydata):
                    yy, kk = self.remove_plateaus(
                        prof,
                        xdata=None,
                        only_at_extrema=only_at_extrema,
                        smooth_per_side=smooth_per_side,
                        relative_gradient_smaller_than=relative_gradient_smaller_than,
                    )
                    y_left += [yy]
                    keeps += [kk]
                if checkAllEqual(keeps):
                    return y_left, numpy.array(xdata[keeps[0]])
                else:
                    return (
                        y_left,
                        [numpy.array(xdata)[ke] for ke in keeps],
                    )  # return multiple x-axis
        # 1D array
        if smooth_per_side is not None:
            y = numpy.array(
                smooth_profile(
                    list(ydata),
                    smooth_per_side=smooth_per_side,
                    use_savgol_filter=False,
                )
            )  # if you want to replace first: [ypoints[smooth_per_side]]*smooth_per_side) + list(ypoints)[smooth_per_side:]
        else:
            y = numpy.array(ydata)
        y1 = numpy.gradient(y)
        candidate_flat_zone_index = numpy.where(
            (numpy.abs(y1) / numpy.abs(y1).max()) <= relative_gradient_smaller_than
        )[
            0
        ]  # only part smaller than
        # print 'candidate_flat_zone_index=',candidate_flat_zone_index
        # print 'candidate_flat_zone_index[0]=',candidate_flat_zone_index[0]
        # print'y',y
        # print 'y1',y1
        flat_zone_index = []
        if only_at_extrema:
            for j, e in enumerate(candidate_flat_zone_index):
                if j == e:
                    flat_zone_index += [e]
                else:
                    break
            n = len(y) - 1
            j = -1
            while (
                abs(j) <= len(candidate_flat_zone_index)
                and n == candidate_flat_zone_index[j]
            ):
                flat_zone_index += [candidate_flat_zone_index[j]]
                j -= 1
                n -= 1
        else:  # get index that are consecutive (at leat two consecutives)
            for j, e in enumerate(candidate_flat_zone_index):
                if j > 0 and e == candidate_flat_zone_index[j - 1] + 1:
                    if candidate_flat_zone_index[j - 1] == flat_zone_index[-1]:
                        flat_zone_index += [e]  # added at previous step
                    else:
                        flat_zone_index += [candidate_flat_zone_index[j - 1], e]
        keep = numpy.array([j for j in range(len(ydata)) if j not in flat_zone_index])
        # print 'keep=',keep
        if xdata is None:
            return ydata[keep], keep
        return ydata[keep], numpy.array(xdata)[keep]

    def get_plot_x_y(
        self, all_x, popt=None, num_points=1000, fit_allow_extra_fraction=0.0
    ):
        if popt is None:
            popt = self.fitted_parameters
        if self.dont_fit_plateau is True:
            mx, Mx = get_min_max_glob(self.xdata_no_plateau)
        else:
            mx, Mx = get_min_max_glob(all_x)
        if (
            hasattr(fit_allow_extra_fraction, "__len__")
            and len(fit_allow_extra_fraction) == 2
        ):  # fit_allow_extra_fraction is x_range of fit
            x_s = numpy.linspace(
                fit_allow_extra_fraction[0], fit_allow_extra_fraction[1], num_points
            )
        else:
            a = fit_allow_extra_fraction * (Mx - mx)
            x_s = numpy.linspace(mx - a, Mx + a, num_points)
        # print "DEBB popt",popt,'mx,Mx,a',mx,Mx,a
        if (
            (
                len(self.ydata.shape) > 1
                and self.ydata.shape[0] > 1
                and self.ydata.shape[1] > 1
            )
            or self.local_args_for_global is not None
            or self.local_params_indices is not None
        ):  # it is a global fit
            if self.local_params_indices is None:
                nglobal = len(popt)
            else:
                #                                                      (       nglobal                                           )== estimated nlocal across all profiles
                if len(popt) > len(self.parameters_guess) and len(popt) - (
                    len(self.parameters_guess) - len(self.local_params_indices)
                ) == len(self.local_params_indices) * len(self.ydata):
                    self.local_params_indices = numpy.arange(
                        min(self.local_params_indices), len(popt)
                    )  # p_guess= numpy.hstack((p_guess,local_params_guess.copy())) # no for loop has we already have a set of guesses per profile.
                nglobal = len(popt) - len(self.local_params_indices)
                # if nglobal<=0 :
                #    nglobal=len(popt) - len(self.local_params_indices)
                nlocal = len(self.local_params_indices) // len(self.ydata)
                if len(self.local_params_indices) % len(self.ydata) != 0:
                    sys.stderr.write(
                        "\n**ERROR** in fitter get_plot_x_y() for global fit with local params len(self.local_params_indices)%%len(self.ydata)!=0  lens=%d  %d nglobal=%d\n"
                        % (len(self.local_params_indices), len(self.ydata), nglobal)
                    )
            if self.local_args_for_global is not None:
                nargs = len(self.local_args_for_global) // self.ydata.shape[0]
                if self.local_params_indices is None:
                    y_s = numpy.array(
                        [
                            self.function(
                                x_s,
                                list(popt)[:nglobal],
                                *self.local_args_for_global[j * nargs : (j + 1) * nargs]
                            )
                            for j in range(self.ydata.shape[0])
                        ]
                    )
                else:
                    y_s = numpy.array(
                        [
                            self.function(
                                x_s,
                                list(popt)[:nglobal]
                                + list(popt[nglobal:])[j * nlocal : (j + 1) * nlocal],
                                *self.local_args_for_global[j * nargs : (j + 1) * nargs]
                            )
                            for j in range(self.ydata.shape[0])
                        ]
                    )
            else:
                y_s = numpy.array(
                    [
                        self.function(
                            x_s,
                            list(popt)[:nglobal]
                            + list(popt[nglobal:])[j * nlocal : (j + 1) * nlocal],
                        )
                        for j in range(self.ydata.shape[0])
                    ]
                )
        else:
            y_s = self.function(x_s, popt)
        return x_s, y_s

    def get_plotCI_x_y(
        self, all_x, num_points=1000, fit_allow_extra_fraction=0.0, ci=None
    ):
        # returns top and bottom ci lines
        if ci is None:
            ci = self.ci
        if ci > 1 and ci < 100:
            ci /= 100.0  # put in 0,1 assuming it has been given as a percent.
        if self.parameter_ensemble is None:
            sys.stderr.write(
                "**WARNING** while fitting in get_plotCI_x_y parameter_ensemble (from bootstrap) not set, using parameter CIs to generate CI\n"
            )
            if self.parameters_CI_down is None:
                raise Exception("self.parameters_CI_down is None - cannot proceed\n")
            x_s, y_s_lower = self.get_plot_x_y(all_x, self.parameters_CI_down)
            x_s, y_s_upper = self.get_plot_x_y(all_x, self.parameters_CI_up)
            return x_s, y_s_lower, y_s_upper

        if (
            hasattr(fit_allow_extra_fraction, "__len__")
            and len(fit_allow_extra_fraction) == 2
        ):  # fit_allow_extra_fraction is x_range of fit
            x_s = numpy.linspace(
                fit_allow_extra_fraction[0], fit_allow_extra_fraction[1], num_points
            )
        else:
            mx, Mx = get_min_max_glob(all_x)
            a = fit_allow_extra_fraction * (Mx - mx)
            x_s = numpy.linspace(mx - a, Mx + a, num_points)
        is_global = False
        if (
            (
                len(self.ydata.shape) > 1
                and self.ydata.shape[0] > 1
                and self.ydata.shape[1] > 1
            )
            or self.local_args_for_global is not None
            or self.local_params_indices is not None
        ):  # it is a global fit
            is_global = True
            if self.local_params_indices is None:
                nglobal = len(self.parameter_ensemble[0])
            else:
                print(
                    "DEB self.local_params_indices:",
                    self.local_params_indices,
                    "self.parameter_ensemble[0]:",
                    self.parameter_ensemble[0],
                    len(self.parameter_ensemble[0]),
                )
                nglobal = len(self.parameter_ensemble[0]) - len(
                    self.local_params_indices
                )  # * len(self.ydata) # will have added len(local parameters) for each fit
                nlocal = len(self.local_params_indices) // len(self.ydata)
                if len(self.local_params_indices) % len(self.ydata) != 0:
                    sys.stderr.write(
                        "\n**ERROR** in fitter get_plotCI_x_y() for global fit with local params len(self.local_params_indices)%%len(self.ydata)!=0  lens=%d %d nglobal=%d\n"
                        % (len(self.local_params_indices), len(self.ydata), nglobal)
                    )
                # if nglobal<=0 :   nglobal=len(self.parameter_ensemble[0]) - len(self.local_params_indices)
            if self.local_args_for_global is not None:
                nargs = len(self.local_args_for_global) // self.ydata.shape[0]
        y_s = []
        for popt in self.parameter_ensemble:  # evaluate for each set in esemble
            if is_global:
                if self.local_args_for_global is not None:
                    if self.local_params_indices is None:
                        y_s += [
                            numpy.array(
                                [
                                    self.function(
                                        x_s,
                                        list(popt)[:nglobal],
                                        *self.local_args_for_global[
                                            j * nargs : (j + 1) * nargs
                                        ]
                                    )
                                    for j in range(self.ydata.shape[0])
                                ]
                            )
                        ]
                    else:
                        y_s += [
                            numpy.array(
                                [
                                    self.function(
                                        x_s,
                                        list(popt)[:nglobal]
                                        + list(popt[nglobal:])[
                                            j * nlocal : (j + 1) * nlocal
                                        ],
                                        *self.local_args_for_global[
                                            j * nargs : (j + 1) * nargs
                                        ]
                                    )
                                    for j in range(self.ydata.shape[0])
                                ]
                            )
                        ]
                else:
                    y_s += [
                        numpy.array(
                            [
                                self.function(
                                    x_s,
                                    list(popt)[:nglobal]
                                    + list(popt[nglobal:])[
                                        j * nlocal : (j + 1) * nlocal
                                    ],
                                )
                                for j in range(self.ydata.shape[0])
                            ]
                        )
                    ]
            else:
                y_s += [self.function(x_s, popt)]
        y_s = numpy.array(y_s)
        if self.debug:
            print(
                "self.parameter_ensemble.shape",
                self.parameter_ensemble.shape,
                "mean=",
                numpy.nanmean(self.parameter_ensemble, axis=0),
                "stdev=",
                numpy.nanstd(self.parameter_ensemble, axis=0),
            )
            # print "len(y_s)",len(y_s)
            # print numpy.nanmax(y_s,axis=0)-numpy.nanmin(y_s,axis=0)
        y_s_lower, y_s_upper = numpy.nanpercentile(
            y_s, [100.0 * (0.5 - ci / 2.0), 100.0 * (0.5 + ci / 2.0)], axis=0
        )
        return x_s, y_s_lower, y_s_upper

    def plot(
        self,
        x=None,
        y=None,
        popt=None,
        popt_dw=None,
        popt_up=None,
        yerr=None,
        xerr=None,
        plot_CI=True,
        permute_ci=None,
        **kwargs
    ):
        if x is None:
            x = self.xdata
        if y is None:
            y = self.ydata
        if yerr is None:
            yerr = self.yerr
        if xerr is None:
            xerr = self.xerr
        if popt is None:
            popt = self.fitted_parameters
        if permute_ci is None:
            permute_ci = self.permute_ci
        if plot_CI:
            if popt_dw is None:
                popt_dw = self.parameters_CI_down
            if popt_up is None:
                popt_up = self.parameters_CI_up
            if permute_ci:
                if len(popt) != 2:
                    sys.stderr.write(
                        "Error in FitCI() permute_ci is implemented for 2 parameters only! (leaving unchanged)"
                    )
                    sys.stderr.flush()
                else:
                    tmp_dw = [popt_dw[0], popt_up[1]]  # swap them
                    popt_up = numpy.array([popt_up[0], popt_dw[1]])
                    popt_dw = numpy.array(tmp_dw)

        x_pl, y_pl = self.get_plot_x_y(x, popt)
        y_pl_dw, y_pl_up = None, None
        if plot_CI:
            _, y_pl_dw = self.get_plot_x_y(x, popt_dw)
            _, y_pl_up = self.get_plot_x_y(x, popt_up)
        figure = None
        # if not just_return_plot_lines :
        zorder_ref = 10
        fitls = "-"
        if "markersize" not in kwargs:
            if yerr is None and xerr is None:
                kwargs["markersize"] = 15
            else:
                kwargs["markersize"] = 10
        if "marker" not in kwargs:
            kwargs["marker"] = "."
        if "ls" in kwargs:
            fitls = kwargs["ls"]
            del kwargs["ls"]
        if "zorder" in kwargs:
            zorder_ref = kwargs["zorder"]
            del kwargs["zorder"]
        # first plot data points
        figure = plotter.profile(
            y, x, yerr=yerr, xerr=xerr, ls="", zorder=zorder_ref, **kwargs
        )
        if "fit_linecolor" in kwargs:
            kwargs["color"] = kwargs["fit_linecolor"]
        if "color" not in kwargs:
            kwargs["color"] = "red"
        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 3
        # delete those that we don't apply to CI lines:
        if "marker" in kwargs:
            del kwargs["marker"]
        if "figure" in kwargs:
            del kwargs["figure"]
        if "ls" not in kwargs:
            kwargs["ls"] = fitls
        # plot fit
        # print kwargs
        figure = plotter.profile(
            y_pl, x_pl, marker="", zorder=zorder_ref - 1, figure=figure, **kwargs
        )
        if plot_CI:
            # plot line for CI
            if "ls" in kwargs:
                del kwargs["ls"]
            figure = plotter.profile(
                y_pl_dw,
                x_pl,
                ls="--",
                zorder=zorder_ref - 1,
                marker="",
                figure=figure,
                **kwargs
            )
            figure = plotter.profile(
                y_pl_up,
                x_pl,
                ls="--",
                zorder=zorder_ref - 1,
                marker="",
                figure=figure,
                **kwargs
            )
        return figure


def curve_fit(
    function,
    x,
    y,
    p0=None,
    sigma=None,
    p_boundaries=None,
    p_are_individual_argv=False,
    maxfev=None,
    full_output=True,
    use_small_ydata_correction=True,
):
    """
    function(x, parms) wher parms is a list of the various parameters unless p_are_individual_argv is True
    replace scipy.optimize.curve_fit by explicitly calling leastsquare and thorugh a custom cost funciton that handles better (safer) errors and
     correct for small ydata  if use_small_ydata_correction is given.
    """
    if p_are_individual_argv:

        def mf(x, args):
            return function(x, *args)

        inpfun = mf
    else:
        inpfun = function
    leastsq_fitter = Leastsq(
        function=inpfun,
        parameters_guess=p0,
        parameters_boundaries=p_boundaries,
        use_small_ydata_correction=use_small_ydata_correction,
        n_iterations=maxfev,
        full_output=full_output,
    )
    return leastsq_fitter(x, y, yerr=sigma)


class Leastsq:
    def __init__(
        self,
        function=None,
        parameters_guess=None,
        parameters_boundaries=None,
        n_iterations=None,
        leastsq_tolerance=1.49012e-08,
        use_small_ydata_correction=True,
        full_output=True,
    ):
        self.function = function
        self.parameters_guess = parameters_guess
        self.parameters_boundaries = parameters_boundaries
        self.n_iterations = n_iterations
        self.leastsq_tolerance = leastsq_tolerance
        self.use_small_ydata_correction = use_small_ydata_correction
        self.full_output = full_output

    def __call__(
        self,
        xdata,
        ydata,
        yerr=None,
        parameters_guess=None,
        parameters_boundaries=None,
        function=None,
    ):
        """
        if given as None parameters_guess, p_boundaries, and function are read from self
        """
        if function is None:
            function = self.function
        if parameters_boundaries is None:
            parameters_boundaries = self.parameters_boundaries
        if parameters_guess is None:
            parameters_guess = self.parameters_guess
        return self.get_fitted_parameters(
            function,
            xdata,
            ydata,
            parameters_guess,
            yerr=yerr,
            parameters_boundaries=parameters_boundaries,
            leastsq_tolerance=self.leastsq_tolerance,
            n_iterations=self.n_iterations,
            use_ugly_correction_for_small_obs=self.use_small_ydata_correction,
        )

    def get_fitted_parameters(
        self,
        function,
        xdata,
        ydata,
        parameters_guess,
        yerr=None,
        parameters_boundaries=None,
        leastsq_tolerance=1.49012e-08,
        n_iterations=None,
        use_ugly_correction_for_small_obs=True,
    ):
        if n_iterations is None or n_iterations < 0:
            n_iterations = int(2 * 200 * (len(parameters_guess) + 1))
        if (
            n_iterations > 1e5
        ):  # leasquare is fast # if it is zero it should do 200*(len(parameters)+1) iterations..
            print(
                "\nAbout to perform %d iterations, this may take some time.."
                % (n_iterations)
            )
        corrected_function = function
        if (
            use_ugly_correction_for_small_obs
        ):  # this correction should hepl yielding better chi2 values, that's all.
            nel = 1
            for s in ydata.shape:
                nel *= s
            # if all(all_y >= -0.2) and (y<=5.).sum()/float(nel) >0.1 : # if all y are positives (or nearly) and 10% or more of them are smaller than 5
            if any(abs(ydata) < 0.5):  # if some points are close to zero
                ydata = numpy.array(ydata) + 5.0

                def corrected_function(xv, parms):
                    return 5.0 + function(xv, parms)

                # print ' Leastsq(): using UGLY correction for small observed numbers. [namely applying ydata+5]'
        if yerr is not None:
            yerr = numpy.array(yerr)
            safe_yerr_den = numpy.mean(yerr)
        else:
            safe_yerr_den = 0.0
        if self.full_output:
            popt, pcov, infodict, message, success = scipy.optimize.leastsq(
                self.cost_function,
                parameters_guess,
                args=(
                    xdata,
                    ydata,
                    corrected_function,
                    yerr,
                    parameters_boundaries,
                    safe_yerr_den,
                ),
                full_output=1,
                maxfev=n_iterations,
                ftol=leastsq_tolerance,
                xtol=leastsq_tolerance,
            )
            print("success:", success)
            print(message)
            return popt, pcov, infodict, message, success
        else:
            return scipy.optimize.leastsq(
                self.cost_function,
                parameters_guess,
                args=(
                    xdata,
                    ydata,
                    corrected_function,
                    yerr,
                    parameters_boundaries,
                    safe_yerr_den,
                ),
                full_output=0,
                maxfev=n_iterations,
                ftol=leastsq_tolerance,
                xtol=leastsq_tolerance,
            )

    def parameters_box_penalty(
        self, par, lower_bound=None, upper_bound=None, cost_penalty=1e6
    ):
        """
        this is used to give a penalty if parameters go outside a range. if no boundaries or if all parameters respect boundaries return 0
        """
        if isinstance(par, numpy.ndarray):
            if lower_bound is not None and any(par < lower_bound):
                return cost_penalty
            if upper_bound is not None and any(par > upper_bound):
                return cost_penalty
        else:
            if lower_bound is not None and par < lower_bound:
                return cost_penalty
            if upper_bound is not None and par > upper_bound:
                return cost_penalty
        return 0.0

    def cost_function(
        self, p, x, y, function, yerr=None, p_boundaries=None, safe_yerr_den=0
    ):
        """
        at this stage p_boundaries, if given, should be a list of tuple (one tuple per parameter)
         with (lower_bound,higher_bound), if one do not wishes to give a specific boundary None can be given (None, higher_bound).
        yerr must be a scalar or a numpy array, in the latter case it must have the same shape of y.
        
        safe_yerr_den should be numpy.mean(yerr) to handle erros that contain zeros or that span multiple order of magnitudes.
        """
        y_fit = function(x, p)  # x at this stage should be a numpy array
        if p_boundaries is not None:
            penalty = 0
            for j, tup in enumerate(p_boundaries):
                penalty += self.parameters_box_penalty(p[j], tup[0], tup[1])
                if penalty > 0:
                    y_fit += penalty
        if yerr is not None:
            # print y.shape, yerr.shape
            return (y - y_fit) / (safe_yerr_den + yerr ** 2)
        return y - y_fit


class FitCI:
    def __init__(
        self,
        function=None,
        function_has_parameter_list=True,
        ci=0.95,
        bootstrap_cycles=True,
        permute_ci=False,
        fit_engine=curve_fit,
    ):
        """
        it is possible that the function in this context should be defined as
        def func(x, a, b, c) (with a,b,c parameters) 
        rather than def func(x,p) with p a list of parameters
        function_has_parameter_list must be set to true if your function is written like
         f(x, params) where params is a list/numpy.array like [parameter[0], parameter[1], ... ]
        otherwise set to False if 
         f(x,p0, p1, p2, ...)
        """
        if fit_engine is None:
            self.fit_engine = scipy.optimize.curve_fit
        else:
            self.fit_engine = fit_engine
        if fit_engine is None and function_has_parameter_list:

            def mf(x, *args):
                return function(x, args)

            self.function = mf
        else:
            self.function = function
        self.bootstrap_cycles = bootstrap_cycles
        self.ci = ci
        self.permute_ci = permute_ci

    def __call__(
        self,
        x,
        y,
        parameter_guess=None,
        yerr=None,
        ci=None,
        plot_with_bootstrap_medians=True,
        print_progress=False,
        bootstrap_cycles=None,
        ci_bounds=None,
        permute_ci=None,
        return_plot_lines=False,
        plot=False,
        debug=False,
        **kwargs
    ):
        """
        ci defines the confidence interval one requires.
         **kwargs are used only if plot=True and represent the arguments for plot.profile
         
        """
        if not isinstance(x, numpy.ndarray):
            x = numpy.array(x)
        if not isinstance(y, numpy.ndarray):
            y = numpy.array(y)
        if (
            yerr is not None
            and hasattr(yerr, "__len__")
            and not isinstance(yerr, numpy.ndarray)
        ):
            yerr = numpy.array(yerr)
            yerr[yerr <= 1e-15] = min(
                yerr[yerr > 1e-15]
            )  # random replacement for zeros!!
        if debug:
            print(yerr)
        # yerr=None
        if self.function is None:
            raise Exception(
                "Error self.function is None, you should give to the class a function in the form f(x,parameters) that you want to fit!!\n"
            )
        if permute_ci is None:
            permute_ci = self.permute_ci
        if ci is None:
            ci = self.ci
        if bootstrap_cycles is None:
            bootstrap_cycles = self.bootstrap_cycles
        if type(bootstrap_cycles) is not int:
            if type(bootstrap_cycles) is float:
                bootstrap_cycles = int(bootstrap_cycles)
            else:
                bootstrap_cycles = 10000
        # Find best fit. -> best guess of parameters
        popt_leastsq_guess, pcov, infodict, errmsg, ier = self.fit_engine(
            self.function, x, y, sigma=yerr, p0=parameter_guess, full_output=True
        )
        perrors_leastsq = None
        if pcov is not None:
            perrors_leastsq = numpy.sqrt(
                numpy.diagonal(pcov)
            )  # *RMS_of_residuals MAYBE I SHOUD?? Disagreements online on whether this should be multiplied or not..
        fit_results_string = 30 * "*" + "BOOTSTRAP FIT RESULTS " + 30 * "*" + "\n"
        fit_results_string += "   (least square)  popt: %s\n" % (
            ", ".join(map(repr, popt_leastsq_guess))
        )
        if pcov is not None:
            fit_results_string += " (least square) perrors: %s\n" % (
                ", ".join(map(repr, perrors_leastsq))
            )
            fit_results_string += "                   pcov: %s\n" % (
                ", ".join(map(repr, pcov))
            )
        if debug or print_progress:
            if debug:
                print("      ier: ", ier)
                print("   errmsg: ", errmsg)
                print(" infodict: ", infodict)
            print(" (leastsq): popt:", popt_leastsq_guess)
            print("            pcov:", pcov)
        if bootstrap_cycles > 0:
            print(
                "  bootstrap_fit(): doing %d bootstrap cycles..." % (bootstrap_cycles)
            )
            popt_median, perr, popt_dw, popt_up = self.bootstrap_fit(
                x, y, parameter_guess, bootstrap_runs=bootstrap_cycles, ci=ci
            )
            if plot_with_bootstrap_medians:
                popt = popt_median
                if print_progress:
                    print("bootstrap popt_median", popt_median, "Using these to plot!")
            elif print_progress:
                print("bootstrap popt_median", popt_median)
                popt = popt_leastsq_guess
            else:
                popt = popt_leastsq_guess
            fit_results_string += (
                " Performed %d bootstrap cycles:popt_median\n   ->  popt_median = %s\n"
                % (bootstrap_cycles, ", ".join(map(repr, popt_median)))
            )
        else:
            popt_dw, popt_up, perr = self.get_CI_from_cov_matrix(
                popt_leastsq_guess, pcov, ci=ci, ci_bounds=ci_bounds
            )
        fit_results_string += "   ->popt_CI_dwn = %s\n" % (
            ", ".join(map(repr, popt_dw))
        )
        fit_results_string += "   -> popt_CI_up = %s\n" % (
            ", ".join(map(repr, popt_up))
        )
        fit_results_string += "   ->       perr = %s\n" % (", ".join(map(repr, perr)))
        fit_results_string += 30 * "*" + "********************" + 30 * "*" + "\n"
        if print_progress:
            print("  popt_CI_dw:", popt_dw)
            print("  popt_CI_up:", popt_up)
            print("  perr:", perr)
        if permute_ci:
            if len(popt) != 2:
                sys.stderr.write(
                    "Error in FitCI() permute_ci is implemented for 2 parameters only! (leaving unchanged)"
                )
                sys.stderr.flush()
            else:
                if print_progress:
                    print("FitCI() permute_ci selected")
                tmp_dw = [popt_dw[0], popt_up[1]]
                popt_up = numpy.array([popt_up[0], popt_dw[1]])
                popt_dw = numpy.array(tmp_dw)
        self.fitted_parameters = popt
        if return_plot_lines or plot:
            x_pl, y_pl = self.get_plot_x_y(x, popt)
            _, y_pl_dw = self.get_plot_x_y(x, popt_dw)
            _, y_pl_up = self.get_plot_x_y(x, popt_up)
            figure = None
            if plot:
                zorder_ref = 10
                fitls = "-"
                if "markersize" not in kwargs:
                    if yerr is None:
                        kwargs["markersize"] = 15
                    else:
                        kwargs["markersize"] = 10
                if "marker" not in kwargs:
                    kwargs["marker"] = "."
                if "ls" in kwargs:
                    fitls = kwargs["ls"]
                    del kwargs["ls"]
                if "zorder" in kwargs:
                    zorder_ref = kwargs["zorder"]
                    del kwargs["zorder"]
                # first plot data points
                figure = plotter.profile(
                    y, x, yerr=yerr, ls="", zorder=zorder_ref, **kwargs
                )
                if "fit_linecolor" in kwargs:
                    kwargs["color"] = kwargs["fit_linecolor"]
                if "color" not in kwargs:
                    kwargs["color"] = "red"
                if "linewidth" not in kwargs:
                    kwargs["linewidth"] = 3
                if "marker" in kwargs:
                    del kwargs["marker"]
                if "figure" in kwargs:
                    del kwargs["figure"]
                if "ls" not in kwargs:
                    kwargs["ls"] = fitls
                # plot fit
                # print kwargs
                figure = plotter.profile(
                    y_pl,
                    x_pl,
                    marker="",
                    zorder=zorder_ref - 1,
                    figure=figure,
                    **kwargs
                )
                # plot line for CI
                if "ls" in kwargs:
                    del kwargs["ls"]
                figure = plotter.profile(
                    y_pl_dw,
                    x_pl,
                    ls="--",
                    zorder=zorder_ref - 1,
                    marker="",
                    figure=figure,
                    **kwargs
                )
                figure = plotter.profile(
                    y_pl_up,
                    x_pl,
                    ls="--",
                    zorder=zorder_ref - 1,
                    marker="",
                    figure=figure,
                    **kwargs
                )
            return (
                popt_leastsq_guess,
                popt_median,
                perr,
                popt_dw,
                popt_up,
                x_pl,
                y_pl,
                y_pl_dw,
                y_pl_up,
                figure,
                fit_results_string,
            )
        return (
            popt_leastsq_guess,
            perrors_leastsq,
            popt_median,
            perr,
            popt_dw,
            popt_up,
            fit_results_string,
        )
        # Plot data and best fit curve.
        # scatter(x, y)
        # plot(x, func(x, *popt), c='g', lw=2.)
        # plot(x, func(x, *popt_up), c='r', lw=2.)
        # plot(x, func(x, *popt_dw), c='r', lw=2.)
        # text(12, 0.5, '{}% confidence interval'.format(ci * 100.))

    def bootstrap_parameter_ensemble(
        self,
        x,
        y,
        yerr=None,
        parameter_guess=None,
        bootstrap_runs=None,
        filter_failures_from_distance_to_guess=50,
        debug=False,
    ):
        """
        There are two approaches to bootstrapping for fitting.

            1- In the first approach, the OLS fit is computed from the original data. The residuals are then resampled. The residuals are then added to the predicted values of the original fit to obtain a new Y vector. This new Y vector is then fit against the original X variables. We call this approach residual resampling (or the Efron approach).
            2- In the second approach, rows of the original data (both the Y vector and the corresponding rows of the X variables) are resampled. The resampled data are then fit. We call this approach data resampling (or the Wu approach).
            
            Hamilton (see Reference below) gives some guidance on the contrasts between these approaches.
            1-Residual resampling assumes fixed X values and independent and identically distributed residuals (although the residuals are not assumed to be normally distributed).
            2-Data resampling does not assume independent and identically distributed residuals.
        
        This function performs 2 only
        
        Efron and Gong, 1983. "A Leisurely Look at the Bootstrap, the Jacknife, and Cross-Validation," The American Statistician.
        Hamilton (1992), "Regression with Graphics: A Second Course in Applied Statistics," Duxbury Press,
        """
        sample_size = len(y)
        if debug:
            print(" bootstrap_runs", bootstrap_runs)
        if type(bootstrap_runs) is not int:
            bootstrap_runs = 10000
        y = numpy.array(y)
        x = numpy.array(x)
        if yerr is not None:
            yerr = numpy.array(yerr)
        choice = numpy.random.random_integers(
            0, sample_size - 1, (bootstrap_runs, sample_size)
        )
        # Find best fit for each sample. We use a for loop as I don't know how to call leastsquare axis=1
        parameter_ensemble = None
        skipped = 0
        if yerr is None:
            for j in range(bootstrap_runs):
                if (
                    len(numpy.unique(choice[j])) <= len(parameter_guess) + 1
                ):  # skip underdetermined fit
                    continue
                popt, _ = self.fit_engine(
                    self.function,
                    x[choice[j]],
                    y[choice[j]],
                    p0=parameter_guess,
                    full_output=False,
                )
                if filter_failures_from_distance_to_guess is not None and any(
                    numpy.abs(popt - parameter_guess)
                    > filter_failures_from_distance_to_guess
                ):
                    skipped += 1
                    continue
                if parameter_ensemble is None:
                    parameter_ensemble = numpy.array(popt)
                else:
                    parameter_ensemble = numpy.vstack(
                        (parameter_ensemble, numpy.array(popt))
                    )
        else:
            for j in range(bootstrap_runs):
                if (
                    len(numpy.unique(choice[j])) <= len(parameter_guess) + 1
                ):  # skip underdetermined fit
                    continue
                popt, _ = self.fit_engine(
                    self.function,
                    x[choice[j]],
                    y[choice[j]],
                    sigma=yerr[choice[j]],
                    p0=parameter_guess,
                    full_output=False,
                )
                if filter_failures_from_distance_to_guess is not None and any(
                    numpy.abs(popt - parameter_guess)
                    > filter_failures_from_distance_to_guess
                ):
                    skipped += 1
                    continue
                if parameter_ensemble is None:
                    parameter_ensemble = numpy.array(popt)
                else:
                    parameter_ensemble = numpy.vstack(
                        (parameter_ensemble, numpy.array(popt))
                    )
        if debug:
            print(" left with parameter_ensemble.shape:", parameter_ensemble.shape)
        if skipped > 0:
            print(
                "  FitCI filter_failures_from_distance_to_guess=%lf -> skipped %d of %d (%5.2lf %%)"
                % (
                    filter_failures_from_distance_to_guess,
                    skipped,
                    bootstrap_runs,
                    100.0 * skipped / bootstrap_runs,
                )
            )
        return parameter_ensemble

    def bootstrap_fit(self, x, y, parameter_guess=None, bootstrap_runs=None, ci=0.95):
        parameter_ensemble = self.bootstrap_parameter_ensemble(
            x, y, parameter_guess=parameter_guess, bootstrap_runs=bootstrap_runs
        )
        # popt_mean=numpy.mean(parameter_ensemble,axis=0) # mean is too sensitive to outliers that in some combinations may be inf or similar
        popt_median = numpy.median(parameter_ensemble, axis=0)
        popt_sterr = numpy.std(parameter_ensemble, axis=0)
        # print [100.*0.5-ci/2.,100.*0.5+ci/2.]
        popt_down, popt_up = numpy.percentile(
            parameter_ensemble,
            [100.0 * (0.5 - ci / 2.0), 100.0 * (0.5 + ci / 2.0)],
            axis=0,
        )  # default 2.5 97.5 percentiles
        return popt_median, popt_sterr, popt_down, popt_up

    def get_CI_from_cov_matrix(self, popt, pcov, ci=0.95, ci_bounds=None):
        # Convert to percentile point of the normal distribution.
        # See: https://en.wikipedia.org/wiki/Standard_score
        pp = (1.0 + ci) / 2.0
        # Convert to number of standard deviations.
        nstd = scipy.stats.norm.ppf(pp)
        # print nstd
        # Standard deviation errors on the parameters.
        perr = numpy.sqrt(numpy.diag(pcov))
        # Add nstd standard deviations to parameters to obtain the upper confidence
        # interval.
        popt_up = popt + nstd * perr
        popt_dw = popt - nstd * perr
        print("before ci_bounds")
        print("  popt_up", popt_up)
        print("  popt_dw", popt_dw)
        if ci_bounds is not None:
            for j in range(len(ci_bounds)):
                if ci_bounds[j] is None:
                    continue
                if ci_bounds[j][0] is not None and popt_dw[j] < ci_bounds[j][0]:
                    popt_dw[j] = ci_bounds[j][
                        0
                    ]  # 0 is parameter 0, then 0 is lower bound 1 upper bound
                if ci_bounds[j][1] is not None and popt_up[j] < ci_bounds[j][1]:
                    popt_up[j] = ci_bounds[j][1]

        return popt_dw, popt_up, perr

    def get_plot_x_y(
        self, all_x, popt=None, num_points=1000, fit_allow_extra_fraction=0.0
    ):
        if popt is None:
            popt = self.fitted_parameters
        mx, Mx = min(all_x), max(all_x)
        a = fit_allow_extra_fraction * (Mx - mx)
        x_s = numpy.linspace(mx - a, Mx + a, num_points)
        y_s = self.function(x_s, popt)
        return x_s, y_s


class Fit:  # fits a function from R^n --> R
    """
    KNOWN BUG. yerr is implemented in a way that if given doesn't actually return canonical chi2..!!
    
    function must be f(x, p) where p are the parameters to be estimated, whic should return a real number y
    it should be written so that if mutliple values of x are given as a numpy array the corresponding values of y are returned
     you should give to the class a function in the form f(x,parameters) that you want to fit!!
      if you want to solve a linear system of equation your functon must return x*p, where x is now the matrix of known coefficients and p is the solution you look for.
        Standard nomenclature in this case would be x=A and p=x)
    EXAMPLE:
    
    you have defined your function 
    f(x, p) where p are the parameters that need to be fitted. When x is a np array this function should return y, the np array with the corresponding results
    you also have a list of initial guesses for the parameters 
    guessed_p=[ 1., 3. ] # clearly this should have the same number of paremters as f is expecting
    
    fitting=misc.Fit( f ) # f is your defined function
    parameter_fitted, y_fit,chi2,chi2_rid,RMS_of_residuals,parameters_standard_errors,fit_results_string = fitting( known_x, known_y, guessed_p, yerr=None,p_boundaries=None,plot=False )
    print fit_results_string
    # if you do not wish to plot above you can also get the line coordinates with
    xs, ys= fitting.get_plot_x_y(x, num_points=1000, fit_allow_extra_fraction=0.025) # where x are your observations and num_points is the number of points in the profile you want to get.
    """

    def __init__(
        self,
        function=None,
        solve_system=True,
        iteration_factor=10,
        leastsq_tolerance=1.49012e-08,
    ):
        """
        function must be f(x, p) where p are the parameters to be estimated, whic should return a real number y
        it should be written so that if mutliple values of x are given as a numpy array the corresponding values of y are returned
         you should give to the class a function in the form f(x,parameters) that you want to fit!!
          if you want to solve a linear system of equation your functon must return x*p, where x is now the matrix of known coefficients and p is the solution you look for.
             General nomenclature in this case would be x=A and p=x)
        """
        self.function = function
        if self.function is None and solve_system:
            self.function = self._lin_system_function
        self.yerr = None
        self.p_boundaries = None
        self.iteration_factor = iteration_factor
        self.leastsq_tolerance = leastsq_tolerance
        self.use_ugly_chi2_autocorrection = True  # correct Chi2 when values are small (chi2 test should be used only for every y > 5 (approximately) )
        self.fitted_parameters = None

    def __call__(
        self,
        all_x,
        all_y,
        parameter_guess,
        yerr=None,
        p_boundaries=None,
        plot=False,
        **kwargs
    ):
        """
        return parameter_fitted, y_fit,chi2,chi2_rid,RMS_of_residuals,parameters_standard_errors,fit_results_string
        if plot is given a plot is drawn and
        returns parameter_fitted, y_fit,chi2,chi2_rid,RMS_of_residuals,parameters_standard_errors,figure
         you should give to the class a function in the form f(x,parameters) that you want to fit!!
          if you want to solve a linear system of equation your functon must return x*p, where x is now the matrix of known coefficients and p is the solution you look for.
             General nomenclature in this case would be x=A and p=x)
        p_boundaries, if given, should be a list of tuple (one tuple per parameter)
         with (lower_bound,higher_bound), if one do not wishes to give a specific boundary None can be given (None, higher_bound).
         , **kwargs are passed to self.plot()
        """
        if not isinstance(all_x, numpy.ndarray):
            all_x = numpy.array(all_x)
        if not isinstance(all_y, numpy.ndarray):
            all_y = numpy.array(all_y)
        if not isinstance(parameter_guess, numpy.ndarray):
            parameter_guess = numpy.array(parameter_guess)
        if self.function is None:
            raise Exception(
                "Error self.function is None, you should give to the class a function in the form f(x,parameters) that you want to fit!!\n if you want to solve a linear system of equation your functon must return x*p, where x is now the matrix of known coefficients and p is the solution you look for.\n   General nomenclature in this case would be x=A and p=x)"
            )
        if p_boundaries is not None:
            if type(p_boundaries) is tuple or type(p_boundaries) is list:
                if len(p_boundaries) == 2 and not hasattr(
                    p_boundaries[0], "__len__"
                ):  # lower and upper bound only, we assume these will be applied to all parameters
                    p_boundaries = [p_boundaries] * len(parameter_guess)
                p_boundaries = numpy.array(p_boundaries)
            if self.p_boundaries is not None:
                print(
                    "WARNING, overwriting p_boundaries as new have been given in __call__"
                )
            self.p_boundaries = p_boundaries
        if yerr is not None:
            if type(yerr) is list or type(yerr) is tuple:
                yerr = numpy.array(yerr)
            if self.yerr is not None:
                print("WARNING, overwriting yerr as new have been given in __call__")
            self.yerr = yerr
        (
            parameter_fitted,
            y_fit,
            chi2,
            chi2_rid,
            RMS_of_residuals,
            parameters_standard_errors,
            fit_results_string,
            popt_dw,
            popt_up,
            perr,
        ) = self.do_fit(
            parameter_guess,
            all_x,
            all_y,
            self.function,
            self.iteration_factor,
            self.leastsq_tolerance,
        )

        if plot:
            print(fit_results_string)
            figure = self.plot(all_x, all_y, parameter_fitted, yerr=yerr, **kwargs)
            fit_results_string = figure
        return (
            parameter_fitted,
            y_fit,
            chi2,
            chi2_rid,
            RMS_of_residuals,
            parameters_standard_errors,
            fit_results_string,
        )

    def _lin_system_function(self, A, x):
        """
        this can be given as a funciton to solve a linear system.
        In this case the systme is b=Ax and we look for the solution x
         Both A and x must be given as numpy array of proper shape!
        if will be use as f(x,p) since we identify A with the known x-values for the fit and p with the parameters we want
        """
        return (A * x).sum(axis=1)  # like a dot product in my limited knowledge

    def do_fit(
        self,
        parameters,
        all_x,
        all_y,
        function,
        iteration_factor=1,
        leastsq_tolerance=1.49012e-08,
        use_ugly_correction_for_small_obs=True,
    ):
        n_iterations = int(iteration_factor * 200 * (len(parameters) + 1))
        if (
            n_iterations > 0
        ):  # if it is zero it should do 200*(len(parameters)+1) iterations..
            print(
                "\nAbout to perform %d iterations, this may take some time.."
                % (n_iterations)
            )
        corrected_function = function
        if (
            use_ugly_correction_for_small_obs
        ):  # this correction should hepl yielding better chi2 values, that's all.
            nel = 1
            for s in all_y.shape:
                nel *= s
            # if all(all_y >= -0.2) and (all_y<=5.).sum()/float(nel) >0.1 : # if all y are positives (or nearly) and 10% or more of them are smaller than 5
            if any(abs(all_y) < 0.5):  # if some points are close to zero
                all_y = numpy.array(all_y) + 5.0

                def corrected_function(x, parms):
                    return 5.0 + function(x, parms)

                print("using UGLY correction for chi2 of small numbers...")
        p, cov_x, infodict, message, success = scipy.optimize.leastsq(
            self.cost_function,
            parameters,
            args=(all_x, all_y, corrected_function, self.yerr, self.p_boundaries),
            full_output=1,
            maxfev=n_iterations,
            ftol=leastsq_tolerance,
            xtol=leastsq_tolerance,
        )

        print(message)
        fit_results_string = 30 * "*" + " FIT RESULTS " + 30 * "*" + "\n"
        fit_results_string += "  Max number of iterations: %d \n" % (n_iterations)
        if cov_x is None:
            # print ' *** WARNING ***, cant estimate cov_x as flat curvature has been encountered (not necessarily a problem, but you wont get estimate of fit parameters errors)'
            fit_results_string += " *** WARNING ***, cant estimate cov_x as flat curvature has been encountered (not necessarily a problem, but you wont get estimate of fit parameters errors)\n"
            popt_dw, popt_up, perr = None, None, None
        else:
            popt_dw, popt_up, perr = self.get_CI_from_cov_matrix(p, cov_x)
        if success > 4:
            # print ' Warning, consider giving a larger value to iteration_factor since the optimal solution has not been reached (not necessarily a problem)'
            fit_results_string += " Warning, consider giving a larger value to iteration_factor since the optimal solution has not been reached (not necessarily a problem)\n"
        elif success != 1:
            # print 'not converged..'
            fit_results_string += "not converged... (not necessarily a problem)\n"
        chi2 = sum(
            self.cost_function(
                p, all_x, all_y, corrected_function, yerr=self.yerr, p_boundaries=None
            )
            ** 2
            / all_y
        )  # we do not give boundaries here since these are the resulting parameters
        degrees_of_freedom = len(all_x) - len(p)
        chi2_rid = chi2 / degrees_of_freedom
        RMS_of_residuals = numpy.sqrt(chi2_rid)
        parameters_standard_errors = None
        if cov_x is not None:
            parameters_standard_errors = numpy.sqrt(
                numpy.diagonal(cov_x)
            )  # *RMS_of_residuals MAYBE I SHOUD?? Disagreements online on whether this should be multiplied or not..
            params_str = " ".join(
                [
                    "P[%d]=%lf +/- %lf" % (j, pj, parameters_standard_errors[j])
                    for j, pj in enumerate(p)
                ]
            )
        else:
            params_str = " ".join(["P[%d]=%lf" % (j, pj) for j, pj in enumerate(p)])
        fit_results_string += (
            "success: %d   chi2= %s   chi2_rid= %s   RMS_of_residuals= %s\nFitted parameters:\n%s\n"
            % (success, repr(chi2), repr(chi2_rid), repr(RMS_of_residuals), params_str)
        )
        fit_results_string += 30 * "*" + "********************" + 30 * "*" + "\n"
        y_fit = function(all_x, p)
        self.fitted_parameters = p
        return (
            p,
            y_fit,
            chi2,
            chi2_rid,
            RMS_of_residuals,
            parameters_standard_errors,
            fit_results_string,
            popt_dw,
            popt_up,
            perr,
        )

    def get_CI_from_cov_matrix(self, popt, pcov, ci=0.95):
        """
        gets confidence interval from fitted parameters and their covariance matrix
        """
        # Convert to percentile point of the normal distribution.
        # See: https://en.wikipedia.org/wiki/Standard_score
        pp = (1.0 + ci) / 2.0
        # Convert to number of standard deviations.
        nstd = scipy.stats.norm.ppf(pp)
        # print nstd
        # Standard deviation errors on the parameters.
        perr = numpy.sqrt(numpy.diag(pcov))
        # Add nstd standard deviations to parameters to obtain the upper confidence
        # interval.
        popt_up = popt + nstd * perr
        popt_dw = popt - nstd * perr
        return popt_dw, popt_up, perr

    def get_plot_x_y(self, all_x, num_points=1000, fit_allow_extra_fraction=0.025):
        mx, Mx = min(all_x), max(all_x)
        a = fit_allow_extra_fraction * (Mx - mx)
        x_s = numpy.linspace(mx - a, Mx + a, num_points)
        y_s = self.function(x_s, self.fitted_parameters)
        return x_s, y_s

    def plot(
        self,
        all_x,
        all_y,
        parameters,
        yerr=None,
        num_points=1000,
        fit_allow_extra_fraction=0.025,
        **kwargs
    ):
        fit_x, fit_y = self.get_plot_x_y(
            all_x,
            num_points=num_points,
            fit_allow_extra_fraction=fit_allow_extra_fraction,
        )

        zorder_ref = 10
        fitls = "-"
        if "markersize" not in kwargs:
            if yerr is None:
                kwargs["markersize"] = 15
            else:
                kwargs["markersize"] = 10
        if "marker" not in kwargs:
            kwargs["marker"] = "."
        if "ls" in kwargs:
            fitls = kwargs["ls"]
            del kwargs["ls"]
        if "zorder" in kwargs:
            zorder_ref = kwargs["zorder"]
            del kwargs["zorder"]
        # first plot data points
        figure = plotter.profile(
            all_y, all_x, yerr=yerr, ls="", zorder=zorder_ref, **kwargs
        )
        if "fit_linecolor" in kwargs:
            kwargs["color"] = kwargs["fit_linecolor"]
        if "color" not in kwargs:
            kwargs["color"] = "red"
        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 3
        if "marker" in kwargs:
            del kwargs["marker"]
        if "figure" in kwargs:
            del kwargs["figure"]
        if "ls" not in kwargs:
            kwargs["ls"] = fitls
        figure = plotter.profile(
            fit_y, fit_x, marker="", zorder=zorder_ref - 1, figure=figure, **kwargs
        )
        return figure

    def parameters_box_penalty(
        self, par, lower_bound=None, upper_bound=None, factor=1e6
    ):
        """
        this is used to give a penalty if parameters go outside a range.. with default options it just return 0
        """
        if isinstance(par, numpy.ndarray):
            if lower_bound is not None and any(par < lower_bound):
                return factor
            if upper_bound is not None and any(par > upper_bound):
                return factor
        else:
            if lower_bound is not None and par < lower_bound:
                return factor
            if upper_bound is not None and par > upper_bound:
                return factor
        return 0

    def cost_function(self, p, x, y, function, yerr=None, p_boundaries=None):
        """
        at this stage p_boundaries, if given, should be a list of tuple (one tuple per parameter)
         with (lower_bound,higher_bound), if one do not wishes to give a specific boundary None can be given (None, higher_bound).
        yerr must be a scalar or a numpy array, in the latter case it must have the same shape of y.
        """
        err = None
        y_fit = function(x, p)  # x at this stage should be a nm
        if p_boundaries is not None:
            penalty = 0
            for j, tup in enumerate(p_boundaries):
                penalty += self.parameters_box_penalty(p[j], tup[0], tup[1])
                if penalty > 0:
                    y_fit += penalty
        if yerr is not None:
            # print y.shape, yerr.shape
            err = (y - y_fit) / (
                numpy.mean(yerr) + yerr ** 2
            )  # the mean is essential for erros that contain zeros or that span multiple order of magnitudes.
        else:
            err = y - y_fit
        return err

    def cost_function_global(
        self,
        p,
        all_X_in_a_row,
        all_Y_in_a_row,
        tuple_with_slicing_idices,
        tuple_of_functions,
        tuple_of_lists_of_indices_in_p,
        tuple_of_Y_err=None,
        heigth_boundaries=(0.0, None),
        mean_boundaries=(None, None),
        sigma_boundaries=(0.0, None),
    ):
        # can be used for global fitting, not implemented yet
        # tuple_with_slicing_idices must have in position -1 len(all_X_in_a_row)
        err = None
        for i in range(len(tuple_of_functions)):
            params = [p[j] for j in tuple_of_lists_of_indices_in_p[i]]
            y_fit = tuple_of_functions[i](
                all_X_in_a_row[
                    tuple_with_slicing_idices[i] : tuple_with_slicing_idices[i + 1]
                ],
                params,
            )
            penalty = 0
            params = numpy.array(params)
            penalty += self.parameters_box_penalty(params[0::3], *heigth_boundaries)
            penalty += self.parameters_box_penalty(params[1::3], *mean_boundaries)
            penalty += self.parameters_box_penalty(params[2::3], *sigma_boundaries)
            if penalty > 0:
                y_fit += penalty
            if tuple_of_Y_err is not None:
                if err is None:
                    err = (
                        all_Y_in_a_row[
                            tuple_with_slicing_idices[i] : tuple_with_slicing_idices[
                                i + 1
                            ]
                        ]
                        - y_fit
                    ) / tuple_of_Y_err[i]
                else:
                    err = numpy.hstack(
                        (
                            err,
                            (
                                all_Y_in_a_row[
                                    tuple_with_slicing_idices[
                                        i
                                    ] : tuple_with_slicing_idices[i + 1]
                                ]
                                - y_fit
                            )
                            / tuple_of_Y_err[i],
                        )
                    )

            else:
                if err is None:
                    err = (
                        all_Y_in_a_row[
                            tuple_with_slicing_idices[i] : tuple_with_slicing_idices[
                                i + 1
                            ]
                        ]
                        - y_fit
                    )
                else:
                    err = numpy.hstack(
                        (
                            err,
                            all_Y_in_a_row[
                                tuple_with_slicing_idices[
                                    i
                                ] : tuple_with_slicing_idices[i + 1]
                            ]
                            - y_fit,
                        )
                    )
            # print len(y_fit),len(all_X_in_a_row[ tuple_with_slicing_idices[i]:tuple_with_slicing_idices[i+1] ]),len(err),len(all_Y_in_a_row[ tuple_with_slicing_idices[i]:tuple_with_slicing_idices[i+1] ])
        return err


class Fit_noisy:
    def __init__(
        self,
        function=None,
        p_boundaries=0.8,
        nsteps=90000,
        T=0.01,
        p_step=0.05,
        log=sys.stdout,
    ):
        self.function = function
        self.NSTEPS_ = nsteps  # number of MC steps
        self.KBTMIN_ = T  # MC temperature
        self.KBTMAX_ = 10.0  # Annealing temperature
        self.NPRINT_ = (
            10000  # print to self.log every NPRINT_ MC steps (can set to None)
        )
        self.p_boundaries = 0.8  # boundaries will be set at 0.8* times guessed value per side if not explicitly given in __call__
        self.p_step = p_step  # can also be given as list to have a specific step for each parameter
        # SIMULATED ANNEALING
        self.NCOLD_ = 5000  # consecutive steps at low temperature
        self.NHOT_ = 800  # consecutive steps at high temperature
        ### I would not touch these
        # tipical error
        self.S0MIN_ = 0.001  # uncertainty min
        self.S0MAX_ = 10.0  # uncertainty max
        self.DS0MAX_ = 0.01  # moves in uncertainty
        # constants
        self.SQRT2PI_ = numpy.sqrt(2.0 * numpy.pi)
        self.SQRT2_DIV_PI_ = numpy.sqrt(2.0) / numpy.pi
        self.log = log
        self.warn = sys.stderr

    def __call__(
        self,
        all_x,
        all_y,
        parameter_guess=None,
        p_boundaries=None,
        yerr=None,
        plot=False,
        guess_from_leastsquares=True,
        function_has_parameter_list=True,
    ):
        """
        return params,y_fit,chi2,chi2_rid, pvalue, output_best, MC_str+FIT_str
        if both x and y have an error then - if you know the fitting line y=f(x) - use the formula for the equivalent error
            sigma_eq= sqrt(  obs_err**2 + ( df/dx * exp_err )**2 )
        if plot is given a plot is drawn
         you should give to the class a function in the form f(x,parameters) that you want to fit!!
          
        p_boundaries, if given, should be a list of tuples/lists (one tuple per parameter)
         with (lower_bound,higher_bound), if one do not wishes to give a specific boundary None can be given, in which case boundaries are 
         obtained from the parameter_guess using self.p_boundaries* times guessed value per side of each guess. (in case guess is zero self.p_boundaries is used as boundaries)
        """
        if not function_has_parameter_list:
            print(
                "ERRRO function_has_parameter_list False not implemented in Fit_noisy"
            )
        if not isinstance(all_x, numpy.ndarray):
            all_x = numpy.array(all_x)
        if not isinstance(all_y, numpy.ndarray):
            all_y = numpy.array(all_y)
        if self.function is None:
            raise Exception(
                "Error self.function is None, you should give to the class a function in the form f(x,parameters) that you want to fit!!\n if you want to solve a linear system of equation your functon must return x*p, where x is now the matrix of known coefficients and p is the solution you look for.\n   General nomenclature in this case would be x=A and p=x)"
            )
        if p_boundaries is None and parameter_guess is None:
            raise Exception(
                "Error either p_boundaries or parameter_guess are compulsory kwargs, both given as None"
            )
        if p_boundaries is None:
            p_boundaries = self.p_boundaries
        if p_boundaries is not None:
            if type(p_boundaries) is float:
                self.warn.write(
                    "Setting p_boundaries from parameter_guess with %g fraction from guess %s\n"
                    % (self.p_boundaries, str(parameter_guess))
                )
                p_boundaries = [
                    [p - (p * p_boundaries), p + (p * p_boundaries)]
                    if not CompareFloats(p, 0, sensibility=0.5)
                    else [p - p_boundaries, p + p_boundaries]
                    for p in parameter_guess
                ]
                self.warn.write("   p_boundaries= %s\n" % (str(p_boundaries)))
            elif len(p_boundaries) == 2 and not hasattr(
                p_boundaries[0], "__len__"
            ):  # lower and upper bound only, we assume these will be applied to all parameters
                p_boundaries = [p_boundaries] * len(
                    parameter_guess
                )  # all the same, or probably only one par_guess
            p_boundaries = numpy.array(p_boundaries)

            self.p_boundaries = p_boundaries
        if parameter_guess is None:
            parameter_guess = [
                random.uniform(pg[0], pg[1]) for pg in p_boundaries
            ]  # randomly get them from the boundaries

        if guess_from_leastsquares == True:
            ls = Fit(function=self.function, solve_system=False)
            parameter_fitted_leastq, _, _, _, _, _, fit_results_string = ls(
                all_x, all_y, parameter_guess, yerr=None, p_boundaries=None, plot=False
            )
            if self.log is not None:
                self.log.write("\nLEAST SQUARE FITTING:\n%s\n" % (fit_results_string))
            p_step = []
            for j, p in enumerate(parameter_fitted_leastq):
                if p_boundaries[j][0] > p or p_boundaries[j][1] < p:
                    self.warn.write(
                        "***WARNING*** discarding guess from least square P[%d]= %lf as it falls outside the boundaries %s!! --> restoring given %lf\n"
                        % (j, p, str(p_boundaries[j]), parameter_guess[j])
                    )
                    p_step += [self.p_step]
                else:
                    parameter_guess[j] = p
                    if CompareFloats(p, 0.0, sensibility=0.3):
                        p_step += [self.p_step]
                    else:
                        p_step += [0.01 * p]  # change by 1%
            self.p_step = p_step[:]
        parameter_guess = numpy.array(parameter_guess).astype("f")

        if (
            yerr is not None
        ):  # we use the error as a weight. For consistency we set the weights with a mean 1
            if type(yerr) is list or type(yerr) is tuple:
                yerr = numpy.array(yerr)
            weights = (
                1.0 / (yerr + numpy.mean(yerr) * 0.001) ** 2
            )  # mask points with error zero
            # weights= weights+1.-numpy.mean(weights) # give them mean 1 so that the other parameters don't need to be changed according to whether yerr is given or not.
        else:
            weights = None
        output_best = self.doMC(
            all_x, all_y, parameter_guess, p_boundaries, weights=weights
        )
        params = [
            output_best[j] for j in range(len(parameter_guess))
        ]  # best parameters
        str_acc = ""
        par_str = ""
        for j in range(len(parameter_guess)):
            str_acc += str(j) + "=%lf " % float((output_best["Acceptance-" + str(j)]))
            par_str += "P[%d]= %lf " % (j, params[j])
        str_acc += " sig0=%lf " % (float(output_best["Acceptance-sig0"]))
        MC_str = (
            " ********************  Noisy Fit MC results ********************\n   Best energy = %lf at step %d 0f %d  sig0=%lf T=%lf\n                         Acceptances:\n  %s\n *****  *****  *****  *****  ***** ***** ***** ***** ***** *****\n"
            % (
                output_best["Total_Energy"],
                output_best["Step"],
                self.NSTEPS_,
                output_best["sig0"],
                output_best["KBT"],
                str_acc,
            )
        )
        self.fitted_parameters = params
        y_fit = self.function(all_x, params)
        chi2, pvalue, chi2_rid = chi2_test(
            abs(y_fit) + 5, abs(all_y) + 5, dof=len(parameter_guess), obs_err=yerr
        )
        FIT_str = (
            " ********************  Noisy Fit results ********************\n   Chi2 = %lf pval = %lf  Chi2rid = %lf [Very ugly way to get Chi2]\n                         PARAMETERS:\n  %s\n *****  *****  *****  *****  ***** ***** ***** ***** ***** *****\n"
            % (chi2, pvalue, chi2_rid, par_str)
        )
        if self.log is not None:
            self.log.write("\n%s%s\n" % (MC_str, FIT_str))
        if plot:
            figure = self.plot(all_x, all_y, params, yerr=yerr)
            if guess_from_leastsquares:
                figure = self.plot(
                    all_x,
                    all_y,
                    parameter_fitted_leastq,
                    linecolor="blue",
                    linestyle="-.",
                    linewidth=2,
                    markersize=None,
                    plot_points=False,
                    figure=figure,
                )

        # parameter_fitted, y_fit,chi2,chi2_rid,RMS_of_residuals,parameters_standard_errors,fit_results_string
        return params, y_fit, chi2, chi2_rid, pvalue, output_best, MC_str + FIT_str

    def doMC(
        self, all_x, all_y, parameter_guess, p_boundaries, weights=None, parms=None
    ):
        sig0 = self.S0MAX_
        # prepare best stuff
        output_best = {}
        e_best = 1e19
        if parms is None:
            parms = numpy.copy(parameter_guess)
        if not hasattr(self.p_step, "__len__"):
            self.p_step = [
                self.p_step for j in range(len(parms))
            ]  # convert pstep to list, same pstep for each parameter
        # initialize counter for acceptance
        accept = {}
        for j in range(len(parms)):
            accept[j] = 0.0
            output_best[j] = 0
        accept["sig0"] = 0.0

        # start MC
        for istep in range(self.NSTEPS_):

            # simulated annealing, kbt is the temperature
            kbt = self.simulated_annealing(istep, self.NCOLD_, self.NHOT_)

            ene = self.get_score(all_x, parms, all_y, sig0, weights=weights)

            # propose move in all parameters, one at a time
            for j, p in enumerate(parms):
                p_new = self.propose_move(
                    p, self.p_step[j], p_boundaries[j][0], p_boundaries[j][1]
                )
                nps = numpy.copy(parms)
                nps[j] = p_new
                # print p,parms,nps,p_new,j,
                e_new = self.get_score(all_x, nps, all_y, sig0, weights=weights)
                parms[j], ene, ac = self.accept_or_reject(p, p_new, ene, e_new, kbt)
                if kbt < self.KBTMAX_:
                    accept[j] += ac  # update only if not annealing
            # propose move in uncertainty
            sig0_new = self.propose_move(sig0, self.DS0MAX_, self.S0MIN_, self.S0MAX_)
            e_new = self.get_score(all_x, parms, all_y, sig0_new, weights=weights)
            sig0, ene, ac = self.accept_or_reject(sig0, sig0_new, ene, e_new, kbt)
            if kbt < self.KBTMAX_:
                accept["sig0"] += ac  # update only if not annealing

            # save best
            if ene < e_best:
                # update best
                e_best = ene
                # step
                output_best["Step"] = istep
                # total energy
                output_best["Total_Energy"] = ene
                # acceptance
                for key in accept:
                    output_best["Acceptance-" + str(key)] = accept[key] / float(
                        istep + 1
                    )
                # value of parameters
                for j, p in enumerate(parms):
                    output_best[j] = p
                output_best["sig0"] = sig0
                # temperature
                output_best["KBT"] = kbt

            # time to print ?
            if self.NPRINT_ is not None and istep % self.NPRINT_ == 0:
                # prepare dictionary
                output = OrderedDict()  # so that parms are printed at beginning
                # value of parameters
                for j, p in enumerate(parms):
                    output[j] = p
                # step
                output["Step"] = istep
                # total energy
                output["Total_Energy"] = ene
                # acceptance
                for key in accept:
                    output["Acceptance-" + str(key)] = accept[key] / float(istep + 1)

                output["sig0"] = sig0
                # temperature
                output["KBT"] = kbt
                # dump stuff
                if self.log is not None:
                    self.log.write("%s \n" % output)
        return output_best

    def get_plot_x_y(
        self,
        all_x,
        fitted_parameters=None,
        num_points=1000,
        fit_allow_extra_fraction=0.025,
    ):
        if fitted_parameters is None:
            fitted_parameters = self.fitted_parameters
        mx, Mx = min(all_x), max(all_x)
        a = fit_allow_extra_fraction * (Mx - mx)
        x_s = numpy.linspace(mx - a, Mx + a, num_points)
        y_s = self.function(x_s, fitted_parameters)
        return x_s, y_s

    def plot(
        self,
        all_x,
        all_y,
        parameters,
        yerr=None,
        num_points=1000,
        fit_allow_extra_fraction=0.025,
        marker=".",
        linecolor="red",
        linestyle="--",
        linewidth=2,
        markersize=None,
        plot_points=True,
        figure=None,
        **kwargs
    ):
        fit_x, fit_y = self.get_plot_x_y(
            all_x,
            fitted_parameters=parameters,
            num_points=num_points,
            fit_allow_extra_fraction=fit_allow_extra_fraction,
        )
        if markersize is None:
            if yerr is None:
                markersize = 15
            else:
                markersize = 10
        if plot_points:
            figure = plotter.profile(
                all_y,
                all_x,
                ls="",
                yerr=yerr,
                markersize=markersize,
                marker=marker,
                figure=figure,
                **kwargs
            )
        figure = plotter.profile(
            fit_y,
            fit_x,
            ls=linestyle,
            color=linecolor,
            linewidth=linewidth,
            figure=figure,
            **kwargs
        )
        return figure

    def get_score(self, xs, params, exps, sig0, weights=None):
        # print len(params),params,xs.shape,exps.shape,self.function(xs,params).shape
        if weights is not None:
            dev = ((self.function(xs, params) - exps) ** 2) * weights
        else:
            dev = (self.function(xs, params) - exps) ** 2
        ene = -(
            numpy.log(self.SQRT2_DIV_PI_ * sig0 / (dev + 2.0 * sig0 * sig0))
        ).mean()  # I use mean rather than sum so that the temperature of the optimisation is less dependent on the number of x,y
        ene += numpy.log(sig0)
        return ene

    def simulated_annealing(self, istep, ncold, nhot):
        if istep % (ncold + nhot) < ncold:
            value = 0.0
        else:
            value = 1.0
        kbt = self.KBTMIN_ + (self.KBTMAX_ - self.KBTMIN_) * value
        return kbt

    def propose_move(self, x, dxmax, xmin, xmax):
        dx = random.uniform(-dxmax, dxmax)
        x_new = x + dx
        # check boundaries
        if x_new > xmax:
            x_new = 2.0 * xmax - x_new
        if x_new < xmin:
            x_new = 2.0 * xmin - x_new
        return x_new

    def accept_or_reject(self, x, x_new, ene, e_new, kbt):
        ac = 0.0
        delta = (e_new - ene) / kbt
        # downhill move -> always accepted
        if delta < 0.0:
            ene = e_new
            x = x_new
            ac = 1.0
        # uphill move -> accept with certain probability
        else:
            r = random.random()
            delta = numpy.exp(-delta)
            if r < delta:
                ene = e_new
                x = x_new
                ac = 1.0
        return x, ene, ac


"""  
Look also:
http://stackoverflow.com/questions/9742739/how-do-i-make-processes-able-to-write-in-an-array-of-the-main-program/9849971#9849971
"""
# given a symmetric matrix (which cuold be a matrix with multiple alignment scores, or maybe a contact matrix).
# It actually looks at the top right triangle so there is no need to fill the matrix completely.
# it extract clusters of entries between lower_threshold and upper_threshold.
# Every entry in a cluster has to sit in this interval when compared to every other entry in the same cluster. One element can't end up in multiple clusters,
# the idea is that element as close to the upper_threshold as possible are put in a cluster
# (badly explained, if one elemente can fit in two clusters it will end up in the one for which it has higher (closer to upper threshold) matrix elements)....
# Anyway two different clusters won't contain elements whose relative score is between lower_threshold and upper_threshold.
# If you have understood this explanation: 1) you must be very smart, 2) you might wish to know that
# it is returned a list of lists, where every list is a cluster and entries are row/column identifiers of the symmetric matrix you gave as an input.
# strict_clustering=False loosens the condition, it has to satisfy the conditions with any of the elements in the cluster, useful for separating connected regions
# in this case the higher_first=False voice become crucial. In fact it changes the separation procedure, higher_first=True favours clustering of entry closer to the upper threshold
# while false to favour clustering of entries closer to the lower threshold. When the matrix is filled with Norms or distances set it to False as 0 menans identical entries!
def clusters_from_matrix(
    symmetric_matrix,
    lower_threshold=0.8,
    upper_threshold=1.1,
    strict_clustering=True,
    higher_first=False,
    avoid_diagonal=True,
    return_scores_and_score_dict=False,
):
    # first we get a vector with all the possible scores
    if avoid_diagonal:
        shift = 1
    else:
        shift = 0
    scores = []
    score_dict = {}  # this will end up being like an "alignemnt tree"
    for s1 in range(0, len(symmetric_matrix)):
        for s2 in range(s1 + shift, len(symmetric_matrix[0])):
            if symmetric_matrix[s1][s2] not in scores:
                scores += [symmetric_matrix[s1][s2]]
            if symmetric_matrix[s1][s2] not in score_dict:
                score_dict[symmetric_matrix[s1][s2]] = [[s1, s2]]
            else:
                score_dict[symmetric_matrix[s1][s2]] += [[s1, s2]]
    scores.sort(reverse=higher_first)  # sort
    clusters = []
    c = 0
    while c < len(scores):
        if (
            clusters == []
            and lower_threshold
            <= symmetric_matrix[score_dict[scores[c]][0][0]][
                score_dict[scores[c]][0][1]
            ]
            <= upper_threshold
        ):
            clusters += [
                score_dict[scores[c]][0][:]
            ]  # we want to actually copy the pair, not to reference to the same piece of memory
        for pair in score_dict[scores[c]]:
            for seq_id in pair:
                if strict_clustering:
                    already_assigned = False
                    assigned = False
                    for cluster in clusters:
                        add_it = True
                        if seq_id not in cluster:
                            for (
                                seq_id2
                            ) in (
                                cluster
                            ):  # the following two if are just to have it working also if only the upper right part of the matrix has been filled
                                if seq_id < seq_id2 and (
                                    lower_threshold > symmetric_matrix[seq_id][seq_id2]
                                    or symmetric_matrix[seq_id][seq_id2]
                                    > upper_threshold
                                ):
                                    add_it = False
                                elif seq_id > seq_id2 and (
                                    lower_threshold > symmetric_matrix[seq_id2][seq_id]
                                    or symmetric_matrix[seq_id2][seq_id]
                                    > upper_threshold
                                ):
                                    add_it = False
                            if add_it and not already_assigned:
                                assigned = True
                                cluster += [
                                    seq_id
                                ]  # technically this could satisfy the conditions for multiple clusters, but since we have ordered the scores we hope it will end up in the most similar one!
                                break
                        else:
                            already_assigned = True
                            break
                else:  # loose condition, it has to satisfy the conditions with any of the elements in the cluster, useful for separating connected regions
                    already_assigned = False
                    assigned = False
                    for cluster in clusters:
                        if seq_id not in cluster:
                            for (
                                seq_id2
                            ) in (
                                cluster
                            ):  # the following two if are just to have it working also if only the upper right part of the matrix has been filled
                                if seq_id < seq_id2 and (
                                    lower_threshold <= symmetric_matrix[seq_id][seq_id2]
                                    and symmetric_matrix[seq_id][seq_id2]
                                    <= upper_threshold
                                ):
                                    assigned = True
                                    cluster += [
                                        seq_id
                                    ]  # technically this could satisfy the conditions for multiple clusters, but since we have ordered the scores we hope it will end up in the most similar one!
                                    break
                                elif seq_id > seq_id2 and (
                                    lower_threshold <= symmetric_matrix[seq_id2][seq_id]
                                    and symmetric_matrix[seq_id2][seq_id]
                                    <= upper_threshold
                                ):
                                    assigned = True
                                    cluster += [
                                        seq_id
                                    ]  # technically this could satisfy the conditions for multiple clusters, but since we have ordered the scores we hope it will end up in the most similar one!
                                    break
                        else:
                            already_assigned = True
                            break
                if not assigned and not already_assigned:
                    clusters += [[seq_id]]  # init a new cluster

        c += 1
    if return_scores_and_score_dict:
        return (clusters, scores, score_dict)
    return clusters
