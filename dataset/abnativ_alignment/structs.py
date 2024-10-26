# (c) 2023 Sormannilab


import copy
import os
import sys
from collections import OrderedDict

import Bio.PDB  # part of biophython
import numpy

try:
    from . import mybio  # custom module
except ImportError:  # will raise an error only if this missing module is actually used
    try:
        import mybio
    except Exception:

        class tmpmybio:
            def pairwise_alignment(self, *a, **ka):
                raise Exception("mybio module not available, CANNOT ALIGN SEQUENCES\n")

        mybio = tmpmybio()

try:
    from .parse_pdb import (
        PDBParser_mod,  # a modified version, but not really necessary for most applications.
    )
except ImportError:
    try:
        from parse_pdb import (
            PDBParser_mod,  # a modified version, but not really necessary for most applications.
        )
    except Exception:
        from Bio.PDB import PDBParser as PDBParser_mod

try:
    import freesasa  # pip install freesasa  would solve this issue if it manifest
except ImportError:

    class tmpfreesasa:  # will raise an error only if this missing module is actually used
        def calcBioPDB(self, *args, **kwargs):
            raise Exception(
                "freesasa module not available Cannot calculates surfaces- try installing it from Github or with command 'pip install freesasa'\n"
            )

        def Parameters(self, *args, **kwargs):
            raise Exception(
                "freesasa module not available Cannot calculates surfaces- try installing it from Github or with command 'pip install freesasa'\n"
            )

        def LeeRichards(self, *args, **kwargs):
            raise Exception(
                "freesasa module not available Cannot calculates surfaces- try installing it from Github or with command 'pip install freesasa'\n"
            )

    freesasa = tmpfreesasa()


# Abstract struct class (allow to define e.g. r=Residue(resname='bla',pos='blabla') )
class Struct:
    def __init__(self, *argv, **argd):
        if len(argd):
            # Update by dictionary
            self.__dict__.update(argd)
        else:
            # Update by position
            attrs = [x for x in dir(self) if x[0:2] != "__"]
            for n in range(len(argv)):
                setattr(self, attrs[n], argv[n])


# class that holds some statistical variables, with useful methods like update and finalize
class Stats:
    # class that holds some statistical variables, with useful methods like update and finalize
    name = None  # name of the class, or description of what it contains
    count = 0
    null_entries = 0
    summ = 0.0
    entries = []
    ids = []  # one might wish to give ids
    weights = None
    value_to_id = (
        {}
    )  # one might wish to be able to go from value to ids (use create_value_to_id())
    null_ids = []
    histogram = []
    average = None
    Weigthed_summ = 0
    Wsumm = 0
    stdev = 0.0
    stdevN = 0.0
    Max = -99999.0
    Max_index = None
    Min = 99999.0
    Min_index = None

    def __init__(self, name="stats_class", names=None):
        self.name = name
        self.average = None
        self.count = 0.0
        self.null_entries = 0
        self.names = names
        self.ids = []  # one might wish to give ids
        self.weights = None
        self.null_ids = []
        self.summ = 0.0
        self.Weigthed_summ = 0.0
        self.Wsumm = 0.0
        self.histogram = []
        self.entries = []
        self.stdev = 0.0
        self.stdevN = 0.0
        self.Max = -99999.0
        self.Min = 99999.0
        self.Max_index = None
        self.Min_index = None

    def __len__(self):
        return len(self.entries)  # ,self.count,self.null_entries

    def __repr__(
        self,
    ):  # used when class is represented (i.e. when in python you press enter!)
        tmp_str = ""
        for attr, value in self.__dict__.items():
            if type(value) is list or type(value) is dict or type(value) is tuple:
                tmp_str += str(attr) + "=" + str(type(value)) + " "
            else:
                tmp_str += str(attr) + "=" + repr(value) + " "
        return tmp_str[:-1]

    def __str__(self):  # used when class is printed
        tmp_str = ""
        for attr, value in self.__dict__.items():
            if type(value) is list or type(value) is dict or type(value) is tuple:
                tmp_str += str(attr) + "=" + str(type(value)) + "\t"
            else:
                tmp_str += str(attr) + "=" + repr(value) + "\t"
        return tmp_str[:-1]

    def retrieve(self, variable_name_str, print_warnings=True):
        # print_warnings useful when embedded in try:
        if variable_name_str not in dir(self):
            if print_warnings:
                sys.stderr.write(
                    "**ERROR** in method retrieve, %s not defined in class\n    %s\n"
                    % (variable_name_str, str(dir(self)))
                )
            raise AttributeError
        return self.__dict__[variable_name_str]

    def update(self, number, en_id=None, weight=None, name=None):
        # add one entry, (this function should be called inside a loop, then .finalize() should be used to get the statistics...
        if not isinstance(number, (int, float, complex)):
            self.null_entries += 1
            self.null_ids += [en_id]
            if self.weights != None:
                self.weights.append(0)
            if self.names != None and name != None:
                self.names.append(name)
            return
        if weight != None:
            if self.weights == None:
                self.weights = [weight]
            else:
                self.weights += [weight]
            self.Weigthed_summ += number * weight
            self.Wsumm += weight
        self.summ += number
        self.entries += [number]
        self.ids += [en_id]
        self.count += 1
        if self.names != None and name != None:
            self.names.append(name)

    def update_list(self, list_of_numbers, list_of_ids=None, weight_list=None):
        # add a list or a dictionary of entries, (then .finalize() should be used to get the statistics...
        if type(list_of_numbers) is dict or isinstance(list_of_numbers, OrderedDict):
            list_of_ids = list(
                list_of_numbers.keys()
            )  # python 2.x guarantees a 1 to 1 correspondence between .values and .keys
            list_of_numbers = list(list_of_numbers.values())
        if weight_list != None:
            if self.weights == None:
                self.weights = []
        for j, number in enumerate(list_of_numbers):
            if not isinstance(number, (int, float, complex)):
                self.null_entries += 1
                if list_of_ids != None:
                    self.null_ids += [list_of_ids[j]]
                else:
                    self.null_ids += [None]
            else:
                if weight_list != None:
                    self.weights += [weight_list[j]]
                    self.Wsumm += weight_list[j]
                    self.Weigthed_summ += number * weight_list[j]
                if list_of_ids != None:
                    self.ids += [list_of_ids[j]]
                else:
                    self.null_ids += [None]
                self.summ += number
                self.entries += [number]
                self.count += 1

    def free(self, free_from_index=None, free_to_index=None):
        if type(self.entries) is not list:
            self.entries = list(self.entries)
        del self.entries[free_from_index:free_to_index]
        if type(self.ids) is not list:
            self.ids = list(self.ids)
        del self.ids[free_from_index:free_to_index]
        if self.names != None:
            if type(self.names) is not list:
                self.names = list(self.names)
            del self.names[free_from_index:free_to_index]

    # compute average and standard deviation. Get max and min
    def finalize(self):
        # compute average and standard deviation. Get max and min
        self.stdev = 0.0
        if (
            self.count > 0
        ):  # works even if we have updated something between two calls of finalize
            self.average = self.summ / float(self.count)
            if self.weights != None:
                self.weight_average = self.Weigthed_summ / self.Wsumm
                self.weight_stdev = 0.0
        else:
            self.average = 0.0
        for j, entry in enumerate(self.entries):
            if entry > self.Max:
                self.Max = entry
                self.Max_index = j
            if entry < self.Min:
                self.Min = entry
                self.Min_index = j
            self.stdev += (entry - self.average) * (entry - self.average)
            if self.weights != None:
                self.weight_stdev += (
                    self.weights[j] * (entry - self.average) * (entry - self.average)
                )
        if self.count > 100:
            self.stdev = numpy.sqrt(
                self.stdev / float(self.count - 1)
            )  # N-1 is the Bessel's correction
            self.stdevN = self.stdev / numpy.sqrt(self.count)
            if self.weights != None:
                self.weight_stdev = numpy.sqrt(self.weight_stdev / self.Wsumm)
        elif self.count > 1:
            self.stdev = numpy.sqrt(
                self.stdev / float(self.count)
            )  # No Bessel's correction
            self.stdevN = self.stdev / numpy.sqrt(self.count)
            if self.weights != None:
                self.weight_stdev = numpy.sqrt(self.weight_stdev / self.Wsumm)
        else:
            self.stdev = 0.0  # in this case it is actually undefined
            self.stdevN = 0.0

    def create_value_to_id(self):
        if len(self.ids) == len(self.entries):
            for j, value in enumerate(self.entries):
                if value in self.value_to_id:
                    self.value_to_id[value] += [self.ids[j]]
                else:
                    self.value_to_id[value] = [self.ids[j]]
        else:
            sys.stderr.write(
                "**ERROR** can't create create_value_to_id dictionary as len(self.ids)!=len(self.entries) %d %d\n"
                % (len(self.ids), len(self.entries))
            )

    # ------------------------------ FUNCTIONS TO GENERATE HISTOGRAMS ------------------------------ #
    # compare two float numbers at a specified sensibliity
    def CompareFloats(self, float1, float2, sensibility=0.0001):
        if float1 - sensibility <= float2 <= float1 + sensibility:
            return True
        else:
            return False

    def sort(self, reverse=False, use_np_array=True):
        """
        sorts the entries and the corresponding ids at once (according to the values in the entries)
        """
        if use_np_array:
            self.entries = numpy.array(self.entries)
            if reverse:
                sorter = numpy.argsort(-1 * self.entries)
            else:
                sorter = numpy.argsort(self.entries)
            self.entries = self.entries[sorter]
            self.ids = numpy.array(self.ids)[sorter]
            if self.names != None and self.names != []:
                self.names = numpy.array(self.names)[sorter]
            return
        if self.names != None and self.names != []:
            self.entries, self.ids, self.names = list(
                zip(*sorted(zip(self.entries, self.ids, self.names), reverse=reverse))
            )
        else:
            self.entries, self.ids = list(
                zip(*sorted(zip(self.entries, self.ids), reverse=reverse))
            )
        return

    # return the bin corresponding to input_number in the range [minimum, maximum].Note that reciprocal_of_bin_size is Number_of_bins/(maximum-minimum)
    def GetBin(self, input_number, minimum, maximum, reciprocal_of_bin_size):
        if self.CompareFloats(input_number, maximum, 0.00001):
            return int(
                reciprocal_of_bin_size * (maximum - minimum) - 1
            )  # in this particular case return nbin-1
        elif input_number > maximum or input_number < minimum:
            return -1
        return int(reciprocal_of_bin_size * (input_number - minimum))

    # return a list of length Number_of_bins corresponding to the Histogram obtained from list_of_input.
    # it also stores it in self.histogram
    # One can enter maximum or minimum if he wants to consider only entries of list of inputs that fall within the given range.
    # if the histogram is needed normalized one can set normalized to True (default)
    def CreateHistogram(
        self,
        nbins=None,
        list_of_input=None,
        minimum=None,
        maximum=None,
        normalized=True,
        QUIET=False,
    ):
        if nbins == None:
            nbins = numpy.sqrt(len(self.entries))
        if list_of_input == None:
            list_of_input = self.entries
        if minimum == None:
            minimum = self.Min
        if maximum == None:
            maximum = self.Max
        Histogram = [0] * nbins  # allocate list of zeroes
        N = len(list_of_input)
        if N == 0:
            return -1
        if N <= nbins:
            sys.stderr.write(
                "***WARNING*** in CreateHistogram number of inputs is smaller than requested number of bins\n"
            )
        if minimum == None:
            minimum = min(list_of_input)
        if maximum == None:
            maximum = max(list_of_input)
        num_entries = 0
        reciprocal_of_bin_size = float(nbins) / (maximum - minimum)
        for entry in list_of_input:  # fill Histogram
            i = self.GetBin(float(entry), minimum, maximum, reciprocal_of_bin_size)
            if i >= 0:
                Histogram[i] += 1
                num_entries += 1
        if not QUIET:
            print(
                "generated Histogram with %d entries in the range [%lf , %lf]"
                % (num_entries, minimum, maximum)
            )
        if normalized and num_entries > 0:
            Histogram = [1.0 * x / float(num_entries) for x in Histogram]
        self.histogram = Histogram
        x_axis = []
        for bi in range(0, nbins):
            x_axis += [minimum + float(bi) / reciprocal_of_bin_size]
        return x_axis, Histogram

    # print the histogram contained in list_with_histogram in a tab separated file with bin_number value_of_histogram.
    # One can set write_mode to append 'a', in which case a double empty line is added before printing the histogram (gnuplot format)
    # CUMULATIVE add a third column corresponding to the cumulative distribution
    def PrintHistogram(
        self,
        filename,
        list_with_histogram=None,
        minimum=None,
        maximum=None,
        write_mode="w",
        HEADER=None,
        CUMULATIVE=True,
    ):
        if list_with_histogram == None:
            if self.histogram == []:
                self.histogram = self.CreateHistogram()
            list_with_histogram = self.histogram
        if minimum == None:
            minimum = self.Min
        if maximum == None:
            maximum = self.Max
        out = open(filename, write_mode)
        if write_mode == "a":
            out.write("\n\n")
        N = len(list_with_histogram)
        if type(HEADER) is str:
            out.write(HEADER + "\n")
        bin_size = float(maximum - minimum) / float(N)
        cum = 0.0  # for cumulative distribution
        for i in range(0, N):
            if CUMULATIVE:
                cum += list_with_histogram[i]
                out.write(
                    "%lf\t%lf\t%lf\n"
                    % ((minimum + float(i + 1) * bin_size), list_with_histogram[i], cum)
                )
            else:
                out.write(
                    "%lf\t%lf\n"
                    % ((minimum + float(i + 1) * bin_size), list_with_histogram[i])
                )
        del i, cum
        out.close()
        del N, out, bin_size
        return

    # print to a two column files, id entry
    def Print(
        self,
        filename=None,
        header=None,
        print_only_entries=False,
        print_stats_before_entries=True,
        delimiter="\t",
    ):
        if filename == None:
            if self.name == None:
                "statsdata.dat"
            else:
                filename = self.name.replace(" ", "_") + "_statsdata.dat"
        out = open(filename, "w")
        if header != None:
            out.write(header + "\n")
        if print_stats_before_entries:
            out.write("# average= " + str(self.average) + "\n")
            out.write("# stdev  = " + str(self.stdev) + "\n")
            out.write("# stdevN = " + str(self.stdevN) + "\n")
            out.write("# count  = " + str(self.count) + "\n")
        for i, e in enumerate(self.entries):
            if e != None:
                if print_only_entries:
                    line = str(e) + "\n"
                elif self.names != None:
                    line = (
                        str(self.names[i])
                        + delimiter
                        + str(self.ids[i])
                        + delimiter
                        + str(e)
                        + "\n"
                    )
                else:
                    line = str(self.ids[i]) + delimiter + str(e) + "\n"
                out.write(line)
        out.close()
        return


chimera_combine_model_script = """
# works assuming that models are loaded as 0.1,0.2,... and the new model will be model 1!!
import sys
import chimera
from collections import OrderedDict
# EXECUTE with: chimera --nogui --nostatus --script "this_script.py input_file.pdb [output_file.pdb]" (if output_file is not given the input_file is overwritten
Debug=False
if Debug : print sys.argv
input_pdb=sys.argv[1]#recall that the command is chimera --nogui this_script.py [files...]
if len(sys.argv)>2 : outfilename=sys.argv[2] # output figure name
else : outfilename=input_pdb # will OVERWRITE!! 


STANDARD_AA = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET','PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
OneToThree = {'X':'UNK','A': "ALA", 'C': "CYS", 'D': "ASP", 'E': "GLU", 'F': "PHE",'G': "GLY", 'H': "HIS", 'I': "ILE", 'K': "LYS", 'L': "LEU",'M': "MET", 'N': "ASN", 'P': "PRO", 'Q': "GLN", 'R': "ARG",'S': "SER", 'T': "THR", 'V': "VAL", 'W': "TRP", 'Y': "TYR" }
ThreeToOne = {'CYS':'C', 'ASP':'D', 'SER':'S', 'ASN':'N', 'GLN':'Q', 'LYS':'K', 'THR':'T', 'PRO':'P', 'HIS':'H', 'PHE':'F', 'ALA':'A', 'GLY':'G', 'ILE':'I', 'LEU':'L', 'ARG':'R', 'TRP':'W', 'VAL':'V', 'GLU':'E', 'TYR':'Y', 'MET':'M'}
def get_seqres_from_pdb(pdb_file,change_modres_name=True,modres_in_lower_case=False):
    old_chain=None
    seqres={}
    mod_residues={}    # dict of dict, first key is chain id and second key is residue id (SSSEQI) (aka resnumber) and content is modres name - residue name 
                    # e.g. missing_residues['A'][21]='DBB-THR' (so that [:3] is modres name and [-3:] is resname)
    modres={}       # a dictionary that contains the modified resname found as key and the original resname as value (e.g. {MSE:MET})
    seqadv_res={}
    seqres_problems=False
    for line in open(pdb_file) :
        if line[:6]=='SEQRES' :
            chain=line[11]
            if chain==' ' : 
                print ' **W** get_seqres_from_pdb() SEQRES problem, no chain id file %s ' % (pdb_file)
                seqres_problems=True
            if chain != old_chain : seqres[chain]=line[19:].split()    #save begginning of new chain
            else : seqres[chain] += line[19:].split()
            old_chain=chain
        elif line[:6]=='MODRES' :           
            if line[24:27]!='   ' : #if they know what should be there instead!
                resid, _ =convert_to_number(line[18:23].strip())
                if line[16] not in mod_residues : mod_residues[line[16]]={}
                mod_residues[line[16]][resid] =line[12:15]+'-'+line[24:27]
                if line[12:15] in modres and modres[line[12:15]]!=line[24:27] :
                    print ' **W** get_seqres_from_pdb() file %s chain %s MODRES CONFLICT! %s was %s now found %s-->%s at %s. Overwritten, all will be changed with latest found.' % (pdb_file,line[16],line[12:15],modres[line[12:15]],line[12:15],line[24:27],str(resid))
                modres[line[12:15]]=line[24:27]
        elif line[:6]=='SEQADV' and  line[12:15]!='   ' :
            tmp=line.split()
            if tmp[2] not in STANDARD_AA : # note that there are a lot of examples where tmp[2] is actually a standard residue (it is the resname in the ATOM) such as HIS tags. However we are only interested in those cases when this is not a standard residue.
                if line[16] not in seqadv_res :seqadv_res[line[16]]={}
                resid, _ =convert_to_number(line[18:23].strip())
                seqadv_res[line[16]][resid]=tmp[2]+'-'+line[39:42] # this includes also engineered mutations. However then residues are renamed only if not standard
                if line[12:15] not in modres and line[39:42]!='   ':
                    if line[12:15] not in ThreeToOne: # it is not a standard residue
                        modres[line[12:15]]=line[39:42] # this should also include engineered mutations when tmp[-1]=='ENGINEERED' (but if mutation is standard it does not go here)                
    # convert the SEQRES to one letter string and change the name of eventual modres
    if not seqres_problems and seqres!={} :
        for ch in seqres :
            seq=''
            for j,res in enumerate(seqres[ch]) :
                if (j==0 or j==len(seqres[ch])-1) and res in ['ACE','NME','NH2','NHH','MYR'] : continue # skip cappings (MYR is not a capping but when it binds at the termini some authors include it as a residue in seqres...)
                if res not in STANDARD_AA :
                    if change_modres_name:
                        if ch in mod_residues and res in modres : # if res pdb number plus insertion code (or space) is in modres
                            res= modres[res]
                            if modres_in_lower_case : res=res.lower()
                        elif ch in seqadv_res and res in modres : # if res pdb number plus insertion code (or space) is in modres
                            res= modres[res]
                            if modres_in_lower_case : res=res.lower()
                if res in ThreeToOne : seq += ThreeToOne[res]
                else : seq += 'X'
            seqres[ch]=seq
    return seqres

def seqres_to_line_list(seqres):
    #prints the seqres field of a pdb from a seqres dict
    #SEQRES   1 A  491  MET THR GLN VAL LEU VAL ARG ASN GLY ILE GLN ALA VAL
    line_list=[]
    for ch in seqres :
        j,i=1,0
        if type(seqres[ch]) is str : seqlist=[ OneToThree[a] for a in seqres[ch] ]# assume single letter notation
        else : seqlist=seqres[ch] # assumes three letter notation
        while i< len(seqlist) :
            line_list+=[ 'SEQRES%4d %s %4d  %s' %(j,ch,i+1,' '.join(seqlist[i:i+13])) ] 
            i+=13;j+=1
    return line_list

seqres= get_seqres_from_pdb(input_pdb)
opened = chimera.openModels.open(input_pdb) # opened is a list of opened model
existing_chains=[ mod.chain for mod in chimera.openModels.list(modelTypes=[chimera.Molecule])[0].sequences() ]
# opened[0].pdbHeaders['SEQRES'] # this is a list of all the seqres lines. but because of modres etc we open the file twice and we use our funciton to get a cleaned version of seqres
title=opened[0].pdbHeaders['TITLE']
chimera.runCommand('combine # close true log false') # run command that combines all models. 
#print dir(chimera.openModels.list(modelTypes=[chimera.Molecule])[0])
#chimera.openModels.list(modelTypes=[chimera.Molecule])[0].addPDBHeader('YooO','seqxx' ) # add a line to the pdb header, not sure what the k (first string is for)
new_chains=[ mod.chain for mod in chimera.openModels.list(modelTypes=[chimera.Molecule])[0].sequences() ]
if Debug :
    print 'existing_chains',existing_chains
    print 'new_chains',new_chains
new_seqres=OrderedDict()
for j,ch in enumerate(new_chains) :
    new_seqres[ch]=seqres[existing_chains[j%len(existing_chains)]][:] # assumes chimera opens and rename sequentially...
chimera.openModels.list(modelTypes=[chimera.Molecule])[0].setPDBHeader(u'TITLE',title )
chimera.openModels.list(modelTypes=[chimera.Molecule])[0].setPDBHeader(u'SEQRES',seqres_to_line_list(new_seqres) )
chimera.runCommand('write format pdb #1 '+outfilename)
"""


chimera_color_figure_script = """
import sys
import chimera
# EXECUTE with: chimera --nogui --nostatus --script "this_script.py input_file.pdb -a/-r" (only input_file.pdb argv is compulsory -a or -r color only atoms or ribbons respecively)
# NOTE a file color_dict.py must be present with the color_dict inside!!!!
Debug=False
if Debug : print sys.argv
input_pdb=sys.argv[1]#recall that the command is chimera --nogui this_script.py [files...]
color_py_file= sys.argv[2]
color_key='' # examples are r or a meaning ribbon or atoms (all atoms but not the ribbon)
if len(sys.argv)>3 : 
    outfilename=sys.argv[3] # output figure name
    if outfilename[-3:]!='.py' : outfilename+='.py'
else : outfilename=input_pdb[:-4]+'_chimera_color_figure.py'
if len(sys.argv)>4 : 
    color_key=','+sys.argv[4][-1] # last letter matters (-r or -a can be given)
    if color_key not in ['a','r'] : 
        sys.stderr.write("Warn chimera_color_figure_script --> color_key %s not recognized. setting to empty string\\n" % (color_key))
        color_key=''
from color_dict import color_dict # a file with .py must be printed with a dict in it that its named (as a variable) color_dict. Keys must be residue_pdb_id.chain_id
str_color_command_list=[]
global_color_command=''
for k in color_dict :
    if type(k) is not str : nk=':'+str(k)
    elif k[0]!=':' : nk=':'+k
    else : nk=k
    if type(color_dict[k]) is not str : 
        if hasattr(color_dict[k],'__len__') : nv=','.join(map(str,color_dict[k])) 
        else : nv=str(color_dict[k])
    else : nv=color_dict[k]
    if k in ['rest','global'] : 
        global_color_command='color '+nv+color_key
    else : str_color_command_list+=[ 'color '+nv+color_key+' '+nk ]
opened = chimera.openModels.open(input_pdb) # opened is a list of opened model
if Debug : 
    print 'opened: ',input_pdb
    print 'global_color_command:',global_color_command
if global_color_command!='' : chimera.runCommand(global_color_command) # color all of one color, so that missing ones are left like this
for com in str_color_command_list :
    chimera.runCommand(com) # run all the commands to color each residue.
#chimera.runCommand('focus '+str(opened[0]))
if '/' not in outfilename : outfilename='./'+outfilename # otherwise chimera yields the error "session directory does not exist"
chimera.runCommand('save '+outfilename)
sys.stdout.flush()
sys.stderr.flush()
"""

chimera_hb_script = """import chimera
import sys
# EXECUTE with: chimera --nogui --nostatus --script "this_script.py input_file.pdb -sel #0 -out tmp2_hb.txt -contacts contact_filename" (only input_file.pdb argv is compulsory)
Debug=False
print_contact_file=False
if Debug : print sys.argv
input_pdb=sys.argv[1]#recall that the command is chimera --nogui this_script.py [files...]
opened = chimera.openModels.open(input_pdb) # opened is a list of opened model
if Debug : print 'opened: ',input_pdb
selection=''
outfilename=input_pdb[:-4]+'_chimera_hbonds.txt'
for j,argv in enumerate(sys.argv[2:]) :
    if argv[:4]=='-sel' :
        chimera.runCommand('select '+sys.argv[2+j+1])
        selection=' selRestrict both' # both - H-bonds with both atoms selected
        if Debug : print 'selection:'+selection+' '+sys.argv[2+j+1]
    elif argv[:4]=='-out' :
        outfilename=sys.argv[2+j+1]
        if Debug : print 'outfilename:'+outfilename
    elif argv[:8]=='-contact' :
        print_contact_file=True
        if 2+j+1<len(sys.argv) and '-'!=sys.argv[2+j+1][0] : contact_filename=sys.argv[2+j+1]
        else : contact_filename=input_pdb[:-4]+'_chimera_contacts.txt'
        if Debug: print 'print_contact_file: '+contact_filename

chimera.runCommand('findhbond'+selection+' makePseudobonds False namingStyle simple saveFile '+outfilename) # namingStyle simple - residue name, residue specifier, and atom name (for example, HIS 16.A ND1)
if print_contact_file : 
    if selection!='' : chimera.runCommand('findclash selection test self overlapCutoff -0.4 hbondAllowance 0.0 interSubmodel False makePseudobonds False namingStyle simple saveFile '+contact_filename)
    else : chimera.runCommand('findclash # test self overlapCutoff -0.4 hbondAllowance 0.0 interSubmodel False makePseudobonds False namingStyle simple saveFile '+contact_filename) # default to all model. 
sys.stdout.flush()
sys.stderr.flush()
"""


# amino acid class
# should include standard: pdb_id=int overall_id=int resname=''
# others: ok=bol chain='A' contacts=<overall_id 'list'> zsurf_alone= rel_sasa_alone= coord=array( CA_coordinates) exposure_weight= rel_sasa= sasa= sasa_alone=  exposure_weight_alone= intrinsic_score= res1letter='Y' zsurf=
class Residue(Struct):
    resname = ""
    pdb_id = None
    overall_id = None
    chain_id = ""
    residue = None  # pointer to corresponding biopython residue class
    color = None  # can be used to make figures

    def __repr__(
        self,
    ):  # used when class is represented (i.e. when in python you press enter!)
        tmp_str = ""
        for attr, value in self.__dict__.items():
            if type(value) is list or type(value) is tuple:
                if len(value) <= 5:
                    tmp_str += str(attr) + "=" + str(value) + "\n"
                else:
                    tmp_str += (
                        str(attr)
                        + "="
                        + str(type(value))
                        + ",len="
                        + str(len(value))
                        + " =="
                        + str(value[5:])
                        + "...\n"
                    )
            elif type(value) is dict:
                tmp_str += (
                    str(attr)
                    + "="
                    + str(type(value))
                    + ",len="
                    + str(len(value))
                    + " items="
                    + str(list(value.items())[5:])
                    + "...\n"
                )
            else:
                tmp_str += str(attr) + "=" + repr(value) + " "
        return tmp_str[:-1]

    def __str__(self):  # used when class is printed
        tmp_str = ""
        for attr, value in self.__dict__.items():
            if type(value) is list or type(value) is tuple:
                if len(value) <= 5:
                    tmp_str += str(attr) + "=" + str(value) + "\n"
                else:
                    tmp_str += (
                        str(attr)
                        + "="
                        + str(type(value))
                        + ",len="
                        + str(len(value))
                        + " =="
                        + str(value[5:])
                        + "...\n"
                    )
            elif type(value) is dict:
                tmp_str += (
                    str(attr)
                    + "="
                    + str(type(value))
                    + ",len="
                    + str(len(value))
                    + " items="
                    + str(list(value.items())[5:])
                    + "...\n"
                )
            else:
                tmp_str += str(attr) + "=" + repr(value) + "\n"
        return tmp_str[:-1]

    # retrieve the value of a variable by calling it as a string. One can do residue.retrieve('variable')
    # so one can use if 'variable' in dir(residues) : and staff like that...
    def retrieve(self, variable_name_str, print_warnings=True, force_number=False):
        # print_warnings useful when embedded in try:
        if variable_name_str not in dir(self):
            if print_warnings:
                sys.stderr.write(
                    "**ERROR** in method retrieve, %s not defined in class\n    %s\n"
                    % (variable_name_str, str(dir(self)))
                )
            raise AttributeError
        if force_number:
            a = self.__dict__[variable_name_str]
            if a is None:
                return numpy.nan
            elif type(a) is str:
                try:
                    return int(a)
                except Exception:
                    pass
                try:
                    return float(a)
                except Exception:
                    pass
            return a
        return self.__dict__[variable_name_str]

    def update(self, variable_name_str, new_variable_value, check=False):
        """
        # used to change the value of a variable by calling it as a str (rather than residue.variable=new_value we do residue.update('variable',new_value))
        # if check is False it can also be used to add a completely new variable with the name  variable_name_str
        """
        if check and variable_name_str not in dir(self):
            sys.stderr.write(
                "**ERROR** in method update, %s not defined in class\n    %s\n"
                % (variable_name_str, str(dir(self)))
            )
            raise AttributeError
        self.__dict__[variable_name_str] = new_variable_value

    # works only with crosslinked structure where residue is actually the correct biopython object
    # if get_as_sum==False it return the average
    # if add_variable_name it adds the computed property to the class as a variable named property
    # (if add_variable_name is str the variable is named in that way)
    def get_atom_property(
        self, property="bfactor", get_as_sum=True, add_variable_name=True
    ):
        summ = 0.0
        if property == "bfactor":
            for atom in self.residue:
                summ += atom.bfactor
        elif property == "occupancy":
            for atom in self.residue:
                summ += atom.occupancy
        else:
            sys.stderr.write(
                "ERROR in get_atom_property() within Residue class, property %s not found or not implemented!!\n Residue: %s\n"
                % (str(property), self.__repr__())
            )
            return -999.0
        if not get_as_sum:
            summ /= float(len(self.residue))  # get average
        if add_variable_name == True:
            self.update(property, summ, check=False)
        elif type(add_variable_name) is str:
            self.update(add_variable_name, summ, check=False)
        return summ


# Polymer class
# Should include standard variables: kind='', seq=<'list' of Residue()>, chain_id=''
# other possibles:  instrinsic_score=float instrinsic_score_profile=  sequence1='AQS...'
class Polymer(Struct):
    chain_id = ""
    kind = ""
    aa_pdb_id_map = (
        {}
    )  # map to get amino acid index in seq knowing only the pdb_id -> seq_index=aa_pdb_id_map[pdb_id]
    seq = []  # list of Residue classes
    chain = None  # pointer to corresponding biopython residue class
    default_for_non_ok_residues = numpy.nan
    seqres = None

    def __repr__(
        self,
    ):  # used when class is represented (i.e. when in python you press enter!)
        tmp_str = ""
        for attr, value in self.__dict__.items():
            if type(value) is list or type(value) is dict:
                tmp_str += (
                    str(attr) + "=" + str(type(value)) + ",len=" + str(len(value)) + " "
                )
            else:
                tmp_str += str(attr) + "=" + repr(value) + " "
        return tmp_str[:-1]

    def __str__(self):  # used when class is printed
        tmp_str = ""
        for attr, value in self.__dict__.items():
            if type(value) is list or type(value) is dict:
                tmp_str += (
                    str(attr)
                    + "="
                    + str(type(value))
                    + ",len="
                    + str(len(value))
                    + "\n"
                )
            else:
                tmp_str += str(attr) + "=" + repr(value) + "\n"
        return tmp_str[:-1]

    def retrieve(self, variable_name_str, print_warnings=True):
        # print_warnings useful when embedded in try:
        if variable_name_str not in dir(self):
            if print_warnings:
                sys.stderr.write(
                    "**ERROR** in method retrieve, %s not defined in class\n    %s\n"
                    % variable_name_str,
                    str(dir(self)),
                )
            raise AttributeError
        return self.__dict__[variable_name_str]

    # update the value of an existing variable or create a new variable with name variable_name_str (if check is False)
    # check controls if the variable already exists and raise an exception if it doens't
    def update(self, variable_name_str, new_variable_value, check=False):
        if check and variable_name_str not in dir(self):
            sys.stderr.write(
                "**ERROR** in method update, %s not defined in class\n    %s\n"
                % (variable_name_str, str(dir(self)))
            )
            raise AttributeError
        self.__dict__[variable_name_str] = new_variable_value

    def profile(
        self,
        variable_name_str,
        only_ok_residues=True,
        default_for_non_ok_residues=None,
        print_retrieve_warnings=True,
    ):
        """
        # extract the profile corresponding to the property variable_name_str from the .seq of polymer[ch_id].
        # it returns profile
        if default_for_non_ok_residues is None then nothing is added and the profile is only of ok
        print_retrieve_warnings useful to suppress error prints when embedded in try: (otherwise exception is raised anyway)
        """
        if default_for_non_ok_residues is None:
            default_for_non_ok_residues = (
                self.default_for_non_ok_residues
            )  # typically numpy.nan but can be changed to allow cahnges in global plotting
        profile = []
        for res in self.seq:
            if only_ok_residues:
                if res.ok:
                    x = res.retrieve(
                        variable_name_str,
                        print_warnings=print_retrieve_warnings,
                        force_number=True,
                    )
                    if type(x) is str:
                        x = float(x)
                    profile += [x]
                elif default_for_non_ok_residues is not None:
                    if variable_name_str in dir(res):
                        profile += [
                            res.retrieve(
                                variable_name_str,
                                print_warnings=print_retrieve_warnings,
                                force_number=True,
                            )
                        ]
                    else:
                        profile += [default_for_non_ok_residues]
            else:
                profile += [
                    res.retrieve(
                        variable_name_str,
                        print_warnings=print_retrieve_warnings,
                        force_number=True,
                    )
                ]
        return profile

    def profile_of_sequence(
        self,
        variable_name_str,
        sequence,
        missing_residue_value=None,
        mismatch_residue_value=False,
        full_return=False,
        force_number=False,
    ):
        """
        extract the profile corresponding to the property variable_name_str from the .seq of polymer[ch_id] after alinging the latter with sequence 
        (useful if using full sequence in pdb with missing residues or similar situations) 
        if mismatch_residue_value is False it uses the value found in corresponding seq residue even though it differs from the residue in sequence
        it returns profile
        """
        profile = []
        gap_symbol = "-"
        if sequence == None:
            sequence = (
                self.sequence
            )  # waist of cpu because alignment is done anyway, in this case the function profile() should be used instead
        # align sequences
        if sequence != None:
            (
                seq1_aligned,
                seq2_aligned,
                score,
                identity_percentage,
                alignment_end,
            ) = mybio.pairwise_alignment(
                sequence,
                self.sequence,
                calculate_identity_percentage=True,
                one_alignment_only=True,
            )[
                0
            ]
            # CAN fix speed to avoid getting identity percentage here and computing it from matches_on_seq in the for loop below
            if identity_percentage < 85:
                sys.stderr.write(
                    "WARNING in profile_of_sequence() identity_percentage=%lf !!\n"
                    % (identity_percentage)
                )
                sys.stderr.write(
                    mybio.print_pairwise(
                        (
                            seq1_aligned,
                            seq2_aligned,
                            score,
                            identity_percentage,
                            alignment_end,
                        )
                    )
                )
                sys.stderr.flush()
            j, i = 0, 0
            start_pol_on_seq = None
            matches_on_seq = 0
            for ji, aa in enumerate(seq2_aligned):  # loop on aligned polymer seq
                if aa != gap_symbol:  # no gap in polymer seq
                    if start_pol_on_seq == None:
                        start_pol_on_seq = i  # if the other sequence starts with a gap it will still be zero the start_pol
                    if aa == seq1_aligned[ji]:
                        profile += [
                            self.seq[j].retrieve(
                                variable_name_str, force_number=force_number
                            )
                        ]
                        end_pol_on_seq = i + 1
                        matches_on_seq += 1
                        # self.seq[j].update( new_variable_name, seq_profile[i],check=False)

                    else:  # there is no gap but the two sequences are different...
                        if seq1_aligned[ji] == gap_symbol:
                            continue
                        if mismatch_residue_value == False:
                            profile += [
                                self.seq[j].retrieve(
                                    variable_name_str, force_number=force_number
                                )
                            ]
                            end_pol_on_seq = (
                                i + 1
                            )  # if mismatch_residue_value==False end_pol_on_seq will be updated including possible mismatches at the end.
                        else:
                            profile += [mismatch_residue_value]
                    j += 1
                elif seq1_aligned[ji] != gap_symbol:
                    profile += [missing_residue_value]
                if seq1_aligned[ji] != gap_symbol:
                    i += 1
        if len(profile) != len(sequence):
            raise Exception(
                "error in polymer.profile_of_sequence() len(weights)!=len(full_seq) %d %d\n%s\n"
                % (
                    len(profile),
                    len(sequence),
                    mybio.print_pairwise(
                        (
                            seq1_aligned,
                            seq2_aligned,
                            score,
                            identity_percentage,
                            alignment_end,
                        )
                    ),
                )
            )
        if full_return:
            covered_positions_on_seq = end_pol_on_seq - start_pol_on_seq
            return (
                profile,
                identity_percentage,
                seq1_aligned,
                seq2_aligned,
                start_pol_on_seq,
                end_pol_on_seq,
                covered_positions_on_seq,
                matches_on_seq,
            )
        return profile

    def cast_seq_info(
        self,
        seq_profile,
        sequence=None,
        new_variable_name="new",
        missing_aa_value=None,
        cast_color=True,
        max_or_fun_to_normalize_color=None,
        print_alignment=False,
    ):
        """
        can be used to add results (stored in seq_profile) of a sequence based prediction
        sequence can be given to verify the pairwise alignment (maybe in polymer there are missing crystal residues or such)
        color_rgb_list can be a list of rgb tuples, default is color_rgb_list
        """
        gap_symbol = "-"
        cast_color_list = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1.0, 1.0, 0.0)]
        if type(cast_color) is list or type(cast_color) is tuple:
            cast_color_list = cast_color
            if type(cast_color_list) is tuple:
                cast_color_list = [cast_color_list]  # list of rgb tuples
            cast_color = True
        if cast_color:
            color_list = profile_to_color(
                seq_profile,
                maximum_to_normalize=max_or_fun_to_normalize_color,
                color_rgb_list=cast_color_list,
            )
        if sequence != None:
            if len(sequence) != len(seq_profile):
                print(
                    "**ERROR** in cast_seq_info len(sequence)!=len(seq_profile) %d and %d"
                    % (len(sequence), len(seq_profile))
                )
            (
                seq1_aligned,
                seq2_aligned,
                score,
                identity_percentage,
                alignment_end,
            ) = mybio.pairwise_alignment(
                sequence,
                self.sequence,
                calculate_identity_percentage=True,
                one_alignment_only=True,
                local=False,
            )[
                0
            ]
            if identity_percentage < 80:
                sys.stderr.write(
                    "WARNING in cast_seq_info() identity_percentage=%lf !!\n"
                    % (identity_percentage)
                )
                sys.stderr.write(
                    mybio.print_pairwise(
                        (
                            seq1_aligned,
                            seq2_aligned,
                            score,
                            identity_percentage,
                            alignment_end,
                        )
                    )
                )
            elif print_alignment:
                sys.stdout.write(
                    mybio.print_pairwise(
                        (
                            seq1_aligned,
                            seq2_aligned,
                            score,
                            identity_percentage,
                            alignment_end,
                        )
                    )
                )
            j, i = 0, 0
            for ji, aa in enumerate(
                seq2_aligned
            ):  # loop on aligned sequence from polymer
                if aa != gap_symbol:
                    if aa == seq1_aligned[ji]:
                        self.seq[j].update(
                            new_variable_name, seq_profile[i], check=False
                        )
                        if cast_color:
                            self.seq[j].color = color_list[i]
                    elif seq1_aligned[ji] != gap_symbol:
                        self.seq[j].update(
                            new_variable_name, seq_profile[i], check=False
                        )
                        if cast_color:
                            self.seq[j].color = color_list[
                                i
                            ]  # give color anyway (even if amino acid is mistmatched)
                    elif cast_color:
                        self.seq[j].color = "gray"  # a gap in the input sequence
                    j += 1
                if (
                    seq1_aligned[ji] != gap_symbol
                ):  # rise the counter of the input sequence
                    i += 1
        else:  # update without aligning...
            if len(self.seq) != len(seq_profile):
                sys.stderr.write(
                    "WARNING in cast_seq_info() len(self.seq)!=len(seq_profile) %d %d\n"
                    % (len(self.seq), len(seq_profile))
                )
            for j, r in enumerate(self.seq):
                self.seq[j].update(new_variable_name, seq_profile[j], check=False)
                if cast_color:
                    self.seq[j].color = color_list[j]
        return


def profile_to_color(
    profile_or_matrix,
    maximum_to_normalize=None,
    color_rgb_list=[(0, 0, 1), (0, 1, 0), (1, 0, 0), (1.0, 1.0, 0.0)],
    last_to_alpha_if_more_than_3=True,
):
    """
    auxiliary function used to convert a profile into RGB colors.
    maximum_to_normalize can be a number (by which each element gets divided)
    or a function that is applied to each element
      an example for counts can be tanh(x/2.) so that elements with x=1 will become about 0.5 and a plateau at one (full color) is reached quickly (at x=4-5)
    """
    try:
        import chroma
    except ImportError:
        sys.stderr.write(
            "\n**WARNING** chroma MODULE NOT AVAILABLE. Will not be able run profile_to_color\n  in terminal try running 'pip install chroma' (may also need to navigate in module folder and run '2to3 -w core.py' for python3 usage)\n"
        )

        class chroma:
            def Color(self, *args, **kwargs):
                raise Exception(
                    "chroma MODULE NOT AVAILABLE. Cannot run profile_to_color\n in terminal try running 'pip install chroma'  (may also need to navigate in module folder and run '2to3 -w core.py' for python3 usage)\n"
                )

        pass
    if maximum_to_normalize == None:
        maximum_to_normalize = max(profile_or_matrix)
        while hasattr(maximum_to_normalize, "__len__"):
            maximum_to_normalize = max(maximum_to_normalize)
    starting_cols = [chroma.Color(c, format="RGB") for c in color_rgb_list]
    # print starting_cols
    out_color_list = []
    profile_or_matrix = numpy.array(profile_or_matrix)
    for mat in profile_or_matrix:
        if hasattr(maximum_to_normalize, "__call__"):
            mat = maximum_to_normalize(mat)
        else:
            mat /= 1.0 * maximum_to_normalize
        if hasattr(mat, "__len__"):
            rc = None
            for j, x in enumerate(mat):
                if (
                    j > 2 and last_to_alpha_if_more_than_3 and j == len(mat) - 1
                ):  # we add an alpha channel rather than mixing a new color (last iteration)
                    rc = chroma.Color(rc.rgb + ((1 - x),), format="RGB")
                    break  # last iteration in this for loop
                nc = list(starting_cols[j].hsv)
                nc[1] = x
                # print nc,starting_cols[j].hsv,res_out
                if not hasattr(rc, "rgb"):
                    rc = chroma.Color(nc, format="HSV")
                else:
                    rc -= chroma.Color(nc, format="HSV")
        else:
            nc = list(starting_cols[0].hsv)
            nc[1] = mat
            rc = chroma.Color(nc, format="HSV")
        out_color_list += [rc.rgb]
    return out_color_list


class Model(OrderedDict):
    """
    an Ordered Dict of Polymer classes, each being a pdb chain
     isinstance with OrderedDict will return True
    """

    pdbfile = None
    pdb_id = None
    structure = None
    overall_map = (
        None  # list that goes from overall_id to Residue() wherever the latter is!
    )
    default_for_non_ok_residues = numpy.nan

    def __repr__(self):
        tmpstr = "PDB: " + str(self.pdbfile) + "\n"
        tmpstr += str(OrderedDict.__repr__(self))
        return tmpstr

    def __str__(self):
        tmpstr = "PDB: " + str(self.pdbfile) + "\n"
        tmpstr += OrderedDict.__str__(self)
        return tmpstr

    def profile(
        self,
        variable_name_str,
        only_ok_residues=True,
        default_for_non_ok_residues=None,
        print_retrieve_warnings=True,
    ):
        """
        # extract the profile corresponding to the property variable_name_str from the .seq of polymer[ch_id].
        # it returns profile corresponding to all chains joined together in the order they are saved in the OrderedDict
        as everything is sorted it will correspond to the overall_id to the residues 
        if default_for_non_ok_residues is None then nothing is added and the profile is only of ok
        print_retrieve_warnings useful to suppress error prints when embedded in try: (otherwise exception is raised anyway)
        """
        if default_for_non_ok_residues is None:
            default_for_non_ok_residues = (
                self.default_for_non_ok_residues
            )  # typically numpy.nan but can be changed to allow cahnges in global plotting
        profile = []
        for ch in self:
            profile += self[ch].profile(
                variable_name_str,
                only_ok_residues=only_ok_residues,
                default_for_non_ok_residues=default_for_non_ok_residues,
                print_retrieve_warnings=print_retrieve_warnings,
            )
        return profile

    def chimera_color_figure(
        self,
        pdbfile=None,
        outfilename=None,
        color_only="",
        global_color=None,
        chimera_script=chimera_color_figure_script,
        tmpscript_file_no_extentions="chimera_color_figure",
        delete_tmp=True,
    ):
        """
        pdbfile is the template of the colored figure. It must be the one loaded in Model() (if left to None this is done automatically!)
          it can also be an existing chimera figure, proviede that it corresponds to the pdb file loaded in Model.
        color_only can be either -r (color only ribbon) or -a (color only atoms). Or '' color both.
        EXAMPLE
import structs,s2D_class
out=s2D_class.read_output('s2D_out.txt')
pol,s,ovmap=structs.pdb_to_polymer('4GIZ.pdb')
pol['A'].cast_seq_info(out[2][0],out[1][0],max_or_fun_to_normalize_color=1,print_alignment=True)
pol.chimera_color_figure(outfilename=None)
#Example 2
pol,s,ovmap=structs.pdb_to_polymer('4GIZ.pdb')
pol['C'].cast_seq_info(e6prof,e6seq,'s2D',max_or_fun_to_normalize_color=1,print_alignment=True) # in this example e6prof,e6seq are the s2D profile (matrix (seqlen,4)) of the s2D result of the uniprot sequence (E6) of chain C in the pdb file!
pol.chimera_color_figure(outfilename=None,global_color=None)
# if your profile is a count can do
tanh = lambda x: numpy.tanh(x/2.)
pol['C'].cast_seq_info( count_profile.T  ,e6seq,'count_profile',max_or_fun_to_normalize_color=tanh  ,cast_color=[ (1,0,0),(1,1,0),(1,0,1) ]) # magenta and yellow
pol.chimera_color_figure('4GIZ_chimera_color_figure.py','4GIZ_chimera_color_figure_atoms_are_good_complementaries.py',color_only='-a')
        """
        if pdbfile == None:
            pdbfile = self.pdbfile
        if color_only != "" and color_only[-1] != "r" and color_only[-1] != "a":
            raise ValueError(
                "Unsupported color_only %s [supported '' '-a' '-r']" % (str(color_only))
            )
        if outfilename == None:
            outfilename = ""
        # first create dict with chain whose residues have a color attribute. Save resnumber and color
        color_dict = {}
        if global_color is not None:
            color_dict["global"] = global_color
        for ch in self:
            for res in self[ch].seq:
                if hasattr(res, "color") and res.color != None:
                    color_dict[":" + str(res.pdb_id) + "." + ch] = res.color
        out = open("color_dict.py", "w")
        out.write("color_dict=" + str(color_dict) + "\n")
        out.close()
        out = open(tmpscript_file_no_extentions + ".py", "w")
        out.write(chimera_script)
        out.close()
        # cwd=os.getcwd()
        os.system(
            'chimera --nogui --nostatus --script "%s %s  color_dict.py %s %s"'
            % (tmpscript_file_no_extentions + ".py", pdbfile, outfilename, color_only)
        )
        if delete_tmp:
            os.system(
                "rm -f " + tmpscript_file_no_extentions + ".py color_dict.py*"
            )  # removes .pyc as well
        return outfilename


def convert_to_number(
    string,
    force_float=False,
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
    converted_string,bool. Bool is True if the sting has been converted, False if the  string is still in string 
     format.
    the function is quite slow
    """
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
            return float(string), True
        except ValueError:
            return string, False


def remove_solvent(structure, key_to_remove="W", remove_everything_but_standard=False):
    """
    # given a pdb structure it removes all solvent residues/atom. 
    # if remove_everything_but_standard it removes everything that has res_id[0] different from ' ' 
    (empty space in biopython denotes ATOM entry in pdb)
    """
    # remove heteroatoms
    res_to_remove = []
    chain_to_remove = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if remove_everything_but_standard and residue.id[0] != " ":
                    #                    print (residue.id,residue.resname,chain.id)
                    res_to_remove += [
                        residue.id
                    ]  # if I remove it here then the loop don't go back so you remove every other residue
                elif residue.id[0] == key_to_remove:
                    res_to_remove += [residue.id]
            for el in res_to_remove:
                chain.detach_child(el)
            res_to_remove = []
            if len(chain) == 0:
                chain_to_remove += [chain.id]
        for ch in chain_to_remove:
            model.detach_child(ch)
        chain_to_remove = []
    del chain_to_remove, res_to_remove
    return structure


def pdb_to_structure(pdbfilename, quiet=False, long_factors=False, structure_id=""):
    parser = PDBParser_mod(QUIET=quiet)
    if structure_id == "":
        structure_id = pdbfilename.split("/")[-1].split(".")[0]
    if long_factors:
        try:
            structure = parser.get_structure(
                structure_id, pdbfilename, long_factors=long_factors
            )
        except Exception:
            sys.stderr.write("**pdb_to_structure cannot get long_factors\n")
            structure = parser.get_structure(structure_id, pdbfilename)
            pass
    else:
        structure = parser.get_structure(structure_id, pdbfilename)
    del parser
    return structure


STANDARD_AA = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
amino_list1 = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
OneToThree = {
    "X": "UNK",
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}
ThreeToOne = {
    "UNK": "X",
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "ASN": "N",
    "GLN": "Q",
    "LYS": "K",
    "THR": "T",
    "PRO": "P",
    "HIS": "H",
    "PHE": "F",
    "ALA": "A",
    "GLY": "G",
    "ILE": "I",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}


def get_seqres_from_pdb(pdb_file, change_modres_name=True, modres_in_lower_case=False):
    old_chain = None
    seqres = {}
    mod_residues = (
        {}
    )  # dict of dict, first key is chain id and second key is residue id (SSSEQI) (aka resnumber) and content is modres name - residue name
    # e.g. missing_residues['A'][21]='DBB-THR' (so that [:3] is modres name and [-3:] is resname)
    modres = (
        {}
    )  # a dictionary that contains the modified resname found as key and the original resname as value (e.g. {MSE:MET})
    seqadv_res = {}
    seqres_problems = False
    for line in open(pdb_file):
        if line[:6] == "SEQRES":
            chain = line[11]
            if chain == " ":
                print(
                    " **W** get_seqres_from_pdb() SEQRES problem, no chain id file %s "
                    % (pdb_file)
                )
                seqres_problems = True
            if chain != old_chain:
                seqres[chain] = line[19:].split()  # save begginning of new chain
            else:
                seqres[chain] += line[19:].split()
            old_chain = chain
        elif line[:6] == "MODRES":
            if line[24:27] != "   ":  # if they know what should be there instead!
                resid, _ = convert_to_number(line[18:23].strip())
                if line[16] not in mod_residues:
                    mod_residues[line[16]] = {}
                mod_residues[line[16]][resid] = line[12:15] + "-" + line[24:27]
                if line[12:15] in modres and modres[line[12:15]] != line[24:27]:
                    print(
                        " **W** get_seqres_from_pdb() file %s chain %s MODRES CONFLICT! %s was %s now found %s-->%s at %s. Overwritten, all will be changed with latest found."
                        % (
                            pdb_file,
                            line[16],
                            line[12:15],
                            modres[line[12:15]],
                            line[12:15],
                            line[24:27],
                            str(resid),
                        )
                    )
                modres[line[12:15]] = line[24:27]
        elif line[:6] == "SEQADV" and line[12:15] != "   ":
            tmp = line.split()
            if (
                tmp[2] not in STANDARD_AA
            ):  # note that there are a lot of examples where tmp[2] is actually a standard residue (it is the resname in the ATOM) such as HIS tags. However we are only interested in those cases when this is not a standard residue.
                if line[16] not in seqadv_res:
                    seqadv_res[line[16]] = {}
                resid, _ = convert_to_number(line[18:23].strip())
                seqadv_res[line[16]][resid] = (
                    tmp[2] + "-" + line[39:42]
                )  # this includes also engineered mutations. However then residues are renamed only if not standard
                if line[12:15] not in modres and line[39:42] != "   ":
                    if line[12:15] not in ThreeToOne:  # it is not a standard residue
                        modres[line[12:15]] = line[
                            39:42
                        ]  # this should also include engineered mutations when tmp[-1]=='ENGINEERED' (but if mutation is standard it does not go here)
    # convert the SEQRES to one letter string and change the name of eventual modres
    if not seqres_problems and seqres != {}:
        for ch in seqres:
            seq = ""
            for j, res in enumerate(seqres[ch]):
                if (j == 0 or j == len(seqres[ch]) - 1) and res in [
                    "ACE",
                    "NME",
                    "NH2",
                    "NHH",
                    "MYR",
                ]:
                    continue  # skip cappings (MYR is not a capping but when it binds at the termini some authors include it as a residue in seqres...)
                if res not in STANDARD_AA:
                    if change_modres_name:
                        if (
                            ch in mod_residues and res in modres
                        ):  # if res pdb number plus insertion code (or space) is in modres
                            res = modres[res]
                            if modres_in_lower_case:
                                res = res.lower()
                        elif (
                            ch in seqadv_res and res in modres
                        ):  # if res pdb number plus insertion code (or space) is in modres
                            res = modres[res]
                            if modres_in_lower_case:
                                res = res.lower()
                if res in ThreeToOne:
                    seq += ThreeToOne[res]
                else:
                    seq += "X"
            seqres[ch] = seq
    return seqres


def atom_type(atom_name):
    """
    from the pdb atom name it returns the atom 
     type. 
    The latter is one of ['C','N','O','H','S','?']
    """
    if atom_name[0] == "C":
        return "C"
    if atom_name[0] == "N":
        return "N"
    if atom_name[0] == "O":
        return "O"
    if atom_name[0] == "H":
        return "H"
    if atom_name[0].isdigit() and atom_name[1] == "H":
        return "H"
    if atom_name[0] == "S":
        return "S"
    return "?"


def three_to_one(amino3letter, quiet=True, unk="X"):
    if amino3letter in ThreeToOne:
        return ThreeToOne[amino3letter]
    else:
        if not quiet:
            sys.stderr.write(
                "**WARN** in amino3letter() amino acid %s not standard. Set to %s\n"
                % (amino3letter, unk)
            )
        return unk


def seqres_to_line_list(seqres):
    """
    prints the seqres field of a pdb from a seqres dict
    SEQRES   1 A  491  MET THR GLN VAL LEU VAL ARG ASN GLY ILE GLN ALA VAL
    """
    line_list = []
    for ch in seqres:
        j, i = 1, 0
        if type(seqres[ch]) is str:
            seqlist = [
                OneToThree[a] for a in seqres[ch]
            ]  # assume single letter notation
        else:
            seqlist = seqres[ch]  # assumes three letter notation
        while i < len(seqlist):
            line_list += [
                "SEQRES%4d %s %4d  %s" % (j, ch, i + 1, " ".join(seqlist[i : i + 13]))
            ]
            i += 13
            j += 1
    return line_list


def pdb_to_polymer(
    pdb_file_name,
    crosslink=True,
    dict_of_res_variable_to_initialize=None,
    long_factors=False,
    add_seqres=False,
    seqres_to_seq=True,
    quiet=False,
):
    """
    this function reads a pdb_file and 
    return polymer,structure,overall_map
     that are a polymer dictionary, a biopython structure and a list of Residues 
      which maps the overall_id (0 to total number of residues in file) to the Residues() 
    if crosslink==True than a pointer to the biopython chain class (in structure[0]) is saved in each polymer and 
    another pointer to the biopython residue class is saved in each Residue() 
    Chain ids are the keys the values are Polymer() structs containing .seq which is a list of Residue() structs 
    if pdb_file_name is already a Bio.PDB.Structure.Structure then the the structure is converted to polymer with no 
    need of reading a file
    add_seqres adds for each chain the corresponding SEQRES sequence (in .seqres variable) with amino acids without atomic coordinates in lower case.
     it also adds a variable atom_seq_indices that corresponds to the indices within the seqres varaible of those residues found in the ATOM sequence (thus with at least some atomic coordinates).
     seqres_to_seq is only used when add_seqres, and if True adds the missing residues (those in seqres but not in ATOM seq) as non-ok residues to the .seq class.
    add_seqres can also be a string (a single sequence) or a dict with chain ids and various sequences as input to use a custom seqres. Giving it this way will match possible gaps 
     with an alignment with the ATOM sequence in the pdb file at places where violations in the peptide bond distance are found.
    """
    if type(pdb_file_name) is str:
        pdb_id = pdb_file_name.split("/")[-1].split(".")[0]
        if not os.path.isfile(pdb_file_name):
            sys.stderr.write(
                "\n**ERROR** in pdb_to_polymer() can't find file %s\n" % (pdb_file_name)
            )
        structure = pdb_to_structure(
            pdb_file_name, quiet=quiet, long_factors=long_factors, structure_id=pdb_id
        )
        input_seqres = None
        if type(add_seqres) is str or hasattr(
            add_seqres, "keys"
        ):  # string, dict or ordered dict containing custom seqres sequences
            input_seqres = add_seqres
            add_seqres = True
        if add_seqres:
            expected_atomseq_indices = {}
            (
                polymer_seq_resid_lists,
                (gap_map, atom_seq_blocks_indices, seqres_aligned),
                atom_seq,
                cluster_list,
                missing_residues,
                mod_residues,
                seqres,
                deletions,
                insertions,
                tags,
                flags,
                chains_raising_flags,
            ) = mybio.pull_sequences_from_pdb(
                pdb_file_name,
                input_seqres=input_seqres,
                change_modres_name=True,
                remove_tags_from_polymer=False,
                remove_insertions_from_polymer=False,
                modres_in_lower_case=False,
                include_missing_res_in_lower_case=True,
                consider_only_declared_MR=False,
                return_gapless_sequence=False,
                ALIGN_CHAINS=False,
                print_flags=True,
            )
            # seqres=get_seqres_from_pdb(pdb_file_name)
    else:
        structure = pdb_file_name
        pdb_file_name = pdb_id = str(structure.id)
        add_seqres = False
    if crosslink:
        structure = remove_solvent(
            structure, key_to_remove="W", remove_everything_but_standard=False
        )
    str_id = 0
    is_multi_model = False
    good_models = [0]
    if len(structure) > 0:  # multi model file
        str_id = 0
        is_multi_model = True
        good_models = list(range(0, len(structure)))
        while (
            len(structure[str_id]) == 0
        ):  # deal with the fact that some models could be empty (as we have removed them...)
            str_id += 1
            if str_id >= len(structure):
                sys.stderr.write(
                    "**ERROR** in pdb_to_polymer() the pdb %s seems empty!!\n"
                    % (pdb_file_name)
                )
                break
            good_models.remove(str_id - 1)  # remove empty model
    # fill first part of residue class
    polymer = (
        Model()
    )  # the sequence is stored as a list (named seq) of Residue() classes inside a Polymer class. each dictionary entry is a Polymer struct corresponding to a chain
    polymer.pdbfile = os.path.abspath(pdb_file_name)
    polymer.pdb_id = pdb_id
    i = 0
    overall_id = 0
    overall_map = (
        []
    )  # list that goes from overall_id to Residue() wherever the latter is!
    for chain in structure[str_id]:
        if chain.id not in polymer:
            polymer[chain.id] = Polymer(
                chain_id=chain.id, kind="protein"
            )  # initialize to protein
            polymer[chain.id].seq = []
            polymer[chain.id].aa_pdb_id_map = {}  # init it
            polymer[chain.id].sequence = ""
            polymer[chain.id].sequenceOK = ""
            i = 0
            if crosslink:
                polymer[chain.id].chain = chain
        iteron = chain
        if add_seqres and seqres_to_seq:
            reslist = [res for res in chain if res.id[0] != "W"]  # water
            iteron = polymer_seq_resid_lists[chain.id][0].replace(
                "-", ""
            )  # will be seqres with those with missing coordinates in lower case
            gi = 0  # good residues index (those with coordinates)
            expected_atomseq_indices[chain.id] = []
        for j, res in enumerate(iteron):
            if add_seqres and seqres_to_seq:
                if res.islower():  # missing from ATOM seq - add to seq as non-ok
                    if res.upper() in OneToThree:
                        nam = OneToThree[res.upper()]
                    else:
                        nam = res
                    polymer[chain.id].seq += [
                        Residue(
                            resname=nam,
                            pdb_id=numpy.nan,
                            chain_id=chain.id,
                            res1letter=res.upper(),
                            overall_id=overall_id,
                            coord=None,
                            ok=False,
                        )
                    ]
                    if long_factors:
                        polymer[chain.id].seq[-1].factor = numpy.nan
                    polymer[chain.id].seq[-1].index = len(polymer[chain.id].seq) - 1
                    polymer[chain.id].sequence += res.upper()
                    i += 1  # reset every chain
                    overall_id += 1  # never reset
                    continue
                else:
                    expected_atomseq_indices[chain.id] += [j]
                    if three_to_one(reslist[gi].resname) != res:
                        sys.stderr.write(
                            "**ERROR** discrepancy in pdb_to_polymer when seqres_to_seq chain %s res from ATOM sequence %s and from seqres %s gi=%d\n"
                            % (chain.id, three_to_one(reslist[gi].resname), res, gi)
                        )
                    res = reslist[gi]
                    gi += 1
            if res.id[0] == "W":
                continue  # remove solvent... anyway it's always better to clean the pdb
            if "CA" in res:
                CA_coord = res["CA"].coord
            else:
                CA_coord = None
            # add res
            r1 = three_to_one(res.resname)
            if res.id[2] not in ["", " "]:
                pdb_number = str(res.id[1]) + str(res.id[2])
            else:
                pdb_number = int(res.id[1])
            polymer[chain.id].seq += [
                Residue(
                    resname=res.resname,
                    pdb_id=pdb_number,
                    chain_id=chain.id,
                    res1letter=r1,
                    overall_id=overall_id,
                    coord=CA_coord,
                    ok=r1 in amino_list1,
                )
            ]
            if long_factors:
                polymer[chain.id].seq[-1].factor = res[
                    "CA"
                ].bfactor  # save CA bfactor to residue - this may be a computational score otherwise long_factor is not needed
            polymer[chain.id].seq[-1].index = len(polymer[chain.id].seq) - 1
            polymer[chain.id].aa_pdb_id_map[pdb_number] = i
            polymer[chain.id].sequence += r1
            if polymer[chain.id].seq[-1].ok:
                polymer[chain.id].sequenceOK += r1
            overall_map += [
                polymer[chain.id].seq[-1]
            ]  # hopefully this will be pointers
            if dict_of_res_variable_to_initialize != None:
                for key in dict_of_res_variable_to_initialize:
                    polymer[chain.id].seq[-1].update(
                        str(key), dict_of_res_variable_to_initialize[key], check=False
                    )
            if crosslink:
                polymer[chain.id].seq[-1].residue = res
            i += 1  # reset every chain
            overall_id += 1  # never reset
    if add_seqres:
        for ch in polymer:
            if ch in polymer_seq_resid_lists:
                polymer[ch].seqres = polymer_seq_resid_lists[ch][0].replace(
                    "-", ""
                )  # will be seqres with those residues with missing coordinates in lower case (or some gaps sometimes if seqres not given but gaps identified from both discontinous residue numbering and violation of peptide bond length)
                if atom_seq_blocks_indices != {}:
                    l = [list(range(*block)) for block in atom_seq_blocks_indices[ch]]
                    polymer[ch].atom_seq_indices = numpy.array(
                        [item for sublist in l for item in sublist]
                    )  # flatten array so that can use numpy array notation on numpy profiles..
                    if any(
                        numpy.array(expected_atomseq_indices[ch])
                        != polymer[ch].atom_seq_indices
                    ):  # control
                        sys.stderr.write(
                            "**ERROR** discrepancy in pdb_to_polymer pdb %s when seqres_to_seq chain %s gaps from lower case residues in seqres dont match those from and from atom_seq_blocks_indices - keeping those from lower case\n"
                            % (pdb_file_name, chain.id)
                        )
                        polymer[ch].atom_seq_indices = numpy.array(
                            expected_atomseq_indices[ch]
                        )
                else:
                    polymer[ch].atom_seq_indices = numpy.array(
                        expected_atomseq_indices[ch]
                    )
        # print ch,polymer[ch].sequence
    polymer.overall_map = overall_map
    return polymer, structure, overall_map


def get_chains_of_polymer(polymer, structure, chain_ids, deepcopy=False):
    if type(chain_ids) is str:
        chain_ids = list(chain_ids)
    delete = []
    pol2 = Model()
    if type(polymer.pdbfile) is str:
        pol2.pdbfile = (
            polymer.pdbfile[: polymer.pdbfile.rfind(".")]
            + "_"
            + ("".join(chain_ids))
            + ".pdb"
        )
    for ch in polymer:
        if ch in chain_ids:
            if deepcopy:
                pol2[ch] = copy.deepcopy(polymer[ch])
            else:
                pol2[ch] = polymer[ch]
        else:
            delete += ch
    stru2 = structure.copy()
    for model in stru2:
        for ch in delete:
            model.detach_child(ch)
    return pol2, stru2


# a selector that extracts a model, a chain or a segment in a chain from a given chain in a given model in a pdb structure
class FragmentSelector_from_polymer(Bio.PDB.Select):
    def __init__(self, polymer, model=None):
        """
        model can be given to focus on one (or some as a list) specific models
        at residue-resolution cannot exclude individual atoms (which may be deleted from structure)
        """
        self.polymer = polymer
        self.MODEL = model
        if self.MODEL is not None and not hasattr(self.MODEL, "__len__"):
            self.MODEL = [self.MODEL]
        self.warned = 0
        self.warned2 = 0

    def accept_residue(self, residue):
        if (
            self.MODEL is None
            or int(residue.get_parent().get_parent().id) in self.MODEL
        ):
            chid = str(residue.get_parent().id)
            if chid in self.polymer:  # chain_id in polymer
                if residue.id[2] not in ["", " "]:
                    resid = str(residue.id[1]) + str(residue.id[2])
                else:
                    resid = residue.id[1]
                if resid in self.polymer[chid].aa_pdb_id_map:
                    if "CA" not in residue:
                        if self.warned2 == 5:
                            sys.stderr.write("  ...Suppressing further warnings\n")
                        elif self.warned2 > 5:
                            return 1
                        sys.stderr.write(
                            "*Potential Warn FragmentSelector_from_polymer() res_pdb_id %s in polymer['%s'].aa_pdb_id_map but 'CA' not in residue %s in structure %s - trying to write anyway\n"
                            % (
                                str(resid),
                                chid,
                                str(residue),
                                residue.get_parent().get_parent().get_parent().id,
                            )
                        )
                        self.warned2 += 1
                        return 1
                    elif (
                        self.polymer[chid]
                        .seq[self.polymer[chid].aa_pdb_id_map[resid]]
                        .residue["CA"]
                        .coord
                        == residue["CA"].coord
                    ).all():
                        return 1
                    else:
                        if self.warned == 5:
                            sys.stderr.write("  ...Suppressing further warnings\n")
                        elif self.warned < 5:
                            sys.stderr.write(
                                "**ERROR** FragmentSelector_from_polymer() res_pdb_id %s in polymer['%s'].aa_pdb_id_map but corresponding biopython residue['CA'].coord is different from that in structure %s\n   respectively: %s and %s for %s and %s\n"
                                % (
                                    str(resid),
                                    chid,
                                    residue.get_parent().get_parent().get_parent().id,
                                    str(
                                        self.polymer[chid]
                                        .seq[self.polymer[chid].aa_pdb_id_map[resid]]
                                        .residue["CA"]
                                        .coord
                                    ),
                                    str(residue["CA"].coord),
                                    str(
                                        self.polymer[chid]
                                        .seq[self.polymer[chid].aa_pdb_id_map[resid]]
                                        .residue
                                    ),
                                    str(residue),
                                )
                            )
                        self.warned += 1
        return 0


def polymer_to_pdb(polymer, structure, pdbfilename="", model_id=None):
    """
    #save a structure to a pdb file by selecting corresponding residues found in polymer. 
    uses the biopython structure to print
    If one wishes to save part of the structure model_id can be given (but will probably fail unless it corresponds to the one used to make polymer)
    """
    if pdbfilename == "":
        pdbfilename = str(structure.id) + ".pdb"
    pdb_write = Bio.PDB.PDBIO()
    pdb_write.set_structure(structure)
    selector = FragmentSelector_from_polymer(polymer, model=model_id)
    pdb_write.save(pdbfilename, selector)  # save it!
    del selector
    del pdb_write
    return


# ---------------------------------------------------- SASA HANDLING ------------------------------------------------#


# value of the sasa for the residues alone in the solvent
res_alone_porter_radii = {
    "ALA": 229.846100,
    "ARG": 365.636800,
    "ASN": 276.554900,
    "ASP": 268.190400,
    "CYS": 263.035200,
    "GLN": 299.415600,
    "GLU": 294.653900,
    "GLY": 198.232900,
    "HIS": 311.246600,
    "ILE": 298.842000,
    "LEU": 299.087800,
    "LYS": 327.984000,
    "MET": 310.846000,
    "PHE": 329.335000,
    "PRO": 264.294800,
    "SER": 242.341500,
    "THR": 270.869100,
    "TRP": 366.552500,
    "TYR": 343.304700,
    "VAL": 280.676300,
}

res_alone_standard_radii = {
    "ALA": 223.043200,
    "ARG": 358.903900,
    "ASN": 272.012200,
    "ASP": 261.196700,
    "CYS": 248.364300,
    "GLN": 295.076000,
    "GLU": 288.814400,
    "GLY": 195.470700,
    "HIS": 305.588400,
    "ILE": 293.583600,
    "LEU": 293.869100,
    "LYS": 325.749300,
    "MET": 304.800000,
    "PHE": 323.684700,
    "PRO": 257.296800,
    "SER": 234.996200,
    "THR": 262.925300,
    "TRP": 363.212000,
    "TYR": 338.742100,
    "VAL": 272.546300,
}
res_alone_standard_radiiMax = {
    "ALA": 232.3586,
    "ARG": 377.1561,
    "ASN": 281.559,
    "ASP": 269.1945,
    "CYS": 258.7806,
    "GLN": 312.1445,
    "GLU": 298.4643,
    "GLY": 205.4119,
    "HIS": 318.0137,
    "ILE": 306.0498,
    "LEU": 312.3443,
    "LYS": 346.2746,
    "MET": 324.1663,
    "PHE": 336.4804,
    "PRO": 266.2991,
    "SER": 245.0882,
    "THR": 266.1156,
    "TRP": 377.7551,
    "TYR": 352.064,
    "VAL": 277.5915,
}
res_alone_standard_radii_allH = {
    "ALA": 247.5093,
    "ARG": 399.2883,
    "ASN": 299.1803,
    "ASP": 281.1734,
    "CYS": 274.8868,
    "GLN": 331.2829,
    "GLU": 312.72612,
    "GLY": 218.34812,
    "HIS": 331.8,
    "ILE": 321.8787,
    "LEU": 326.9812,
    "LYS": 363.0983,
    "MET": 339.4616,
    "PHE": 356.0118,
    "PRO": 284.7761,
    "SER": 262.2028,
    "THR": 290.0436,
    "TRP": 399.3182,
    "TYR": 372.4161,
    "VAL": 295.3025,
}

# older generally smaller radii
Gly_X_Gly_sasa_standard_radii = {
    "CYS": 137.5403,
    "ASP": 164.8272,
    "SER": 137.8495,
    "GLN": 187.3957,
    "LYS": 232.3219,
    "PRO": 162.7688,
    "THR": 158.9018,
    "PHE": 226.8894,
    "ALA": 124.5884,
    "HIS": 209.52451,
    "GLY": 104.1654,
    "ILE": 198.2432,
    "GLU": 195.0637,
    "LEU": 209.1485,
    "ARG": 272.5771,
    "TRP": 265.9949,
    "VAL": 173.8507,
    "ASN": 169.766600,
    "TYR": 237.741,
    "MET": 211.1405,
}

# the below is obtained by generating multiple 3-peptide G-X-G models and taking the maximum sasa value for amino acid X
Gly_X_Gly_sasa_standard_radiiMax = {
    "CYS": 155.1929,
    "ASP": 167.159,
    "SER": 135.5286,
    "GLN": 208.3877,
    "LYS": 245.8032,
    "PRO": 165.3191,
    "THR": 162.8606,
    "PHE": 229.7791,
    "ALA": 129.5636,
    "HIS": 212.151,
    "GLY": 100.1121,
    "ILE": 203.7346,
    "GLU": 196.2796,
    "LEU": 207.9027,
    "ARG": 273.0128,
    "TRP": 269.6858,
    "VAL": 178.4592,
    "ASN": 177.277,
    "TYR": 246.0931,
    "MET": 215.94,
}

Gly_X_Gly_sasa_standard_radii_allH = {
    "CYS": 161.3921,
    "ASP": 168.9211,
    "SER": 144.8628,
    "ASN": 183.3684,
    "GLN": 216.1981,
    "LYS": 255.8972,
    "ILE": 210.7685,
    "PRO": 173.0591,
    "THR": 171.9532,
    "PHE": 239.5635,
    "ALA": 137.121,
    "GLY": 105.9325,
    "HIS": 221.906,
    "LEU": 211.3939,
    "ARG": 284.8122,
    "TRP": 286.4021,
    "VAL": 186.5904,
    "GLU": 198.3556,
    "TYR": 260.2168,
    "MET": 227.7568,
}
"""
Obtained from:
Term_res={}
reload(sasa)
three_pep_dict=sasa.Gly_X_Gly_sasa_standard_radiiMax # or Gly_X_Gly_sasa_standard_radii
res_alone_dict=sasa.res_alone_standard_radiiMax # or res_alone_standard_radii
for k in three_pep_dict :
    Term_res[k]=three_pep_dict[k]+(res_alone_dict[k]-three_pep_dict[k])/2.
"""
Term_resMax = {
    "ALA": 180.9611,
    "ARG": 325.08445,
    "ASN": 229.418,
    "ASP": 218.17675,
    "CYS": 206.98675,
    "GLN": 260.2661,
    "GLU": 247.37194999999997,
    "GLY": 152.762,
    "HIS": 265.08235,
    "ILE": 254.8922,
    "LEU": 260.1235,
    "LYS": 296.0389,
    "MET": 270.05314999999996,
    "PHE": 283.12975,
    "PRO": 215.8091,
    "SER": 190.3084,
    "THR": 214.48809999999997,
    "TRP": 323.72045,
    "TYR": 299.07855,
    "VAL": 228.02535,
}
Term_res_allH = {
    "ALA": 192.31515000000002,
    "ARG": 342.05025,
    "ASN": 241.27435,
    "ASP": 225.04725000000002,
    "CYS": 218.13945,
    "GLN": 273.7405,
    "GLU": 255.54086,
    "GLY": 162.14031,
    "HIS": 276.853,
    "ILE": 266.3236,
    "LEU": 269.18755,
    "LYS": 309.49775,
    "MET": 283.6092,
    "PHE": 297.78765,
    "PRO": 228.9176,
    "SER": 203.5328,
    "THR": 230.9984,
    "TRP": 342.86015,
    "TYR": 316.31645,
    "VAL": 240.94645,
}
Term_res = {
    "CYS": 192.95229999999998,
    "ASP": 213.01195,
    "SER": 186.42284999999998,
    "GLN": 241.23585000000003,
    "LYS": 276.3679,
    "ASN": 220.54645000000002,
    "PRO": 210.0328,
    "THR": 210.91355,
    "PHE": 275.28705,
    "ALA": 173.8158,
    "HIS": 257.55645499999997,
    "GLY": 146.1138,
    "ILE": 244.52114999999998,
    "LEU": 246.8529,
    "ARG": 306.8865,
    "TRP": 314.60344999999995,
    "VAL": 221.55845,
    "GLU": 241.93905,
    "TYR": 288.24155,
    "MET": 257.97025,
}


# quite approximate without H
Atom_sasa_Gly_X_Gly = {
    "CYS": {
        "C": 2.3014,
        "CB": 43.5433,
        "CA": 18.1716,
        "O": 22.7947,
        "N": 5.4289,
        "SG": 63.835,
    },
    "ASP": {
        "C": 0.4758,
        "CB": 44.8724,
        "CA": 14.7185,
        "CG": 7.1384,
        "O": 29.9894,
        "N": 5.2612,
        "OD1": 29.1247,
        "OD2": 40.0469,
    },
    "SER": {
        "C": 2.2854,
        "OG": 46.5852,
        "CB": 43.6754,
        "CA": 10.1883,
        "O": 26.4986,
        "N": 9.1256,
    },
    "GLN": {
        "C": 0.8559,
        "CB": 25.2991,
        "CA": 12.535,
        "CG": 23.6232,
        "O": 18.0916,
        "CD": 5.0421,
        "N": 0.9343,
        "NE2": 59.3053,
        "OE1": 39.207,
    },
    "LYS": {
        "C": 0.5748,
        "CB": 35.1397,
        "CA": 11.1026,
        "CG": 33.8123,
        "CE": 36.3099,
        "CD": 17.3203,
        "NZ": 80.2166,
        "O": 25.2061,
        "N": 9.1277,
    },
    "PRO": {
        "C": 1.1333,
        "CB": 40.7363,
        "CA": 15.603,
        "CG": 47.4626,
        "O": 29.6308,
        "CD": 33.8537,
        "N": 0.0321,
    },
    "THR": {
        "C": 1.3341,
        "CB": 20.2877,
        "CA": 10.3743,
        "OG1": 34.7814,
        "O": 13.2496,
        "N": 3.7183,
        "CG2": 67.1579,
    },
    "PHE": {
        "C": 0.2537,
        "CD2": 26.7896,
        "CB": 33.2932,
        "CA": 9.7733,
        "CG": 0.5978,
        "O": 19.5725,
        "N": 9.3556,
        "CZ": 38.0282,
        "CD1": 24.6558,
        "CE1": 38.0586,
        "CE2": 38.0864,
    },
    "ALA": {"CB": 73.3679, "CA": 14.0006, "C": 0.6154, "O": 21.9373, "N": 9.8272},
    "HIS": {
        "C": 2.8008,
        "CD2": 37.1075,
        "CB": 28.6518,
        "CA": 18.9697,
        "CG": 2.8404,
        "O": 25.1511,
        "N": 10.0334,
        "CE1": 53.1662,
        "ND1": 18.8681,
        "NE2": 31.3876,
    },
    "GLY": {"CA": 42.4908, "C": 3.4926, "O": 24.1925, "N": 13.9281},
    "ILE": {
        "C": 1.1965,
        "CB": 10.3568,
        "CA": 0.2168,
        "O": 19.124,
        "N": 5.4548,
        "CD1": 67.7592,
        "CG1": 28.448,
        "CG2": 59.4196,
    },
    "GLU": {
        "C": 0.0,
        "CB": 29.5425,
        "CA": 16.1576,
        "CG": 19.8556,
        "O": 25.4739,
        "CD": 10.6519,
        "OE1": 37.2931,
        "OE2": 41.3987,
        "N": 6.5004,
    },
    "LEU": {
        "C": 2.0362,
        "CB": 26.9138,
        "CA": 2.6213,
        "CG": 6.3073,
        "O": 24.9902,
        "N": 4.46,
        "CD1": 68.5152,
        "CD2": 58.5228,
    },
    "ARG": {
        "C": 0.2153,
        "CB": 27.4888,
        "CA": 12.0459,
        "CG": 21.7017,
        "NE": 14.4843,
        "O": 19.5801,
        "CD": 36.8625,
        "CZ": 3.8751,
        "NH1": 56.5475,
        "NH2": 68.0737,
        "N": 9.1158,
    },
    "TRP": {
        "C": 0.2654,
        "CZ2": 38.3929,
        "CB": 33.0853,
        "CA": 11.28,
        "CG": 0.8923,
        "O": 23.0844,
        "N": 4.9631,
        "CH2": 37.6549,
        "CE3": 28.7496,
        "CE2": 3.9817,
        "CD2": 3.297,
        "CZ3": 37.5502,
        "NE1": 29.585,
        "CD1": 28.6116,
    },
    "VAL": {
        "C": 0.0,
        "CB": 10.7375,
        "CA": 7.4062,
        "O": 18.8622,
        "N": 1.6681,
        "CG1": 69.2922,
        "CG2": 57.7869,
    },
    "ASN": {
        "C": 2.2733,
        "CB": 35.834,
        "CA": 9.079,
        "CG": 3.341,
        "O": 20.9494,
        "N": 10.2006,
        "OD1": 36.2756,
        "ND2": 57.2352,
    },
    "TYR": {
        "C": 0.6412,
        "CD2": 19.742,
        "OH": 53.9062,
        "CB": 36.5035,
        "CA": 2.823,
        "CG": 0.6212,
        "O": 21.5179,
        "N": 5.6755,
        "CZ": 3.5214,
        "CD1": 25.6591,
        "CE1": 35.3932,
        "CE2": 28.6421,
    },
    "MET": {
        "C": 0.4772,
        "CB": 31.0654,
        "CA": 12.9384,
        "CG": 23.4488,
        "CE": 83.8018,
        "N": 6.6608,
        "O": 24.8407,
        "SD": 38.4237,
    },
}


def get_charges_exposure(
    residue_with_sasa,
    Atom_sasa_resalone=Atom_sasa_Gly_X_Gly,
    res_pdb_id=-1,
    print_warns=True,
):
    """
    residue_with_sasa must be a biopython residue where we have added the sasa of each atom (with add_sasa_to_polymer() or by calling calculate_polymer_surface() with add_to_atom_within_structure=True)
    """
    res = residue_with_sasa
    exposure = 0
    ok = True
    if res.resname == "ASP":
        ok = False
        if "OD1" not in res:
            s = (
                " pdb warning atom 'OD1' not in 'ASP' %d skipping get_charges_exposure for this residue"
                % (res_pdb_id)
            )
        elif "OD2" not in res:
            s = (
                " pdb warning atom 'OD2' not in 'ASP' %d skipping get_charges_exposure for this residue"
                % (res_pdb_id)
            )
        else:
            ok = True
            s = "'OD1': %lf , 'OD2': %lf " % (res["OD1"].SASA, res["OD2"].SASA)
            exposure = (res["OD1"].SASA + res["OD2"].SASA) / (
                Atom_sasa_resalone["ASP"]["OD1"] + Atom_sasa_resalone["ASP"]["OD2"]
            )
    elif res.resname == "GLU":
        ok = False
        if "OE1" not in res:
            s = (
                " pdb warning atom 'OE1' not in 'GLU' %d skipping get_charges_exposure for this residue"
                % (res_pdb_id)
            )
        elif "OE2" not in res:
            s = (
                " pdb warning atom 'OE2' not in 'GLU' %d skipping get_charges_exposure for this residue"
                % (res_pdb_id)
            )
        else:
            ok = True
            s = "'OE1': %lf , 'OE2': %lf " % (res["OE1"].SASA, res["OE2"].SASA)
            exposure = (res["OE1"].SASA + res["OE2"].SASA) / (
                Atom_sasa_resalone["GLU"]["OE1"] + Atom_sasa_resalone["GLU"]["OE2"]
            )
    elif res.resname == "LYS":
        ok = False
        if "NZ" not in res:
            s = (
                " pdb warning atom 'NZ' not in 'LYS' %d skipping get_charges_exposure for this residue"
                % (res_pdb_id)
            )
        else:
            ok = True
            s = "'NZ': %lf " % (res["NZ"].SASA)
            exposure = res["NZ"].SASA / Atom_sasa_resalone["LYS"]["NZ"]
    elif res.resname == "ARG":
        ok = False
        if "NH2" not in res:
            s = (
                " pdb warning atom 'NH2' not in 'ARG' %d skipping get_charges_exposure for this residue"
                % (res_pdb_id)
            )
        else:
            ok = True
            s = "'NH1': %lf , 'NH2': %lf " % (res["NH1"].SASA, res["NH2"].SASA)
            exposure = (res["NH1"].SASA + res["NH2"].SASA) / (
                Atom_sasa_resalone["ARG"]["NH1"] + Atom_sasa_resalone["ARG"]["NH2"]
            )
    elif res.resname == "HIS":
        ok = False
        if "ND1" not in res:
            s = (
                " pdb warning atom 'ND1' not in 'HIS' %d skipping get_charges_exposure for this residue"
                % (res_pdb_id)
            )
        elif "NE2" not in res:
            s = (
                " pdb warning atom 'NE2' not in 'HIS' %d skipping get_charges_exposure for this residue"
                % (res_pdb_id)
            )
        else:
            ok = True
            s = "'ND1': %lf , 'NE2': %lf " % (res["ND1"].SASA, res["NE2"].SASA)
            exposure = (res["ND1"].SASA + res["NE2"].SASA) / (
                Atom_sasa_resalone["HIS"]["ND1"] + Atom_sasa_resalone["HIS"]["NE2"]
            )
    if print_warns:
        if exposure > 1.2 or exposure < -0.1:
            sys.stderr.write(
                "  WARN get_charges_exposure for charged %s rel_value %lf should be in [0,1]\n  %s\n"
                % (res.resname, exposure, s)
            )
        if not ok:
            sys.stderr.write("  WARN %s\n" % (s))
    if exposure > 1:
        exposure = 1.0
    elif exposure < 0:
        exposure = 0.0
    return exposure


def add_sasa_to_polymer(
    polymer, structure, add_to_atom_within_structure=True, accurate=True,calculate_solvent_exposure=True):
    """
    uses freesasa to add the SASA of residues (and atoms if add_to_atom_within_structure) to a biopython structure.
    However freesasa is not ideally integrated with biopython so the procedure relies on a loop on all atoms hoping that 
    their order in the freesasa result is exactly the same as in the biopython structure. 
      Note that the pymol-like surface selection available in freesasa did not work for biopython structure in the tested version:
         File "freesasa.pyx", line 931, in freesasa.selectArea
        AttributeError: 'Structure' object has no attribute '_get_address'
    SASA from freesasa of individual residues obtained with this function correlate well (R>0.98) with those from former calculation with few slight outliers.  
    """
    if accurate:
        result, sasa_classes = freesasa.calcBioPDB(
            structure,
            freesasa.Parameters({"algorithm": freesasa.LeeRichards, "n-slices": 100}),
        )
    else:
        result, sasa_classes = freesasa.calcBioPDB(structure)
    done_residues = {}
    has_Hydrogens=False
    for j, atom in enumerate(
        structure.get_atoms()
    ):  # assumes and hopes order will be the same as freesas result!
        (
            _,
            _,
            chain,
            residue_id,
            atom_name,
        ) = atom.get_full_id()  # '1ax8', 0, 'A', (' ', 146, ' '), ('OXT', ' ')
        if atom_name[0][0]=='H' :
            has_Hydrogens=True
        if residue_id[0] != " ":
            sys.stderr.write(
                "**WARNING** in add_sasa_to_polymer identified HETATM at atom index %d num %d %s file %s - may cause problems [Try CLEANING pdb file]\n"
                % (
                    j,
                    atom.get_serial_number(),
                    str(atom.get_full_id()),
                    str(structure.id),
                )
            )
            continue  # HETATM
        if residue_id[-1] != " ":
            residue_id = str(residue_id[1]) + residue_id[-1]
        else:
            residue_id = residue_id[1]
        if (chain, residue_id) not in done_residues:
            done_residues[(chain, residue_id)] = 0
            polymer[chain].seq[polymer[chain].aa_pdb_id_map[residue_id]].SASA = 0
        # switch to done_residues as this line was adding more sasa if function was called multiple times on same polymer:if 'SASA' not in dir(polymer[chain].seq[polymer[chain].aa_pdb_id_map[residue_id]] ) : polymer[chain].seq[polymer[chain].aa_pdb_id_map[residue_id]].SASA=0
        # you cannot skip hydrogens as these would shield other atoms, must be removed from pdb before reading structure.
        atom_sasa = result.atomArea(j)
        if add_to_atom_within_structure:
            polymer[chain].seq[ polymer[chain].aa_pdb_id_map[residue_id]].residue[
                atom_name[0]].SASA = atom_sasa
        polymer[chain].seq[polymer[chain].aa_pdb_id_map[residue_id]].SASA += atom_sasa
        done_residues[(chain, residue_id)] += atom_sasa
    if result.nAtoms() != j + 1:
        raise Exception(
            "in add_sasa_to_polymer() using freesasa module sasa exist for %d atoms, but processed %d\n"
            % (result.nAtoms(), j + 1)
        )
    if calculate_solvent_exposure :
        for chain in polymer :
            for j,res in enumerate(polymer[chain].seq) :
                if hasattr(res,'SASA') :
                    if j in [0, len(polymer[chain].seq)-1] : # use termini sasa (Gly-AA)
                        if has_Hydrogens :
                            polymer[chain].seq[j].solvent_exposure = res.SASA / Term_res_allH[ res.resname ]
                        else :
                            polymer[chain].seq[j].solvent_exposure = res.SASA / Term_resMax[ res.resname ]
                    else :
                        if has_Hydrogens :
                            polymer[chain].seq[j].solvent_exposure = res.SASA / Gly_X_Gly_sasa_standard_radii_allH[ res.resname ]
                        else :
                            polymer[chain].seq[j].solvent_exposure = res.SASA / Gly_X_Gly_sasa_standard_radiiMax[ res.resname ]
                    if polymer[chain].seq[j].solvent_exposure>1 :
                        polymer[chain].seq[j].solvent_exposure=1
    return polymer


def res_sasa_dict_from_structure(structure, accurate=True):
    """
    uses freesasa
    return sasa_dict  where keys are (chain_id,residue_id) and values are the sasa of the residues
    accurate=False makes it ~ 2.5x faster
    """
    if accurate:
        result, sasa_classes = freesasa.calcBioPDB(
            structure,
            freesasa.Parameters({"algorithm": freesasa.LeeRichards, "n-slices": 100}),
        )
    else:
        result, sasa_classes = freesasa.calcBioPDB(structure)
    sasa_dict = {}
    for j, atom in enumerate(
        structure.get_atoms()
    ):  # assumes and hopes order will be the same as freesas result!
        (
            _,
            _,
            chain,
            residue_id,
            atom_name,
        ) = atom.get_full_id()  # '1ax8', 0, 'A', (' ', 146, ' '), ('OXT', ' ')
        if residue_id[0] != " ":
            sys.stderr.write(
                "**WARNING** in res_sasa_dict_from_structure identified HETATM at atom index %d num %d %s file %s - may cause problems [Try CLEANING pdb file]\n"
                % (
                    j,
                    atom.get_serial_number(),
                    str(atom.get_full_id()),
                    str(structure.id),
                )
            )
            continue  # HETATM
        if residue_id[-1] != " ":
            residue_id = str(residue_id[1]) + residue_id[-1]
        else:
            residue_id = residue_id[1]
        if (chain, residue_id) not in sasa_dict:
            sasa_dict[(chain, residue_id)] = 0
        # switch to done_residues as this line was adding more sasa if function was called multiple times on same polymer:if 'SASA' not in dir(polymer[chain].seq[polymer[chain].aa_pdb_id_map[residue_id]] ) : polymer[chain].seq[polymer[chain].aa_pdb_id_map[residue_id]].SASA=0
        # you cannot skip hydrogens as these would shield other atoms, must be removed from pdb before reading structure.
        sasa_dict[(chain, residue_id)] += result.atomArea(j)
    if result.nAtoms() != j + 1:  # this is the best control one can make
        raise Exception(
            "in res_sasa_dict_from_structure() using freesasa module sasa exist for %d atoms, but processed %d\n"
            % (result.nAtoms(), j + 1)
        )
    return sasa_dict


def sasa_list_from_structure(structure, accurate=True):
    """
    uses freesasa  
    return sasa_list,done_residues  
    sasa_list is a numpy.array with the sasa of each residue as they appear from the biopython structure in the order of structure.get_atoms()
    done_residues is a list of the same order with (chain_id,residue_id)
    accurate=False makes it ~ 2.5x faster
    """
    if accurate:
        result, sasa_classes = freesasa.calcBioPDB(
            structure,
            freesasa.Parameters({"algorithm": freesasa.LeeRichards, "n-slices": 100}),
        )
    else:
        result, sasa_classes = freesasa.calcBioPDB(structure)
    done_residues = [None]
    sasa_list = []
    for j, atom in enumerate(
        structure.get_atoms()
    ):  # assumes and hopes order will be the same as freesas result!
        (
            _,
            _,
            chain,
            residue_id,
            atom_name,
        ) = atom.get_full_id()  # '1ax8', 0, 'A', (' ', 146, ' '), ('OXT', ' ')
        if residue_id[0] != " ":
            sys.stderr.write(
                "**WARNING** in sasa_list_from_structure identified HETATM at atom index %d num %d %s file %s - may cause problems [Try CLEANING pdb file]\n"
                % (
                    j,
                    atom.get_serial_number(),
                    str(atom.get_full_id()),
                    str(structure.id),
                )
            )
            continue  # HETATM
        if residue_id[-1] != " ":
            residue_id = str(residue_id[1]) + residue_id[-1]
        else:
            residue_id = residue_id[1]
        if (chain, residue_id) != done_residues[-1]:
            done_residues += [(chain, residue_id)]
            sasa_list += [0.0]
        # switch to done_residues as this line was adding more sasa if function was called multiple times on same polymer:if 'SASA' not in dir(polymer[chain].seq[polymer[chain].aa_pdb_id_map[residue_id]] ) : polymer[chain].seq[polymer[chain].aa_pdb_id_map[residue_id]].SASA=0
        # you cannot skip hydrogens as these would shield other atoms, must be removed from pdb before reading structure.
        atom_sasa = result.atomArea(j)
        sasa_list[-1] += atom_sasa
    if result.nAtoms() != j + 1:
        raise Exception(
            "in sasa_list_from_structure() using freesasa module sasa exist for %d atoms, but processed %d\n"
            % (result.nAtoms(), j + 1)
        )
    del done_residues[0]  # remove None at beginning
    return numpy.array(sasa_list), done_residues


def SASA_contacts_of_residue(
    structure,
    residue_index,
    polymer,
    chain=None,
    sasa_list_full_structure=None,
    index_is_pdb_id=False,
    resname=None,
    model_id=0,
    accurate=False,
):
    """
    uses freesasa
    it will remove sidechain atoms of each residue (including CA atom as otherwise Gly becomes problematic)
     then calcualtes the difference in SASA between a structure with the sidechain at residue_index and one without -
     those residues with a singnificant SASA change between the two will be considered in contact with  residue_index
     return sasa_difference , sasa_list_full_structure, done_residues
      sasa_list_full_structure is the SASA of the full structure and can be given as input to future calls of the function to avoid calculating it multiple times
      done_residues is the corresponding list of tuples like (chain_id, residue_pdb_id)
    note it will MODIFY the sasa in polymer at
    polymer can be None as it is used only for some consinstency checks
    """
    # shortcut for atom = structure[0]['A'][100]['CA']  HOWEVER 100 is not index is pdb_id!! must do -> residue=list(structure[0]['A'])[100]
    if type(residue_index) is str:
        residue_index, chain, resname, _ = mybio.parse_mutation_str(residue_index)
    if chain is None:
        if len(structure[model_id]) == 1:
            chain = structure[model_id].get_chains().next().id
        else:
            chain = structure[model_id].get_chains().next().id
            sys.stderr.write(
                "*Warn in SASA_contacts_of_residue chain is None but %d chains in structure - assinging to first chain %s\n"
                % (len(structure[model_id]), chain)
            )
    if polymer is not None:
        if index_is_pdb_id:
            residue_index = polymer[chain].aa_pdb_id_map[residue_index]
        if (
            resname is not None
            and resname != polymer[chain].seq[residue_index].res1letter
        ):
            sys.stderr.write(
                "*ERROR* in SASA_contacts_of_residue input str hints at residue %s at index %d of chain %s but found residue %s instead!\n"
                % (
                    resname,
                    residue_index,
                    chain,
                    polymer[chain].seq[residue_index].res1letter,
                )
            )
    # check if SASA already computed - if not add
    # if not hasattr(polymer[chain].seq[residue_index], 'SASA') :
    #    polymer=add_sasa_to_polymer(polymer,structure, add_to_atom_within_structure=False, accurate=accurate)
    if sasa_list_full_structure is None:
        sasa_list_full_structure, done_residues = sasa_list_from_structure(
            structure, accurate=accurate
        )
        if polymer is not None and (
            done_residues[polymer[chain].seq[residue_index].overall_id][0] != chain
            or done_residues[polymer[chain].seq[residue_index].overall_id][1]
            != polymer[chain].seq[residue_index].pdb_id
        ):
            sys.stderr.write(
                "*ERROR* in SASA_contacts_of_residue order of sasa_list_full_structure does not seem compatible with that of polymer:\n  sasa list = %s and polymer %s\n"
                % (
                    str(done_residues[polymer[chain].seq[residue_index].overall_id]),
                    str((chain, polymer[chain].seq[residue_index].pdb_id)),
                )
            )
    stru2 = structure.copy()
    res = list(stru2[model_id][chain])[residue_index]
    for atom in list(res):
        if atom.id not in ["N", "C", "O", "OXT"]:
            res.detach_child(atom.id)

    sasa_list2, done_residues = sasa_list_from_structure(stru2, accurate=accurate)
    return (
        sasa_list2 - sasa_list_full_structure,
        sasa_list_full_structure,
        done_residues,
    )


def SASA_contact_map(structure, polymer=None, model_id=0, accurate=False):
    """
    quite SLOW for a structure with only 144 residues took 16.5 seconds on laptop
way to 'pass selection':
from Bio.PDB import *
s=Bio.PDB.Structure.Structure(0)
s.add(Model.Model(0))
s[0].add(Bio.PDB.Chain.Chain('A'))
s[0]['A']
in id=A>
s[0]['A'].add(next(residues)) # Only one residue obtained as residues=structure.get_residues() from a full structure

    calculate a SASA based contact map which stores at row j the differences in SASA between a structure where the sidechain of residue j is removed and one where it is there.
    The diagonal will have negative values

import time
sta=time.time()
contact_map=structs.SASA_contact_map(structure, polymer) # Took 34.5 seconds on 132 globular protein
print("Took %g seconds"%(time.time()-sta))

    """
    sasa_list_full_structure = None
    contact_map = None
    overall_id = 0
    for chain in structure[model_id]:
        for j, res in enumerate(chain):
            diff, sasa_list_full_structure, done_residues = SASA_contacts_of_residue(
                structure,
                residue_index=j,
                polymer=polymer,
                chain=chain.id,
                sasa_list_full_structure=sasa_list_full_structure,
                accurate=accurate,
            )
            if contact_map is None:
                contact_map = numpy.zeros(
                    (len(sasa_list_full_structure), len(sasa_list_full_structure))
                )
            if (
                done_residues[overall_id][0] != chain.id
                or done_residues[overall_id][1] != res.id[1]
            ):  # res.id = e.g. (' ', 144, ' ')
                sys.stderr.write(
                    "*ERROR* in SASA_contact_map overall_id discrepancy at %d expecting from structure (%s,%s) but found (%s,%s) in sasa_list instead!\n"
                    % (
                        overall_id,
                        chain.id,
                        str(res.id[1]),
                        done_residues[overall_id][0],
                        str(done_residues[overall_id][1]),
                    )
                )
            contact_map[overall_id] = diff
            overall_id += 1
    return contact_map


def _get_structure_of_shell(
    shell_residues, additional_atoms=None, stru_id=0, model_id=0
):
    """
    :param shell_residues: is a list of biopython residue objects that one want to include in the shell structure
    :return: shell_structure as a biopython strucutre objects comprisning residues (and corresponding chains) of shell_residues
    """
    s = Bio.PDB.Structure.Structure(stru_id)
    s.add(Bio.PDB.Model.Model(model_id))
    for residue in shell_residues:  # loop on shell
        chid = residue.get_parent().get_id()
        if chid not in s[0]:
            s[0].add(Bio.PDB.Chain.Chain(chid))
        s[0][chid].add(residue)
    if additional_atoms is not None:
        for atom in additional_atoms:
            residue = atom.get_parent()
            chid = residue.get_parent().get_id()
            resid = residue.get_id()
            if chid not in s[0]:
                s[0].add(Bio.PDB.Chain.Chain(chid))
            if resid not in s[0][chid]:
                s[0][chid].add(
                    Bio.PDB.Residue.Residue(
                        id=resid, resname=residue.resname, segid=residue.segid
                    )
                )
            s[0][chid][resid].add(atom)
    return s


def SASA_contact_map_fast(
    polymer,
    structure,
    shell_size=14,
    larger_shell_size=28,
    accurate=False,
    method="with/without",
    add_contact_list_to_residues=True,
    sasa_difference_threshold=2,
    reset=False,
):
    """
    MISSING: leave backbone or part of backbone of residue under scrutiny (use special recipipe for Pro if you don't delete entirely)
           to implement can use additional_atoms as argument of function _get_structure_of_shell
    Calculates a residue contact map based on the sasa criterion, that is a residue res2 is considered in contact
     with the residue under scrutiny res1 if its sasa changes when calculated in the presence and absence of res1.
    :param polymer:
    :param structure: can be None if method 'with/without' is used, otherwise must be a Bio.PDB structure object.
    :param shell_size: radius of CA distances to define a shell for which to calculate sasa with/without res1
    :param larger_shell_size: second radius used only for method 'double-shell', must be > shell_size
    :param accurate: parameter for freesasa calculation, False is faster and should be good enough for purpose.
           for example for a mAb with 8977 contacts (sasa_difference_threshold=2) and matrix shape (1315, 1315) only 3
              elements have a greater than 2 absoulte diff when calcualted with accurate=True or False.
              Number of contacts change to 8977 when accurate=True all around the threshold
    :param method: either 'with/without' or 'double-shell'.
                'with/without' per each residue res1 calculates the sasa of the shell around shell_size twice,
                once with res1 and once without # 17.7/20.5 seconds on 132 residues; 120 seconds on mAb with 1315 residues
                  (258 s with accurate=True)
                'double-shell' per each residue res1 calculates the sasa of the shell around larger_shell_size once,
                and then considers sasa differnces of only those residues within shell_size ignoring the others
                  29.9/37.8 seconds on 132 residues  289/272 seconds on mAb with 1315 residues
                'with/without' is faster and also more robust, 'double-shell' is just an old implementation.
    :param add_contact_list_to_residues: will create two variable in each Residue (res1) within polymer.seq:
                sasa_contacts will be the sasa differences for each contacting residue observed with/without res1
                and one sasa_contact_ids are the corresponding overall_id of those residues in contact with res1
    :param sasa_difference_threshold: sasa difference required for a residue to be considered in contact with res1
    :param reset: whether to recompute shells with get_contact_dict if values already present (otherwise computed anyway)
                 useful if one changes shell_size as otherwise it's not updated.
    :return: sasa_difference_mat a numpy matrix with rows and colums whose index correpond to the overall_id of residues
      as given in input polymer. Values of res_row,res_column are sasa_res_column_without_res_row- sasa_res_column_with_res_row
       therefore this matrix is not symmetric.


#tests:
import structs,time
sta=time.time()
sasa_difference_mat= structs.SASA_contact_map_fast( polymer,structure,reset=True,method='with/without')
print ("Took %g seconds"%(time.time()-sta))

sta=time.time()
sasa_difference_mat2= structs.SASA_contact_map_fast( polymer,structure,reset=True,method='double-shell') # 29 seconds on 132 residues
print ("Took %g seconds"%(time.time()-sta))

sta=time.time()
sasa_difference_mat= structs.SASA_contact_map_fast( polMab,struMab,reset=True,method='with/without')
print ("Took %g seconds"%(time.time()-sta))

sta=time.time()
sasa_difference_matA= structs.SASA_contact_map_fast( polMab,struMab,reset=True,method='with/without',accurate=True)
print ("Took %g seconds"%(time.time()-sta))

reload(structs)
sta=time.time()
sasa_difference_mat2= structs.SASA_contact_map_fast( polMab,struMab,reset=True,method='double-shell')
print ("Took %g seconds"%(time.time()-sta))


    """
    Nres = sum([len(polymer[ch].seq) for ch in polymer])
    sasa_difference_mat = numpy.zeros(
        (Nres, Nres)
    )  # will be a matrix (dict of dict) of overall_id1, overall_id2 with values the difference of SASA with/without residue 1
    for jc1, chid1 in enumerate(polymer):
        for j1, res1 in enumerate(polymer[chid1].seq):
            if j1 == 0:  # only once
                if (
                    not hasattr(res1, "dinstance_contacts")
                    or not hasattr(res1, "long_dinstance_contacts")
                    or reset
                ):  # Make corse distance contacts to speed up subsequent sasa calculation
                    if method == "with/without":
                        larger_shell_size = None  # no need to compute
                    _ = get_contact_dict(
                        polymer,
                        shell_size=shell_size,
                        larger_shell_size=larger_shell_size,
                        add_contact_list_to_residues=True,
                    )
                if method == "double-shell" and (
                    (not hasattr(res1, "SASA")) or reset
                ):  # calculate sasa of residues, not needed
                    polymer = add_sasa_to_polymer(
                        polymer,
                        structure,
                        add_to_atom_within_structure=False,
                        accurate=accurate,
                    )
            if add_contact_list_to_residues:
                res1.sasa_contacts = []
                res1.sasa_contact_ids = []
            if method == "double-shell":
                # calculate sasa around a shell in once res1 (sidechain) has been removed
                # first make a dummy structure that contains only this shell
                shell_structure = _get_structure_of_shell(res1.long_dinstance_contacts)
                if len([a for a in shell_structure.get_residues()]) == 0:
                    sys.stderr.write(
                        "**WARNING** in structs.SASA_contact_map_fast() shell_structure of residue:[%s] contains no residues! Skipping this\n"
                        % (repr(res1))
                    )
                    continue
                # calculate shell sasa without Res1
                sasa_dict = res_sasa_dict_from_structure(
                    shell_structure, accurate=accurate
                )  # done_residues is like [('A', 2), ('A', 3)... chain, res id
                # find out which residues now have a difference sasa
                for (ch, resid) in sasa_dict:
                    res2_ovid = (
                        polymer[ch].seq[polymer[ch].aa_pdb_id_map[resid]].overall_id
                    )
                    if res2_ovid in res1.dinstance_contact_ids:
                        sasa_difference_mat[res1.overall_id][res2_ovid] = (
                            sasa_dict[(ch, resid)]
                            - polymer[ch].seq[polymer[ch].aa_pdb_id_map[resid]].SASA
                        )
                        if (
                            add_contact_list_to_residues
                            and sasa_difference_mat[res1.overall_id][res2_ovid]
                            > sasa_difference_threshold
                        ):
                            res1.sasa_contacts += [
                                sasa_difference_mat[res1.overall_id][res2_ovid]
                            ]
                            res1.sasa_contact_ids += [res2_ovid]
            elif method == "with/without":
                # calculate sasa around a shell in once res1 (sidechain) has been removed
                # first make a dummy structure that contains only this shell
                shell_structure = _get_structure_of_shell(res1.dinstance_contacts)
                if len([a for a in shell_structure.get_residues()]) == 0:
                    sys.stderr.write(
                        "**WARNING** in structs.SASA_contact_map_fast() shell_structure of residue:[%s] contains no residues! Skipping this\n"
                        % (repr(res1))
                    )
                    continue
                # calculate shell sasa without Res1
                sasa_dict = res_sasa_dict_from_structure(
                    shell_structure, accurate=accurate
                )  # done_residues is like [('A', 2), ('A', 3)... chain, res id
                # now add res1 and calculate again
                shell_structure[0][chid1].add(
                    res1.residue
                )  # puts it at the end of the chain
                sasa_dict1 = res_sasa_dict_from_structure(
                    shell_structure, accurate=accurate
                )
                # find out which residues now have a difference sasa
                for (ch, resid) in sasa_dict:
                    res2_ovid = (
                        polymer[ch].seq[polymer[ch].aa_pdb_id_map[resid]].overall_id
                    )
                    # if done_residues1[j]!=(ch, resid) :
                    #    raise Exception("**ERROR** in SASA_contact_map_fast incongruent residues %s %s in two sasa lists under consideration j=%d len=%d\ndone_residues=%s\ndone_residues1=%s\n" %(str(done_residues1[j]),str((ch, resid)),j,len(done_residues),str(done_residues),str(done_residues1)))
                    sasa_difference_mat[res1.overall_id][res2_ovid] = (
                        sasa_dict[(ch, resid)] - sasa_dict1[(ch, resid)]
                    )
                    if (
                        add_contact_list_to_residues
                        and sasa_difference_mat[res1.overall_id][res2_ovid]
                        > sasa_difference_threshold
                    ):
                        res1.sasa_contacts += [
                            sasa_difference_mat[res1.overall_id][res2_ovid]
                        ]
                        res1.sasa_contact_ids += [res2_ovid]
            else:
                raise Exception(
                    "method variable must be either 'with/without' or 'double-shell'\n"
                )
    return sasa_difference_mat


def atom_squared_distance(residue_one, residue_two, atom_name1="CA", atom_name2="CA"):
    """
    Returns the C-alpha (or other atoms) distance between two residues
    """
    diff_vector = residue_one[atom_name1].coord - residue_two[atom_name2].coord
    return numpy.sum(diff_vector * diff_vector)


def get_contact_dict(
    polymer,
    shell_size=16.0,
    larger_shell_size=None,
    target_atom_name="CA",
    add_contact_list_to_residues=False,
):
    """
    get a overall_id-based dictionary of residues in contact
      keep in mind that the distance between a CA and the further away H atom in a TRP can be more than 9 A when you set the threshold.
    This coarse-grained function can be used to speed up subsequent more fine calculations of atom distances. 
    it returns contact_dict a dict whose keys are overall_ids and values a list of residue overall_ids that are in contact with it.
      it does not contain self-contacts
    if add_contact_list_to_residues adds to each res in .seq a variable named
      dinstance_contacts which is a list of Bio.PDB residue objects corresponding to the residues in contact with the one
      under scrutiny
    """
    if larger_shell_size is not None:
        long_contact_dict = {}
        if larger_shell_size < shell_size:
            a = shell_size
            shell_size = larger_shell_size
            larger_shell_size = a
            sys.sterr.write(
                "**WARNING** in get_contact_dict() given larger_shell_size of %g, but this must be > shell_size %g | These two have been SWAPPED\n"
                % (larger_shell_size, shell_size)
            )
        squared_larger_shell_size = larger_shell_size ** 2
    squared_shell = shell_size ** 2
    contact_dict = {}
    tot_len = sum([len(polymer[ch].seq) for ch in polymer])
    contact_matrix_squared = numpy.zeros((tot_len, tot_len))
    for jc1, chid1 in enumerate(polymer):
        for j1, res1 in enumerate(polymer[chid1].seq):
            if res1.overall_id not in contact_dict:
                contact_dict[res1.overall_id] = []
                if larger_shell_size is not None:
                    long_contact_dict[res1.overall_id] = []
                if add_contact_list_to_residues:
                    res1.dinstance_contacts = []
                    res1.dinstance_contact_ids = []
                    if larger_shell_size is not None:
                        res1.long_dinstance_contacts = []
                        res1.long_dinstance_contact_ids = []
            for jc2, chid2 in enumerate(polymer):
                if jc2 >= jc1:
                    if jc2 == jc1:
                        start = j1 + 1
                    else:
                        start = 0
                    for res2 in polymer[chid2].seq[start:]:
                        if res2.overall_id not in contact_dict:
                            contact_dict[res2.overall_id] = []
                            if larger_shell_size is not None:
                                long_contact_dict[res2.overall_id] = []
                            if add_contact_list_to_residues:
                                res2.dinstance_contacts = []
                                res2.dinstance_contact_ids = []
                                if larger_shell_size is not None:
                                    res2.long_dinstance_contacts = []
                                    res2.long_dinstance_contact_ids = []
                        if res1.overall_id == res2.overall_id:
                            continue  # self-contact are pointless
                        if res1.ok and res2.ok:
                            D2 = atom_squared_distance(
                                res1.residue,
                                res2.residue,
                                atom_name1=target_atom_name,
                                atom_name2=target_atom_name,
                            )
                        else:
                            D2 = numpy.nan
                        contact_matrix_squared[res1.overall_id, res2.overall_id] = D2
                        contact_matrix_squared[res2.overall_id, res1.overall_id] = D2
                        if D2 < squared_shell:
                            contact_dict[res1.overall_id] += [res2.overall_id]
                            contact_dict[res2.overall_id] += [res1.overall_id]
                            # print("DEB: contact j1", j1, res1.overall_id, 'adding', res2.overall_id,'contact_dict[res2.overall_id]=',contact_dict[res2.overall_id],'contact_dict[res1.overall_id]=',contact_dict[res1.overall_id])
                            if add_contact_list_to_residues:
                                res1.dinstance_contacts += [res2.residue]
                                res1.dinstance_contact_ids += [res2.overall_id]
                                res2.dinstance_contacts += [res1.residue]
                                res2.dinstance_contact_ids += [res1.overall_id]
                        if (
                            larger_shell_size is not None
                            and D2 < squared_larger_shell_size
                        ):
                            long_contact_dict[res1.overall_id] += [res2.overall_id]
                            long_contact_dict[res2.overall_id] += [res1.overall_id]
                            if add_contact_list_to_residues:
                                res1.long_dinstance_contacts += [res2.residue]
                                res1.long_dinstance_contact_ids += [res2.overall_id]
                                res2.long_dinstance_contacts += [res1.residue]
                                res2.long_dinstance_contact_ids += [res1.overall_id]
    return contact_dict, contact_matrix_squared


def is_hydrogen(atom_name, ignore_backbone_hydrogen=False, ignore_N_termini=False):
    if ignore_N_termini and atom_name[:2] == "HT":
        return False
    if ignore_backbone_hydrogen and (atom_name == "H" or atom_name == "HN"):
        return False  # usefull to keep only backbone hydrogens
    if atom_name[0] == "H":
        return True
    elif atom_name[0].isdigit() and atom_name[1] == "H":
        return True
    return False


def add_residue_contacts(
    polymer_crosslinked,
    overall_map,
    AVOID_CONTACTS_WITH_COSECUTIVE_RESIDUES=2,
    AVOID_FOR_BACKBONE_ONLY=False,
    backbone_atoms=["CA", "N", "C", "O"],
    contact_distance=5.0,
    h_bond_distance_cutoff=3.2,
    compute_protection_factor=True,
    keep_only_best_hb=True,
    beta_c=0.35,
    beta_hb=2.0,
):
    """
    h_bond_distance_cutoff is the N to O distance, not the distance to the actual H atom.
    can also compute_protection_factor
    """
    contact_dict, _ = get_contact_dict(polymer_crosslinked)
    print("  add_residue_contacts(): contact_dict computed")
    contact_squared_distance = contact_distance * contact_distance
    hbond_squared_distance = h_bond_distance_cutoff * h_bond_distance_cutoff
    for ch in polymer_crosslinked:
        for residue in polymer_crosslinked[ch].seq:
            ov_id1 = residue.overall_id
            residue.N_hc = 0  # number of heavy atom contacts, note that HB are also counted as heavy contacts
            residue.heavy_contacts = (
                {}
            )  # a dictionary whose keys are tuples with (residue_in_contact_ov_id, atom_name_of_res_in_contact) and value is the ( atom_name_closer_to_residue_in_contact, distance )
            residue.N_bhb = 0  # number of backbone HB
            residue.bhb = (
                {}
            )  # backbone Hydrogen bonds. Keys are N or O value is (residue_in_contact_ov_id, atom_name_of_res_in_contact, distance)
            residue.N_shb = 0  # number of sidechain HB
            residue.shb = (
                {}
            )  # sidechain Hydrogen bonds. Keys are sidechain atom names value is (residue_in_contact_ov_id, atom_name_of_res_in_contact, distance)
            for ov_id in contact_dict[ov_id1]:  # loops on residues in the vicinity
                res2 = overall_map[ov_id]
                close_on_sequence = False
                if (
                    abs(ov_id1 - ov_id) <= AVOID_CONTACTS_WITH_COSECUTIVE_RESIDUES
                    and residue.chain_id == res2.chain_id
                ):
                    if not AVOID_FOR_BACKBONE_ONLY:
                        continue
                    close_on_sequence = True
                for (
                    target_atom
                ) in residue.residue:  # loop on atoms of the residue we are considering
                    if is_hydrogen(target_atom.name):
                        continue
                    for (
                        atom
                    ) in (
                        res2.residue
                    ):  # loop on atoms of the other residue in the vicinity
                        if close_on_sequence and atom.name in backbone_atoms:
                            continue
                        if not is_hydrogen(
                            atom.name
                        ):  # we want the contacts between heavy atoms
                            dist2 = atom_squared_distance(
                                residue.residue,
                                res2.residue,
                                atom_name1=target_atom.name,
                                atom_name2=atom.name,
                            )
                            if dist2 < contact_squared_distance:
                                # update heavy atom contacts
                                d = numpy.sqrt(dist2)
                                if (ov_id, atom.name) not in residue.heavy_contacts:
                                    residue.N_hc += 1
                                    residue.heavy_contacts[(ov_id, atom.name)] = (
                                        target_atom.name,
                                        d,
                                    )
                                elif d < residue.heavy_contacts[(ov_id, atom.name)][1]:
                                    residue.heavy_contacts[(ov_id, atom.name)] = (
                                        target_atom.name,
                                        d,
                                    )
                                # Hydrogen Bonds
                                if dist2 < hbond_squared_distance:  # check for HB
                                    t1 = atom_type(
                                        target_atom.name
                                    )  # get atom type from atom name
                                    t2 = atom_type(atom.name)
                                    if (t1 == "N" or t1 == "S") and t2 == "O":
                                        if target_atom.name == "N":  # backbone HB
                                            residue.N_bhb += 1  # update backbone HB
                                            update_hb(
                                                residue.bhb,
                                                target_atom,
                                                ov_id,
                                                atom,
                                                d,
                                                keep_only_best_hb=keep_only_best_hb,
                                            )
                                        else:
                                            residue.N_shb += 1  # update sidechain HB
                                            update_hb(
                                                residue.shb,
                                                target_atom,
                                                ov_id,
                                                atom,
                                                d,
                                                keep_only_best_hb=keep_only_best_hb,
                                            )
                                    elif t1 == "O" and (t2 == "N" or t2 == "S"):
                                        if target_atom.name == "O":  # backbone HB
                                            residue.N_bhb += 1  # update backbone HB
                                            update_hb(
                                                residue.bhb,
                                                target_atom,
                                                ov_id,
                                                atom,
                                                d,
                                                keep_only_best_hb=keep_only_best_hb,
                                            )
                                        else:
                                            residue.N_shb += 1  # update sidechain HB
                                            update_hb(
                                                residue.shb,
                                                target_atom,
                                                ov_id,
                                                atom,
                                                d,
                                                keep_only_best_hb=keep_only_best_hb,
                                            )

            if compute_protection_factor:
                residue.protection_factor = float(
                    beta_c * residue.N_hc + beta_hb * residue.N_bhb
                )
    return


def update_hb(dict_to_update, target_atom, ov_id, atom, d, keep_only_best_hb=True):
    """
    auxiliary function used by add_residue_contacts()
    """
    if target_atom.name in dict_to_update:
        if keep_only_best_hb:
            if d < dict_to_update[target_atom.name][2]:
                dict_to_update[target_atom.name] = (
                    ov_id,
                    atom.name,
                    d,
                )  # overwrite with better donor/acceptor pair
        else:
            dict_to_update[target_atom.name] += [(ov_id, atom.name, d)]
    else:
        if keep_only_best_hb:
            dict_to_update[target_atom.name] = (ov_id, atom.name, d)
        else:
            dict_to_update[target_atom.name] = [(ov_id, atom.name, d)]  # start a list


def add_contact_order(polymer, debug=False):
    if "N_hc" not in dir(list(polymer.values())[0].seq[0]):
        add_residue_contacts_from_chimera(polymer.pdbfile, polymer)
    for ch in polymer:
        contacts = []
        total_DS = 0
        for j, res in enumerate(polymer[ch].seq):
            try:
                for contact_ovid in res.residue_contact:
                    if set([res.overall_id, contact_ovid]) not in contacts:
                        contacts += [set([res.overall_id, contact_ovid])]
                        total_DS += abs(res.overall_id - contact_ovid)
            except Exception:
                sys.stderr.write(
                    "\nException for residue at index %d chain %s pdb_id %d\n"
                    % (j, ch, res.pdb_id)
                )
                raise
        polymer[ch].contact_order = total_DS / float(
            len(polymer[ch].seq) * len(contacts)
        )
        polymer[ch].total_DS = total_DS
        polymer[ch].residue_contact = contacts
        if debug:
            print(
                "chain ",
                ch,
                "total_DS",
                total_DS,
                "L",
                len(polymer[ch].seq),
                "N",
                len(contacts),
            )
    return


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


def chimera_hbond(
    pdbfile,
    get_also_contacts=True,
    run_only_on_selection=None,
    out_path=None,
    out_red=" >/dev/null 2>&1",
    chimera_script=chimera_hb_script,
    remove_tmps=True,
    path_to_chimera_bin="",
):
    """
    runs chimera nogui to get results of function findhbonds within chiemra 
      if get_also_contacts it runs also findclashes but using the cutoff parameters to find atoms in contact (which includes clashes). Contacts are established using the VDW radii
    out_path determines the folder where the Hbond chimera file (and if asked the contacts file) is printed
    return HB_filename, Contacts_filename
    """
    if len(path_to_chimera_bin) > 0 and path_to_chimera_bin[-1] != "/":
        path_to_chimera_bin += "/"
    path, pdb_id, ext = get_file_path_and_extension(pdbfile)
    if out_red == None:
        out_red = ""
    tmpscript = "tmp_chimera_hb.py"
    out = open(path + tmpscript, "w")
    out.write(chimera_script)
    out.close()
    cwd = os.getcwd()
    if out_path == None or out_path == "":
        out_path = cwd
    if out_path[-1] != "/":
        out_path += "/"
    add_to_path = ""
    try:
        os.chdir(path)
        if " " in out_path:
            add_to_path = os.getcwd() + "/"
            sys.stderr.write(
                "**WARNING** in chimera_hbond() empty space found in out_path. Trying to save Chimera files in %s\n"
                % (add_to_path)
            )
            out_path = ""
        if run_only_on_selection != None and run_only_on_selection != "":
            selection = " -sel " + run_only_on_selection
        else:
            selection = ""
        inputfile = pdb_id + ext
        out_hb = out_path + pdb_id + "_HB.txt"
        if get_also_contacts:
            out_cc = out_path + pdb_id + "_contacts.txt"
            command = (
                path_to_chimera_bin
                + 'chimera --nogui  --nostatus --script "%s %s%s -out %s -contacts %s"'
                % (tmpscript, inputfile, selection, out_hb, out_cc)
            )
        else:
            out_cc = None
            command = (
                path_to_chimera_bin
                + 'chimera --nogui  --nostatus --script "%s %s%s -out %s"'
                % (tmpscript, inputfile, selection, out_hb)
            )
        print(command)
        os.system(command)
        if remove_tmps:
            os.system("rm -f %s %sc" % (tmpscript, tmpscript))  # deletes .py and .pyc
    except Exception:
        os.chdir(cwd)
        raise
    os.chdir(cwd)
    return add_to_path + out_hb, add_to_path + out_cc


def add_residue_contacts_from_chimera(
    pdbfile,
    polymer,
    chimera_selection=None,
    resnames_to_skip=["HOH"],
    remove_chimera_files=True,
):
    """
    it adds to each residue class (in seq) the following fileds :
      N_hc -> number of heavy atom contacts done with the residue, note that HB are also counted as heavy contacts. Atoms belonging to the adjacent residues are excluded.
      atom_contact -> a list of tuple with the heavy atoms in contact in the form (atom_in_this_residue, res_in_contact_overall_id, atom_in_res_in_contact, atom_distance)
      residue_contact -> a list of tuple of unique residue overall_id corresponding to the residue contacting this one (it does not distinguish how many atoms are in contact between the two residues).
      N_hb -> number of atoms involved in hydrogen bonds with atoms of the residue (either donor or acceptors). Atoms belonging to the adjacent residues are excluded.
      atom_hb -> a list of tuple with the atoms involved in hydrogen bonds in the form (atom_in_this_residue, res_in_contact_overall_id, atom_in_res_in_contact, atom_distance)
      residue_hb -> a list of tuple of unique residue overall_id corresponding to the residue forming hydrogen bonds with this one (it does not distinguish how many atoms are in contact between the two residues).
    """
    HB_filename, Contacts_filename = chimera_hbond(
        pdbfile, get_also_contacts=True, run_only_on_selection=chimera_selection
    )
    # first add Heavy atoms contacts
    add_chimera_info_to_residues(
        polymer,
        Contacts_filename,
        resnames_to_skip=resnames_to_skip,
        atom_contact_count="N_hc",
        atom_contact_list="atom_contact",
        residue_contanct_list="residue_contact",
        fun_rest=lambda x: float(x[1]),
    )
    # then add HB
    add_chimera_info_to_residues(
        polymer,
        HB_filename,
        resnames_to_skip=resnames_to_skip,
        atom_contact_count="N_hb",
        atom_contact_list="atom_hb",
        residue_contanct_list="residue_hb",
        fun_rest=lambda x: float(x[2]),
    )
    if remove_chimera_files:
        os.remove(Contacts_filename)
        os.remove(HB_filename)
    return


def add_chimera_info_to_residues(
    polymer,
    chimera_filename,
    resnames_to_skip=["HOH"],
    dont_process_hydrogens=True,
    skip_adiacent_residues=True,
    check_empties=True,
    atom_contact_count="N_hc",
    atom_contact_list="atom_contact",
    residue_contanct_list="residue_contact",
    fun_rest=lambda x: float(x[1]),
):
    """
    auxiliary function used by add_residue_contacts_from_chimera()
      fun_rest returns the distance from rest
    """
    if fun_rest == None:
        fun_rest = lambda x: x  # identity function
    for j, line in enumerate(open(chimera_filename)):
        try:
            out = parse_chimerafile_line(line)
            if out == None:
                continue
            aa1, pdb_id1, ch1, atom1, aa2, pdb_id2, ch2, atom2, rest = out
        except Exception:
            sys.stderr.write(
                "\nException at line %d of file %s\n" % (j + 1, chimera_filename)
            )
            raise
        if aa1 in resnames_to_skip or aa2 in resnames_to_skip:
            continue  # skip contacts with non interesting residues
        if dont_process_hydrogens and (is_hydrogen(atom1) or is_hydrogen(atom2)):
            continue
        if ch1 == None:
            ch1 = list(polymer.keys())[0]
        if ch2 == None:
            ch2 = list(polymer.keys())[0]
        id1 = polymer[ch1].aa_pdb_id_map[pdb_id1]
        id2 = polymer[ch2].aa_pdb_id_map[pdb_id2]
        if aa1 != polymer[ch1].seq[id1].resname:
            sys.stderr.write(
                "**WARNING** in add_residue_contacts_from_chimera() expecting %s at pdb id %d chain %s but found %s instead!\n"
                % (aa1, pdb_id1, ch1, polymer[ch1].seq[id1].resname)
            )
        if aa2 != polymer[ch2].seq[id2].resname:
            sys.stderr.write(
                "**WARNING** in add_residue_contacts_from_chimera() expecting %s at pdb id %d chain %s but found %s instead! (2)\n"
                % (aa2, pdb_id2, ch2, polymer[ch2].seq[id2].resname)
            )
        if skip_adiacent_residues and ch1 == ch2 and abs(id1 - id2) <= 1:
            continue
        ov_id1 = polymer[ch1].seq[id1].overall_id
        ov_id2 = polymer[ch2].seq[id2].overall_id
        if atom_contact_count not in dir(polymer[ch1].seq[id1]):
            polymer[ch1].seq[id1].update(
                atom_contact_count, 0, check=False
            )  # number of atoms contacting this residue
            polymer[ch1].seq[id1].update(
                atom_contact_list, [], check=False
            )  # a list of tuple with (atom_in_this_residue, res_in_contact_overall_id, atom_in_res_in_contact, atom_distance)
            polymer[ch1].seq[id1].update(
                residue_contanct_list, [], check=False
            )  # a list of tuple of unique residue overall_id (it does not distinguish how many atoms are in contact between the two residues).
        if atom_contact_count not in dir(polymer[ch2].seq[id2]):
            polymer[ch2].seq[id2].update(
                atom_contact_count, 0, check=False
            )  # number of heavy atom contacts, note that HB are also counted as heavy contacts
            polymer[ch2].seq[id2].update(atom_contact_list, [], check=False)
            polymer[ch2].seq[id2].update(residue_contanct_list, [], check=False)
        # update existing classes
        polymer[ch1].seq[id1].__dict__[atom_contact_count] += 1
        polymer[ch2].seq[id2].__dict__[atom_contact_count] += 1
        if ov_id2 not in polymer[ch1].seq[id1].__dict__[residue_contanct_list]:
            polymer[ch1].seq[id1].__dict__[residue_contanct_list] += [ov_id2]
        if ov_id1 not in polymer[ch2].seq[id2].__dict__[residue_contanct_list]:
            polymer[ch2].seq[id2].__dict__[residue_contanct_list] += [ov_id1]
        polymer[ch1].seq[id1].__dict__[atom_contact_list] += [
            (atom1, ov_id2, atom2, fun_rest(rest))
        ]
        polymer[ch2].seq[id2].__dict__[atom_contact_list] += [
            (atom2, ov_id1, atom1, fun_rest(rest))
        ]

    if (
        check_empties
    ):  # add variables to residues not found in file so that errors won't be generated
        for ch in polymer:
            for res in polymer[ch].seq:
                if atom_contact_count not in dir(res):
                    res.update(atom_contact_count, 0, check=False)
                    res.update(atom_contact_list, [], check=False)
                    res.update(residue_contanct_list, [], check=False)
    return


def parse_chimerafile_line(line):
    line = line.split()
    if len(line) < 8 or len(line[0]) >= 5 or line[0] == "H-bonds":
        return None
    ch1, ch2 = None, None
    aa1, pdb_id1, atom1, aa2, pdb_id2, atom2 = line[:6]
    if "." in pdb_id1:
        pdb_id1, ch1 = pdb_id1.split(".")
    if "." in pdb_id2:
        pdb_id2, ch2 = pdb_id2.split(".")
    return aa1, int(pdb_id1), ch1, atom1, aa2, int(pdb_id2), ch2, atom2, line[6:]
