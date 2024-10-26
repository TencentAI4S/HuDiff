# (c) 2023 Sormannilab


import sys, inspect, os
from traceback import print_exc

import matplotlib


if os.environ.get("DISPLAY") is not None:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        # print_exc(file=sys.stderr)
        sys.stderr.write(
            "  when import plotter -> not using TkAgg as failed matplotlib.use (not usually a problem)\n"
        )
        pass

from matplotlib import pyplot as plt
import matplotlib.font_manager
from matplotlib.colors import TwoSlopeNorm

import numpy, scipy.stats, scipy.signal, scipy.spatial


from collections import OrderedDict

import colorsys


try:  # custom module, not necessarily needed (only in some functions if plot=True is given)
    from . import misc  # see if it is in same folder
except ImportError:
    try:  # see if it's in pythonpath
        import misc
    except ImportError:

        class tmpmisc:
            def Fit(self, *args, **kwargs):
                raise Exception("misc MODULE NOT AVAILABLE. Cannot Fit\n")

            def FitNew(self, *args, **kwargs):
                raise Exception("misc MODULE NOT AVAILABLE. Cannot Fit\n")

            def ScaleInRange(self, *args, **kwargs):
                raise Exception("misc MODULE NOT AVAILABLE\n")

        misc = tmpmisc()


try:  # custom module, not necessarily needed (only in some functions if plot=True is given)
    from . import csv_dict  # see if it is in same folder
except ImportError:
    try:  # see if it's in pythonpath
        import csv_dict
    except ImportError:

        class tmpcsv_dict:
            def Data(self, *args, **kwargs):
                raise Exception("csv_dict module not available, cannot use Data class")

        csv_dict = tmpcsv_dict()

try:  # custom module, not necessarily needed (only in some functions if plot=True is given)
    from . import mybio  # see if it is in same folder
except ImportError:
    try:  # see if it's in pythonpath
        import mybio
    except ImportError:

        class tmpmybio:
            def get_SeqRecords(self, *args, **kwargs):
                raise Exception("mybio module not available, cannot use get_SeqRecords")

            def tcoffee_alignment(self, *args, **kwargs):
                raise Exception(
                    "mybio module not available, cannot use tcoffee_alignment"
                )

            def get_CDR(self, *args, **kwargs):
                raise Exception("mybio module not available, cannot use get_CDR")

            def Anarci_alignment(self, *args, **kwargs):
                raise Exception(
                    "mybio module not available, cannot use Anarci_alignment"
                )

            def conservation_index_variance_based(self, *args, **kwargs):
                raise Exception(
                    "mybio module not available, cannot use conservation_index_variance_based"
                )

        mybio = tmpmybio()


plt.rcdefaults()
plt.rc("figure", facecolor="white")


# COLOR GENERATOR: http://www.2createawebsite.com/build/hex-colors.html#hextool


monospaced_font = "monospace"  # font-family actually

text_sizes = {
    "value_labels": 18,
    "xlabels": 18,
    "xlabels_many": 11,
    "xlabel": 22,
    "ylabel": 22,
    "title": 24,
    "legend_size": 18,
}

publication = {
    "value_labels": 22,
    "xlabels": 22,
    "xlabels_many": 12,
    "xlabel": 30,
    "ylabel": 30,
    "title": 30,
    "legend_size": 21,
}
publication_cm = {
    "value_labels": 6,
    "xlabels": 6,
    "xlabel": 8,
    "ylabel": 8,
    "title": 8,
    "legend_size": 8,
}
publication_small = {
    "value_labels": 26,
    "xlabels": 30,
    "xlabels_many": 14,
    "xlabel": 30,
    "ylabel": 30,
    "title": 30,
    "legend_size": 26,
}
# if you set the size for 'default' all the figures will come out of that size disregarding their type. Otherwise you can change the figure size for each type (key in dictionary)
default_figure_sizes = {
    "all_tight": False,
    "use_cm": False,
    "dpi": 300,
    "default": None,
    "histogram": (8, 8),
    "scatter": (8, 8),
    "bar": (8, 8),
    "pie": (8, 8),
    "multibar": (8, 8),
    "boxplot": (8, 8),
    "sequence": (14.2, 8),
    "profile": (12, 8),
}
default_error_bars = {"capsize": 4, "capthick": 1.0, "elinewidth": 1.0}
default_parameters = {
    "hgrid": True,
    "vgrid": True,
    "frame": True,
    "set_publish": False,
    "ticks_nbins": 5,
    "seq_max_res_per_plot": None,
    "markersize": 8,
    "linewidth": 1.5,
    "barlinewidth": 0.5,
    "value_label_text_offset": 8,
    "scatter_marker": "x",
    "fit_allow_extra_fraction": 0.005,
}
grid_parameters = (
    {  # hgrid and vgrid are in default_parameters. these contain only grid styles
        "hcolor": "DarkGrey",
        "vcolor": "DarkGrey",
        "h_ls": ":",
        "v_ls": ":",
        "h_lw": 0.75,
        "v_lw": 0.75,
    }
)

plt.rc("xtick", labelsize=text_sizes["xlabels"])
plt.rc("ytick", labelsize=text_sizes["xlabels"])
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
def set_tiny_labels(
    labelsize=10, major_width=1, major_size=3, minor_width=0.75, minor_size=2
):
    plt.rc("xtick", labelsize=labelsize)
    plt.rc("ytick", labelsize=labelsize)
    plt.rc("ytick.major", width=major_width, size=major_size)
    plt.rc("ytick.minor", width=minor_width, size=minor_size)
    plt.rc("xtick.major", width=major_width, size=major_size)
    plt.rc("xtick.minor", width=minor_width, size=minor_size)
    text_sizes["value_labels"] = 10
    text_sizes["xlabels"] = 10
    text_sizes["xlabels_many"] = 7
    text_sizes["xlabel"] = 12
    text_sizes["ylabel"] = 12
    text_sizes["title"] = 12
    text_sizes["legend_size"] = 12
    return


def set_publish(
    all_same_figure_size=False,
    thick_ticks=True,
    axis_tickness=True,
    all_tight=False,
    small_figure=True,
    no_grids=True,
    text_sizes=text_sizes,
    publication=publication,
    publication_small=publication_small,
    default_figure_sizes=default_figure_sizes,
):
    default_parameters["set_publish"] = True
    default_error_bars["capsize"] = 8
    default_error_bars["capthick"] = 2
    default_error_bars["elinewidth"] = 2
    default_parameters["value_label_text_offset"] = 10
    grid_parameters["h_lw"], grid_parameters["v_lw"] = 1.25, 1.25
    grid_parameters["hcolor"], grid_parameters["vcolor"] = "black", "black"
    if small_figure:
        for k in publication:
            text_sizes[k] = publication_small[
                k
            ]  # with an = or with .copy it does not work
        # default_parameters['seq_max_res_per_plot']=50 # now auto-set
        default_parameters["markersize"] = 15
        default_parameters["linewidth"] = 2.5
    else:
        for k in publication:
            text_sizes[k] = publication[k]  # with an = or with .copy it does not work
        # default_parameters['seq_max_res_per_plot']=100 # now auto-set
        default_parameters["markersize"] = 12
        default_parameters["linewidth"] = 2
    plt.rc("xtick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick.major", width=1.5, size=6)
    plt.rc("ytick.minor", width=1.0, size=3)
    plt.rc("xtick.major", width=1.5, size=6)
    plt.rc("xtick.minor", width=1.0, size=3)
    plt.rcParams["lines.linewidth"] = 2
    if all_tight:
        default_figure_sizes["all_tight"] = True
    if no_grids:
        for p in ["frame", "vgrid", "hgrid"]:
            default_parameters[p] = False
    if thick_ticks != False:
        plt.rc("ytick.major", width=2.5, size=10)
        plt.rc("ytick.minor", width=2, size=6)
        plt.rc("xtick.major", width=2.5, size=10)
        plt.rc("xtick.minor", width=2, size=6)
    if axis_tickness:
        plt.rc("axes", linewidth=2, edgecolor="black")
    if (
        all_same_figure_size != False
    ):  # I could just change default_figure_sizes['default'] but this is safer
        if type(all_same_figure_size) is tuple:
            for s in default_figure_sizes:
                if s == "all_tight" or s == "use_cm" or s == "dpi":
                    continue
                default_figure_sizes[s] = all_same_figure_size
        else:
            for s in default_figure_sizes:
                if s == "all_tight" or s == "use_cm" or s == "dpi":
                    continue
                default_figure_sizes[s] = (10, 10)
    return


def set_publish_s2D(
    all_same_figure_size=False,
    thick_ticks=True,
    axis_tickness=True,
    no_grids=True,
    text_sizes=text_sizes,
    publication=publication,
    default_figure_sizes=default_figure_sizes,
):
    default_parameters["set_publish"] = True
    default_error_bars["capsize"] = 8
    default_error_bars["capthick"] = 2
    default_error_bars["elinewidth"] = 2
    for k in publication:
        text_sizes[k] = publication[k]  # with an = or with .copy it does not work
    default_parameters["seq_max_res_per_plot"] = 100
    default_parameters["all_tight"] = True
    plt.rc("xtick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick.major", width=1.5, size=6)
    plt.rc("ytick.minor", width=1.0, size=3)
    plt.rc("xtick.major", width=1.5, size=6)
    plt.rc("xtick.minor", width=1.0, size=3)
    default_figure_sizes["sequence"] = (14.2, 8)
    default_figure_sizes["all_tight"] = True
    if no_grids:
        for p in ["frame", "vgrid", "hgrid"]:
            default_parameters[p] = False

    if thick_ticks:
        plt.rc("ytick.major", width=2.5, size=10)
        plt.rc("ytick.minor", width=2, size=6)
        plt.rc("xtick.major", width=2.5, size=10)
        plt.rc("xtick.minor", width=2, size=6)
    if axis_tickness:
        plt.rc("axes", linewidth=2, edgecolor="black")
    if (
        all_same_figure_size != False
    ):  # I could just change default_figure_sizes['default'] but this is safer
        if type(all_same_figure_size) is tuple:
            for s in default_figure_sizes:
                if s == "all_tight" or s == "use_cm" or s == "dpi":
                    continue
                default_figure_sizes[s] = all_same_figure_size
        else:
            for s in default_figure_sizes:
                if s == "all_tight" or s == "use_cm" or s == "dpi":
                    continue
                default_figure_sizes[s] = 10, 10
    return


def set_publish_cm(
    all_same_figure_size=False,
    thick_ticks=False,
    axis_tickness=True,
    small_figure=True,
    text_sizes=text_sizes,
    publication=publication,
    publication_small=publication_small,
    default_figure_sizes=default_figure_sizes,
):
    """
    very beta, for figure sizes in centimeters rather than in inches..
    """
    default_figure_sizes["use_cm"] = True
    if small_figure:
        default_error_bars["capsize"] = 3
        default_error_bars["capthick"] = 1
        default_error_bars["elinewidth"] = 1
        for k in publication:
            text_sizes[k] = publication_cm[
                k
            ]  # with an = or with .copy it does not work
    else:
        default_error_bars["capsize"] = 2
        for k in publication:
            text_sizes[k] = publication_cm[
                k
            ]  # with an = or with .copy it does not work
    plt.rc("xtick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick.major", width=1.0, size=3)
    plt.rc("ytick.minor", width=0.5, size=1.5)
    plt.rc("xtick.major", width=0.5, size=3)
    plt.rc("xtick.minor", width=1.0, size=1.5)

    default_figure_sizes["all_tight"] = True
    if thick_ticks != False:
        plt.rc("ytick.major", width=2.5, size=10)
        plt.rc("ytick.minor", width=2, size=6)
        plt.rc("xtick.major", width=2.5, size=10)
        plt.rc("xtick.minor", width=2, size=6)
    if axis_tickness:
        plt.rc("axes", linewidth=1, edgecolor="black")
    if (
        all_same_figure_size != False
    ):  # I could just change default_figure_sizes['default'] but this is safer
        if type(all_same_figure_size) is tuple:
            for s in default_figure_sizes:
                if s == "all_tight" or s == "use_cm" or s == "dpi":
                    continue
                default_figure_sizes[s] = all_same_figure_size
        else:
            for s in default_figure_sizes:
                if s == "all_tight" or s == "use_cm" or s == "dpi":
                    continue
                default_figure_sizes[s] = (5, 5)
    return


class cycle_list(list):
    def __init__(self, l=[]):
        self.n = 0
        list.__init__(self, l)

    def __getitem__(self, y):
        # x.__getitem__(y) <==> x[y]
        if type(y) is int and y >= len(self):
            y = y % len(self)
        return list.__getitem__(self, y)

    def next(self):
        self.n += 1
        return self.__getitem__(self.n - 1)


# RGB colors, 0 to 255
# iworkpalette=[ (64,114,202),(108,164,62),(216,194,53),(209,136,39),(198,75,28),(116,83,166)  ]
iworkpalette = cycle_list(
    [
        (0.25098039215686274, 0.4470588235294118, 0.792156862745098),
        (0.4235294117647059, 0.6831372549019608, 0.24313725490196078),
        (0.8470588235294118, 0.7507843137254902, 0.16784313725490197),
        (0.8196078431372549, 0.5333333333333333, 0.15294117647058825),
        (0.7764705882352941, 0.29411764705882354, 0.10980392156862745),
        (0.4549019607843137, 0.3254901960784314, 0.6509803921568628),
    ]
)

# iworkpalette2=cycle_list( [(92,194,252),(115   , 253   , 234    ),(136,    250,    78    ),(250 ,   226   , 51    ),(255  , 147   ,1       ),(255   , 150   , 141    ),(238,34,13    ),(255,141,198    ) ])
iworkpalette2 = cycle_list(
    [
        (0.3607843137254902, 0.7607843137254902, 0.9882352941176471),
        (0.45098039215686275, 0.9921568627450981, 0.9176470588235294),
        (0.5333333333333333, 0.9803921568627451, 0.3058823529411765),
        (0.9803921568627451, 0.8862745098039215, 0.2),
        (1.0, 0.5764705882352941, 0.00392156862745098),
        (1.0, 0.5882352941176471, 0.5529411764705883),
        (0.9333333333333333, 0.13333333333333333, 0.050980392156862744),
        (1.0, 0.5529411764705883, 0.7764705882352941),
    ]
)
transp_iworkpalette = cycle_list(
    [
        (0.25098039215686274, 0.4470588235294118, 0.792156862745098, 0.4),
        (0.4235294117647059, 0.6831372549019608, 0.24313725490196078, 0.4),
        (0.8470588235294118, 0.7507843137254902, 0.16784313725490197, 0.4),
        (0.8196078431372549, 0.5333333333333333, 0.15294117647058825, 0.4),
        (0.7764705882352941, 0.29411764705882354, 0.10980392156862745, 0.4),
        (0.4549019607843137, 0.3254901960784314, 0.6509803921568628, 0.4),
    ]
)
twocols_palette = cycle_list(
    [(62 / 255.0, 112 / 255.0, 182 / 255.0), (255 / 255.0, 133 / 255.0, 72 / 255.0)]
)
threecols_palette = cycle_list(["#E44424", "#67BCDB", "#A2AB58"])
fourcols_palette = cycle_list(["#2B2B2B", "#DE1B1B", "#F6F6F6", "#E9E581"])

try:
    import brewer2mpl

    color_set = cycle_list(
        brewer2mpl.get_map("Paired", "qualitative", 12).mpl_colors
    )  # Get "Set3" colors from ColorBrewer (all colorbrewer scales: http://bl.ocks.org/mbostock/5577023)
    color_set1 = cycle_list(
        brewer2mpl.get_map("Set1", "qualitative", 9).mpl_colors
    )  # Get "Set1" colors from ColorBrewer (all colorbrewer scales: http://bl.ocks.org/mbostock/5577023)
    color_set2 = cycle_list(
        brewer2mpl.get_map("Set2", "qualitative", 8).mpl_colors
    )  # Get "Set2" colors from ColorBrewer (all colorbrewer scales: http://bl.ocks.org/mbostock/5577023)
except ImportError:
    color_set = iworkpalette
    color_set1 = threecols_palette
    color_set2 = fourcols_palette

almost_black = "#262626"  # Save a nice dark grey as a variable
# see http://www.w3schools.com/html/html_colornames.asp
html_dict = {
    "Blue": "#0000FF",
    "Pink": "#FFC0CB",
    "Purple": "#800080",
    "Fuchsia": "#FF00FF",
    "LawnGreen": "#7CFC00",
    "AliceBlue": "#F0F8FF",
    "Crimson": "#DC143C",
    "White": "#FFFFFF",
    "NavajoWhite": "#FFDEAD",
    "Cornsilk": "#FFF8DC",
    "Bisque": "#FFE4C4",
    "PaleGreen": "#98FB98",
    "Brown": "#A52A2A",
    "DarkTurquoise": "#00CED1",
    "DarkGreen": "#006400",
    "DarkGoldenRod": "#B8860B",
    "MediumOrchid": "#BA55D3",
    "Chocolate": "#D2691E",
    "PapayaWhip": "#FFEFD5",
    "Olive": "#808000",
    "LightSlateGray": "#778899",
    "PeachPuff": "#FFDAB9",
    "Plum": "#DDA0DD",
    "MediumAquaMarine": "s",
    "MintCream": "#F5FFFA",
    "CornflowerBlue": "#6495ED",
    "HotPink": "#FF69B4",
    "DarkBlue": "#00008B",
    "LimeGreen": "#32CD32",
    "DeepSkyBlue": "#00BFFF",
    "DarkKhaki": "#BDB76B",
    "Yellow": "#FFFF00",
    "Gainsboro": "#DCDCDC",
    "MistyRose": "#FFE4E1",
    "SandyBrown": "#F4A460",
    "DeepPink": "#FF1493",
    "SeaShell": "#FFF5EE",
    "Magenta": "#FF00FF",
    "DarkCyan": "#008B8B",
    "GreenYellow": "#ADFF2F",
    "DarkOrchid": "#9932CC",
    "LightGoldenRodYellow": "s",
    "OliveDrab": "#6B8E23",
    "Chartreuse": "#7FFF00",
    "Peru": "#CD853F",
    "MediumTurquoise": "s",
    "Orange": "#FFA500",
    "Red": "#FF0000",
    "Wheat": "#F5DEB3",
    "LightCyan": "#E0FFFF",
    "LightSeaGreen": "#20B2AA",
    "BlueViolet": "#8A2BE2",
    "Cyan": "#00FFFF",
    "MediumPurple": "#9370DB",
    "MidnightBlue": "#191970",
    "Coral": "#FF7F50",
    "PaleTurquoise": "#AFEEEE",
    "Gray": "#808080",
    "MediumSeaGreen": "#3CB371",
    "Moccasin": "#FFE4B5",
    "Turquoise": "#40E0D0",
    "DarkSlateBlue": "#483D8B",
    "Green": "#008000",
    "Beige": "#F5F5DC",
    "Teal": "#008080",
    "Azure": "#F0FFFF",
    "LightSteelBlue": "#B0C4DE",
    "Tan": "#D2B48C",
    "AntiqueWhite": "#FAEBD7",
    "SkyBlue": "#87CEEB",
    "GhostWhite": "#F8F8FF",
    "HoneyDew": "#F0FFF0",
    "FloralWhite": "#FFFAF0",
    "LavenderBlush": "#FFF0F5",
    "SeaGreen": "#2E8B57",
    "Lavender": "#E6E6FA",
    "BlanchedAlmond": "#FFEBCD",
    "DarkOliveGreen": "#556B2F",
    "DarkSeaGreen": "#8FBC8F",
    "SpringGreen": "#00FF7F",
    "Navy": "#000080",
    "Orchid": "#DA70D6",
    "Salmon": "#FA8072",
    "IndianRed": "#CD5C5C",
    "Snow": "#FFFAFA",
    "SteelBlue": "#4682B4",
    "MediumSlateBlue": "s",
    "Black": "#000000",
    "LightBlue": "#ADD8E6",
    "Ivory": "#FFFFF0",
    "MediumVioletRed": "s",
    "DarkViolet": "#9400D3",
    "DarkGray": "#A9A9A9",
    "SaddleBrown": "#8B4513",
    "DarkMagenta": "#8B008B",
    "Tomato": "#FF6347",
    "WhiteSmoke": "#F5F5F5",
    "MediumSpringGreen": "s",
    "DodgerBlue": "#1E90FF",
    "Aqua": "#00FFFF",
    "ForestGreen": "#228B22",
    "LemonChiffon": "#FFFACD",
    "Silver": "#C0C0C0",
    "LightGray": "#D3D3D3",
    "GoldenRod": "#DAA520",
    "Indigo": "#4B0082",
    "CadetBlue": "#5F9EA0",
    "LightYellow": "#FFFFE0",
    "DarkOrange": "#FF8C00",
    "PowderBlue": "#B0E0E6",
    "RoyalBlue": "#4169E1",
    "Sienna": "#A0522D",
    "Thistle": "#D8BFD8",
    "Lime": "#00FF00",
    "SlateGray": "#708090",
    "DarkRed": "#8B0000",
    "LightSkyBlue": "#87CEFA",
    "SlateBlue": "#6A5ACD",
    "YellowGreen": "#9ACD32",
    "Aquamarine": "#7FFFD4",
    "LightCoral": "#F08080",
    "DarkSlateGray": "#2F4F4F",
    "Khaki": "#F0E68C",
    "BurlyWood": "#DEB887",
    "MediumBlue": "#0000CD",
    "DarkSalmon": "#E9967A",
    "RosyBrown": "#BC8F8F",
    "LightSalmon": "#FFA07A",
    "PaleVioletRed": "#DB7093",
    "FireBrick": "#B22222",
    "Violet": "#EE82EE",
    "LightGreen": "#90EE90",
    "Linen": "#FAF0E6",
    "OrangeRed": "#FF4500",
    "PaleGoldenRod": "#EEE8AA",
    "DimGray": "#696969",
    "Maroon": "#800000",
    "LightPink": "#FFB6C1",
    "Gold": "#FFD700",
    "OldLace": "#FDF5E6",
}


palette_contrasting = [
    "Blue",
    "DeepSkyBlue",
    "cyan",
    "Lime",
    "#12AD2B",
    "DarkGreen",
    "#4E9258",
    "olive",
    "gold",
    "orange",
    "Pink",
    "Purple",
    "magenta",
    "red",
    "FireBrick",
    "Chocolate",
]

palette8 = cycle_list(
    [
        "#003f5c",
        "#2f4b7c",
        "#665191",
        "#a05195",
        "#d45087",
        "#f95d6a",
        "#ff7c43",
        "#ffa600",
    ]
)
palette20 = cycle_list(
    [
        "Navy",
        "Blue",
        "DarkCyan",
        "Cyan",
        "SkyBlue",
        "Lime",
        "SeaGreen",
        "Green",
        "Olive",
        "DarkGoldenRod",
        "Sienna",
        "Brown",
        "DarkRed",
        "Red",
        "OrangeRed",
        "Orange",
        "Gold",
        "Tan",
        "RosyBrown",
        "Fuchsia",
        "Plum",
        "LightPink",
        "MediumPurple",
        "DarkOrchid",
        "Gray",
        "Black",
    ]
)
# OLD:cycle_list( [ 'Navy', 'Blue','DarkCyan', 'Cyan', 'SkyBlue', 'Lime', 'Olive', 'Green','SeaGreen', 'DarkRed', 'Red','OrangeRed','Orange', 'Gold', 'DarkGoldenRod','Sienna','Brown','Tan','RosyBrown','Plum','Fuchsia','MediumPurple','DarkOrchid', 'LightPink', 'Gray','Black'] ) # , 'MidnightBlue','Turquoise'
palette20hex = cycle_list([html_dict[c] for c in palette20])
palette10 = cycle_list(
    [
        (0.176470588, 0.082352941, 0.509803922),
        (0.08627451, 0.141176471, 0.537254902),
        (0.090196078, 0.301960784, 0.560784314),
        (0.090196078, 0.478431373, 0.588235294),
        (0.094117647, 0.611764706, 0.552941176),
        (0.094117647, 0.635294118, 0.396078431),
        (0.098039216, 0.662745098, 0.223529412),
        (0.156862745, 0.68627451, 0.098039216),
        (0.364705882, 0.709803922, 0.098039216),
        (0.588235294, 0.737254902, 0.101960784),
    ]
)
palette_OG = cycle_list(
    [  # orange green
        (0.929411765, 0.678431373, 0.109803922),
        (0.878431373, 0.749019608, 0.121568627),
        (0.82745098, 0.803921569, 0.129411765),
        (0.71372549, 0.776470588, 0.137254902),
        (0.592156863, 0.729411765, 0.141176471),
        (0.482352941, 0.678431373, 0.145098039),
        (0.384313725, 0.62745098, 0.145098039),
        (0.301960784, 0.576470588, 0.145098039),
        (0.231372549, 0.525490196, 0.145098039),
        (0.184313725, 0.474509804, 0.160784314),
        (0.039215686, 0.364705882, 0.094117647),
        (0.02745098, 0.254901961, 0.035294118),
    ]
)

# can do
# iworkpalette_full=plotter.cycle_list(plotter.iworkpalette_full[1::4]+plotter.iworkpalette_full[::4]+plotter.iworkpalette_full[2::4]+plotter.iworkpalette_full[3::4]) # to make it blue, green ..., and than darker tones of the same
iworkpalette_full = cycle_list(
    [
        # greens:
        (0.435294118, 0.752941176, 0.254901961),
        (0.023529412, 0.529411765, 0.168627451),
        (0.039215686, 0.364705882, 0.094117647),
        (0.02745098, 0.254901961, 0.035294118),
        # blue:
        # (120/255.,   212/255., 250/255.  )  ,
        (0.321568627, 0.654901961, 0.976470588),
        (0.000000, 0.396078431, 0.752941176),
        (0.090196078, 0.305882353, 0.529411765),
        (0.00000, 0.141176471, 0.321568627),
        # yellows:
        (0.964705882, 0.82745098, 0.152941176),
        (0.862745098, 0.745098039, 0.133333333),
        (0.764705882, 0.596078431, 0.101960784),
        (0.639215686, 0.458823529, 0.066666667),
        # oranges:
        # (243/255.,    190/255.,    93 /255.   ),
        (0.952941176, 0.564705882, 0.098039216),
        (0.870588235, 0.415686275, 0.062745098),
        (0.745098039, 0.352941176, 0.043137255),
        (0.57254902, 0.274509804, 0.02745098),
        # reds:
        (0.925490196, 0.364705882, 0.341176471),
        (0.784313725, 0.145098039, 0.023529412),
        (0.525490196, 0.062745098, 0.007843137),
        (0.341176471, 0.02745098, 0.019607843),
        # purples:
        (0.701960784, 0.415686275, 0.882352941),
        (0.466666667, 0.243137255, 0.611764706),
        (0.368627451, 0.196078431, 0.482352941),
        (0.235294118, 0.121568627, 0.305882353),
        # greys:
        # (0.866666667,    0.870588235    ,0.878431373),
        # (0.654901961,    0.666666667    ,0.662745098),
        # (0.321568627,    0.345098039    ,0.37254902 )
    ]
)
iworkpalette_dict = {
    "green": cycle_list(
        [
            (0.435294118, 0.752941176, 0.254901961),
            (0.023529412, 0.529411765, 0.168627451),
            (0.039215686, 0.364705882, 0.094117647),
            (0.02745098, 0.254901961, 0.035294118),
        ]
    ),
    "blue": cycle_list(
        [
            (120 / 255.0, 212 / 255.0, 250 / 255.0),
            (0.321568627, 0.654901961, 0.976470588),
            (0.000000, 0.396078431, 0.752941176),
            (0.090196078, 0.305882353, 0.529411765),
            (0.00000, 0.141176471, 0.321568627),
        ]
    ),
    "red": cycle_list(
        [
            (0.925490196, 0.364705882, 0.341176471),
            (0.784313725, 0.145098039, 0.023529412),
            (0.525490196, 0.062745098, 0.007843137),
            (0.341176471, 0.02745098, 0.019607843),
        ]
    ),
    "yellow": cycle_list(
        [
            (0.964705882, 0.82745098, 0.152941176),
            (0.862745098, 0.745098039, 0.133333333),
            (0.764705882, 0.596078431, 0.101960784),
            (0.639215686, 0.458823529, 0.066666667),
        ]
    ),
    "orange": cycle_list(
        [
            (243 / 255.0, 190 / 255.0, 93 / 255.0),
            (0.952941176, 0.564705882, 0.098039216),
            (0.870588235, 0.415686275, 0.062745098),
            (0.745098039, 0.352941176, 0.043137255),
            (0.57254902, 0.274509804, 0.02745098),
        ]
    ),
    "purple": cycle_list(
        [
            (0.701960784, 0.415686275, 0.882352941),
            (0.466666667, 0.243137255, 0.611764706),
            (0.368627451, 0.196078431, 0.482352941),
            (0.235294118, 0.121568627, 0.305882353),
        ]
    ),
    "grey": cycle_list(
        [
            (0.866666667, 0.870588235, 0.878431373),
            (0.654901961, 0.666666667, 0.662745098),
            (0.321568627, 0.345098039, 0.37254902),
            (0.2, 0.2, 0.2),
            (0, 0, 0),
        ]
    ),
}
Reds = matplotlib.colors.LinearSegmentedColormap.from_list(
    "Reds",
    [
        (0, (0.925490196, 0.364705882, 0.341176471)),
        (0.33, (0.784313725, 0.145098039, 0.023529412)),
        (0.66, (0.525490196, 0.062745098, 0.007843137)),
        (1, (0.341176471, 0.02745098, 0.019607843)),
    ],
    N=256,
)
Blues = matplotlib.colors.LinearSegmentedColormap.from_list(
    "Blues",
    [
        (0, (0.321568627, 0.654901961, 0.976470588)),
        (0.33, (0.000000, 0.396078431, 0.752941176)),
        (0.66, (0.090196078, 0.305882353, 0.529411765)),
        (1, (0.00000, 0.141176471, 0.321568627)),
    ],
    N=256,
)
Greens = matplotlib.colors.LinearSegmentedColormap.from_list(
    "Greens",
    [
        (0, (0.435294118, 0.752941176, 0.254901961)),
        (0.33, (0.023529412, 0.529411765, 0.168627451)),
        (0.66, (0.039215686, 0.364705882, 0.094117647)),
        (1, (0.02745098, 0.254901961, 0.035294118)),
    ],
    N=256,
)

Oranges = matplotlib.colors.LinearSegmentedColormap.from_list(
    "Oranges",
    [
        (0, (243 / 255.0, 190 / 255.0, 93 / 255.0)),
        (1, (0.57254902, 0.274509804, 0.02745098)),
    ],
)
Yellows = matplotlib.colors.LinearSegmentedColormap.from_list(
    "Yellows",
    [
        (0, (0.964705882, 0.82745098, 0.152941176)),
        (1, (0.639215686, 0.458823529, 0.066666667)),
    ],
)
RedsBlues = matplotlib.colors.LinearSegmentedColormap.from_list(
    "RedsBlues",
    [
        (0, (0.925490196, 0.364705882, 0.341176471)),
        (0.14, (0.784313725, 0.145098039, 0.023529412)),
        (0.28, (0.525490196, 0.062745098, 0.007843137)),
        (0.43, (1, 0, 0)),
        (0.57, (0.0, 0.141176471, 0.321568627)),
        (0.71, (0.090196078, 0.305882353, 0.529411765)),
        (0.86, (0.0, 0.396078431, 0.752941176)),
        (1, (0.321568627, 0.654901961, 0.976470588)),
    ],
    N=256,
)

camsol_colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "camsol_colormap",
    [
        (0, "red"),
        (0.2, "orange"),
        (0.25, "DeepSkyBlue"),
        (0.95, "DeepSkyBlue"),
        (1, "blue"),
    ],
    N=256,
)
camsol_strucorr_colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "strucorr_colormap",
    [
        (0, "magenta"),
        (0.25, "purple"),
        (0.25, "DimGray"),
        (0.95, "DimGray"),
        (1, "ForestGreen"),
    ],
    N=256,
)


def camsol_to_color_val(profile):
    """
    put a profile in 0,1 in such a way that it can be colored
    """
    NewList = (
        numpy.array(profile) + 1.5
    ) * 0.4  # 0.4 is new range/old range aka 1/2.5 and new min is 0.
    NewList[NewList < 0] = 0
    NewList[NewList > 1] = 1
    return NewList


"""
color mixing with Chroma (color blending)
import chroma
c,y,m = chroma.Color((1,0,0),format='CMY'),chroma.Color((0,1,0),format='CMY'),chroma.Color((0,0,1),format='CMY') # Cyan Yellow Magenta. format='RGB' is possible as well
plotter.plot_palette([c.rgb,m.rgb,y.rgb,(c-m).rgb,(y-m).rgb,(y-c).rgb,(c-y).rgb]) # the minus mixes the colors! (+ makes them lighter. Not too sure what is it for.)

# gradient. To make a gradient color  -> white you can use HSV. let's convert 1=red 0=white (as a score)
starting_color=chroma.Color((1,0,0),format="RGB") # red then starting_color.hsv is the hsv equivalent.
score=0.5 # input in 0,1 (your score)
x=list(starting_color.hsv)
x[1]=score
new_color=chroma.Color(tuple(x), format='HSV')
"""


def Gencols6():
    i = 0
    yield iworkpalette[i]
    while True:
        i += 1
        yield iworkpalette[i]


def Gencols20():
    i = 0
    yield palette20[i]
    while True:
        i += 1
        yield palette20[i]


colors6 = Gencols6()
colors20 = Gencols20()


def mix_colors(c1, c2=None, add=False):
    """
    c1 can be a list of more than one color and c2 None or they can be two colors
    """
    try:
        import chroma
    except ImportError:
        sys.stderr.write(
            "\n**WARNING** chroma MODULE NOT AVAILABLE. Will not be able run s2D_profile_to_color\n  in terminal try running 'pip install chroma'\n"
        )

        class chroma:
            def Color(self, *args, **kwargs):
                raise Exception(
                    "chroma MODULE NOT AVAILABLE. Cannot run s2D_profile_to_color\n in terminal try running 'pip install chroma'\n"
                )

        pass
    color = None
    conv = matplotlib.colors.ColorConverter()
    if c2 is None:
        starting_cols = [chroma.Color(conv.to_rgba(c), format="RGB") for c in c1]
    else:
        starting_cols = [
            chroma.Color(conv.to_rgba(c1), format="RGB"),
            chroma.Color(conv.to_rgba(c2), format="RGB"),
        ]
    # print starting_cols
    for nc in starting_cols:
        if not hasattr(color, "rgb"):
            color = nc
        elif add:
            color += nc
        else:
            color -= nc
    return color.rgb


def RGBA_to_hex(RGBA_tuple):
    if len(RGBA_tuple) == 3:
        RGBA_tuple = list(RGBA_tuple) + [0]
    if all(numpy.array(RGBA_tuple) <= 1.0):  # assume is not given up to 255
        RGBA_tuple = (255 * numpy.array(RGBA_tuple)).astype("int")
    return "#{:02x}{:02x}{:02x}".format(*RGBA_tuple)


def gradient(
    color,
    Ncolors,
    alpha=1.0,
    vibrancy=0.6,
    spiral=False,
    spiral_revolutions=1.0,
    start_with_white=False,
    end_with_black=False,
    debug=False,
    show=False,
    return_hex=False,
):
    """
    create a gradient of N colors starting from color. By default color is in the middle
    the lower the vibrancy the brightest the colors...
     see help of generate_HSV_gradients
    """
    if type(color) is tuple and len(color) == 4:
        alpha = color[-1]
    conv = matplotlib.colors.ColorConverter()
    HSV1 = colorsys.rgb_to_hsv(*conv.to_rgb(color))
    RGBg = HSV_gradient(
        HSV1[0],
        HSV1[1],
        HSV1[2],
        Ncolors,
        spiral=spiral,
        convert_to_RGB=True,
        start_with_white=start_with_white,
        end_with_black=end_with_black,
        spiral_revolutions=spiral_revolutions,
        debug=debug,
        show=False,
        plot_points=show,
    )
    # RGBg=  generate_HSV_gradients(HSV1[0], N=N,vibrancy=HSV1[1], spiral=spiral, spiral_revolutions=spiral_revolutions, start_with_white=start_with_white, end_with_black=end_with_black, convert_to_RGB=True, show=show)
    cols = [t + (alpha,) for t in RGBg]
    if return_hex:
        cols = [RGBA_to_hex(c) for c in cols]
    return cols


def gradientRGB(
    color, N, to_color=None, factor=4.0, alpha=1.0, color_is_mid=True, to_dark=True
):
    """
    create a gradient of N colors starting from color. By default color is in the middle
     unless
     (i) color_is_mid is False, in which case it goes to darker (to_dark=True) or to lighter (to_dark=False) colors
     (ii) to_color is another color, in which case the colors are between color and to_color
    """
    if factor > 10.0:
        factor = 9.0
        print("in plotter.gradient factor>10 not allowed, settng to %lf" % (factor))
    if type(color) is tuple and len(color) == 4:
        alpha = color[-1]
    conv = matplotlib.colors.ColorConverter()
    RGB1 = conv.to_rgb(color)
    if to_color is not None:
        RGB2 = conv.to_rgb(to_color)
    if not color_is_mid:
        if to_dark:
            RGB2 = tuple([a / float(factor) for a in RGB1])
        else:
            RGB2 = tuple([min([1, a + factor / 10.0]) for a in RGB1])
    else:
        RGB1 = tuple([a / float(factor) for a in RGB1])
        RGB2 = tuple([min([1, a + factor / 10.0]) for a in conv.to_rgb(color)])
    print("RGB1=", RGB1, "RGB2=", RGB2)
    return generateColorGradient(RGB1, RGB2, N, alpha=alpha)


def generateColorGradient(RGB1, RGB2, n, alpha=1):
    dRGB = [float(x2 - x1) / (n - 1) for x1, x2 in zip(RGB1, RGB2)]
    gradient = [
        tuple([min([1, x + k * dx]) for x, dx in zip(RGB1, dRGB)] + [alpha])
        for k in range(int(n))
    ]
    return gradient


markers = [
    ".",
    "o",
    "s",
    "d",
    "v",
    "^",
    "<",
    ">",
    "8",
    "p",
    "h",
    "H",
    "D",
    "+",
    "x",
    "*",
    "1",
    "2",
    "3",
    "4",
    4,
    5,
    6,
    7,
    "_",
    "|",
]


def color_generator(NUM_COLORS=22, from_map="gist_rainbow"):
    cm = matplotlib.cm.get_cmap(from_map)
    for i in range(NUM_COLORS):
        color = cm(1.0 * i / NUM_COLORS)  # color will now be an RGBA tuple
        yield color


def plot_palette(palette, defaultfor_cmaps=255):
    if type(palette) is str:
        palette = matplotlib.cm.get_cmap(palette)
        L = defaultfor_cmaps
        xlabs = list(range(0, L))
    else:
        L = len(palette)
        xlabs = list(map(str, palette))
    val = [1] * L
    profile(
        val,
        color=palette,
        xlabels=xlabs,
        bar=True,
        markeredgecolor=True,
        xlabels_rotation="vertical",
        bar_sep=0,
        linewidth=0.0,
        x_major_tick_every=1,
        xlabel="Colors N=" + str(L),
    )
    print(palette)
    return


def HSV_gradient(
    Hue,
    Saturation,
    Value,
    Ncolors,
    spiral=False,
    convert_to_RGB=True,
    start_with_white=False,
    end_with_black=False,
    spiral_revolutions=1,
    show=False,
    debug=False,
    plot_points=False,
):
    """
    Value is almost like 1-Lightness withe is 0 Saturation and 1 Vaue
    spiral make changes also in the hue value and thus color change (rainbow like) hue is tha angle in the HSV cilinder
    start_with_white can be a float corresponding to starting Saturation and starting value will be 1-start_with_white
    end_with_black can be a float corresponding to starting Value
    """
    if Hue > 1.0:
        Hue = Hue / 360.0  # angle in degrees
    if type(start_with_white) is bool and start_with_white == True:
        start = (Hue, 0, 1)
    elif type(start_with_white) is float or type(start_with_white) is int:
        start = (
            Hue,
            min([Saturation, start_with_white]),
            max([Value, 1 - start_with_white]),
        )
    else:
        start = (Hue, min([Saturation, 0.2]), max([Value, 0.95]))
    if type(end_with_black) is bool and end_with_black == True:
        end = (Hue, 1, 0)
    elif type(end_with_black) is float or type(end_with_black) is int:
        end = (
            Hue,
            1,
            end_with_black,
        )  # real black has 0 saturation in truth but both convert to rgb black
    else:
        end = (Hue, 1, min([Value, 0.2]))
    d1 = (
        numpy.linalg.norm(numpy.array(start[1:]) - numpy.array([Saturation, Value]))
        - 0.12
    )  # the 0.1 correction is because after we include the input color within d2 and not d1
    d2 = (
        numpy.linalg.norm(numpy.array(end[1:]) - numpy.array([Saturation, Value]))
        + 0.12
    )
    N1, N2 = int(numpy.round(Ncolors * d1 / (d1 + d2), 0)), int(
        numpy.round(Ncolors * d2 / (d1 + d2), 0)
    )
    if N1 + N2 == Ncolors - 1:  # happens when they are exactly the same
        N2 += 1
    if N1 == 1 and N2 > 2:
        N1 += 1
        N2 -= 1
    elif N2 == 1 and N1 > 2:
        N2 += 1
        N1 -= 1
    lin_sat = list(numpy.linspace(start[1], Saturation, N1, endpoint=False)) + list(
        numpy.linspace(Saturation, end[1], N2, endpoint=True)
    )
    lin_val = list(numpy.linspace(start[2], Value, N1, endpoint=False)) + list(
        numpy.linspace(Value, end[2], N2, endpoint=True)
    )
    if lin_val.count(0) > 1:  # multiple blacks
        if Value == 0:
            lin_val = list(numpy.linspace(start[2], Value, Ncolors, endpoint=True))
    if debug:
        print("Hue, Saturation, Value=", Hue, Saturation, Value)
        print("start=", start)
        print("end=", end)
        print("d1,d2=", d1, ",", d2, ";N1, N2=", N1, ",", N2)
        print("lin_val=", lin_val)
        print("lin_sat=", lin_sat)
        print("input_index=", lin_sat.index(Saturation), lin_val.index(Value))
    if spiral:
        lin_hue = numpy.array(
            (
                [
                    Hue - (j + 1) * float(spiral_revolutions) / Ncolors
                    for j in range(N1)
                ][::-1]
            )
            + ([Hue + j * float(spiral_revolutions) / Ncolors for j in range(N2)])
        )
        while not all(lin_hue <= 1):
            lin_hue[lin_hue > 1] -= 1
        while not all(lin_hue >= 0):
            lin_hue[lin_hue < 0] += 1
    else:
        lin_hue = [Hue] * Ncolors
    HSV_tuples = list(zip(lin_hue, lin_sat, lin_val))
    if show:
        RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
        val = [1] * len(RGB_tuples)
        bar(
            val,
            color=RGB_tuples,
            xlabels=list(map(str, RGB_tuples)),
            xlabels_rotation="vertical",
            bar_sep=0,
            bar_width=1,
            linewidth=0.0,
            xlabel="Colors from generate_HSV_gradients() N=" + str(len(RGB_tuples)),
            block=False,
            show=True,
        )
    if plot_points:
        RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
        fig = scatter(
            lin_sat,
            lin_val,
            xlabel="Saturation",
            ylabel="Value",
            markerfacecolor=RGB_tuples,
            marker="s",
            x_range=(-0.02, 1.05),
            y_range=(-0.02, 1.05),
            markersize=100,
        )
        fig = point(
            (Saturation, Value),
            fig,
            marker="o",
            markersize=350,
            markerfacecolor="none",
            markeredgewidth=3,
            markeredgecolor=colorsys.hsv_to_rgb(Hue, Saturation, Value),
        )
    if convert_to_RGB:
        return [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    return HSV_tuples


def generate_HSV_gradients(
    h_start_color,
    N=5,
    vibrancy=1,
    spiral=False,
    spiral_revolutions=1.0,
    start_with_white=False,
    end_with_black=False,
    convert_to_RGB=True,
    show=False,
):
    """
    SUPERSEDED by HSV_gradient
    the lower the vibrancy the brightest the colors...
        play by giving close to 1 values to start_from_white ( <1 )
        and close to 0 to end_with_black ( >0 )
    if spiral it generates raimbow like staff rather than gradients
    40./360 is a good yellow to start with
    """
    if type(h_start_color) is str:
        if h_start_color == "red":
            h_start_color = 0.0
        elif h_start_color == "yellow":
            h_start_color = 60.0 / 360.0
        elif h_start_color == "green":
            h_start_color = 120.0 / 360.0
        elif h_start_color == "light blue" or h_start_color == "cyan":
            h_start_color = 180.0 / 360.0
        elif h_start_color == "blue":
            h_start_color = 240.0 / 360.0
        elif h_start_color == "purple" or h_start_color == "magenta":
            h_start_color = 300.0 / 360.0
        else:
            sys.stderr.write(
                "ERROR in generate_HSV_gradients() %s not recognized!\n"
                % (h_start_color)
            )
            return
    elif h_start_color > 1.0:
        h_start_color = h_start_color / 360.0  # angle in degrees
    if start_with_white is False:
        start_with_white = 0.99
    elif type(start_with_white) is not float and type(start_with_white) is not int:
        start_with_white = 1.0
    if end_with_black is False:
        end_with_black = 0.3
    elif type(end_with_black) is not float and type(end_with_black) is not int:
        end_with_black = 0.0
    tparms = numpy.linspace(end_with_black, start_with_white, N)
    tparms = tparms[::-1]
    print("tparms=", tparms)
    if vibrancy == 0:
        vibrancy = 0.05
        spiral = True
    if spiral:
        sp = numpy.linspace(
            0.0, spiral_revolutions * numpy.arctan(0.0535898384 * N), N
        )  # arctan tailored so that it does a max of 3 revolutions and for N about 5 it does half revolution..
        HSV_tuples = [
            (h_start_color + sp[j], t**vibrancy, t ** (1.0 / vibrancy))
            for j, t in enumerate(tparms)
        ]
    else:
        HSV_tuples = [
            (h_start_color, t**vibrancy, t ** (1.0 / vibrancy)) for t in tparms
        ]
    if convert_to_RGB:
        RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
        if show:
            val = [1] * len(RGB_tuples)
            bar(
                val,
                color=RGB_tuples,
                xlabels=list(map(str, RGB_tuples)),
                xlabels_rotation="vertical",
                bar_sep=0,
                bar_width=1,
                linewidth=0.0,
                xlabel="Colors from generate_HSV_gradients() N=" + str(len(RGB_tuples)),
                block=False,
                show=True,
            )
        return RGB_tuples
    return HSV_tuples


def generate_grey_scale(
    values,
    flatten_percentiles=None,
    black_means_higher=True,
    lighter_gray="0.8",
    darker_grey="0",
    return_also_colorbar_info=True,
):
    entries = numpy.array(values[:])  # copy the values
    percs = None
    if flatten_percentiles is not None:
        percs = numpy.percentile(values, flatten_percentiles)
        if hasattr(percs, "__len__"):  # we have also an inf percentile
            entries[entries < percs[0]] = percs[0]
            entries[entries > percs[1]] = percs[1]
        else:
            entries[entries > percs] = percs
    if black_means_higher:
        p = -1.0
    else:
        p = 1.0
    if return_also_colorbar_info:
        tok, mymap = linear_color_map(
            min(entries), max(entries), lighter_gray, darker_grey
        )

    entries = ScaleInRange(
        p * entries, NewMin=float(darker_grey), NewMax=float(lighter_gray)
    )
    if return_also_colorbar_info:
        return list(map(str, entries)), tok, mymap
    return list(map(str, entries)), percs


def linear_color_map(minval, maxval, mincol, maxcol, step=100, close_tocken=True):
    # then do plt.colorbar(CS3) and then show. much better to avoid this and use add_color_map
    mymap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "mycolors", [mincol, maxcol]
    )

    mymap.set_under(mincol)
    mymap.set_over(maxcol)
    f2 = plt.figure()
    # Using contourf to provide my colorbar info, then clearing the figure
    Z = [[0, 0], [0, 0]]
    levels = numpy.linspace(minval, maxval, step)
    # print levels
    CS3 = plt.contourf(Z, levels, cmap=mymap, figure=f2)
    f2.clf()
    if close_tocken:
        plt.close(f2)
    return CS3, mymap


def create_color_map(
    minval,
    maxval,
    colors=["0.8", "0"],
    masked_vals_color=None,
    return_sm=False,
    set_under=None,
    set_over=None,
):
    mymap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "mycolors", colors, N=256
    )
    if set_under is None:
        set_under = colors[0]
    if set_over is None:
        set_over = colors[-1]
    mymap.set_under(set_under)
    mymap.set_over(set_over)
    if masked_vals_color is not None:
        mymap.set_bad(masked_vals_color)
    if return_sm:
        sm = plt.cm.ScalarMappable(
            cmap=mymap, norm=plt.Normalize(vmin=minval, vmax=maxval)
        )
        sm._A = []
        return mymap, sm
    return mymap


def plot_color_map_simple(colormap, npoints=256):
    if type(colormap) is str:
        colormap = matplotlib.cm.get_cmap(colormap)
    f = plt.figure(figsize=(2, 8))
    ax = f.gca()
    ax.scatter(
        [0] * npoints,
        numpy.linspace(0, 1, npoints),
        color=colormap(numpy.linspace(0, 1, npoints)),
    )
    ax.set_xticklabels([])
    plt.show(block=False)


def plot_color_map(mymap, ncolors=255, upwith=400, **kwargs):
    if type(mymap) is str:
        mymap = matplotlib.cm.get_cmap(mymap)
    yss = list(range(1, ncolors + 1))
    profiles = numpy.vstack([yss] * 10).T
    xss = list(range(0, 10))
    profile(
        profiles,
        xss,
        label=yss,
        color=mymap,
        y_range=(0.9, ncolors + 0.1),
        linewidth=float(upwith) / ncolors,
    )


def custom_color_map(
    minval,
    maxval,
    mincol,
    maxcol,
    figure=None,
    maxbeyond=None,
    minbeyond=None,
    nticks=10,
    label=None,
):
    # this kind of works...
    mymap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "mycolors", [mincol, maxcol], N=256
    )

    mymap.set_under(mincol)
    mymap.set_over(maxcol)
    sm = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(vmin=minval, vmax=maxval))
    sm._A = []
    if figure is None:
        figure = plt.figure()
    if maxbeyond is not None or minbeyond is not None:
        size = maxval - minval
        fmax, fmin = 0, 0
        if type(nticks) is list:
            ticks = nticks
        else:
            ticks = list(
                numpy.arange(
                    Round_To_n(minval, 0),
                    maxval,
                    Round_To_n((maxval - minval) / float(nticks), 0),
                )
            )
        draw_me = []
        if maxbeyond is not None:
            fmax = (maxbeyond - maxval) / size
            draw_me += [(maxbeyond, 1.0 + fmax)]
        if minbeyond is not None:
            fmin = (minval - minbeyond) / size
            draw_me += [(minbeyond, 0 - fmin)]
        print(ticks, fmin, fmax)
        cbar = figure.colorbar(
            sm,
            extend="both",
            extendfrac=(fmin, fmax),
            extendrect=True,
            spacing="proportional",
            drawedges=False,
        )  # ,ticks=ticks)
        cbar.set_ticks(ticks)
        if label is not None:
            cbar.set_label(label, rotation=270, fontsize=text_sizes["xlabel"])
        for t in draw_me:
            cbar.ax.text(
                1.5, t[1], t[0], ha="left", va="center", fontsize=text_sizes["xlabels"]
            )
    else:
        if type(nticks) is list:
            ticks = nticks
        else:
            ticks = None
        cbar = figure.colorbar(sm, spacing="proportional", drawedges=False, ticks=ticks)
    return cbar, mymap


def colors_from_color_map(
    data, cmap_name="Spectral", color_bad="none", log_scale=False
):
    """
        import matplotlib

    cmap = matplotlib.cm.get_cmap('Spectral')
    rgba = cmap(0.5)
    print(rgba) # (0.99807766255210428, 0.99923106502084169, 0.74602077638401709, 1.0)
    For values outside of the range [0.0, 1.0] it will return the under and over colour (respectively). This, by default, is the minimum and maximum colour within the range (so 0.0 and 1.0). This default can be changed with cmap.set_under() and cmap_set_over().
    For "special" numbers such as numpy.nan and numpy.inf the default is to use the 0.0 value, this can be changed using cmap.set_bad() similarly to under and over as above.
    Finally it may be necessary for you to normalize your data such that it conforms to the range [0.0, 1.0]. This can be done using matplotlib.colors.Normalize simply as shown in the small example below where the arguments vmin and vmax describe what numbers should be mapped to 0.0 and 1.0 respectively.

    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=10.0, vmax=20.0)
    print(norm(15.0)) # 0.5
    A logarithmic normaliser (matplotlib.colors.LogNorm) is also available for data ranges with a large range of values.
    """
    cmap = matplotlib.cm.get_cmap(cmap_name)
    if color_bad is not None:
        cmap.set_bad(color_bad)
    if hasattr(data[0], "__len__"):
        Min, Max = numpy.nanmin([numpy.nanmin(p) for p in data]), numpy.nanmax(
            [numpy.nanmax(p) for p in data]
        )
    else:
        Min, Max = numpy.nanmin(data), numpy.nanmax(data)
    if log_scale:
        norm = matplotlib.colors.LogNorm(vmin=Min, vmax=Max)
    else:
        norm = matplotlib.colors.Normalize(vmin=Min, vmax=Max)
    return [cmap(norm(x)) for x in data]


def colorline(
    x,
    y,
    z=None,
    cmap=plt.get_cmap("copper"),
    ax=None,
    norm=plt.Normalize(0.0, 1.0),
    linewidth=2,
    alpha=1.0,
    interpolate=None,
    zorder=4,
):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    z can also be a function in the form z(x,y), in this case for any (x,y) it shoul return values in 0,1
    """
    # Default colors equally spaced on [0,1]:
    segments, x, y = make_segments(x, y, interpolate=interpolate)
    if z is None:
        z = ScaleInRange(y, 0, 1)  # numpy.linspace(0.0, 1.0, len(x))
    elif hasattr(z, "__call__"):
        if len(inspect.getargspec(z).args) == 1:
            z = z(y)
        else:
            z = z(x, y)
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = numpy.array([z])

    z = numpy.asarray(z)

    lc = matplotlib.collections.LineCollection(
        segments,
        array=z,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y, interpolate=None):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    it return segments,x,y if interpolate the x and y are the interpolated ones
    """
    if type(interpolate) is int:
        xx = numpy.linspace(min(x), max(x), interpolate)
        yy = numpy.interp(xx, *list(zip(*sorted(zip(x, y)))))
        points = numpy.array([xx, yy]).T.reshape(-1, 1, 2)
        segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
        return segments, xx, yy
    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
    return segments, x, y


def align_seqs_profiles(
    sequences,
    profiles,
    annotation_string=None,
    align_with_modeller=False,
    gap_profile_value=numpy.nan,
    gap_symbol="-",
):
    """
    if align_with_modeller is False it will use tcoffee
    it takes some sequences and some profiles (must be one per sequence but same sequence can be given many times and should make no difference
    return al_profile,aligned_records,printed_sequence,annotation_string
      if the input annotation_string is None and only 2 unique sequences are given as input
        then the pair printed_sequences, annotation_string correponds to the 2 input sequences with special characters
        and capitalisation to highlight the regions that are identical and those that are not
       otherwise printed_sequence is a sort of fake consensus sequence from the alignment again with special characters
        and capitalisation to highlight conservation
      aligned_records is all sequecnes after the alignment (thus they may contain gaps)
    """
    uniq_seq, uniq_ind = [], []
    lengths = []
    for p in profiles:
        if len(p) not in lengths:
            lengths += [len(p)]
    # maxlength=max(lengths)
    for ij, s in enumerate(sequences):
        if s not in uniq_seq:
            uniq_seq += [s]
            uniq_ind += [ij]

    aligned_records = mybio.tcoffee_alignment(sequences, quiet=True)
    al_profile = []
    seq_mat = []
    for j, seq in enumerate(aligned_records):
        al_profile += [
            seq_profile_to_alignment(
                profiles[j],
                seq,
                gap_profile_value=gap_profile_value,
                gap_symbol=gap_symbol,
            )
        ]
        seq_mat += [seq]

    seq_mat = numpy.array(seq_mat)
    if len(uniq_seq) == 2 and annotation_string is None:
        annotation_string = ""
        printed_sequence = ""
        top, bottom = uniq_ind
        for j in range(len(al_profile[-1])):
            if "-" in seq_mat[:, j]:
                annotation_string += seq_mat[:, j][bottom]  # one of these will be -
                printed_sequence += seq_mat[:, j][top]
            elif (seq_mat[:, j] == seq_mat[:, j][0]).all():  # the two are identical
                annotation_string += seq_mat[:, j][bottom]
                printed_sequence += seq_mat[:, j][top]
            else:
                annotation_string += seq_mat[:, j][
                    bottom
                ].lower()  # we represent different aa in lower case
                printed_sequence += seq_mat[:, j][top].lower()
    else:
        fake_consensus = ""
        for j in range(len(al_profile[-1])):
            mat_values, mat_counts = numpy.unique(
                seq_mat[:, j], return_counts=True
            )  # return_counts only in numpy versions >= 1.9
            most_common_ind = numpy.argmax(mat_counts)
            most_common, numb_occurrences = (
                mat_values[most_common_ind],
                mat_counts[most_common_ind],
            )  # prints the most frequent element
            if numb_occurrences == len(aligned_records):
                fake_consensus += most_common  # they are all identical
            elif numb_occurrences >= len(aligned_records) / 2.0:
                fake_consensus += (
                    most_common.lower()
                )  # this is more than 50% conserved, hence lower case
            elif "-" in seq_mat[:, j]:
                fake_consensus += (
                    ","  # non-conserved site where at least one sequence has a gap
                )
            else:
                fake_consensus += "*"  # non-conserved site where no sequence has a gap
        printed_sequence = fake_consensus

    # sys.stderr.write("** in plot_seq_profile() given profiles of different lengths (found %s). Done alignemnt of sequences and assigning Nan to gaps!!\n\n" % (str(lengths)) )
    return al_profile, aligned_records, printed_sequence, annotation_string


def seq_profile_to_alignment(
    profile, seq_from_alignment, gap_profile_value=numpy.nan, gap_symbol="-"
):
    """
    given a sequence profile (list of floats) and a sequence as read from an alignemt file
     it adds in the profile list entries equal to gap_profile_value
     at positions corresponding to the gaps in the sequences '-'
    return al_profile
    """
    if len(profile) == 2 and hasattr(
        profile[0], "__len__"
    ):  # most likely a CI from errors
        return [
            seq_profile_to_alignment(
                pr,
                seq_from_alignment,
                gap_profile_value=gap_profile_value,
                gap_symbol=gap_symbol,
            )
            for pr in profile
        ]
    al_profile = []
    i = 0
    for res in seq_from_alignment:
        if res != gap_symbol:
            al_profile += [float(profile[i])]
            i += 1
        else:
            al_profile += [gap_profile_value]
    if len(profile) != i:
        sys.stderr.write(
            "WARNING in seq_profile_to_alignment() len(profile)!= number of non-gap residues in seq_from_alignment (%d!=%d)\n"
            % (len(profile), i)
        )
    return al_profile


# it plot a profile corresponfing to a sequence. It contains some nice trick that vary with the length of the sequence in order to put the labels. Unless print_all_sequence=True
def plot_seq_profile(
    sequence,
    profile,
    annotation_string=None,
    do_matrix=False,
    cbar_label=None,
    bar=False,
    bar_sep=0.2,
    log_scale=False,
    avoid_scientific_notation=True,
    max_res_per_plot=None,
    stacked=False,
    start_rescount=1,
    label="",
    xlabel=None,
    ylabel="Score",
    ylabels=None,
    title=None,
    value_labels=None,
    value_labels_rotation=None,
    zygg_like_lines=True,
    return_alignment=False,
    align_with_modeller=False,
    hline=0,
    vline=None,
    hgrid=False,
    vgrid=None,
    frame=None,
    zorder=0,
    print_all_sequence=True,
    color=None,
    add_colorline=False,
    plot_legend=True,
    ncol=1,
    do_matrix_cmap="ocean_r",
    center_cmap_on_value=None,
    plot_colorbar=True,
    show=True,
    block=False,
    sequence_fontsize=None,
    annotate_antibody=False,
    plot_antibody_legend=None,
    fill_error=False,
    sequence_extra_numbering=None,
    sequence_extra_numbering_name=None,
    fill_error_alpha=0.2,
    yerr=None,
    y_range=None,
    y_major_tick_every=None,
    y_minor_tick_every=None,
    x_major_tick_every=None,
    x_minor_tick_every=None,
    xlabels=None,
    xlabels_rotation="horizontal",
    ls="-",
    linewidth=None,
    marker="",
    markerfacecolor=True,
    markeredgecolor=True,
    markersize=18,
    upper_label_rotation="horizontal",
    annotation_string_position=None,
    annotation_string_rotation="horizontal",
    legend_location="upper right",
    legend_size=None,
    vmin=None,
    vmax=None,
    figure_size=None,
    figure_and_axts_tuple=None,
    dpi=None,
    save="",
):
    """
    figure_and_axts_tuple can be given to superimpose
    return fig,axt
    unless return_alignment and multiple sequences are given (and the alignemnt is actually perform) in which case it
     return fig, axt, aln, fake_consensus, al_profiles
    add_colorline can be set to 'camsol' to use camsol default, otherwise to True or to (zf,colormap) (zf can be None or a custom normalisation)
    value_labels now only for matrix
    annotate_antibody highlights CDR regions by adding gray boxes shaded in the background
    """
    if frame is None:
        if sequence is None:
            frame = default_parameters["frame"]
        else:
            frame = True
    if hgrid is None:
        hgrid = default_parameters["hgrid"]
    if vgrid is None:
        vgrid = default_parameters["vgrid"]
    if figure_size is None:
        if default_figure_sizes["default"] is None:
            figure_size = default_figure_sizes["sequence"]
        else:
            figure_size = default_figure_sizes["default"]
        if type(sequence) is str and len(sequence) > 600:
            figure_size = (figure_size[0], min([20, int(len(sequence) / 100)]))
    if default_figure_sizes["use_cm"]:
        figure_size = (cmToinch(figure_size[0]), cmToinch(figure_size[1]))
    if sequence_fontsize is None:
        sequence_fontsize = text_sizes["xlabels_many"]
    if max_res_per_plot is None:
        max_res_per_plot = default_parameters["seq_max_res_per_plot"]
        if max_res_per_plot is None:  # may have been default parameter
            if type(sequence_fontsize) is not str:
                max_res_per_plot = int(
                    (120 / float(sequence_fontsize)) * figure_size[0]
                )
    if legend_size is None:
        legend_size = text_sizes["legend_size"]
    plt.rc("xtick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick", labelsize=text_sizes["xlabels"])
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    to_return = None
    cdr_profile = None
    antibody_scheme = "AHo"  # default, also Chothia or imgt or others
    antibody_numbers = None
    if type(annotate_antibody) is bool and annotate_antibody == True:
        annotate_antibody = 0.333
    elif type(annotate_antibody) is str:
        antibody_scheme = annotate_antibody
        annotate_antibody = 0.333
    if annotate_antibody and (
        type(sequence) is str or "seq" in dir(sequence)
    ):  # only one sequence, ismulti=False but will become True to plot also annotated CDRs
        try:
            cdr_profile, antibody_numbers = mybio.get_CDR(
                sequence, return_cdr_profile=True, scheme=antibody_scheme
            )[-2:]
            if xlabels is None:
                xlabels = antibody_numbers
                xlabels_size = text_sizes["xlabels_many"]
                if x_major_tick_every is None:
                    x_major_tick_every = 1
                if x_minor_tick_every is None:
                    x_minor_tick_every = False
                if xlabels_rotation == "horizontal":
                    xlabels_rotation = "vertical"
                if (
                    xlabel is None
                    or xlabel == "Residue"
                    or xlabel.lower() == "residue position"
                ):
                    xlabel = antibody_scheme + " residue number"
            cdr_profile = annotate_antibody * numpy.array(cdr_profile)
            if len(cdr_profile) != len(sequence):
                sys.stderr.write(
                    "*ERROR in annotate_antibody plot_seq_profile() len(cdr_profile) != len(sequence) %d %d\n"
                    % (len(cdr_profile), len(sequence))
                )
            # if not do_matrix :
            #    if hasattr(profile[0],'__len__') :
            #        if type(profile) is not list : profile=list(profile)
            #        profile+=[ cdr_profile ]
            #    else :
            #        profile=[profile]
            if return_alignment:
                to_return = (antibody_scheme, sequence, cdr_profile, antibody_numbers)
        except Exception:
            sys.stderr.write(
                "*potential error in annotate_antibody plot_seq_profile() - IGNORE if input sequence was not an antibody sequence containing VH and/or VL domain\n"
            )
            print_exc(file=sys.stderr)
            # sys.stderr.write( "")
            sys.stderr.flush()
            pass

    if hasattr(profile[0], "__len__"):
        ismulti = True
        lengths = []
        for p in profile:
            if len(p) not in lengths:
                lengths += [len(p)]
        maxlength = max(lengths)
        # if len(lengths)>1 :
        # algin multiple sequences if needed
        if (
            sequence is not None
            and (type(sequence) is list or type(sequence) is tuple)
            and not do_matrix
        ):  # more than one sequence given, align
            try:
                if annotate_antibody > 0:
                    # print 'DEB: annotate_antibody=True multiple sequences'
                    for seq in sequence:
                        try:
                            # only check that they are all antibodies, otherwise use tcoffee
                            _ = mybio.get_CDR(
                                seq, return_cdr_profile=True, scheme=antibody_scheme
                            )[-2:]
                        except Exception:
                            print(
                                "\nERROR annotate_antibody=True multiple sequences - at least one not antibody or lead to failure - will align with tcoffee"
                            )
                            print_exc(file=sys.stderr)
                            annotate_antibody = False
                            pass

                uniq_seq, uniq_ind = (
                    [],
                    [],
                )  # will still align all sequences, in case only 2 uniqs it will display the alignment
                for ij, s in enumerate(sequence):
                    if s not in uniq_seq:
                        uniq_seq += [s]
                        uniq_ind += [ij]

                if (
                    annotate_antibody > 0
                ):  # align antibodies here, we already verified these are all antibodies
                    aln = mybio.Anarci_alignment(scheme=antibody_scheme)
                    aln.add_from_records(sequence)
                    aln.align_constant_region()  # include constant regions
                    align = aln.to_recs()
                    antibody_numbers = aln.HD
                    cdr_profile = [1 if "CDR" in seid else 0 for seid in aln.seq_region]
                    cdr_profile = annotate_antibody * numpy.array(cdr_profile)
                    if xlabels is None:
                        xlabels = antibody_numbers
                        xlabels_size = text_sizes["xlabels_many"]
                        if x_major_tick_every is None:
                            x_major_tick_every = 1
                        if x_minor_tick_every is None:
                            x_minor_tick_every = False
                        if xlabels_rotation == "horizontal":
                            xlabels_rotation = "vertical"
                        if (
                            xlabel == "Residue"
                            or xlabel.lower() == "residue positon"
                            or xlabel is None
                        ):
                            xlabel = antibody_scheme + " residue number"
                    # print 'DEB: cdr_profile=',cdr_profile
                elif len(sequence) == 2:
                    aln_pair = mybio.pairwise_alignment(
                        sequence[0], sequence[1], one_alignment_only=True
                    )
                    align = [aln_pair[0][0], aln_pair[0][1]]
                else:
                    align = mybio.tcoffee_alignment(sequence, quiet=True)
                al_profile = []
                if yerr is not None:
                    al_yerr = []
                seq_mat = []
                # print 'alignlen:',len(align),align
                for j, seq in enumerate(align):
                    seq_mat += [list(seq)]
                    al_profile += [seq_profile_to_alignment(profile[j], seq)]
                    if yerr is not None:
                        if j < len(yerr) and yerr[j] is not None:
                            al_yerr += [seq_profile_to_alignment(yerr[j], seq)]
                        else:
                            al_yerr += [None]
                profile = al_profile
                if yerr is not None:
                    yerr = al_yerr
                seq_mat = numpy.array(seq_mat)
                # print 'seq_mat.shape',seq_mat.shape,len(al_profile),len(al_profile[-1]),seq_mat
                if len(uniq_seq) == 2 and annotation_string is None:
                    annotation_string = ""
                    sequence = ""
                    top, bottom = uniq_ind
                    for j in range(len(al_profile[-1])):
                        if "-" in seq_mat[:, j]:
                            annotation_string += seq_mat[:, j][
                                bottom
                            ]  # one of these will be -
                            sequence += seq_mat[:, j][top]
                        elif (
                            seq_mat[:, j] == seq_mat[:, j][0]
                        ).all():  # the two are identical
                            annotation_string += seq_mat[:, j][bottom]
                            sequence += seq_mat[:, j][top]
                        else:
                            annotation_string += seq_mat[:, j][
                                bottom
                            ].lower()  # we represent different aa in lower case
                            sequence += seq_mat[:, j][top].lower()
                else:
                    fake_consensus = ""
                    for j in range(len(al_profile[-1])):
                        mat_values, mat_counts = numpy.unique(
                            seq_mat[:, j], return_counts=True
                        )  # return_counts only in numpy versions >= 1.9
                        most_common_ind = numpy.argmax(mat_counts)
                        most_common, numb_occurrences = (
                            mat_values[most_common_ind],
                            mat_counts[most_common_ind],
                        )  # prints the most frequent element
                        if numb_occurrences == len(align):
                            fake_consensus += most_common  # they are all identical
                        elif numb_occurrences >= len(align) / 2.0:
                            fake_consensus += (
                                most_common.lower()
                            )  # this is more than 50% conserved, hence lower case
                        elif "-" in seq_mat[:, j]:
                            fake_consensus += ","  # non-conserved site where at least one sequence has a gap
                        else:
                            fake_consensus += (
                                "*"  # non-conserved site where no sequence has a gap
                            )
                    sequence = fake_consensus

                if len(lengths) > 1:
                    sys.stderr.write(
                        "** in plot_seq_profile() given profiles of different lengths (found %s). Done alignemnt of sequences and assigning Nan to gaps!!\n\n"
                        % (str(lengths))
                    )
                if return_alignment:
                    to_return = (align, sequence, al_profile, antibody_numbers)
            except Exception:
                raise
                sys.stderr.write(
                    "**WARNING in plot_seq_profile() given profiles of different lengths (found %s - failde to aling sequences). Appending zeros at end of shorter profiles!\n\n"
                    % (str(lengths))
                )
                for j, p in enumerate(profile):
                    if len(p) < maxlength:
                        profile[j] = list(p) + [0.0] * (maxlength - len(p))
                        if yerr is not None and yerr[j] is not None:
                            yerr[j] = list(yerr[j]) + [0.0] * (maxlength - len(yerr[j]))
                pass
        elif len(lengths) > 1:
            sys.stderr.write(
                "**WARNING in plot_seq_profile() given profiles of different lengths (found %s). Appending zeros at end of shorter profiles!\n\n"
                % (str(lengths))
            )
            for j, p in enumerate(profile):
                if len(p) < maxlength:
                    profile[j] = list(p) + [0.0] * (maxlength - len(p))

        if y_range is None:
            Min, Max = numpy.nanmin([numpy.nanmin(p) for p in profile]), numpy.nanmax(
                [numpy.nanmax(p) for p in profile]
            )
        if not do_matrix and value_labels is None:
            value_labels = len(profile) * [None]  # not implemented after
        if not hasattr(bar, "__len__"):
            bar = len(profile) * [bar]
        if not hasattr(zorder, "__len__"):
            zorder = len(profile) * [zorder]
        if len(bar) != len(profile):
            sys.stderr.write(
                "WARNING len(bar)!=len(profile) (number of profiles given) %d %d. Setting all bars to False\n"
                % (len(bar), len(profile))
            )
            bar = [False] * len(profile)
        n_bar_profiles = sum(bar)
        if (
            label != ""
            and label is not None
            and type(label) is not list
            and type(label) is not tuple
        ):
            label = len(profile) * [label]
        if not hasattr(add_colorline, "__len__") or type(add_colorline) is str:
            add_colorline = [add_colorline] * len(profile)
        if type(ls) is str or ls is None:
            ls = [ls] * len(profile)
        if type(linewidth) is int or type(linewidth) is float:
            lw = [linewidth] * len(profile)
        elif linewidth is None:
            lw = [
                default_parameters["barlinewidth"]
                if b
                else default_parameters["linewidth"]
                for b in bar
            ]
        else:
            lw = linewidth
        if type(marker) is str or not hasattr(marker, "__len__"):
            marker = [marker] * len(profile)
        # print 'markers:',len(profile),len(marker),len(lw),len(ls),len(color),len(yerr)
        profile = numpy.array(profile)
        if yerr is not None:
            yerr = numpy.array(yerr)
        # print 'ismulti',profile.shape,yerr.shape
        prof = profile[0]
    else:
        ismulti = False
        if hasattr(bar, "__len__"):
            bar = bar[0]
        if linewidth is None:
            if bar:
                lw = default_parameters["barlinewidth"]
            else:
                lw = default_parameters["linewidth"]
        else:
            lw = linewidth
        prof = profile
        if y_range is None:
            Min = numpy.nanmin(profile)
            Max = numpy.nanmax(profile)
    if y_range is None:
        ymin = int(Min - 1.0)
        ymax = int(Max + 1.0)
    else:
        ymin, ymax = y_range
    if sequence is not None and len(sequence) != len(prof):
        sys.stderr.write(
            "**WARNING** in plot_seq_profile() len(sequence)!=len(profile) %d!=%d\n"
            % (len(sequence), len(prof))
        )
    # if ecolor is None : ecolor=color
    if type(markeredgecolor) is bool and markeredgecolor == True:
        markeredgecolor = color
    if type(markerfacecolor) is bool and markerfacecolor == True:
        markerfacecolor = color
    if type(color) is list or isinstance(color, cycle_list):
        if type(markerfacecolor) is not list and not isinstance(
            markerfacecolor, cycle_list
        ):
            markerfacecolor = cycle_list([markerfacecolor] * len(color))
        if type(markeredgecolor) is not list and not isinstance(
            markeredgecolor, cycle_list
        ):
            markeredgecolor = cycle_list([markeredgecolor] * len(color))
        # if type(ecolor) is not list and not isinstance(ecolor,cycle_list) : ecolor=cycle_list([ecolor]*len(color))

    if figure_and_axts_tuple is not None:
        fig, axt = figure_and_axts_tuple
        n_profs = len(axt)
        do_tight = False
    else:
        if len(prof) % int(max_res_per_plot) > 0:
            add = 1
        else:
            add = 0.001  # in case of rounding errors from the float conversion
        n_profs = max(
            [1, int(len(prof) / float(max_res_per_plot) + add)]
        )  # up to max_res_per_plot residues per plot
        fig, axt = plt.subplots(
            n_profs, sharey=True, figsize=figure_size
        )  # axt is a tuple of n_profs size Use , sharex=True to share x axis
        if n_profs == 1:
            axt = (axt,)  # make iterable
        do_tight = True

    # auto set xlabels size for many labels
    xlabels_size = text_sizes["xlabels"]
    if (xlabels is not None and len(xlabels) > 15) or (
        sequence_extra_numbering is not None
    ):
        xlabels_size = text_sizes["xlabels_many"]
        if xlabels is not None and len(xlabels) / float(n_profs) > 90:
            xlabels_size = text_sizes["xlabels_many"] - 2
        elif (
            sequence_extra_numbering is not None
            and len(sequence_extra_numbering) / float(n_profs) > 80
        ):
            xlabels_size = text_sizes["xlabels_many"] - 2
        plt.rc("xtick", labelsize=xlabels_size)
    # print("DEB: xlabels_size=", xlabels_size, sequence_extra_numbering is not None,(xlabels is not None and len(xlabels) > 15), 'max_res_per_plot', max_res_per_plot)

    # determine the number of residues per subplot
    res_per_plot = (
        len(prof) // n_profs
    )  # floor division are like integer divisions as long as positive int
    rem = len(prof) % n_profs
    pzise = [res_per_plot] * n_profs
    j = 0
    while rem > 0:
        pzise[j] += 1
        rem -= 1
        j += 1
    start = 0

    line_styles = []
    for j, nres in enumerate(pzise):  # nres is the number of residues in subplot j
        xlim_m = start + start_rescount - 0.5
        xlim_M = start + nres + start_rescount - 0.5

        if (
            default_parameters["set_publish"]
            and figure_and_axts_tuple is None
            and not log_scale
        ):
            if y_major_tick_every is None:
                axt[j].locator_params(axis="y", tight=None, nbins=5)
            if x_major_tick_every is None:
                axt[j].locator_params(axis="x", tight=None, nbins=5)
            if x_minor_tick_every is None:
                axt[j].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(n=2))
            if y_minor_tick_every is None:
                axt[j].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(n=2))

        if ismulti:
            to_plot = profile[:, start : start + nres]
            if yerr is not None:
                pl_yerr = []
                for ye in yerr:
                    if ye is None:
                        pl_yerr += [None]
                    elif len(ye) == 2 and hasattr(
                        ye[0], "__len__"
                    ):  # given as asymmetric error bars
                        pl_yerr += [
                            (ye[0][start : start + nres], ye[1][start : start + nres])
                        ]
                    else:
                        pl_yerr += [ye[start : start + nres]]
            else:
                pl_yerr = [None] * len(to_plot)
        else:
            to_plot = profile[start : start + nres]
            if yerr is not None:
                if len(yerr) == 2 and hasattr(
                    yerr[0], "__len__"
                ):  # given as asymmetric error bars
                    pl_yerr = (
                        yerr[0][start : start + nres],
                        yerr[1][start : start + nres],
                    )
                else:
                    pl_yerr = yerr[start : start + nres]
            else:
                pl_yerr = None

        # if log_scale :
        #    if j==0 and y_range is not None : print "WARNING log_scale is overwriting y_range"
        #    _,to_plot,y_range =logscale(axt[j], entries=to_plot,add_one=True,add_zero=True)

        x_pos = list(range(start + start_rescount, start + nres + start_rescount))
        if ismulti:
            sep, bar_width = bar_sep / 2.0, (1.0 - bar_sep)
            if (
                n_bar_profiles > 0
            ):  # particularly important if some profles are bar and some are not.
                if stacked:
                    bottom = numpy.zeros(nres)
                    sep, bar_width = bar_sep, (1.0 - bar_sep)
                    left = numpy.array(x_pos) - 0.5 + sep / 2.0
                else:
                    sep, bar_width = (
                        float(bar_sep) / (n_bar_profiles + 1),
                        (1.0 - bar_sep) / n_bar_profiles,
                    )  # +1 is there so that bar groups will be separated by 2*sep
                    bottom = None  # n_bar_profiles above is the number of multiple profiles for which bar is true
                    left = numpy.array(x_pos) - 0.5 + sep

            if do_matrix:
                zygg_like_lines = False
                if hline == 0:
                    hline = None
                vlabs = None
                if value_labels is not None:
                    if (
                        type(value_labels) is tuple or type(value_labels) is list
                    ) and len(value_labels) == 2:
                        vlabs = (
                            value_labels[0][:, start : start + nres],
                            value_labels[1][:, start : start + nres],
                        )
                    else:
                        vlabs = value_labels[:, start : start + nres]

                if xlabels is not None:
                    xllab = xlabels[start : start + nres]
                else:
                    xllab = None
                xlabels_fontname = None
                if sequence_extra_numbering is not None:
                    if xllab is None:
                        # print("DEB: start", start, "nres", nres)
                        # print("DEB: sequence_extra_numbering",sequence_extra_numbering)
                        xllab = sequence_extra_numbering[start : start + nres]
                    else:
                        # xlabels_rotation='vertical'
                        xllab = xlabels[start : start + nres]
                        xlabels_fontname = monospaced_font
                        for jel, ell in enumerate(
                            sequence_extra_numbering[start : start + nres]
                        ):  # extra numbering goes first here to go last in xlabels when put vertically (default in this case)
                            if type(ell) is int or type(ell) is float:
                                ell = "%g" % (ell)
                            if type(xllab[jel]) is int:
                                xell = "%d" % (xllab[jel])
                            elif type(xllab[jel]) is float:
                                xell = " %g" % (xllab[jel])
                            elif xllab[jel] != "":
                                try:
                                    xell = "%d" % (int(xllab[jel]))
                                except Exception:
                                    xell = str(xllab[jel])
                            else:
                                xell = str(xllab[jel])
                            if xlabels_rotation == "vertical" or (
                                (
                                    type(xlabels_rotation) is int
                                    or type(xlabels_rotation) is float
                                )
                                and 25 <= xlabels_rotation <= 165
                            ):
                                if len(xell) >= 4:
                                    xllab[jel] = str(ell) + " " + xell
                                else:
                                    xllab[jel] = str(ell) + ("%4s" % (xell))
                            else:
                                xllab[jel] = str(ell) + "\n" + str(xell)
                    # print('DEB: xllab=',xllab)
                if cdr_profile is not None:
                    if plot_antibody_legend is None and plot_legend:
                        plot_antibody_legend = True
                    # print("DEB: to_plot.shape[0]",to_plot.shape[0],numpy.nanmax(cdr_profile))
                    l = axt[j].bar(
                        x_pos,
                        to_plot.shape[0]
                        * numpy.array(cdr_profile)[start : start + nres]
                        / numpy.nanmax(cdr_profile)
                        / (-18.0),
                        bottom=-0.05,
                        align="center",
                        width=0.8,
                        yerr=None,
                        linewidth=0,
                        color=(0.5, 0.5, 0.5, 0.4),
                        zorder=20,
                    )

                fig, image = plot_matrix(
                    to_plot,
                    vmin=vmin,
                    vmax=vmax,
                    subplot=axt[j],
                    figure=fig,
                    ylabels=ylabels,
                    extent=(
                        start + start_rescount - 0.5,
                        start + nres + start_rescount - 0.5,
                        -0.5,
                        to_plot.shape[0] - 0.5,
                    ),
                    cmap=do_matrix_cmap,
                    center_cmap_on_value=center_cmap_on_value,
                    value_labels=vlabs,
                    value_labels_size=text_sizes["xlabels_many"],
                    interpolation="nearest",
                    aspect="auto",
                    y_range=y_range,
                    xlabels=xllab,
                    xlabels_fontname=xlabels_fontname,
                    save=None,
                    show=False,
                    xlabels_rotation=xlabels_rotation,
                    labelsize=xlabels_size,
                )

            else:
                if annotate_antibody > 0 and cdr_profile is not None:
                    if plot_antibody_legend is None and plot_legend:
                        plot_antibody_legend = True
                    l = axt[j].bar(
                        x_pos,
                        cdr_profile[start : start + nres],
                        bottom=None,
                        align="center",
                        width=bar_width,
                        yerr=None,
                        linewidth=0,
                        color=(0.5, 0.5, 0.5, 0.4),
                        zorder=20,
                    )
                for i, prof in enumerate(to_plot):
                    if bar[i]:
                        if color is None:
                            l = axt[j].bar(
                                left,
                                prof,
                                bottom=bottom,
                                align="edge",
                                width=bar_width,
                                yerr=pl_yerr[i],
                                linewidth=lw[i],
                                color=color_set[i],
                                zorder=zorder[i],
                            )
                        elif type(color) is list or isinstance(color, cycle_list):
                            l = axt[j].bar(
                                left,
                                prof,
                                bottom=bottom,
                                align="edge",
                                yerr=pl_yerr[i],
                                width=bar_width,
                                linewidth=lw[i],
                                color=color[i],
                                zorder=zorder[i],
                            )
                        else:
                            l = axt[j].bar(
                                left,
                                prof,
                                bottom=bottom,
                                width=bar_width,
                                align="edge",
                                yerr=pl_yerr[i],
                                linewidth=lw[i],
                                color=color,
                                zorder=zorder[i],
                            )
                        if stacked:
                            bottom += numpy.array(prof)
                        else:
                            left += sep + bar_width
                        if start == 0:
                            line_styles += [l]  # save this only for the first subplot
                        del l
                    else:
                        if (
                            fill_error and pl_yerr[i] is not None
                        ):  # will set pl_yerr[i] to avoid plotting the bar (hence need to be called now)
                            # determine ytop and ybottom depending on error given and then set yerr to all None
                            ytop, ybottom, pl_yerr[i] = yerr_to_fill_errors(
                                prof, pl_yerr[i], ismulti=False, bar_list=None
                            )
                            # print 'ybottom, ytop', ybottom, ytop,pl_yerr[i],
                            # print len(prof),len(pl_yerr[i])
                        else:
                            ytop, ybottom = None, None
                        if color is None:
                            l = axt[j].errorbar(
                                x_pos,
                                prof,
                                linewidth=lw[i],
                                yerr=pl_yerr[i],
                                ls=ls[i],
                                elinewidth=default_error_bars["elinewidth"],
                                capsize=default_error_bars["capsize"],
                                capthick=default_error_bars["capthick"],
                                marker=marker[i],
                                markersize=markersize,
                                markeredgecolor=markeredgecolor,
                                markerfacecolor=markerfacecolor,
                                zorder=zorder[i],
                            )
                        elif type(color) is list or isinstance(color, cycle_list):
                            l = axt[j].errorbar(
                                x_pos,
                                prof,
                                yerr=pl_yerr[i],
                                linewidth=lw[i],
                                ls=ls[i],
                                color=color[i],
                                elinewidth=default_error_bars["elinewidth"],
                                capsize=default_error_bars["capsize"],
                                capthick=default_error_bars["capthick"],
                                marker=marker[i],
                                markersize=markersize,
                                markeredgecolor=markeredgecolor[i],
                                markerfacecolor=markerfacecolor[i],
                                zorder=zorder[i],
                            )
                        else:
                            l = axt[j].errorbar(
                                x_pos,
                                prof,
                                linewidth=lw[i],
                                yerr=pl_yerr[i],
                                ls=ls[i],
                                color=color,
                                elinewidth=default_error_bars["elinewidth"],
                                capsize=default_error_bars["capsize"],
                                capthick=default_error_bars["capthick"],
                                marker=marker[i],
                                markersize=markersize,
                                markeredgecolor=markeredgecolor,
                                markerfacecolor=markerfacecolor,
                                zorder=zorder[i],
                            )

                        if fill_error and ytop is not None:
                            axt[j].fill_between(
                                x_pos,
                                ybottom,
                                ytop,
                                alpha=fill_error_alpha,
                                color=l.lines[0].get_color(),
                                interpolate=False,
                            )  # edgecolor='#CC4F1B', facecolor='#FF9848')
                        if add_colorline[i] is not None and add_colorline[i] != False:
                            l[0].remove()
                            if type(add_colorline[i]) is str:
                                if add_colorline[i].lower() == "camsol":
                                    zf, cmap = (camsol_to_color_val, camsol_colormap)
                                    zygg_like_lines = True
                                elif "strucorr" in add_colorline[i].lower():
                                    zf, cmap = (
                                        camsol_to_color_val,
                                        camsol_strucorr_colormap,
                                    )
                                    zygg_like_lines = True
                            elif hasattr(add_colorline[i], "__len__"):
                                zf, cmap = add_colorline[i]
                            else:
                                zf, cmap = None, plt.get_cmap("copper")
                            colorline(
                                x_pos,
                                prof,
                                z=zf,
                                cmap=cmap,
                                ax=axt[j],
                                norm=plt.Normalize(0.0, 1.0),
                                linewidth=lw[i] * 1.2,
                                alpha=1.0,
                                interpolate=len(x_pos) * 10,
                                zorder=zorder[i] + 1,
                            )
                        if start == 0:
                            line_styles += [l]
                        del l
                    # if value_labels is not None and value_labels[j] is not None :
                    # we use another function below that relies on plt.annotate
                    # add_value_labels(value_labels[j], x_pos, prof, fontsize=None ,ax=axt[j],extra_points=5,initial_off_x=10,initial_off_y=10,max_attempts=10,textcolor='black',initial_offset_ha_va=None,alternate=False,offset_from_linfit=True,avoid_hitting_points=True,xerr=None,yerr=None,bbox = dict(boxstyle = 'round,pad=0.1', fc = 'white', alpha = 0.5), arrowprops = dict(arrowstyle = '-',lw=None,connectionstyle=None) )
        else:
            if annotate_antibody > 0 and cdr_profile is not None:
                if plot_antibody_legend is None:
                    plot_antibody_legend = True
                l = axt[j].bar(
                    x_pos,
                    cdr_profile[start : start + nres],
                    bottom=None,
                    align="center",
                    width=bar_width,
                    yerr=None,
                    linewidth=0,
                    color=(0.5, 0.5, 0.5, 0.4),
                    zorder=0,
                )
            if bar:
                sep, bar_width = bar_sep, (1.0 - bar_sep)
                left = numpy.array(x_pos) - 0.5 + sep / 2.0
                l = axt[j].bar(
                    left,
                    to_plot,
                    width=bar_width,
                    align="edge",
                    yerr=pl_yerr,
                    linewidth=lw,
                    color=color,
                    zorder=zorder,
                )
                if start == 0:
                    line_styles += [l]
                del l
            else:
                if fill_error and pl_yerr is not None:
                    # determine ytop and ybottom depending on error given and then set yerr to all None
                    ytop, ybottom, pl_yerr = yerr_to_fill_errors(
                        to_plot, pl_yerr, ismulti=False, bar_list=None
                    )
                else:
                    ytop, ybottom = None, None
                if color is None:
                    l = axt[j].errorbar(
                        x_pos,
                        to_plot,
                        linewidth=lw,
                        yerr=pl_yerr,
                        ls=ls,
                        elinewidth=default_error_bars["elinewidth"],
                        capsize=default_error_bars["capsize"],
                        capthick=default_error_bars["capthick"],
                        marker=marker,
                        markersize=markersize,
                        markeredgecolor=markeredgecolor,
                        markerfacecolor=markerfacecolor,
                        zorder=zorder,
                    )
                else:
                    l = axt[j].errorbar(
                        x_pos,
                        to_plot,
                        linewidth=lw,
                        yerr=pl_yerr,
                        ls=ls,
                        color=color,
                        elinewidth=default_error_bars["elinewidth"],
                        capsize=default_error_bars["capsize"],
                        capthick=default_error_bars["capthick"],
                        marker=marker,
                        markersize=markersize,
                        markeredgecolor=markeredgecolor,
                        markerfacecolor=markerfacecolor,
                        zorder=zorder,
                    )
                if fill_error and ytop is not None:
                    axt[j].fill_between(
                        x_pos,
                        ybottom,
                        ytop,
                        alpha=fill_error_alpha,
                        color=l.lines[0].get_color(),
                        interpolate=False,
                    )  # edgecolor='#CC4F1B', facecolor='#FF9848')
                if add_colorline is not None and add_colorline != False:
                    l[0].remove()
                    if type(add_colorline) is str:
                        if add_colorline.lower() == "camsol":
                            zf, cmap = (camsol_to_color_val, camsol_colormap)
                            zygg_like_lines = True
                        elif "strucorr" in add_colorline[i].lower():
                            zf, cmap = (camsol_to_color_val, camsol_strucorr_colormap)
                            zygg_like_lines = True
                    elif hasattr(add_colorline, "__len__"):
                        zf, cmap = add_colorline
                    else:
                        zf, cmap = None, plt.get_cmap("copper")
                    colorline(
                        x_pos,
                        to_plot,
                        z=zf,
                        cmap=cmap,
                        ax=axt[j],
                        norm=plt.Normalize(0.0, 1.0),
                        interpolate=len(to_plot) * 10,
                        linewidth=lw * 1.2,
                        alpha=1.0,
                        zorder=zorder + 1,
                    )
                if start == 0:
                    line_styles += [l]
                del l
        if value_labels is not None and not do_matrix:
            if ismulti:
                vlabs = [
                    vals[start : start + nres] if vals is not None else None
                    for vals in value_labels
                ]
                zord = numpy.nanmax(zorder) + 1
            else:
                vlabs = value_labels[start : start + nres]
                zord = zorder + 1
            handle_value_labels(
                vlabs,
                x_pos,
                to_plot,
                yerr=pl_yerr,
                ismulti=ismulti,
                axis=axt[j],
                value_labels_rotation=value_labels_rotation,
                value_labels_ypos="top",
                zorder=zord,
                default_text_offset=None,
            )
        # axt[j].set_xlim(start+start_rescount,start+nres+start_rescount-1)

        if hline is not None:
            if not hasattr(hline, "__len__"):
                hline = [hline]
            for ypos in hline:
                axt[j].axhline(
                    ypos, color="black", ls="-"
                )  # use thick line in this case, it represent the axis)
        if vline is not None:
            if not hasattr(vline, "__len__"):
                vline = [vline]
            for xvl in vline:
                if xvl < x_pos[0] or xvl > x_pos[-1]:
                    continue  # not in this j subplot
                axt[j].axvline(
                    xvl, color="black", ls="-"
                )  # use thick line in this case, it represent the axis)
        if zygg_like_lines != False:
            if type(zygg_like_lines) is not list or type(zygg_like_lines) is not tuple:
                zygg_like_lines = (-1, 1)
            for ypos in zygg_like_lines:
                axt[j].axhline(
                    ypos, color="black", ls="--", lw=0.5
                )  # use thick line in this case, it represent the axis)

        if log_scale:
            if type(log_scale) is int:
                logbase = log_scale
            else:
                logbase = 10
            axt[j].set_yscale("symlog", base=logbase)
            if avoid_scientific_notation:
                yticks = axt[j].yaxis.get_majorticklocs()
                xlab = [int(i) for i in yticks]
                print(
                    "avoid_scientific_notation is rounding to int:", yticks, "==>", xlab
                )
                axt[j].set_yticklabels(
                    xlab,
                    rotation="horizontal",
                    verticalalignment="center",
                    horizontalalignment="right",
                    fontsize=text_sizes["xlabels"],
                )

        if y_minor_tick_every is not None:
            if type(y_minor_tick_every) is bool and y_minor_tick_every == False:
                axt[j].yaxis.set_ticks([], minor=True)
            else:
                yminorLocator = matplotlib.ticker.MultipleLocator(y_minor_tick_every)
                # for the minor ticks, use no labels; default NullFormatter
                axt[j].yaxis.set_minor_locator(yminorLocator)
        if y_major_tick_every is not None:
            if type(y_major_tick_every) is bool and y_major_tick_every == False:
                axt[j].yaxis.set_ticks([])
            else:
                ymajorLocator = matplotlib.ticker.MultipleLocator(y_major_tick_every)
                axt[j].yaxis.set_major_locator(ymajorLocator)
        if x_minor_tick_every is not None:
            if type(x_minor_tick_every) is bool and x_minor_tick_every == False:
                axt[j].xaxis.set_ticks([], minor=True)
            else:
                xminorLocator = matplotlib.ticker.MultipleLocator(x_minor_tick_every)
                # for the minor ticks, use no labels; default NullFormatter
                axt[j].xaxis.set_minor_locator(xminorLocator)
        already_done_xlabels = False
        # print('DEB: x_major_tick_every',x_major_tick_every,'xlabels',type(xlabels))
        if x_major_tick_every is not None:  # and not do_matrix :
            if type(x_major_tick_every) is bool and x_major_tick_every == False:
                axt[j].xaxis.set_ticks([])
            elif type(x_major_tick_every) is int:
                # xmajorLocator   = matplotlib.ticker.MultipleLocator(x_major_tick_every)
                # axt[j].xaxis.set_major_locator(xmajorLocator)
                if (
                    x_pos[0] == 1 and x_major_tick_every > 1
                ):  # common for sequences, in this way ticks will be e.g. 1,5,10,15,... rather than 1,6,11,...
                    axt[j].set_xticks(
                        [x_pos[0]]
                        + list(x_pos[x_major_tick_every - 1 :: x_major_tick_every])
                    )
                else:
                    axt[j].set_xticks(x_pos[::x_major_tick_every])
            elif hasattr(x_major_tick_every, "__len__"):
                xmm = numpy.array(x_major_tick_every)
                inds_xmm = numpy.where((xmm >= x_pos[0]) & (xmm <= x_pos[-1]))[0]
                # print ('x_major_tick_every:',numpy.array(x_major_tick_every),'-->xmm=',xmm[inds_xmm],x_pos[0],x_pos[-1])
                axt[j].set_xticks(xmm[inds_xmm])
                if (
                    xlabels is not None
                    and hasattr(xlabels, "__len__")
                    and len(xlabels) == len(x_major_tick_every)
                ):  # given together as matching pairs
                    axt[j].set_xticklabels(
                        numpy.array(xlabels)[inds_xmm],
                        verticalalignment="top",
                        fontsize=xlabels_size,
                        rotation=xlabels_rotation,
                    )
                    already_done_xlabels = True
        # axt[j].set_ylim(ymin, ymax )
        if not do_matrix:  # if do matrix these are handled therein
            if ylabels is False or ylabels == []:
                axt[j].set_yticklabels([])
            if xlabels is not None or sequence_extra_numbering is not None:
                if xlabels is False:
                    axt[j].set_xticklabels([])
                else:
                    if xlabels is None:
                        xllab = sequence_extra_numbering[start : start + nres]
                    elif sequence_extra_numbering is not None:
                        xllab = xlabels[start : start + nres]
                        for jel, ell in enumerate(
                            sequence_extra_numbering[start : start + nres]
                        ):  # extra numbering goes first here to go last in xlabels when put vertically (default in this case)
                            if type(ell) is int or type(ell) is float:
                                ell = "%g" % (ell)
                            if type(xllab[jel]) is int:
                                xell = "%d" % (xllab[jel])
                            elif type(xllab[jel]) is float:
                                xell = " %g" % (xllab[jel])
                            elif xllab[jel] != "":
                                try:
                                    xell = "%d" % (int(xllab[jel]))
                                except Exception:
                                    xell = str(xllab[jel])
                            else:
                                xell = str(xllab[jel])
                            if xlabels_rotation == "vertical" or (
                                (
                                    type(xlabels_rotation) is int
                                    or type(xlabels_rotation) is float
                                )
                                and 25 <= xlabels_rotation <= 165
                            ):
                                if len(xell) >= 4:
                                    xllab[jel] = str(ell) + " " + xell
                                else:
                                    xllab[jel] = str(ell) + ("%4s" % (xell))
                            else:
                                xllab[jel] = str(ell) + "\n" + str(xell)
                        # print('DEB: xllab=',xllab)
                    elif not already_done_xlabels:
                        xllab = xlabels[start : start + nres]
                    if not already_done_xlabels and len(xllab) == len(x_pos):
                        if x_major_tick_every is None:
                            axt[j].set_xticks(x_pos)
                        elif type(x_major_tick_every) is int:
                            if x_pos[0] == 1 and x_major_tick_every > 1:
                                xllab = [xllab[0]] + list(
                                    xllab[x_major_tick_every - 1 :: x_major_tick_every]
                                )
                            else:
                                xllab = xllab[::x_major_tick_every]
                    # print "DEB: xlabels_size",xlabels_size
                    if sequence_extra_numbering is not None:
                        axt[j].set_xticklabels(
                            xllab,
                            verticalalignment="top",
                            fontsize=xlabels_size,
                            fontname=monospaced_font,
                            rotation=xlabels_rotation,
                        )
                    elif not already_done_xlabels:
                        axt[j].set_xticklabels(
                            xllab,
                            verticalalignment="top",
                            fontsize=xlabels_size,
                            rotation=xlabels_rotation,
                        )  # Monospace font not required
            elif x_major_tick_every is None:  # adjust numbering
                # if x_major_tick_every is None :
                xticks = axt[j].xaxis.get_majorticklocs()
                xticks = list(map(float, xticks))
                sp = 1.0 * nres
                to_remove = []
                for x in xticks:
                    if abs((x - start - start_rescount) / sp) < 0.1:
                        to_remove.append(
                            x
                        )  # if too close to start or end remove - then start and end will be put back
                    elif abs((start + nres + start_rescount - 1 - x) / sp) < 0.1:
                        to_remove.append(x)
                # print to_remove,abs((xticks[0]-start-start_rescount)/sp),abs((start+nres+start_rescount-1-xticks[-1])/sp)
                for x in to_remove:
                    xticks.remove(x)
                    xticks += [
                        start + start_rescount,
                        start + nres + start_rescount - 1,
                    ]  # add first and last tick, the set_xticks below will sort them correctly.
                    axt[j].set_xticks(xticks)
            axt[j].set_xlim(xlim_m, xlim_M)

        handle_grid(axt[j], vgrid=False, hgrid=hgrid)  # custom vgrid for this plot
        if vgrid:
            if hasattr(vgrid, "__len__"):
                for vl in vgrid:
                    axt[j].axvline(
                        vl,
                        color=grid_parameters["vcolor"],
                        ls=grid_parameters["v_ls"],
                        lw=grid_parameters["v_lw"],
                    )
            else:
                for count in x_pos:
                    if type(vgrid) is int:
                        if count % vgrid == 0:
                            axt[j].axvline(
                                count,
                                color=grid_parameters["vcolor"],
                                ls=grid_parameters["v_ls"],
                                lw=grid_parameters["v_lw"],
                            )
                    elif count % 10 == 0:
                        axt[j].axvline(
                            count,
                            color=grid_parameters["vcolor"],
                            ls=grid_parameters["v_ls"],
                            lw=grid_parameters["v_lw"],
                        )
                # axt[j].annotate(sequence[count-1], xy=(count,ymin), xytext=(0, -5),rotation=rotation, textcoords='offset points', va='top', ha='center',size='small')
        if sequence is None and annotation_string is not None:
            sequence = annotation_string
            annotation_string = None
            upper_label_rotation = annotation_string_rotation
        if sequence is not None or annotation_string is not None:
            try:
                ax2 = axt[j].secondary_xaxis("top")
                axt[j].ax2 = ax2
            except Exception:
                sys.stderr.write(
                    "  matplotlib secondary_xaxis failed, consider upgrading matplotlib version\n"
                )
                ax2 = axt[j].twiny()
                axt[j].ax2 = ax2
            ax2.set_xlim(axt[j].get_xlim())
            if print_all_sequence == True:
                ju = 1
            elif type(print_all_sequence) is int:
                ju = print_all_sequence
            else:
                ju = 3
            ax2.set_xticks(x_pos, minor=True)  #
            ax2.set_xticks(
                list(range(start + start_rescount, start + nres + start_rescount, ju))
            )
            ax2_ticks = ax2.get_xticks()
            if sequence is not None:
                ax2.set_xticklabels(
                    sequence[start : start + nres : ju],
                    rotation=upper_label_rotation,
                    verticalalignment="bottom",
                    fontsize=sequence_fontsize,
                )
            plt.sca(axt[j])  # set back current axis if needed outside function
            # an=list(sequence[start:start+nres:ju])
            # for ja,xt in enumerate(ax2_ticks[::ju]) :
            #    axt[j].annotate( an[ja], (xt,ymax),(0,5), xycoords='data' \
            #    , size=text_sizes['xlabels_many'],textcoords = 'offset points', ha = 'center', va = 'bottom' )
        if annotation_string is not None:
            ha = "center"
            if (
                annotation_string_position is None
                or annotation_string_position == "lower"
                or annotation_string_position == "bottom"
            ):
                annotation_string_position, va = -5, "top"
                if annotation_string_rotation == "vertical":
                    annotation_string_rotation, va = 270, "top"
            elif annotation_string_position == "top":
                if type(text_sizes["xlabels_many"]) is int:
                    offf = (
                        text_sizes["xlabels_many"]
                        + int(0.5 * text_sizes["xlabels_many"] + 0.5)
                        + 1
                    )
                else:
                    offf = 15
                annotation_string_position, va = offf + 5, "bottom"
                if annotation_string_rotation == "vertical":
                    annotation_string_rotation, va, ha = (
                        90,
                        "bottom",
                        "center",
                    )  # change ha for some reason center puts it to the left on the saved png
                elif annotation_string_rotation == "horizontal":
                    annotation_string_position -= int(
                        0.1 * annotation_string_position + 0.5
                    )
            if do_matrix:
                ymax = axt[j].get_ylim()[1]
            an = list(annotation_string[start : start + nres : ju])
            for ja, xt in enumerate(ax2_ticks):
                axt[j].annotate(
                    an[ja],
                    (xt, ymax),
                    (0, annotation_string_position),
                    xycoords="data",
                    size=text_sizes["xlabels_many"],
                    textcoords="offset points",
                    rotation=annotation_string_rotation,
                    ha=ha,
                    va=va,
                )
        start = start + nres
        if not frame:
            # this remove top and right axis
            axt[j].spines["right"].set_visible(False)
            axt[j].spines["top"].set_visible(False)
            axt[j].get_xaxis().tick_bottom()  # ticks only on bottom axis
            axt[j].get_yaxis().tick_left()  # ticks only on left axis
    # yticks=axt[0].yaxis.get_majorticklocs() # the y axis is shared
    # yticks=map(float,yticks)
    # yticks.remove(min(yticks))
    # axt[0].set_yticks(yticks)
    if (
        sequence_extra_numbering_name is not None
        and sequence_extra_numbering is not None
    ):
        if xlabel is not None and xlabel != "":
            xlabel = xlabel + "\n& " + sequence_extra_numbering_name
        else:
            xlabel = sequence_extra_numbering_name
    if not do_matrix:
        axt[0].set_ylim(ymin, ymax)
    if xlabel is not None and xlabel != False:
        fig.text(
            0.5, 0.03, xlabel, fontsize=text_sizes["xlabel"], ha="center", va="center"
        )
    if ylabel is not None:
        fig.text(
            0.015,
            0.5,
            ylabel,
            fontsize=text_sizes["ylabel"],
            rotation="vertical",
            ha="center",
            va="center",
        )
    if title is not None:
        fig.text(
            0.5, 0.97, title, horizontalalignment="center", fontsize=text_sizes["title"]
        )
    legend = None
    if plot_legend and label != "" and label is not None:
        if type(label) is not list and type(label) is not tuple:
            label = [label]
        legend = plt.legend(
            line_styles,
            label,
            loc=legend_location,
            prop={"size": legend_size},
            frameon=True,
            framealpha=0.5,
            ncol=ncol,
        )
        legendframe = legend.get_frame()
        legendframe.set_facecolor((1.0, 1.0, 1.0, 0.7))
    if do_tight:
        fig.tight_layout(pad=3.5, h_pad=1.08, w_pad=1.08, rect=(0, 0, 1, 1))
    # if default_figure_sizes['all_tight'] : figure.tight_layout()

    plt.draw()
    if do_matrix and plot_colorbar:
        fig = add_colorbar_to_seq_profile(
            fig,
            image,
            cbar_major_ntick=None,
            cbar_label=cbar_label,
            cbar_label_rotation=270,
            cbar_fraction=0.02,
        )
    if plot_antibody_legend is not None and plot_antibody_legend == True:
        # if legend_location=='lower right' : cdr_legend_location='lower left' # will be outside axis anyway
        # else :
        cdr_legend_location = "lower right"  # (0.8 , 0.05)#
        cdr_legend_fit_in_axis = False
        add_custom_legend(
            ["CDR region"],
            facecolors=[(0.5, 0.5, 0.5, 0.4)],
            figure=fig,
            fit_in_axis=cdr_legend_fit_in_axis,
            ncol=1,
            legend_location=cdr_legend_location,
            legend_size=legend_size - 4,
        )
    if save is not None and save != "":
        if "." not in save:
            save += ".pdf"
        if dpi is None:
            dpi = default_figure_sizes["dpi"]
        if legend is not None:
            fig.savefig(
                save,
                dpi=dpi,
                transparent=True,
                bbox_inches="tight",
                bbox_extra_artists=[fig, legend] + [ax for ax in axt],
            )
        else:
            fig.savefig(
                save,
                dpi=dpi,
                transparent=True,
                bbox_inches="tight",
                bbox_extra_artists=[fig] + [ax for ax in axt],
            )
        # else : fig.savefig(save, dpi=dpi,bbox_inches="tight",transparent=True) #  bbox_inches=0 remove white space around the figure.. ,
    if show:
        plt.show(block=block)
    if to_return is not None:
        return (fig, axt) + to_return
    return fig, axt


def broken_axis(
    y_values,
    y_break,
    y_restart,
    x_values=None,
    bar=False,
    label=None,
    show=True,
    linewidth=1,
):
    """
    Broken axis example, where the y-axis will have a portion cut out.
    """
    # If we were to simply plot y_values, we'd lose most of the interesting
    # details due to the outliers. So let's 'break' or 'cut-out' the y-axis
    # into two portions - use the top (ax) for the outliers, and the bottom
    # (ax2) for the details of the majority of our data
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    edge = ["blue", "green", "red"]

    if type(y_values[0]) is not list:  # trick to superimpose
        y_values = [y_values]
        label = [label]
        if x_values is not None:
            x_values = [x_values]
    if label is None:
        label = [None] * len(y_values)
    for j, yval in enumerate(y_values):
        # plot the same data on both axes
        ax.plot(yval)
        ax2.plot(yval)
        if bar:
            if x_values is not None:
                width = abs(x_values[j][1] - x_values[j][0])
                ax.bar(
                    x_values[j],
                    yval,
                    label=label[j],
                    linewidth=linewidth,
                    width=width,
                    align="edge",
                    edgecolor=edge[j],
                    color=(0, 0, 0.0, 0.0),
                )  # ,color_code,label=label,linewidth=linewidth
                ax2.bar(
                    x_values[j],
                    yval,
                    label=label[j],
                    linewidth=linewidth,
                    width=width,
                    align="edge",
                    edgecolor=edge[j],
                    color=(0, 0, 0.0, 0.0),
                )  # ,color_code,label=label,linewidth=linewidth
            else:
                width = 0.8
                ax.bar(
                    list(range(0, len(yval))),
                    yval,
                    label=label[j],
                    linewidth=linewidth,
                    width=width,
                    align="edge",
                )  # ,color_code,label=label,linewidth=linewidth
                ax2.bar(
                    list(range(0, len(yval))),
                    yval,
                    label=label[j],
                    linewidth=linewidth,
                    width=width,
                    align="edge",
                )  # ,color_code,label=label,linewidth=linewidth
        else:
            if x_values is not None:
                ax.plot(x_values[j], yval, label=label[j], linewidth=linewidth)
                ax2.plot(x_values[j], yval, label=label[j], linewidth=linewidth)
            else:
                ax.plot(yval, label=label[j], linewidth=linewidth)
                ax2.plot(yval, label=label[j], linewidth=linewidth)
    # zoom-in / limit the view to different portions of the data
    Min, Max = ax.get_ylim()
    ax.set_ylim(y_restart, Max)  # outliers only
    ax2.set_ylim(Min, y_break)  # most of the data

    # hide the spines between ax and ax2
    ax.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop="off")  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # This looks pretty good, and was fairly painless, but you can get that
    # cut-out diagonal lines look with just a bit more work. The important
    # thing to know here is that in axes coordinates, which are always
    # between 0-1, spine endpoints are at these locations (0,0), (0,1),
    # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
    # appropriate corners of each of our axes, and so long as we use the
    # right transform and disable clipping.

    d = 0.015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # What's cool about this is that now if we vary the distance between
    # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
    # the diagonal lines will move accordingly, and stay right at the tips
    # of the spines they are 'breaking'
    plt.legend()
    plt.draw()
    if show:
        plt.show()


def pvalue_to_stars(p, non_significant="", symbol="*"):
    if hasattr(p, "__len__"):  # handles up to one list
        return [
            pvalue_to_stars(pv, non_significant=non_significant, symbol=symbol)
            for pv in p
        ]
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


def add_custom_legend(
    labels,
    facecolors,
    edgecolors=None,
    marker_types=None,
    markersize=None,
    linewidth=None,
    linestyle=None,
    figure=None,
    fit_in_axis=False,
    proxy_point=1,
    frame=True,
    ncol=2,
    legend_location="upper right",
    bbox_to_anchor=None,
    legend_size=None,
    frame_facecolor=None,
    shadow=False,
    framealpha=None,
    save=None,
):
    """
    this will draw a custom legend to the existing figure (or to figure if given)
    if you want to represent a line for label j give facecolor[j]=None and the desired edgecolor,
    if you wish to represent a marker give marker_types[j]  is not None
    frame_facecolor=(1.,1.,1.,0.7) give alpha of 0.7 to a white background
    legend_location can also be a 2-tuple giving the coordinates of the lower-left corner of the legend in axes coordinates (fraction)
    if save is given it saves the figure after adding the legend
    return leg
    """
    if figure is not None:
        ax_fig = figure.gca()
    else:
        figure = plt.gcf()
        ax_fig = figure.gca()
    ylims = ax_fig.get_ylim()
    if legend_size is None:
        legend_size = text_sizes["legend_size"]
    proxy_point = int(proxy_point)
    if type(labels) is str or not hasattr(labels, "__len__"):
        labels = [labels]
    if type(facecolors) is not list and not isinstance(facecolors, cycle_list):
        facecolors = [facecolors] * len(labels)
    if type(edgecolors) is not list and not isinstance(edgecolors, cycle_list):
        if edgecolors is None:
            edgecolors = [None] * len(labels)
        else:
            edgecolors = [edgecolors] * len(labels)
    if type(marker_types) is not list and not isinstance(marker_types, cycle_list):
        if marker_types is None:
            marker_types = [None] * len(labels)
        else:
            marker_types = [marker_types] * len(labels)
    if type(markersize) is not list and not isinstance(markersize, cycle_list):
        markersize = [markersize] * len(labels)
    if type(linewidth) is not list and not isinstance(linewidth, cycle_list):
        if linewidth is None:
            linewidth = [plt.rcParams["lines.linewidth"]] * len(labels)
        else:
            linewidth = [linewidth] * len(labels)
    if type(linestyle) is not list and not isinstance(linestyle, cycle_list):
        if linestyle is None:
            linestyle = [plt.rcParams["lines.linestyle"]] * len(labels)
        else:
            linestyle = [linestyle] * len(labels)

    proxy_artists = []
    for j in range(len(labels)):
        if marker_types[j] is not None:
            pro = plt.Line2D(
                list(range(proxy_point)),
                list(range(proxy_point)),
                color="white",
                marker=marker_types[j],
                markersize=markersize[j],
                markerfacecolor=facecolors[j],
                markeredgecolor=edgecolors[j],
                linewidth=linewidth[j],
                linestyle=linestyle[j],
            )
        elif facecolors[j] is None:
            pro = plt.hlines(
                y=proxy_point,
                xmin=proxy_point,
                xmax=proxy_point,
                color=edgecolors[j],
                linewidth=linewidth[j],
                linestyle=linestyle[j],
            )
        else:
            pro = plt.Rectangle(
                (proxy_point, proxy_point),
                0,
                0,
                facecolor=facecolors[j],
                edgecolor=edgecolors[j],
                linewidth=linewidth[j],
                linestyle=linestyle[j],
            )  # ,linewidth=2)
        proxy_artists += [pro]
    if fit_in_axis:
        leg = ax_fig.legend(
            proxy_artists,
            labels,
            frameon=frame,
            numpoints=1,
            bbox_to_anchor=bbox_to_anchor,
            loc=legend_location,
            prop={"size": legend_size},
            ncol=ncol,
            shadow=shadow,
            framealpha=framealpha,
        )
    else:
        leg = figure.legend(
            proxy_artists,
            labels,
            frameon=frame,
            numpoints=1,
            bbox_to_anchor=bbox_to_anchor,
            loc=legend_location,
            prop={"size": legend_size},
            ncol=ncol,
            shadow=shadow,
            framealpha=framealpha,
        )
    if frame_facecolor is not None:
        # leg=fig.legend(line_styles, label, loc=legend_location,prop={'size':legend_size},frameon=True,framealpha=0.5)
        legendframe = leg.get_frame()
        legendframe.set_facecolor(frame_facecolor)
    ax_fig.set_ylim(ylims)
    # print('ax_fig.set_ylim(ylims)',ylims)
    plt.draw()
    if save is not None:
        if ".png" in save:
            figure.savefig(
                save,
                dpi=default_figure_sizes["dpi"],
                bbox_inches="tight",
                bbox_extra_artists=[figure.gca(), figure, leg],
                transparent=True,
            )  # ,bbox_inches="tight" is such that everything is in the figure... even though margins are removed.
        else:
            figure.savefig(
                save,
                dpi=default_figure_sizes["dpi"],
                bbox_inches="tight",
                bbox_extra_artists=[figure.gca(), figure, leg],
            )  # ,bbox_inches="tight" is such that everything is in the figure... even though margins are removed.
    return leg


def get_min_max_glob(entries, dilate_range_by_fraction=0):
    # up to three dimensions where various profiles within entries could have different shapes or lengths
    entries = [numpy.nan if type(x) is str else x for x in entries]
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
    return m - dilate_range_by_fraction * (M - m), M + dilate_range_by_fraction * (
        M - m
    )


def adjust_ax_ranges(ax, dilate_range_by_fraction=0.02, shape_filter=None):
    """
    automatically adjust x_range and y_range (axes lims) by retrieving plotted data
    doesn't really work for subplots unless you have a shape filter that corresponds to the shape of the data given as input
        e.g. shape_filter=(1,2) (you plotted one point at a time)
    """
    xydat = None
    for line in ax.get_lines():
        dat = line.get_xydata()
        if shape_filter is not None and shape_filter != dat.shape:
            continue
        if xydat is None:
            xydat = dat
        else:
            xydat = numpy.vstack((xydat, dat))
    if xydat is not None:
        xm, xM = get_min_max_glob(
            xydat.T[0], dilate_range_by_fraction=dilate_range_by_fraction
        )
        ax.set_xlim((xm, xM))
        ym, yM = get_min_max_glob(
            xydat.T[1], dilate_range_by_fraction=dilate_range_by_fraction
        )
        ax.set_ylim((ym, yM))
        plt.draw()
    else:
        sys.stderr.write(
            "**WARNING** in plotter adjust_ax_ranges no data retrieved (shape_filter=%s)- nothing done!\n"
            % (str(shape_filter))
        )
    return ax


def Round_To_n(value, n=0, only_decimals=False):
    """
    rounds to n significant digits
    Note that 0 would leave 1 significant digit,
    if not only_decimals then
       n=0 ==> 1111 --> 1000
       n=0 ==> 1.1534142 --> 1.0
       n=1 ==> 1.1534142 --> 1.2
    if only_decimals then
       n=0 ==> 1.1534142 --> 1.2
       n=1 ==> 1.1534142 --> 1.15
    will return a float (or array of floats)
    """
    if not only_decimals:
        n += 1  # back compatibility
    value = numpy.asarray(value).copy()
    if only_decimals:
        int_part = numpy.array(value).astype("int")
        return int_part + Round_To_n(value - int_part, n)
    zero_mask = value == 0
    value[zero_mask] = 1.0
    sign_mask = value < 0
    value[sign_mask] *= -1
    exponent = numpy.ceil(numpy.log10(value))
    result = 10**exponent * numpy.round(value * 10 ** (-exponent), n)
    if not hasattr(result, "__len__"):
        if zero_mask:
            return 0
        if sign_mask:
            return -1 * result
        return result
    result[sign_mask] *= -1
    result[zero_mask] = 0.0
    return result
    """ OLD CODE:
    if isinstance(x, numpy.ndarray) :
        if only_decimals :
            int_part=int(x)
            return float(int_part) + Round_To_n(x-int_part,n)
        return round(x, -int(numpy.floor(numpy.sign(x) * numpy.log10(abs(x)))) + n)
    if x==0 : return 0.
    if only_decimals :
        if x>1 :
            if n==0 : return int(x+0.5)
            n-=1
        int_part=int(x)
        return float(int_part) + Round_To_n(x-int_part,n)
    return round(x, -int(numpy.floor(numpy.sign(x) * numpy.log10(abs(x)))) + n)
    """


def round_with_error(x, xerr, nextra_significant_err=1, only_decimals=True):
    """
    rounds a value according to the signficant digits of its error
    A CI can be given as a tuple instead of the error
    """
    if hasattr(xerr, "__len__"):  # CI given
        if xerr[0] < x < xerr[1]:
            n = get_order_of_magnitude(xerr[0])
            n += numpy.sign(n) * nextra_significant_err
            if n >= 0 and only_decimals:
                n = -nextra_significant_err
            m = get_order_of_magnitude(xerr[1])
            m += numpy.sign(m) * nextra_significant_err
            if m >= 0 and only_decimals:
                m = -nextra_significant_err
            if n < 0 and m < n:
                n = m
            return (
                numpy.round(x, -n),
                numpy.round(xerr[0], -n),
                numpy.round(xerr[1], -n),
            )
        else:
            print(
                "ERROR in () CI given but not xerr[0]<x<xerr[1] %lf< %lf< %lf"
                % (xerr[0], x, xerr[1])
            )
            return
    n = get_order_of_magnitude(xerr)
    n += numpy.sign(n) * nextra_significant_err
    if n >= 0 and only_decimals:
        n = -nextra_significant_err
    return numpy.round(x, -n), numpy.round(xerr, -n)


def get_order_of_magnitude(x):
    """
    returns the order of magnitude of abs(x), i.e. the exponent you would give to x to write it in scientific notation
    """
    x = abs(x)
    if x > 1:
        mg = int(numpy.log10(x))
    elif x == 0:
        return 0
    else:
        mg = -1 * int(numpy.log10(1.0 / x)) - 1  # get order of magnitude
        mg = (
            -1 * int(numpy.log10(1.0 / (x + 10 ** (mg - 1)))) - 1
        )  # necessary as 0.01 ->1 while 0.011->2
    return mg


# ax,add_to_axis_label = consider_scientific_notation(ax,axis='y', publication=default_parameters['set_publish']) # at the moment we use all_tight as an alias for publication quality y/n
def consider_scientific_notation(
    ax,
    publication=False,
    up_limit=10000,
    low_limit=0.1,
    axis="y",
    print_warn=True,
    rotation="horizontal",
    fontsize=text_sizes["xlabels"],
):
    add_to_axis_label = ""
    return ax, add_to_axis_label
    ## NEEDS FIXING IT CAUSES ALL SORTS OF MESS
    if publication:
        if axis == "y":
            ticks = ax.yaxis.get_majorticklocs()
        elif axis == "x":
            ticks = ax.xaxis.get_majorticklocs()
        else:
            raise Exception(
                " Error in consider_scientific_notation() axis %s not recognized! "
                % (str(axis))
            )
        if hasattr(ticks, "__len__") and ticks.shape != () and len(ticks) > 0:
            try:
                tmp_ticks = list(ticks)
                if 0 in tmp_ticks:
                    tmp_ticks.remove(0)  # use to estimate order of magintude
                tmp_ticks = numpy.array(tmp_ticks)
                ma = max(abs(tmp_ticks))
                mi = min(abs(tmp_ticks))
                if ma > up_limit or ma < low_limit:
                    mg = get_order_of_magnitude(ma)
                    mg2 = get_order_of_magnitude(mi)
                    if mg2 < mg:
                        mg = mg2

                    add_to_axis_label = r" x $10^{%d}$" % (mg)
                    ###
                    print(ticks, "then scientific notation -> ", end=" ")
                    ticks = ticks / (10**mg)
                    ticks = [
                        int(numpy.round(x)) if abs(x - numpy.round(x)) < 0.001 else x
                        for x in ticks
                    ]  # try to round to closer int
                    print(ticks)
                    if axis == "y":
                        ax.set_yticklabels(
                            ticks,
                            rotation=rotation,
                            verticalalignment="center",
                            fontsize=fontsize,
                        )
                    else:
                        ax.set_xticklabels(
                            ticks,
                            rotation=rotation,
                            verticalalignment="top",
                            fontsize=fontsize,
                        )
                    if print_warn:
                        print(
                            "WARNING "
                            + axis
                            + " change changed to scientific notation with "
                            + add_to_axis_label
                        )
            except Exception:
                sys.stderr.write(
                    "\n***ERROR*** in consider_scientific_notation() tmp_ticks=%s\n"
                    % (str(tmp_ticks))
                )
                print_exc(file=sys.stderr)
                sys.stderr.flush()
                pass
    else:
        try:
            ax.ticklabel_format(
                style="sci", scilimits=(-2, 3), axis=axis
            )  # use scientific notation
        except Exception:
            sys.stderr.write("\n***ERROR*** in consider_scientific_notation() \n")
            print_exc(file=sys.stderr)
            sys.stderr.flush()
            pass
    return ax, add_to_axis_label


def logscale(ax, entries=None, add_one=True, add_zero=True):
    """
    if entries is given it must be called before plotting
    """
    M = None
    if entries is not None:
        entries = numpy.array(entries)
        if add_one:
            a = 1.0
        else:
            a = 0.0
        entries = numpy.log10(entries + a)
        M = entries.max()
        while hasattr(M, "__len__"):
            M = max(M)
        M = int(M + 1)
    if M is None:
        yticks = ax.yaxis.get_majorticklocs()
        M = max(yticks)
        M = int(M + 0.9999)
    if add_zero:
        a = [0]
    else:
        a = []
    print(M, list(range(1, M + 1)), a)
    ax.set_yticks(a + list(range(0, M + 1)))
    y_range = (0, M)
    xlab = a
    for i in range(M + 1):
        xlab += [10 ** (i)]
    ax.set_yticklabels(
        xlab,
        rotation="horizontal",
        verticalalignment="center",
        horizontalalignment="right",
        fontsize=text_sizes["xlabels"],
    )
    print(xlab)
    return ax, entries, y_range


# def apply_function_to_errors(data, error_bars, function):
#    return new_bars

"""
Plot non-overlapping labels
http://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations
"""


class OverlappingRectangles:  # improve: init with axis boundaries!
    """
    a rectangle is obtained from most objects by applying .get_window_extent() After the thing has been drawn!!
    """

    def __init__(self):
        self.rectangles = []

    def __call__(self, new_rectangle):
        for rec in self.rectangles:
            if self.overlap(new_rectangle, rec):
                return False, rec
        self.rectangles += [new_rectangle]  # add new one to existing ones..
        return True, None

    def range_overlap(self, a_min, a_max, b_min, b_max):
        """Neither range is completely greater than the other"""
        return (a_min <= b_max) and (b_min <= a_max)

    def overlap(self, r1, r2):
        """Overlapping rectangles overlap both horizontally & vertically"""
        return self.range_overlap(r1.x0, r1.x1, r2.x0, r2.x1) and self.range_overlap(
            r1.y0, r1.y1, r2.y0, r2.y1
        )


class ToyRec:
    def __init__(self, x0=None, x1=None, y0=None, y1=None):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1


def add_value_labels(
    labels,
    x_data,
    y_data,
    fontsize=None,
    ax=None,
    extra_points=5,
    initial_off_x=10,
    initial_off_y=10,
    max_attempts=10,
    textcolor="black",
    initial_offset_ha_va=None,
    alternate=False,
    offset_from_linfit=True,
    avoid_hitting_points=True,
    xerr=None,
    yerr=None,
    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.5),
    arrowprops=dict(arrowstyle="-", lw=None, connectionstyle=None),
):
    """
    NB> handle_value_labels should be used instead when possible
      arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10"
      see also http://matplotlib.org/examples/pylab_examples/annotation_demo2.html
    """
    if ax is None:
        ax = plt.gca()
    if fontsize is None:
        fontsize = text_sizes["value_labels"]
    # in display units, get bottom left and top right corners coordinates of the axis
    bottom_left, top_right = ax.transData.transform(
        list(zip(*[ax.get_xlim(), ax.get_ylim()]))
    )
    ax_width, ax_height = top_right - bottom_left

    # init class of overlaps
    OV = OverlappingRectangles()
    if avoid_hitting_points:
        for j, x in enumerate(x_data):
            r = ToyRec()
            y = y_data[j]
            if xerr is not None:
                r.x0, r.x1 = ax.transData.transform((x - xerr[j], x + xerr[j]))
            else:
                r.x0, r.x1 = ax.transData.transform((x, x))
                r.x0 -= extra_points
                r.x1 += extra_points
            if yerr is not None:
                r.y0, r.y1 = ax.transData.transform((y - yerr[j], y + yerr[j]))
            else:
                r.y0, r.y1 = ax.transData.transform((y, y))
                r.y0 -= extra_points
                r.y1 += extra_points
            OV.rectangles += [r]

    if initial_offset_ha_va is None:
        if offset_from_linfit:
            initial_offset_ha_va = []
            _, pol_fun, _ = polfit(x_data, y_data)
            for j, x in enumerate(x_data):
                yf = pol_fun(x)
                ##
                print(labels[j], yf, y_data[j])
                if yf > y_data[j]:
                    initial_offset_ha_va += [
                        ([-initial_off_x, initial_off_y], "right", "bottom")
                    ]
                else:
                    initial_offset_ha_va += [
                        ([initial_off_x, -initial_off_y], "left", "bottom")
                    ]
        elif alternate:
            initial_offset_ha_va = [
                ([-initial_off_x, initial_off_y], "right", "bottom"),
                ([initial_off_x, -initial_off_y], "left", "bottom"),
            ] * (len(x_data) // 2 + 1)
        else:
            initial_offset_ha_va = [
                ([-initial_off_x, initial_off_y], "right", "bottom")
            ] * len(x_data)

    # sort data in a convenient way
    points = list(zip(x_data, y_data, labels))
    points.sort(key=lambda x: x[0])  # first by x
    points.sort(key=lambda x: x[1])  # then by y
    xytext = None

    for j, p in enumerate(points):
        lab = p[2]
        p = (p[0], p[1])
        xytext, ha, va = initial_offset_ha_va[j]
        attempt = 0
        while True:
            obj = plt.annotate(
                lab,
                xy=p,
                xytext=xytext,
                xycoords="data",
                size=fontsize,
                color=textcolor,
                textcoords="offset points",
                ha=ha,
                va=va,
                bbox=bbox,
                arrowprops=arrowprops,
            )
            plt.draw()
            rect = obj.get_window_extent()
            ok, conflict = OV(rect)
            last_dir = None
            if not ok:
                go_right = 0
                if (
                    conflict.x0 < rect.x0 < conflict.x1 and last_dir != "l"
                ):  # should move to the right (we assume all rect are more or less same size)
                    go_right = conflict.x1 - rect.x0 + extra_points
                if (
                    conflict.x0 < rect.x1 < conflict.x1 and last_dir != "r"
                ):  # should move to the left
                    tmp = (
                        conflict.x0 - rect.x1 - extra_points
                    )  # save a negative quanitity, so that we actually go left
                    if abs(tmp) < go_right or go_right == 0:
                        go_right = tmp
                go_up = 0
                if (
                    conflict.y0 < rect.y0 < conflict.y1 and last_dir != "d"
                ):  # we should only go up because of the way we have sorted the points
                    go_up = conflict.y1 - rect.y0 + extra_points
                elif (
                    conflict.y0 < rect.y1 < conflict.y1 and last_dir != "u"
                ):  # but just in case...
                    tmp = (
                        conflict.y0 - rect.y1 - extra_points
                    )  # save a negative quanitity, so that we actually go down
                    if abs(tmp) < go_up or go_up == 0:
                        go_up = tmp
                if go_right == 0:  # we go up/down
                    xytext = [xytext[0], xytext[1] + go_up]
                    if go_up < 0:
                        last_dir = "d"
                    else:
                        last_dir = "u"
                elif go_up == 0:  # we go right/left!
                    xytext = [xytext[0] + go_right, xytext[1]]
                    if go_right < 0:
                        last_dir = "l"
                    else:
                        last_dir = "r"
                elif abs(go_right) < abs(go_up):  # we go right/left!
                    xytext = [xytext[0] + go_right, xytext[1]]
                    if go_right < 0:
                        last_dir = "l"
                    else:
                        last_dir = "r"
                else:  # we go up/down
                    xytext = [xytext[0], xytext[1] + go_up]
                    if go_up < 0:
                        last_dir = "d"
                    else:
                        last_dir = "u"
                attempt += 1
                if attempt >= max_attempts:
                    ##
                    print("aborting ", lab)
                    OV.rectangles += [rect]
                    break  # leave the damn things overlap!!
                obj.remove()
                del rect
            else:
                break  # exit why loop.
        del rect
        del obj

    return


def font_size_to_figure_fraction(
    fontsize, figuresize, string=None
):  # this is very rough!!
    """
    returns the size of one character at that fontsize in axis fraction of the figure size (very approximate)
    at fontsize 50 in figure size 10*10 the figure comes as 25.4 cm.
    10 letters are 11.94 cm width and 1 capital letter is 1.2 cm tall

    PostScript (current DTP ?) 0.138889 is 10 pt
    Didot System 0.148008 is 10p
    Pica System 0.138349 is 10 pt
    """
    fig_frac10_50W = 1.194 / 25.4
    fig_frac10_50H = 1.2 / 25.4
    current_frac50W = figuresize[0] * fig_frac10_50W / 10.0
    current_frac50H = figuresize[1] * fig_frac10_50H / 10.0
    width = fontsize * current_frac50W / 50.0
    height = fontsize * current_frac50H / 50.0
    if string is not None:
        return width * len(string), height
    return width, height


def get_text_positions(x_data, y_data, txt_width, txt_height):
    """
    #set the bbox for the text. Increase txt_width for wider text.
    txt_height = 0.04*(plt.ylim()[1] - plt.ylim()[0])
    txt_width = 0.02*(plt.xlim()[1] - plt.xlim()[0])
    """

    if type(txt_width) is not list:
        txt_width = [txt_width] * len(x_data)
    if type(txt_height) is not list:
        txt_width = [txt_height] * len(x_data)
    x_data = numpy.array(x_data, "f")
    y_data = numpy.array(y_data, "f")
    a = list(zip(y_data, x_data))

    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [
            i
            for i in a
            if i[0] > (y - txt_height[index])
            and (abs(i[1] - x) < txt_width[index] * 2)
            and i != (y, x)
        ]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height[index]:  # True == collision
                differ = numpy.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height[index], a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height[index]
                for k, (j, m) in enumerate(differ):
                    # j is the vertical distance between words
                    if j > txt_height[index] * 2:  # if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height[index], a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height[index]
                        break
    return text_positions


def text_plotter(
    x_data,
    y_data,
    text_positions,
    axis,
    txt_width,
    txt_height,
    text,
    fontsize=None,
    rotation=0,
    always_arrow=True,
    bbox=True,
    textcolor="red",
    arrowcolor="black",
):
    """
    text must be a list with the actual text one wants to write
    can give (e.g. text=numpy.round(y_data,decimals=2) to plot the y_data)
    text_positions is as returned by the get_text_positions()
    """
    if fontsize is None:
        fontsize = text_sizes["value_labels"]
    for j, (x, y, t) in enumerate(zip(x_data, y_data, text_positions)):
        if bbox:
            axis.text(
                x - txt_width[j],
                1.01 * t,
                str(text[j]),
                rotation=rotation,
                color=textcolor,
                fontsize=fontsize,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5),
            )
        else:
            axis.text(
                x - txt_width[j],
                1.01 * t,
                str(text[j]),
                rotation=rotation,
                color=textcolor,
                fontsize=fontsize,
            )

        if y != t or always_arrow:
            axis.arrow(
                x,
                t,
                0,
                y - t,
                color=arrowcolor,
                alpha=1,
                zorder=0,
                length_includes_head=True,
            )


def add_custom_valuelabels(
    labels, x_data, y_data, fontsize=None, figuresize=None, axis=None
):
    if axis is None:
        axis = plt.gca()
    if figuresize is None:
        figure = axis.get_figure()
        figuresize = figure.get_size_inches()
    if fontsize is None:
        fontsize = text_sizes["value_labels"]
    txt_width = []
    txt_height = []
    for lab in labels:
        w, h = font_size_to_figure_fraction(fontsize, figuresize, string=lab)
        txt_width += [w * (axis.get_xlim()[1] - axis.get_xlim()[0])]
        txt_height += [h * (axis.get_ylim()[1] - axis.get_ylim()[0])]
    print(txt_height)
    print(txt_width)
    text_positions = get_text_positions(x_data, y_data, txt_width, txt_height)
    text_plotter(
        x_data,
        y_data,
        text_positions,
        axis,
        txt_width,
        txt_height,
        labels,
        fontsize=fontsize,
        textcolor="red",
        arrowcolor="black",
    )
    plt.draw()
    return


def add_custom_flaglabels(
    labels,
    x_pos=None,
    y_pos=None,
    figure=None,
    axis=None,
    fontsize=None,
    facecolor="white",
    ha="right",
    va="bottom",
    labels_offset=(-10, 10),
):
    if figure is None:
        figure = plt.gcf()
    if axis is None:
        axis = figure.gca()
    if fontsize is None:
        if len(labels) < 15:
            fontsize = text_sizes["value_labels"]
        else:
            fontsize = text_sizes["xlabels_many"]
    if x_pos is None and y_pos is None:
        if (
            type(labels) is dict
        ):  # assumes keys are tuples of x_pos and y_pos, while values are labels
            x_pos, y_pos = list(zip(*list(labels.keys())))
            labels = list(labels.values())
        else:  # assumes labels is a list of tuples, each tuple with (xpos,ypos,label)
            x_pos, y_pos, labels = list(zip(*labels))
    # Beta this assumes labels is dict
    elif y_pos is None:  # assumes x_pos is given and keys are y_pos
        y_pos = list(map(float, list(labels.keys())))
        labels = list(labels.values())
    elif x_pos is None:  # assumes y_pos is given and keys are x_pos
        x_pos = list(map(float, list(labels.keys())))
        labels = list(labels.values())

    if type(facecolor) is not list and not isinstance(facecolor, cycle_list):
        facecolor = [facecolor] * len(labels)
    if type(labels_offset) is not list:
        labels_offset = [labels_offset] * len(labels)
    jc = 0
    for labl, X, Y in zip(labels, x_pos, y_pos):
        kl = plt.annotate(
            labl,
            xy=(X, Y),
            xytext=labels_offset[jc],
            fontsize=fontsize,
            xycoords="data",
            textcoords="offset points",
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.2", fc=facecolor[jc]),
            arrowprops=dict(arrowstyle="-", connectionstyle=None),
        )
        kl.draggable()
        jc += 1
    return


def handle_grid(
    ax,
    vgrid,
    hgrid,
    vcolor=None,
    hcolor=None,
    v_ls=None,
    h_ls=None,
    v_lw=None,
    h_lw=None,
    zorder=None,
):
    """
    if it is int or float it will assume that you want the grid every x units in the x/y lims
    to give specific value give as a list
    """
    if hgrid is None:
        hgrid = default_parameters["hgrid"]
    if vgrid is None:
        vgrid = default_parameters["vgrid"]
    if vcolor is None:
        vcolor = grid_parameters["vcolor"]
    if hcolor is None:
        hcolor = grid_parameters["hcolor"]
    if v_ls is None:
        v_ls = grid_parameters["v_ls"]
    if h_ls is None:
        h_ls = grid_parameters["h_ls"]
    if v_lw is None:
        v_lw = grid_parameters["v_lw"]
    if h_lw is None:
        h_lw = grid_parameters["h_lw"]

    if hgrid:
        if type(hgrid) is bool:
            yticks = ax.yaxis.get_majorticklocs()
            for yt in yticks:
                plt.axhline(yt, color=hcolor, ls=h_ls, lw=h_lw, zorder=zorder)
        elif type(hgrid) is int or type(hgrid) is float:
            for hl in numpy.arange(ax.get_ylim()[0], ax.get_ylim()[1], hgrid):
                ax.axhline(hl, color=hcolor, ls=h_ls, lw=h_lw, zorder=zorder)
        elif type(hgrid) is list or type(hgrid) is tuple:
            for hl in hgrid:
                ax.axhline(hl, color=hcolor, ls=h_ls, lw=h_lw, zorder=zorder)
    if vgrid:
        if type(vgrid) is bool:
            xticks = ax.xaxis.get_majorticklocs()
            for xt in xticks:
                ax.axvline(xt, color=vcolor, ls=v_ls, lw=v_lw, zorder=zorder)
        elif type(vgrid) is int or type(vgrid) is float:
            for vl in numpy.arange(ax.get_xlim()[0], ax.get_xlim()[1], vgrid):
                ax.axvline(vl, color=vcolor, ls=v_ls, lw=v_lw, zorder=zorder)
        elif type(vgrid) is list or type(vgrid) is tuple:
            for vl in vgrid:
                ax.axvline(vl, color=vcolor, ls=v_ls, lw=v_lw, zorder=zorder)
    return ax


def handle_value_labels(
    value_labels,
    x_values,
    profile,
    yerr=None,
    ismulti=False,
    axis=None,
    value_labels_ypos="top",
    zorder=0,
    default_text_offset=None,
    value_labels_rotation="horizontal",
    label_size=None,
    increase_star_size=4,
):
    """
    one can give value_labels_ypos= 'scatter' to put labels in flags that point to the points
    increase_star_size is for p value ***
    """
    if axis is None:
        axis = plt
    if default_text_offset is None:
        default_text_offset = default_parameters["value_label_text_offset"]
    if label_size is None:
        label_size = text_sizes["value_labels"]
    scat = False
    if not ismulti:
        profile = [profile]
        value_labels = [value_labels]
        yerr = [yerr]
    family = None
    ha = "center"
    if "top" in value_labels_ypos:
        va = "bottom"
    elif "bottom" in value_labels_ypos:
        va = "top"
    else:
        va = "center"
    if value_labels_rotation in ["vertical", 90, 270]:
        ha = "center"

        # va='center'
    if "right" in value_labels_ypos:
        ha = "left"
    elif "left" in value_labels_ypos:
        ha = "right"
    # print 'handle_value_labels ha,va',ha,va,ismulti
    for j, prof in enumerate(profile):
        if value_labels[j] is None:
            continue
        for k, val in enumerate(prof):
            yend = val
            if yerr[j] is not None:
                if "bottom" in value_labels_ypos or "top" in value_labels_ypos:
                    if len(yerr[j]) == 2 and hasattr(yerr[j][0], "__len__"):  # like CI
                        if "bottom" in value_labels_ypos:
                            yend -= yerr[j][0][k]
                        else:
                            yend += yerr[j][1][k]
                    else:
                        if "bottom" in value_labels_ypos:
                            yend -= yerr[j][k]
                        else:
                            yend += yerr[j][k]
            if type(value_labels_ypos) is tuple:
                xytext = value_labels_ypos
            elif type(value_labels_ypos) is list:
                xytext = value_labels_ypos[k]
            elif type(value_labels_ypos) is str:
                if "scatter" in value_labels_ypos:
                    scat = True
                    xytext = (-10, 10)
                if "top" in value_labels_ypos:
                    xytext = (0, default_text_offset)
                elif "bottom" in value_labels_ypos:
                    xytext = (0, -default_text_offset)
                elif "right" in value_labels_ypos:
                    xytext = (default_text_offset, 0)
                elif "left" in value_labels_ypos:
                    xytext = (0, default_text_offset)
                elif "mid" in value_labels_ypos or "cent" in value_labels_ypos:
                    xytext = (0, 0)
            if type(value_labels[j]) is bool and value_labels[j] == True:
                vlab = repr(val)
            else:
                vlab = value_labels[j][k]
            if hasattr(x_values[0], "__len__"):
                xvlab = x_values[j][
                    k
                ]  # ismulti and different x_values for different profiles
            else:
                xvlab = x_values[k]
            if (
                type(vlab) is str
                and increase_star_size != False
                and (vlab == "*" * len(vlab))
            ):
                lsize = label_size + int(increase_star_size)
                family = "monospace"
            else:
                lsize = label_size
                family = None
            # print label_size,lsize
            if scat:
                kl = axis.annotate(
                    vlab,
                    xy=(xvlab, yend),
                    xytext=xytext,
                    size=lsize,
                    xycoords="data",
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    rotation=value_labels_rotation,
                    zorder=zorder + 1,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
                    family=family,
                    arrowprops=dict(arrowstyle="-", connectionstyle=None),
                )
                kl.draggable()
            else:

                kl = axis.annotate(
                    vlab,
                    xy=(xvlab, yend),
                    xytext=xytext,
                    xycoords="data",
                    size=lsize,
                    textcoords="offset points",
                    ha=ha,
                    va=va,
                    rotation=value_labels_rotation,
                    zorder=zorder
                    + 1  # , bbox = dict(boxstyle = 'round,pad=0.3', fc = 'white', alpha = 1.) \
                    # , arrowprops = dict(arrowstyle = '-',connectionstyle=connectionstyle) \
                )
                kl.draggable()
    return


def handle_ticks(
    ax,
    x_major_tick_every=None,
    y_major_tick_every=None,
    x_minor_tick_every=None,
    y_minor_tick_every=None,
    minor_ticks_between_two_major=1,
    cbar_in_figure=False,
    log_scale_x=False,
    new_figure=True,
    log_scale_y=False,
    xlabels=None,
    ylabels=None,
    entries_xpos=None,
    entries_ypos=None,
    use_right_axis=False,
    xlabels_rotation="horizontal",
    ylabels_rotation="horizontal",
):
    """
    x_major_tick_every and y_major_tick_every can be given as string integer (e.g. '3') if one wants to specify the total number of ticks rather than their spacing
    """
    # print('DEB x_major_tick_every', x_major_tick_every, 'entries_xpos', entries_xpos,hasattr(xlabels,'__len__'),xlabels )
    if hasattr(xlabels, "__len__"):
        xlabels = numpy.array(xlabels)
        if xlabels.dtype.kind == "i" or xlabels.dtype.kind == "f":
            # NB even if you turn these into int they will be represented as float..
            # print 'DEB******',x_major_tick_every,xlabels.dtype.kind,xlabels
            make_int = True
            if not hasattr(xlabels[0], "__len__"):
                for x in xlabels:
                    if int(x) != x:
                        make_int = False
                        break
                if make_int:
                    xlabels = list(
                        map(int, xlabels)
                    )  # will be set as ticks in this way
            if entries_xpos is None:
                x_major_tick_every = xlabels  # will be set as ticks in this way
        if hasattr(x_major_tick_every, "__len__") and len(xlabels) == len(
            x_major_tick_every
        ):  # and entries_xpos is None :
            entries_xpos = x_major_tick_every
            # print("DEB2", entries_xpos)
        elif (
            type(x_major_tick_every) is not bool
            and not hasattr(x_major_tick_every, "__len__")
            and x_major_tick_every is not None
            and x_major_tick_every > 1
        ):
            xlabels = xlabels[
                ::x_major_tick_every
            ]  # when x_major_tick_every is None it does nothing
        elif x_major_tick_every is None and entries_xpos is None:
            ticks = ax.get_xticks()
            # print 'xlabels',xlabels
            if len(ticks) == len(xlabels):
                x_major_tick_every = ticks  # will be set as ticks in this way

    if hasattr(ylabels, "__len__"):
        ylabels = numpy.array(ylabels)
        if ylabels.dtype.kind == "i" or ylabels.dtype.kind == "f":
            # NB even if you turn these into int they will be represented as float..
            # print 'DEB******',y_major_tick_every,ylabels.dtype.kind,ylabels
            make_int = True
            for y in ylabels:
                if int(y) != y:
                    make_int = False
                    break
            if make_int:
                ylabels = list(map(int, ylabels))  # will be set as ticks in this way
            if entries_ypos is None:
                y_major_tick_every = ylabels  # will be set as ticks in this way
        elif (
            type(y_major_tick_every) is not bool
            and not hasattr(y_major_tick_every, "__len__")
            and y_major_tick_every is not None
        ):
            ylabels = ylabels[
                ::y_major_tick_every
            ]  # when y_major_tick_every is None it does nothing
        elif hasattr(y_major_tick_every, "__len__") and len(ylabels) == len(
            y_major_tick_every
        ):  # and entries_ypos is None :
            entries_ypos = y_major_tick_every
        elif y_major_tick_every is None and entries_ypos is None:
            ticks = ax.get_yticks()
            if len(ticks) == len(ylabels):
                y_major_tick_every = ticks  # will be set as ticks in this way

    # SET PUBLICATION QUALITY if no custom option is provided
    if default_parameters[
        "set_publish"
    ]:  # we use this to know whether we did set publish or not.
        # print xlabels,x_major_tick_every,ax.get_xticks()
        if new_figure:
            if not cbar_in_figure:
                ax.figure.set_tight_layout(True)
        if log_scale_x:
            if type(log_scale_x) is int and log_scale_x > 0:
                logbase = log_scale_x
            else:
                logbase = 10
            ax.set_xscale(
                "log", base=logbase, subs=[2, 4, 6, 8], nonpositive="clip"
            )  # clip non positive values, previously nonposx and nonposy
            # set a minor tick every order of magnitute (if span less than 100 order of magnitudes, otherwise 100 ticks)
            # minorLocator = matplotlib.ticker.LogLocator(base=logbase)#,numticks=100)
            # set a maximum of 6 major ticks
            # ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=logbase, subs=[1.0], numticks=5)) #, numdecs=4
            # ax.xaxis.set_minor_locator(minorLocator)
            # labels = [item.get_text() for item in ax.get_xticklabels()]
            # every=int(0.5+len(labels)/6.)
            # if labels[0]==u'0.00000' : labels = [labels[0]]+[l if j%every==0 else '' for j,l in enumerate(labels[1:])]
            # else : labels = [l if j%every==0 else '' for j,l in enumerate(labels)]
            # ax.set_xticklabels(labels)
            # Turn off x-axis minor ticks
            ax.xaxis.set_tick_params(which="minor", bottom=False)
        # set major
        elif (
            not hasattr(x_major_tick_every, "__len__")
            and x_major_tick_every is None
            and x_major_tick_every != False
        ):
            ax.xaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(nbins=default_parameters["ticks_nbins"])
            )

        if log_scale_y:
            if type(log_scale_y) is int and log_scale_y > 0:
                logbase = log_scale_y
            else:
                logbase = 10
            ax.set_yscale("log", base=logbase, subs=[2, 4, 6, 8], nonpositive="clip")
            # minorLocator = matplotlib.ticker.LogLocator(base=logbase) #,numticks=20)
            # ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=logbase, numticks=5)) #, numdecs=4
            # ax.yaxis.set_minor_locator(minorLocator)
            # Turn off y-axis minor ticks
            ax.yaxis.set_tick_params(which="minor", bottom=False)
        elif (
            not hasattr(y_major_tick_every, "__len__")
            and y_major_tick_every != False
            and y_major_tick_every is None
        ):
            ax.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(nbins=default_parameters["ticks_nbins"])
            )
        # set MINOR PUBLISH DEFAULT
        if (
            not log_scale_x
            and not hasattr(x_minor_tick_every, "__len__")
            and x_minor_tick_every != False
            and x_minor_tick_every is None
        ):
            ax.xaxis.set_minor_locator(
                matplotlib.ticker.AutoMinorLocator(n=minor_ticks_between_two_major + 1)
            )
        if (
            not log_scale_y
            and not hasattr(y_minor_tick_every, "__len__")
            and y_minor_tick_every != False
            and y_minor_tick_every is None
        ):
            ax.yaxis.set_minor_locator(
                matplotlib.ticker.AutoMinorLocator(n=minor_ticks_between_two_major + 1)
            )

        # print xlabels,x_major_tick_every, ax.get_xticks()
    else:
        if log_scale_x:
            if type(log_scale_x) is int and log_scale_x > 0:
                logbase = log_scale_x
            else:
                logbase = 10
            # subs = Where to place the subticks between each major tick. For example, in a log10 scale, [2, 3, 4, 5, 6, 7, 8, 9] will place 8 logarithmically spaced minor ticks between each major tick.
            ax.set_xscale(
                "log", base=logbase, subs=[2, 3, 4, 5, 6, 7, 8, 9], nonpositive="clip"
            )  # nonpositive previously nonposx and nonposy
        if log_scale_y:
            if type(log_scale_y) is int and log_scale_y > 0:
                logbase = log_scale_y
            else:
                logbase = 10
            ax.set_yscale(
                "log", base=logbase, subs=[2, 3, 4, 5, 6, 7, 8, 9], nonpositive="clip"
            )

    # Y MAJOR
    if type(y_major_tick_every) is str:
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=int(y_major_tick_every))
        )
    elif hasattr(y_major_tick_every, "__len__"):
        ax.yaxis.set_ticks(y_major_tick_every)
    elif y_major_tick_every != False and y_major_tick_every is not None:
        ymajorLocator = matplotlib.ticker.MultipleLocator(y_major_tick_every)
        ax.yaxis.set_major_locator(ymajorLocator)
    elif type(y_major_tick_every) is bool and y_major_tick_every == False:
        ax.yaxis.set_ticks([])
    # X MAJOR
    # print xlabels,x_major_tick_every, ax.get_xticks(),map(str,ax.get_xticklabels())
    if type(x_major_tick_every) is str:
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=int(x_major_tick_every))
        )
    elif hasattr(x_major_tick_every, "__len__"):
        ax.xaxis.set_ticks(x_major_tick_every)
    elif x_major_tick_every != False and x_major_tick_every is not None:
        xmajorLocator = matplotlib.ticker.MultipleLocator(x_major_tick_every)
        ax.xaxis.set_major_locator(xmajorLocator)
    elif type(x_major_tick_every) is bool and x_major_tick_every == False:
        ax.xaxis.set_ticks([])
    # print xlabels,x_major_tick_every, ax.get_xticks(),map(str,ax.get_xticklabels())
    # Y MINOR
    if hasattr(y_minor_tick_every, "__len__"):
        ax.yaxis.set_ticks(y_minor_tick_every, minor=True)
    elif y_minor_tick_every != False and y_minor_tick_every is not None:
        yminorLocator = matplotlib.ticker.MultipleLocator(y_minor_tick_every)
        # for the minor ticks, use no labels; default NullFormatter
        ax.yaxis.set_minor_locator(yminorLocator)
    elif type(y_minor_tick_every) is bool and y_minor_tick_every == False:
        ax.yaxis.set_ticks([], minor=True)
    # X MINOR
    if hasattr(x_minor_tick_every, "__len__"):
        ax.xaxis.set_ticks(x_minor_tick_every, minor=True)
    elif x_minor_tick_every != False and x_minor_tick_every is not None:
        xminorLocator = matplotlib.ticker.MultipleLocator(x_minor_tick_every)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(xminorLocator)
    elif type(x_minor_tick_every) is bool and x_minor_tick_every == False:
        ax.xaxis.set_ticks([], minor=True)
    # XLABELS
    if type(xlabels) is bool and xlabels == False:
        ax.set_xticklabels([])
    elif xlabels is not None:
        if entries_xpos is not None and hasattr(xlabels, "__len__"):
            # print 'entries_xpos',entries_xpos,'xlabels',xlabels
            if len(entries_xpos) == len(xlabels) and not hasattr(
                entries_xpos[0], "__len__"
            ):
                ax.set_xticks([], minor=True)
                # print 'entries_xpos:',entries_xpos
                ax.set_xticks(
                    entries_xpos
                )  # all major ticks restored, one for each entry
            elif hasattr(entries_xpos[0], "__len__") and len(entries_xpos[0]) == len(
                xlabels
            ):  # could be multi profile
                ax.set_xticks([], minor=True)
                # print('DEB 3 entries_xpos',entries_xpos)
                ax.set_xticks(
                    numpy.mean(numpy.array(entries_xpos), axis=0)
                )  # all major ticks restored, one for each entry
            elif hasattr(entries_xpos[0], "__len__") and len(
                misc.uniq(misc.flatten(entries_xpos))
            ) == len(
                xlabels
            ):  # multi profiles but with same xpos
                ax.set_xticks([], minor=True)
                ax.set_xticks(
                    misc.uniq(misc.flatten(entries_xpos))
                )  # all major ticks restored, one for each entry
            else:
                sys.stderr.write(
                    "WARNING in plotter.handle_ticks len(entries_xpos)!=len(xlabels) %d %d not setting entries_xpos\n   entries_xpos=%s\n   xlabels=%s\n"
                    % (len(entries_xpos), len(xlabels), str(entries_xpos), str(xlabels))
                )
        horizontalalignment = "center"
        verticalalignment = "top"
        if type(xlabels_rotation) is int:
            if 15 < xlabels_rotation < 90:
                horizontalalignment = "right"
                # verticalalignment='baseline'
            elif xlabels_rotation < -15:
                horizontalalignment = "left"
                # verticalalignment='baseline'
        ticks = ax.get_xticks(minor=False)
        if len(ticks) != len(xlabels):
            sys.stderr.write(
                "*Warn* in plotter.handle_ticks() xlabels given as input but len(ticks)!=len(xlabels) [%d,%d] or issues in types:\n"
                % (len(ticks), len(xlabels))
            )
            if len(xlabels) == len(ticks) - 2:  # tmp fix
                xlabels = [""] + list(xlabels) + [""]
        ax.set_xticklabels(
            xlabels,
            rotation=xlabels_rotation,
            verticalalignment=verticalalignment,
            fontsize=text_sizes["xlabels"],
            horizontalalignment=horizontalalignment,
        )
    # print xlabels
    # YLABELS
    if type(ylabels) is bool and ylabels == False:
        ax.set_yticklabels([])
    elif ylabels is not None:
        if entries_ypos is not None and hasattr(ylabels, "__len__"):
            if len(entries_ypos) == len(ylabels):
                ax.set_yticks([], minor=True)
                # print 'entries_ypos:',entries_ypos
                ax.set_yticks(
                    entries_ypos
                )  # all major ticks restored, one for each entry
            else:
                sys.stderr.write(
                    "WARNING in plotter.handle_ticks len(entries_ypos)!=len(ylabels) %d %d not setting entries_ypos\n"
                    % (len(entries_ypos), len(ylabels))
                )
        if use_right_axis:
            horizontalalignment = "left"
        else:
            horizontalalignment = "right"
        verticalalignment = "center"
        if type(ylabels_rotation) is int:
            if 15 < ylabels_rotation < 90:
                verticalalignment = "top"
            elif ylabels_rotation < -15:
                verticalalignment = "bottom"
        ticks = ax.get_yticks(minor=False)
        print("DEB:", ticks)
        if len(ticks) != len(ylabels):
            sys.stderr.write(
                "*Warn* in plotter.handle_ticks() ylabels given as input but len(ticks)!=len(ylabels) [%d,%d]\n"
                % (len(ticks), len(ylabels))
            )
        ax.set_yticklabels(
            ylabels,
            rotation=ylabels_rotation,
            verticalalignment=verticalalignment,
            fontsize=text_sizes["xlabels"],
            horizontalalignment=horizontalalignment,
        )
    # print ylabels
    return ax


def process_smoothk_to_fun(smooth):
    yfunct = None
    if type(smooth) is tuple or type(smooth) is list:
        if len(smooth) == 2:
            win, pol = smooth
            mod = "interp"
        elif len(smooth) == 3:
            win, pol, mod = smooth
        else:
            win, pol, mod = smooth[0], 5, "interp"
        yfunct = lambda x: misc.smooth_profile(
            x, use_savgol_filter=(win, pol), mode=mod, interpolate_nan=True
        )  # scipy.signal.savgol_filter(x,window_length=win, polyorder=pol, mode=mod)
    elif type(smooth) is int:
        yfunct = lambda x: misc.smooth_profile(x, smooth_per_side=smooth)
    elif hasattr(smooth, "__call__"):
        yfunct = lambda x: smooth(x)
    else:
        yfunct = lambda x: misc.smooth_profile(
            x, smooth_per_side=5
        )  # assumes it is a smoothing # default window of 5
    return yfunct


"""

GENERATION OF (almost) PUBLICATION QUALITY FIGURES

"""


def histogram(
    entries,
    cumulative=False,
    nbins=None,
    normalized=True,
    stacked=False,
    histtype="stepfilled",
    orientation="vertical",
    weights=None,
    label="",
    intcount=False,
    plot_legend=True,
    color=None,
    hatch=None,
    alpha=None,
    edgecolor=None,
    linewidth=None,
    xlabels=None,
    flag_labels=None,
    flag_labels_color=(1, 1, 1, 0.6),
    xlabels_rotation="vertical",
    xlabel=None,
    ylabel=None,
    title=None,
    x_range=None,
    y_range=None,
    hline=None,
    vline=None,
    vline_color="black",
    vline_style="--",
    figure=None,
    frame=None,
    hgrid=None,
    log_scale=False,
    save=False,
    figure_size=None,
    legend_size=None,
    legend_location="upper right",
    show=True,
    block=False,
    only_cumulative=False,
    x_major_tick_every=None,
    y_major_tick_every=None,
    x_minor_tick_every=None,
    y_minor_tick_every=None,
    fit_gaussian=False,
):
    """
    draw an histogram
    flag_labels if not None must be a dictionary whose keys are xvalues and values are str labels,
      these are drawn on top of the histogram and look like flags...
     histtype : ['bar' | 'barstacked' | 'step' | 'stepfilled']
    """
    if frame is None:
        frame = default_parameters["frame"]
    if hgrid is None:
        hgrid = default_parameters["hgrid"]
    if stacked and histtype == "bar":
        histtype = "barstacked"
    if only_cumulative:
        cumulative = True

    if linewidth is None:
        linewidth = default_parameters["linewidth"]
    if figure_size is None:
        if default_figure_sizes["default"] is None:
            figure_size = default_figure_sizes["histogram"]
        else:
            figure_size = default_figure_sizes["default"]
    if default_figure_sizes["use_cm"]:
        figure_size = (cmToinch(figure_size[0]), cmToinch(figure_size[1]))
    if legend_size is None:
        legend_size = text_sizes["legend_size"]
    # histtype can be bar, step barstacked stepfilled
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    plt.rc("xtick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick", labelsize=text_sizes["xlabels"])
    if type(entries) is dict:
        label = list(entries.keys())
        entries = list(entries.values())

    m, M = get_min_max_glob(entries)
    if intcount and nbins is None:
        nbins = numpy.arange(m - 0.5, M + 1.5, 1)
    elif nbins is None:
        if hasattr(entries[0], "__len__"):
            nbins = 2 * int(
                numpy.sqrt(1.0 * len(entries[0]))
            )  # use numpy.sqrt rule (like excel)
        elif len(entries) > 25:
            nbins = 2 * int(
                numpy.sqrt(1.0 * len(entries))
            )  # use numpy.sqrt rule (like excel)
        else:
            nbins = int(len(entries) // 3)
    if label == "":
        if hasattr(nbins, "__len__"):
            label += " N=%d nbins=%d" % (len(entries), len(nbins) - 1)
        else:
            label += " N=%d nbins=%d" % (len(entries), nbins)
    if vline is not None:
        if not hasattr(vline, "__len__") or type(vline) is str:
            vline = [vline]
    if hline is not None:
        if not hasattr(hline, "__len__") or type(hline) is str:
            hline = [hline]
    if hasattr(entries[0], "__len__"):
        if vline is not None and len(vline) == 1:
            vline *= len(
                entries
            )  # meaningful only if 'median' or 'mean' is given as vline
        if (
            color is not None
            and type(color) is not str
            and type(color) is not tuple
            and len(color) > len(entries)
        ):
            color = color[: len(entries)]

    if histtype == "step" and edgecolor is None:
        edgecolor = color

    if ylabel == "" or ylabel == True:
        if normalized:
            ylabel = "PDF"
        else:
            ylabel = "Count"
    new_figure = False
    if figure is None:
        new_figure = True
        figure = plt.figure(figsize=figure_size)

    ax = figure.gca()
    if log_scale:
        if normalized:
            sys.stderr.write(
                "**WARNING** for some reasons normalized does not work with log_scale (it would plot the wrong histogram). Setting it to False.\n   if you want the distribution normalized you can try giving the numpy.log10(entries) as an input and log_scale=False.\n"
            )
            normalized = False
        if hasattr(nbins, "__len__"):
            if intcount:
                nbins = 10 ** numpy.linspace(
                    numpy.log10(m - 0.5), numpy.log10(M + 1.5), len(nbins) - 1
                )
            else:
                nbins = 10 ** numpy.linspace(numpy.log10(m), numpy.log10(M), len(nbins))
        else:
            nbins = 10 ** numpy.linspace(numpy.log10(m), numpy.log10(M), nbins)
    if cumulative:
        if edgecolor is None:
            edg = None
        else:
            edg = edgecolor
        n_data, n_bins, n_pathces = plt.hist(
            entries,
            bins=nbins,
            range=None,
            weights=weights,
            orientation=orientation,
            color=color,
            hatch=hatch,
            stacked=stacked,
            density=int(normalized),
            histtype="step",
            align="mid",
            cumulative=True,
            figure=figure,
            label=label,
            edgecolor=edg,
            linewidth=linewidth,
            alpha=alpha,
        )
    if not only_cumulative:
        n_data, n_bins, n_pathces = plt.hist(
            entries,
            bins=nbins,
            range=None,
            weights=weights,
            orientation=orientation,
            color=color,
            hatch=hatch,
            stacked=stacked,
            density=int(normalized),
            histtype=histtype,
            align="mid",
            cumulative=False,
            figure=figure,
            label=label,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
    if fit_gaussian:
        xs = numpy.linspace(m, M, 2000)
        if hasattr(entries[0], "__len__"):
            for j, en in enumerate(entries):
                mean, std = scipy.stats.norm.fit(en)
                print(
                    "fit_gaussian distribution %d mean= %lf std = %lf" % (j, mean, std)
                )
                ys = scipy.stats.norm.pdf(xs, mean, std)
                plt.plot(xs, ys, color="red")
        else:
            # entries=numpy.array(entries)
            # lp,up=numpy.percentile(entries,[1,99])
            # mean,std = scipy.stats.norm.fit(entries[ numpy.where( (entries>lp) & (entries<up)) ])
            mean, std = scipy.stats.norm.fit(entries)
            print("fit_gaussian mean= %lf std = %lf" % (mean, std))
            ys = scipy.stats.norm.pdf(xs, mean, std)
            plt.plot(xs, ys, color="red")

    # if numpy.array(n_data).max()<0.1: ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y') # use scientific notation

    if vline is not None:
        for j, l in enumerate(vline):
            if l is None:
                continue
            if l == "median":
                if hasattr(entries[0], "__len__"):
                    l = numpy.median(entries[j])
                else:
                    l = numpy.median(entries)
                print("  histogram: distribution %d median %g" % (j, l))
            elif l == "mean":
                if hasattr(entries[0], "__len__"):
                    l = numpy.mean(entries[j])
                else:
                    l = numpy.mean(entries)
                print("  histogram: distribution %d mean %g" % (j, l))
            ax.axvline(l, color=vline_color, ls=vline_style, lw=linewidth)
    if hline is not None:
        for j, l in enumerate(hline):
            if l is None:
                continue
            if l == "median":  # in case orientation is 'horizontal'
                if hasattr(entries[0], "__len__"):
                    l = numpy.median(entries[j])
                else:
                    l = numpy.median(entries)
                print("  histogram: distribution %d median %g" % (j, l))
            elif l == "mean":
                if hasattr(entries[0], "__len__"):
                    l = numpy.mean(entries[j])
                else:
                    l = numpy.mean(entries)
                print("  histogram: distribution %d mean %g" % (j, l))
            ax.axhline(l, color=vline_color, ls=vline_style, lw=linewidth)

    if (
        x_major_tick_every is None
        and not log_scale
        and not hasattr(entries[0], "__len__")
    ):
        xticks = ax.xaxis.get_majorticklocs()
        xticks = list(map(float, xticks))
        # xticks.remove(min(xticks))
        # xticks.remove(max(xticks))
        sp = M - m
        to_remove = []
        for x in xticks:
            if abs((x - m) / sp) < 0.1:
                to_remove.append(x)
            elif abs((M - x) / sp) < 0.1:
                to_remove.append(x)
        for x in to_remove:
            xticks.remove(x)
        xticks += [m, M]
        ax.set_xticks(xticks)

    # change to log_scale if asked to, handle ticks anyway
    ax = handle_ticks(
        ax,
        x_major_tick_every,
        y_major_tick_every,
        x_minor_tick_every,
        y_minor_tick_every,
        log_scale_x=log_scale,
        new_figure=new_figure,
        xlabels=xlabels,
        xlabels_rotation=xlabels_rotation,
    )
    if not frame:
        # this remove top and right axis
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    ax.get_xaxis().tick_bottom()  # ticks only on bottom axis
    ax.get_yaxis().tick_left()  # ticks only on left axis
    if y_range is not None:
        ax.set_ylim(y_range)
    if x_range is not None:
        ax.set_xlim(x_range)
    ax, add_to_axis_label = consider_scientific_notation(
        ax,
        axis="y",
        publication=default_parameters["set_publish"],
        fontsize=text_sizes["xlabels"],
    )  # at the moment we use all_tight as an alias for publication quality y/n
    if ylabel is not None:
        ylabel += add_to_axis_label
    if not log_scale:
        ax, add_to_axis_label = consider_scientific_notation(
            ax,
            axis="x",
            publication=default_parameters["set_publish"],
            fontsize=text_sizes["xlabels"],
        )  # at the moment we use all_tight as an alias for publication quality y/n
    if xlabel is not None:
        xlabel += add_to_axis_label

    if xlabel is not None and xlabel != "":
        plt.xlabel(xlabel, fontsize=text_sizes["xlabel"], labelpad=10)
    if ylabel != "" and ylabel is not None:
        plt.ylabel(ylabel, fontsize=text_sizes["ylabel"], labelpad=10)
    if title is not None:
        plt.text(
            0.5,
            1.03,
            title,
            horizontalalignment="center",
            fontsize=24,
            transform=ax.transAxes,
        )

    if label is not None and label != "" and plot_legend:
        plt.legend(loc=legend_location, frameon=False, prop={"size": legend_size})

    ax = handle_grid(ax, vgrid=False, hgrid=hgrid)

    if flag_labels is not None:
        done_ytext = {}
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        spx = xmax - xmin
        spy = ymax - ymin
        ticks = list(flag_labels.keys())
        labels = list(flag_labels.values())
        # SHOULDNT IT BE type(n_data[0]) - no n_data is a numpy array unless stacked.. (i believe and hope)
        if (
            type(n_data) is list and len(n_data) > 1
        ):  # this should happen when the histo is stacked.. (or when there are multiple histo, which should not be the case here as one should call histogram() more than once giving the same figure).
            n_data = n_data[
                0
            ]  # n_data=sum(n_data) # obtain outer distribution function..
        if type(flag_labels_color) is not list and not isinstance(
            flag_labels_color, cycle_list
        ):
            flag_labels_color = [flag_labels_color] * len(labels)
        # print list(n_bins), list(n_data)
        for j, tick in enumerate(ticks):
            bin_value, bin_index = find_nearest(
                n_bins[:-1], tick
            )  # keep in mind that len(n_data)==len(n_bins)-1 as the last n_bins[-1] is the maximum x
            if tick < bin_value and bin_index != 0:
                bin_index -= 1
            # print tick,bin_value,bin_index,n_data[bin_index],len(n_data),len(n_bins)
            yend = 0.999 * n_data[bin_index]  # end at the top of the distribution
            connectionstyle = None
            # HERE MIGHT NEED TWEAKING
            yt = 20
            # xpc=int(100.*numpy.round(2.*tick/spx,1)/2.) # round so that you get 0.1, 0.15, 0.2, 0.25,...
            # ypc=int(100.*numpy.round(2.*yend/spy,1)/2.) # round so that you get 0.1, 0.15, 0.2, 0.25,...
            xpc = int(100.0 * numpy.round(tick / spx, 1))
            ypc = int(100.0 * numpy.round(2.0 * yend / spy, 1) / 2.0)
            if (xpc, ypc) in done_ytext:
                yt = done_ytext[(xpc, ypc)] + 30
            done_ytext[(xpc, ypc)] = yt
            xytext = (0, yt)
            kl = plt.annotate(
                labels[j],
                xy=(tick, yend),
                xytext=xytext,
                xycoords="data",
                size=text_sizes["xlabels_many"],
                textcoords="offset points",
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc=flag_labels_color[j]),
                arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle),
            )
            kl.draggable()
            # plt.vlines(tick,n_data[bin_index],yend,color='black',lw=1.)

    plt.draw()
    if save != False and save is not None:
        if ".png" in save:
            figure.savefig(
                save,
                dpi=default_figure_sizes["dpi"],
                bbox_inches="tight",
                bbox_extra_artists=[ax, figure],
                transparent=True,
            )  # ,bbox_inches="tight" is such that everything is in the figure... even though margins are removed.
        else:
            figure.savefig(
                save,
                dpi=default_figure_sizes["dpi"],
                bbox_inches="tight",
                bbox_extra_artists=[ax, figure],
            )  # ,bbox_inches="tight" is such that everything is in the figure... even though margins are removed.
    if show:
        plt.show(block=block)
    return figure


def bar(
    entries,
    label="",
    yerr=None,
    bar_width=0.8,
    bar_sep=None,
    first_left_pos=None,
    color=None,
    hatch=None,
    linewidth=None,
    xlabels=None,
    xlabel=None,
    ylabel=None,
    title=None,
    xlabels_rotation="horizontal",
    hline=None,
    y_range=None,
    x_range=None,
    ecolor="black",
    upper_labels=None,
    upper_labels_rotation="horizontal",
    start_xlabels=1,
    start_xticks_at_index=0,
    x_minor_tick_every=None,
    x_major_tick_every=None,
    y_major_tick_every=None,
    y_minor_tick_every=None,
    value_labels=None,
    value_labels_rotation="horizontal",
    print_all_labels=False,
    figure=None,
    figure_size=None,
    hgrid=None,
    vgrid=None,
    frame=None,
    save=False,
    legend_size=None,
    legend_location="upper right",
    show=True,
    block=False,
    zorder=0,
):
    """
    if xlabels is None and value_labels is None
        start_xlabels is used if custom xlabels are not provided to start the counting
        similarly start_xticks_at_index is the tick that will have the fist xlabel
    """
    if frame is None:
        frame = default_parameters["frame"]
    if hgrid is None:
        hgrid = default_parameters["hgrid"]
    if vgrid is None:
        vgrid = default_parameters["vgrid"]
    if linewidth is None:
        linewidth = default_parameters["barlinewidth"]

    if figure_size is None:
        if default_figure_sizes["default"] is None:
            figure_size = default_figure_sizes["bar"]
        else:
            figure_size = default_figure_sizes["default"]
    if default_figure_sizes["use_cm"]:
        figure_size = (cmToinch(figure_size[0]), cmToinch(figure_size[1]))
    if legend_size is None:
        legend_size = text_sizes["legend_size"]
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    plt.rc("xtick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick", labelsize=text_sizes["xlabels"])
    if type(entries) is dict:
        xlabels = list(entries.keys())
        entries = list(entries.values())
        xlabels, entries = list(zip(*sorted(zip(xlabels, entries), reverse=False)))
        if type(yerr) is dict:
            yerr = [yerr[ke] for ke in xlabels]
    elif isinstance(entries, OrderedDict):
        xlabels = list(entries.keys())
        entries = list(entries.values())
        if type(yerr) is dict or isinstance(yerr, OrderedDict):
            yerr = [yerr[ke] for ke in xlabels]
    barpos = []
    if bar_sep is None:
        bar_sep = bar_width / 4.0
    cols = []
    if first_left_pos is None:
        first_left_pos = bar_sep
    for j, x in enumerate(entries):
        if j == 0:
            left_bar_pos = [first_left_pos]
        else:
            left_bar_pos += [left_bar_pos[-1] + bar_width + bar_sep]
        if type(color) is dict and xlabels is not None and xlabels != False:
            if xlabels[j] in color:
                cols += [color[xlabels[j]]]
            else:
                print("WARNING in bar() xlabels[j]=%s not in color dict" % (xlabels[j]))
        barpos += [left_bar_pos[-1] + bar_width / 2.0]
    if cols != []:
        color = cols
    new_figure = False
    if figure is None:
        new_figure = True
        figure = plt.figure(figsize=figure_size)

    ax = figure.gca()

    #    xminorLocator   = matplotlib.ticker.MultipleLocator(x_minor_tick_every)
    #    #for the minor ticks, use no labels; default NullFormatter
    #    ax.xaxis.set_minor_locator(xminorLocator)

    if title is not None:
        plt.text(
            0.5,
            1.03,
            title,
            horizontalalignment="center",
            fontsize=text_sizes["title"],
            transform=ax.transAxes,
        )
    # if min(entries)>10000 or max(entries)<0.001: ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y') # use scientific notation
    plt.bar(
        left_bar_pos,
        entries,
        width=bar_width,
        align="edge",
        color=color,
        hatch=hatch,
        label=label,
        linewidth=linewidth,
        yerr=yerr,
        figure=figure,
        zorder=zorder,
        error_kw=dict(
            elinewidth=default_error_bars["elinewidth"],
            ecolor=ecolor,
            capsize=default_error_bars["capsize"],
            capthick=default_error_bars["capthick"],
            zorder=zorder + 1,
        ),
    )
    ax.set_xlim(min(left_bar_pos) - bar_sep, max(left_bar_pos) + bar_width + bar_sep)
    ax = handle_ticks(
        ax,
        x_major_tick_every,
        y_major_tick_every,
        x_minor_tick_every,
        y_minor_tick_every,
        new_figure=new_figure,
        xlabels=xlabels,
        xlabels_rotation=xlabels_rotation,
    )
    """
    if xlabels is not None and  xlabels!=False : 
        ax.set_xticks(barpos[::x_major_tick_every]) 
        xticks_set=barpos[::x_major_tick_every]
        xlabels=xlabels[::x_major_tick_every]# when x_major_tick_every is None it does nothing
    else : 
        if x_major_tick_every is not None : ju=x_major_tick_every
        else : ju=int(numpy.sqrt(len(entries))+1)
        xticks_set=barpos[start_xticks_at_index::ju]
        if xlabels!=False: xlabels=range(start_xlabels,start_xlabels+len(entries),ju)
        ax.set_xticks(xticks_set)
        if xlabels==False :
            ax.set_xticklabels([])
    """
    if hline is not None:
        plt.axhline(hline, color="black", ls="-", lw=plt.rcParams["axes.linewidth"])
    # ymin,ymax=ax.get_ylim()
    if not frame:
        # this remove top and right axis
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    ax.get_xaxis().tick_bottom()  # ticks only on bottom axis
    ax.get_yaxis().tick_left()  # ticks only on left axis

    if y_range is not None:
        ax.set_ylim(y_range)
    if x_range is not None:
        ax.set_xlim(x_range)

    relative_space_for_each_bar = (bar_width + bar_sep) / (
        max(left_bar_pos) + bar_width + bar_sep - min(left_bar_pos)
    )
    if label is not None and label != "":
        plt.legend(loc=legend_location, prop={"size": legend_size}, figure=figure)
    if value_labels is not None:
        handle_value_labels(
            value_labels,
            barpos,
            entries,
            yerr,
            ismulti=False,
            zorder=zorder,
            value_labels_rotation=value_labels_rotation,
        )

    ax = handle_grid(ax, vgrid, hgrid)
    if xlabels is not None and xlabels != False:
        # remove default labels
        # ax.set_xticklabels([])
        if type(xlabels) is tuple:
            xlab = list(xlabels)
        else:
            xlab = xlabels[:]
        jump = 1
        fontsize = text_sizes["xlabels"]
        if len(xlabels) < 50:
            fontsize = text_sizes["xlabels"]
        elif print_all_labels:
            fontsize = text_sizes["xlabels_many"]  #'small'
        if (
            relative_space_for_each_bar < 0.0125 and not print_all_labels
        ):  # condition means at most 80 xlabels (0.0125=1/80)
            jump = 1 + len(xlabels) // 80
        for j, x in enumerate(barpos):
            if j % jump == 0:
                xlab[j] = xlabels[j]
                # ax.annotate(xlabels[j], xy=(x,ymin), xytext=(0, -5),rotation=xlabels_rotation, textcoords='offset points', va='top', ha='center',size=fontsize)
            else:
                xlab[j] = ""
        horizontalalignment = "center"
        if type(xlabels_rotation) is int:
            if 15 < xlabels_rotation < 90:
                horizontalalignment = "right"
            elif xlabels_rotation < -15:
                horizontalalignment = "left"
        ax.set_xticklabels(
            xlab,
            rotation=xlabels_rotation,
            verticalalignment="top",
            horizontalalignment=horizontalalignment,
            fontsize=fontsize,
        )
        # figure.autofmt_xdate()
    ax, add_to_axis_label = consider_scientific_notation(
        ax,
        axis="y",
        publication=default_parameters["set_publish"],
        fontsize=text_sizes["xlabels"],
    )  # at the moment we use all_tight as an alias for publication quality y/n
    if ylabel is not None:
        ylabel += add_to_axis_label
    ax, add_to_axis_label = consider_scientific_notation(
        ax,
        axis="x",
        publication=default_parameters["set_publish"],
        fontsize=text_sizes["xlabels"],
    )  # at the moment we use all_tight as an alias for publication quality y/n
    if xlabel is not None:
        xlabel += add_to_axis_label
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=text_sizes["xlabel"], labelpad=10)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=text_sizes["ylabel"], labelpad=10)

    if upper_labels is not None:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(barpos)
        if type(upper_labels_rotation) is int:
            if 15 < upper_labels_rotation < 90:
                horizontalalignment = "left"
            elif upper_labels_rotation < -15:
                horizontalalignment = "right"
        ax2.set_xticklabels(
            upper_labels,
            rotation=upper_labels_rotation,
            verticalalignment="bottom",
            horizontalalignment=horizontalalignment,
            fontsize=text_sizes["xlabels"],
        )
    plt.draw()
    if save != False and save is not None:
        figure.savefig(
            save,
            dpi=default_figure_sizes["dpi"],
            bbox_inches="tight",
            bbox_extra_artists=[ax, figure],
            transparent=True,
        )  # ,bbox_inches="tight" is such that everything is in the figure... even though margins are removed.
    if show:
        plt.show(block=block)
    return figure


def multibar(
    entries,
    label="",
    yerr=None,
    bar_width=0.8,
    first_left_pos=None,
    space_per_entry=None,
    bar_sep=None,
    sub_bar_sep=0.0,
    use_bar_xticks=True,
    empty_space_after_ngroups=None,
    stacked=False,
    color=iworkpalette,
    hatch=None,
    linewidth=None,
    hline=None,
    vline=None,
    xlabels=None,
    xlabel=None,
    ylabel=None,
    title=None,
    xlabels_rotation="horizontal",
    upper_labels=None,
    upper_labels_rotation="horizontal",
    value_labels=None,
    value_labels_ypos=None,
    value_labels_offset_points=3,
    value_labels_rotation="horizontal",
    print_all_labels=False,
    y_range=None,
    x_range=None,
    subplot=None,
    figure=None,
    figure_size=None,
    hgrid=None,
    group_vgrid=False,
    vgrid=None,
    frame=None,
    save=False,
    plot_legend=True,
    legend_size=None,
    legend_location="upper right",
    show=True,
    block=False,
    zorder=2,
    y_major_tick_every=None,
    x_minor_tick_every=None,
    x_major_tick_every=None,
    reverse=False,
    y_minor_tick_every=None,
):
    """
    display multiple bars next to one another, or stacked on top of each other
     It assumes that each series has the same number of entries (so each contribute the same number of bars).
    if a dictionary (of lists for example) is given the keys will be used as labels for the series. Each value in each list is a column,
      values with the same index create a group of column
    xlabels is False won't set even the ticks
    """
    if frame is None:
        frame = default_parameters["frame"]
    if hgrid is None:
        hgrid = default_parameters["hgrid"]
    if vgrid is None:
        vgrid = default_parameters["vgrid"]
    if linewidth is None:
        linewidth = default_parameters["barlinewidth"]
    if figure_size is None:
        if default_figure_sizes["default"] is None:
            figure_size = default_figure_sizes["multibar"]
        else:
            figure_size = default_figure_sizes["default"]
    if default_figure_sizes["use_cm"]:
        figure_size = (cmToinch(figure_size[0]), cmToinch(figure_size[1]))
    if legend_size is None:
        legend_size = text_sizes["legend_size"]
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    plt.rc("xtick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick", labelsize=text_sizes["xlabels"])
    dkeys = None
    if type(entries) is dict:
        dkeys, entries = list(
            zip(
                *sorted(
                    zip(list(entries.keys()), list(entries.values())), reverse=reverse
                )
            )
        )
    elif isinstance(entries, OrderedDict):
        dkeys = list(entries.keys())
        entries = list(entries.values())
    if dkeys is not None:
        if label == "":
            label = dkeys
            label = list(map(str, label))
        if type(value_labels) is dict or isinstance(value_labels, OrderedDict):
            tmp = []
            for k in dkeys:
                tmp += [value_labels[k]]
            value_labels = tmp[:]
        if type(yerr) is dict or isinstance(yerr, OrderedDict):
            tmp = []
            for k in dkeys:
                tmp += [yerr[k]]
            yerr = tmp[:]
        if type(color) is dict or isinstance(color, OrderedDict):
            tmp = []
            for k in dkeys:
                tmp += [color[k]]
            color = tmp[:]

    if bar_sep is None:
        bar_sep = bar_width / 2.0

    new_figure = False
    if figure is None and subplot is None:
        new_figure = True
        figure = plt.figure(figsize=figure_size)

    if subplot is None:
        ax = figure.gca()
    else:
        ax = subplot
    if default_figure_sizes["all_tight"]:
        figure.set_tight_layout(True)  # plt.tight_layout()
    n_series = len(entries)  # the number of series to plot
    n_entries = len(entries[0])  # number of entries per series.
    entries = numpy.array(entries)
    if space_per_entry is not None:
        if stacked:
            bar_width = 0.8 * space_per_entry
            bar_sep = 0.2 * space_per_entry
            sub_bar_sep = 0.0
        else:
            if sub_bar_sep != 0.0 and sub_bar_sep is not None:
                sub_bar_sep = (0.1 / n_series) * space_per_entry
                bar_width = (0.7 / n_series) * space_per_entry
                bar_sep = 0.2 * space_per_entry
            else:
                sub_bar_sep = 0.0
                bar_width = (0.9 / n_series) * space_per_entry
                bar_sep = 0.1 * space_per_entry
    else:
        if stacked:
            space_per_entry = bar_width + bar_sep
        else:
            space_per_entry = (
                bar_width * n_series + sub_bar_sep * (n_series - 1) + bar_sep
            )
    if first_left_pos is None:
        first_left_pos = bar_width
    left = numpy.arange(
        first_left_pos,
        space_per_entry * n_entries + first_left_pos - 0.000005,
        space_per_entry,
    )
    if empty_space_after_ngroups is not None:
        if hasattr(empty_space_after_ngroups, "__len__"):
            ng, sp = empty_space_after_ngroups
        else:
            ng, sp = empty_space_after_ngroups, space_per_entry / 2.0
        left += numpy.array([0] * ng + [sp] * (n_entries - ng))
    ##  print left
    relative_space_for_each_entry = space_per_entry / (
        max(left) + space_per_entry - min(left)
    )
    xlabels_pos = left + (space_per_entry - bar_sep) / 2.0
    if type(color) is not list and not isinstance(color, cycle_list):
        color = n_series * [color]
    if type(hatch) is not list and not isinstance(hatch, cycle_list):
        hatch = n_series * [hatch]
    if type(label) is not list and type(label) is not tuple:
        label = n_series * [label]
    if type(yerr) is not list and not isinstance(yerr, numpy.ndarray):
        if yerr is None:
            yerr = n_series * [yerr]
        elif stacked:
            yerr = (n_series - 1) * [None] + [yerr]
        else:
            yerr = n_series * [yerr]
    if stacked:
        bottom = numpy.zeros(n_entries)
    else:
        bottom = None

    every_bar_pos = []
    for j, ent in enumerate(entries):
        if type(ent) is dict or isinstance(ent, OrderedDict):
            if type(ent) is dict:
                keys, ent = list(
                    zip(*sorted(zip(list(ent.keys()), list(ent.values()))))
                )
            else:
                keys, ent = list(ent.keys()), list(ent.values())
            if value_labels is not None and type(value_labels[j]) is dict:
                value_labels[j] = [value_labels[j][kel] for kel in keys]
            if xlabels is None:
                xlabels = list(keys)
        if type(color[j]) is dict and xlabels is not None:
            cols = []
            if xlabels[j] in color[j]:
                cols += [color[j][xlabels[j]]]
            else:
                print(
                    "WARNING in multibar() xlabels[j]=%s not in color dict"
                    % (xlabels[j])
                )

            if cols != []:
                color[j] = cols

        plt.bar(
            left,
            ent,
            width=bar_width,
            align="edge",
            color=color[j],
            bottom=bottom,
            hatch=hatch[j],
            label=label[j],
            linewidth=linewidth,
            figure=figure,
            zorder=zorder,
            yerr=yerr[j],
            error_kw=dict(
                elinewidth=default_error_bars["elinewidth"],
                ecolor="black",
                capsize=default_error_bars["capsize"],
                capthick=default_error_bars["capthick"],
                zorder=20,
            ),
        )

        every_bar_pos += list(left + bar_width / 2.0)
        if value_labels is not None:
            if value_labels == True:
                vlab = [numpy.round(val, 2) for val in value_labels[j]]
                handle_value_labels(
                    vlab,
                    every_bar_pos[-len(left) :],
                    ent,
                    yerr[j],
                    value_labels_rotation=value_labels_rotation,
                    ismulti=False,
                    zorder=zorder,
                )
            elif (
                not hasattr(value_labels[j], "__len__") or type(value_labels[j]) is str
            ):  # j will be zero the first time
                vlab = value_labels
                # xlabels_pos
                ypos = entries.max(axis=0)
                # yerr[j] is not ideal but given the fact that some may be none getting it from numpy is difficult..
                handle_value_labels(
                    vlab,
                    xlabels_pos,
                    ypos,
                    yerr[j],
                    value_labels_rotation=value_labels_rotation,
                    ismulti=False,
                    zorder=zorder,
                )
                value_labels = None
            else:
                vlab = value_labels[j][:]
                handle_value_labels(
                    vlab,
                    every_bar_pos[-len(left) :],
                    ent,
                    yerr[j],
                    value_labels_rotation=value_labels_rotation,
                    ismulti=False,
                    zorder=zorder,
                )
        if j == 0:
            group_left = list(left)[:]
        if stacked:
            bottom = bottom + numpy.array(ent)
        else:
            left = left + bar_width + sub_bar_sep

    # this remove top and right axis
    if not frame:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()  # ticks only on bottom axis
        ax.get_yaxis().tick_left()  # ticks only on left axis

    if title is not None:
        plt.text(
            0.5,
            1.03,
            title,
            horizontalalignment="center",
            fontsize=text_sizes["title"],
            transform=ax.transAxes,
        )
    if use_bar_xticks:
        # set yticks:
        ax = handle_ticks(
            ax,
            x_major_tick_every=False,
            y_major_tick_every=y_major_tick_every,
            x_minor_tick_every=False,
            y_minor_tick_every=y_minor_tick_every,
            new_figure=new_figure,
            xlabels=None,
        )

        if x_major_tick_every is False:
            ax.set_xticks([])
        else:
            ax.set_xticks(xlabels_pos[::x_major_tick_every])  # set the major ticks
        if xlabels is False or xlabels == []:
            ax.set_xticklabels([])
        if x_minor_tick_every == True and type(x_minor_tick_every) is not int:
            ax.set_xticks(every_bar_pos, minor=True)
        elif x_minor_tick_every is not None and x_minor_tick_every != False:
            ax.set_xticks(xlabels_pos[::x_minor_tick_every], minor=True)

    else:
        ax = handle_ticks(
            ax,
            x_major_tick_every=x_major_tick_every,
            y_major_tick_every=y_major_tick_every,
            x_minor_tick_every=x_minor_tick_every,
            y_minor_tick_every=y_minor_tick_every,
            new_figure=new_figure,
            xlabels=xlabels,
            xlabels_rotation=xlabels_rotation,
        )

    ax.ticklabel_format(
        style="sci", scilimits=(-2, 3), axis="y"
    )  # use scientific notation

    if y_range is not None:
        ax.set_ylim(y_range)
    if x_range is not None:
        ax.set_xlim(x_range)
    else:
        if stacked:
            ax.set_xlim(min(left) - bar_sep, max(left) + space_per_entry + bar_sep)
        else:
            ax.set_xlim(
                min(left) - space_per_entry - bar_sep,
                max(left) + space_per_entry / float(n_series) + bar_sep,
            )
    if hline is not None:
        if not hasattr(hline, "__len__"):
            hline = [hline]
        for hl in hline:
            plt.axhline(
                hl, color="black", ls="-", lw=plt.rcParams["axes.linewidth"], zorder=-2
            )
    if vline is not None:
        if not hasattr(vline, "__len__"):
            vline = [vline]
        for vl in vline:
            plt.axvline(
                vl, color="black", ls="--", lw=plt.rcParams["axes.linewidth"], zorder=-2
            )

    if group_vgrid:
        for xt in group_left[1:]:
            plt.axvline(xt - bar_sep / 2.0, color="black", ls=":", lw=0.5)

    if y_major_tick_every != False:
        ax, add_to_axis_label = consider_scientific_notation(
            ax,
            axis="y",
            publication=default_parameters["set_publish"],
            fontsize=text_sizes["xlabels"],
        )  # at the moment we use all_tight as an alias for publication quality y/n
    if ylabel is not None:
        ylabel += add_to_axis_label
    if xlabels != False and xlabels != [] and x_major_tick_every != False:
        ax, add_to_axis_label = consider_scientific_notation(
            ax,
            axis="x",
            publication=default_parameters["set_publish"],
            fontsize=text_sizes["xlabels"],
        )  # at the moment we use all_tight as an alias for publication quality y/n
        if xlabel is not None:
            xlabel += add_to_axis_label
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=text_sizes["xlabel"], labelpad=10)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=text_sizes["ylabel"], labelpad=10)
    ax = handle_grid(ax, vgrid, hgrid)

    if use_bar_xticks and xlabels is not None and xlabels != False:
        # remove default labels
        # ax.set_xticklabels([])
        xlab = xlabels[:]
        fontsize = text_sizes["xlabels"]
        if len(xlabels) < 50:
            fontsize = text_sizes["xlabels"]
        elif print_all_labels:
            fontsize = text_sizes["xlabels_many"]
        if x_major_tick_every is None:
            x_major_tick_every = 1
            if (
                relative_space_for_each_entry < 0.0125 and not print_all_labels
            ):  # condition means at most 80 xlabels (0.0125=1/80)
                x_major_tick_every = 1 + len(xlabels) // 80
            for j, x in enumerate(xlabels_pos):
                if j % x_major_tick_every == 0:
                    xlab[j] = xlabels[j]
                    # ax.annotate(xlabels[j], xy=(x,ymin), xytext=(0, -5),rotation=xlabels_rotation, textcoords='offset points', va='top', ha='center',size=fontsize)
                else:
                    xlab[j] = ""
        elif x_major_tick_every != False:
            xlab = []
            for j, x in enumerate(xlabels_pos):
                if hasattr(x_major_tick_every, "__len__"):
                    xlab += [xlabels[j]]
                elif j % x_major_tick_every == 0:
                    xlab += [xlabels[j]]
                    # ax.annotate(xlabels[j], xy=(x,ymin), xytext=(0, -5),rotation=xlabels_rotation, textcoords='offset points', va='top', ha='center',size=fontsize)

        horizontalalignment = "center"
        if type(xlabels_rotation) is int:
            if 15 < xlabels_rotation < 90:
                horizontalalignment = "right"
            elif xlabels_rotation < -15:
                horizontalalignment = "left"
        ax.set_xticklabels(
            xlab,
            rotation=xlabels_rotation,
            verticalalignment="top",
            fontsize=fontsize,
            horizontalalignment=horizontalalignment,
        )
        # figure.autofmt_xdate()

    if upper_labels is not None:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(xlabels_pos)
        ax2.set_xticklabels(
            upper_labels,
            rotation=upper_labels_rotation,
            verticalalignment="bottom",
            fontsize=text_sizes["xlabels"],
        )

    if label is not None and label != "" and plot_legend:
        plt.legend(
            loc=legend_location,
            frameon=False,
            prop={"size": legend_size},
            borderaxespad=0.01,
        )
    plt.draw()
    if save != False and save is not None:
        plt.savefig(
            save,
            dpi=default_figure_sizes["dpi"],
            bbox_inches="tight",
            bbox_extra_artists=[ax, figure],
            transparent=True,
        )  # ,bbox_inches="tight" is such that everything is in the figure... even though margins are removed.
    if show:
        plt.show(block=block)
    return figure


def cloudplot(
    entries,
    normal=False,
    plot_mean=True,
    plot_median=False,
    marker=None,
    markerfacecolor=True,
    markeredgecolor=True,
    markersize=15,
    x_range=None,
    yerr=None,
    zorder=1,
    figure=None,
    figure_size=None,
    save=None,
    xlabels=None,
    **kwargs
):
    """
    superseded by swarmplot 
    entries are like boxplot - list of lists
    can also give (but probably only if doing one cloud at the moment): labels=None, point_labels=None, labels_offset=None, flag_labels_color=(1,1,1,0.8) \
    """
    if figure_size is None:
        figure_size = default_figure_sizes["boxplot"]
    if marker is None:
        marker = "."
    if not hasattr(entries[0], "__len__"):
        entries = [entries]
    if figure is None:
        figure = plt.figure(figsize=figure_size)
    ewidth, capsize, capthick = (
        default_error_bars["elinewidth"],
        default_error_bars["capsize"],
        default_error_bars["capthick"],
    )  # save for restoring
    (
        default_error_bars["elinewidth"],
        default_error_bars["capsize"],
        default_error_bars["capthick"],
    ) = (ewidth + 2, capsize + 2, capthick + 2)
    if marker in [",", "."]:
        ms = markersize - 0.4 * markersize
    else:
        ms = markersize + 2
    if plot_mean:
        xv, yv, ye = [], [], []
        for j, en in enumerate(entries):
            if hasattr(yerr, "__len__"):
                w_average, w_stdev, _, SE = misc.weigthed_average_from_errors(
                    en, vals_errs=yerr[j]
                )
            else:
                w_average, w_stdev, _, SE = misc.weigthed_average_from_errors(
                    en, vals_errs=None
                )
            yv += [w_average]
            ye += [SE]
            xv += [j]
        figure = profile(
            yv,
            xv,
            yerr=ye,
            color="black",
            marker="s",
            markeredgecolor="black",
            ecolor="black",
            markersize=ms,
            markeredgewidth=default_error_bars["capthick"],
            markerfacecolor="white",
            zorder=zorder + 1,
            figure=figure,
            ls="",
        )
    if plot_median:
        ym = [numpy.median(en) for en in entries]
        figure = profile(
            ym,
            list(range(0, len(entries))),
            yerr=None,
            color="white",
            markeredgecolor="black",
            markersize=ms,
            markeredgewidth=default_error_bars["capthick"],
            marker="*",
            zorder=zorder + 1,
            markerfacecolor="black",
            figure=figure,
            ls="",
        )
    (
        default_error_bars["elinewidth"],
        default_error_bars["capsize"],
        default_error_bars["capthick"],
    ) = (
        ewidth,
        capsize,
        capthick,
    )  # restore
    if xlabels is not None and type(xlabels) is not str and hasattr(xlabels, "__len__"):
        xlabels = list(map(str, xlabels))
    if normal:  # gaussian
        xvals = [
            list(j + numpy.random.normal(loc=0, scale=0.1, size=len(en)))
            for j, en in enumerate(entries)
        ]  # without list get_min_max_glob failes
    else:  # uniform
        xvals = [
            list(j + numpy.random.uniform(low=-0.3, high=0.3, size=len(en)))
            for j, en in enumerate(entries)
        ]  # without list get_min_max_glob failes
    if x_range is None:
        x_range = (-0.5, len(entries) - 0.5)
    return profile(
        entries,
        x_values=xvals,
        ls="",
        x_major_tick_every=list(range(0, len(entries))),
        markersize=markersize,
        x_range=x_range,
        yerr=yerr,
        marker=marker,
        markeredgecolor=markeredgecolor,
        markerfacecolor=markerfacecolor,
        figure=figure,
        zorder=zorder,
        xlabels=xlabels,
        save=save,
        **kwargs
    )


def adjustx(in_start_value, increment, value_count):
    """
    used by swarm plot preprocess
     return an adjusted list of (x - increment_value, x, x + increment_value)
    """
    adjusted = []
    for i in range(value_count):
        directionality = numpy.power(-1, i)
        adjustment = i - (i % 2) - (i // 2) + 1
        increment_factor = increment * directionality * adjustment
        adjusted.append(in_start_value + increment_factor)
    return adjusted


def preprocess(x_value, y_series, increment_value=0.03, round_to_digits=None):
    """
    used by swarm plot
    x_value is the x-axis position (one number)
     return an adjusted list of (x - increment_value, x, x + increment_value)
    """
    if round_to_digits is None:
        round_to_digits = -1 * (
            get_order_of_magnitude(numpy.diff(get_min_max_glob(y_series))[0]) - 1
        )  # order of magnitude of data spread -2 (so about 100 point levels in spread)
        # print 'round_to_digits:',round_to_digits
    y_tracker = {}
    rounded_ys = {}
    for y in y_series:
        yr = numpy.round(y, round_to_digits)  # essentially acts like binning
        # print(yr,type(yr),y)
        y_tracker[yr] = (
            y_tracker.get(yr, 0) + 1
        )  # set to zero if not present, then add 1 (if present just adds 1)
        if yr not in rounded_ys:
            rounded_ys[yr] = [y]
        else:
            rounded_ys[yr] += [y]
    xvals, yvals = [], []
    for y_value, y_count in list(y_tracker.items()):
        if y_count == 1:
            append_x, append_y = [x_value], rounded_ys[y_value]
        else:
            start_value = x_value - (increment_value / 2.0 if y_count % 2 == 0 else 0)
            adjusted = adjustx(start_value, increment_value, y_count - 1)
            append_x, append_y = [start_value] + adjusted, rounded_ys[y_value]
        xvals += append_x
        yvals += append_y
    return xvals, yvals


def violinplot(
    entries,
    nbins=None,
    xpos=None,
    xlabels=None,
    show_average=True,
    connect_with_lines=False,
    use_percentile_and_median=34.1,
    boxes_linewidth=3,
    boxes_edges_color="black",
    bar_sep=0.3,
    median_as_upper_labels=True,
    bar_zorder=None,
    zorder=1,
    color=None,
    log_scale_y=False,
    y_major_tick_every=None,
    y_minor_tick_every=None,
    ylabels=None,
    y_range=None,
    vgrid=None,
    hgrid=None,
    xlabels_rotation="horizontal",
    markerfacecolor=True,
    markersize=None,
    markeredgecolor="black",
    x_range=None,
    save=None,
    show=True,
    figure_size=None,
    figure=None,
    ax=None,
    use_right_axis=False,
):
    """
    plots a histogram horizontally plot of entries

    """
    if hasattr(entries, "keys"):
        if xlabels is None:
            xlabels = list(entries.keys())
        entries = list(entries.values())
    if not hasattr(entries[0], "__len__"):
        entries = [entries]  # multiple distributions
    if figure_size is None:
        if len(entries) <= 4:
            figure_size = default_figure_sizes["boxplot"]
        else:
            figure_size = default_figure_sizes["profile"]
    if markersize is None:
        markersize = default_parameters["markersize"]
    if nbins is None:
        nbins = min([int(round(numpy.sqrt(len(e))) / 2) for e in entries])
        if nbins <= 1:
            nbins = 2
    if bar_zorder is None:
        bar_zorder = zorder + 1

    diff = 1
    if xpos is None:
        # numpy.arange(0, sum(binned_maximums), numpy.max(binned_maximums))
        # xpos = numpy.cumsum(binned_maximums)
        # xpos=numpy.array([j*numpy.max(binned_maximums) for j in range(len(entries))])
        xpos = list(range(0, len(entries)))
    elif len(xpos) > 1:
        diff = abs(xpos[1] - xpos[0])

    # define ranges and create histograms
    hist_range = get_min_max_glob(entries)
    binned_data_sets = [
        numpy.histogram(d, range=hist_range, bins=nbins)[0] for d in entries
    ]
    # print("DEB: hist_range",hist_range, "nbins=",nbins,"binned_data_sets=",binned_data_sets)

    binned_maximums = numpy.max(binned_data_sets, axis=1)
    binned_data_sets = numpy.array(binned_data_sets) / (
        1.03 * max(binned_maximums) / diff
    )

    if (
        color is None
        or type(color) is str
        or (type(color) is tuple and len(color) in [3, 4])
    ):
        color = [color] * len(entries)

    # The bin_edges are the same for all of the histograms
    bin_edges = numpy.linspace(hist_range[0], hist_range[1], nbins + 1)

    if hgrid is None and len(bin_edges) < 51:
        hgrid = list(bin_edges)

    centers = 0.5 * (bin_edges + numpy.roll(bin_edges, 1))[1:]
    heights = numpy.diff(bin_edges)
    # print("DEB: bin_edges=",list(bin_edges), "centers=",list(centers),"heights=",list(heights))

    if xlabels is None:
        xlabels = list(map(str, xpos))

    # print("DEB: len(entries)=%d len(xpos)=%d nbins=%d"%(len(entries),len(xpos),nbins),xpos,binned_maximums)
    if show_average or use_percentile_and_median:
        yv, ye = [], []
        lw = boxes_linewidth
        for j, en in enumerate(entries):
            if len(en) < 2:
                # if bar : yv+=[ en[0] ]
                # else : yv+=[numpy.nan]
                yv += [en[0]]
                if use_percentile_and_median:
                    ye += [(numpy.nan, numpy.nan)]
                else:
                    ye += [numpy.nan]
                continue
            if use_percentile_and_median:
                yv += [numpy.median(en)]
                ye += [
                    (
                        yv[-1] - numpy.percentile(en, 50 - use_percentile_and_median),
                        numpy.percentile(en, 50 + use_percentile_and_median) - yv[-1],
                    )
                ]
            else:
                w_average, w_stdev, _, SE = misc.weigthed_average_from_errors(
                    en, vals_errs=None
                )
                yv += [w_average]
                ye += [SE]
        if use_percentile_and_median:
            ye = list(zip(*ye))
        ms = markersize + 4
        markerav = "_"
        profcolor, ls = "black", ""
        if type(connect_with_lines) is str or type(connect_with_lines) is tuple:
            ls, connect_with_lines, profcolor = "-", True, connect_with_lines
        elif connect_with_lines:
            ls, profcolor = "-", boxes_edges_color
        if type(boxes_edges_color) is not str and type(boxes_edges_color) is not tuple:
            sys.stderr.write(
                "**Potential warn in violinplot - boxes_edges_color as list of colors not working for averages and error bars, setting these to black\n"
            )
            boxes_edges_color = "black"
        elinewidth = default_error_bars["elinewidth"]
        default_error_bars["elinewidth"] = lw  # temporary change error bar width
        figure = profile(
            yv,
            xpos,
            yerr=ye,
            color=profcolor,
            xlabels=xlabels,
            marker=markerav,
            markeredgecolor=boxes_edges_color,
            ecolor=boxes_edges_color,
            hgrid=False,
            vgrid=False,
            markersize=ms,
            markeredgewidth=lw,
            markerfacecolor="black",
            zorder=bar_zorder,
            figure=figure,
            ax=ax,
            ls=ls,
            figure_size=figure_size,
        )
        default_error_bars["elinewidth"] = elinewidth

    new_figure = False
    if figure is None:
        new_figure = True
        figure = plt.figure(figsize=figure_size)

    if ax is None:
        ax = figure.gca()

    # Cycle through and plot each histogram
    all_lefts, all_rights = [], []
    for j, (x_loc, binned_data) in enumerate(zip(xpos, binned_data_sets)):
        lefts = x_loc - 0.5 * binned_data
        all_rights += list(x_loc + 0.5 * binned_data)
        ax.barh(
            centers,
            binned_data,
            height=heights,
            left=lefts,
            color=color[j],
            edgecolor=None,
            zorder=zorder,
        )
        all_lefts += list(lefts)

    # print("DEB: xpos=", xpos, "xlabels=", xlabels)
    ax.set_xticks(xpos, minor=False)
    ax.set_xticklabels(xlabels, rotation=xlabels_rotation)
    # Must be before the range for log scales
    ax = handle_ticks(
        ax,
        x_major_tick_every=None,
        y_major_tick_every=y_major_tick_every,
        x_minor_tick_every=None,
        y_minor_tick_every=y_minor_tick_every,
        log_scale_y=log_scale_y,
        new_figure=new_figure,
        ylabels=ylabels,
        entries_xpos=xpos,
        use_right_axis=use_right_axis,
    )
    ax = handle_grid(ax, vgrid=vgrid, hgrid=hgrid)

    if x_range is None:
        m = min(all_lefts)
        M = max(all_rights)
        ax.set_xlim((m - 0.025 * (M - m), M + 0.025 * (M - m)))
    else:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)
    plt.draw()
    if save is not None and save != "":
        if "." not in save:
            save += ".png"
        plt.savefig(
            save,
            dpi=default_figure_sizes["dpi"],
            bbox_inches="tight",
            bbox_extra_artists=[ax, figure],
            transparent=True,
        )  # ,bbox_inches="tight" is such that everything is in the figure... even though margins are removed.
    if show:
        plt.show(block=False)
    return figure


def swarmplot(
    entries,
    increment_value=None,
    xpos=None,
    bar=False,
    add_boxplot=False,
    show_average=True,
    yerr=None,
    connect_with_lines=False,
    use_percentile_and_median=34.1,
    box_width=None,
    boxes_linewidth=3,
    boxes_edges_color="black",
    bar_sep=0.3,
    marker=".",
    markersize=None,
    notch=None,
    median_as_upper_labels=True,
    size_as_upper_labels=False,
    labels=None,
    zorder=1,
    bar_zorder=None,
    color=transp_iworkpalette,
    x_major_tick_every=None,
    markerfacecolor=True,
    markeredgecolor="black",
    x_range=None,
    save=None,
    show=True,
    figure_size=None,
    figure=None,
    upper_label_size=None,
    **kwargs_profile
):
    """
    plots a swarm plot of entries
    median_as_upper_labels only if add_boxplot
    increment_value may need adjusting depending on the number of points and compactness of the distributions.
     it represents the amount by which each point is offsetted in the x direction when multiple points of close value are present
     use_percentile will plot mean +/- use_percentile instead of standard error - ignored if add_boxplot
    """
    if bar_zorder is None:
        bar_zorder = zorder + 1
    if hasattr(entries, "keys"):
        xlabels = list(entries.keys())
        entries = list(entries.values())
    if not hasattr(entries[0], "__len__"):
        entries = [entries]  # multiple swarms
    if figure_size is None:
        if len(entries) <= 4:
            figure_size = default_figure_sizes["boxplot"]
        else:
            figure_size = default_figure_sizes["profile"]
    if markersize is None:
        markersize = default_parameters["markersize"]
    if size_as_upper_labels:
        median_as_upper_labels = True
    if upper_label_size is None:
        if size_as_upper_labels:
            upper_label_size = int(0.8 * text_sizes["xlabels"])
        else:
            upper_label_size = text_sizes["xlabels_many"]
    if increment_value is None:
        if hasattr(entries[0], "__len__"):
            increment_value = 0.1 / numpy.sqrt(max([len(e) for e in entries]))
        else:
            increment_value = 0.1 / numpy.sqrt(len(entries))
    xvalues = []
    yvalues = []
    if xpos is None:
        xpos = list(range(len(entries)))
    if x_major_tick_every is None:
        x_major_tick_every = xpos
    if add_boxplot:
        if notch is None:
            if any([len(xv) < 33 for xv in entries]):
                notch = False
            else:
                notch = True
        figure = boxplot(
            list(entries),
            x_values=xpos,
            notch=notch,
            x_major_tick_every=x_major_tick_every,
            median_as_upper_labels=False,
            hgrid=False,
            vgrid=False,
            figure=figure,
            color="none",
            showfliers=False,
            median_linewidth=boxes_linewidth + 0.5,
            box_width=box_width,
            boxes_linewidth=boxes_linewidth,
            boxes_edges_color=boxes_edges_color,
            show_average=show_average,
            zorder=bar_zorder,
            figure_size=figure_size,
        )
    elif show_average or use_percentile_and_median or bar:
        yv, ye = [], []
        lw = boxes_linewidth
        for j, en in enumerate(entries):
            if len(en) < 2:
                # if bar : yv+=[ en[0] ]
                # else : yv+=[numpy.nan]
                yv += [en[0]]
                if use_percentile_and_median:
                    ye += [(numpy.nan, numpy.nan)]
                else:
                    ye += [numpy.nan]
                continue
            if use_percentile_and_median:
                yv += [numpy.median(en)]
                ye += [
                    (
                        yv[-1] - numpy.percentile(en, 50 - use_percentile_and_median),
                        numpy.percentile(en, 50 + use_percentile_and_median) - yv[-1],
                    )
                ]
            else:
                if hasattr(yerr, "__len__"):
                    w_average, w_stdev, _, SE = misc.weigthed_average_from_errors(
                        en, vals_errs=yerr[j]
                    )
                else:
                    w_average, w_stdev, _, SE = misc.weigthed_average_from_errors(
                        en, vals_errs=None
                    )
                yv += [w_average]
                ye += [SE]
        if use_percentile_and_median:
            ye = list(zip(*ye))
        ms = markersize + 4
        markerav = "_"
        if bar:
            markerav = ","
            figure = profile(
                yv,
                xpos,
                bar=True,
                yerr=None,
                markeredgecolor=boxes_edges_color,
                ecolor=boxes_edges_color,
                hgrid=False,
                vgrid=False,
                bar_sep=bar_sep,
                linewidth=lw,
                color="none",
                zorder=bar_zorder,
                figure=figure,
                ls="",
                figure_size=figure_size,
            )
        profcolor, ls = "black", ""
        if type(connect_with_lines) is str or type(connect_with_lines) is tuple:
            ls, connect_with_lines, profcolor = "-", True, connect_with_lines
        elif connect_with_lines:
            ls, profcolor = "-", boxes_edges_color
        if type(boxes_edges_color) is not str and type(boxes_edges_color) is not tuple:
            sys.stderr.write(
                "**Potential warn in swarmplot - boxes_edges_color as list of colors not working for averages and error bars, setting these to black\n"
            )
            boxes_edges_color = "black"
        elinewidth = default_error_bars["elinewidth"]
        default_error_bars["elinewidth"] = lw  # temporary change error bar width
        figure = profile(
            yv,
            xpos,
            yerr=ye,
            color=profcolor,
            marker=markerav,
            markeredgecolor=boxes_edges_color,
            ecolor=boxes_edges_color,
            hgrid=False,
            vgrid=False,
            markersize=ms,
            markeredgewidth=lw,
            markerfacecolor="white",
            zorder=bar_zorder,
            figure=figure,
            ls=ls,
            figure_size=figure_size,
        )
        default_error_bars["elinewidth"] = elinewidth
    for j, en in enumerate(entries):
        # print en # it will change order of ys
        xv, yv = preprocess(xpos[j], en, increment_value=increment_value)
        # print yv,'\n'
        xvalues += [xv]
        yvalues += [yv]
    ax = figure.gca()
    if labels is not None:
        sys.stderr.write(
            "**WARNING** in swarmplot labels given as input but this cannot be processed as order of ypoints is changed by preprocessing, setting labels to None\n"
        )
        labels = None
    figure = profile(
        yvalues,
        xvalues,
        x_major_tick_every=x_major_tick_every,
        ls="",
        yerr=yerr,
        marker=marker,
        zorder=zorder,
        color=color,
        markersize=markersize,
        figure=figure,
        markerfacecolor=markerfacecolor,
        markeredgecolor=markeredgecolor,
        figure_size=figure_size,
        labels=labels,
        **kwargs_profile
    )
    # adjust range
    if x_range is None:
        ax.set_xlim((min(xpos) - 0.6, max(xpos) + 0.6))
    else:
        ax.set_xlim(x_range)
    if median_as_upper_labels:
        ndec = 3
        if type(median_as_upper_labels) is int:
            ndec = int(median_as_upper_labels)
        upperLabels = []
        x_label_pos = []
        for ji, v in enumerate(entries):
            if hasattr(v, "__len__"):
                le = len(v)
                if le == 0:
                    continue  # we do not increase the counter, we skip one label since this box plot is empty
                med = numpy.median(v)
                xm = xpos[ji]
            else:
                le = len(entries)
                xm = xpos[0]
                med = numpy.median(entries)
            if size_as_upper_labels:
                upperlabel = "%d" % (le)
            else:
                upperlabel = "N=%d\nm=%.3g" % (le, numpy.round(med, ndec))
            upperLabels += [upperlabel]
            x_label_pos += [xm]
        # if vertical_upper_labels :uplabel_rotation='vertical'
        # else :
        uplabel_rotation = "horizontal"
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x_label_pos)
        ax2.set_xticklabels(
            upperLabels,
            rotation=uplabel_rotation,
            verticalalignment="bottom",
            fontsize=upper_label_size,
        )  # , weight=upper_label_fontweight)
        plt.sca(ax)  # set back

    plt.draw()
    if save is not None and save != "":
        if "." not in save:
            save += ".png"
        plt.savefig(
            save,
            dpi=default_figure_sizes["dpi"],
            bbox_inches="tight",
            bbox_extra_artists=[ax, figure],
            transparent=True,
        )  # ,bbox_inches="tight" is such that everything is in the figure... even though margins are removed.
    if show:
        plt.show(block=False)
    return figure


def boxplot(
    entries,
    xlabels=False,
    x_values=None,
    bootstrap=None,
    sym="",
    notch=True,
    vert=True,
    whis=[5, 95],
    title=None,
    font_size_title=None,
    showfliers=True,
    whiskers_color="black",
    whisker_linewidth=None,
    whisker_linestyle="-",
    boxes_linewidth=None,
    boxes_edges_color="black",
    color=color_set2[1],
    box_width=None,
    boxes_alpha=1.0,
    fliers_color=almost_black,
    fliers_size=None,
    median_color="black",
    median_linewidth=2,
    vline=None,
    hline=None,
    show_average=True,
    average_color="w",
    average_marker=".",
    average_markeredgecolor="black",
    average_markersize=12,
    xlabel=None,
    ylabel=None,
    font_size_xlabel=None,
    font_size_ylabel=None,
    hgrid=None,
    vgrid=None,
    alpha_hgrid=0.7,
    color_hgrid="lightgrey",
    linestyle_hgrid="-",
    linestyle_hgrid_minor="--",
    xlabels_rotation=0,
    fontsize_xlabels=None,
    y_range=None,
    x_range=None,
    y_major_tick_every=None,
    y_minor_tick_every=None,
    x_major_tick_every=None,
    x_minor_tick_every=False,
    vline_every_nboxes=None,
    set_ylogscale=False,
    median_as_upper_labels=2,
    vertical_upper_labels=False,
    upper_label_size=None,
    upper_label_fontweight="bold",
    vertical_upper_labels_verticalalignment="bottom",
    figure=None,
    save=False,
    show=True,
    figure_size=None,
    zorder=0,
    block=False,
    close=False,
    ax_aspect=None,
    adjust_plot=True,
    frame=None,
    tight_layout=True,
):
    """
    whis : float, sequence (default = 1.5) or string -- Consider also [5,95]
        As a float, determines the reach of the whiskers past the first
        and third quartiles (e.g., Q3 + whis*IQR, IQR = interquartile
        range, Q3-Q1). Beyond the whiskers, data are considered outliers
        and are plotted as individual points. Set this to an unreasonably
        high value to force the whiskers to show the min and max values.
        Alternatively, set this to an ascending sequence of percentile
        (e.g., [5, 95]) to set the whiskers at specific percentiles of
        the data. Finally, *whis* can be the string 'range' to force the
        whiskers to the min and max of the data. In the edge case that
        the 25th and 75th percentiles are equivalent, *whis* will be
        automatically set to 'range'.
    bootstrap : None (default) or integer
        Specifies whether to bootstrap the confidence intervals
        around the median for notched boxplots. If bootstrap is None,
        no bootstrapping is performed, and notches are calculated
        using a Gaussian-based asymptotic approximation  (see McGill, R.,
        Tukey, J.W., and Larsen, W.A., 1978, and Kendall and Stuart,
        1967). Otherwise, bootstrap specifies the number of times to
        bootstrap the median to determine it's 95% confidence intervals.
        Values between 1000 and 10000 are recommended.

    """
    if font_size_xlabel is None:
        font_size_xlabel = text_sizes["xlabel"]
    if font_size_ylabel is None:
        font_size_ylabel = text_sizes["ylabel"]
    if font_size_title is None:
        font_size_title = text_sizes["title"]
    if fontsize_xlabels is None:
        fontsize_xlabels = text_sizes["xlabels"]
    if upper_label_size is None:
        upper_label_size = text_sizes["xlabels_many"]
    if frame is None:
        frame = default_parameters[
            "frame"
        ]  # it will be shown anyway if median_as_upper_labels is given
    elif frame is False:
        median_as_upper_labels = False
    if hgrid is None:
        hgrid = default_parameters["hgrid"]
    if fliers_size is None:
        fliers_size = default_parameters["markersize"]
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rc("xtick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick", labelsize=text_sizes["xlabels"])
    if figure_size is None:
        if default_figure_sizes["default"] is None:
            figure_size = default_figure_sizes["boxplot"]
        else:
            figure_size = default_figure_sizes["default"]

    if type(entries) is dict:
        xlabels = list(entries.keys())
        if type(color) is dict:
            color = [color[e] for e in xlabels]
        entries = list(entries.values())
    elif isinstance(entries, numpy.ndarray):
        entries = list(entries)
    new_figure = False
    if figure is None:
        new_figure = True
        figure = plt.figure(figsize=figure_size)
    else:
        show = None
    if whisker_linewidth is None:
        whisker_linewidth = default_parameters["linewidth"]
    if boxes_linewidth is None:
        boxes_linewidth = default_parameters["linewidth"]

    ax = figure.gca()

    if ax_aspect is not None:
        ax.set_aspect(ax_aspect)
    if adjust_plot:
        plt.subplots_adjust(left=0.075, right=0.95, top=0.94, bottom=0.20)

    # if set_ylogscale:
    #    ax.set_yscale('log')
    #    if y_major_tick_every  is not None:
    #        #ax.get_yaxis().get_major_formatter().labelOnlyBase = False
    #        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if showfliers and (sym == "" or type(sym) is not str):
        sym = "b+"
    elif showfliers is False:
        sym = ""
    if box_width is not None and not hasattr(box_width, "__len__"):
        box_width = len(entries) * [box_width]
    if x_values is None:
        x_values = list(range(len(entries)))
    if x_major_tick_every is None:
        x_major_tick_every = x_values
    if color is not None:
        bp = plt.boxplot(
            entries,
            bootstrap=bootstrap,
            notch=notch,
            sym=sym,
            vert=vert,
            whis=whis,
            positions=x_values,
            widths=box_width,
            patch_artist=True,
            showfliers=showfliers,
        )
        if boxes_alpha is None:
            boxes_alpha = 1
        if type(color) is list or isinstance(color, cycle_list):
            for j, patch in enumerate(bp["boxes"]):
                if type(color[j]) is tuple and len(color[j]) < 4:
                    col = color[j] + (boxes_alpha,)
                else:
                    col = color[j]
                patch.set_facecolor(col)
        else:
            for patch in bp["boxes"]:
                if type(color) is tuple and len(color) < 4:
                    col = color + (boxes_alpha,)
                else:
                    col = color
                patch.set_facecolor(col)

        if boxes_edges_color is not None:
            if type(boxes_edges_color) is list or isinstance(
                boxes_edges_color, cycle_list
            ):
                for patch, col in zip(bp["boxes"], boxes_edges_color):
                    patch.set_edgecolor(col)
            else:
                for patch in bp["boxes"]:
                    patch.set_edgecolor(boxes_edges_color)
    else:
        bp = plt.boxplot(
            entries,
            bootstrap=bootstrap,
            notch=notch,
            sym=sym,
            vert=vert,
            whis=whis,
            positions=x_values,
            widths=box_width,
            patch_artist=False,
            showfliers=showfliers,
            zorder=zorder,
        )
        if boxes_edges_color is not None:
            if type(boxes_edges_color) is list:
                i = 0
                for j, item in enumerate(bp["boxes"]):
                    if hasattr(entries[i], "__len__") and len(entries[i]) == 0:
                        i += 1  # move one more haead
                    #                 plt.setp(item, edgecolor=boxes_edges_color[i])
                    item.setp(edgecolor=boxes_edges_color[i])
                    i += 1
            else:
                #             plt.setp(bp['boxes'], edgecolor=boxes_edges_color)
                bp["boxes"].setp(edgecolor=boxes_edges_color)
    plt.setp(bp["boxes"], linewidth=boxes_linewidth)

    if whiskers_color is not None:
        if type(whiskers_color) is list:
            i = 0
            for j, item in enumerate(bp["whiskers"]):
                if (
                    hasattr(entries[i], "__len__")
                    and len(entries[i]) == 0
                    and j % 2 == 0
                ):
                    i += 1  # move one more haead
                plt.setp(
                    item,
                    color=whiskers_color[i],
                    linewidth=whisker_linewidth,
                    linestyle=whisker_linestyle,
                )
                plt.setp(
                    bp["caps"][j],
                    color=whiskers_color[i],
                    linewidth=whisker_linewidth,
                    linestyle=whisker_linestyle,
                )
                if j % 2 == 1:
                    i += 1
        else:
            plt.setp(
                bp["whiskers"],
                color=whiskers_color,
                linewidth=whisker_linewidth,
                linestyle=whisker_linestyle,
            )
            plt.setp(
                bp["caps"],
                color=whiskers_color,
                linewidth=whisker_linewidth,
                linestyle=whisker_linestyle,
            )
            # print dir(bp['whiskers']),bp.keys()

    if fliers_color is None:
        fliers_color = almost_black
    if len(sym) > 0:
        marker = sym[-1]
    else:
        marker = "+"
    if type(fliers_color) is list:
        i = 0
        for j, item in enumerate(bp["fliers"]):
            if hasattr(entries[i], "__len__") and len(entries[i]) == 0 and j % 2 == 0:
                i += 1  # move one more haead
            plt.setp(
                item,
                color=fliers_color[i],
                markerfacecolor=fliers_color[i],
                markeredgecolor=fliers_color[i],
                markersize=fliers_size,
                marker=marker,
            )
            if j % 2 == 1:
                i += 1
    else:
        plt.setp(
            bp["fliers"],
            color=fliers_color,
            markeredgecolor=fliers_color,
            markerfacecolor=fliers_color,
            markersize=fliers_size,
            marker=marker,
        )  # , marker='+')

    if median_color is not None or median_color != "default":
        if median_linewidth is not None or median_linewidth != "default":
            for median in bp["medians"]:
                median.set(color=median_color, linewidth=median_linewidth)
        else:
            for median in bp["medians"]:
                median.set(color=median_color)

    if (
        show_average
    ):  # Finally, overplot the sample averages, with horizontal alignment in the center of each box
        for i, median in enumerate(bp["medians"]):
            plt.plot(
                x_values[i],
                [numpy.mean(entries[i])],
                color=average_color,
                marker=average_marker,
                markeredgecolor=average_markeredgecolor,
                markersize=average_markersize,
            )

    if vline is not None:
        if not hasattr(vline, "__len__"):
            vline = [vline]
        for vl in vline:
            plt.axvline(
                vl,
                color="black",
                ls="--",
                lw=plt.rcParams["axes.linewidth"],
                zorder=zorder - 2,
            )
    if hline is not None:
        if not hasattr(hline, "__len__"):
            hline = [hline]
        for hl in hline:
            plt.axhline(
                hl,
                color="black",
                ls="--",
                lw=plt.rcParams["axes.linewidth"],
                zorder=zorder - 2,
            )
    ax = figure.gca()
    if title is not None:
        if font_size_title is None:
            ax.set_title(title)
        elif font_size_title == "default":
            plt.text(
                0.5,
                1.0,
                title,
                horizontalalignment="center",
                fontsize=text_sizes["title"],
                transform=ax.transAxes,
                weight="bold",
            )
        else:
            ax.set_title(title, fontsize=font_size_title)

    if xlabel is not None:
        if font_size_xlabel is None:
            ax.set_xlabel(xlabel)
        elif font_size_xlabel == "default":
            ax.set_xlabel(xlabel, fontsize=text_sizes["xlabel"], labelpad=10)
        else:
            ax.set_xlabel(xlabel, fontsize=font_size_xlabel)

    if ylabel is not None:
        if font_size_ylabel is None:
            ax.set_ylabel(ylabel)
        elif font_size_ylabel == "default":
            ax.set_ylabel(ylabel, fontsize=text_sizes["ylabel"], labelpad=10)
        else:
            ax.set_ylabel(ylabel, fontsize=font_size_ylabel)

    ax = handle_grid(ax, vgrid, hgrid)

    if median_as_upper_labels == True or type(median_as_upper_labels) is int:
        ndec = int(median_as_upper_labels)
        upperLabels = []
        x_label_pos = []
        j = 0
        for ji, v in enumerate(entries):
            if hasattr(v, "__len__"):
                le = len(v)
                if le == 0:
                    continue  # we do not increase the counter, we skip one label since this box plot is empty
            else:
                le = len(entries)
                xm = x_values[0]
            if j >= len(bp["medians"]):
                break
            m = bp["medians"][j]
            ym = float(m.get_ydata()[0])
            xm = x_values[ji]
            upperlabel = "N=%d\nm=%.3g" % (le, numpy.round(ym, ndec))
            upperLabels += [upperlabel]
            #           upperLabels += ['N='+str(le)+'\nm='+str(numpy.round(ym, ndec))]
            x_label_pos += [xm]
            """
            if not vertical_upper_labels :
                transOffset = matplotlib.transforms.offset_copy(ax.transData, fig=plt.gcf(), x = 0, y=-0.50, units='inches')
                top = ax.get_ylim()[1]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.05
                if vertical_upper_labels_verticalalignment == 'default':
                    ax.text(xm, top, upperLabels[-1], transform=transOffset, horizontalalignment='center',verticalalignment='center',size=upper_label_size,linespacing=1.2,weight=upper_label_fontweight)
                elif vertical_upper_labels_verticalalignment is None:
                    ax.text(xm, top, upperLabels[-1], transform=transOffset, horizontalalignment='center',size=upper_label_size,linespacing=1.2,weight=upper_label_fontweight)
                else:
                    ax.text(xm, top, upperLabels[-1], transform=transOffset, horizontalalignment='center',verticalalignment=vertical_upper_labels_verticalalignment,size=upper_label_size,linespacing=1.2,weight=upper_label_fontweight)
            """

            j += 1
        if vertical_upper_labels:
            rotation = "vertical"
        else:
            rotation = "horizontal"
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x_label_pos)
        ax2.set_xticklabels(
            upperLabels,
            rotation=rotation,
            verticalalignment="bottom",
            fontsize=upper_label_size,
            weight=upper_label_fontweight,
        )
        # if xlabels==[] :
        #    for tic in ax2.xaxis.get_major_ticks():
        #        tic.tick1On = tic.tick2On = False
        plt.sca(ax)  # set back
    elif median_as_upper_labels is False or median_as_upper_labels is None:
        # this remove top and right axis
        if not frame:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.get_xaxis().tick_bottom()  # ticks only on bottom axis
            ax.get_yaxis().tick_left()  # ticks only on left axis

    if vline_every_nboxes:
        if type(vline_every_nboxes) == list:
            vline_every_nboxes.insert(0, 0)
            for i, j in enumerate(vline_every_nboxes):
                if j > 0:
                    new_j = j + vline_every_nboxes[i - 1]
                    if new_j < len(bp["medians"]):
                        xms = [
                            float(bp["medians"][new_j].get_xdata()[0]),
                            float(bp["medians"][new_j - 1].get_xdata()[1]),
                        ]
                        ax.axvline(
                            (xms[0] + xms[1]) / 2.0, color="black", ls="--", lw=1
                        )
        else:
            for j in range(0, len(bp["medians"]), vline_every_nboxes):
                if j > 0:
                    xms = [
                        float(bp["medians"][j].get_xdata()[0]),
                        float(bp["medians"][j - 1].get_xdata()[1]),
                    ]
                    ax.axvline((xms[0] + xms[1]) / 2.0, color="black", ls="--", lw=1)
    # print 'x_values:',x_values
    # print 'xlabels',xlabels
    ax = handle_ticks(
        ax,
        x_major_tick_every=x_major_tick_every,
        y_major_tick_every=y_major_tick_every,
        x_minor_tick_every=x_minor_tick_every,
        y_minor_tick_every=y_minor_tick_every,
        new_figure=False,
        log_scale_y=set_ylogscale,
        entries_xpos=x_values,
        xlabels=xlabels,
        xlabels_rotation=xlabels_rotation,
    )

    # Upper X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)

    if x_range is not None:
        ax.set_xlim(x_range)
    # else : ax.set_xlim(0.5, (len(entries)+0.5))     # Set the axes ranges and axes labels
    if y_range is not None:
        ax.set_ylim(y_range)

    plt.draw()
    if tight_layout:
        figure.set_tight_layout(True)  # plt.tight_layout()
    if save != False and save is not None:
        if "." not in save:
            save += ".pdf"
        plt.savefig(
            save,
            dpi=default_figure_sizes["dpi"],
            bbox_inches="tight",
            bbox_extra_artists=[ax, figure],
            transparent=True,
        )  # ,bbox_inches="tight" is such that everything is in the figure... even though margins are removed.
        # plt.savefig(save)
    if show:
        plt.show(block=block)

    if close:
        plt.close()
    return figure


def handle_flag_labels(
    ax,
    x,
    y,
    labels,
    point_labels=None,
    ismulti=False,
    labels_offset=None,
    edgecolor="black",
    fontcolor="black",
    flag_labels_color=(1, 1, 1, 0.6),
    labels_alpha=None,
    markersize=30,
    boxstyle=None,
    labels_size=None,
    x_range=None,
    y_range=None,
    plot_cbar=False,
    scalarMap=None,
    N_uniq_points=None,
    cbar_labels=None,
    flag_fit_outliers=False,
    fit_tuple=None,
    fit_results=None,
    zorder=0,
):
    """

    flag_fit_outliers can be True (defaults to 5%) or a number in 0, 100 to select the percent of outliers to label
    boxstyle includes 'circle' 'round' 'round,pad=0.2' 'square'
    """

    if type(edgecolor) is str or (
        type(edgecolor) is tuple and (len(edgecolor) == 3 or len(edgecolor) == 4)
    ):
        edgecolor = [edgecolor] * len(x)
    if boxstyle is None:
        if point_labels is not None and point_labels is not False:
            boxstyle = "circle"
        else:
            boxstyle = "round,pad=0.2"
    if flag_fit_outliers > 0:
        if ismulti:
            sys.stderr.write(
                "**WARNING** plotter.handle_flag_labels cannot flag_fit_outliers with ismulti True - not implemented\n"
            )
            flag_fit_outliers = False
        elif fit_tuple is None or fit_results is None:
            sys.stderr.write(
                "*Warn plotter.handle_flag_labels cannot flag_fit_outliers as either fit_tuple with function or fit_results with parameters are None\n"
            )
            flag_fit_outliers = False
        else:
            if type(fit_tuple) is tuple or hasattr(fit_tuple, "__len__"):
                fit_tuple = fit_tuple[0]  # just the function
            elif fit_tuple == True or fit_tuple == 1:
                fit_tuple = lambda xx, p: xx * p[0] + p[1]
            elif fit_tuple == 2:
                fit_tuple = lambda xx, p: xx * xx * p[0] + xx * p[1] + p[2]
            if not hasattr(fit_tuple, "__call__"):
                sys.stderr.write(
                    "\n*ERROR** plotter.handle_flag_labels cannot flag_fit_outliers as fit_function non callable\n"
                )
            if type(flag_fit_outliers) is bool:
                flag_fit_outliers = 5  # flag 5% outliers
            sorted_residuals_inds = numpy.argsort(
                -1
                * (
                    (
                        numpy.array(y)
                        - fit_tuple(numpy.array(x), fit_results["parameters_fitted"])
                    )
                    ** 2
                )
            )
            N_to_show = int(
                numpy.round(0.01 * flag_fit_outliers * len(sorted_residuals_inds))
            )
            if labels is None:
                labels = numpy.arange(
                    0, len(sorted_residuals_inds)
                )  # set to index of data
            else:
                labels = numpy.array(labels)
            labels = labels.astype("str")  # otherwise below raise error
            labels[sorted_residuals_inds[N_to_show + 1 :]] = ""  # set to empty

    if point_labels is not None and point_labels is not False:
        if not hasattr(point_labels, "__len__"):
            point_labels = list(range(len(y)))
        if ismulti:
            sys.stderr.write(
                "\n\n**WARNING** plotter.handle_flag_labels cannot put point_labels with ismulti True - not implemented (use two figures)\n"
            )
        if (
            "colormap" in str(type(flag_labels_color)).lower()
        ):  # it is a colormap, convert to list with required colors - in this case points are coloured according to value in point_labels and color bar is plotted
            N_uniq_points = len(misc.uniq(point_labels))
            if point_labels[0] is not None and type(point_labels[0]) is not str:
                cNorm = matplotlib.colors.Normalize(
                    vmin=min(point_labels), vmax=max(point_labels)
                )
                scalarMap = matplotlib.cm.ScalarMappable(
                    norm=cNorm, cmap=flag_labels_color
                )
                flag_labels_color = [
                    scalarMap.to_rgba(idx, alpha=labels_alpha) for idx in point_labels
                ]  # list of color spaced with colormap
                try:
                    # cbar_labels=None #sorted(N_uniq_points) # [-1] will raise exception if no number in str(l)
                    print("plotter.handle_flag_labels cbar_labels=", cbar_labels)
                    scalarMap._A = []
                    plot_cbar = True
                except Exception:
                    sys.stderr.write(
                        "Warn in plotter.handle_flag_labels failed processing colorbar ticks\n"
                    )
                    scalarMap._A = []
                    # print 'plotter.profile label=',label
                    sys.stderr.flush()
                    plot_cbar = False
                # else : ncol=max([1,len(label)/6])
            else:
                cNorm = matplotlib.colors.Normalize(vmin=0, vmax=N_uniq_points - 0.98)
                scalarMap = matplotlib.cm.ScalarMappable(
                    norm=cNorm, cmap=flag_labels_color
                )
                flag_labels_color = [
                    scalarMap.to_rgba(idx, alpha=labels_alpha)
                    for idx in range(N_uniq_points)
                ]  # list of color spaced with colormap
        elif type(flag_labels_color) is not list and not isinstance(
            flag_labels_color, cycle_list
        ):
            flag_labels_color = [flag_labels_color] * len(x)

        for i in range(len(x)):
            # print 'DEB point_labels:',point_labels[i],x[i],y[i]
            lsize = labels_size
            if plot_cbar:
                if lsize is None:
                    lsize = markersize / 3
                ax.annotate(
                    " ",
                    (x[i], y[i]),
                    va="center",
                    ha="center",
                    fontsize=lsize,
                    color=fontcolor,
                    bbox=dict(
                        boxstyle=boxstyle,
                        fc=flag_labels_color[i],
                        ec=edgecolor[i],
                        alpha=labels_alpha,
                    ),
                    zorder=zorder,
                )
            else:
                if lsize is None:
                    lsize = markersize / 2
                ax.annotate(
                    str(point_labels[i]),
                    (x[i], y[i]),
                    va="center",
                    ha="center",
                    fontsize=lsize,
                    color=fontcolor,
                    bbox=dict(
                        boxstyle=boxstyle,
                        fc=flag_labels_color[i],
                        alpha=labels_alpha,
                        ec=edgecolor[i],
                    ),
                    zorder=zorder,
                )
        # obscure re-range
        if x_range is None:
            x_range = (min(x), max(x))
            sp = x_range[1] - x_range[0]
            x_range = (
                x_range[0] - 0.04 * sp,
                x_range[1] + 0.04 * sp,
            )  # (Round_To_n(x_range[0]-0.04*sp ,n=0, only_decimals=True) , Round_To_n(x_range[1]+0.04*sp ,n=0, only_decimals=True) )
        if y_range is None:
            y_range = (min(y), max(y))
            sp = y_range[1] - y_range[0]
            y_range = (
                y_range[0] - 0.04 * sp,
                y_range[1] + 0.04 * sp,
            )  # (Round_To_n(y_range[0]-0.04*sp ,n=0, only_decimals=True) , Round_To_n(y_range[1]+0.04*sp ,n=0, only_decimals=True) )
    if ismulti:
        # see if all profiles have same labels or each has different
        if labels is not None:
            if len(labels) == len(y):  # same as number of profiles!
                for j, prof in enumerate(y):
                    if labels[j] is None:
                        continue  # no labels for this profile
                    if hasattr(x[0], "__len__") and len(x) == len(
                        y
                    ):  # one x-axis per profile
                        xvals = x[j]
                    else:
                        xvals = x
                    if labels_offset is None and hasattr(
                        labels, "__len__"
                    ):  # slightly randomise
                        labels_offset = numpy.array(
                            [[-10, 10]] * len(labels[j])
                        ) + numpy.random.uniform(
                            low=-5, high=5, size=(len(labels[j]), 2)
                        )
                    if labels_size is None:
                        if len(labels) < 15:
                            fontsize = text_sizes["value_labels"]
                        else:
                            fontsize = text_sizes["xlabels_many"]
                    else:
                        fontsize = labels_size
                    if (
                        type(flag_labels_color) is not list
                        and not isinstance(flag_labels_color, cycle_list)
                        and not "colormap" in str(type(flag_labels_color)).lower()
                    ):
                        flag_color = [flag_labels_color] * len(labels[j])
                    elif len(flag_labels_color) == len(y):
                        flag_color = [flag_labels_color[j]] * len(
                            labels[j]
                        )  # one color per profile
                    else:
                        flag_color = flag_labels_color
                    if not hasattr(labels_offset, "__len__") or len(labels_offset) == 2:
                        labels_offset = [labels_offset] * len(labels[j])
                    jc = 0
                    for labl, X, Y in zip(labels[j], xvals, prof):
                        kl = ax.annotate(
                            labl,
                            xy=(X, Y),
                            xytext=labels_offset[jc],
                            fontsize=fontsize,
                            xycoords="data",
                            textcoords="offset points",
                            ha="right",
                            va="bottom",
                            bbox=dict(
                                boxstyle=boxstyle, fc=flag_color[jc], alpha=labels_alpha
                            ),
                            arrowprops=dict(arrowstyle="-", connectionstyle=None),
                            zorder=zorder,
                        )
                        kl.draggable()
                        jc += 1
                labels = None  # WE DID them already
            else:
                if type(labels[0]) is str or not hasattr(
                    labels[0], "__len__"
                ):  # same labels for all profiles
                    labels = list(labels) * len(y)
                else:
                    labels = [item for sublist in labels for item in sublist]
                # flatten all via a deep copy
                y = [item for sublist in y for item in sublist]
                x = [item for sublist in x for item in sublist]
        if point_labels is not None and point_labels != False:
            if type(point_labels[0]) is str or not hasattr(
                point_labels[0], "__len__"
            ):  # same labels for all profiles
                point_labels = list(point_labels) * len(y)
            else:
                point_labels = [item for sublist in point_labels for item in sublist]
            # flatten all via a deep copy
            y = [item for sublist in y for item in sublist]
            x = [item for sublist in x for item in sublist]
    if labels is not None:
        if labels_offset is None and hasattr(labels, "__len__"):  # slightly randomise
            labels_offset = numpy.array(
                [[-10, 10]] * len(labels)
            ) + numpy.random.uniform(low=-5, high=5, size=(len(labels), 2))
        if labels_size is None:
            if len(labels) < 15:
                fontsize = text_sizes["value_labels"]
            else:
                fontsize = text_sizes["xlabels_many"]
        else:
            fontsize = labels_size
        if (
            type(flag_labels_color) is not list
            and not isinstance(flag_labels_color, cycle_list)
            and not "colormap" in str(type(flag_labels_color)).lower()
        ):
            flag_labels_color = [flag_labels_color] * len(labels)
        if not hasattr(labels_offset, "__len__") or len(labels_offset) == 2:
            labels_offset = [labels_offset] * len(labels)
        jc = 0
        for labl, X, Y in zip(labels, x, y):
            if labl == "" or labl is None:
                continue
            kl = ax.annotate(
                labl,
                xy=(X, Y),
                xytext=labels_offset[jc],
                fontsize=fontsize,
                xycoords="data",
                textcoords="offset points",
                ha="right",
                va="bottom",
                bbox=dict(
                    boxstyle=boxstyle,
                    fc=flag_labels_color[jc],
                    ec=edgecolor[jc],
                    alpha=labels_alpha,
                ),
                arrowprops=dict(arrowstyle="-", connectionstyle=None),
                zorder=zorder,
            )
            kl.draggable()
            jc += 1
    return x_range, y_range, plot_cbar, N_uniq_points, cbar_labels, scalarMap


def scatter(
    x,
    y,
    linfit=False,
    print_r_pval=False,
    label=None,
    yerr=None,
    xerr=None,
    contour=False,
    nbins=None,
    log=None,
    log_scale_x=False,
    log_scale_y=False,
    marker=None,
    title=None,
    figure_size=None,
    swapfit=False,
    vline=None,
    hline=None,
    hline_style="-",
    vline_style="-",
    xlabels=None,
    ylabels=None,
    zorder=0,
    xlabel=None,
    ylabel=None,
    x_range=None,
    y_range=None,
    same_scale=False,
    cmap=None,
    markerfacecolor="black",
    markeredgecolor="black",
    markeredgewidth=None,
    ecolor=None,
    alpha=None,
    markersize=30,
    flag_fit_outliers=False,
    point_labels=False,
    flag_labels_boxstyle=None,
    labels=None,
    labels_fontcolor="black",
    labels_offset=None,
    flag_labels_color=(1, 1, 1, 0.6),
    labels_size=None,
    labels_alpha=1,
    figure=None,
    ax=None,
    draw_unity_line=False,
    hgrid=None,
    vgrid=None,
    linewidth=None,
    fit_linecolor=None,
    fit_linestyle=None,
    linestyle="-",
    linecolor="black",
    save=False,
    legend_size=None,
    legend_location="best",
    frame=True,
    show=True,
    block=False,
    plot_colorbar_if_needed=True,
    x_major_tick_every=None,
    y_major_tick_every=None,
    x_minor_tick_every=None,
    y_minor_tick_every=None,
    prefer_noisy_fit=False,
    calculate_fit_CI=False,
    fit_CI_from_resample=False,
    plot_fit_CI=False,
    fit_results=None,
    fit_label=None,
    fit_ignore_yerr=False,
    fit_allow_extra_fraction=None,
):
    """
    can plot colorbar with different color points like:
    f=plotter.scatter( x, y ,marker='',point_labels=numpy.array(z), flag_labels_color=plotter.plt.get_cmap('jet') )
    labels_alpha=0 makes the shape around the labels disappear.
    swapfit simply swaps x and y to perform the fit..
    prefer_noisy_fit may do a sort of bayesan fit but doublecheck
    flag_fit_outliers can be True (defaults to 5%) or a number in 0, 100 to select the percent of outliers to label
        if labels is not given it will use the index of the data
    """
    if fit_results is None:
        fit_results = {}
    if frame is None:
        frame = default_parameters["frame"]
    if hgrid is None:
        hgrid = default_parameters["hgrid"]
    if vgrid is None:
        vgrid = default_parameters["vgrid"]
    if marker is None:
        marker = default_parameters["scatter_marker"]
    if linewidth is None:
        linewidth = plt.rcParams["axes.linewidth"]
    if figure_size is None:
        if default_figure_sizes["default"] is None:
            figure_size = default_figure_sizes["scatter"]
        else:
            figure_size = default_figure_sizes["default"]
    if default_figure_sizes["use_cm"]:
        figure_size = (cmToinch(figure_size[0]), cmToinch(figure_size[1]))
    if legend_size is None:
        legend_size = text_sizes["legend_size"]
    if fit_linecolor is None:
        fit_linecolor = linecolor
    if fit_linestyle is None:
        fit_linestyle = linestyle
    if fit_allow_extra_fraction is None:
        fit_allow_extra_fraction = default_parameters["fit_allow_extra_fraction"]
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    plt.rc("xtick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick", labelsize=text_sizes["xlabels"])

    if print_r_pval or linfit:
        r_pears, pvalue = scipy.stats.pearsonr(x, y)
        print(
            "correlation coefficient (R) = %s\t pvalue= %s"
            % (repr(r_pears), repr(pvalue))
        )
    new_figure = False
    if figure is None:
        new_figure = True
        figure = plt.figure(figsize=figure_size)
    if hasattr(figure, "set_xlim"):  # it is actually an axis object
        ax = figure
        figure = ax.figure
    elif ax is None:
        ax = figure.gca()

    # MOVED DOWN  but don't remember why this was here
    # ax= handle_ticks(ax, x_major_tick_every, y_major_tick_every, x_minor_tick_every, y_minor_tick_every,new_figure=new_figure,xlabels=xlabels, cbar_in_figure=(hasattr(point_labels, '__len__')  and point_labels[0] is not None and type(point_labels[0]) is not str) )

    if title is not None:
        plt.text(
            0.5,
            1.03,
            title,
            horizontalalignment="center",
            fontsize=text_sizes["title"],
            transform=ax.transAxes,
        )

    if log is not None:
        xMasked = numpy.ma.masked_where(x <= 0, x)  # Mask pixels with a value of zero
        yMasked = numpy.ma.masked_where(y <= 0, y)  # Mask pixels with a value of zero
        if log == 10:
            y = numpy.log10(yMasked)
            x = numpy.log10(xMasked)
        elif log == 2:
            y = numpy.log2(yMasked)
            x = numpy.log2(xMasked)
        else:
            y = numpy.log(yMasked)
            x = numpy.log(xMasked)

    if x_range is not None and hasattr(
        x, "__len__"
    ):  # we may be plotting a single point
        for j, xx in enumerate(x):
            if hasattr(xx, "__len__"):
                continue  # don't check for multi
            if (x_range[0] is not None and x_range[0] > xx) or (
                x_range[1] is not None and x_range[1] < xx
            ):
                if labels is not None:
                    l = labels[j]
                else:
                    l = ""
                sys.stderr.write(
                    "Warn in plotter.scatter() x_range excluding point %s (%lf,%lf) from plot\n"
                    % (str(l), xx, y[j])
                )
    if y_range is not None and hasattr(
        y, "__len__"
    ):  # we may be plotting a single point
        for j, yy in enumerate(y):
            if hasattr(yy, "__len__"):
                continue  # don't check for multi
            if (y_range[0] is not None and y_range[0] > yy) or (
                y_range[1] is not None and y_range[1] < yy
            ):
                if labels is not None:
                    l = labels[j]
                else:
                    l = ""
                sys.stderr.write(
                    "Warn in plotter.scatter() y_range excluding point %s (%lf,%lf) from plot\n"
                    % (str(l), x[j], yy)
                )

    if yerr is not None or xerr is not None:
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            xerr=xerr,
            ls="none",
            ecolor=ecolor,
            elinewidth=default_error_bars["elinewidth"],
            capsize=default_error_bars["capsize"],
            capthick=default_error_bars["capthick"],
            errorevery=1,
            label=label,
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markeredgewidth=markeredgewidth,
            figure=figure,
            markerfacecolor=markerfacecolor,
            zorder=zorder,
        )
    elif point_labels is None or point_labels is False:
        # else:
        if (
            type(markerfacecolor) is tuple
            and len(markerfacecolor) in [3, 4]
            and len(markerfacecolor) == len(x)
        ):  # probably RGB tuple but would plot different points in different colors
            if len(markerfacecolor) == 3:
                markerfacecolor = markerfacecolor + (1,)  # add alpha
            elif len(markerfacecolor) == 4 and alpha is None or alpha == 1:
                alpha = markerfacecolor[-1]
                markerfacecolor = markerfacecolor[:3]
        ax.scatter(
            x,
            y,
            c=markerfacecolor,
            s=markersize,
            edgecolor=markeredgecolor,
            lw=markeredgewidth,
            label=label,
            marker=marker,
            alpha=alpha,
            figure=figure,
            cmap=cmap,
            zorder=zorder,
        )

    # add a fit
    # print 'DEB fit_label',fit_label,'linfit',linfit
    ax = add_fit(
        ax,
        x,
        y,
        linfit,
        yerr=yerr,
        fit_ignore_yerr=fit_ignore_yerr,
        swapfit=swapfit,
        linewidth=linewidth,
        linecolor=fit_linecolor,
        linestyle=fit_linestyle,
        fit_allow_extra_fraction=fit_allow_extra_fraction,
        prefer_noisy_fit=prefer_noisy_fit,
        calculate_CI=calculate_fit_CI,
        permute_CI=swapfit,
        do_data_resampling=fit_CI_from_resample,
        plot_fit_CI=plot_fit_CI,
        fit_results=fit_results,
        zorder=zorder + 1,
        label=fit_label,
    )

    # add flag labels or point_labels if not None
    (
        x_range,
        y_range,
        plot_cbar,
        N_uniq_points,
        cbar_labels,
        scalarMap,
    ) = handle_flag_labels(
        ax,
        x,
        y,
        labels,
        point_labels=point_labels,
        fontcolor=labels_fontcolor,
        labels_offset=labels_offset,
        edgecolor=markeredgecolor,
        boxstyle=flag_labels_boxstyle,
        flag_labels_color=flag_labels_color,
        labels_size=labels_size,
        labels_alpha=labels_alpha,
        markersize=markersize,
        x_range=x_range,
        y_range=y_range,
        flag_fit_outliers=flag_fit_outliers,
        fit_tuple=linfit,
        fit_results=fit_results,
        zorder=zorder,
    )
    if not plot_colorbar_if_needed :
        plot_cbar=False

    # handle ticks, before it was almost at the beginning
    ax = handle_ticks(
        ax,
        x_major_tick_every,
        y_major_tick_every,
        x_minor_tick_every,
        y_minor_tick_every,
        log_scale_x=log_scale_x,
        log_scale_y=log_scale_y,
        new_figure=new_figure,
        xlabels=xlabels,
        ylabels=ylabels,
        cbar_in_figure=(
            hasattr(point_labels, "__len__")
            and point_labels[0] is not None
            and type(point_labels[0]) is not str
        ),
    )

    if not frame:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()  # ticks only on bottom axis
        ax.get_yaxis().tick_left()  # ticks only on left axis

    # ax,add_to_axis_label = consider_scientific_notation(ax,axis='y', publication=default_parameters['set_publish'],fontsize=text_sizes['xlabels']) # at the moment we use all_tight as an alias for publication quality y/n
    # if ylabel is not None : ylabel+=add_to_axis_label
    # ax,add_to_axis_label = consider_scientific_notation(ax,axis='x', publication=default_parameters['set_publish'],fontsize=text_sizes['xlabels']) # at the moment we use all_tight as an alias for publication quality y/n
    # if xlabel is not None : xlabel+=add_to_axis_label
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=text_sizes["xlabel"], labelpad=10)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=text_sizes["ylabel"], labelpad=10)
    # contour
    if contour:
        figure = draw_contour(
            x,
            y,
            nbins=nbins,
            H=None,
            xedges=None,
            yedges=None,
            figure=figure,
            linewidth="auto",
        )
    if draw_unity_line:
        xmi, xma = ax.get_xlim()
        ymi, yma = ax.get_ylim()
        ra = (min(xmi, ymi), max(yma, xma))
        xs = numpy.linspace(ra[0], ra[1], num=400)
        # chi2,p=scipy.stats.chisquare(y, x)
        # print('  chi2=%s chi2red=%s  p=%s' % (repr(chi2),repr(chi2/float(len(x)-1)),repr(p)))
        ax.plot(
            xs, xs, color=linecolor, linewidth=linewidth, linestyle=linestyle, zorder=0
        )

    ax = handle_grid(ax, vgrid, hgrid)

    if hline is not None:
        if not hasattr(hline, "__len__"):
            hline = [hline]
        for hl in hline:
            plt.axhline(
                hl, color=linecolor, ls=hline_style, lw=linewidth, zorder=zorder - 2
            )
    if vline is not None:
        if not hasattr(vline, "__len__"):
            vline = [vline]
        for vl in vline:
            plt.axvline(
                vl, color=linecolor, ls=vline_style, lw=linewidth, zorder=zorder - 2
            )

    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)
    if same_scale:
        if (x_range is not None or y_range is not None) and x_range != y_range:
            sys.stderr.write(
                "Plotter.scatter same_scale overwrites x_range and y_range and sets them to the same.\n"
            )
        xmi, xma = min(x), max(x)  # ax.get_xlim()
        ymi, yma = min(y), max(y)  # ax.get_ylim()
        ra = [min(xmi, ymi), max(yma, xma)]
        rang = ra[1] - ra[0]
        ra[0] -= 0.03 * rang
        ra[1] += 0.03 * rang
        ax.set_xlim(ra)
        ax.set_ylim(ra)

    # rotate and align the tick labels so they look better
    # figure.autofmt_xdate(self, bottom=0.2, rotation=30, ha=u'right')
    if show or save != False:
        if (label is not None and label != False) or (
            fit_label is not None and fit_label != False
        ):
            plt.legend(
                loc=legend_location,
                prop={"size": legend_size},
                numpoints=1,
                frameon=False,
                borderpad=0,
                handletextpad=0,
            )

    if plot_cbar:
        jump = N_uniq_points // 10
        if jump <= 0:
            jump = 1
        if cbar_labels is None:
            cbarlabels = None
        else:
            cbarlabels = [
                c if j in range(0, N_uniq_points, jump) else ""
                for j, c in enumerate(cbar_labels)
            ]
            cbarlabels[-1] = cbar_labels[-1]
            for j in range(jump - 1):
                cbarlabels[-2 - j] = ""
        add_colorbar_to_seq_profile(
            figure, scalarMap, ax=ax, cbar_major_ntick=None, cbar_labels=cbarlabels
        )
    plt.draw()
    if save != False and save is not None:
        figure.savefig(
            save,
            dpi=default_figure_sizes["dpi"],
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[ax, figure],
        )  # ,bbox_inches="tight" is such that everything is in the figure... even though margins are removed.
        # plt.savefig(save, dpi=plt.gcf().dpi)
    if show:
        plt.show(block=block)
    return figure


def draw_contour(
    x,
    y,
    nbins=None,
    H=None,
    xedges=None,
    yedges=None,
    figure=None,
    advanced=False,
    linewidth="auto",
    figure_size=None,
):
    # extent = [xedges.min(),xedges.max(),yedges.min(),yedges.max()]# [yedges[0], yedges[-1], xedges[0], xedges[-1]]
    # levels=[0.95*len(x), 0.75*len(x), 0.50*len(x),0.25*len(x),0.05*len(x)][::-1]
    # print 100.*H/float(len(x))
    # cset = plt.contour(H,extent=extent, colors=['black','DarkBlue','blue','DarkRed','red','DimGray'],levels=levels,linewidths=numpy.linspace(1,3,len(levels)) )#,origin='lower'
    # plt.clabel(cset, inline=1, fontsize=text_sizes['value_labels'], fmt='%1.0i')
    # for c in cset.collections:
    #    c.set_linestyle('solid')
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    if linewidth is None:
        linewidth = default_parameters["linewidth"]
    if figure_size is None:
        if default_figure_sizes["default"] is None:
            figure_size = default_figure_sizes["scatter"]
        else:
            figure_size = default_figure_sizes["default"]
    if default_figure_sizes["use_cm"]:
        figure_size = (cmToinch(figure_size[0]), cmToinch(figure_size[1]))
    new_figure = False
    if figure is None:
        new_figure = True
        figure = plt.figure(figsize=figure_size)
        plt.rcParams["xtick.direction"] = "out"
        plt.rcParams["ytick.direction"] = "out"
        plt.rc("xtick", labelsize=text_sizes["xlabels"])
        plt.rc("ytick", labelsize=text_sizes["xlabels"])

    ax = figure.gca()
    if nbins is None:
        if len(x) > 25:
            nbins = 0.5 * int(
                numpy.sqrt(1.0 * len(x))
            )  # use numpy.sqrt rule (like excel)
        else:
            nbins = int(len(x) // 3)
    if H is None:
        H, xedges, yedges = numpy.histogram2d(x, y, bins=nbins)
        # H needs to be rotated and flipped (so that origin is bottom left rather than top left?)
        H = numpy.rot90(H)
        H = numpy.flipud(H)
    if advanced:
        levels = []
        M = H.max()
        targets = sorted(
            [0.1, 0.25, 0.5, 0.75, 0.9]
        )  # will be reversed so that start from 10% up
        for jj in numpy.linspace(M, 0, 100):
            s = sum(H[numpy.where(H > jj)])
            if s > targets[0] * len(
                x
            ):  # len(x) is the total number of points so the first asks whether s contains at least 10% of points - hence it is the top 90%
                print(
                    "Contour:", jj, s, targets[0]
                )  # depedning on how H is built may never reach 90%
                levels += [jj]  # save cutoff value of H that contains x% of points
                targets = targets[1:]
                if targets == []:
                    break
        # print M,levels
        extent = [
            xedges.min(),
            xedges.max(),
            yedges.min(),
            yedges.max(),
        ]  # [yedges[0], yedges[-1], xedges[0], xedges[-1]]
        if linewidth == "auto":
            linewidth = numpy.linspace(1, 3, len(levels))
        cset = ax.contour(
            H, extent=extent, levels=levels[::-1], linewidths=linewidth
        )  # ,origin='lower',colors=['black','DarkBlue','blue','DarkRed','red','DimGray']
        # cset.legend_elements()
        label_percent = False
        # if label_percent :
        #    fmt = {} # label percentages
        #    strs = [('%d'%(100*(1-f)))+'%' for f in targets ]
        #    for l, s in zip(cset.levels, strs):
        #        fmt[l] = s
        # else : fmt='%1.0i'
        # ax.clabel(cset,cset.levels, inline=True, fontsize=text_sizes['value_labels'], fmt=fmt)
        # for c in cset.collections:
        #    c.set_linestyle('solid')
    else:
        k = scipy.stats.kde.gaussian_kde(numpy.vstack((x, y)))
        xi, yi = numpy.mgrid[
            x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j
        ]
        zi = k(numpy.vstack([xi.flatten(), yi.flatten()]))
        M = zi.max()
        levels = [
            0.1 * M,
            0.25 * M,
            0.5 * M,
            0.75 * M,
            0.9 * M,
        ]  # quite good in getting the points (in terms fo percent) of the higher percentiles in particualr
        # print levels
        if linewidth == "auto":
            linewidth = numpy.linspace(1, 3, len(levels))
        ax.contour(
            xi,
            yi,
            zi.reshape(xi.shape),
            levels=levels,
            linewidths=numpy.linspace(1, 3, len(levels)),
        )
        # print numpy.percentile(zi,[5,10,50,90,95,99]),'->',len(x)*numpy.percentile(zi,[5,10,50,90,95,99])
        # plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
    return figure


def histogram2d(
    x,
    y,
    nbins=None,
    linfit=False,
    contour=False,
    draw_unity_line=False,
    log=None,
    normalize_max=False,
    mask_zeros=True,
    log_scale=False,
    figure=None,
    x_range=None,
    y_range=None,
    print_warn=True,
    cbar_label=None,
    xlabel=None,
    ylabel=None,
    title=None,
    hgrid=None,
    vgrid=None,
    same_scale=False,
    print_r_pval=True,
    figure_size=None,
    swapfit=False,
    linewidth=1.0,
    linestyle="-",
    linecolor="black",
    cmap="coolwarm",
    cbar_label_rotation=270,
    cbar_major_ntick=None,
    cbar_fraction=0.04,
    plot_colorbar=True,
    xlabels=None,
    ylabels=True,
    frame=True,
    x_major_tick_every=None,
    y_major_tick_every=None,
    x_minor_tick_every=None,
    y_minor_tick_every=None,
    hline=None,
    vline=None,
    block=False,
    save=None,
    show=True,
):

    if hgrid is None:
        hgrid = default_parameters["hgrid"]
    if vgrid is None:
        vgrid = default_parameters["vgrid"]
    if frame is None:
        frame = default_parameters["frame"]
    if figure_size is None:
        if default_figure_sizes["default"] is None:
            figure_size = default_figure_sizes["scatter"]
        else:
            figure_size = default_figure_sizes["default"]
    if default_figure_sizes["use_cm"]:
        figure_size = (cmToinch(figure_size[0]), cmToinch(figure_size[1]))

    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    plt.rc("xtick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick", labelsize=text_sizes["xlabels"])

    if type(cmap) is str:
        cmap = matplotlib.cm.get_cmap(cmap)

    if print_r_pval:
        r_pears, pvalue = scipy.stats.pearsonr(x, y)
        print(
            "correlation coefficient (R) = %s\t pvalue= %s"
            % (repr(r_pears), repr(pvalue))
        )
    new_figure = False
    if figure is None:
        new_figure = True
        figure = plt.figure(figsize=figure_size)

    # Estimate the 2D histogram
    if nbins is None:
        if len(x) > 25:
            nbins = int(
                0.5 * int(numpy.sqrt(1.0 * len(x)))
            )  # use numpy.sqrt rule (like excel)
        else:
            nbins = int(len(x) // 3)
        print("nbins=", nbins)
    nbins = int(nbins)
    if x_range is not None:
        xxx = []
        yyy = []
        for j, xx in enumerate(x):
            if x_range[0] <= xx <= x_range[1]:
                xxx += [xx]
                yyy += [y[j]]
            elif print_warn:
                # if labels is not None : l=labels[j]
                l = ""
                sys.stderr.write(
                    "Warn in plotter.histogram2d() x_range excluding point %s (%lf,%lf) from plot\n"
                    % (str(l), xx, y[j])
                )
        x = xxx
        y = yyy
    if y_range is not None:
        xxx = []
        yyy = []
        for j, yy in enumerate(y):
            if y_range[0] <= yy <= y_range[1]:
                yyy += [yy]
                xxx += [x[j]]
            elif print_warn:
                # if labels is not None :l=labels[j]
                l = j
                sys.stderr.write(
                    "Warn in plotter.histogram2d() y_range excluding point %s (%lf,%lf) from plot\n"
                    % (str(l), x[j], yy)
                )
        x = xxx
        y = yyy
    H, xedges, yedges = numpy.histogram2d(x, y, bins=nbins)
    # H needs to be rotated and flipped (so that origin is bottom left rather than top left?)
    H = numpy.rot90(H)
    H = numpy.flipud(H)
    # Mask zeros
    if mask_zeros:
        Hmasked = numpy.ma.masked_where(H == 0, H)  # Mask pixels with a value of zero
    else:
        Hmasked = H
    # contour
    if contour:
        figure = draw_contour(
            x,
            y,
            nbins=nbins,
            H=H,
            xedges=xedges,
            yedges=yedges,
            figure=figure,
            linewidth="auto",
        )
    # Plot 2D histogram using pcolor
    ax = figure.gca()
    ax = handle_ticks(
        ax,
        x_major_tick_every,
        y_major_tick_every,
        x_minor_tick_every,
        y_minor_tick_every,
        new_figure=new_figure,
        xlabels=xlabels,
    )

    if log is not None:
        if log == 10:
            Hmasked = numpy.log10(Hmasked)
            if cbar_label is not None and "log" not in cbar_label.lower():
                cbar_label = "Log " + cbar_label
        elif log == 2:
            Hmasked = numpy.log2(Hmasked)
            if cbar_label is not None and "log2" not in cbar_label.lower():
                cbar_label = "log2 " + cbar_label
        else:
            Hmasked = numpy.log(Hmasked)
            if (
                cbar_label is not None
                and "log" not in cbar_label.lower()
                and "ln" not in cbar_label.lower()
            ):
                cbar_label = "ln " + cbar_label
    if normalize_max:
        Hmasked = float(normalize_max) * Hmasked / Hmasked.max()
    img = plt.pcolormesh(xedges, yedges, Hmasked, figure=figure, cmap=cmap)
    if log_scale:
        ax.set_yscale("symlog", base=10)
        ax.set_xscale("symlog", base=10)
        # if avoid_scientific_notation:
        #    yticks=ax.yaxis.get_majorticklocs()
        #    xlab=[ 10**i for i in xrange(len(yticks))]
        #    ax.set_yticklabels(xlab,rotation='horizontal',verticalalignment='center',horizontalalignment='right',fontsize=text_sizes['xlabels'])
    """
    *fraction*    0.15; fraction of original axes to use for colorbar
    *pad*         0.05 if vertical, 0.15 if horizontal; fraction
                  of original axes between colorbar and new image axes
    *shrink*      1.0; fraction by which to shrink the colorbar
    *aspect*      20; ratio of long to short dimensions
    *anchor*      (0.0, 0.5) if vertical; (0.5, 1.0) if horizontal;
                  the anchor point of the colorbar axes
    *panchor*     (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal;
                  the anchor point of the colorbar parent axes. If
                  False, the parent axes' anchor will be unchanged
    ANd much more see help(plt.colorbar)
    """
    if plot_colorbar:
        # figure=add_colorbar_to_seq_profile(figure, img, cbar_major_ntick=cbar_major_ntick ,cbar_label=cbar_label, cbar_label_rotation=cbar_label_rotation,cbar_fraction=cbar_fraction)

        if cbar_major_ntick is None and default_figure_sizes["all_tight"]:
            cbar_major_ntick = 5
        cbar = plt.colorbar(
            orientation="vertical",
            fraction=cbar_fraction,
            drawedges=False,
            spacing="proportional",
            ticks=None,
        )
        # cbar=figure.colorbar(image_or_mappable, ax=figure.axes,orientation='vertical',drawedges=False,spacing='proportional',ticks=None,fraction=cbar_fraction)
        if cbar_label is not None:
            cbar.set_label(
                cbar_label,
                fontsize=text_sizes["xlabel"],
                horizontalalignment="right",
                rotation=cbar_label_rotation,
                labelpad=7,
            )
            # print "HERE"
        if cbar_major_ntick is not None:
            cmin, cmax = cbar.get_clim()
            tmp_min = Round_To_n(cmin, 0)
            if abs(tmp_min) < 10:
                tmp_min = 0
            if hasattr(cbar_major_ntick, "__len__"):
                ticks = cbar_major_ntick
            else:
                ticks = list(
                    numpy.arange(
                        tmp_min,
                        cmax,
                        Round_To_n((cmax - cmin) / float(cbar_major_ntick), 0),
                    )
                )
            if cmin == int(cmin):
                ticks[0] = cmin
            if cmax == int(cmax):
                if abs(ticks[-1] - cmax) / (cmax - cmin) < 0.03:
                    ticks[-1] = cmax
                else:
                    ticks += [cmax]
            cbar.set_ticks(ticks)
        """
        cbar = plt.colorbar(orientation='vertical',fraction=cbar_fraction,drawedges=False,spacing='proportional',ticks=None)
        # the below is a workaround for bugs in some renderers which might results in white bars within colors
        #cbar.solids.set_edgecolor("face")
        if cbar_label is not None : cbar.ax.set_ylabel(cbar_label,fontsize=text_sizes['xlabel'],rotation=cbar_label_rotation)
        if cbar_major_ntick is not None : 
            cmin,cmax= cbar.get_clim()
            tmp_min=Round_To_n(cmin,0)
            if abs(tmp_min)<10 : tmp_min=0
            ticks=list( numpy.arange(tmp_min,cmax,Round_To_n( (cmax-cmin)/float(cbar_major_ntick),0)))
            if cmin==int(cmin) : ticks[0]=cmin
            if cmax==int(cmax) : ticks[-1]=cmax
            cbar.set_ticks( ticks )
            #cmajorLocator   = matplotlib.ticker.MultipleLocator(cbar_major_tick_every)
            #bar.ax.yaxis.set_major_locator(cmajorLocator)  #JUST NOT WORKING
            #cbar.update_ticks()
        """
    if linfit == True or linfit == 1:
        if swapfit == "both":
            swapped_coefficients, pol_fun, R2 = polfit(x, y, pol_order=1, swap=True)
            print(
                "Swapped linfit coefficients: "
                + str(swapped_coefficients)
                + " R^2= "
                + str(R2)
                + " pol: "
                + str(pol_fun)[2:]
            )
            lab2 = str(pol_fun) + " R^2= %4.2lf" % (R2)
            xs = numpy.linspace(min(x), max(x), num=40)
            ys = pol_fun(xs)
            ax.plot(
                xs,
                ys,
                label=lab2,
                color=linecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=19,
            )
            swapfit = False
        coefficients, pol_fun, R2 = polfit(x, y, pol_order=1, swap=swapfit)
        print(
            "linfit coefficients: "
            + str(coefficients)
            + " R^2= "
            + str(R2)
            + " pol: "
            + str(pol_fun)[2:]
        )
        lab2 = str(pol_fun) + " R^2= %4.2lf" % (R2)
        xs = numpy.linspace(min(x), max(x), num=40)
        ys = pol_fun(xs)
        ax.plot(
            xs,
            ys,
            label=lab2,
            color=linecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=20,
        )
    elif type(linfit) is int:
        coefficients, pol_fun = polfit(x, y, pol_order=linfit)
        print(
            "Polfit"
            + str(linfit)
            + " coefficients: "
            + str(coefficients)
            + " pol: "
            + str(pol_fun)[2:]
        )
        lab2 = str(pol_fun)
        xs = numpy.linspace(min(x), max(x), num=400)
        ys = pol_fun(xs)
        ax.plot(
            xs,
            ys,
            label=lab2,
            color=linecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=20,
        )

    if not frame:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()  # ticks only on bottom axis
        ax.get_yaxis().tick_left()  # ticks only on left axis
    if not log_scale:
        ax, add_to_axis_label = consider_scientific_notation(
            ax,
            axis="y",
            publication=default_parameters["set_publish"],
            fontsize=text_sizes["xlabels"],
        )  # at the moment we use all_tight as an alias for publication quality y/n
    if ylabel is not None:
        ylabel += add_to_axis_label
    ax, add_to_axis_label = consider_scientific_notation(
        ax,
        axis="x",
        publication=default_parameters["set_publish"],
        fontsize=text_sizes["xlabels"],
    )  # at the moment we use all_tight as an alias for publication quality y/n
    if xlabel is not None:
        xlabel += add_to_axis_label
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=text_sizes["xlabel"], labelpad=10)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=text_sizes["ylabel"], labelpad=10)
    if hline is not None:
        if not hasattr(hline, "__len__"):
            hline = [hline]
        for hl in hline:
            plt.axhline(
                hl, color="black", ls="--", lw=plt.rcParams["axes.linewidth"], zorder=2
            )
    if vline is not None:
        if not hasattr(vline, "__len__"):
            vline = [vline]
        for vl in vline:
            plt.axvline(
                vl, color="black", ls="--", lw=plt.rcParams["axes.linewidth"], zorder=2
            )
    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)
    if ylabels is False:
        ax.set_yticklabels([])
    if same_scale and x_range is None and y_range is None:
        xmi, xma = ax.get_xlim()
        ymi, yma = ax.get_ylim()
        ra = (min(xmi, ymi), max(yma, xma))
        ax.set_xlim(ra)
        ax.set_xlim(ra)
    if title is not None:
        plt.text(
            0.5,
            1.03,
            title,
            horizontalalignment="center",
            fontsize=text_sizes["title"],
            transform=ax.transAxes,
        )
    ax = handle_grid(ax, vgrid, hgrid)
    if draw_unity_line:
        xmi, xma = ax.get_xlim()
        ymi, yma = ax.get_ylim()
        ra = (min(xmi, ymi), max(yma, xma))
        xs = numpy.linspace(ra[0], ra[1], num=10)
        chi2, p = scipy.stats.chisquare(y, x)
        print(
            "  chi2=%s chi2red=%s  p=%s"
            % (repr(chi2), repr(chi2 / float(len(x) - 1)), repr(p))
        )
        ax.plot(
            xs, xs, color=linecolor, linewidth=linewidth, linestyle=linestyle, zorder=20
        )
    plt.draw()
    if save != False and save is not None:
        if ".png" in save:
            transparent = True
        else:
            transparent = False
        figure.savefig(
            save,
            dpi=default_figure_sizes["dpi"],
            bbox_inches="tight",
            bbox_extra_artists=[ax, figure],
            transparent=transparent,
        )  # ,bbox_inches="tight" is such that everything is in the figure... even though margins are removed.
        # plt.savefig(save, dpi=plt.gcf().dpi)
    if show:
        plt.show(block=block)
    return figure


def point(
    xytuple_or_xytuple_list,
    axis_or_figure,
    marker="s",
    markersize=30,
    color=None,
    x_range=None,
    y_range=None,
    **kwargs
):
    """
    can add one or more higlighted point to existing plot
    **kargs are those of the scatter function
    can add yerr and xerr one per tuple
    """
    if not hasattr(xytuple_or_xytuple_list[0], "__len__"):
        xytuple_or_xytuple_list = [xytuple_or_xytuple_list]
    xs, ys = list(zip(*xytuple_or_xytuple_list))
    if color is not None:
        if "markerfacecolor" not in kwargs:
            kwargs["markerfacecolor"] = color
        if "markeredgecolor" not in kwargs:
            kwargs["markeredgecolor"] = color
        if "ecolor" not in kwargs:
            kwargs["ecolor"] = color
    if axis_or_figure is not None:
        if x_range is None:
            if hasattr(axis_or_figure, "get_xlim"):
                x_range = axis_or_figure.get_xlim()
            else:
                x_range = axis_or_figure.gca().get_xlim()
            # see if it needs adjustment to accomodate new point
            adj = False
            if min(xs) < x_range[0]:
                adj, x_range = True, (min(xs), x_range[1])
            if max(xs) > x_range[1]:
                adj, x_range = True, (x_range[0], max(xs))
            if adj:
                x_range = (
                    x_range[0] - 0.25 * (x_range[1] - x_range[0]),
                    x_range[1] + 0.25 * (x_range[1] - x_range[0]),
                )
        if y_range is None:
            if hasattr(axis_or_figure, "get_ylim"):
                y_range = axis_or_figure.get_ylim()
            else:
                y_range = axis_or_figure.gca().get_ylim()
            # see if it needs adjustment to accomodate new point
            adj = False
            if min(ys) < y_range[0]:
                adj, y_range = True, (min(ys), y_range[1])
            if max(ys) > y_range[1]:
                adj, y_range = True, (y_range[0], max(ys))
            if adj:
                y_range = (
                    y_range[0] - 0.25 * (y_range[1] - y_range[0]),
                    y_range[1] + 0.25 * (y_range[1] - y_range[0]),
                )
    f = scatter(
        xs,
        ys,
        figure=axis_or_figure,
        marker=marker,
        markersize=markersize,
        y_range=y_range,
        x_range=x_range,
        **kwargs
    )
    del kwargs
    return f


def apply_yfunct(funct, profile, yerr, ismulti=False):
    """
    mostly disused by apply_scale_function albeit error propagation is different
     (here is done canoncally  in apply_scale_function is done by applying the function
     to x +/- errx
    """
    if funct is None:
        return profile, yerr
    if ismulti:
        if yerr is None:
            nyerr = None
        else:
            nyerr = []
        np = []
        for j, prof in enumerate(profile):
            p1 = numpy.array(prof)
            if yerr[j] is None:
                np += [funct(p1)]
                nyerr += [None]
            else:
                p2, err2 = misc.propagate_error(funct, [p1], yerr[j])
                np += [p2]
                nyerr += [err2]
        return np, nyerr
    else:
        if yerr is None:
            return funct(profile), yerr
        else:
            return misc.propagate_error(funct, [profile], [yerr])


def apply_scale_function(funct, profile, yerr):
    """
    similar to apply_yfunct above but support asymmetric error bars (e.g. for logscales or for CI)
    return profie, yerr   yerr is always in the CI format ( ys-funct(low), funct(up)-ys ) whether error bars are or not asymmetric
    autodetects ismulti
    """
    if hasattr(profile[0], "__len__"):  # ismulti
        if yerr is None:
            yerr = [None] * len(profile)
        return numpy.array(
            [
                apply_scale_function(funct, prof, yerr[j])
                for j, prof in enumerate(profile)
            ]
        )
    if yerr is None:
        return funct(numpy.array(profile))
    if not hasattr(yerr[0], "__len__"):  # not a confidence interval
        low = numpy.array(profile) - numpy.array(yerr)
        up = numpy.array(profile) + numpy.array(yerr)
    else:  # a confidence interval
        low = numpy.array(profile) - numpy.array(yerr[0])
        up = numpy.array(profile) + numpy.array(yerr[1])
    ys = funct(numpy.array(profile))
    return ys, (ys - funct(low), funct(up) - ys)


def yerr_to_fill_errors(profile, yerr, ismulti=False, bar_list=None):
    """
    ytop,ybottom,yerr=yerr_to_fill_errors(profile,yerr,ismulti=False,bar_list=None)
    used to convert error bars in top and bottom interval to be used in
    plt.fill_between(x, ybottom, ytop ,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    """
    if ismulti:
        ytop, ybottom = [], []
        if yerr is None:
            ytop, ybottom = [None] * len(profile), [None] * len(profile)
            return ytop, ybottom, yerr
        for i, prof in enumerate(profile):
            if yerr[i] is None or (bar_list is not None and bar_list[i] == True):
                ytop += [None]
                ybottom += [None]
                continue  # yerr[i] was already None in case of non-bar plot
            elif len(yerr[i]) == len(prof) and not hasattr(
                yerr[i][0], "__len__"
            ):  # error bars not CI
                ytop += [numpy.array(prof) + numpy.array(yerr[i])]
                ybottom += [numpy.array(prof) - numpy.array(yerr[i])]
            elif len(yerr[i]) == 2 and len(yerr[i][0]) == len(prof):  # it is CI
                ytop += [numpy.array(prof) + numpy.array(yerr[i][1])]
                ybottom += [numpy.array(prof) - numpy.array(yerr[i][0])]
            else:
                raise Exception(
                    "**ERROR** in plotter.profile fill_error error bar format not recongnized!!\n"
                )
            yerr[i] = None  # set  to None so these are not plotted
    else:
        if bar_list is not None and type(bar_list) is bool and bar_list is True:
            return None, None, yerr
        if yerr is None:
            ytop = None
            ybottom = None
        elif len(yerr) == len(profile) and not hasattr(
            yerr[0], "__len__"
        ):  # error bars not CI
            ytop = numpy.array(profile) + numpy.array(yerr)
            ybottom = numpy.array(profile) - numpy.array(yerr)
        elif len(yerr) == 2 and len(yerr[0]) == len(profile):  # it is CI
            ytop = numpy.array(profile) + numpy.array(yerr[1])
            ybottom = numpy.array(profile) - numpy.array(yerr[0])
        else:
            raise Exception(
                "**ERROR** in plotter.profile fill_error error bar format not recongnized!!\n"
            )
        yerr = None  # set all to None so these are not plotted
    return ytop, ybottom, yerr


def profile(
    profile,
    x_values=None,
    yerr=None,
    xerr=None,
    bar=False,
    hbar=False,
    bar_sep=0.2,
    yfunct=None,
    smooth=None,
    label="",
    ls="-",
    linfit=False,
    marker="",
    xlabels=None,
    ylabels=None,
    upper_labels=None,
    xlabel=None,
    ylabel=None,
    title=None,
    title_correlation=False,
    xlabels_rotation="horizontal",
    vgrid=None,
    hgrid=None,
    frame=None,
    color=None,
    show=True,
    use_right_axis=False,
    color_axis=None,
    color_ticks=None,
    markerfacecolor="black",
    markeredgecolor="black",
    markeredgewidth=None,
    alpha=None,
    markersize=15,
    ecolor=None,
    hatch=None,
    fill_error=False,
    fill_error_alpha=0.2,
    draw_unity_line=False,
    value_labels=None,
    value_labels_kwargs={"value_labels_ypos": "top"},
    x_range=None,
    y_range=None,
    same_scale=False,
    linewidth=None,
    plot_legend=True,
    legend_location="best",
    upper_labels_rotation="horizontal",
    legend_size=None,
    ncol=None,
    figure=None,
    ax=None,
    figure_size=None,
    save=None,
    group_bar_sep=None,
    labels=None,
    point_labels=None,
    labels_offset=None,
    flag_labels_color=(1, 1, 1, 0.8),
    flag_labels_boxstyle=None,
    labels_fontcolor="black",
    flag_labels_size=None,
    flag_fit_outliers=False,
    avoid_scientific_notation=False,
    x_major_tick_every=None,
    y_major_tick_every=None,
    x_minor_tick_every=None,
    y_minor_tick_every=None,
    cbar_min_max=None,
    plot_cbar_if_given=True,
    cbar_labels=None,
    log_scale_x=False,
    log_scale_y=False,
    hline=None,
    hline_color="black",
    hline_style="-",
    vline=None,
    vline_color="black",
    vline_style="-",
    zorder=0,
    fit_linewidth=None,
    fit_linecolor=None,
    fit_linestyle="-",
    fit_alpha=None,
    fit_allow_extra_fraction=None,
    fit_ignore_yerr=False,
    dont_fit_plateau=False,
    prefer_noisy_fit=False,
    calculate_fit_CI=False,
    swapfit=False,
    fit_CI_from_resample=False,
    plot_fit_CI=False,
    fit_results=None,
    fit_global=False,
    local_params_indices=None,
    local_args_for_global=None,
    fit_label=None,
):
    """
        markeredgecolor can be used to adjust bar edge color
        color can also be a colormap such as for example plotter.plt.get_cmap('jet')
        yfunct can be given as a function,applied to both y values and yerr. This can be a smoothing or a log
        errorbar options : ecolor=None, elinewidth=None, capsize=default_error_bars['capsize'], capthick=default_error_bars['capthick'],
                   barsabove=False, lolims=False, uplims=False,
                   xlolims=False, xuplims=False, errorevery=1,
                   )
        linfit can be True for a lienar fit, and int number for a polynomial fit of that degree
         else a tuple or list :
            p_boundaries=None
            if len(linfit)==3 : fun,guessed_params,p_boundaries=linfit
            else : fun,guessed_params=linfit
    Maybe
        can plot colorbar with different color points like:
        f=plotter.profile( y, x ,marker='',ls='',point_labels=numpy.array(z), flag_labels_color=plotter.plt.get_cmap('jet') ) for various points in SINGLE profile
        to color MULTIPLE profile just give the z to label instead than point_labels and color instead of flag_labels
        f=plotter.profile( y, x ,label=numpy.array(z), color=plotter.plt.get_cmap('jet') )

        flag_fit_outliers can be True (defaults to 5%) or a number in 0, 100 to select the percent of outliers to label
            if labels is not given it will use the index of the data
        flag_labels_boxstyle='round,pad=0.01' so you can also adjust text padding 'square,pad=0.5'
    see help(misc.fit_function) for how to do global fits etc.
    """
    if fit_results is None:
        fit_results = {}
    if frame is None:
        frame = default_parameters["frame"]
    if hbar:
        bar = True
    if linewidth is None:
        if hasattr(bar, "__len__"):
            linewidth = [
                default_parameters["barlinewidth"]
                if b
                else default_parameters["linewidth"]
                for b in bar
            ]
        elif bar == True:
            linewidth = default_parameters["barlinewidth"]
        else:
            linewidth = default_parameters["linewidth"]
    if figure_size is None:
        if default_figure_sizes["default"] is None:
            if same_scale:
                figure_size = default_figure_sizes["scatter"]
            elif bar:
                figure_size = default_figure_sizes["bar"]
            else:
                figure_size = default_figure_sizes["profile"]
        else:
            figure_size = default_figure_sizes["default"]
    if default_figure_sizes["use_cm"]:
        figure_size = (cmToinch(figure_size[0]), cmToinch(figure_size[1]))
    if legend_size is None:
        legend_size = text_sizes["legend_size"]
    if fit_allow_extra_fraction is None:
        fit_allow_extra_fraction = default_parameters["fit_allow_extra_fraction"]
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    entries_ypos = None
    plot_cbar, scalarMap = False, None
    plt.rc("xtick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick", labelsize=text_sizes["xlabels"])
    if ls == "" and marker == "" and not (yerr is not None and xerr is not None):
        marker = "."
    if fit_alpha is None:
        fit_alpha = alpha
    if type(profile) is int or type(profile) is float:
        profile = [profile]
        if x_values is not None and not hasattr(x_values, "__len__"):
            x_values = [x_values]
        if xerr is not None and not hasattr(xerr, "__len__"):
            xerr = [xerr]
        if yerr is not None and not hasattr(yerr, "__len__"):
            yerr = [yerr]
    if type(profile) is dict or isinstance(profile, OrderedDict):
        xvals, profile = list(zip(*list(profile.items())))
        try:
            if all([is_number(v) for v in xvals]):
                x_values = xvals
            else:
                xlabels = xvals
        except Exception:
            xlabels = list(map(str, xvals))

    if isinstance(label, numpy.ndarray):
        label = list(label)
    if hasattr(profile[0], "__len__"):
        ismulti = True
        # if label!='' and label is not None and type(label) is not list and type(label) is not tuple : label=len(profile)*[label]
        if type(label) is not list and type(label) is not tuple:
            label = len(profile) * [label]
        n_points = len(profile[0])
        n_profiles = len(profile)
        # profile=numpy.array(profile)
        if yerr is None:
            yerr = [None] * len(profile)
        if xerr is None:
            xerr = [None] * len(profile)
        if (
            type(marker) is not list
            and type(marker) is not tuple
            and not isinstance(marker, cycle_list)
        ):
            marker = [marker] * len(profile)
        if (
            type(ls) is not list
            and type(ls) is not tuple
            and not isinstance(ls, cycle_list)
        ):
            ls = [ls] * len(profile)
        if type(alpha) is not list and type(alpha) is not tuple:
            alpha = [alpha] * len(profile)
        if type(fit_alpha) is not list and type(fit_alpha) is not tuple:
            fit_alpha = [fit_alpha] * len(profile)

        if x_values is None:
            x_values = [list(range(len(prof))) for prof in profile]
        elif type(x_values) is int or type(x_values) is float:
            x_values = [
                numpy.arange(x_values, x_values + len(prof)) for prof in profile
            ]
    else:
        ismulti = False
        n_profiles = 1
        if x_values is None:
            x_values = list(range(0, len(profile)))
        elif type(x_values) is int or type(x_values) is float:
            x_values = numpy.arange(x_values, x_values + len(profile))
    # assign default colors if None given
    if color is None:
        if not ismulti:
            if ls == "-":
                color = next(colors6)
        elif n_profiles > 6:
            color = palette20
        else:
            color = iworkpalette

    if ismulti and hasattr(x_values, "__len__") and not hasattr(x_values[0], "__len__"):
        x_values = [x_values] * len(profile)

    if (ismulti or bar) and "colormap" in str(
        type(color)
    ).lower():  # it is a colormap, convert to list with required colors
        if cbar_min_max is None:
            if n_profiles == 1 and bar:
                cbar_min_max = min(profile), max(profile)
            else:
                cbar_min_max = 0, n_profiles - 0.98
        cNorm = matplotlib.colors.Normalize(vmin=cbar_min_max[0], vmax=cbar_min_max[1])
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=color)
        if n_profiles == 1 and bar:
            color = [scalarMap.to_rgba(x) for x in profile]
        else:
            color = [scalarMap.to_rgba(idx) for idx in range(n_profiles)]
        # print ("DEB: plot_cbar_if_given",plot_cbar_if_given)
        if (
            label is not None
            and not (type(label) is bool and label == False)
            and len(label) == n_profiles
            and label[0] is not None
        ):
            if plot_cbar_if_given:
                try:
                    if cbar_labels is None:
                        if type(label[0]) is str:
                            cbar_labels = [
                                misc.get_numbers_from_string(str(l))[-1] for l in label
                            ]  # [-1] will raise exception if no number in str(l)
                        else:
                            cbar_labels = label  # maybe already int or float
                        print("plotter.profile label=", cbar_labels)
                    scalarMap._A = []
                    plot_cbar = True
                except Exception:
                    sys.stderr.write(
                        "Warn in plotter.profile failed processing colorbar ticks\n"
                    )
                    # print 'plotter.profile label=',label
                    sys.stderr.flush()
                    plot_cbar = False
                if plot_cbar:
                    label = len(profile) * [None]
            else:
                plot_cbar = False
            if not plot_cbar:
                ncol = max([1, len(label) // 6])

    if ncol is None:
        ncol = 1
    if type(markeredgecolor) is bool and markeredgecolor == True:
        markeredgecolor = color
    if type(markerfacecolor) is bool and markerfacecolor == True:
        markerfacecolor = color
    if type(fit_linecolor) is bool and fit_linecolor == True:
        fit_linecolor = color
    if ecolor is None:
        ecolor = markerfacecolor
    if ismulti and (
        type(color) is list or isinstance(color, cycle_list)
    ):  # this is used to color also each point in different ways, not only for multiple profiles
        if (
            not hasattr(markerfacecolor, "__len__")
            or type(markerfacecolor) is str
            or type(markerfacecolor) is tuple
        ):
            markerfacecolor = cycle_list([markerfacecolor] * len(color))
        if (
            not hasattr(markeredgecolor, "__len__")
            or type(markeredgecolor) is str
            or type(markeredgecolor) is tuple
        ):
            markeredgecolor = cycle_list([markeredgecolor] * len(color))
        if (
            not hasattr(ecolor, "__len__")
            or type(ecolor) is str
            or type(ecolor) is tuple
        ):
            ecolor = cycle_list([ecolor] * len(color))
    if ismulti and type(hatch) is not list and type(hatch) is not tuple:
        hatch = [hatch] * len(profile)
    if ismulti and type(linewidth) is not list and type(linewidth) is not tuple:
        linewidth = [linewidth] * len(profile)
    if ismulti and (
        not hasattr(fit_linecolor, "__len__")
        or type(fit_linecolor) is str
        or type(fit_linecolor) is tuple
    ):
        fit_linecolor = cycle_list([fit_linecolor] * len(profile))
    new_figure = False
    if figure is None:
        new_figure = True
        figure = plt.figure(figsize=figure_size)
    # if ax is None :pylab.figure(figure.number)

    if use_right_axis:
        if ax is not None:
            axL = ax
        else:
            axL = figure.gca()
        ax = axL.twinx()
        if color_axis is None:
            color_axis = True
        if color_ticks is None:
            color_ticks = True
    else:
        if ax is not None:
            ax = ax
        else:
            ax = figure.gca()
        if color_axis is None:
            color_axis = False
        if color_ticks is None:
            color_ticks = False

    if smooth is not None:
        yfunct = process_smoothk_to_fun(smooth)
    if type(yfunct) is int:
        smooth_per_side = int(yfunct)
        yfunct = lambda x: misc.smooth_profile(
            x, smooth_per_side=smooth_per_side
        )  # assumes it is a smoothing
    if hasattr(yfunct, "__call__"):
        profile, yerr = apply_scale_function(
            yfunct, profile, yerr=yerr
        )  # ,ismulti=ismulti)

    if (
        fill_error and not bar
    ):  # determine ytop and ybottom depending on error given and then set yerr to all None
        yerr = yerr[:]
        ytop, ybottom, yerr = yerr_to_fill_errors(
            profile, yerr, ismulti=ismulti, bar_list=None
        )

    line_styles = []
    if bar:
        if x_range is None:
            xrm, xrM = get_min_max_glob(x_values)
            x_range = (xrm - 0.5, xrM + 0.5)
        bar_mid = []  # also for multi profiles
        if ismulti:
            if len(x_values[0]) <= 1:
                xsep = 3 * bar_sep
            else:
                xsep = numpy.mean(
                    numpy.abs(numpy.diff(x_values[0]))
                )  # spearation between to xvalues (center of bar group)
            sep, bar_width = (
                float(bar_sep) / (n_profiles + 1),
                float(xsep - bar_sep) / n_profiles,
            )
            if group_bar_sep is not None:
                gr_sep = group_bar_sep / 2.0
                sep, bar_width = (
                    float(bar_sep) / (n_profiles + 1),
                    float(xsep - gr_sep) / n_profiles,
                )
            else:
                gr_sep = sep
            if bar_width < 0:
                sys.stderr.write(
                    "*Warning* in profile() with bar_sep=%lf bar_width becomes negative (%lf xsep=%lf),"
                    % (bar_sep, bar_width, xsep)
                )
                aa = numpy.mean(numpy.diff(x_values[0]))
                sep, bar_width = 0.2 * aa, 0.8 * aa
                sys.stderr.write(
                    "setting to %lf (sep) and %lf (width)\n" % (sep, bar_width)
                )
            for i, prof in enumerate(profile):

                left = (
                    numpy.array(x_values[i])
                    - 0.5 * xsep
                    + gr_sep
                    + i * (sep + bar_width)
                )  # +gr_sep is accounted twice at end and beginning.. as we want group_bar_sep separation between clusters of bars
                if (type(color) is list or isinstance(color, cycle_list)) or (
                    type(color) is not tuple
                    and type(color) is not str
                    and color is not None
                ):
                    if hbar:
                        l = ax.barh(
                            left,
                            prof,
                            height=bar_width,
                            align="edge",
                            xerr=xerr[i],
                            label=label[i],
                            linewidth=linewidth[i],
                            edgecolor=markeredgecolor[i],
                            color=color[i],
                            ecolor=ecolor[i],
                            hatch=hatch[i],
                            zorder=zorder,
                            alpha=alpha[i],
                        )
                    else:
                        l = ax.bar(
                            left,
                            height=prof,
                            width=bar_width,
                            align="edge",
                            yerr=yerr[i],
                            label=label[i],
                            linewidth=linewidth[i],
                            edgecolor=markeredgecolor[i],
                            color=color[i],
                            ecolor=ecolor[i],
                            hatch=hatch[i],
                            zorder=zorder,
                            alpha=alpha[i],
                        )
                else:
                    if hbar:
                        l = ax.barh(
                            left,
                            prof,
                            height=bar_width,
                            align="edge",
                            xerr=xerr[i],
                            label=label[i],
                            linewidth=linewidth[i],
                            edgecolor=markeredgecolor,
                            color=color,
                            hatch=hatch[i],
                            ecolor=ecolor,
                            zorder=zorder,
                            alpha=alpha[i],
                        )
                    else:
                        l = ax.bar(
                            left,
                            height=prof,
                            width=bar_width,
                            align="edge",
                            yerr=yerr[i],
                            label=label[i],
                            linewidth=linewidth[i],
                            edgecolor=markeredgecolor,
                            color=color,
                            hatch=hatch[i],
                            ecolor=ecolor,
                            zorder=zorder,
                            alpha=alpha[i],
                        )
                line_styles += [l]
                bar_mid += [left + bar_width / 2.0]
                del l
                # the following line does something only if linfit is not None and linfit!=False
                ax = add_fit(
                    ax,
                    x_values[i],
                    prof,
                    linfit,
                    yerr=yerr[i],
                    fit_ignore_yerr=fit_ignore_yerr,
                    fit_global=fit_global,
                    local_params_indices=local_params_indices,
                    swapfit=swapfit,
                    linewidth=fit_linewidth,
                    linecolor=fit_linecolor[i],
                    linestyle=fit_linestyle,
                    fit_allow_extra_fraction=fit_allow_extra_fraction,
                    dont_fit_plateau=dont_fit_plateau,
                    permute_CI=swapfit,
                    prefer_noisy_fit=prefer_noisy_fit,
                    calculate_CI=calculate_fit_CI,
                    do_data_resampling=fit_CI_from_resample,
                    plot_fit_CI=plot_fit_CI,
                    fit_results=fit_results,
                    label=fit_label,
                    alpha=fit_alpha[i],
                    zorder=zorder + 1,
                )
        else:
            if len(x_values) <= 1:
                xsep = 3 * bar_sep
            else:
                xsep = numpy.mean(numpy.abs(numpy.diff(x_values)))
            sep, bar_width = bar_sep, (xsep - bar_sep)
            if bar_width < 0:
                sys.stderr.write(
                    "*Warning* in profile() with bar_sep=%lf bar_width becomes negative (%lf xsep=%lf),"
                    % (bar_sep, bar_width, xsep)
                )
                sep, bar_width = 0.2 * xsep, 0.8 * xsep
                sys.stderr.write(
                    "setting to %lf (sep) and %lf (width)\n" % (sep, bar_width)
                )
            left = numpy.array(x_values) - (xsep - sep) / 2.0
            # print 'sep,bar_width,xsep',sep,bar_width,xsep,'\nleft:',left,'\nx_values:',x_values
            if hbar:
                l = ax.barh(
                    left,
                    profile,
                    height=bar_width,
                    align="edge",
                    xerr=xerr,
                    label=label,
                    linewidth=linewidth,
                    color=color,
                    edgecolor=markeredgecolor,
                    ecolor=ecolor,
                    hatch=hatch,
                    zorder=zorder,
                    alpha=alpha,
                )  # THese are not in bar. Overlay to plots to customize error bars, elinewidth=default_error_bars['elinewidth'], ecapsize=default_error_bars['capsize'])
            else:
                l = ax.bar(
                    left,
                    height=profile,
                    width=bar_width,
                    align="edge",
                    yerr=yerr,
                    label=label,
                    linewidth=linewidth,
                    color=color,
                    edgecolor=markeredgecolor,
                    ecolor=ecolor,
                    hatch=hatch,
                    zorder=zorder,
                    alpha=alpha,
                )  # THese are not in bar. Overlay to plots to customize error bars, elinewidth=default_error_bars['elinewidth'], ecapsize=default_error_bars['capsize'])
            line_styles += [l]
            bar_mid = x_values
            del l
            # the following line does something only if linfit is not None and linfit!=False
            ax = add_fit(
                ax,
                x_values,
                profile,
                linfit,
                yerr=yerr,
                fit_ignore_yerr=fit_ignore_yerr,
                fit_global=fit_global,
                local_params_indices=local_params_indices,
                swapfit=swapfit,
                linewidth=fit_linewidth,
                linecolor=fit_linecolor,
                linestyle=fit_linestyle,
                fit_allow_extra_fraction=fit_allow_extra_fraction,
                dont_fit_plateau=dont_fit_plateau,
                permute_CI=swapfit,
                prefer_noisy_fit=prefer_noisy_fit,
                calculate_CI=calculate_fit_CI,
                do_data_resampling=fit_CI_from_resample,
                plot_fit_CI=plot_fit_CI,
                fit_results=fit_results,
                label=fit_label,
                alpha=fit_alpha,
                zorder=zorder + 1,
            )
    else:
        if ismulti:
            for i, prof in enumerate(profile):
                if len(prof) != len(x_values[i]):
                    sys.stderr.write(
                        "WARNING in profile len(prof)!=len(x_values[i]) while plotting multiple profiles (%d!=%d). Cropping end of xvalues.\n"
                        % (len(prof), len(x_values[i]))
                    )

                if color is None:
                    l = ax.errorbar(
                        x_values[i][: len(prof)],
                        prof,
                        yerr=yerr[i],
                        label=label[i],
                        xerr=xerr[i],
                        ls=ls[i],
                        elinewidth=default_error_bars["elinewidth"],
                        capsize=default_error_bars["capsize"],
                        capthick=default_error_bars["capthick"],
                        errorevery=1,
                        marker=marker[i],
                        markersize=markersize,
                        markeredgecolor=markeredgecolor,
                        markeredgewidth=markeredgewidth,
                        figure=figure,
                        markerfacecolor=markerfacecolor,
                        linewidth=linewidth[i],
                        ecolor=ecolor,
                        zorder=zorder,
                        alpha=alpha[i],
                    )
                elif (
                    type(color) is list
                    or isinstance(color, cycle_list)
                    or (
                        type(color) is not tuple
                        and type(color) is not str
                        and color is not None
                    )
                ):
                    l = ax.errorbar(
                        x_values[i][: len(prof)],
                        prof,
                        yerr=yerr[i],
                        label=label[i],
                        xerr=xerr[i],
                        ls=ls[i],
                        elinewidth=default_error_bars["elinewidth"],
                        capsize=default_error_bars["capsize"],
                        capthick=default_error_bars["capthick"],
                        errorevery=1,
                        marker=marker[i],
                        markersize=markersize,
                        markeredgecolor=markeredgecolor[i],
                        markeredgewidth=markeredgewidth,
                        figure=figure,
                        markerfacecolor=markerfacecolor[i],
                        linewidth=linewidth[i],
                        color=color[i],
                        ecolor=ecolor[i],
                        zorder=zorder,
                        alpha=alpha[i],
                    )
                else:
                    l = ax.errorbar(
                        x_values[i][: len(prof)],
                        prof,
                        yerr=yerr[i],
                        label=label[i],
                        xerr=xerr[i],
                        ls=ls[i],
                        elinewidth=default_error_bars["elinewidth"],
                        capsize=default_error_bars["capsize"],
                        capthick=default_error_bars["capthick"],
                        errorevery=1,
                        marker=marker[i],
                        markersize=markersize,
                        markeredgecolor=markeredgecolor,
                        markeredgewidth=markeredgewidth,
                        figure=figure,
                        markerfacecolor=markerfacecolor,
                        linewidth=linewidth[i],
                        color=color,
                        ecolor=ecolor,
                        zorder=zorder,
                        alpha=alpha[i],
                    )
                if fill_error and ybottom[i] is not None:
                    ax.fill_between(
                        x_values[i][: len(prof)],
                        ybottom[i],
                        ytop[i],
                        alpha=fill_error_alpha,
                        color=l.lines[0].get_color(),
                        interpolate=False,
                    )  # edgecolor='#CC4F1B', facecolor='#FF9848')
                line_styles += [l]
                del l
                # the following line does something only if linfit is not None and linfit!=False
                ax = add_fit(
                    ax,
                    x_values[i][: len(prof)],
                    prof,
                    linfit,
                    yerr=yerr[i],
                    fit_ignore_yerr=fit_ignore_yerr,
                    fit_global=fit_global,
                    local_params_indices=local_params_indices,
                    swapfit=swapfit,
                    linewidth=fit_linewidth,
                    linecolor=fit_linecolor[i],
                    linestyle=fit_linestyle,
                    fit_allow_extra_fraction=fit_allow_extra_fraction,
                    dont_fit_plateau=dont_fit_plateau,
                    permute_CI=swapfit,
                    prefer_noisy_fit=prefer_noisy_fit,
                    calculate_CI=calculate_fit_CI,
                    do_data_resampling=fit_CI_from_resample,
                    plot_fit_CI=plot_fit_CI,
                    fit_results=fit_results,
                    label=fit_label,
                    alpha=fit_alpha[i],
                    zorder=zorder + 1,
                )
        else:
            if color is None:
                l = ax.errorbar(
                    x_values,
                    profile,
                    yerr=yerr,
                    xerr=xerr,
                    label=label,
                    ls=ls,
                    elinewidth=default_error_bars["elinewidth"],
                    capsize=default_error_bars["capsize"],
                    capthick=default_error_bars["capthick"],
                    errorevery=1,
                    marker=marker,
                    markersize=markersize,
                    markeredgecolor=markeredgecolor,
                    markeredgewidth=markeredgewidth,
                    figure=figure,
                    markerfacecolor=markerfacecolor,
                    linewidth=linewidth,
                    ecolor=ecolor,
                    zorder=zorder,
                    alpha=alpha,
                )
            else:
                l = ax.errorbar(
                    x_values,
                    profile,
                    yerr=yerr,
                    xerr=xerr,
                    label=label,
                    ls=ls,
                    elinewidth=default_error_bars["elinewidth"],
                    capsize=default_error_bars["capsize"],
                    capthick=default_error_bars["capthick"],
                    errorevery=1,
                    marker=marker,
                    markersize=markersize,
                    markeredgecolor=markeredgecolor,
                    markeredgewidth=markeredgewidth,
                    figure=figure,
                    markerfacecolor=markerfacecolor,
                    linewidth=linewidth,
                    color=color,
                    ecolor=ecolor,
                    zorder=zorder,
                    alpha=alpha,
                )
            if fill_error and ybottom is not None:
                ax.fill_between(
                    x_values,
                    ybottom,
                    ytop,
                    alpha=fill_error_alpha,
                    color=l.lines[0].get_color(),
                    interpolate=False,
                )
            line_styles += [l]
            del l
            # the following line does something only if linfit is not None and linfit!=False
            ax = add_fit(
                ax,
                x_values,
                profile,
                linfit,
                yerr=yerr,
                fit_ignore_yerr=fit_ignore_yerr,
                fit_global=fit_global,
                local_params_indices=local_params_indices,
                swapfit=swapfit,
                linewidth=fit_linewidth,
                linecolor=fit_linecolor,
                linestyle=fit_linestyle,
                fit_allow_extra_fraction=fit_allow_extra_fraction,
                dont_fit_plateau=dont_fit_plateau,
                permute_CI=swapfit,
                prefer_noisy_fit=prefer_noisy_fit,
                calculate_CI=calculate_fit_CI,
                do_data_resampling=fit_CI_from_resample,
                plot_fit_CI=plot_fit_CI,
                fit_results=fit_results,
                label=fit_label,
                alpha=fit_alpha,
                zorder=zorder + 1,
            )

        # add flag labels or point_labels if not None (not implemented for ismulti
        if type(markeredgecolor) is tuple or type(markeredgecolor) is str:
            ec = markeredgecolor
        else:
            ec = "black"
        (
            x_range,
            y_range,
            plot_cbar,
            N_uniq_points,
            cbar_labels,
            scalarMap,
        ) = handle_flag_labels(
            ax,
            x_values,
            profile,
            labels,
            point_labels=point_labels,
            edgecolor=ec,
            ismulti=ismulti,
            fontcolor=labels_fontcolor,
            labels_offset=labels_offset,
            flag_labels_color=flag_labels_color,
            labels_size=flag_labels_size,
            markersize=markersize,
            x_range=x_range,
            y_range=y_range,
            plot_cbar=plot_cbar,
            scalarMap=scalarMap,
            cbar_labels=cbar_labels,
            zorder=zorder,
            boxstyle=flag_labels_boxstyle,
            labels_alpha=alpha,
            flag_fit_outliers=flag_fit_outliers,
            fit_tuple=linfit,
            fit_results=fit_results,
        )
    if fit_global:
        ax = add_fit(
            ax,
            x_values,
            numpy.array(profile),
            linfit,
            yerr=yerr,
            fit_global=False,
            local_params_indices=local_params_indices,
            local_args_for_global=local_args_for_global,
            swapfit=swapfit,
            linewidth=fit_linewidth,
            linecolor=fit_linecolor,
            linestyle=fit_linestyle,
            fit_allow_extra_fraction=fit_allow_extra_fraction,
            dont_fit_plateau=dont_fit_plateau,
            permute_CI=swapfit,
            prefer_noisy_fit=prefer_noisy_fit,
            calculate_CI=calculate_fit_CI,
            do_data_resampling=fit_CI_from_resample,
            plot_fit_CI=plot_fit_CI,
            fit_results=fit_results,
            label=fit_label,
            alpha=fit_alpha,
            zorder=zorder + 1,
        )
    # set also log scale:

    if (
        not ismulti and title_correlation and title is None
    ):  # works only when ismulti=False
        x, y = numpy.array(x_values), numpy.array(profile)
        nans = numpy.logical_or(numpy.isnan(x), numpy.isnan(y))
        R, p = scipy.stats.pearsonr(x[~nans], y[~nans])
        title = "R=%.2lf p=%g" % (R, Round_To_n(p, 1, only_decimals=True))
    if title is not None:
        ax.set_title(title, horizontalalignment="center", fontsize=text_sizes["title"])
        # ax.text(0.5, 1.04, title,horizontalalignment='center',fontsize=text_sizes['title'],transform = ax.transAxes) # maybe fancy but won't save in bbox='tight'

    if hbar:
        entries_ypos = x_values
        x_values = None

    # Must be before the range for log scales
    ax = handle_ticks(
        ax,
        x_major_tick_every,
        y_major_tick_every,
        x_minor_tick_every,
        y_minor_tick_every,
        log_scale_x=log_scale_x,
        log_scale_y=log_scale_y,
        new_figure=new_figure,
        xlabels=xlabels,
        ylabels=ylabels,
        entries_xpos=x_values,
        entries_ypos=entries_ypos,
        use_right_axis=use_right_axis,
        xlabels_rotation=xlabels_rotation,
    )
    ax = handle_grid(ax, vgrid, hgrid)

    if x_range is not None:
        ax.set_xlim(x_range)
    elif same_scale and x_range is None and y_range is None:
        xmi, xma = ax.get_xlim()
        ymi, yma = ax.get_ylim()
        ra = (min(xmi, ymi), max(yma, xma))
        ax.set_xlim(ra)
        ax.set_ylim(ra)
    elif not log_scale_x and not hbar:
        Min, Max = get_min_max_glob(x_values)
        if Min < Max:
            if bar:
                d = n_profiles * bar_width / 2.0 + 0.015 * (Max - Min)
            else:
                d = 0.015 * (Max - Min)
            ax.set_xlim((Min - d, Max + d))
    if y_range is not None:
        ax.set_ylim(y_range)
    elif hbar:
        Min, Max = get_min_max_glob(entries_ypos)
        d = n_profiles * bar_width / 2.0 + 0.015 * (Max - Min)
        ax.set_ylim((Min - d, Max + d))

    # ax= handle_ticks(ax, x_major_tick_every, y_major_tick_every, x_minor_tick_every, y_minor_tick_every,log_scale_x=log_scale_x,log_scale_y=log_scale_y,new_figure=new_figure ,xlabels=xlabels,ylabels=ylabels, entries_xpos=x_values ,entries_ypos=entries_ypos,use_right_axis=use_right_axis,xlabels_rotation=xlabels_rotation)
    # ax=handle_grid( ax , vgrid , hgrid )

    if hline is not None:
        if not hasattr(hline, "__len__"):
            hline = [hline]
        for hl in hline:
            ax.axhline(
                hl,
                color=hline_color,
                ls=hline_style,
                lw=plt.rcParams["axes.linewidth"],
                zorder=zorder - 2,
            )
    if vline is not None:
        if not hasattr(vline, "__len__"):
            vline = [vline]
        for vl in vline:
            ax.axvline(
                vl,
                color=vline_color,
                ls=vline_style,
                lw=plt.rcParams["axes.linewidth"],
                zorder=zorder - 2,
            )
    if upper_labels is not None:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        if ismulti :
            ax2.set_xticks(x_values[0])
        else :
            ax2.set_xticks(x_values)
        ax2.set_xticklabels(
            upper_labels,
            rotation=upper_labels_rotation,
            verticalalignment="bottom",
            fontsize=text_sizes["xlabels"],
        )
        plt.sca(ax)  # set back current axis if needed outside function

    if value_labels is not None:
        if bar:
            handle_value_labels(
                value_labels,
                bar_mid,
                profile,
                yerr=yerr,
                ismulti=ismulti,
                zorder=zorder + 1,
                **value_labels_kwargs
            )
        else:
            handle_value_labels(
                value_labels,
                x_values,
                profile,
                yerr=yerr,
                ismulti=ismulti,
                zorder=zorder + 1,
                **value_labels_kwargs
            )
    # fig.tight_layout(pad=3.5,h_pad=1.08,w_pad=1.08,rect=(0, 0, 1, 1))

    if color_axis:
        if type(color_axis) is bool:
            if color is None:
                axcolor = markerfacecolor
            elif type(color) is list or isinstance(color, cycle_list):
                axcolor = color[0]
            else:
                axcolor = color
        else:
            axcolor = color_axis
        ax.tick_params(
            axis="y", which="both", colors=axcolor
        )  # which is both minor and major
        if use_right_axis:
            ax.spines["right"].set_color(axcolor)
        else:
            ax.spines["left"].set_color(axcolor)
        print("axcolor,", axcolor, use_right_axis)
    if color_ticks:
        if type(color_ticks) is bool:
            if color is None:
                txcolor = markerfacecolor
            elif type(color) is list or isinstance(color, cycle_list):
                txcolor = color[0]
            else:
                txcolor = color
        else:
            txcolor = color_ticks
        for tl in ax.get_yticklabels():  # change color of all tick labels
            tl.set_color(txcolor)
        ax.yaxis.label.set_color(txcolor)  # change color of eventual ylabel
    # this remove top and right axis
    if not frame:
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()  # ticks only on bottom axis
        if use_right_axis:
            axL.spines["left"].set_visible(False)
            axL.set_yticks([])
            axL.spines["top"].set_visible(False)
            axL.get_xaxis().tick_bottom()  # ticks only on bottom axis
            ax.spines["left"].set_visible(False)
            ax.get_yaxis().tick_right()  # ticks only on left axis
        else:
            ax.spines["right"].set_visible(False)
            ax.get_yaxis().tick_left()  # ticks only on left axis

    if (
        not log_scale_y
        and y_major_tick_every != False
        and ylabels is None
        and not avoid_scientific_notation
    ):
        ax, add_to_axis_label = consider_scientific_notation(
            ax,
            axis="y",
            publication=default_parameters["set_publish"],
            fontsize=text_sizes["xlabels"],
        )  # at the moment we use all_tight as an alias for publication quality y/n
        if ylabel is not None and type(ylabel) is str:
            ylabel += add_to_axis_label
    # print('DEB: x_major_tick_every',x_major_tick_every,'xlabels',xlabels,avoid_scientific_notation,'log_scale_x',log_scale_x)
    if (
        not log_scale_x
        and (type(x_major_tick_every) is not bool or x_major_tick_every != False)
        and xlabels is None
        and not avoid_scientific_notation
    ):
        ax, add_to_axis_label = consider_scientific_notation(
            ax,
            axis="x",
            publication=default_parameters["set_publish"],
            fontsize=text_sizes["xlabels"],
        )  # at the moment we use all_tight as an alias for publication quality y/n
        if xlabel is not None and type(xlabel) is str:
            xlabel += add_to_axis_label

    if xlabel is not None and type(xlabel) is str:
        ax.set_xlabel(xlabel, fontsize=text_sizes["xlabel"], labelpad=10)
    if ylabel is not None and type(ylabel) is str:
        ax.set_ylabel(ylabel, fontsize=text_sizes["ylabel"], labelpad=10)

    if draw_unity_line:
        xmi, xma = ax.get_xlim()
        ymi, yma = ax.get_ylim()
        ra = (min(xmi, ymi), max(yma, xma))
        xs = numpy.linspace(ra[0], ra[1], num=400)
        if fit_linewidth is None:
            fit_linewidth = default_parameters["linewidth"]
        l_linecolor = "black"
        ax.plot(
            xs,
            xs,
            color=l_linecolor,
            linewidth=fit_linewidth,
            linestyle=fit_linestyle,
            zorder=0,
        )

    if (
        plot_legend
        and not plot_cbar
        and (
            (label != "" and label is not None)
            or (fit_label is not None and fit_label != "")
        )
    ):
        # if ismulti :
        #    if type(label) is not list and type(label) is not tuple : label=[label]
        #    figure.legend(line_styles, label, loc=legend_location,prop={'size':legend_size})
        # else :
        plt.legend(
            loc=legend_location,
            frameon=False,
            prop={"size": legend_size},
            ncol=ncol,
            borderpad=0,
        )
    elif plot_cbar_if_given and plot_cbar:
        if N_uniq_points is None:
            N_uniq_points = n_profiles
        Nlabels_minus1or2 = 4
        nj = N_uniq_points // Nlabels_minus1or2
        if nj <= 0:
            nj = 1
        inds_labels_to_plot = [j * nj for j in range(Nlabels_minus1or2 + 1)]
        if N_uniq_points - inds_labels_to_plot[-1] > nj / 2:
            inds_labels_to_plot += [N_uniq_points - 1]  # add last
        else:
            inds_labels_to_plot[-1] = N_uniq_points - 1  # replace last

        if cbar_labels is None:
            cbarlabels = None
        else:
            cbarlabels = [
                cbar_labels[jj] for jj in inds_labels_to_plot
            ]  # [ c if j in range(0,N_uniq_points,jump) else '' for j,c in enumerate(cbar_labels)]
            # cbarlabels[-1]=cbar_labels[-1]
        # for j in xrange(jump-1) :cbarlabels[-2-j]=''
        # cbar_major_ntick=None
        add_colorbar_to_seq_profile(
            figure,
            scalarMap,
            cbar_major_ntick=inds_labels_to_plot,
            cbar_labels=cbarlabels,
        )

    plt.draw()
    if save is not None and save != "":
        if "." not in save:
            save += ".png"
        plt.savefig(
            save,
            dpi=default_figure_sizes["dpi"],
            bbox_inches="tight",
            bbox_extra_artists=[ax, figure],
            transparent=True,
        )  # ,bbox_inches="tight" is such that everything is in the figure... even though margins are removed.
    if show:
        plt.show(block=False)
    return figure


def ROC_curve(true_classification, predicted_value, **kwargs):
    """
    plots a ROC curve for a binary classifier and computes the area under it
     kwargs are the same as the function profile
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(true_classification, predicted_value)
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)
    sq_dist = (fpr) ** 2 + (
        1 - tpr
    ) ** 2  # distance of the ROC curve and the point of ideal prediction (0,1)
    ind = sq_dist.argmin()
    print(
        "  Best threshold: %f for TPR %f FPR %f" % (thresholds[ind], tpr[ind], fpr[ind])
    )
    figure = profile(tpr, x_values=fpr, same_scale=True, **kwargs)
    return fpr, tpr, thresholds, roc_auc, figure


def add_fit(
    ax,
    x,
    y,
    linfit,
    yerr=None,
    xerr=None,
    fit_global=False,
    local_params_indices=None,
    local_args_for_global=None,
    fit_ignore_yerr=False,
    swapfit=None,
    linewidth=None,
    linecolor=None,
    alpha=None,
    linestyle="-",
    prefer_noisy_fit=False,
    dont_fit_plateau=False,
    fit_allow_extra_fraction=None,
    calculate_CI=False,
    plot_fit_CI=True,
    permute_CI=False,
    plot_filled_ci=True,
    use_new_rather_than_resample=True,
    do_data_resampling=False,
    label=None,
    fit_results=None,
    zorder=1,
):
    """
    linfit can be a tuple such that: fitting_function,guessed_params,p_boundaries=linfit . Where p_boundaries are the boundaries
      in which parameters are allowde to vary, one can also not give this argument and just give fitting_function,guessed_params=linfit
    see help(misc.fit_function) for help on advanced options.
    fit_allow_extra_fraction can be a fraction of extra points to add consider at both sides of the data range to define the fit x_range (fraction of the range not number of points)
     or a tuple that specifies the x_range for the fit
    """
    if fit_results is None:
        fit_results = {}
    if (
        fit_global is True
    ):  # when is_global add_fit is called for every profile, and then one last time with all profiles and fit_global set to False
        return ax
    print_results = False
    if fit_allow_extra_fraction is None:
        fit_allow_extra_fraction = default_parameters["fit_allow_extra_fraction"]
    if type(fit_results) is str:
        fit_results = {}
        par_names = None
        print_results = True
    elif type(fit_results) is list:
        par_names = fit_results
        print_results = True
        fit_results = {}
    if prefer_noisy_fit:
        calculate_CI = False
    if fit_ignore_yerr:
        yerr = None
        xerr = None
    if linfit is False or linfit is None:
        return ax
    elif (type(linfit) is str and linfit.lower() in ["interpolate", "trendline"]) or (
        type(linfit) is tuple
        and type(linfit[0]) is str
        and linfit[0].lower() in ["interpolate", "trendline"]
    ):
        k = 2
        s = 3.0 * len(y)
        if type(linfit) is tuple:
            if len(linfit) == 2:
                k = linfit[1]
            elif len(linfit) > 2:
                k, s = linfit[1:]
        print("DEB: interpolating with k,s", k, s)
        SPL, xs, ys = misc.interpolate(
            x, y, k=k, s=s, yerr=yerr, plot=False, pre_smooth=False
        )
        ax.plot(
            xs,
            ys,
            color=linecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )
        print_results = False
        fit_results["interpolation"] = SPL
        return ax
    extra_str = ""
    if linecolor is None:
        linecolor = next(colors6)
    if linewidth is None:
        linewidth = plt.rcParams["lines.linewidth"]
    if linfit is True or type(linfit) is int:
        if (
            hasattr(fit_allow_extra_fraction, "__len__")
            and len(fit_allow_extra_fraction) == 2
        ):  # like x_range for the fit
            xs = numpy.linspace(
                fit_allow_extra_fraction[0], fit_allow_extra_fraction[1], num=400
            )
        else:
            m, M = numpy.nanmin(x), numpy.nanmax(x)
            a = fit_allow_extra_fraction * (M - m)
            xs = numpy.linspace(m - a, M + a, num=400)
    if linfit == True or linfit == 1:
        r_pears, pvalue = scipy.stats.pearsonr(x, y)
        if swapfit == "both":
            swapped_coefficients, pol_fun, R2 = polfit(x, y, pol_order=1, swap=True)
            print(
                "Swapped linfit coefficients: "
                + str(swapped_coefficients)
                + " R^2= "
                + str(R2)
                + " pol: "
                + str(pol_fun)[2:]
            )
            if label is False or label is None:
                lab2 = None
            elif label == "" or (label == True and type(label) is bool):
                lab2 = str(pol_fun)[2:] + "\nR^2= %4.2lf R=%4.2lf P=%g" % (
                    R2,
                    r_pears,
                    Round_To_n(pvalue, 2),
                )
            else:
                lab2 = label
            ys = pol_fun(xs)
            ax.plot(
                xs,
                ys,
                label=lab2,
                color=linecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=zorder,
                alpha=alpha,
            )
            swapfit = False
        coefficients, pol_fun, R2 = polfit(x, y, pol_order=1, swap=swapfit)
        fit_results["parameters_fitted"] = coefficients
        fit_results["R2"] = R2
        fit_results["pol_fun"] = pol_fun
        print(
            "linfit coefficients: "
            + str(coefficients)
            + " R^2= "
            + str(R2)
            + " pol: "
            + str(pol_fun)[2:]
        )
        if label is False or label is None:
            lab2 = None
        elif label == "" or (label == True and type(label) is bool):
            lab2 = str(pol_fun)[2:] + "\nR^2= %4.2lf R=%4.2lf P=%g" % (
                R2,
                r_pears,
                Round_To_n(pvalue, 2),
            )
        else:
            lab2 = label
        # print 'DEB lab2',repr(str(lab2))
        ys = pol_fun(xs)
        if linestyle != "":
            ax.plot(
                xs,
                ys,
                label=lab2,
                color=linecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
            )
    elif type(linfit) is int:
        coefficients, pol_fun, R2 = polfit(x, y, pol_order=linfit)
        fit_results["parameters_fitted"] = coefficients
        fit_results["R2"] = R2
        fit_results["pol_fun"] = pol_fun
        print(
            "Polfit"
            + str(linfit)
            + " coefficients: "
            + str(coefficients)
            + " pol: "
            + str(pol_fun)[2:]
            + " R^2= "
            + str(R2)
        )
        if label is False or label is None:
            lab2 = None
        elif label == "" or (label == True and type(label) is bool):
            lab2 = str(pol_fun)[2:] + " R^2= %4.2lf" % (R2)
        else:
            lab2 = label
        ys = pol_fun(xs)
        if linestyle != "":
            ax.plot(
                xs,
                ys,
                label=lab2,
                color=linecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder,
            )
    elif type(linfit) is tuple or type(linfit) is list:
        p_boundaries = None
        if len(linfit) == 3:
            fun, guessed_params, p_boundaries = linfit
        else:
            fun, guessed_params = linfit
        if not hasattr(fun, "__call__"):
            raise Exception(
                "In add_fit given tuple or list but first object is not callable\n"
            )
        if not prefer_noisy_fit and use_new_rather_than_resample:
            if calculate_CI == True:
                calculate_CI = 1000
            fitter = misc.FitNew(
                fun,
                parameters_guess=guessed_params,
                parameters_boundaries=p_boundaries,
                bootstrap_cycles=calculate_CI,
                do_data_resampling=do_data_resampling,
                dont_fit_plateau=dont_fit_plateau,
                local_params_indices=local_params_indices,
                local_args_for_global=local_args_for_global,
                permute_ci=False,
            )
            if calculate_CI:
                (
                    popt_leastsq_guess,
                    perrors_leastsq,
                    parameters_fitted,
                    parameters_standard_errors,
                    popt_dw,
                    popt_up,
                    _,
                    chi2red,
                    sum_of_squared_residuals,
                ) = fitter(x, y, yerr=yerr, xerr=xerr)
                fit_results["popt_leastsq_guess"] = popt_leastsq_guess
                fit_results["perrors_leastsq"] = perrors_leastsq
                fit_results["parameters_lower_CI"] = popt_dw
                fit_results["parameters_upper_CI"] = popt_up
                if plot_fit_CI:
                    xss, y_pl_dw, y_pl_up = fitter.get_plotCI_x_y(
                        x, fit_allow_extra_fraction=fit_allow_extra_fraction
                    )  # gets CI from percentile of ys obtained with parameter ensemble
                    if plot_filled_ci:
                        # print 'y_pl_up-y_pl_dw=',y_pl_up-y_pl_dw
                        # print 'linecolor=',linecolor
                        plot_filled_profile(
                            xss,
                            y_pl_dw,
                            y_pl_up,
                            figure=ax.figure,
                            axes=ax,
                            alpha=0.2,
                            color=linecolor,
                            interpolate=False,
                            zorder=-2,
                        )
                    else:
                        profile(
                            y_pl_dw,
                            xss,
                            label=None,
                            color=linecolor,
                            linewidth=linewidth,
                            ls="--",
                            alpha=alpha,
                            ax=ax,
                            figure=ax.figure,
                        )
                        profile(
                            y_pl_up,
                            xss,
                            label=None,
                            color=linecolor,
                            linewidth=linewidth,
                            ls="--",
                            alpha=alpha,
                            ax=ax,
                            figure=ax.figure,
                        )
                    # ax.plot(xss, y_pl_dw,label=None, color=linecolor,linewidth=linewidth, linestyle='--',alpha=alpha)
                    # ax.plot(xss,y_pl_up,label=None, color=linecolor,linewidth=linewidth, linestyle='--',alpha=alpha)
            else:
                (
                    parameters_fitted,
                    parameters_standard_errors,
                    chi2red,
                    sum_of_squared_residuals,
                ) = fitter(x, y, yerr=yerr, xerr=xerr)
            fit_results["parameters_fitted"] = parameters_fitted
            fit_results["parameters_standard_errors"] = parameters_standard_errors
            fit_results["sum_of_squared_residuals"] = sum_of_squared_residuals
            fit_results["chi2red"] = chi2red
            if hasattr(y[0], "__len__"):
                SS_tot = (
                    (
                        numpy.array(misc.flatten(y))
                        - numpy.mean(numpy.array(misc.flatten(y)))
                    )
                    ** 2
                ).sum()
            else:
                SS_tot = ((numpy.array(y) - numpy.mean(numpy.array(y))) ** 2).sum()
            R_2 = 1 - sum_of_squared_residuals / SS_tot
            fit_results["R2"] = R_2
        else:
            if prefer_noisy_fit:
                if isinstance(prefer_noisy_fit, int):
                    fitter = misc.Fit_noisy(fun, log=None, nsteps=prefer_noisy_fit)
                else:
                    fitter = misc.Fit_noisy(fun, log=None)
            elif calculate_CI:
                fitter = misc.FitCI(fun, function_has_parameter_list=True)
            else:
                fitter = misc.Fit(fun)
            if (
                calculate_CI
            ):  # Assumes ``ydata = f(xdata, *params) + eps`` otherwise give function_has_parameter_list=True below (probably given by default)
                (
                    popt_leastsq_guess,
                    perrors_leastsq,
                    parameters_fitted,
                    parameters_standard_errors,
                    popt_dw,
                    popt_up,
                    fit_results_string,
                ) = fitter(
                    x,
                    y,
                    parameter_guess=guessed_params,
                    yerr=yerr,
                    p_boundaries=p_boundaries,
                    plot=False,
                )
                fit_results["popt_leastsq_guess"] = popt_leastsq_guess
                fit_results["perrors_leastsq"] = perrors_leastsq
                fit_results["parameters_lower_CI"] = popt_dw
                fit_results["parameters_upper_CI"] = popt_up
                if plot_fit_CI:
                    if permute_CI:
                        if len(popt_leastsq_guess) != 2:
                            sys.stderr.write(
                                "Error in FitCI() permute_ci is implemented for 2 parameters only! (leaving unchanged)"
                            )
                            sys.stderr.flush()
                        else:
                            tmp_dw = [popt_dw[0], popt_up[1]]  # swap them
                            popt_up = numpy.array([popt_up[0], popt_dw[1]])
                            popt_dw = numpy.array(tmp_dw)
                    _, y_pl_dw = fitter.get_plot_x_y(x, popt_dw)
                    xss, y_pl_up = fitter.get_plot_x_y(x, popt_up)
                    ax.plot(
                        xss,
                        y_pl_dw,
                        label=None,
                        color=linecolor,
                        linewidth=linewidth,
                        linestyle="--",
                        alpha=alpha,
                    )
                    ax.plot(
                        xss,
                        y_pl_up,
                        label=None,
                        color=linecolor,
                        linewidth=linewidth,
                        linestyle="--",
                        alpha=alpha,
                    )
            else:
                (
                    parameters_fitted,
                    y_fit,
                    chi2,
                    chi2_rid,
                    RMS_of_residuals,
                    parameters_standard_errors,
                    fit_results_string,
                ) = fitter(
                    x,
                    y,
                    guessed_params,
                    yerr=yerr,
                    p_boundaries=p_boundaries,
                    plot=False,
                )
                fit_results["y_fit"] = y_fit
                fit_results["chi2"] = chi2
                fit_results["RMS_of_residuals"] = RMS_of_residuals
            fit_results["parameters_fitted"] = parameters_fitted
            fit_results["parameters_standard_errors"] = parameters_standard_errors
            print(fit_results_string)
            if local_args_for_global is None:  # otherwise the call to fun will be wrong
                try:
                    SS_tot = ((y - numpy.mean(y)) ** 2).sum()
                    SS_res = ((y - fun(x, parameters_fitted)) ** 2).sum()
                    R_2 = 1 - SS_res / SS_tot
                    fit_results["R2"] = R_2
                except Exception:
                    print_exc(file=sys.stderr)
                    pass
        xs, ys = fitter.get_plot_x_y(
            x, num_points=3000, fit_allow_extra_fraction=fit_allow_extra_fraction
        )
        if label is False or label is None:
            lab2 = None
        elif label == True and type(label) is bool:
            lab2 = "p:%s" % (str(fit_results["parameters_fitted"]))
        else:
            lab2 = label
        if linestyle != "":
            if len(xs) != len(ys) and local_args_for_global is None:
                sys.stderr.write(
                    "**POTENTIAL ERROR** in add_fit len(xs)!=len(ys) %d %d fit_allow_extra_fraction=%s\n"
                    % (len(xs), len(ys), str(fit_allow_extra_fraction))
                )
            profile(
                ys,
                xs,
                label=lab2,
                color=linecolor,
                linewidth=linewidth,
                ls=linestyle,
                alpha=alpha,
                ax=ax,
                figure=ax.figure,
                zorder=zorder,
            )
        # ax.plot(xs, ys,label=lab2, color=linecolor,linewidth=linewidth, linestyle=linestyle,alpha=alpha)
    # elif type(linfit) is str and 'interpolate' in linfit.lower():
    #    SPL,xs,ys,candidate_max_inds,candidate_min_inds=misc.get_interpolated_steepest(x,y,yerr=yerr)
    #    fit_results['parameters_fitted']= numpy.hstack((candidate_max_inds,candidate_min_inds))
    #    if label is False or label is None : lab2=None
    #    elif label=='' or (label==True and type(label) is bool): lab2='x,y='+'; '.join(['%g,%g'%(xs[i],ys[i]) for i in sorted(numpy.hstack((candidate_max_inds,candidate_min_inds))) ])
    #    else : lab2=label
    #    print_results=True
    #    if print_results:
    #        par_names=['IndMax' for i in candidate_max_inds]+['IndMin' for i in candidate_min_inds]
    #        print "len(fit_results['parameters_fitted'])",len(fit_results['parameters_fitted']),len(par_names)
    #        extra_str='x,y=\t'+'; '.join(['%g,%g'%(xs[i],ys[i]) for i in numpy.hstack((candidate_max_inds,candidate_min_inds)) ])
    #    profile( ys, xs,label=lab2, color=linecolor,linewidth=linewidth, ls=linestyle,alpha=alpha,  ax=ax,figure=ax.figure ,zorder=zorder)
    #    if len(candidate_max_inds)>0 : point( zip( xs[candidate_max_inds], ys[candidate_max_inds]), ax, marker='d',markersize=42, markerfacecolor=linecolor,zorder=9)
    #    if len(candidate_min_inds)>0 : point( zip( xs[candidate_min_inds], ys[candidate_min_inds]), ax, marker='d',markersize=42, markerfacecolor=linecolor,zorder=10)
    if print_results:
        print_fit_results(fit_results, parameters_name=par_names, extra_str=extra_str)
        # sys.stdout.write("************************ FIT RESULTS ***********************\n")
        # sys.stdout.write(" parameters_fitted=%s\n" % (', '.join(map(repr,fit_results['parameters_fitted']))))
        # sys.stdout.write(" parameters_standard_errors=%s\n" % (', '.join(map(repr,fit_results['parameters_standard_errors']))))
        # sys.stdout.write("***********************************************************\n")
    return ax


def plot_filled_profile(
    x,
    y_dw,
    y_up,
    figure=None,
    axes=None,
    color=None,
    alpha=0.2,
    interpolate=False,
    **kw
):
    if figure is None:
        if axes is not None:
            figure = axes.figure
        else:
            figure = plt.figure(figsize=default_figure_sizes["profile"])
    if axes is None:
        axes = figure.gca()
    if hasattr(y_dw[0], "__len__"):
        if not hasattr(x[0], "__len__"):  # same x for all
            x = [x] * len(y_dw)
        if type(color) is str or (
            type(color) is tuple and len(color) in [3, 4]
        ):  # change color to one per profile
            color = [color] * len(y_dw)
        for j, y_pl_dw in enumerate(y_dw):
            collection = axes.fill_between(
                x[j],
                y_pl_dw,
                y_up[j],
                figure=figure,
                axes=axes,
                alpha=alpha,
                color=color[j],
                interpolate=interpolate,
                **kw
            )
    else:
        collection = axes.fill_between(
            x,
            y_dw,
            y_up,
            figure=figure,
            axes=axes,
            alpha=alpha,
            color=color,
            interpolate=interpolate,
            **kw
        )
    plt.draw()
    return collection


def pretty_print_line(
    mean,
    stderr,
    low_CI=None,
    up_CI=None,
    nsignificant=1,
    out=sys.stdout,
    add_at_beginning="",
    add_at_end="\n",
):
    """
    pretty print mean with signficant digits from error, can include CI in round brakets
    """
    pr, err = round_with_error(mean, stderr, nextra_significant_err=nsignificant - 1)
    ret = [pr, err]
    if low_CI is not None:
        lowCI, upCI = Round_To_n(
            low_CI, n=nsignificant, only_decimals=True
        ), Round_To_n(up_CI, n=nsignificant, only_decimals=True)
        if out is not None:
            out.write(
                add_at_beginning
                + ("%g +/- %g in (%g , %g)" % (pr, err, lowCI, upCI))
                + add_at_end
            )
        ret += [lowCI, upCI]
    elif out is not None:
        out.write(add_at_beginning + ("%g +/- %g" % (pr, err)) + add_at_end)
    return ret


def print_fit_results(
    fit_results,
    out=sys.stdout,
    parameters_name=None,
    return_data_class=False,
    local_params_indices=None,
    extra_str="",
):
    '''
    return name_dict
    OR
    return data_class  (when return_data_class=True)
    '''
    # a list with the corresponding parameters name or a dict whose keys are indices
    close = False
    name_dict = (
        {}
    )  # dict with keys that are parameters_name, values are lists [fitted_value, fitted_error,fitted_CI] latter is tuple (lowCI,upCI)
    if return_data_class:  # returns instead of name_dict
        data = csv_dict.Data()
        data.key_column_hd_name = "Parameter"
        data.hd = {}
    # or if no errors are calculated only values
    if type(out) is str:
        close = True
        out = open(out, "w")
    if parameters_name is None:
        parameters_name = [
            "[%d]" % j for j in range(0, len(fit_results["parameters_fitted"]))
        ]
    elif len(parameters_name) < len(fit_results["parameters_fitted"]):
        if local_params_indices is not None : # does big assumptions
            #parameters_name=parameters_name[:] # re allocate memory so hopefully won't add to input
            pg,pl = [],[]
            for j,p in enumerate(parameters_name) :
                if j not in local_params_indices : 
                    pg+=[p]
                else :
                    pl+=[ str(p) ]
            parameters_name= pg + (len(fit_results["parameters_fitted"])-len(pg))*pl
        else :
            sys.stderr.write(
                "\n**WARNING** in print_fit_results() len(parameters_name)<number of fitted parameters (%d and %d)\n Maybe it was global fit with some local paramters?\n"
                % (len(parameters_name), len(fit_results["parameters_fitted"]))
            )
            i = len(parameters_name)
            parameters_name=parameters_name[:] # re allocate memory so hopefully won't add to input
            while i < len(fit_results["parameters_fitted"]):
                parameters_name += ["[%d]" % i]
                i += 1
    if out is not None:
        out.write("************************ FIT RESULTS ***********************\n")
        out.write(
            " parameters_fitted: [%d parameters]\n"
            % (len(fit_results["parameters_fitted"]))
        )
    par_lines = []
    if (
        "parameters_lower_CI" in fit_results
        and "parameters_standard_errors" in fit_results
    ):
        par_hd = ["parameter", "mean", "error", "lowCI", "highCI"]
        for j, p in enumerate(fit_results["parameters_fitted"]):
            if (
                fit_results["parameters_standard_errors"][j] < p
                and fit_results["parameters_standard_errors"][j] > 0
            ):
                pr, err = round_with_error(
                    p, fit_results["parameters_standard_errors"][j]
                )
            else:
                pr, err = Round_To_n(p, n=1, only_decimals=True), Round_To_n(
                    fit_results["parameters_standard_errors"][j],
                    n=1,
                    only_decimals=True,
                )
            lowCI, upCI = Round_To_n(
                fit_results["parameters_lower_CI"][j], n=1, only_decimals=True
            ), Round_To_n(
                fit_results["parameters_upper_CI"][j], n=1, only_decimals=True
            )
            if out is not None:
                out.write(
                    " %s= %g +/- %g in (%g , %g)\n"
                    % (parameters_name[j], pr, err, lowCI, upCI)
                )
            name, jj = parameters_name[j], 1
            while name in name_dict:  # make it unique
                name = name + "%d" % (jj)
                jj += 1
            par_lines += [
                "%s\t%g\t%g\t%g\t%g"
                % (
                    str(name),
                    p,
                    fit_results["parameters_standard_errors"][j],
                    fit_results["parameters_lower_CI"][j],
                    fit_results["parameters_upper_CI"][j],
                )
            ]
            name_dict[name] = [
                p,
                fit_results["parameters_standard_errors"][j],
                (
                    fit_results["parameters_lower_CI"][j],
                    fit_results["parameters_upper_CI"][j],
                ),
            ]
            if return_data_class:  # returns instead of name_dict
                data[name] = [
                    p,
                    fit_results["parameters_standard_errors"][j],
                    fit_results["parameters_lower_CI"][j],
                    fit_results["parameters_upper_CI"][j],
                ]
                if data.hd == {}:
                    for jhd, ahd in enumerate(["value", "sterr", "low_CI", "up_CI"]):
                        data.hd[ahd] = jhd
    elif "parameters_standard_errors" in fit_results:
        par_hd = ["parameter", "mean", "error"]
        for j, p in enumerate(fit_results["parameters_fitted"]):
            if (
                fit_results["parameters_standard_errors"][j] < p
                and fit_results["parameters_standard_errors"][j] > 0
            ):
                pr, err = round_with_error(
                    p, fit_results["parameters_standard_errors"][j]
                )
            else:
                pr, err = Round_To_n(p, n=1, only_decimals=True), Round_To_n(
                    fit_results["parameters_standard_errors"][j],
                    n=1,
                    only_decimals=True,
                )
            if out is not None:
                out.write(" %s= %g +/- %g\n" % (parameters_name[j], pr, err))
                # out.write('     %g +/-%g\n'%(p, fit_results['parameters_standard_errors'][j]))
            name, jj = parameters_name[j], 1
            while name in name_dict:  # make it unique
                name = name + "%d" % (jj)
                jj += 1
            par_lines += [
                "%s\t%g\t%g"
                % (str(name), p, fit_results["parameters_standard_errors"][j])
            ]
            name_dict[name] = [p, fit_results["parameters_standard_errors"][j]]
            if return_data_class:  # returns instead of name_dict
                data[name] = [p, fit_results["parameters_standard_errors"][j]]
                if data.hd == {}:
                    for jhd, ahd in enumerate(["value", "sterr"]):
                        data.hd[ahd] = jhd
    else:
        par_hd = ["parameter", "mean"]
        for j, p in enumerate(fit_results["parameters_fitted"]):
            if out is not None:
                out.write(" %s= %g\n" % (parameters_name[j], p))
        name, jj = parameters_name[j], 1
        while name in name_dict:  # make it unique
            name = name + "%d" % (jj)
            jj += 1
        par_lines += ["%s\t%g" % (str(name), p)]
        name_dict[name] = p
        if return_data_class:  # returns instead of name_dict
            data[name] = [p]
            if data.hd == {}:
                for jhd, ahd in enumerate(["value"]):
                    data.hd[ahd] = jhd
    if out is not None:
        if "popt_leastsq_guess" in fit_results:
            if "perrors_leastsq" in fit_results:
                for j, p in enumerate(fit_results["popt_leastsq_guess"]):
                    pr, err = round_with_error(p, fit_results["perrors_leastsq"][j])
                    out.write(
                        " from leastsq %s= %g +/- %g\n" % (parameters_name[j], pr, err)
                    )
            else:
                for j, p in enumerate(fit_results["popt_leastsq_guess"]):
                    out.write(" from leastsq %s= %g\n" % (parameters_name[j], p))
        if "sum_of_squared_residuals" in fit_results:
            out.write(
                " sum_of_squared_residuals = %g "
                % (fit_results["sum_of_squared_residuals"])
            )
        if "chi2" in fit_results:
            out.write(" Chi2 = %g\n" % (fit_results["chi2"]))
        if "chi2red" in fit_results:
            if type(fit_results["chi2red"]) is not float:
                out.write(" Chi2red = %s\n" % (str(fit_results["chi2red"])))
            else:
                out.write(" Chi2red = %g\n" % (fit_results["chi2red"]))
        if (
            "RMS_of_residuals" in fit_results
            and type(fit_results["RMS_of_residuals"]) is float
        ):
            out.write(" RMS_of_residuals = %g\n" % (fit_results["RMS_of_residuals"]))
        if "R2" in fit_results:
            out.write(" R^2 = %g\n" % (fit_results["R2"]))
        out.write(extra_str)
        out.write("***********************************************************\n")
    if close and out is not None:  # print results also in non-rounded table format
        out.write("\t".join(par_hd) + "\n")
        out.write("\n".join(par_lines) + "\n")
        out.close()
    if return_data_class:  # returns instead of name_dict
        if "sum_of_squared_residuals" in fit_results:
            data["sum_of_squared_residuals"] = [
                fit_results["sum_of_squared_residuals"]
            ] + ["" for i in range(len(data.hd) - 1)]
        if "R2" in fit_results:
            data["R2"] = [fit_results["R2"]] + ["" for i in range(len(data.hd) - 1)]
        return data
    return name_dict


def CompareFloats(float1, float2, sensibility=0.0001):
    """
    #compare two float numbers at a specified sensibility, Returns True/False
    # or if input are two numpy array returns the corresponding bool numpy array
    #return float1-sensibility <= float2 <= float1+sensibility
    """
    return (float1 - sensibility <= float2) & (float2 <= float1 + sensibility)


def polfit(x, y, pol_order=1, swap=False):
    """
    note that R will be the same even if you swap
    """
    x = numpy.array(x)
    y = numpy.array(y)
    idx = numpy.isfinite(x) & numpy.isfinite(y)
    if swap and pol_order == 1:
        coeff = numpy.polyfit(y[idx], x[idx], pol_order)
        coefficients = numpy.array([1.0 / coeff[0], -1.0 * coeff[1] / coeff[0]])
    else:
        coefficients = numpy.polyfit(x[idx], y[idx], pol_order)
    pol_fun = numpy.poly1d(coefficients)
    SS_tot = ((y[idx] - numpy.mean(y[idx])) ** 2).sum()
    SS_res = ((y[idx] - pol_fun(x[idx])) ** 2).sum()
    R_2 = 1 - SS_res / SS_tot
    return coefficients, pol_fun, R_2


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def find_nearest(numpy_array, value):
    """
    given a numpy array it finds in it the element closest to value.
    it returns the element and its index
    """
    idx = (numpy.abs(numpy_array - value)).argmin()
    return numpy_array[idx], idx


def magnitude(x):
    return int(numpy.floor(numpy.log10(x)))


def ScaleInRange(OldList, NewMin, NewMax):
    NewRange = 1.0 * (NewMax - NewMin)
    OldMin = numpy.amin(OldList, 0)
    OldMax = numpy.amax(OldList, 0)
    OldRange = 1.0 * (OldMax - OldMin)
    ScaleFactor = NewRange / OldRange
    #    print '\nEquation:  NewValue = ((OldValue - ' + str(OldMin) + ') x '+ str(ScaleFactor) + ') + ' + str(NewMin) + '\n'
    NewList = ((numpy.array(OldList) - OldMin) * ScaleFactor) + NewMin
    return NewList


def cmToinch(value):
    return value / 2.54


def close():
    return plt.close("all")


def figure_with_buttons(
    list_of_button_messages,
    message=None,
    buttons_per_row=2,
    figsize=(4, 3),
    fontsize=12,
    figure=None,
):
    n_buttons = len(list_of_button_messages)
    if n_buttons <= 0:
        raise Exception("No buttons given")
    if figure is None:
        figure = plt.figure(figsize=figsize)
    ax = figure.gca()
    if message is not None:
        ax.set_title(message, fontsize=fontsize + 1)
    ax.axis("off")
    n_rows = n_buttons // buttons_per_row
    last_row = n_buttons % buttons_per_row
    if last_row > 0:
        n_rows += 1
    ax.set_xlim((0, 10.001))
    ax.set_ylim((0, 10.001))
    hlines = numpy.linspace(0, 10, n_rows + 1)
    vlines = numpy.linspace(0, 10, buttons_per_row + 1)
    for yt in hlines:
        plt.axhline(yt, color="black", ls="-", lw=1)
    for xt in vlines:
        plt.axvline(xt, color="black", ls="-", lw=1)
    button_corners_list = []
    k = 0
    finished = False
    hlines = hlines[::-1]
    for j, y in enumerate(hlines):
        if j == 0:
            continue
        for i, x in enumerate(vlines[:-1]):
            button_corners_list += [[(x, y), (vlines[i + 1], hlines[j - 1])]]
            text_coord = ((x + vlines[i + 1]) / 2.0, (y + hlines[j - 1]) / 2.0)
            ax.text(
                text_coord[0],
                text_coord[1],
                list_of_button_messages[k],
                fontsize=fontsize,
                verticalalignment="center",
                horizontalalignment="center",
            )
            k += 1
            # print k,hlines,vlines,j,i,x,y,text_coord,button_corners_list[-1]
            if k == len(list_of_button_messages):
                finished = True
                break
        if finished:
            break
    plt.draw()
    plt.show(block=False)

    def get_button_index(coord, corner_list=button_corners_list):
        # returns the index of the bottom chosen
        for j, (left_bottom, right_top) in enumerate(corner_list):
            if (
                left_bottom[0] <= coord[0] <= right_top[0]
                and left_bottom[1] <= coord[1] <= right_top[1]
            ):
                return j
        return None

    return figure, get_button_index


def draw_buttons(
    subplot,
    message,
    n_buttons,
    list_of_button_messages,
    y_rel_size=0.25,
    x_rel_size=1,
    ythreshold=None,
    figure=None,
):
    if n_buttons > 2:
        print("***ERROR*** in draw_buttons() n_buttons>2. solution not implemented yet")
        return
    added_stuff = []
    xmin, xmax, ymin, ymax = subplot.axis(figure=figure)
    if ythreshold is None:
        ythreshold = ymax
    heigth = y_rel_size * (ythreshold - ymin)
    width = x_rel_size * (xmax - xmin)

    left_bottom = (xmin, ythreshold)
    right_top = (xmin + width, ythreshold + heigth)

    all_patch = patch_from_corners(
        left_bottom, right_top, facecolor="white", alpha=None
    )
    added_stuff.append(subplot.add_patch(all_patch))

    subplot.set_ylim(ymin, right_top[1])
    if n_buttons == 0:
        x_split = right_top[0]
    elif n_buttons <= 2:
        x_split = 2 * (right_top[0] + left_bottom[0]) / 3
    else:
        x_split = (right_top[0] + left_bottom[0]) / 2.0
    x_mid_message = (left_bottom[0] + x_split) / 2.0
    y_mid = (right_top[1] + left_bottom[1]) / 2.0
    added_stuff.append(
        subplot.text(
            x_mid_message,
            y_mid,
            message,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="center",
        )
    )
    button_corners_list = []
    if n_buttons == 1:
        button_corners_list.append(
            [(x_split, left_bottom[1]), (right_top[0], right_top[1])]
        )  # format left_bottom, right_top
        patch = patch_from_corners(
            *button_corners_list[-1], facecolor="grey", lw=1.3, alpha=0.1
        )
        subplot.add_patch(patch)
        subplot.text(
            (button_corners_list[-1][1][0] + button_corners_list[-1][0][0]) / 2,
            (button_corners_list[-1][1][1] + button_corners_list[-1][0][1]) / 2,
            list_of_button_messages[0],
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="center",
        )
    elif n_buttons == 2:
        button_corners_list.append(
            [
                (x_split, left_bottom[1]),
                (right_top[0], (right_top[1] + left_bottom[1]) / 2.0),
            ]
        )  # format left_bottom, right_top
        patch = patch_from_corners(
            *button_corners_list[-1], facecolor="grey", lw=1.3, alpha=0.1
        )
        subplot.add_patch(patch)
        subplot.text(
            (button_corners_list[-1][1][0] + button_corners_list[-1][0][0]) / 2,
            (button_corners_list[-1][1][1] + button_corners_list[-1][0][1]) / 2,
            list_of_button_messages[0],
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="center",
        )

        button_corners_list.append(
            [
                (x_split, (right_top[1] + left_bottom[1]) / 2.0),
                (right_top[0], right_top[1]),
            ]
        )  # format left_bottom, right_top
        patch = patch_from_corners(
            *button_corners_list[-1], facecolor="grey", lw=1.3, alpha=0.1
        )
        subplot.add_patch(patch)
        subplot.text(
            (button_corners_list[-1][1][0] + button_corners_list[-1][0][0]) / 2,
            (button_corners_list[-1][1][1] + button_corners_list[-1][0][1]) / 2,
            list_of_button_messages[1],
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="center",
        )

    def get_button_index(coord, corner_list=button_corners_list):
        # returns the index of the bottom chosen
        for j, (left_bottom, right_top) in enumerate(corner_list):
            if (
                left_bottom[0] <= coord[0] <= right_top[0]
                and left_bottom[1] <= coord[1] <= right_top[1]
            ):
                return j
        return None

    return subplot, button_corners_list, get_button_index, added_stuff


def patch_from_corners(
    left_bottom, right_top, close_shape=True, facecolor="white", lw=2, alpha=0.1
):
    codes = [
        matplotlib.path.Path.MOVETO,
        matplotlib.path.Path.LINETO,
        matplotlib.path.Path.LINETO,
        matplotlib.path.Path.LINETO,
    ]
    if close_shape:
        codes += [matplotlib.path.Path.CLOSEPOLY]
    verts = corners_to_verts(left_bottom, right_top, add_ignored=close_shape)
    path = matplotlib.path.Path(verts, codes)
    patch = matplotlib.patches.PathPatch(path, facecolor=facecolor, lw=lw, alpha=alpha)
    return patch


def corners_to_verts(left_bottom, right_top, add_ignored=True):
    # add_ignored is useful to close a polymer
    verts = [
        (left_bottom[0], left_bottom[1]),  # left, bottom
        (left_bottom[0], right_top[1]),  # left, top
        (right_top[0], right_top[1]),  # right, top
        (right_top[0], left_bottom[1]),  # right, bottom
    ]
    if add_ignored:
        verts += [(0.0, 0.0)]
    return verts


def pie(
    entries,
    value_labels=None,
    explode=None,
    autopct=None,
    shadow=False,
    label_size=None,
    pctdistance=0.6,
    labeldistance=1.1,
    startangle=90,
    label=None,
    radius=None,
    color=None,
    hatch=None,
    linewidth=0.7,
    linestyle="solid",
    title=None,
    value_labels_rotation="horizontal",
    print_all_labels=False,
    figure=None,
    figure_size=None,
    frame=None,
    save=False,
    legend_size=None,
    legend_location="best",
    show=True,
    block=False,
):
    """
    explode=(0, 0.05, 0, 0) assume 4 entries, with the second one exploded by 5%
    autopct='%1.1f%%' writes the percentages with one decimal precision
    # The default startangle is 0, which would start
    # the first slice on the x-axis.  With startangle=90,
    # everything is rotated counter-clockwise by 90 degrees,
    # so the plotting starts on the positive y-axis.
    """
    if frame is None:
        frame = default_parameters["frame"]
    if figure_size is None:
        if default_figure_sizes["default"] is None:
            figure_size = default_figure_sizes["pie"]
        else:
            figure_size = default_figure_sizes["default"]
    if default_figure_sizes["use_cm"]:
        figure_size = (cmToinch(figure_size[0]), cmToinch(figure_size[1]))
    if legend_size is None:
        legend_size = text_sizes["legend_size"]
    if label_size is None:
        label_size = text_sizes["value_labels"]
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    plt.rc("xtick", labelsize=text_sizes["xlabels"])
    plt.rc("ytick", labelsize=text_sizes["xlabels"])
    if type(entries) is dict:
        value_labels = list(entries.keys())
        entries = list(entries.values())
        value_labels, entries = list(
            zip(*sorted(zip(value_labels, entries), reverse=False))
        )
    elif isinstance(entries, OrderedDict):
        value_labels = list(entries.keys())
        entries = list(entries.values())

    if type(label) is bool and label == True:
        if value_labels is not None:
            label = value_labels
            value_labels = None

    cols = []
    for j, x in enumerate(entries):

        if type(color) is dict and value_labels is not None and value_labels != False:
            if value_labels[j] in color:
                cols += [color[value_labels[j]]]
            else:
                print(
                    "WARNING in pie() value_labels[j]=%s not in color dict"
                    % (value_labels[j])
                )
    if cols != []:
        color = cols
    if figure is None:
        figure = plt.figure(figsize=figure_size)

    ax = figure.gca()
    ax.set_aspect("equal")
    if default_figure_sizes["all_tight"]:
        plt.tight_layout()

    # normalize to 100
    # entries=numpy.array(entries)
    # if not CompareFloats(entries.sum(),100) :
    #    entries*=100./entries.sum()
    # colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']#
    # patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    # plt.legend(patches, labels, loc="best")
    if title is not None:
        plt.text(
            0.5,
            1.03,
            title,
            horizontalalignment="center",
            fontsize=text_sizes["title"],
            transform=ax.transAxes,
        )
    # if min(entries)>10000 or max(entries)<0.001: ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y') # use scientific notation
    PieObjects = plt.pie(
        entries,
        explode=explode,
        labels=value_labels,
        colors=color,
        autopct=autopct,
        pctdistance=pctdistance,
        shadow=shadow,
        labeldistance=labeldistance,
        startangle=startangle,
        radius=radius,
    )
    for j, wedge in enumerate(PieObjects[0]):
        wedge.set_lw(linewidth)  # adjust the line width of each slice
        wedge.set_linestyle(linestyle)
        PieObjects[1][j].set_fontsize(label_size)
        if autopct is not None:
            PieObjects[2][j].set_fontsize(label_size)

    if label is not None and label != "":
        plt.legend(
            PieObjects[0], label, loc=legend_location, prop={"size": legend_size}
        )

    if not frame:
        # this remove top and right axis
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.draw()
    if save != False and save is not None:
        figure.savefig(
            save,
            dpi=default_figure_sizes["dpi"],
            bbox_inches="tight",
            bbox_extra_artists=[ax, figure],
            transparent=True,
        )  # ,bbox_inches="tight" is such that everything is in the figure... even though margins are removed.
    if show:
        plt.show(block=block)
    return figure, wedge


def fmt(x, y):
    return "x: {x:0.2f}\ny: {y:0.2f}".format(x=x, y=y)


class DataCursor(object):
    """
        # http://stackoverflow.com/a/4674445/190597
        A simple data cursor widget that displays the x,y location of a
        matplotlib artist when it is selected.
        USAGE:
    x=[1,2,3,4,5]
    y=[6,7,8,9,10]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    scat = ax.scatter(x, y)
    curs= DataCursor(ax, x, y)
    plt.show(block=False)
    """

    def __init__(
        self,
        artists,
        x=[],
        y=[],
        tolerance=5,
        offsets=(-20, 20),
        formatter=fmt,
        display_all=False,
    ):
        """
        Create the data cursor and connect it to the relevant figure.
        "artists" is the matplotlib artist or sequence of artists that will be
            selected.
        "tolerance" is the radius (in points) that the mouse click must be
            within to select the artist.
        "offsets" is a tuple of (x,y) offsets in points from the selected
            point to the displayed annotation box
        "formatter" is a callback function which takes 2 numeric arguments and
            returns a string
        "display_all" controls whether more than one annotation box will
            be shown if there are multiple axes.  Only one will be shown
            per-axis, regardless.
        """
        self._points = numpy.column_stack((x, y))
        self.clicked = []
        self.formatter = formatter
        self.offsets = offsets
        self.display_all = display_all
        if not matplotlib.cbook.iterable(artists):
            artists = [artists]
        self.artists = artists
        self.axes = tuple(set(art.axes for art in self.artists))
        self.figures = tuple(set(ax.figure for ax in self.axes))

        self.annotations = {}
        for ax in self.axes:
            self.annotations[ax] = self.annotate(ax)

        for artist in self.artists:
            artist.set_picker(tolerance)
        for fig in self.figures:
            fig.canvas.mpl_connect("pick_event", self)

    def annotate(self, ax):
        """Draws and hides the annotation box for the given axis "ax"."""
        annotation = ax.annotate(
            self.formatter,
            xy=(0, 0),
            ha="right",
            xytext=self.offsets,
            textcoords="offset points",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )
        annotation.set_visible(False)
        return annotation

    def snap(self, x, y):
        """Return the value in self._points closest to (x, y)."""
        idx = numpy.nanargmin(((self._points - (x, y)) ** 2).sum(axis=-1))
        return self._points[idx]

    def __call__(self, event):
        """Intended to be called through "mpl_connect"."""
        # Rather than trying to interpolate, just display the clicked coords
        # This will only be called if it's within "tolerance", anyway.
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        # print 'deb',x,y
        annotation = self.annotations[event.artist.axes]
        if x is not None:
            if not self.display_all:
                # Hide any other annotation boxes...
                for ann in list(self.annotations.values()):
                    ann.set_visible(False)
            # Update the annotation in the current axis..
            x, y = self.snap(x, y)  # value of closest point
            annotation.xy = x, y
            annotation.set_text(self.formatter(x, y))
            annotation.set_visible(True)
            event.canvas.draw()
            self.clicked += [(x, y)]
        return


class SaveCursor(object):
    """
        # http://stackoverflow.com/a/4674445/190597
        A simple data cursor widget that displays the x,y location of a
        matplotlib artist when it is selected.
        USAGE:
    exit()
    python

    from plotter import *
    x=[1,2,3,4,5]
    y=[6,7,8,9,10]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    scat = ax.scatter(x, y)
    curs= SaveCursor(ax, x, y)
    print 'close figure %d to continue' % (fig.number)
    plt.show() # don't put block
    print curs.saved

    # this does not work:
    print 'close figure %d to continue' % (fig.number)
    while plt.fignum_exists(fig.number) :
        pass

    print curs.saved
    """

    def __init__(
        self,
        artists,
        x=[],
        y=[],
        tolerance=5,
        offsets=(-20, 20),
        formatter=fmt,
        display_all=False,
        add_vline=True,
        first_message=None,
    ):
        """
        Create the data cursor and connect it to the relevant figure.
        "artists" is the matplotlib artist or sequence of artists that will be
            selected.
        "tolerance" is the radius (in points) that the mouse click must be
            within to select the artist.
        "offsets" is a tuple of (x,y) offsets in points from the selected
            point to the displayed annotation box
        "formatter" is a callback function which takes 2 numeric arguments and
            returns a string
        "display_all" controls whether more than one annotation box will
            be shown if there are multiple axes.  Only one will be shown
            per-axis, regardless.
        """
        self._points = numpy.column_stack((x, y))
        self.clicked = []
        self.saved = []
        self.savediag = None
        self.display_savediag = True
        self.formatter = formatter
        self.offsets = offsets
        self.display_all = display_all
        if not matplotlib.cbook.iterable(artists):
            artists = [artists]
        self.artists = artists
        self.axes = tuple(set(art.axes for art in self.artists))
        self.figures = tuple(set(ax.figure for ax in self.axes))
        self.add_vline = add_vline
        self.annotations = {}
        for ax in self.axes:
            self.annotations[ax] = self.annotate(ax)
        if first_message is not None:
            self.savediag = plt.annotate(
                first_message,
                xy=(0.5, 0.9),
                ha="center",
                xycoords="figure fraction",
                xytext=self.offsets,
                textcoords="offset points",
                va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="gray", alpha=0.5),
                # arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
            )
        for artist in self.artists:
            artist.set_picker(tolerance)
        for fig in self.figures:
            fig.canvas.mpl_connect("pick_event", self)

    def annotate(self, ax):
        """Draws and hides the annotation box for the given axis "ax".
        this then get moved and made visible at every click"""
        annotation = ax.annotate(
            self.formatter,
            xy=(0, 0),
            ha="right",
            xytext=self.offsets,
            textcoords="offset points",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )
        annotation.set_visible(False)
        return annotation

    def snap(self, x, y):
        """Return the value in self._points closest to (x, y)."""
        idx = numpy.nanargmin(((self._points - (x, y)) ** 2).sum(axis=-1))
        return self._points[idx]

    def __call__(self, event):
        """Intended to be called through "mpl_connect"."""
        # Rather than trying to interpolate, just display the clicked coords
        # This will only be called if it's within "tolerance", anyway.
        # print dir(event.mouseevent)
        # print event.mouseevent.dblclick
        if (
            event.mouseevent.dblclick and self.savediag is not None
        ):  # note that the first clikc of the double click will go through as false and the second will be here
            print("saving:", self.clicked[-2])
            self.saved += [self.clicked[-2]]
            self.savediag.set_visible(False)
            if self.add_vline:
                for ax in self.axes:
                    ax.axvline(
                        self.saved[-1][0],
                        color="black",
                        ls="--",
                        lw=plt.rcParams["axes.linewidth"],
                        zorder=-2,
                    )
            self.savediag.set_text("%s saved" % (self.formatter(*self.saved[-1])))
            self.savediag.set_visible(True)
            self.display_savediag = True
            plt.draw()
            return
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        # print 'deb',x,y
        annotation = self.annotations[event.artist.axes]
        if x is not None:
            if not self.display_all:
                # Hide any other annotation boxes...
                for ann in list(self.annotations.values()):
                    ann.set_visible(False)
            # Update the annotation in the current axis..
            x, y = self.snap(x, y)  # value of closest point
            annotation.xy = x, y
            annotation.set_text(self.formatter(x, y))
            annotation.set_visible(True)
            event.canvas.draw()
            self.clicked += [(x, y)]
        if self.display_savediag:
            self.display_savediag = False
            if self.savediag is not None:
                self.savediag.set_visible(False)
            self.savediag = plt.annotate(
                "doubleclick in axis to save selected point",
                xy=(0.5, 0.9),
                ha="center",
                xycoords="figure fraction",
                xytext=self.offsets,
                textcoords="offset points",
                va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="gray", alpha=0.5),
                # arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
            )
            self.savediag.set_visible(True)
            plt.draw()
        return


class FollowDotCursor(object):
    """
        Display the x,y location of the nearest data point.
    USAGE:
    x=[1,2,3,4,5]
    y=[6,7,8,9,10]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y)
    cursor = FollowDotCursor(ax, x, y)
    plt.show(block=False)
    """

    def __init__(self, ax, x, y, tolerance=5, formatter=fmt, offsets=(-20, 20)):
        try:
            x = numpy.asarray(x, dtype="float")
        except (TypeError, ValueError):
            raise
            # x = numpy.asarray(mdates.date2num(x), dtype='float')
        y = numpy.asarray(y, dtype="float")
        self._points = numpy.column_stack((x, y))
        self.offsets = offsets
        self.scale = x.ptp()
        self.scale = y.ptp() / self.scale if self.scale else 1
        self.tree = scipy.spatial.cKDTree(self.scaled(self._points))
        self.formatter = formatter
        self.tolerance = tolerance
        self.ax = ax
        self.fig = ax.figure
        self.ax.xaxis.set_label_position("top")
        self.dot = ax.scatter([x.min()], [y.min()], s=130, color="green", alpha=0.7)
        self.annotation = self.setup_annotation()
        plt.connect("motion_notify_event", self)

    def scaled(self, points):
        points = numpy.asarray(points)
        return points * (self.scale, 1)

    def __call__(self, event):
        ax = self.ax
        # event.inaxes is always the current axis. If you use twinx, ax could be
        # a different axis.
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
        elif event.inaxes is None:
            return
        else:
            inv = ax.transData.inverted()
            x, y = inv.transform([(event.x, event.y)]).ravel()
        annotation = self.annotation
        x, y = self.snap(x, y)
        annotation.xy = x, y
        annotation.set_text(self.formatter(x, y))
        self.dot.set_offsets((x, y))
        bbox = ax.viewLim
        event.canvas.draw()

    def setup_annotation(self):
        """Draw and hide the annotation box."""
        annotation = self.ax.annotate(
            "",
            xy=(0, 0),
            ha="right",
            xytext=self.offsets,
            textcoords="offset points",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.75),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )
        return annotation

    def snap(self, x, y):
        """Return the value in self.tree closest to x, y."""
        dist, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
        try:
            return self._points[idx]
        except IndexError:
            # IndexError: index out of bounds
            return self._points[0]


readable_map_for_chars = create_color_map(
    1, 0, colors=["white", "Gold", "SpringGreen", "DeepSkyBlue"]
)
readable_map_for_chars_centered = create_color_map(
    1, 0, colors=["Gold", "Moccasin", "white", "SpringGreen", "DeepSkyBlue"]
)
readable_map_for_chars_vmin = create_color_map(
    0,
    1,
    colors=[
        "AntiqueWhite",
        "Bisque",
        "Gold",
        "GoldenRod",
        "GreenYellow",
        "SpringGreen",
        "LightSkyBlue",
        "DeepSkyBlue",
    ],
    masked_vals_color=None,
    return_sm=False,
    set_under="white",
    set_over="blue",
)
# also decent cmap = plt.get_cmap('rainbow')


def plot_pssm_of_seq(
    sequence,
    pssm,
    plot_only_top=None,
    plot_conservation_index_num_seqs=None,
    center_cmap_on_value=None,
    y_labs=[
        "-",
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
    ],
    color_different_from_consensus=True,
    aa_extra_labels=None,
    sort=True,
    func_to_sort=None,
    small_at_bottom=True,
    cmap=None,
    max_res_per_plot=None,
    sequence_fontsize=None,
    annotation_string=None,
    annotation_string_position="top",
    annotation_string_rotation="vertical",
    vgrid=5,
    save=None,
    show=True,
    block=False,
    sequence_extra_numbering=None,
    figure_size=None,
    vmin=None,
    vmax=None,
    y_major_tick_every=None,
    **kwargs
):
    """
    pssm may be obtained from an alignment with
      pssm, aa_freq, aa_freq_ignoring_gaps, gap_freqs, consensus,key_seq=mybio.process_aln_records(records, key_id=None)
    # consider also mybio.alignment_correlations_at_position(records,index,key_id=None) , which will provide the subset of alignment with given residue at given position
    # aa_extra_labels puts an extra label in the cell of each amino acid.. (could be frequency value or similar.
    default cmap will be readable_map_for_chars unless center_cmap_on_value is given, in which case it will be readable_map_for_chars_centered
    plot_conservation_index_num_seqs can either be the number of sequences to be used to calculate the conservation index or already a profile (an array)
      of the same length of the sequence to be plotted overlaid onto the pssm.
    """
    if pssm.shape[0] == 20 and len(y_labs) == 21:
        y_labs = y_labs[1:]  # remove '-'
    aa_values = None
    ylabels = False
    hgrid = None
    if y_major_tick_every is None:
        y_major_tick_every = False  # default for PSSM plot
    if sequence_fontsize is None:
        sequence_fontsize = text_sizes["xlabels_many"]
    if max_res_per_plot is None:
        max_res_per_plot = default_parameters["seq_max_res_per_plot"]
        if max_res_per_plot is None:  # may have been default parameter
            if type(sequence_fontsize) is not str:
                if figure_size is None:
                    max_res_per_plot = int(
                        (120 / float(sequence_fontsize))
                        * (default_figure_sizes["sequence"][0])
                    )
                else:
                    max_res_per_plot = int(
                        (120 / float(sequence_fontsize)) * figure_size[0]
                    )
    nplots = int(len(sequence) / max_res_per_plot)
    nplots = max(
        [
            1,
            int(
                len(sequence) / float(max_res_per_plot)
                + int((len(sequence) % int(max_res_per_plot)) > 0)
            ),
        ]
    )
    if figure_size is None:
        if plot_only_top is None:
            figure_size = (
                default_figure_sizes["sequence"][0],
                min([20, len(y_labs) / 2.5]),
            )
        else:
            figure_size = (
                default_figure_sizes["sequence"][0],
                min([20, nplots * (1 + plot_only_top / 2.5)]),
            )
        if type(sequence) is str and len(sequence) > 600:
            figure_size = (figure_size[0], min([20, int(len(sequence) / 100)]))

    if func_to_sort is None:
        func_to_sort = lambda x: x  # identity
    to_plot = numpy.array(pssm).copy()
    if sort:
        aa_values = numpy.tile(numpy.array([y_labs]).transpose(), (1, to_plot.shape[1]))
        if small_at_bottom:
            sort_indices = numpy.argsort(-1.0 * func_to_sort(to_plot), axis=0)
        else:
            sort_indices = numpy.argsort(func_to_sort(to_plot), axis=0)
        static_indices = numpy.indices(to_plot.shape)
        to_plot = to_plot[sort_indices, static_indices[1]]
        if aa_extra_labels is not None:
            aa_extra_labels = numpy.array(aa_extra_labels)[
                sort_indices, static_indices[1]
            ]
        aa_values = aa_values[sort_indices, static_indices[1]]
        if plot_only_top is not None:
            # print 'DEB: plot_pssm_of_seq',to_plot.shape,aa_values.shape
            to_plot = to_plot[:plot_only_top]
            aa_values = aa_values[:plot_only_top]
            if aa_extra_labels is not None:
                aa_extra_labels = aa_extra_labels[:plot_only_top]

        if aa_extra_labels is not None:
            aa_values = (aa_values, aa_extra_labels)  # should plot both supreimposed
            # print  'DebShapes2',pssm.shape,aa_values[0].shape,aa_values[1].shape
        # else :    print 'DebShapes',pssm.shape,aa_values.shape
    if aa_values is None:
        ylabels = y_labs[::-1]
        hgrid = 1
    # print('DEB:',ylabels, len(aa_values))
    out = plot_seq_profile(
        sequence,
        to_plot,
        do_matrix=True,
        value_labels=aa_values,
        ylabels=ylabels,
        sequence_extra_numbering=sequence_extra_numbering,
        zygg_like_lines=False,
        hgrid=hgrid,
        ylabel="",
        vgrid=False,
        print_all_sequence=True,
        max_res_per_plot=max_res_per_plot,
        sequence_fontsize=sequence_fontsize,
        do_matrix_cmap=cmap,
        center_cmap_on_value=center_cmap_on_value,
        annotation_string=annotation_string,
        annotation_string_position=annotation_string_position,
        annotation_string_rotation=annotation_string_rotation,
        save=None,
        show=False,
        figure_size=figure_size,
        vmin=vmin,
        vmax=vmax,
        y_major_tick_every=y_major_tick_every,
        **kwargs
    )
    fig, axt = out[:2]

    try:
        plt.draw()
        cons_index = None
        if (
            type(plot_conservation_index_num_seqs) is int
            and plot_conservation_index_num_seqs > 0
        ):
            if (
                pssm < 0
            ).any():  # probably log2 enrichments, correction is still not very good as it should be done on frequencies not enrichments!
                sys.stderr.write(
                    "**WARNING** in plotter.plot_pssm_of_seq asked to calculate conservation index on what looks like an erichment PSSM. This should be calculated on the frequency PSSM, the results will be inaccurate!!\n"
                )
                cons_index = mybio.conservation_index_variance_based(
                    2**pssm,
                    num_sequences=plot_conservation_index_num_seqs,
                    gapless=True,
                )
            else:
                cons_index = mybio.conservation_index_variance_based(
                    pssm, num_sequences=plot_conservation_index_num_seqs, gapless=True
                )
        elif hasattr(
            plot_conservation_index_num_seqs, "__len__"
        ):  # likely an array with the already computed conservation index
            if len(plot_conservation_index_num_seqs) == pssm.shape[1]:
                cons_index = plot_conservation_index_num_seqs
            else:
                sys.stderr.write(
                    "**WARNING** in plot_pssm_of_seq() given plot_conservation_index_num_seqs as profile of length %d but this is not compatible with input pssm of shape %s\n"
                    % (len(plot_conservation_index_num_seqs), str(pssm.shape))
                )
        # print('DEB: cons_index',cons_index,'plot_conservation_index_num_seqs',plot_conservation_index_num_seqs)
        if cons_index is not None:
            # cons_index should be in 0 to 1 by definition, however, if calculated from log-likelihood it won't be.
            # this puts it in the range of the existing plot.
            ylims = axt[0].get_ylim()
            yvals = misc.ScaleInRange(cons_index, ylims[0], ylims[1])
            for ax in axt:
                xlims = ax.get_xlim()
                # print('DEB: plotting cons_index xlims',xlims,'ylims:',ylims,numpy.nanmax(yvals),numpy.nanmin(yvals),len(yvals))
                lims = (numpy.array(xlims)).astype(int)
                xpos = numpy.arange(lims[0] + 1, lims[1] + 1)
                # print plot_conservation_index_num_seqs,cons_index[lims[0]:lims[1]]
                ax.plot(
                    xpos,
                    yvals[lims[0] : lims[1]],
                    ls="-",
                    color="DarkRed",
                    lw=default_parameters["linewidth"] - 0.5,
                )  # ,alpha=1)
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)
                plt.draw()
    except Exception:
        sys.stderr.write(
            "\n Error in plot_pssm_of_seq while trying to add conservation index\n"
        )
        print_exc(file=sys.stderr)
        sys.stderr.flush()
        pass
    if vgrid:
        for j in range(len(axt)):
            if hasattr(vgrid, "__len__"):
                for vl in vgrid:
                    if type(vl) is int:
                        vl -= 0.5
                    axt[j].axvline(
                        vl,
                        color=grid_parameters["vcolor"],
                        ls=grid_parameters["v_ls"],
                        lw=grid_parameters["v_lw"],
                    )
            else:
                xlims = axt[j].get_xlim()
                # print('DEB: plotting cons_index xlims',xlims,'ylims:',ylims,numpy.nanmax(yvals),numpy.nanmin(yvals),len(yvals))
                lims = (numpy.array(xlims)).astype(int)
                x_pos = numpy.arange(lims[0] + 1, lims[1] + 1)
                for count in x_pos:
                    if type(vgrid) is int:
                        if count % vgrid == 0:
                            axt[j].axvline(
                                count - 0.5,
                                color=grid_parameters["vcolor"],
                                ls=grid_parameters["v_ls"],
                                lw=grid_parameters["v_lw"],
                            )
                    elif count % 10 == 0:
                        axt[j].axvline(
                            count - 0.5,
                            color=grid_parameters["vcolor"],
                            ls=grid_parameters["v_ls"],
                            lw=grid_parameters["v_lw"],
                        )
    # try to color red those residues that differ from consensus
    if (
        color_different_from_consensus is not None
        and aa_values is not None
        and color_different_from_consensus != False
    ):
        for ax in axt:
            if hasattr(ax, "ax2"):
                xlims = ax.get_xlim()
                lims = (numpy.array(xlims)).astype(int)
                coldiff, col_neg = "#641E16", "red"
                is_loglikelihood = False
                if (pssm < 0).any():
                    is_loglikelihood = True
                for j, aa in enumerate(sequence[lims[0] : lims[1]]):
                    if (
                        aa in y_labs
                        and lims[0] + j < pssm.shape[1]
                        and aa_values[0][lims[0] + j] != aa
                        and (pssm[:, lims[0] + j] > 0).any()
                    ):
                        newcol = coldiff
                        if cons_index is not None:
                            if (
                                cons_index[lims[0] + j] >= 0.25
                            ):  # position conserved more than 25%
                                if is_loglikelihood:
                                    if pssm[y_labs.index(aa)][lims[0] + j] <= 0:
                                        newcol = col_neg
                                elif pssm[y_labs.index(aa)][lims[0] + j] <= 0.05:
                                    newcol = col_neg  # less than 5% frequency
                        else:
                            if is_loglikelihood:
                                if pssm[y_labs.index(aa)][lims[0] + j] <= 0:
                                    newcol = col_neg
                            elif pssm[y_labs.index(aa)][lims[0] + j] <= 0.05:
                                newcol = col_neg  # less than 5% frequency
                        ax.ax2.get_xticklabels()[j].set_color(newcol)
                # print('DEB: color cons xlims',xlims,'len(ax.ax2.get_xticklabels())',len(ax.ax2.get_xticklabels(minor=True)),len(sequence),ax.ax2.get_xticklabels())
            else:
                sys.stderr.write(
                    "  plot_pssm_of_seq: cannot color sequence according to color_different_from_consensus\n"
                )
        plt.draw()
    if save is not None and save != "":
        if "." not in save:
            save += ".png"
        dpi = default_figure_sizes["dpi"]
        fig.savefig(
            save, dpi=dpi, bbox_inches="tight", transparent=True
        )  #  bbox_inches=0 remove white space around the figure.. ,
    if show:
        plt.show(block=block)
    return fig, axt


alpha_map = create_color_map(1, 0, colors=["Black", "DarkRed", (1, 1, 1, 0)][::-1])


def plot_matrix(
    matrix,
    log=None,
    log_scale=False,
    vmin=None,
    vmax=None,
    figure=None,
    subplot=None,
    x_range=None,
    y_range=None,
    interpolation="nearest",
    aspect="auto",
    extent=True,
    print_warn=True,
    cbar_label=None,
    xlabel=None,
    ylabel=None,
    title=None,
    hgrid=None,
    vgrid=None,
    grid_color="DarkRed",
    same_scale=False,
    figure_size=None,
    norm=None,
    cmap=None,
    center_cmap_on_value=None,
    cbar_label_rotation=270,
    cbar_major_ntick=None,
    cbar_fraction=0.04,
    plot_colorbar=None,
    xlabels=True,
    xlabels_fontname=None,
    frame=None,
    xlabels_rotation="horizontal",
    labelsize=None,
    ylabels=False,
    value_labels=None,
    extra_labels_size=None,
    value_labels_size=None,
    zorder=0,
    x_major_tick_every=None,
    y_major_tick_every=None,
    x_minor_tick_every=None,
    y_minor_tick_every=None,
    block=False,
    save=None,
    show=True,
):
    """
        vmin and vmax can be used to determine the range of the colormap
        consider center_cmap_on_value to center it e.g. on zero, or as an alterative give the norm explicitly like:
        from matplotlib.colors import TwoSlopeNorm
        norm=TwoSlopeNorm(0),cmap= plt.cm.seismic to centre the colormap on a specific value (0 in example)
        this function is called by plot_seq_profile in certain instances.
        value_labels can be true to plot the matrix value up to 1 significant digit (only among decimals) or an int to round the values to N decimals or any custom matrix of same shape as matrix.
    deafault map is plotter.readable_map_for_chars

    """

    # if hgrid is None : hgrid=default_parameters['hgrid']
    # if vgrid is None : vgrid=default_parameters['vgrid']
    if type(cmap) is str:
        cmap = matplotlib.cm.get_cmap(cmap)
    if center_cmap_on_value is not None:
        if norm is None:
            norm = TwoSlopeNorm(center_cmap_on_value)
        if cmap is None:
            cmap = readable_map_for_chars_centered
    if cmap is None:
        cmap = readable_map_for_chars
    if frame is None:
        frame = default_parameters["frame"]
    if subplot is not None:
        ax = subplot
        if plot_colorbar is None:
            plot_colorbar = False
        if figure is None:
            figure = ax.figure

        # print figure.number
    else:
        if figure_size is None:
            if default_figure_sizes["default"] is None:
                figure_size = default_figure_sizes["scatter"]
            else:
                figure_size = default_figure_sizes["default"]
        if default_figure_sizes["use_cm"]:
            figure_size = (cmToinch(figure_size[0]), cmToinch(figure_size[1]))
        plt.rcParams["xtick.direction"] = "out"
        plt.rcParams["ytick.direction"] = "out"
        if plot_colorbar is None:
            plot_colorbar = True
        new_figure = False
        if figure is None:
            new_figure = True
            figure = plt.figure(figsize=figure_size)

        ax = figure.gca()
    if labelsize is None:
        ylabels_size = xlabels_size = text_sizes["xlabels"]
        if xlabels is not None and hasattr(xlabels, "__len__") and len(xlabels) > 19:
            xlabels_size = text_sizes["xlabels_many"]
        if ylabels is not None and hasattr(ylabels, "__len__") and len(ylabels) > 19:
            ylabels_size = text_sizes["xlabels_many"]
    else:
        ylabels_size = xlabels_size = labelsize
    # print 'DEB: labelsize',labelsize
    Hmasked = numpy.ma.masked_where(
        matrix == 0, matrix
    )  # Mask pixels with a value of zero
    if default_figure_sizes["all_tight"]:
        cbar_major_ntick = 5

    if log is not None:
        if log == 10:
            Hmasked = numpy.log10(Hmasked)
            if cbar_label is not None and "log" not in cbar_label.lower():
                cbar_label = "Log " + cbar_label
        else:
            Hmasked = numpy.log(Hmasked)
            if cbar_label is not None and "log" not in cbar_label.lower():
                cbar_label = "log " + cbar_label

    if same_scale:
        ax.set_aspect("equal")
    if type(extent) is bool and extent == True:
        extent = (-0.5, matrix.shape[1] - 0.5, -0.5, matrix.shape[0] - 0.5)

    # print extent,type(value_labels)
    # if 'shape' in dir(value_labels) : print('value_labels.shape:',value_labels.shape)
    image = ax.imshow(
        matrix,
        interpolation=interpolation,
        aspect=aspect,
        extent=extent,
        norm=norm,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # plt.pcolormesh(xedges,yedges,Hmasked,figure=figure,cmap=cmap)
    if log_scale:
        ax.set_yscale("symlog", base=10)
        ax.set_xscale("symlog", base=10)
        # if avoid_scientific_notation:
        #    yticks=ax.yaxis.get_majorticklocs()
        #    xlab=[ 10**i for i in xrange(len(yticks))]
        #    ax.set_yticklabels(xlab,rotation='horizontal',verticalalignment='center',horizontalalignment='right',fontsize=text_sizes['xlabels'])

    if subplot is None:
        ax = handle_ticks(
            ax,
            x_major_tick_every,
            y_major_tick_every,
            x_minor_tick_every,
            y_minor_tick_every,
            new_figure=new_figure,
        )
        ax, add_to_axis_label = consider_scientific_notation(
            ax,
            axis="y",
            publication=default_parameters["set_publish"],
            fontsize=text_sizes["xlabels"],
        )  # at the moment we use all_tight as an alias for publication quality y/n
        if ylabel is not None:
            ylabel += add_to_axis_label
        ax, add_to_axis_label = consider_scientific_notation(
            ax,
            axis="x",
            publication=default_parameters["set_publish"],
            fontsize=text_sizes["xlabels"],
        )  # at the moment we use all_tight as an alias for publication quality y/n
        if xlabel is not None:
            xlabel += add_to_axis_label

    if xlabels is False:
        ax.set_xticklabels([])
    elif type(xlabels) is list or type(xlabels) is tuple or type(xlabels) is str:
        if type(xlabels) is str:
            xlabels = list(xlabels)  # probably a sequence
        extent = image.get_extent()
        addx = ((extent[1] - extent[0]) / float(matrix.shape[1])) / 2.0
        m, M = extent[0], extent[1]
        # print "DEB: extent=",extent,"m,M",m,M,'lims',ax.get_xlim(),'addx=',addx ,'len(numpy.linspace(m+addx,M-addx, len(xlabels)))',len(numpy.linspace(m+addx,M-addx, len(xlabels))) ,len(xlabels)
        # print 'xlabels[0:2]',xlabels[0:2],'...',xlabels[-2:],'at',numpy.linspace(m+addx,M-addx, len(xlabels))[:2],numpy.linspace(m+addx,M-addx, len(xlabels))[-2:]
        ax.set_xticks(numpy.linspace(m + addx, M - addx, len(xlabels)))
        if xlabels_fontname is not None:
            ax.set_xticklabels(
                xlabels,
                rotation=xlabels_rotation,
                verticalalignment="top",
                fontsize=xlabels_size,
                fontname=xlabels_fontname,
            )
        else:
            ax.set_xticklabels(
                xlabels,
                rotation=xlabels_rotation,
                verticalalignment="top",
                fontsize=xlabels_size,
            )
        plt.draw()
    if ylabels is False:
        ax.set_yticklabels([])
    elif type(ylabels) is list or type(ylabels) is tuple or type(ylabels) is str:
        if type(ylabels) is str:
            ylabels = list(ylabels)
        extent = image.get_extent()
        addy = ((extent[3] - extent[2]) / float(matrix.shape[0])) / 2.0
        m, M = extent[2], extent[3]
        ax.set_yticks(numpy.linspace(m + addy, M - addy, len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=ylabels_size)
    if y_range is None:
        extent = image.get_extent()
        m, M = extent[2], extent[3]
        dd = 0  # =(M-m)*0.03
        ax.set_ylim((m - dd, M + dd))
    if type(value_labels) is bool and value_labels == True:
        value_labels = Round_To_n(
            matrix, n=0, only_decimals=True
        )  # round to 1 significant digit in decimal numbers
    elif type(value_labels) is int:
        value_labels = numpy.round(matrix, value_labels)
    if value_labels is not None:
        extra_labels, fontweight = None, "normal"
        if type(value_labels) is tuple and len(value_labels) == 2:
            value_labels, extra_labels = value_labels
            fontweight = "bold"
        if (
            "shape" in dir(value_labels)
            and "shape" in dir(matrix)
            and value_labels.shape != matrix.shape
        ):
            sys.stderr.write(
                "***WARNING*** value_labels.shape != matrix.shape %s %s still trying to add\n"
                % (str(value_labels.shape), str(matrix.shape))
            )
        extent = image.get_extent()
        addx = ((extent[1] - extent[0]) / float(matrix.shape[1])) / 2.0
        m, M = extent[0], extent[1]
        xpos = numpy.linspace(m + addx, M - addx, matrix.shape[1])
        addy = ((extent[3] - extent[2]) / float(matrix.shape[0])) / 2.0
        m, M = extent[2], extent[3]
        ypos = numpy.linspace(m + addy, M - addy, matrix.shape[0])
        if value_labels_size is None:
            value_labels_size = text_sizes["value_labels"]
        if extra_labels_size is None:
            extra_labels_size = text_sizes["xlabels_many"]
            if type(extra_labels_size) is int:
                extra_labels_size /= 2.0
        for r in range(len(value_labels)):
            for c, vlab in enumerate(value_labels[r]):
                if type(vlab) is not str and not hasattr(vlab, "upper"):
                    vlab = "%g" % (
                        vlab
                    )  # second condition is for things like numpy._string
                kl = ax.annotate(
                    vlab,
                    xy=(xpos[c], ypos[-1 - r]),
                    xytext=(0, 0),
                    xycoords="data",
                    weight=fontweight,
                    zorder=zorder + 8,
                    size=value_labels_size,
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    rotation="horizontal"  # , bbox = dict(boxstyle = 'round,pad=0.3', fc = 'white', alpha = 1.) \
                    # , arrowprops = dict(arrowstyle = '-',connectionstyle=connectionstyle) \
                )
                if extra_labels is not None:
                    kl = ax.annotate(
                        extra_labels[r][c],
                        xy=(
                            xpos[c],
                            ypos[-1 - r]
                            - 0.03 * (value_labels_size + extra_labels_size) / 2.0,
                        ),
                        xytext=(0, 0),
                        xycoords="data",
                        weight="light",
                        fontstretch="ultra-condensed",
                        zorder=zorder + 7,
                        size=extra_labels_size,
                        textcoords="offset points",
                        ha="center",
                        va="center",
                        rotation="horizontal",
                    )
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=text_sizes["xlabel"], labelpad=10)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=text_sizes["ylabel"], labelpad=10)

    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)

    if title is not None:
        plt.text(
            0.5,
            1.03,
            title,
            horizontalalignment="center",
            fontsize=text_sizes["title"],
            transform=ax.transAxes,
        )
    if not frame:
        # this remove top and right axis
        # ax.spines["right"].set_visible(False)
        # ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()  # ticks only on bottom axis
        ax.get_yaxis().tick_left()  # ticks only on left axis

    if hgrid is not None:
        if type(hgrid) is bool:
            hgrid = int(hgrid)
        if type(hgrid) is int:
            if hgrid > 0:
                yticks = ax.yaxis.get_majorticklocs()[
                    ::-1
                ]  # matrix start from top, but ticks from bottom
                for yt in (yticks[:-1] + numpy.diff(yticks) / 2.0)[
                    hgrid - 1 :: hgrid
                ]:  # mid points between major ticks
                    plt.axhline(yt, color=grid_color, ls=":", lw=0.5)
        elif hasattr(
            hgrid, "__len__"
        ):  # put grids below these points assuming these are indices of matrix rows
            for yt in hgrid:
                plt.axhline(yt - 0.5, color=grid_color, ls=":", lw=0.5)
            # yticks=ax.yaxis.get_majorticklocs()[::-1]
            # all_candidate_locations= yticks[:-1]+numpy.diff(yticks)/2.# mid points between major ticks, select those given as input
            # hgrid=numpy.array(hgrid)
            # for yt in all_candidate_locations[ hgrid[(hgrid>0)&(hgrid<len(yticks))]-1 ] : # so that 1 in input is a line below top row of matrix, 0 in input is ignored
            #    plt.axhline(yt,color=grid_color,ls=':',lw=0.5)

    if vgrid is not None:
        if type(vgrid) is bool:
            vgrid = int(vgrid)
        if type(vgrid) is int:
            if vgrid > 0:
                xticks = ax.xaxis.get_majorticklocs()
                for yt in (xticks[:-1] + numpy.diff(xticks) / 2.0)[
                    vgrid - 1 :: vgrid
                ]:  # mid points between major ticks
                    plt.axvline(yt, color=grid_color, ls=":", lw=0.5)
        elif hasattr(
            vgrid, "__len__"
        ):  # put grids below these points assuming these are indices of matrix rows
            for yt in vgrid:
                plt.axvline(yt - 0.5, color=grid_color, ls=":", lw=0.5)
            # xticks=ax.xaxis.get_majorticklocs()
            # all_candidate_locations= xticks[:-1]+numpy.diff(xticks)/2.# mid points between major ticks, select those given as input
            # vgrid=numpy.array(vgrid)
            # for yt in all_candidate_locations[ vgrid[(vgrid>0)&(vgrid<len(xticks))]-1 ] : # so that 1 in input is a line below top row of matrix, 0 in input is ignored
            #    plt.axvline(yt,color=grid_color,ls=':',lw=0.5)

    if plot_colorbar:
        # if N_uniq_points is None : N_uniq_points=n_profiles
        # jump=N_uniq_points/10
        # if jump<=0 : jump=1
        # if cbar_labels is None : cbarlabels=None
        # else :
        #    cbarlabels= [ c if j in range(0,N_uniq_points,jump) else '' for j,c in enumerate(cbar_labels)]
        #    cbarlabels[-1]=cbar_labels[-1]
        # for j in xrange(jump-1) :cbarlabels[-2-j]=''
        # cbar_major_ntick=None
        figure = add_colorbar_to_seq_profile(
            figure,
            image,
            cbar_major_ntick=cbar_major_ntick,
            cbar_label=cbar_label,
            cbar_label_rotation=270,
            cbar_fraction=0.04,
        )
    """
    *fraction*    0.15; fraction of original axes to use for colorbar
    *pad*         0.05 if vertical, 0.15 if horizontal; fraction
                  of original axes between colorbar and new image axes
    *shrink*      1.0; fraction by which to shrink the colorbar
    *aspect*      20; ratio of long to short dimensions
    *anchor*      (0.0, 0.5) if vertical; (0.5, 1.0) if horizontal;
                  the anchor point of the colorbar axes
    *panchor*     (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal;
                  the anchor point of the colorbar parent axes. If
                  False, the parent axes' anchor will be unchanged
    ANd much more see help(plt.colorbar)
    """
    plt.draw()
    if save != False and save is not None:
        if ".png" in save:
            transparent = True
        else:
            transparent = False
        figure.savefig(
            save,
            dpi=default_figure_sizes["dpi"],
            bbox_inches="tight",
            bbox_extra_artists=[ax, figure],
            transparent=transparent,
        )  # ,bbox_inches="tight" is such that everything is in the figure... even though margins are removed.
        # plt.savefig(save, dpi=plt.gcf().dpi)
    if show:
        plt.show(block=block)
    return figure, image


def add_colorbar_to_seq_profile(
    figure,
    image_or_mappable,
    ax=None,
    cbar_major_ntick=None,
    cbar_labels=None,
    cbar_label=None,
    cbar_label_rotation=270,
    cbar_fraction=0.03,
    orientation="vertical",
    pad_fraction_between_colorbar_and_axes=0.0005,
):
    if cbar_major_ntick is None and default_parameters["set_publish"]:
        cbar_major_ntick = 5
    # figure.subplots_adjust(right=0.9)
    # cbar_ax = figure.add_axes([0.9, 0.15, 0.05, 0.7]) # Add an axes at position rect [left, bottom, width, height] where all quantities are in fractions of figure width and height
    # cbar=figure.colorbar(image, cax=cbar_ax,orientation='vertical',drawedges=False,spacing='proportional',ticks=None,aspect=30 )
    if ax is None:
        ax = figure.axes
    cbar = figure.colorbar(
        image_or_mappable,
        ax=ax,
        orientation=orientation,
        drawedges=False,
        spacing="uniform",
        aspect=25,
        ticks=None,
        fraction=cbar_fraction,
        pad=pad_fraction_between_colorbar_and_axes,
    )  # Uniform spacing gives each discrete color the same space; proportional makes the space proportional to the data interval.
    # the below is a workaround for bugs in some renderers which might results in white bars within colors
    # cbar.solids.set_edgecolor("face")
    if cbar_label is not None:
        cbar.set_label(
            cbar_label,
            fontsize=text_sizes["xlabel"],
            horizontalalignment="center",
            verticalalignment="top",
            rotation=cbar_label_rotation,
            labelpad=27,
        )
        # print "HERE"
    if cbar_major_ntick is not None:
        # print 'cbar_major_ntick',cbar_major_ntick
        if hasattr(cbar_major_ntick, "__len__"):
            ticks = cbar_major_ntick
        else:
            # The get_clim function was deprecated in Matplotlib 3.1 and will be removed in 3.3. Use ScalarMappable.get_clim instead.
            # cmin,cmax= cbar.get_clim()
            cmin, cmax = image_or_mappable.get_clim()
            ndig = -1 * magnitude(cmax - cmin)
            tmp_min = numpy.round(cmin, ndig)
            ticks = list(
                numpy.arange(
                    tmp_min,
                    cmax,
                    Round_To_n((cmax - cmin) / float(cbar_major_ntick), 0),
                )
            )
            if ndig < 0:
                ndig = 0
            ticks[0] = numpy.round(cmin, ndig)
            if abs(ticks[-1] - cmax) / (cmax - cmin) < 0.05:
                ticks[-1] = numpy.round(cmax, ndig)
            else:
                ticks += [numpy.round(cmax, ndig)]
        # print 'cabar ticks',ticks,cmin,cmax,ndig
        cbar.set_ticks(ticks)
        if cbar_labels is not None:
            cbar.set_ticklabels(cbar_labels)
        # cmajorLocator   = matplotlib.ticker.MultipleLocator(cbar_major_tick_every)
        # bar.ax.yaxis.set_major_locator(cmajorLocator)  #JUST NOT WORKING
        # cbar.update_ticks()
    plt.draw()
    return figure


aa_markers = {
    "-": "_",
    "A": "o",
    "D": "v",
    "E": "^",
    "R": "<",
    "K": ">",
    "G": ".",
    "L": "1",
    "I": "2",
    "V": "3",
    "T": "4",
    "M": "8",
    "F": "s",
    "P": "p",
    "C": "*",
    "H": "h",
    "S": 7,
    "N": "+",
    "Q": "x",
    "W": "D",
    "Y": "d",
}
tiny_markers = ["_", "1", "2", "3", "4", "+", "x"]


def volcano_plot_of_MSA_comparison(
    diff_pssm,
    ttest_pvals,
    cmap_name="gist_rainbow",
    markersize=70,
    alpha=0.8,
    aa_order=[
        "-",
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
    ],
    markers_to_use=aa_markers,
    tiny_markers=tiny_markers,
    x_range=None,
    y_range=None,
    legend_size=20,
    show=True,
    dpi=None,
    save=None,
):
    colors = colors_from_color_map(
        list(range(0, ttest_pvals.shape[1])), cmap_name=cmap_name
    )
    figure = None
    if y_range is None:
        y_range = (0, None)
    if x_range is None:
        xMin, xMax = numpy.nanmin([numpy.nanmin(p) for p in diff_pssm]), numpy.nanmax(
            [numpy.nanmax(p) for p in diff_pssm]
        )
        sp = 0.05 * (xMax - xMin)
        x_range = (xMin - sp, xMax + sp)
    for j, aa in enumerate(aa_order):
        if type(markers_to_use) is dict or isinstance(markers_to_use, OrderedDict):
            mm = markers_to_use[aa]
        else:
            mm = markers_to_use[j]
        if tiny_markers is not None and mm in tiny_markers:
            msize = 2 * markersize
        else:
            msize = markersize
        figure = scatter(
            diff_pssm[j],
            -numpy.log10(ttest_pvals[j]),
            y_range=y_range,
            x_range=x_range,
            markersize=msize,
            alpha=alpha,
            hline=-numpy.log10(0.05),
            markerfacecolor=colors,
            marker=mm,
            label=aa,
            figure=figure,
            show=False,
        )
    plt.legend(
        loc="right",
        bbox_to_anchor=(1.2, 0.5),
        prop={"size": legend_size},
        scatterpoints=1,
        frameon=False,
        borderpad=0.01,
        handletextpad=0.0,
    )
    sm = plt.cm.ScalarMappable(
        cmap=cmap_name, norm=plt.Normalize(vmin=1, vmax=ttest_pvals.shape[1] + 1)
    )
    sm._A = []
    cbar = figure.colorbar(
        sm,
        spacing="proportional",
        drawedges=False,
        ticks=[1] + list(range(10, ttest_pvals.shape[1] + 1, 10)),
        alpha=alpha,
        fraction=0.1,
        pad=0.16,
    )  # pad regulates the distance from the ax
    plt.draw()
    if show:
        plt.show(block=False)
    if save is not None and save != "":
        if "." not in save:
            save += ".pdf"
        if dpi is None:
            dpi = default_figure_sizes["dpi"]
        figure.savefig(
            save, dpi=dpi, bbox_inches="tight", transparent=True
        )  #  bbox_inches=0 remove white space around the figure.. ,
    return figure
