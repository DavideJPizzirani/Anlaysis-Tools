import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib.ticker import AutoLocator, AutoMinorLocator

#----------------------------------------------------------------#
    
#                         SIMPLE PLOTTING 

#                           np.ndarray

#----------------------------------------------------------------#

def simple_plot_arrays(array, x_label, y_label, title, label, format, save):
    assert type(x_label) is str
    assert type(y_label) is str
    assert type(title) is str
    assert type(format) is str and format in ["line","scatter"]
    assert type(save) is str and save in ["yes", "no"]
    if save == "yes" and format == "line":
        fig, ax = plt.subplots()
        ax.get_xaxis().set_major_locator(AutoLocator())
        ax.get_xaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.get_yaxis().set_major_locator(AutoLocator())
        ax.get_yaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.autoscale(enable=True, axis="both", tight=False)

        ax.grid(which="major", axis="both", linewidth=0.5, color="green")
        ax.grid(which="minor", axis="both", linewidth=0.5, color="lightgreen")

        ax.tick_params(which="both", direction="in",
                    width=2, pad=10, top=True, right=True)
        ax.tick_params(which="major", length=8)
        ax.tick_params(which="minor", length=4)

        ax.set_xlabel(x_label, fontsize=20, labelpad=14)
        ax.set_ylabel(y_label, fontsize=20, labelpad=14)
        ax.set_title(title, fontsize=24, pad=14)
        ax.plot(*zip(*array), label = label)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()
        plt.legend(loc="best")
        fig.savefig("test.png")
        plt.show()
    if save == "no" and format == "line":
        fig, ax = plt.subplots()
        ax.get_xaxis().set_major_locator(AutoLocator())
        ax.get_xaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.get_yaxis().set_major_locator(AutoLocator())
        ax.get_yaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.autoscale(enable=True, axis="both", tight=False)

        ax.grid(which="major", axis="both", linewidth=0.5, color="green")
        ax.grid(which="minor", axis="both", linewidth=0.5, color="lightgreen")

        ax.tick_params(which="both", direction="in",
                    width=2, pad=10, top=True, right=True)
        ax.tick_params(which="major", length=8)
        ax.tick_params(which="minor", length=4)

        ax.set_xlabel(x_label, fontsize=20, labelpad=14)
        ax.set_ylabel(y_label, fontsize=20, labelpad=14)
        ax.set_title(title, fontsize=24, pad=14)
        ax.plot(*zip(*array), label = label)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()
        plt.legend(loc="best")
        plt.show()
    if save == "yes" and format == "scatter":
        fig, ax = plt.subplots()
        ax.get_xaxis().set_major_locator(AutoLocator())
        ax.get_xaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.get_yaxis().set_major_locator(AutoLocator())
        ax.get_yaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.autoscale(enable=True, axis="both", tight=False)

        ax.grid(which="major", axis="both", linewidth=0.5, color="green")
        ax.grid(which="minor", axis="both", linewidth=0.5, color="lightgreen")

        ax.tick_params(which="both", direction="in",
                    width=2, pad=10, top=True, right=True)
        ax.tick_params(which="major", length=8)
        ax.tick_params(which="minor", length=4)

        ax.set_xlabel(x_label, fontsize=20, labelpad=14)
        ax.set_ylabel(y_label, fontsize=20, labelpad=14)
        ax.set_title(title, fontsize=24, pad=14)
        ax.scatter(*zip(*array), label = label)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()
        plt.legend(loc="best")
        fig.savefig("test.png")
        plt.show()
    if save == "no" and format == "scatter":
        fig, ax = plt.subplots()
        ax.get_xaxis().set_major_locator(AutoLocator())
        ax.get_xaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.get_yaxis().set_major_locator(AutoLocator())
        ax.get_yaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.autoscale(enable=True, axis="both", tight=False)

        ax.grid(which="major", axis="both", linewidth=0.5, color="green")
        ax.grid(which="minor", axis="both", linewidth=0.5, color="lightgreen")

        ax.tick_params(which="both", direction="in",
                    width=2, pad=10, top=True, right=True)
        ax.tick_params(which="major", length=8)
        ax.tick_params(which="minor", length=4)

        ax.set_xlabel(x_label, fontsize=20, labelpad=14)
        ax.set_ylabel(y_label, fontsize=20, labelpad=14)
        ax.set_title(title, fontsize=24, pad=14)
        ax.scatter(*zip(*array), label = label)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()
        plt.legend(loc="best")
        plt.show()
    return fig , ax




#----------------------------------------------------------------#
    
#                         SIMPLE PLOTTING 

#                             pandas

#----------------------------------------------------------------#


def simple_plot_df_Tsweep(dataset, format, label, save = False, x_label = "T(K)", y_label = "R", title = "R(T)"):
    assert type(label) is str
    assert type(x_label) is str
    assert type(y_label) is str
    assert type(title) is str
    assert type(format) is str and format in ["line","scatter"]
    plt.rcParams["font.family"] = "Times New Roman"
    if save and format == "line":
        fig, ax = plt.subplots()
        ax.get_xaxis().set_major_locator(AutoLocator())
        ax.get_xaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.get_yaxis().set_major_locator(AutoLocator())
        ax.get_yaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.autoscale(enable=True, axis="both", tight=False)

        ax.grid(which="major", axis="both", linewidth=0.5, color="green")
        ax.grid(which="minor", axis="both", linewidth=0.5, color="lightgreen")

        ax.tick_params(which="both", direction="in",
                    width=1.5, pad=10, top=True, right=True)
        ax.tick_params(which="both", length=8, labelsize=14)
        ax.tick_params(which="both", length=4, labelsize=14)

        
        ax.set_xlabel(x_label, fontsize=20, labelpad=14)
        ax.set_ylabel(y_label, fontsize=20, labelpad=14)
        ax.set_title(title, fontsize=24, pad=14)
        for df in dataset:
            ax.plot(df.x, df.y, label = label, linewidth = 2.5)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()
        plt.legend(loc="best", fontsize=12)
        fig.savefig("test.png")
        plt.show()
    if not save and format == "line":
        fig, ax = plt.subplots()
        ax.get_xaxis().set_major_locator(AutoLocator())
        ax.get_xaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.get_yaxis().set_major_locator(AutoLocator())
        ax.get_yaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.autoscale(enable=True, axis="both", tight=False)

        ax.grid(which="major", axis="both", linewidth=0.5, color="green")
        ax.grid(which="minor", axis="both", linewidth=0.5, color="lightgreen")

        ax.tick_params(which="both", direction="in",
                    width=1.5, pad=10, top=True, right=True)
        ax.tick_params(which="both", length=8, labelsize=14)
        ax.tick_params(which="both", length=4, labelsize=14)

        
        ax.set_xlabel(x_label, fontsize=20, labelpad=14)
        ax.set_ylabel(y_label, fontsize=20, labelpad=14)
        ax.set_title(title, fontsize=24, pad=14)
        for df in dataset:
            ax.plot(df.x, df.y, label = label, linewidth = 2.5)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()
        plt.legend(loc="best", fontsize=12)
        plt.show()
    if save and format == "scatter":
        fig, ax = plt.subplots()
        ax.get_xaxis().set_major_locator(AutoLocator())
        ax.get_xaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.get_yaxis().set_major_locator(AutoLocator())
        ax.get_yaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.autoscale(enable=True, axis="both", tight=False)

        ax.grid(which="major", axis="both", linewidth=0.5, color="green")
        ax.grid(which="minor", axis="both", linewidth=0.5, color="lightgreen")

        ax.tick_params(which="both", direction="in",
                    width=1.5, pad=10, top=True, right=True)
        ax.tick_params(which="both", length=8, labelsize=14)
        ax.tick_params(which="both", length=4, labelsize=14)

        
        ax.set_xlabel(x_label, fontsize=20, labelpad=14)
        ax.set_ylabel(y_label, fontsize=20, labelpad=14)
        ax.set_title(title, fontsize=24, pad=14)
        for df in dataset:
            ax.scatter(df.x, df.y, label = label, linewidth = 2.5)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()
        plt.legend(loc="best", fontsize=12)
        fig.savefig("test.png")
        plt.show()
    if not save and format == "scatter":
        fig, ax = plt.subplots()
        ax.get_xaxis().set_major_locator(AutoLocator())
        ax.get_xaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.get_yaxis().set_major_locator(AutoLocator())
        ax.get_yaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.autoscale(enable=True, axis="both", tight=False)

        ax.grid(which="major", axis="both", linewidth=0.5, color="green")
        ax.grid(which="minor", axis="both", linewidth=0.5, color="lightgreen")

        ax.tick_params(which="both", direction="in",
                    width=1.5, pad=10, top=True, right=True)
        ax.tick_params(which="both", length=8, labelsize=14)
        ax.tick_params(which="both", length=4, labelsize=14)

        
        ax.set_xlabel(x_label, fontsize=20, labelpad=14)
        ax.set_ylabel(y_label, fontsize=20, labelpad=14)
        ax.set_title(title, fontsize=24, pad=14)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        for df in dataset:
            ax.scatter(df.x, df.y, label = label, linewidth = 2.5)
        ax.grid()
        plt.legend(loc="best", fontsize=12)  
        plt.show()
    return fig , ax



def simple_plot_df_signal(dataset, format, label, title, save = False, x_label = "B (T)", y_label = "R"):
    assert type(label) is str
    assert type(x_label) is str
    assert type(y_label) is str
    assert type(title) is str
    assert type(format) is str and format in ["line","scatter"]
    plt.rcParams["font.family"] = "Times New Roman"
    if save and format == "line":
        fig, ax = plt.subplots()
        ax.get_xaxis().set_major_locator(AutoLocator())
        ax.get_xaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.get_yaxis().set_major_locator(AutoLocator())
        ax.get_yaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.autoscale(enable=True, axis="both", tight=False)

        ax.grid(which="major", axis="both", linewidth=0.5, color="green")
        ax.grid(which="minor", axis="both", linewidth=0.5, color="lightgreen")

        ax.tick_params(which="both", direction="in",
                    width=1.5, pad=10, top=True, right=True)
        ax.tick_params(which="both", length=8, labelsize=14)
        ax.tick_params(which="both", length=4, labelsize=14)

        
        ax.set_xlabel(x_label, fontsize=20, labelpad=14)
        ax.set_ylabel(y_label, fontsize=20, labelpad=14)
        ax.set_title(title, fontsize=24, pad=14)
        for df in dataset:
            ax.plot(df.x, df.y, label = label, linewidth = 2.5)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()
        plt.legend(loc="best", fontsize=12)
        fig.savefig("test.png")
        plt.show()
    if not save and format == "line":
        fig, ax = plt.subplots()
        ax.get_xaxis().set_major_locator(AutoLocator())
        ax.get_xaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.get_yaxis().set_major_locator(AutoLocator())
        ax.get_yaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.autoscale(enable=True, axis="both", tight=False)

        ax.grid(which="major", axis="both", linewidth=0.5, color="green")
        ax.grid(which="minor", axis="both", linewidth=0.5, color="lightgreen")

        ax.tick_params(which="both", direction="in",
                    width=1.5, pad=10, top=True, right=True)
        ax.tick_params(which="both", length=8, labelsize=14)
        ax.tick_params(which="both", length=4, labelsize=14)

        
        ax.set_xlabel(x_label, fontsize=20, labelpad=14)
        ax.set_ylabel(y_label, fontsize=20, labelpad=14)
        ax.set_title(title, fontsize=24, pad=14)
        for df in dataset:
            ax.plot(df.x, df.y, label = label, linewidth = 2.5)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()
        plt.legend(loc="best", fontsize=12)
        plt.show()
    if save and format == "scatter":
        fig, ax = plt.subplots()
        ax.get_xaxis().set_major_locator(AutoLocator())
        ax.get_xaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.get_yaxis().set_major_locator(AutoLocator())
        ax.get_yaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.autoscale(enable=True, axis="both", tight=False)

        ax.grid(which="major", axis="both", linewidth=0.5, color="green")
        ax.grid(which="minor", axis="both", linewidth=0.5, color="lightgreen")

        ax.tick_params(which="both", direction="in",
                    width=1.5, pad=10, top=True, right=True)
        ax.tick_params(which="both", length=8, labelsize=14)
        ax.tick_params(which="both", length=4, labelsize=14)

        
        ax.set_xlabel(x_label, fontsize=20, labelpad=14)
        ax.set_ylabel(y_label, fontsize=20, labelpad=14)
        ax.set_title(title, fontsize=24, pad=14)
        for df in dataset:
            ax.scatter(df.x, df.y, label = label, linewidth = 2.5)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()
        plt.legend(loc="best", fontsize=12)
        fig.savefig("test.png")
        plt.show()
    if not save and format == "scatter":
        fig, ax = plt.subplots()
        ax.get_xaxis().set_major_locator(AutoLocator())
        ax.get_xaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.get_yaxis().set_major_locator(AutoLocator())
        ax.get_yaxis().set_minor_locator(AutoMinorLocator(n=2))

        ax.autoscale(enable=True, axis="both", tight=False)

        ax.grid(which="major", axis="both", linewidth=0.5, color="green")
        ax.grid(which="minor", axis="both", linewidth=0.5, color="lightgreen")

        ax.tick_params(which="both", direction="in",
                    width=1.5, pad=10, top=True, right=True)
        ax.tick_params(which="both", length=8, labelsize=14)
        ax.tick_params(which="both", length=4, labelsize=14)

        
        ax.set_xlabel(x_label, fontsize=20, labelpad=14)
        ax.set_ylabel(y_label, fontsize=20, labelpad=14)
        ax.set_title(title, fontsize=24, pad=14)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        for df in dataset:
            ax.scatter(df.x, df.y, label = label, linewidth = 2.5)
        ax.grid()
        plt.legend(loc="best", fontsize=12)  
        plt.show()
    return fig , ax