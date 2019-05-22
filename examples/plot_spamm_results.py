#! /usr/bin/env python

import os
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import subprocess

import triangle
from spamm.Samples import Samples
from spamm.Analysis import (median_values, mean_values, plot_chains,
    plot_posteriors)

#-----------------------------------------------------------------------------#

def plot_posteriors_pdf(S):
    pdfname = os.path.join(S.outdir, "{}_posterior.pdf".format(S.model_name))
    pdf_pages = PdfPages(pdfname)    

    for i in range(S.total_parameter_count):
        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(111)
        
        chain = S.samples[:,i]
        hist,bins = np.histogram(chain, S.histbins)
        binsize = bins[1]-bins[0]

        maxm = S.maxs[i]
        med = S.medians[i] + binsize/2.
        avg = S.means[i] + binsize/2.
        mode = S.modes[i] + binsize/2.
        
        std = np.std(chain)
        
        if S.params is not None:
            try:
                actual = S.params[S.model_parameter_names[i]]
#                print("New limits for {}:\n{:0.20f}\n{:0.20f}\n".format(S.model_parameter_names[i], maxm-(1.5*std), maxm+(1.5*std)))
                ax.axvspan(actual-std, actual+std, facecolor="grey", 
                           alpha=0.25, label=r"1$\sigma$={:1.3e}".format(std))
                ax.axvline(S.params[S.model_parameter_names[i]], 
                           color="red", linestyle="solid", linewidth=1.5, 
                           label="Actual value={:1.3e}".format(actual))
            except KeyError:
                actual = maxm
                ax.axvspan(actual-std, actual+std, facecolor="grey", 
                           alpha=0.25, label=r"1$\sigma$={:1.3e}".format(std))
                 
        ax.hist(chain, bins, color="skyblue")

        xlo = actual - binsize*12
        xhi = actual + binsize*12
        
        vmin = min([med, avg, maxm])
        vmax = max([med, avg, maxm])

        if vmin <= xlo:
            xlo = vmin - binsize
        if vmax >= xhi:
            xhi = vmax + binsize

        ax.set_xlim(xlo, xhi)
        
#        ax.axvline(center, color="red", linestyle="dotted", linewidth=1.5, label="Max")
        ax.axvline(avg, color="darkblue", linestyle="--", linewidth=1.5, 
                   label="Mean={:1.3e}".format(avg))
        ax.axvline(med, color="darkviolet", linestyle="--", linewidth=1.5, 
                   label="Median={:1.3e}".format(med))
        ax.axvline(mode, color="blue", linestyle="--", linewidth=1.5, 
                   label="Mode={:1.3e}".format(mode))
        ax.axvline(maxm, color="fuchsia", linestyle="--", linewidth=1.5, 
                   label="Maximum={:1.3e}".format(maxm))
        ax.legend(loc="best")

        ax.set_xlabel(S.model_parameter_names[i])
        ax.set_ylabel("Posterior PDF")
        ax.set_title(S.model_parameter_names[i])

        pdf_pages.savefig(fig)

    pdf_pages.close()
    print("Saved {}".format(pdfname))

#--------------------------------------------------------------------------#

def plot_best_models(S):
    data_spectrum = S.model.data_spectrum
    actualcolor = "deepskyblue"
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(111)
    ax.errorbar(data_spectrum.spectral_axis, data_spectrum.flux,
                    data_spectrum.flux_error, mfc=actualcolor, mec=actualcolor,
                    ecolor=actualcolor, fmt=".", zorder=-100, label="Actual Flux") 
    ax.plot(data_spectrum.spectral_axis,
            S.model.model_flux(params=S.means),
            color="darkblue", label="Mean")
    ax.plot(data_spectrum.spectral_axis,
            S.model.model_flux(params=S.medians),
            color="darkviolet", label="Median")
    ax.plot(data_spectrum.spectral_axis,
            S.model.model_flux(params=S.modes),
            color="blue", label="Mode")
    ax.plot(data_spectrum.spectral_axis,
            S.model.model_flux(params=S.maxs),
            color="fuchsia", label="Max")
    ax.set_title("Best Fits")
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel(r"ergs/s/cm$^2$")
    ax.legend(loc="upper left", framealpha=0.25)
    figname = "{}_best.png"
    fig.savefig(os.path.join(S.outdir, figname))
    print("\tSaved {}".format(figname))

#--------------------------------------------------------------------------#

def plot_models(S, ymax=None):
    data_spectrum = S.model.data_spectrum
    actualcolor = "deepskyblue"
    gifdir = os.path.join(S.outdir, "gifplots_" + S.model_name)
    if not os.path.exists(gifdir):
        os.mkdir(gifdir)

    actual_comps = {}
    for component in S.model.components:
        actual_params = [S.params[x] for x in component.model_parameter_names]
        actual_comp_flux = component.flux(spectrum=data_spectrum,
                                          parameters=actual_params)
        actual_comps[component.name] = actual_comp_flux

    if S.last is True:
        sample_range = [len(S.samples)-1]
    else:
        sample_range = range(0, len(S.samples), S.step)
    for i in sample_range:
        print("Iteration {}".format(i))
        j = 0
        for component in S.model.components:
            fig = plt.figure(figsize=(15,7))
            ax = fig.add_subplot(111)
            ax.plot(data_spectrum.spectral_axis, actual_comps[component.name],
                    color=actualcolor, label="Actual Flux")
            
            comp_flux = component.flux(spectrum=data_spectrum,
                                       parameters=S.samples[i, j:j+len(component.model_parameter_names)])
            ax.plot(data_spectrum.spectral_axis, comp_flux, color="deeppink", 
                    label="Model Flux")
            compmax = max(actual_comps[component.name])
            if ymax is None:
                ymax = compmax + .1*compmax
            ax.set_ylim(0, ymax)
            ax.set_title("{}, Iteration {}".format(component.name, i))
            ax.set_xlabel(r"Wavelength [$\AA$]")
            ax.set_ylabel(r"ergs/s/cm$^2$")
            ax.legend(loc="upper left", framealpha=0.25)
            figname = os.path.join(gifdir, "{}_iter{:06d}.png".format(component.name, i))
            fig.savefig(os.path.join(S.outdir, figname))
            if S.last is True:
                print("\tSaved {}".format(figname))
            j += len(component.model_parameter_names)
            plt.close(fig)

        model_spectrum = S.model.model_flux(params=S.samples[i,:])
        fig = plt.figure(figsize=(15,7))
        ax = fig.add_subplot(111)
        ax.errorbar(data_spectrum.spectral_axis, data_spectrum.flux,
                    data_spectrum.flux_error, mfc=actualcolor, mec=actualcolor,
                    ecolor=actualcolor, fmt=".", zorder=-100, label="Actual Flux") 
        ax.plot(data_spectrum.spectral_axis, model_spectrum, 
                color="deeppink", label="Model Flux")
        modelmax = max(data_spectrum.flux)
        if ymax is None:
            ymax = modelmax + .1*modelmax
        ax.set_ylim(0, ymax)
        ax.set_title("Sum Of Model Components, Iteration {}".format(i))
        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel(r"ergs/s/cm$^2$")
        ax.legend(loc="upper left", framealpha=0.25)
        figname = os.path.join(outdir, "model_iter{:06d}.png".format(i))
        fig.savefig(os.path.join(S.outdir, figname))
        if S.last is True:
            print("\tSaved {}".format(figname))
        plt.close(fig)

    if S.gif is True:
        for component in S.model.components:
            cname = component.name
            gifname = os.path.join(outdir, "{}.gif".format(cname))
            subprocess.check_call(["convert", "-delay", "15", "-loop", "1", 
                                   os.path.join(outdir, "{}*png".format(cname)), 
                                   gifname])
            print("\tSaved {}".format(gifname))
        gifname = os.path.join(outdir, "{}.gif".format(S.model_name))
        subprocess.check_call(["convert", "-delay", "15", "-loop", "1", 
                               os.path.join(outdir, "model*png".format(S.model_name)), 
                               gifname])
        print("\tSaved {}".format(gifname))
    
#--------------------------------------------------------------------------#

def make_all_plots(S):

    # Create the triangle plot.
    fig = triangle.corner(S.samples, labels=S.model_parameter_names)
    figname = "{0}_triangle.png".format(S.model_name)
    fig.savefig(os.path.join(S.outdir, figname))
    print("\tSaved {0}".format(figname))

    # Plot the MCMC chains as a function of iteration. You can easily tell if 
    # the chains are converged because you can no longer tell where the individual 
    # particle chains are sliced together.
    fig = plot_chains(S.samples, labels=S.model_parameter_names)
    figname = "{0}_chain.png".format(S.model_name)
    fig.savefig(os.path.join(S.outdir, figname))
    print("\tSaved {0}".format(figname))

    # Plot the posterior PDFs for each parameter. These are histograms of the 
    # MCMC chains. Boxes = 20 is the default.
    fig = plot_posteriors(S.samples, labels=S.model_parameter_names, 
                          boxes=20, params=S.params)
    figname = "{0}_posterior.png".format(S.model_name)
    fig.savefig(os.path.join(S.outdir, figname))
    print("\tSaved {0}".format(figname))

    # Make a PDF (the plot kind!) with the histogram for each parameter a
    # different page in the PDF for detailed analysis.
    plot_posteriors_pdf(S)

    # Make a gif of the model spectrum as a function of iteration,
    # or if last is True, only save the last model spectrum.
    if S.gif is True:
        plot_models(S)

#-----------------------------------------------------------------------------#

def make_plots_from_pickle(pname, outdir, gif=False, last=False, step=100):
    S = Samples(pname, outdir=outdir, gif=gif, last=last, step=step)
    make_all_plots(S)

#-----------------------------------------------------------------------------#

def first_concat():
    models = ["model_20180807_0442.pickle.gz", "model_20180807_2917.pickle.gz", "model_20180807_3906.pickle.gz", "model_20180808_0244.pickle.gz", "model_20180810_0857.pickle.gz", "model_20180811_0849.pickle.gz", "model_20180812_2600.pickle.gz", "model_20180814_2104.pickle.gz"]
    S = Samples(models)
    make_all_plots(S)

#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pname", help="SPAMM model pickle file", type=str)
    parser.add_argument("--o", dest="outdir", default="plots", 
                        help="Name of output directory to save plots in")
    parser.add_argument("--gif", dest="gif", action="store_true", default=False,
                        help="Switch to make plots to create gif")
    parser.add_argument("--last", dest="last", action="store_true", default=False,
                        help="Switch to only plot last model iteration")
    parser.add_argument("--step", dest="step", default=100,
                        help="Step size for plotting chain iterations")
    args = parser.parse_args()

    make_plots_from_pickle(args.pname, args.outdir, args.gif, args.last, int(args.step))
