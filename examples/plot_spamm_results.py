#! /usr/bin/env python

import os
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import gzip
import numpy as np
import subprocess
import statistics

import triangle
from spamm.Model import Model
from spamm.Analysis import (median_values, mean_values, plot_chains,
    plot_posteriors)

#-----------------------------------------------------------------------------#
class Samples(object):
#class Samples(Model):
    def __init__(self, pickle_files, burn=50):
        self.model, self.samples, self.params, self.model_name = get_samples(pickle_files, burn)
        self.total_parameter_count = self.model.total_parameter_count
        self.model_parameter_names = self.model.model_parameter_names()
        self.get_stats()

#-----------------------------------------------------------------------------#

    def get_stats(self, histbins=100):
        self.means = []
        self.medians = []
        self.modes = []
        self.maxs = []
        for i in range(self.model.total_parameter_count):
            hist, bins = np.histogram(chain, histbins)
            binsize = bins[1]-bins[0]

            # Calculate maximum, median, average, and mode
            max_bin = np.max(bins)
            self.maxs.append(max_bin + binsize/2.)
            self.medians.append(np.median(chain))
            self.means.append(np.average(chain))
            self.modes.append(statistics.mode(chain))

#-----------------------------------------------------------------------------#

def read_pickle(pname):
    try:
        p_data = pickle.loads(gzip.open(pname).read())
    except UnicodeDecodeError:
        p_data = pickle.loads(gzip.open(pname).read(), encoding="latin1")

    model = p_data["model"]
    params = p_data["comp_params"]
    if pname == "model_20180627_1534.pickle.gz":
        params = {'fe_norm_2': 3.5356725072091589e-15, 
                  'fe_norm_3': 8.9351374726858118e-15, 
                  'no_templates': 3, 
                  'fe_width': 4208.055598607859, 
                  'fe_norm_1': 9.4233576501633248e-15}
    elif pname == "model_20180627_4259.pickle.gz":
        params = {'fe_norm_2': 8.68930476e-15, 
                  'fe_width': 5450,
                  'no_templates': 3, 
                  'fe_norm_1': 1.07988504e-14, 
                  'fe_norm_3': 6.91877436e-15,
                  'norm_PL': 5e-15,
                  'slope1': 2.5}
    elif pname == "model_20180711_4746.pickle.gz":
        params = {'bc_Te': 50250.0,
                  'bc_lines': 201.5,
                  'bc_loffset': 0.0,
                  'bc_logNe': 5.5,
                  'bc_lwidth': 5050.0,
                  'bc_norm': 3e-14,
                  'bc_tauBE': 1.0,
                  'broken_pl': False,
                  'fe_norm_1': 1.07988504e-14,
                  'fe_norm_2': 8.68930476e-15,
                  'fe_norm_3': 6.91877436e-15,
                  'fe_width': 5450,
                  'norm_PL': 5e-15,
                  'slope1': 2.5}
    elif pname == "model_20180711_3027.pickle.gz":
        params['fe_norm_1'] = 1.07988504e-14
        params['fe_norm_2'] = 8.68930476e-15
        params['fe_norm_3'] = 6.91877436e-15
        params['fe_width'] = 5450 

    
    return model, params

#-----------------------------------------------------------------------------#

def plot_posteriors_pdf(pdfname, samples, labels, params=None, histbins=100):
    num_params = np.size(samples[0,:])                                   
    pdf_pages = PdfPages(pdfname)    

    for i in range(num_params):
        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(111)
        
        chain = samples[:,i]
        hist,bins = np.histogram(chain, histbins)

        maxind = np.argmax(hist)
        max_bin = bins[maxind] 
        binsize = bins[1]-bins[0]
        maxm = max_bin + binsize/2.

        med_bin = np.median(chain)
        med = med_bin + binsize/2.

        avg_bin = np.average(chain)
        avg = avg_bin + binsize/2.
        
        ax.hist(chain, bins, color="skyblue")
        
        if params is not None:
            try:
                actual = params[labels[i]]
                std = np.std(chain)
                ax.axvspan(actual-std, actual+std, facecolor="grey", alpha=0.25, label=r"1$\sigma$={:1.3e}".format(std))
                ax.axvline(params[labels[i]], color="red", linestyle="solid", linewidth=1.5, label="Actual value={:1.3e}".format(actual))
            except KeyError:
                pass
        
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
        ax.axvline(med, color="darkviolet", linestyle="--", linewidth=1.5, label="Median={:1.3e}".format(med))
        ax.axvline(avg, color="darkblue", linestyle="--", linewidth=1.5, label="Mean={:1.3e}".format(avg))
        ax.axvline(maxm, color="fuchsia", linestyle="--", linewidth=1.5, label="Maximum={:1.3e}".format(maxm))
        ax.legend(loc="best")

        ax.set_xlabel(labels[i])
        ax.set_ylabel("Posterior PDF")
        ax.set_title(labels[i])

        pdf_pages.savefig(fig)

    pdf_pages.close()
    print("Saved {}".format(pdfname))

#-----------------------------------------------------------------------------#

def plot_models(model, samples, model_name, params, ymax=None, make_gif=True, 
                step=100, only_last=False, only_best=False):
    data_spectrum = model.data_spectrum
    actualcolor = "deepskyblue"
    outdir = "gifplots_" + model_name
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    actual_comps = {}
    for component in model.components:
        actual_params = [params[x] for x in component.model_parameter_names]
        actual_comp_flux = component.flux(spectrum=data_spectrum,
                                          parameters=actual_params)
        actual_comps[component.name] = actual_comp_flux

    if only_last is True:
        sample_range = [len(samples)-1]
    else:
        sample_range = range(0, len(samples), step)
    for i in sample_range:
        print("Iteration {}".format(i))
        j = 0
        for component in model.components:
            fig = plt.figure(figsize=(15,7))
            ax = fig.add_subplot(111)
            ax.plot(data_spectrum.wavelengths, actual_comps[component.name],
                    color=actualcolor, label="Actual Flux")
            
            comp_flux = component.flux(spectrum=data_spectrum,
                                       parameters=samples[i, j:j+len(component.model_parameter_names)])
            ax.plot(data_spectrum.wavelengths, comp_flux, color="deeppink", label="Model Flux")
            compmax = max(actual_comps[component.name])
            if ymax is None:
                ymax = compmax + .1*compmax
            ax.set_ylim(0, ymax)
            ax.set_title("{}, Iteration {}".format(component.name, i))
            ax.set_xlabel(r"Wavelength [$\AA$]")
            ax.set_ylabel(r"ergs/s/cm$^2$")
            ax.legend(loc="upper left", framealpha=0.25)
            figname = os.path.join(outdir, "{}_iter{:06d}.png".format(component.name, i))
            fig.savefig(figname)
            if only_last is True:
                print("Saved {}".format(figname))
            j += len(component.model_parameter_names)
            plt.close(fig)

        model_spectrum = model.model_flux(params=samples[i,:])
        fig = plt.figure(figsize=(15,7))
        ax = fig.add_subplot(111)
        ax.errorbar(data_spectrum.wavelengths, data_spectrum.flux,
                    data_spectrum.flux_error, mfc=actualcolor, mec=actualcolor,
                    ecolor=actualcolor, fmt=".", zorder=-100, label="Actual Flux") 
        ax.plot(data_spectrum.wavelengths, model_spectrum, color="deeppink", label="Model Flux")
        modelmax = max(data_spectrum.flux)
        if ymax is None:
            ymax = modelmax + .1*modelmax
        ax.set_ylim(0, ymax)
        ax.set_title("Sum Of Model Components, Iteration {}".format(i))
        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel(r"ergs/s/cm$^2$")
        ax.legend(loc="upper left", framealpha=0.25)
        figname = os.path.join(outdir, "model_iter{:06d}.png".format(i))
        fig.savefig(figname)
        if only_last is True:
            print("Saved {}".format(figname))
        plt.close(fig)

    if make_gif is True:
        for component in model.components:
            cname = component.name
            gifname = os.path.join(outdir, "{}.gif".format(cname))
            subprocess.check_call(["convert", "-delay", "15", "-loop", "1", 
                                   os.path.join(outdir, "{}*png".format(cname)), 
                                   gifname])
            print("Saved {}".format(gifname))
        gifname = os.path.join(outdir, "{}.gif".format(model_name))
        subprocess.check_call(["convert", "-delay", "15", "-loop", "1", 
                               os.path.join(outdir, "model*png".format(model_name)), 
                               gifname])
        print("Saved {}".format(gifname))
        

#-----------------------------------------------------------------------------#

def get_samples(pname, burn=50):
    """ 
    Some pickled model files I made were incorrectly made and a try/except
    needs to be inserted when creating sample:
        try:
            sample = model.sampler.chain[:, burn:, :].reshape((-1, model.total_parameter_count))
        except TypeError:
            sample = model.sampler.chain[:, burn:, :].reshape((-1, 10))
            #samples = model.sampler.chain[:, burn:, :].reshape((-1, 16))
    """ 

    if isinstance(pname, str):
        model_name = pname.split(".p")[0]
        model, params = read_pickle(pname)
        samples = model.sampler.chain[:, burn:, :].reshape((-1, model.total_parameter_count))
    else:
        allsamples = []
        for pfile in pname:
            model, params = read_pickle(pfile)
            sample = model.sampler.chain[:, burn:, :].reshape((-1, model.total_parameter_count))
            allsamples.append(sample)
        samples = np.concatenate(tuple(allsamples))
        model_name = "concat_{}".format(len(samples))
    
    return model, samples, params, model_name

#-----------------------------------------------------------------------------#

def make_plots(pname, gif=False, last=False, step=100, burn=50):
    model, samples, params, model_name = get_samples(pname)

    # Create the triangle plot.
    fig = triangle.corner(samples, labels=model.model_parameter_names())
    figname = "plots/{0}_triangle.png".format(model_name)
    fig.savefig(figname)
    print("\tSaved {0}".format(figname))

    # Calculate the median and confidence intervals where frac is the 
    # fraction of samples within the quoted uncertainties.  frac = 0.68 is 
    # the default. Columns are median, -error1, +error2.
    print(median_values(samples, frac=0.68))
    
    # Calculate the mean and standard deviation. Columns are mean, 
    # standard deviation.
    print(mean_values(samples))

    # Plot the MCMC chains as a function of iteration. You can easily tell if 
    # the chains are converged because you can no longer tell where the individual 
    # particle chains are sliced together.
    fig = plot_chains(samples, labels=model.model_parameter_names())
    figname = "plots/{0}_chain.png".format(model_name)
    fig.savefig(figname)
    print("\tSaved {0}".format(figname))

    # Plot the posterior PDFs for each parameter. These are histograms of the 
    # MCMC chains. Boxes = 20 is the default.
    fig = plot_posteriors(samples, labels=model.model_parameter_names(), boxes=20, params=params)
    figname = "plots/{0}_posterior.png".format(model_name)
    fig.savefig(figname)
    print("\tSaved {0}".format(figname))

    # Make a PDF (the plot kind!) with the histogram for each parameter a
    # different page in the PDF for detailed analysis.
    pdfname = "{}_posterior.pdf".format(model_name)
    plot_posteriors_pdf(pdfname, samples, model.model_parameter_names(), params)

    # Make a gif of the model spectrum as a function of iteration,
    # or if last is True, only save the last model spectrum.
    if gif is True:
        plot_models(model, samples, model_name, params, step=step, only_last=last)


#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pname", help="SPAMM model pickle file", type=str)
    parser.add_argument("--gif", dest="gif", action="store_true", default=False,
                        help="Switch to make plots to create gif")
    parser.add_argument("--last", dest="last", action="store_true", default=False,
                        help="Switch to only plot last model iteration")
    parser.add_argument("--step", dest="step", default=100,
                        help="Step size for plotting chain iterations")
    args = parser.parse_args()

    make_plots(args.pname, args.gif, args.last, int(args.step))
