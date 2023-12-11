#!/usr/bin/python

import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import numpy as np
import subprocess

from spamm.Samples import Samples
from spamm.Model import model_flux
from utils.mask_utils import inverse_bool_mask

###############################################################################
# This code is for analyzing the final posterior samples.

def plot_posteriors_pdf(S, interactive=False):
    # if interactive is False:
    #     matplotlib.use('agg')
    
    pdfname = os.path.join(S.outdir, f"{S.model_name}_posterior.pdf")
    pdf_pages = PdfPages(pdfname)    
    figs = []

    for i in range(S.total_parameter_count):
        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(111)
        
        chain = S.samples[:,i]
        hist, bins = np.histogram(chain, S.histbins)
        binsize = bins[1]-bins[0]

        maxm = S.maxs[i]
        med = S.medians[i] 
        avg = S.means[i]
        mode = S.modes[i]
        
        std = np.std(chain)
        
        if S.params is not None:
            try:
                actual = S.params[S.model_parameter_names[i]]
#                print(f"New limits for {S.model_parameter_names[i]}:\n{maxm-(1.5*std):0.20f}\n{maxm+(1.5*std):0.20f}\n")
                ax.axvspan(actual-std, actual+std, facecolor="grey", 
                           alpha=0.25, label=rf"1$\sigma$={std:1.3e}")
                ax.axvline(S.params[S.model_parameter_names[i]], 
                           color="red", linestyle="solid", linewidth=1.5, 
                           label=f"Actual value={actual:1.3e}")
            except KeyError:
                actual = maxm
                ax.axvspan(actual-std, actual+std, facecolor="grey", 
                           alpha=0.25, label=rf"1$\sigma$={std:1.3e}")
                 
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
                   label=f"Mean={avg:1.3e}")
        ax.axvline(med, color="darkviolet", linestyle="--", linewidth=1.5, 
                   label=f"Median={med:1.3e}")
        ax.axvline(mode, color="blue", linestyle="--", linewidth=1.5, 
                   label=f"Mode={mode:1.3e}")
        ax.axvline(maxm, color="fuchsia", linestyle="--", linewidth=1.5, 
                   label=f"Maximum={maxm:1.3e}")
        ax.legend(loc="best")

        ax.set_xlabel(S.model_parameter_names[i])
        ax.set_ylabel("Posterior PDF")
        ax.set_title(S.model_parameter_names[i])

        figs.append(fig)
        pdf_pages.savefig(fig)
        plt.close(fig)

    pdf_pages.close()
    print(f"Saved {pdfname}")

    return figs

###############################################################################

def plot_best_models(S):
    data_spectrum = S.model.data_spectrum

    actualcolor = "deepskyblue"
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(111)

    # Shade masked regions
    boolmask = S.model.mask
    if boolmask is not None:
        masks = inverse_bool_mask(data_spectrum.spectral_axis, boolmask)
        for mask in masks:
            ax.axvspan(mask[0], mask[1], color='red', alpha=0.1)

    ax.errorbar(data_spectrum.spectral_axis, data_spectrum.flux,
                    data_spectrum.flux_error, mfc=actualcolor, mec=actualcolor,
                    ecolor=actualcolor, fmt=".", zorder=-100, label="Actual Flux") 
    ax.plot(data_spectrum.spectral_axis,
            model_flux(params=S.means, data_spectrum=data_spectrum, components=S.model.components),
            color="darkblue", label="Mean")
    ax.plot(data_spectrum.spectral_axis,
            model_flux(params=S.medians, data_spectrum=data_spectrum, components=S.model.components),
            color="darkviolet", label="Median")
    ax.plot(data_spectrum.spectral_axis,
            model_flux(params=S.modes, data_spectrum=data_spectrum, components=S.model.components),
            color="blue", label="Mode")
    ax.plot(data_spectrum.spectral_axis,
            model_flux(params=S.maxs, data_spectrum=data_spectrum, components=S.model.components),
            color="fuchsia", label="Max")

    ax.set_title("Best Fits")
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel(r"ergs/s/cm$^2$")
    ax.legend(loc="upper left", framealpha=0.25)
    figname = "{}_best.png"
    fig.savefig(os.path.join(S.outdir, figname))
    plt.close(fig)
    print(f"\tSaved {figname}")

###############################################################################

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
        print(f"Iteration {i}")
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
            ax.set_title(f"{component.name}, Iteration {i}")
            ax.set_xlabel(r"Wavelength [$\AA$]")
            ax.set_ylabel(r"ergs/s/cm$^2$")
            ax.legend(loc="upper left", framealpha=0.25)
            figname = os.path.join(gifdir, f"{component.name}_iter{i:06d}.png")
            fig.savefig(os.path.join(S.outdir, figname))
            if S.last is True:
                print(f"\tSaved {figname}")
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
        ax.set_title(f"Sum Of Model Components, Iteration {i}")
        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel(r"ergs/s/cm$^2$")
        ax.legend(loc="upper left", framealpha=0.25)
        figname = os.path.join(outdir, f"model_iter{i:06d}.png")
        fig.savefig(os.path.join(S.outdir, figname))
        if S.last is True:
            print(f"\tSaved {figname}")
        plt.close(fig)

    if S.gif is True:
        for component in S.model.components:
            cname = component.name
            gifname = os.path.join(outdir, f"{cname}.gif")
            subprocess.check_call(["convert", "-delay", "15", "-loop", "1", 
                                   os.path.join(outdir, f"{cname}*png"), 
                                   gifname])
            print(f"\tSaved {gifname}")
        gifname = os.path.join(outdir, f"{S.model_name}.gif")
        subprocess.check_call(["convert", "-delay", "15", "-loop", "1", 
                               os.path.join(outdir, f"{S.model_name}*png"), 
                               gifname])
        print(f"\tSaved {gifname}")
    
###############################################################################

def make_all_plots(S):

    # Create the triangle plot.
    fig = corner(S.samples, labels=S.model_parameter_names)
    figname = f"{S.model_name}_triangle.png"
    fig.savefig(os.path.join(S.outdir, figname))
    plt.close(fig)
    print(f"\tSaved {figname}")

    # Plot the MCMC chains as a function of iteration. You can easily tell if 
    # the chains are converged because you can no longer tell where the individual 
    # particle chains are sliced together.
    fig = plot_chains(S.samples, labels=S.model_parameter_names)
    figname = f"{S.model_name}_chain.png"
    fig.savefig(os.path.join(S.outdir, figname))
    plt.close(fig)
    print(f"\tSaved {figname}")

    # Plot the posterior PDFs for each parameter. These are histograms of the 
    # MCMC chains. Boxes = 20 is the default.
    fig = plot_posteriors(S.samples, labels=S.model_parameter_names, 
                          boxes=20, params=S.params)
    figname = f"{S.model_name}_posterior.png"
    fig.savefig(os.path.join(S.outdir, figname))
    plt.close(fig)
    print(f"\tSaved {figname}")

    # Make a PDF (the plot kind!) with the histogram for each parameter a
    # different page in the PDF for detailed analysis.
    figs = plot_posteriors_pdf(S)

    # Make a gif of the model spectrum as a function of iteration,
    # or if last is True, only save the last model spectrum.
    if S.gif is True:
        plot_models(S)

###############################################################################

def make_plots_from_pickle(pname, outdir, gif=False, last=False, step=100):
    S = Samples(pname, outdir=outdir, gif=gif, last=last, step=step)
    make_all_plots(S)

###############################################################################

def first_concat():
    models = ["model_20180807_0442.pickle.gz", "model_20180807_2917.pickle.gz", "model_20180807_3906.pickle.gz", "model_20180808_0244.pickle.gz", "model_20180810_0857.pickle.gz", "model_20180811_0849.pickle.gz", "model_20180812_2600.pickle.gz", "model_20180814_2104.pickle.gz"]
    S = Samples(models)
    make_all_plots(S)

###############################################################################

def corner(xs, interactive=False, labels=None, extents=None, truths=None, truth_color="#4682b4",
           scale_hist=False, quantiles=[], **kwargs):
    """
    Make a *sick* corner plot showing the projections of a set of samples
    drawn in a multi-dimensional space.

    __version__ = "0.0.5"
    __author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
    __copyright__ = "Copyright 2013 Daniel Foreman-Mackey"
    __contributors__ = [    # Alphabetical by first name.
                            "Ekta Patel @ekta1224",
                            "Geoff Ryan @geoffryan",
                            "Phil Marshall @drphilmarshall",
                            "Pierre Gratier @pirg"]

    :param xs: ``(nsamples, ndim)``
        The samples. This should be a 1- or 2-dimensional array. For a 1-D
        array this results in a simple histogram. For a 2-D array, the zeroth
        axis is the list of samples and the next axis are the dimensions of
        the space.

    :param labels: ``ndim`` (optional)
        A list of names for the dimensions.

    :param truths: ``ndim`` (optional)
        A list of reference values to indicate on the plots.

    :param truth_color: (optional)
        A ``matplotlib`` style color for the ``truths`` makers.

    :param quantiles: (optional)
        A list of fractional quantiles to show on the 1-D histograms as
        vertical dashed lines.

    :param scale_hist: (optional)
        Should the 1-D histograms be scaled in such a way that the zero line
        is visible?

    """
    
    # if interactive is False:
    #     matplotlib.use('agg')
    
    print("Plotting the sample projections.")

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"

    K = len(xs)
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.05 * factor  # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim
    fig = plt.figure(figsize=(dim, dim))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    if extents is None:
        extents = [[x.min(), x.max()] for x in xs]

    for i, x in enumerate(xs):
        # Plot the histograms.
        ax = fig.add_subplot(K, K, i * (K + 1) + 1)
        n, b, p = ax.hist(x, bins=kwargs.get("bins", 50), range=extents[i],
                histtype="step", color=kwargs.get("color", "k"))
        if truths is not None:
            ax.axvline(truths[i], color=truth_color)

        # Plot quantiles if wanted.
        if len(quantiles) > 0:
            xsorted = sorted(x)
            for q in quantiles:
                ax.axvline(xsorted[int(q * len(xsorted))], ls="dashed",
                           color=kwargs.get("color", "k"))

        # Set up the axes.
        ax.set_xlim(extents[i])
        if scale_hist:
            maxn = np.max(n)
            ax.set_ylim(-0.1 * maxn, 1.1 * maxn)
        else:
            ax.set_ylim(0, 1.1 * np.max(n))
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(5))

        # Not so DRY.
        if i < K - 1:
            ax.set_xticklabels([])
        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                ax.set_xlabel(labels[i])
                ax.xaxis.set_label_coords(0.5, -0.3)

        for j, y in enumerate(xs[:i]):
            ax = fig.add_subplot(K, K, (i * K + j) + 1)
            hist2d(y, x, ax=ax, extent=[extents[j], extents[i]], **kwargs)

            if truths is not None:
                ax.plot(truths[j], truths[i], "s", color=truth_color)
                ax.axvline(truths[j], color=truth_color)
                ax.axhline(truths[i], color=truth_color)

            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j])
                    ax.xaxis.set_label_coords(0.5, -0.3)

            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.3, 0.5)

    return fig

###############################################################################

def error_ellipse(mu, cov, ax=None, factor=1.0, **kwargs):
    """
    Plot the error ellipse at a point given it's covariance matrix.
    
    __version__ = "0.0.5"
    __author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
    __copyright__ = "Copyright 2013 Daniel Foreman-Mackey"
    __contributors__ = [    # Alphabetical by first name.
                            "Ekta Patel @ekta1224",
                            "Geoff Ryan @geoffryan",
                            "Phil Marshall @drphilmarshall",
                            "Pierre Gratier @pirg"]

    """
    # some sane defaults
    facecolor = kwargs.pop('facecolor', 'none')
    edgecolor = kwargs.pop('edgecolor', 'k')

    x, y = mu
    U, S, V = np.linalg.svd(cov)
    theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ellipsePlot = Ellipse(xy=[x, y],
            width=2 * np.sqrt(S[0]) * factor,
            height=2 * np.sqrt(S[1]) * factor,
            angle=theta,
            facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    if ax is None:
        ax = plt.gca()
    ax.add_patch(ellipsePlot)

###############################################################################

def hist2d(x, y, *args, **kwargs):
    """
    Plot a 2-D histogram of samples.

    __version__ = "0.0.5"
    __author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
    __copyright__ = "Copyright 2013 Daniel Foreman-Mackey"
    __contributors__ = [    # Alphabetical by first name.
                            "Ekta Patel @ekta1224",
                            "Geoff Ryan @geoffryan",
                            "Phil Marshall @drphilmarshall",
                            "Pierre Gratier @pirg"]
    
    """
   
    ax = kwargs.pop("ax", plt.gca())

    extent = kwargs.pop("extent", [[x.min(), x.max()], [y.min(), y.max()]])
    bins = kwargs.pop("bins", 50)
    color = kwargs.pop("color", "k")
    plot_datapoints = kwargs.get("plot_datapoints", True)

    cmap = cm.get_cmap("gray")
    cmap._init()
    cmap._lut[:-3, :-1] = 0.
    cmap._lut[:-3, -1] = np.linspace(1, 0, cmap.N)

    X = np.linspace(extent[0][0], extent[0][1], bins + 1)
    Y = np.linspace(extent[1][0], extent[1][1], bins + 1)
    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=(X, Y))

    V = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]

    for i, v0 in enumerate(V):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]

    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    X, Y = X[:-1], Y[:-1]

    if plot_datapoints:
        ax.plot(x, y, "o", color=color, ms=1.5, zorder=-1, alpha=0.1,
                rasterized=True)
        V2 = np.array(list( set([V[-1], H.max()])) )
        V2.sort()
        if len(V2) == 1:
            V2 = np.array([0, V2[0]])
        ax.contourf(X1, Y1, H.T, V2,
                    cmap=LinearSegmentedColormap.from_list("cmap", [[1] * 3, [1] * 3],
                    N=2), antialiased=False)

    ax.pcolor(X, Y, H.max() - H.T, cmap=cmap)
    # Contour levels must be sorted for countour to work.
    # There cannot be repeats in contour levels.
    V = np.array(list(set(V)))
    V.sort()
    ax.contour(X1, Y1, H.T, V, colors=color)

    data = np.vstack([x, y])
    mu = np.mean(data, axis=1)
    cov = np.cov(data)
    if kwargs.pop("plot_ellipse", False):
        error_ellipse(mu, cov, ax=ax, edgecolor="r", ls="dashed")

    ax.set_xlim(extent[0])
    ax.set_ylim(extent[1])

###############################################################################

def median_values(samples, frac=0.68):
    num_params = np.size(samples[0,:])
    result = np.zeros((num_params, 3))
    for i in range(0, num_params):
        chain = np.sort(samples[:,i])
        # calculate the median values:
        if np.size(chain)%2 == 1:   # for odd sizes of chain
            index = int(np.size(chain)/2)
            median = chain[index]
        if np.size(chain)%2 == 0:   # for even sizes of chain
            lower = int(np.size(chain)/2)-1
            higher = int(np.size(chain)/2)
            median = (chain[lower] + chain[higher])/2.
        result[i,0] = median
        # calculate the confidence intervals
        num_spread = np.size(chain)/2.0*frac
        max_index = int(np.size(chain)/2.0 + num_spread)
        min_index = int(np.size(chain)/2.0 - num_spread)
        max_value = chain[max_index]
        min_value = chain[min_index]
        max_error = abs(median - max_value)
        min_error = abs(median - min_value)
        result[i,1] = min_error
        result[i,2] = max_error

    print("Calculating median and "+str(frac*100)+"% confidence intervals (min, max).")
    return result

###############################################################################

def mean_values(samples):
    num_params = np.size(samples[0,:])
    result = np.zeros((num_params, 2))
    for i in range(0, num_params):
        chain = samples[:,i]
        result[i,0] = np.mean(chain)
        result[i,1] = np.std(chain)

    print("Calculating mean and standard deviation.")
    return result

###############################################################################

def plot_chains(samples, labels, interactive=False):
    # if interactive is False:
    #     matplotlib.use('agg')
    
    num_params = np.size(samples[0,:])
    fig = plt.figure()
    #####
    # The following is from corner:
    K = num_params
    factor = 2.0           # size of one side of one panel
    lbdim = 0.4 * factor   # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.15         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim
    fig = plt.figure(figsize=(dim, dim))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                    wspace=whspace, hspace=whspace)
    #####
    for i in range(0, num_params):
        chain = samples[:,i]
        ax = fig.add_subplot(num_params, 1, i+1)
        ax.plot(chain, '-b')
        ax.set_ylabel(labels[i])
    ax.set_xlabel("MCMC Chain Iteration")
    print("Plotting the MCMC chains.")
    return fig

###############################################################################

def plot_posteriors(samples, labels, boxes=20, params=None):
    num_params = np.size(samples[0,:])
    fig = plt.figure()
    #####
    # The following is from corner:
    K = num_params
    factor = 2.0           # size of one side of one panel
    lbdim = 0.4 * factor   # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.3         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim
    fig = plt.figure(figsize=(dim, dim))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                    wspace=whspace, hspace=whspace)
    #####
    for i in range(0, num_params):
        chain = samples[:,i]
        ax = fig.add_subplot(num_params, 1, i+1)
        ax.hist(chain, boxes)
        if params is not None:
            try:
                ax.axvline(params[labels[i]], color="r", linestyle="dashed", linewidth=2)
                ax.set_title(f"Actual {labels[i]}={params[labels[i]]}")
            except KeyError:
                pass
        ax.set_xlabel(labels[i])
        ax.set_ylabel("Posterior PDF")
    print("Plotting the model posterior PDFs.")
    return fig

###############################################################################

def plot_spectra(model, samples):
    data_spectrum = model.data_spectrum   # lambda, flux, flux_error
    num_samples = 110  #np.size(samples[:,0])

    plt.clf()
    plt.ion()
    for i in range(0, num_samples):
        # plot the data
        plt.errorbar(data_spectrum.spectral_axis, data_spectrum.flux, 
                yerr=data_spectrum.flux_error, mfc='blue', mec='blue', ecolor='b',
                fmt='.')

        # plot each of the model components individually
        k = 0
    for component in model.components:
        component_flux = component.flux(spectrum=model.data_spectrum,
                parameters=samples[i,k:len(component.model_parameter_names)])  # flux
        k = len(component.model_parameter_names)
        plt.plot(data_spectrum.spectral_axis, component_flux, '-r')

    # plot the sum of the model components
        model_spectrum = model.model_flux(params=samples[i,:])  # flux
        plt.plot(data_spectrum.spectral_axis, model_spectrum)

###############################################################################

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

