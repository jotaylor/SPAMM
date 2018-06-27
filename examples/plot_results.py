#! /usr/bin/env python

import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import gzip
import numpy as np

def read_pickle(pname):
    try:
        p_data = pickle.loads(gzip.open(pname).read())
    except UnicodeDecodeError:
        p_data = pickle.loads(gzip.open(pname).read(), encoding="latin1")

    model = p_data["model"]
    params = p_data["comp_params"]
    return model, params

def plot_posteriors(pdfname, samples, labels, params=None):
    num_params = np.size(samples[0,:])                                   
    pdf_pages = PdfPages(pdfname)    

    for i in range(num_params):
        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(111)
        
        chain = samples[:,i]
        hist,bins = np.histogram(chain, bins=100)
        ind = np.argmax(hist)
        center_bin = bins[ind] 
        binsize = bins[1]-bins[0]
        center = center_bin + binsize/2.
        med = np.median(chain)

        std = np.std(chain)
        print(std)
        ax.axvspan(center-std, center+std, facecolor="grey", alpha=0.25, label="1sigma")
        ax.hist(chain, bins, color="skyblue")
        
        if params is not None:
            try:
                ax.axvline(params[labels[i]], color="r", linestyle="dashed", linewidth=2)
                ax.set_title("Actual {0}={1}".format(labels[i], params[labels[i]]))
            except KeyError:
                pass
        
        ax.set_xlim(center_bin-binsize*15, center_bin+binsize*15)
        
        ax.axvline(center, color="red", linestyle="dotted", linewidth=2, label="Max")
        ax.axvline(med, color="darkviolet", linestyle="dashed", linewidth=2, label="Median")
        ax.legend(loc="best")

        ax.set_xlabel(labels[i])
        ax.set_ylabel("Posterior PDF")
        ax.set_title(labels[i])

        pdf_pages.savefig(fig)

    pdf_pages.close()
    print("Saved {}".format(pdfname))

def make_plots(pname, burn=50):
    model, params = read_pickle(pname)
    samples = model.sampler.chain[:, burn:, :].reshape((-1, model.total_parameter_count))
    pdfname = "{}_posterior.pdf".format(pname)
    plot_posteriors(pdfname, samples, model.model_parameter_names(), params)
#    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pname", help="SPAMM model pickle file", type=str)
    args = parser.parse_args()

    make_plots(args.pname)
