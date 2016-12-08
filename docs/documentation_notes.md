Documenation Notes
------------------

#### Building the Documentation

The documentation is built from comments embedded within the source code. To build the documentation, you run a script wuich creates a folder called `_build`. To view the documentation, open the file `_build/html/index.html` in your browser. Since the documentation can be easily built and will change frequently, do not check in the `_build` folder into the git repository, but rather build it yourself when you want to view/update it.

**Prerequsite:** The documenation uses the [Sphinx](http://sphinx-doc.org) Python package which you must have installed to build the pages. To check to see if you have it installed, start Python and try to import the package:

    % python 
    Python 2.7.6 |Anaconda 1.8.0 (x86_64)| (default, Jan 10 2014, 11:23:15) 
    [GCC 4.0.1 (Apple Inc. build 5493)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import sphinx
    >>> 

If you get an `ImportError`, install it using:

    % pip install sphinx
    
or

    % easy_install sphinx
    


Command to build the documentation, enter the `documentation` directory and run:

    % make html


##### LaTeX inside documentation

Plain LaTeX equations may be placed inside te documentation. Since LaTeX uses backslash characters liberally which Python interprets as an escape character, you can either double the backslashes (i.e. `\\alpha`), or preferably mark the string as "raw":

    def myFunction(x):
        r''' This is the doc string where you can add LaTeX. Note the leading "r". '''

LaTeX can be added in one of several ways.

* Inline, preceded by `:math:`

        r''' This is an equation: :math: E = mc^{2} '''
    
* An equation on its own line:

        r''' This equation is
        .. math:: E = mc^{2}
        '''

See more examples here: <http://sphinx-doc.org/ext/math.html>

##### Sphinx Themes

More themes here:
<http://www.reddit.com/r/Python/comments/18b6v0/any_freeopensource_modern_sphinx_theme_out_there/>
<http://stackoverflow.com/questions/2075691/sphinx-some-good-customization-examples>
<http://pythonhosted.org/cloud_sptheme/install.html>

Bootstrap: <http://loose-bits.com/2011/12/09/sphinx-twitter-bootstrap-theme.html>
