Documenation Notes
------------------

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
