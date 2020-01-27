# tidyPlate

 Utility for cleaning, organizing, and analyzing bioassay plate results in Excel/xlsx spreadsheet format.

## tidyPlate Goals

* Take xlsx (eventually csv as well) containing 96-well plate bioassay (i.e. ELISA) data and layout as input in an 8Rx12C layout
* Process the data to create a table (pandas dataframe in program) containing a single row for each sample/standard
* Process data to subtract any included blanks as necessary
* Use regression to generate a standard curve
  * 5PL is ultimate goal
  * Examples:
    * [MyAssays](https://www.myassays.com/five-parameter-fit.assay)
    * [MATLAB MathWorks from Giuseppe Cardillo](https://www.mathworks.com/matlabcentral/fileexchange/38043-five-parameters-logistic-regression-there-and-back-again)
    * [4PL Regression in Python from Duke](https://people.duke.edu/~ccc14/pcfb/analysis.html)
* Use generated standard curve to calculate unknown values
* Output final table and regression results (as plot image) into original xlsx file
* After all else is functional, may implement GUI for ease-of-access
