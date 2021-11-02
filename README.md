# IO-PS
Public repository of developmental Python code related to research on the input-output product space (IO-PS) 
[Described in Bam, W., &amp; De Bruyne, K. (2019). Improving Industrial Policy Intervention: The Case of Steel in South Africa. The Journal of Development Studies, 55(11), 2460–2475. https://doi.org/10.1080/00220388.2018.1528354]

## Package

### Installation
The package is available from the Python Package Index: https://pypi.org/project/iops/

```text
pip install iops
pip install ecomplexity
```

### Usage
CEPII-BACI trade data is a required input (.csv). The BACI data is available at: http://www.cepii.fr/CEPII/fr/bdd_modele/presentation.asp?id=37

Full IO-PS analysis requires a value chain input (.csv). Three columns are required: 'Tier', 'Category' and 'HS Trade Code'.

```python
import pandas as pd
from iops import main

tradedata_df = pd.read_csv('BACI_HSXX_YXXXX_V202001.csv')
valuechain_df = pd.read_csv('X_Value_Chain.csv')

main.iops(tradedata_df,valuechain_df)
```

### Value Chain Output
Results are generated at tier, category and product level. Results are written to an Excel spreadsheet and headless CSV for each.
```text
Tier_Results.csv
Tier_Results.xlsx
Product_Category_Results.csv
Product_Category_Results.xlsx
Product_Results.csv
Product_Results.xlsx
```

### Function
```Python
def iops(tradedata, valuechain=None, countrycode=710, tradedigit=6, statanorm=False):
    """ IO-PS calculation function that writes the results to .xls and .csv
        Arguments:
            tradedata: pandas dataframe containing raw CEPII-BACI trade data.
            valuechain: .csv of the value chain the IO-PS will map.
                columns - 'Tier', 'Category', 'HS Trade Code'
                default - None
            countrycode: integer indicating which country the IO-PS will map.
                default - 710 
            tradedigit: Integer of 6 or 4 to indicate the raw trade digit summation level.
                default - 6 
            statanorm: Boolean indicator of literature based or CID-Harvard STATA normalization.
                default - False
    """
```
## Future Considerations
* User error warnings
* Investigate use of ecomplexity package fork
* Additional IO-PS metrics
* ECI and distance alignment
## References
### IO-PS

* Bam, W., & De Bruyne, K. (2017). Location policy and downstream mineral processing: A research agenda. Extractive Industries and Society, 4(3), 443–447. https://doi.org/10.1016/j.exis.2017.06.009
* Marais, M., & Bam, W. (2019). Developmental potential of the aerospace industry: the case of South Africa. In 2019 IEEE International Conference on Engineering, Technology and Innovation (ICE/ITMC) (pp. 1–9). IEEE. https://doi.org/10.1109/ICE.2019.8792812

### Economic Complexity and Product Complexity
This packages uses a modified copy of the Growth Lab at Harvard's Center for International Development py-ecomplexity package. The ecomplexity package is used to calculate economic complexity indices: https://github.com/cid-harvard/py-ecomplexity

