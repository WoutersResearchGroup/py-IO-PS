import pandas as pd
import numpy as np
# CID-Harvard ecomplexity package imports
import warnings
from ecomplexity.calc_proximity import calc_discrete_proximity
from ecomplexity.calc_proximity import calc_continuous_proximity
from ecomplexity.ComplexityData import ComplexityData
from ecomplexity.density import calc_density
from ecomplexity.coicog import calc_coi_cog


# IO-PS framework value chain mapping
def iops(tradedata, valuechain=None, countrycode=710, tradedigit=6, statanorm=False):
    """ IO-PS calculation function that writes the results to .xls and .csv
        Arguments:
            tradedata: pandas dataframe containing raw CEPII-BACI trade data.
            valuechain: .csv of the value chain the IO-PS will map.
                columns - 'tier', 'category', ' HS trade code'
                default - None
            countrycode: integer indicating which country the IO-PS will map.
                default - 710
            tradedigit: Integer of 6 or 4 to indicate the raw trade digit summation level.
                default - 6
            statanorm: Boolean indicator of literature based or CID-Harvard STATA normalization.
                default - False
        """

    # Trade data formatting using digitsumm function
    edata = digitsumm(tradedata, tradedigit)
    # Computation of product space metrics using psmetrics function
    cdata = psmetrics(edata, statanorm)

    # Default the IO-PS mapping to all products, at trade digit levels, if no value chain is input.
    if valuechain is None:
        # Store the subset of data associated with the input country code.
        country_df = cdata[cdata['i'] == countrycode].copy(deep=True)
        print("Pre-mappings complete")

        # Binary RCA assignment for IO-PS calculations.
        # Country dataframe Mcp column.
        country_df['Mcp'] = 0
        # Set Mcp column to a value of 1 for all products in the value chain for which the country has an RCA > 1.
        country_df.loc[country_df['rca'] >= 1, 'Mcp'] = 1
        # All data dataframe Mcp column.
        cdata['Mcp'] = 0
        cdata.loc[cdata['rca'] >= 1, 'Mcp'] = 1
        print("Mcp assignment complete")

        print('No value chain input, calculating default at trade digit levels')
        # Aggregate raw trade data to the user specified trade digit level.
        if tradedigit == 4:
            print('No value chain input, calculating default for 4-digit input')
            # Write 4-digit aggregation
            country_df = cdata.loc[cdata['i'] == countrycode].copy(deep=True)
            fourcountry_df = country_df[['t', 'k', 'Mcp', 'rca', 'distance', 'pci']]
            # Store column headings
            fourcountry_df.columns = ['year', 'code', 'Mcp', 'Average RCA', 'Average distance',
                                      'Average product complexity']
            # Write the results to .xls and .csv
            fourcountry_df.to_csv('noGVC4Digit_Results.csv', header=None, index=False)
            fourcountry_df.to_excel('noGVC4Digit_Results.xlsx', index=False)
            print('No value chain 4-digit results calculated')

            print('No value chain input, calculating default for 2-digit input')
            # Create 2-digit level codes
            digit_estimate = country_df.k / 100
            # Store codes as nearest lower integer
            digit_new = np.floor(digit_estimate).astype(int)
            # Add new column at 2-digit level
            country_df['2k'] = digit_new
            # Store new digits and key metrics
            twocountry_df = country_df[['t', '2k', 'rca', 'distance', 'pci']]
            # Sum export values to new 2-digit level and reset index
            twocountry_df = twocountry_df.groupby(['t', '2k']).mean().reset_index()
            # Store column headings
            twocountry_df.columns = ['year', 'code', 'Average RCA', 'Average distance', 'Average product complexity']
            # Write the results to .xls and .csv
            twocountry_df.to_csv('noGVC2Digit_Results.csv', header=None, index=False)
            twocountry_df.to_excel('noGVC2Digit_Results.xlsx', index=False)
            print('No value chain 2-digit results calculated')
        else:
            print('No value chain input, calculating default for 6-digit input')
            # Store 6-digit aggregation data
            country_df = cdata.loc[cdata['i'] == countrycode].copy(deep=True)
            sixcountry_df = country_df[['t', 'k', 'Mcp', 'rca', 'distance', 'pci']]
            # Store column headings
            sixcountry_df.columns = ['year', 'code', 'Mcp', 'Average RCA', 'Average distance',
                                     'Average product complexity']
            # Write the results to .xls and .csv
            sixcountry_df.to_csv('noGVC6Digit_Results.csv', header=None, index=False)
            sixcountry_df.to_excel('noGVC6Digit_Results.xlsx', index=False)
            print('No value chain 6-digit results calculated')

            print('No value chain input, calculating default for 4-digit input')
            # Create 4-digit codes
            digit_estimate = country_df.k / 100
            # Store 4-digit estimates as nearest lower integer
            digit_new = np.floor(digit_estimate).astype(int)
            # Add a new column containing the 4-digit level codes
            country_df['4k'] = digit_new
            # Store new digits and key metrics
            fourcountry_df = country_df[['t', '4k', 'rca', 'distance', 'pci']]
            # Sum export values to new 4-digit level and reset index
            fourcountry_df = fourcountry_df.groupby(['t', '4k']).mean().reset_index()
            # Store column headings
            fourcountry_df.columns = ['year', 'code', 'Average RCA', 'Average distance', 'Average product complexity']
            # Write the results to .xls and .csv
            fourcountry_df.to_csv('noGVC4Digit_Results.csv', header=None, index=False)
            fourcountry_df.to_excel('noGVC4Digit_Results.xlsx', index=False)
            print('No value chain 4-digit results calculated')

            print('No value chain input, calculating default for 2-digit input')
            # Create 2-digit level codes
            digit_estimate = country_df.k / 10000
            # Store codes as nearest lower integer
            digit_new = np.floor(digit_estimate).astype(int)
            # Add new column at 2-digit level
            country_df['2k'] = digit_new
            # Store new digits and key metrics
            twocountry_df = country_df[['t', '2k', 'rca', 'distance', 'pci']]
            # Sum export values to new 2-digit level and reset index
            twocountry_df = twocountry_df.groupby(['t', '2k']).mean().reset_index()
            # Store column headings
            twocountry_df.columns = ['year', 'code', 'Average RCA', 'Average distance', 'Average product complexity']
            # Write the results to .xls and .csv
            twocountry_df.to_csv('noGVC2Digit_Results.csv', header=None, index=False)
            twocountry_df.to_excel('noGVC2Digit_Results.xlsx', index=False)
            print('No value chain 2-digit results calculated')
    else:
        # Create column headings for the value chain dataframe to enable merging on column names.
        valuechain.columns = ['Tier', 'Category', 'k']
        # Store the subset of data associated with the input country
        country_df = cdata[cdata['i'] == countrycode].copy(deep=True)
        # Map the value chain to the country
        valuechain_df = pd.merge(valuechain, country_df, on='k', how='left')
        print("Pre-mappings complete")

        # Binary RCA assignment for value chain calculations.
        # Value Chain Mcp column created and initialized to 0.
        valuechain_df['Mcp'] = 0
        # Set Mcp column to a value of 1 for all products in the value chain for which the country has an RCA > 1.
        valuechain_df.loc[valuechain_df['rca'] >= 1, 'Mcp'] = 1
        # Country Mcp column
        country_df['Mcp'] = 0
        country_df.loc[country_df['rca'] >= 1, 'Mcp'] = 1
        # All data Mcp
        cdata['Mcp'] = 0
        cdata.loc[cdata['rca'] >= 1, 'Mcp'] = 1
        print("Mcp assignment complete")

        # Value chain tier results for value chain input
        print('Starting GVC tier result calculation')
        vc_tier_results = []
        # Loop through each tier of the value chain and calculate the IO-PS metrics.
        for tierCount in range(len(valuechain.Tier.unique())):
            vc_tier_results.append([])
            # Store the subset of the value chain dataframe that corresponds to the current tier counter.
            tiertemp = valuechain_df[valuechain_df['Tier'] == tierCount + 1]
            # Tier number
            vc_tier_results[tierCount].append(tierCount + 1)
            # Average RCA of the tier
            vc_tier_results[tierCount].append(tiertemp['rca'].mean())
            # Average distance of the tier
            vc_tier_results[tierCount].append(tiertemp['distance'].mean())
            # Average complexity of the tier
            vc_tier_results[tierCount].append(tiertemp['pci'].mean())
            # Average RCA of products yet to be obtained
            vc_tier_results[tierCount].append(tiertemp[tiertemp['Mcp'] == 0]['rca'].mean())
            # Average distance of products yet to be obtained
            vc_tier_results[tierCount].append(tiertemp[tiertemp['Mcp'] == 0]['distance'].mean())
            # Average complexity of products yet to be obtained
            vc_tier_results[tierCount].append(tiertemp[tiertemp['Mcp'] == 0]['pci'].mean())

        # Store results as a dataframe
        vc_tier_results = pd.DataFrame(vc_tier_results)
        # Label the column headings for readability
        vc_tier_results.columns = ['Tier', 'Average RCA', 'Average distance', 'Average product complexity',
                                   'Average RCA of products with RCA < 1', 'Average distance of products with RCA < 1',
                                   'Average complexity of products with RCA < 1']
        # Write the results to .xls and .csv
        vc_tier_results.to_csv('Tier_Results.csv', header=None, index=False)
        vc_tier_results.to_excel('Tier_Results.xlsx', index=False)
        print('GVC tier results calculated')

        # Value chain category results
        print('Starting GVC category result calculation')
        vc_category_results = []
        # Loop through each category of the value chain and calculate the IO-PS metrics.
        for catCount in range(len(valuechain.Category.unique())):
            vc_category_results.append([])
            # Store the subset of the value chain dataframe that corresponds to the current category counter.
            cat_temp = valuechain_df[valuechain_df['Category'] == catCount + 1]
            # Append the tier of current category
            vc_category_results[catCount].append(cat_temp['Tier'].iloc[0])
            # Append the current category
            vc_category_results[catCount].append(catCount + 1)
            # Append the aggregate RCA of the current category
            vc_category_results[catCount].append(cat_temp['rca'].mean())
            # Append the aggregate distance of the current category
            vc_category_results[catCount].append(cat_temp['distance'].mean())
            # Append the aggregate product complexity of the current category
            vc_category_results[catCount].append(cat_temp['pci'].mean())
            # Append the aggregate RCA of products in this category for which an RCA has not been obtained
            vc_category_results[catCount].append(cat_temp[cat_temp['Mcp'] == 0]['rca'].mean())
            # Append the aggregate distance of products in this category for which an RCA has not been obtained
            vc_category_results[catCount].append(cat_temp[cat_temp['Mcp'] == 0]['distance'].mean())
            # Append the aggregate complexity of products in this category for which an RCA has not been obtained
            vc_category_results[catCount].append(cat_temp[cat_temp['Mcp'] == 0]['pci'].mean())

        # Store the results as a dataframe which places each set in the loop in a new row.
        vc_category_results = pd.DataFrame(vc_category_results)
        # Label the column headings for readability
        vc_category_results.columns = ['Tier', 'Category', 'Average RCA', 'Average distance',
                                       'Average product complexity',
                                       'Average RCA of products with RCA < 1',
                                       'Average distance of products with RCA < 1',
                                       'Average complexity of products with RCA < 1']
        # Write the results to .xls and .csv
        vc_category_results.to_csv('Product_Category_Results.csv', header=None, index=False)
        vc_category_results.to_excel('Product_Category_Results.xlsx', index=False)
        print('GVC category results calculated')

        # Value chain product results
        print('Starting GVC product results calculation')
        # Subset the columns from the value chain for the product level output.
        vc_product_results = valuechain_df[['Tier', 'Category', 'k', 'rca', 'distance', 'pci']]
        # Rename the columns for readability
        vc_product_results.columns = ['Tier', 'Category', 'Code', 'RCA', 'Distance', 'Complexity']
        # vc_product_results = pd.DataFrame(vc_product_results)
        # Write the results to .xls and .csv
        vc_product_results.to_csv('Product_Results.csv', header=None, index=False)
        vc_product_results.to_excel('Product_Results.xlsx', index=False)
        print('GVC product results calculated')
    print("IOPS execution complete")
    return


def digitsumm(tradedata, tradedigit):
    """Trade data pre-processing, returns pandas dataframe of export data at 4-digit or 6-digit level"""

    print("Export data trade code summation started")
    # Trade data transformation to retain exports at the 4-digit level
    if tradedigit == 4:
        # Remove the import value data from the raw trade data
        edata = tradedata.drop(['j', 'q'], axis=1)
        # Aggregate export value at the 6-digit level
        edata = edata.groupby(['t', 'i', 'k']).sum().reset_index()
        # Create the 4-digit codes
        digit_estimate = edata.k / 100
        # Store codes as nearest lower integer
        digit_new = np.floor(digit_estimate).astype(int)
        # Add a new column to the export data with the 4-digit level code
        edata['k'] = digit_new
        # Sum export values to 4-digit level and reset index
        edata = edata.groupby(['t', 'i', 'k']).sum().reset_index()
        print("Export data 4-digit summation complete")
    else:
        # Trade data transformation to retain exports at the 6-digit level
        # Remove import data from the raw trade data
        edata = tradedata.drop(['j', 'q'], axis=1)
        # Sum export value to 6-digit code level and reset the index
        edata = edata.groupby(['t', 'i', 'k']).sum().reset_index()
        print("Export data 6-digit summation complete")
    return edata


def psmetrics(edata, statanorm):
    """Product Space metric calculations on the aggregated export data, returns pandas dataframe"""

    print("Product Space metric function started")
    # CEPII-BACI column values are changed to ecomplexity package equivalents.
    tradecols = {'time': 't', 'loc': 'i', 'prod': 'k', 'val': 'v'}
    # Complexity and other metrics ar calculated using the ecomplexity package.
    cdata = ecomplexity(edata, tradecols, statanorm)
    # The distance metric is added as a new column.
    cdata['distance'] = 1 - cdata['density']
    print("Product Space metric function complete")
    return cdata


# The following functions are from the CID-Harvard University ecomplexity package (2018) used under the MIT license
# ecomplexity function inputs have been altered to facilitate use within this package:
# https://github.com/cid-harvard/py-ecomplexity/blob/fed5d9b377b133a8431a05c347c4408086d789b2/ecomplexity/ecomplexity.py#L113
def reshape_output_to_data(cdata, t):
    """Reshape output ndarrays to df"""
    diversity = (
        cdata.diversity_t[:, np.newaxis].repeat(cdata.mcp_t.shape[1], axis=1).ravel()
    )
    ubiquity = (
        cdata.ubiquity_t[np.newaxis, :].repeat(cdata.mcp_t.shape[0], axis=0).ravel()
    )
    eci = cdata.eci_t[:, np.newaxis].repeat(cdata.mcp_t.shape[1], axis=1).ravel()
    pci = cdata.pci_t[np.newaxis, :].repeat(cdata.mcp_t.shape[0], axis=0).ravel()
    coi = cdata.coi_t[:, np.newaxis].repeat(cdata.mcp_t.shape[1], axis=1).ravel()

    out_dict = {
        "diversity": diversity,
        "ubiquity": ubiquity,
        "mcp": cdata.mcp_t.ravel(),
        "eci": eci,
        "pci": pci,
        "density": cdata.density_t.ravel(),
        "coi": coi,
        "cog": cdata.cog_t.ravel(),
    }

    if hasattr(cdata, "rpop_t"):
        out_dict["rca"] = cdata.rca_t.ravel()
        out_dict["rpop"] = cdata.rpop_t.ravel()

    elif hasattr(cdata, "rca_t"):
        out_dict["rca"] = cdata.rca_t.ravel()

    output = pd.DataFrame.from_dict(out_dict).reset_index(drop=True)

    cdata.data_t["time"] = t
    cdata.output_t = pd.concat([cdata.data_t.reset_index(), output], axis=1)
    cdata.output_list.append(cdata.output_t)
    return cdata


def conform_to_original_data(cdata, data):
    """Reset column names and add dropped columns back"""
    cdata.output = cdata.output.rename(columns=cdata.cols_input)
    cdata.output = cdata.output.merge(
        data, how="outer", on=list(cdata.cols_input.values())
    )
    return cdata


def calc_eci_pci(cdata):
    # Check if diversity or ubiquity is 0 or nan, can cause problems
    if ((cdata.diversity_t == 0).sum() > 0) | ((cdata.ubiquity_t == 0).sum() > 0):
        warnings.warn(
            f"In year {cdata.t}, diversity / ubiquity is 0 for some locs/prods"
        )

    # Extract valid elements only
    cntry_mask = np.argwhere(cdata.diversity_t == 0).squeeze()
    prod_mask = np.argwhere(cdata.ubiquity_t == 0).squeeze()
    diversity_valid = cdata.diversity_t[cdata.diversity_t != 0]
    ubiquity_valid = cdata.ubiquity_t[cdata.ubiquity_t != 0]
    mcp_valid = cdata.mcp_t[cdata.diversity_t != 0, :][:, cdata.ubiquity_t != 0]

    # Calculate ECI and PCI eigenvectors
    mcp1 = mcp_valid / diversity_valid[:, np.newaxis]
    mcp2 = mcp_valid / ubiquity_valid[np.newaxis, :]
    # Make copy of transpose to ensure contiguous array for performance reasons
    mcp2_t = mcp2.T.copy()
    # These matrix multiplication lines are very slow
    Mcc = mcp1 @ mcp2_t
    Mpp = mcp2_t @ mcp1

    try:
        # Calculate eigenvectors
        eigvals, eigvecs = np.linalg.eig(Mpp)
        eigvecs = np.real(eigvecs)
        # Get eigenvector corresponding to second largest eigenvalue
        eig_index = eigvals.argsort()[-2]
        kp = eigvecs[:, eig_index]
        kc = mcp1 @ kp

        # Adjust sign of ECI and PCI so it makes sense, as per book
        s1 = np.sign(np.corrcoef(diversity_valid, kc)[0, 1])
        eci_t = s1 * kc
        pci_t = s1 * kp

        # Add back the deleted elements
        for x in cntry_mask:
            eci_t = np.insert(eci_t, x, np.nan)
        for x in prod_mask:
            pci_t = np.insert(pci_t, x, np.nan)

    except Exception as e:
        warnings.warn(f"Unable to calculate eigenvectors for year {cdata.t}")
        print(e)
        eci_t = np.empty(cdata.mcp_t.shape[0])
        pci_t = np.empty(cdata.mcp_t.shape[1])
        eci_t[:] = np.nan
        pci_t[:] = np.nan

    return (eci_t, pci_t)


def ecomplexity(data, cols_input, statanorm=False):
    """Complexity calculations through the ComplexityData class
    Args:
        data: pandas dataframe containing production / trade data.
            Including variables indicating time, location, product and value
        cols_input: dict of column names for time, location, product and value.
            Example: {'time':'year', 'loc':'origin', 'prod':'hs92', 'val':'export_val'}
        presence_test: str for test used for presence of industry in location.
            One of "rca" (default), "rpop", "both", or "manual".
            Determines which values are used for M_cp calculations.
            If "manual", M_cp is taken as given from the "value" column in data
        val_errors_flag: {'coerce','ignore','raise'}. Passed to pd.to_numeric
            *default* coerce.
        rca_mcp_threshold: numeric indicating RCA threshold beyond which mcp is 1.
            *default* 1.
        rpop_mcp_threshold: numeric indicating RPOP threshold beyond which mcp is 1.
            *default* 1. Only used if presence_test is not "rca".
        pop: pandas df, with time, location and corresponding population, in that order.
            Not required if presence_test is "rca" (default).
        continuous: Used to calculate product proximities, indicates whether
            to consider correlation of every product pair (True) or product
            co-occurrence (False). *default* False.
        asymmetric: Used to calculate product proximities, indicates whether
            to generate asymmetric proximity matrix (True) or symmetric (False).
            *default* False.
        verbose: Print year being processed
        statanorm: Set method of PCI normalization
    Returns:
        Pandas dataframe containing the data with the following additional columns:
            - diversity: k_c,0
            - ubiquity: k_p,0
            - rca: Balassa's RCA
            - rpop: (available if presence_test!="rca") RPOP
            - mcp: MCP used for complexity calculations
            - eci: Economic complexity index
            - pci: Product complexity index
            - density: Density of the network around each product
            - coi: Complexity Outlook Index
            - cog: Complexity Outlook Gain
    """

    # Set ecomplexity function variables to their default and assign the two remaining arguments as normal.
    presence_test = "rca"
    val_errors_flag = "coerce"
    rca_mcp_threshold = 1
    rpop_mcp_threshold = 1
    pop = None
    continuous = False
    asymmetric = False
    verbose = False

    cdata = ComplexityData(data, cols_input, val_errors_flag)

    cdata.output_list = []

    # Iterate over time stamps
    for t in cdata.data.index.unique("time"):
        if verbose:
            print(t)
        # Rectangularize df
        cdata.create_full_df(t)

        # Check if Mcp is pre-computed
        if presence_test != "manual":
            cdata.calculate_rca()
            cdata.calculate_mcp(
                rca_mcp_threshold, rpop_mcp_threshold, presence_test, pop, t
            )
        else:
            cdata.calculate_manual_mcp()

        # Calculate diversity and ubiquity
        cdata.diversity_t = np.nansum(cdata.mcp_t, axis=1)
        cdata.ubiquity_t = np.nansum(cdata.mcp_t, axis=0)

        # Calculate ECI and PCI
        cdata.eci_t, cdata.pci_t = calc_eci_pci(cdata)

        # Calculate proximity and density
        if continuous == False:
            prox_mat = calc_discrete_proximity(
                cdata.mcp_t, cdata.ubiquity_t, asymmetric
            )
            cdata.density_t = calc_density(cdata.mcp_t, prox_mat)
        elif continuous == True and presence_test == "rpop":
            prox_mat = calc_continuous_proximity(cdata.rpop_t, cdata.ubiquity_t)
            cdata.density_t = calc_density(cdata.rpop_t, prox_mat)
        elif continuous == True and presence_test != "rpop":
            prox_mat = calc_continuous_proximity(cdata.rca_t, cdata.ubiquity_t)
            cdata.density_t = calc_density(cdata.rca_t, prox_mat)

        # Calculate COI and COG
        cdata.coi_t, cdata.cog_t = calc_coi_cog(cdata, prox_mat)

        # Output normalization has been altered from the original CID-Harvard code:
        # https://github.com/cid-harvard/py-ecomplexity/blob/fed5d9b377b133a8431a05c347c4408086d789b2/ecomplexity/ecomplexity.py#L209
        if statanorm == True:
            # Normalize pci as per STATA package
            print('Normalizing PCI as per STATA package')
            cdata.pci_t = (cdata.pci_t - cdata.eci_t.mean()) / cdata.eci_t.std()
        else:
            # Normalize pci as per literature
            cdata.pci_t = (cdata.pci_t - cdata.pci_t.mean()) / cdata.pci_t.std()
            print('Normalizing PCI as per literature')
        # Normalize eci and cog as per STATA package
        cdata.cog_t = cdata.cog_t / cdata.eci_t.std()
        cdata.eci_t = (cdata.eci_t - cdata.eci_t.mean()) / cdata.eci_t.std()

        cdata.coi_t = (cdata.coi_t - cdata.coi_t.mean()) / cdata.coi_t.std()

        # Reshape ndarrays to df
        cdata = reshape_output_to_data(cdata, t)

    cdata.output = pd.concat(cdata.output_list)
    cdata = conform_to_original_data(cdata, data)

    return cdata.output
