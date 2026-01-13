import astropy.units as u
import numpy as np
import cdflib
import pandas as pd


def cdf2df(path):
    with cdflib.cdfread.CDF(path) as file:
        # Extract epoch times
        # epoch = file.varget(variable="EPOCH", expand=False, to_np=True)
        epoch = file.varget(variable="EPOCH")

        # Extract epoch times
        CDF_epoch_class = cdflib.epochs.CDFepoch()
        # time = CDF_epoch_class.to_datetime(epoch, to_np=True)
        time = CDF_epoch_class.to_datetime(epoch)

        # Extract B vectors times
        # B = file.varget(variable="B_RTN", expand=False)
        B = file.varget(variable="B_RTN")
        norm = np.linalg.norm(B, axis=1)

        # Get data attributes
        # attributes = file.globalattsget(expand=True)
        attributes = file.globalattsget()

        df = pd.DataFrame(
            {"BR": B.T[0], "BT": B.T[1], "BN": B.T[2], "|B|": norm}, index=time
        )

        return df


def PS_angle(distance, speed):
    """returns the parker spiral angle. This will be a negative number,
    and is only for positive polarity.

    Parameters
    ----------
    distance : u.Quantity
        Distance of the spacecraft from the Sun
    speed : u.Quantity
        Speed of the solar wind parcel

    Returns
    -------
    float
        Angle in degrees
    """
    # Solar rate of rotation
    omega = (14.713 * np.pi / 180.0) / (24 * 3600 * u.s)
    return np.arctan((-1.0 * distance * omega) / (speed)).to(u.deg).value + 360


def array_dot(r1, t1, r2, t2):
    return r1 * r2 + t1 * t2


def angle_between_components(r1, t1, r2, t2):
    return np.arccos(
        array_dot(r1, t1, r2, t2)
        / (np.sqrt(r1 * r1 + t1 * t1) * np.sqrt(r2 * r2 + t2 * t2))
    )


def polarity_at_sc(r, t, distance, speed, tolerance=45):
    """Works out if magnetic field falls within the Parker Spiral given some tolerance

    Parameters
    ----------
    r : array
        R component
    t : array
        T component
    distance : array with units
        Distance from Sun
    speed : array with units
        Speed of Solar Wind
    tolerance : float, optional
        In degrees, by default 45

    Returns
    -------
    array
        returns an array with -1 for negative polarity and +1 for positive polarity
    """

    # angle of the outwards (positive) Parker spiral
    ps_pos_angle = PS_angle(distance, speed)
    # angle of inwards (negative) polarity Parker spiral
    ps_neg_angle = ps_pos_angle - 180

    # e.g. ps_pos_angle will be around 330

    ps_pos_r = np.cos(ps_pos_angle * np.pi / 180)
    ps_pos_t = np.sin(ps_pos_angle * np.pi / 180)
    ps_neg_r = np.cos(ps_neg_angle * np.pi / 180)
    ps_neg_t = np.sin(ps_neg_angle * np.pi / 180)

    # angle of magnetic field to positive Parker spiral
    angle2pos = angle_between_components(r, t, ps_pos_r, ps_pos_t)
    # angle of magnetic field to negative Parker spiral
    angle2neg = angle_between_components(r, t, ps_neg_r, ps_neg_t)

    is_pos = (abs(angle2pos) * 180 / np.pi < tolerance).astype(int)
    is_neg = (abs(angle2neg) * 180 / np.pi < tolerance).astype(int) * -1
    return is_pos + is_neg


def add_polarity2df(df, ds_period="12H", tolerance=45):
    df_downsampled = df.resample(ds_period).mean()

    df_downsampled["polarity"] = polarity_at_sc(
        df_downsampled["B_R"].values,
        df_downsampled["B_T"].values,
        df_downsampled["Radius"].values * u.au,
        df_downsampled["V"].values * u.km / u.s,
        tolerance=tolerance,
    )

    # I want to forward fill, so I need to shift the indices back half a window length
    df_downsampled = df_downsampled.shift(periods=int(ds_period[:-1]) / 2, freq="H")

    # now upsample back to df
    df_upsampled = df.combine_first(df_downsampled)
    # # fill in nans
    df_upsampled.fillna(method="ffill", inplace=True)
    df["polarity"] = df_upsampled["polarity"]
    return df


def unwrap_lons(arr, threshold=0):
    try:
        idx = np.argwhere(np.diff(arr) > threshold).flatten()[0]
    except IndexError:
        return arr
    if isinstance(arr[0], u.Quantity):
        val_to_subtract = 360 * u.deg
    else:
        val_to_subtract = 360
    arr[idx + 1:] -= val_to_subtract
    return arr


def load_data(StartTime, EndTime, data_name, override=False, starting_letter=None, coordinates=None):
    from sunpy.net import Fido
    from sunpy.net import attrs as a
    from sunpy.timeseries import TimeSeries
    import pandas as pd
    import numpy as np

    print(data_name)

    # Define datatype based on data_name
    if data_name == 'SOMAG1MIN':
        datatype = 'SOLO_L2_MAG-RTN-NORMAL-1-MINUTE'
    elif data_name == 'SOMAGNORMAL':
        datatype = 'SOLO_L2_MAG-RTN-NORMAL'
    elif data_name == 'SOMAGBURST':
        datatype = 'SOLO_L2_MAG-RTN-BURST'
    elif data_name == 'SOPAS':
        datatype = 'SOLO_L2_SWA-PAS-GRND-MOM'
    elif data_name == 'SORPW':
        datatype = 'SOLO_L3_RPW-BIA-DENSITY'  ## DENSITY FROM RPW
    elif data_name == 'SORPW-E':  ## electric field data
        datatype = 'SOLO_L3_RPW-BIA-EFIELD'

    elif data_name == 'PSPMAG1MIN':
        datatype = 'PSP_FLD_L2_MAG_RTN_1MIN'
    elif data_name == 'PSPMAG':
        datatype = 'PSP_FLD_L2_MAG_RTN'

    #  elif data_name == 'PSPELE-DC':
    #     datatype = 'PSP_FLD_L2_DFB_DBM_SCM'

    # elif data_name == 'PSPELE-AC':
    #     datatype = 'PSP_FLD_L2_DFB_DBM_SCM'

    elif data_name == 'PSPELE-WF-differential':
        datatype = 'PSP_FLD_L2_DFB_WF_DVDC'

    elif data_name == 'PSPELE-WF-single-ended':
        datatype = 'PSP_FLD_L2_DFB_WF_VDC'


    elif data_name == 'PSPSPI':
        datatype = 'PSP_SWP_SPI_SF00_L3_MOM'  # density data
    else:
        if override:
            print(f"Using data_name as datatype : {data_name}")
            datatype = data_name
        else:
            print(f"Unrecognized data_name: {data_name}")
            return None

    # Prepare the search criteria
    dataset = a.cdaweb.Dataset(datatype)
    trange = a.Time(StartTime, EndTime)

    # Perform the search
    result = Fido.search(trange, dataset)

    if len(result) == 0:
        print(f"No files found for {StartTime} to {EndTime} with type {datatype}.")

        return None

    try:
        print(result)
        FilesToDownload = Fido.fetch(result)
        print(FilesToDownload)
        data = TimeSeries(FilesToDownload, concatenate=True)
        #print(data)
        #display(data)


    except Exception as e:
        print(f"Error downloading or processing data: {e}")
        raise e
        return None

    # '01234'[3
    # Convert to DataFrame
    df = data.to_dataframe()
    print( "The columns of the selected data_type are ", df.columns )
    chosen = []
    if override:
        chosen_raw = input("Choose index of coordinate columns") #'0,1,2' -> [0,1,2]
        # 1. Split input string by comma -> '0, 1,2' -> ['0', '1', '2']
        splitted_input = chosen_raw.split(",")
        # 'hello','my,'name','is kostis'.split(' ') -> [
        # 2. Remove any random spaces from the characters [' 0', '1 ', '2 '] -> ['0', '1', '2']
        splitted_input = [ ind.replace(' ', '') for ind in splitted_input ]
        # 3. Finally make all characters to integer int('132') -> 132 , ['0', '1', '2'] -> [int('0'), int('1'), int('2')] -> [0,1,2]
        chosen = [ int(character) for character in splitted_input ]
        print(chosen)

    # Data manipulation based on data_name
    if data_name in ['SOMAG1MIN', 'SOMAGNORMAL', 'SOMAGBURST']:
        df['|B|'] = np.sqrt(np.square(df['B_RTN_0']) + np.square(df['B_RTN_1']) + np.square(df['B_RTN_2']))
        df.rename(columns={'B_RTN_0': 'B_R', 'B_RTN_1': 'B_T', 'B_RTN_2': 'B_N'}, inplace=True)
    elif data_name == 'PSPMAG':
        df.rename(columns={'psp_fld_l2_mag_RTN_0': 'B_R', 'psp_fld_l2_mag_RTN_1': 'B_T', 'psp_fld_l2_mag_RTN_2': 'B_N'},
                  inplace=True)
        df['|B|'] = np.sqrt(np.square(df['B_R']) + np.square(df['B_T']) + np.square(df['B_N']))
        df = df.dropna(subset=['B_R', 'B_T', 'B_N'])

    elif data_name in ['SOPAS']:
        df.rename(columns={'V_RTN_0': 'V_R', 'V_RTN_1': 'V_T', 'V_RTN_2': 'V_N'}, inplace=True)
        df['|V|'] = np.sqrt(np.square(df['V_R']) + np.square(df['V_T']) + np.square(df['V_N']))

    elif data_name in ['PSPSPI']:
        df.rename(columns={'VEL_RTN_SUN_0': 'V_R', 'VEL_RTN_SUN_1': 'V_T', 'VEL_RTN_SUN_2': 'V_N', 'DENS': 'Np',
                           'TEMP': 'Tp'}, inplace=True)
        df['|V|'] = np.sqrt(np.square(df['V_R']) + np.square(df['V_T']) + np.square(df['V_N']))

    elif data_name in ['SORPW-E']:
        df.rename(columns={'EDC_SRF_0': 'E_X', 'EDC_SRF_1': 'E_Y', 'EDC_SRF_2': 'E_Z'}, inplace=True)
        df['|E|'] = np.sqrt(np.square(df['E_X']) + np.square(df['E_Y']) + np.square(df['E_Z']))

    elif data_name in ['PSPELE-WF-differential']:
        df.rename(columns={'psp_fld_l2_dfb_wf_dVdc_sc_0': 'E_0', 'psp_fld_l2_dfb_wf_dVdc_sc_1': 'E_1'}, inplace=True)
    # df['|E|'] = np.sqrt(np.square(df['E_X']) + np.square(df['E_Y']) + np.square(df['E_Z']))
    elif override:
        cols = df.columns # return the list of the column names of the dataframe
        # [1,2,3]
        for i, c in enumerate(chosen):
            df = df.rename( columns={cols[c] : starting_letter + '_' + coordinates[i]} )

        calculation_helper = np.square(df[starting_letter + '_' + coordinates[0]])
        for coord in coordinates[1:]:
            calculation_helper = calculation_helper + np.square(df[starting_letter + '_' + coord])
        df['|' + starting_letter + '|'] = np.sqrt(calculation_helper)


    return df