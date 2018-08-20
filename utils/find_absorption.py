import numpy as np

def Determine_absorption_features(Wavelengths_of_region,Fluxes_of_region,Error_of_region,minimum_wavelength):
    """ Code to roughly locate absorption features to fit for to improve emission line profile fitting """

    location_region = 10. # How many angstroms left and right of the absorption minimia to search
    mean_window_width = 5. # Width in angstrom around the absorption line centre to average to determine the degree of fluctuation (for removing some additional candidate lines)
    sigma_depth = 5. # absorption must be larger than sigma depth * the error at that point

    sigma_fluct_low = 3. # For fluctuations away from the mean the flux needs to be to be considered an absorption feature (smaller value due to larger error)
    sigma_fluct_high = 1. # Same as above but > 1250 A, when the error is much smaller

    # Basically searches for all local minima, then tests whether these are large/deep enough to consitute an absorption feature
    # For each minimia, it computes the volume below an incrimentally increasing line of constant flux
    # If the volume ends up sufficiently large, set by the parameters above, it is flagged an absorption feature


    if len(Wavelengths_of_region) < 2.:

        final_absorption_features = []

        return final_absorption_features

    else:
        min_separation = Wavelengths_of_region[1] - Wavelengths_of_region[0]

        for i in range(len(Wavelengths_of_region)-1):

            min_separation_new = Wavelengths_of_region[i+1] - Wavelengths_of_region[i]

            if min_separation_new < min_separation:
                min_separation = min_separation_new

        wavelength_separation = 2.*min_separation # Search on an arbitrary 2 Angstrom width

        surrounding_bins = np.ceil(wavelength_separation/min_separation)

        if surrounding_bins % 2 == 0:
            surrounding_bins = surrounding_bins + 1

        surrounding_bins = int(surrounding_bins)

        location_min = []
        updated_location_min = []
        location_min_int = []
        updated_location_min_int = []
        flux_mins = []

        for i in range(len(Wavelengths_of_region)):

            flux_at_point = Fluxes_of_region[i]

            flux_surrounding = []
            for j in range(surrounding_bins):
                if i > (surrounding_bins-1)/2 and i < len(Wavelengths_of_region)-1-(surrounding_bins-1)/2:
                    if j != (surrounding_bins-1)/2:
                        flux_surrounding.append(Fluxes_of_region[i+j-(surrounding_bins-1)/2])
                elif i < (surrounding_bins-1)/2:
                    if i+j-(surrounding_bins-1)/2 > -1: # to include i=0
                        if j != (surrounding_bins-1)/2:
                            flux_surrounding.append(Fluxes_of_region[i+j-(surrounding_bins-1)/2])
                else:
                    if i+j-(surrounding_bins-1)/2 < len(Wavelengths_of_region)-2: # to include i=0
                        if j != (surrounding_bins-1)/2:
                            flux_surrounding.append(Fluxes_of_region[i+j-(surrounding_bins-1)/2])

            if i != 0 and i != len(Wavelengths_of_region)-1:

                if flux_at_point < np.amin(flux_surrounding):
                    location_min.append(Wavelengths_of_region[i])
                    location_min_int.append(i)
                    flux_mins.append(flux_at_point)
        #print('flux_mins',flux_mins)
        peak_flux = np.nanmax(Fluxes_of_region)
        min_flux = np.nanmin(Fluxes_of_region)

        n_levels = int(np.ceil((peak_flux-min_flux)/0.2))
        print('dfd',(peak_flux-min_flux)/0.2)

        flux_step = (peak_flux - min_flux)/(n_levels-1)
        print('flux_step',flux_step)
        print('n_levels',n_levels)
        record_flux_level = []
        all_loc_mins_left = []
        all_loc_mins_right = []

        for n_pos in range(len(location_min)):
            if location_min[n_pos] > minimum_wavelength:

                for i in range(n_levels):
                    F = min_flux + flux_step*i
                    if flux_mins[n_pos] <= F:
                        record_flux_level.append(i)
                        break

                flux_left = []
                flux_right = []

                for ii in range(len(Wavelengths_of_region)):

                    if Wavelengths_of_region[ii] > (Wavelengths_of_region[location_min_int[n_pos]-1] - location_region) and Wavelengths_of_region[ii] < Wavelengths_of_region[location_min_int[n_pos]-1]:
                        flux_left.append(Fluxes_of_region[ii])
                    if Wavelengths_of_region[ii] < (Wavelengths_of_region[location_min_int[n_pos]+1] + location_region) and Wavelengths_of_region[ii] > Wavelengths_of_region[location_min_int[n_pos]+1]:
                        flux_right.append(Fluxes_of_region[ii])

                if len(flux_left) == 0:
                    flux_left.append(Fluxes_of_region[0])

                if len(flux_right) == 0:
                    flux_right.append(Fluxes_of_region[-1])

                max_flux_left = np.amax(flux_left)
                max_flux_right = np.amax(flux_right)

                starting_point = record_flux_level[n_pos]
                levels_left = 0
                levels_right = 0
                for ii in range(location_min_int[n_pos]):

                    for i in range(n_levels - starting_point):
                        F = min_flux + flux_step*(i + starting_point)

                        if F > Fluxes_of_region[location_min_int[n_pos]-ii-1]:
                            wavelength_left = Wavelengths_of_region[location_min_int[n_pos]-1-ii]
                            starting_point = i + starting_point
                            break

                    if Fluxes_of_region[location_min_int[n_pos]-ii-1] >= max_flux_left:
                        levels_left = starting_point - record_flux_level[n_pos]
                        break

                starting_point = record_flux_level[n_pos]
                for ii in range(len(Wavelengths_of_region)-location_min_int[n_pos]-1):

                    for i in range(n_levels - starting_point):
                        F = min_flux + flux_step*(i + starting_point)

                        if F > Fluxes_of_region[location_min_int[n_pos]+ii+1]:
                            wavelength_right = Wavelengths_of_region[location_min_int[n_pos]+1+ii]
                            starting_point = i + starting_point
                            break

                    if Fluxes_of_region[location_min_int[n_pos]+ii+1] >= max_flux_right:
                        levels_right = starting_point - record_flux_level[n_pos]
                        break

                if levels_left < levels_right:
                    min_num_levels = levels_left
                else:
                    min_num_levels = levels_right

                approx_depth = min_num_levels*flux_step

                if approx_depth > sigma_depth*Error_of_region[location_min_int[n_pos]]:
                    updated_location_min.append(location_min[n_pos])
                    updated_location_min_int.append(location_min_int[n_pos])
            else:
                record_flux_level.append(0)

        final_absorption_features = []
        for i in range(len(updated_location_min)):

            mean_window = []
            for j in range(len(Wavelengths_of_region)):
                if Wavelengths_of_region[j] > (updated_location_min[i] - mean_window_width) and Wavelengths_of_region[j] < (updated_location_min[i] + mean_window_width):
                    mean_window.append(Fluxes_of_region[j])

            mean_val = np.mean(mean_window)

            if updated_location_min[i] < 1250.:

                if mean_val - Fluxes_of_region[updated_location_min_int[i]] > sigma_fluct_low*Error_of_region[updated_location_min_int[i]]:
                    final_absorption_features.append(updated_location_min[i])
            else:
                if mean_val - Fluxes_of_region[updated_location_min_int[i]] > sigma_fluct_high*Error_of_region[updated_location_min_int[i]]:
                    final_absorption_features.append(updated_location_min[i])

        return final_absorption_features
