import colour
import numpy as np
from colour.plotting import *




XYZ = [[53.1250, 100.0000, -152.9297],
       [312.5926, 100.0000, -412.4074],
       [33.6634, 100.0000, -133.5396],
       [200.4785, 100.0000, -300.3589],
       [25.2632, 100.0000, -125.1754],
       [20.5556, 100.0000, -120.4861],
       [15.6600, 100.0000, -115.6040],
       [12.3791, 100.0000, -112.3308],
       [11.0783, 100.0000, -111.0414],
       [65.1852, 100.0000, -165.1481],
       [8.7404,  100.0000, -108.7082],
       [55.2124, 100.0000, -155.1802],
       [7.7013, 100.0000, -107.6721],
       [6.7236, 100.0000, -106.6969],
       [6.2812, 100.0000, -106.2562],
       [5.4228, 100.0000, -105.3998],
       [5.2800, 100.0000, -105.2600],
       [33.3598, 100.0000, -133.3400],
       [4.9927, 100.0000, -104.9743],
       [31.1812, 100.0000, -131.1629],
       [4.5710, 100.0000, -104.5534],
       [4.2384, 100.0000, -104.2219],
       [3.8192, 100.0000, -103.8033],
       [3.2511, 100.0000, -103.2360],
       [3.0928, 100.0000, -103.0792],
       [22.7519, 100.0000, -122.7384],
       [2.8616, 100.0000, -102.8486],
       [21.5727, 100.0000, -121.5598],
       [2.9044, 100.0000, -102.8918],
       [2.4566, 100.0000, -102.4446],
       [2.5151, 100.0000, -102.5035],
       [2.2645, 100.0000, -102.2532]]

RGB = []


illuminant_XYZ = [0.34570, 0.35850]
illuminant_RGB = [0.31270, 0.32900]
chromatic_adaptation_transform = 'Bradford'
matrix_XYZ_to_RGB = [
         [3.24062548, -1.53720797, -0.49862860],
         [-0.96893071, 1.87575606, 0.04151752],
         [0.05571012, -0.20402105, 1.05699594]]



rgb_list  = colour.XYZ_to_RGB(
         XYZ,
         illuminant_XYZ,
         illuminant_RGB,
         matrix_XYZ_to_RGB,
         chromatic_adaptation_transform)

print(rgb_list)

RGB = colour.models.eotf_inverse_sRGB(np.array(rgb_list) / 255)
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(RGB)

# for xyz_list in XYZ:
    
#     print("inside loop")
#     rgb_list  = colour.XYZ_to_RGB(
#          xyz_list,
#          illuminant_XYZ,
#          illuminant_RGB,
#          matrix_XYZ_to_RGB,
#          chromatic_adaptation_transform)
   
#     RGB.append(rgb_list)

    
    
    
# print(RGB)


#for values in RGB:
#    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(RGB)
#RGB = colour.models.eotf_inverse_sRGB(np.array([[79, 2, 45], [87, 12, 67]]) / 255)

#plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(RGB)