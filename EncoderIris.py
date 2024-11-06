import numpy as np
import warnings
warnings.filterwarnings("ignore")


class EncoderIris:
    def encode_iris(self, arr_polar, arr_noise, minw_length, mult, sigma_f):
        """
        Generate iris template and noise mask from the normalized iris region.
        """
        # Convolve with Gabor filters
        filterb = self.gaborconvolve_f(arr_polar, minw_length, mult, sigma_f)
        l = arr_polar.shape[1]
        template = np.zeros([arr_polar.shape[0], 2 * l])
        h = np.arange(arr_polar.shape[0])

        # Making the iris template
        mask_noise = np.zeros(template.shape)
        filt = filterb[:, :]

        # Quantization and check to see if the phase data is useful
        H1 = np.real(filt) > 0
        H2 = np.imag(filt) > 0

        H3 = np.abs(filt) < 0.0001
        for i in range(l):
            ja = 2 * i

            # Biometric template
            template[:, ja] = H1[:, i]
            template[:, ja + 1] = H2[:, i]
            # Noise mask_noise
            mask_noise[:, ja] = arr_noise[:, i] | H3[:, i]
            mask_noise[:, ja + 1] = arr_noise[:, i] | H3[:, i]

        return template, mask_noise

    def gaborconvolve_f(self, img, minw_length, mult, sigma_f):
        """
        Convolve each row of an image with 1D log-Gabor filters.
        """
        rows, ndata = img.shape
        logGabor_f = np.zeros(ndata)
        filterb = np.zeros([rows, ndata], dtype=complex)

        radius = np.arange(ndata / 2 + 1) / (ndata / 2) / 2
        radius[0] = 1

        # Filter wavelength
        wavelength = minw_length

        # Radial filter component
        fo = 1 / wavelength
        logGabor_f[0: int(ndata / 2) + 1] = np.exp((-(np.log(radius / fo)) ** 2) / (2 * np.log(sigma_f) ** 2))
        logGabor_f[0] = 0

        # Convolution for each row
        for r in range(rows):
            signal = img[r, 0:ndata]
            imagefft = np.fft.fft(signal)
            filterb[r, :] = np.fft.ifft(imagefft * logGabor_f)

        return filterb